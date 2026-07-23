import 'dart:async';
import 'dart:convert';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart'
    hide NotificationSettings;
import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

import 'native_platform_bridge.dart';
import 'notification_settings.dart';

const _channel = AndroidNotificationChannel(
  'keiba_general_notifications',
  '更新通知',
  description: 'レース更新や日報公開のお知らせを表示します',
  importance: Importance.defaultImportance,
);

@pragma('vm:entry-point')
Future<void> firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp();
  final settings = await NotificationSettingsRepository().loadMirror();
  if (!settings.allows(NotificationSettings.scopeFromData(message.data))) {
    return;
  }
  final notifications = FlutterLocalNotificationsPlugin();
  await _initializeLocalNotifications(notifications);
  await _showRemoteMessage(notifications, message);
}

class NotificationDestination {
  const NotificationDestination({this.url, this.destination, this.title});

  factory NotificationDestination.fromData(Map<String, dynamic> data) {
    String? value(String key) {
      final normalized = data[key]?.toString().trim() ?? '';
      return normalized.isEmpty ? null : normalized;
    }

    return NotificationDestination(
      url: value('url'),
      destination: value('destination'),
      title: value('title'),
    );
  }

  factory NotificationDestination.fromPayload(String payload) {
    final decoded = jsonDecode(payload);
    if (decoded is! Map) return const NotificationDestination();
    return NotificationDestination.fromData(
      decoded.map((key, value) => MapEntry(key.toString(), value)),
    );
  }

  final String? url;
  final String? destination;
  final String? title;

  bool get hasTarget => url != null || destination != null;
}

class PushNotificationService {
  PushNotificationService._();

  static final instance = PushNotificationService._();
  static const _topicRetryDelays = [
    Duration(seconds: 2),
    Duration(seconds: 5),
    Duration(seconds: 10),
    Duration(seconds: 20),
    Duration(seconds: 30),
  ];

  final _notifications = FlutterLocalNotificationsPlugin();
  final _destinations = StreamController<NotificationDestination>.broadcast();
  StreamSubscription<NotificationSettings>? _settingsSubscription;
  StreamSubscription<String>? _tokenRefreshSubscription;
  StreamSubscription<RemoteMessage>? _foregroundMessageSubscription;
  StreamSubscription<RemoteMessage>? _openedAppSubscription;
  NotificationDestination? _pendingDestination;
  NotificationSettings _latestSettings = const NotificationSettings();
  Future<void> _topicSyncTail = Future<void>.value();
  Future<void>? _initializationInProgress;
  Timer? _topicRetryTimer;
  int _topicRetryAttempt = 0;
  bool _initialized = false;

  Stream<NotificationDestination> get destinations => _destinations.stream;

  Future<String?> getRegistrationToken() async {
    try {
      final token = await FirebaseMessaging.instance.getToken();
      final normalized = token?.trim() ?? '';
      return normalized.isEmpty ? null : normalized;
    } catch (error) {
      debugPrint('FCM token could not be loaded: $error');
      return null;
    }
  }

  NotificationDestination? takePendingDestination() {
    final value = _pendingDestination;
    _pendingDestination = null;
    return value;
  }

  Future<void> initialize() {
    if (_initialized) return Future<void>.value();
    final existing = _initializationInProgress;
    if (existing != null) return existing;

    late final Future<void> operation;
    operation = _initializeOnce().then((_) => _initialized = true).whenComplete(
      () {
        if (identical(_initializationInProgress, operation)) {
          _initializationInProgress = null;
        }
      },
    );
    _initializationInProgress = operation;
    return operation;
  }

  Future<void> _initializeOnce() async {
    FirebaseMessaging.onBackgroundMessage(firebaseMessagingBackgroundHandler);
    await _initializeLocalNotifications(
      _notifications,
      onResponse: _onLocalNotificationResponse,
    );

    final settingsRepository = NotificationSettingsRepository();
    NotificationSettings settings;
    try {
      settings = await settingsRepository.load();
    } catch (error) {
      debugPrint('Notification settings could not be loaded: $error');
      settings = const NotificationSettings();
    }
    _latestSettings = settings;
    await _syncTopics(settings);
    _settingsSubscription = NotificationSettingsRepository.changes.listen(
      (value) => unawaited(_syncTopics(value)),
    );
    _tokenRefreshSubscription = FirebaseMessaging.instance.onTokenRefresh
        .listen(
          (_) => unawaited(_syncTopics(_latestSettings)),
          onError: (Object error) {
            debugPrint('FCM token refresh failed: $error');
            _scheduleTopicRetry();
          },
        );

    _foregroundMessageSubscription = FirebaseMessaging.onMessage.listen(
      (message) => unawaited(_handleForegroundMessage(message)),
    );
    _openedAppSubscription = FirebaseMessaging.onMessageOpenedApp.listen(
      _handleRemoteMessageTap,
    );

    try {
      final initialRemoteMessage = await FirebaseMessaging.instance
          .getInitialMessage();
      if (initialRemoteMessage != null) {
        _setPendingDestination(
          NotificationDestination.fromData(initialRemoteMessage.data),
        );
      }
    } catch (error) {
      debugPrint('Initial FCM message could not be read: $error');
    }

    try {
      final localLaunch = await _notifications
          .getNotificationAppLaunchDetails();
      final localPayload = localLaunch?.notificationResponse?.payload;
      if ((localLaunch?.didNotificationLaunchApp ?? false) &&
          localPayload != null) {
        _setPendingDestination(_decodePayload(localPayload));
      }
    } catch (error) {
      debugPrint('Initial local notification could not be read: $error');
    }
  }

  Future<void> requestPermissionOnFirstLaunch() async {
    if (!await NativePlatformBridge.shouldRequestNotificationPermission()) {
      return;
    }
    final requestCompleted = await requestPermissionFromSettings();
    if (requestCompleted) {
      await NativePlatformBridge.markNotificationPermissionPrompted();
    }
  }

  Future<bool> requestPermissionFromSettings() async {
    try {
      if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
        await _notifications
            .resolvePlatformSpecificImplementation<
              AndroidFlutterLocalNotificationsPlugin
            >()
            ?.requestNotificationsPermission();
      }
      await FirebaseMessaging.instance.requestPermission(
        alert: true,
        badge: true,
        sound: true,
      );
      await _syncTopics(await NotificationSettingsRepository().loadMirror());
      return true;
    } catch (error) {
      debugPrint('Notification permission request failed: $error');
      _scheduleTopicRetry();
      return false;
    }
  }

  Future<void> _handleForegroundMessage(RemoteMessage message) async {
    final settings = await NotificationSettingsRepository().loadMirror();
    if (!settings.allows(NotificationSettings.scopeFromData(message.data))) {
      return;
    }
    await _showRemoteMessage(_notifications, message);
  }

  void _handleRemoteMessageTap(RemoteMessage message) {
    _emitDestination(NotificationDestination.fromData(message.data));
  }

  void _onLocalNotificationResponse(NotificationResponse response) {
    final payload = response.payload;
    if (payload == null || payload.isEmpty) return;
    _emitDestination(_decodePayload(payload));
  }

  NotificationDestination _decodePayload(String payload) {
    try {
      return NotificationDestination.fromPayload(payload);
    } on FormatException {
      return const NotificationDestination();
    }
  }

  void _setPendingDestination(NotificationDestination destination) {
    if (destination.hasTarget) _pendingDestination = destination;
  }

  void _emitDestination(NotificationDestination destination) {
    if (destination.hasTarget) _destinations.add(destination);
  }

  Future<void> _syncTopics(
    NotificationSettings settings, {
    bool resetRetry = true,
  }) {
    _latestSettings = settings;
    if (resetRetry) {
      _topicRetryTimer?.cancel();
      _topicRetryTimer = null;
      _topicRetryAttempt = 0;
    }
    final operation = _topicSyncTail.then((_) => _syncLatestTopics());
    _topicSyncTail = operation;
    return operation;
  }

  Future<void> _syncLatestTopics() async {
    final config = NativePlatformBridge.runtimeConfig;
    if (config == null) return;

    if (!await _isMessagingReady()) {
      _scheduleTopicRetry();
      return;
    }

    final settings = _latestSettings;
    final results = await Future.wait([
      _syncTopic(config.fcmTopic, settings.enabled),
      _syncTopic(
        config.fcmCentralTopic,
        settings.enabled && settings.centralEnabled,
      ),
      _syncTopic(
        config.fcmLocalTopic,
        settings.enabled && settings.localEnabled,
      ),
    ]);
    if (results.every((result) => result)) {
      _topicRetryTimer?.cancel();
      _topicRetryTimer = null;
      _topicRetryAttempt = 0;
      return;
    }
    _scheduleTopicRetry();
  }

  Future<bool> _isMessagingReady() async {
    try {
      if (!kIsWeb && defaultTargetPlatform == TargetPlatform.iOS) {
        final permission = await FirebaseMessaging.instance
            .getNotificationSettings();
        if (permission.authorizationStatus ==
            AuthorizationStatus.notDetermined) {
          return false;
        }
        final apnsToken = await FirebaseMessaging.instance.getAPNSToken();
        if (apnsToken == null || apnsToken.trim().isEmpty) return false;
      }
      final fcmToken = await FirebaseMessaging.instance.getToken();
      return fcmToken != null && fcmToken.trim().isNotEmpty;
    } catch (error) {
      debugPrint('FCM token is not ready: $error');
      return false;
    }
  }

  Future<bool> _syncTopic(String topic, bool subscribe) async {
    if (topic.trim().isEmpty) return true;
    try {
      if (subscribe) {
        await FirebaseMessaging.instance.subscribeToTopic(topic.trim());
      } else {
        await FirebaseMessaging.instance.unsubscribeFromTopic(topic.trim());
      }
      return true;
    } catch (error) {
      debugPrint('FCM topic sync failed for ${topic.trim()}: $error');
      return false;
    }
  }

  void _scheduleTopicRetry() {
    if (_topicRetryTimer != null ||
        _topicRetryAttempt >= _topicRetryDelays.length) {
      return;
    }
    final delay = _topicRetryDelays[_topicRetryAttempt];
    _topicRetryAttempt += 1;
    _topicRetryTimer = Timer(delay, () {
      _topicRetryTimer = null;
      unawaited(_syncTopics(_latestSettings, resetRetry: false));
    });
  }

  Future<void> dispose() async {
    _topicRetryTimer?.cancel();
    await _settingsSubscription?.cancel();
    await _tokenRefreshSubscription?.cancel();
    await _foregroundMessageSubscription?.cancel();
    await _openedAppSubscription?.cancel();
    await _destinations.close();
  }
}

Future<void> _initializeLocalNotifications(
  FlutterLocalNotificationsPlugin notifications, {
  DidReceiveNotificationResponseCallback? onResponse,
}) async {
  const initializationSettings = InitializationSettings(
    android: AndroidInitializationSettings('ic_stat_name'),
    iOS: DarwinInitializationSettings(
      requestAlertPermission: false,
      requestBadgePermission: false,
      requestSoundPermission: false,
    ),
  );
  await notifications.initialize(
    settings: initializationSettings,
    onDidReceiveNotificationResponse: onResponse,
  );
  await notifications
      .resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin
      >()
      ?.createNotificationChannel(_channel);
}

Future<void> _showRemoteMessage(
  FlutterLocalNotificationsPlugin notifications,
  RemoteMessage message,
) async {
  final title =
      message.notification?.title ?? message.data['title'] ?? 'いかいもAI競馬';
  final body = message.notification?.body ?? message.data['body'];
  if (body == null || body.trim().isEmpty) return;

  final payload = jsonEncode(<String, String>{
    ...message.data,
    if (!message.data.containsKey('title')) 'title': title,
  });
  final details = NotificationDetails(
    android: AndroidNotificationDetails(
      'keiba_general_notifications',
      '更新通知',
      channelDescription: 'レース更新や日報公開のお知らせを表示します',
      icon: 'ic_stat_name',
      importance: Importance.defaultImportance,
      priority: Priority.defaultPriority,
      styleInformation: BigTextStyleInformation(body),
    ),
    iOS: const DarwinNotificationDetails(),
  );
  await notifications.show(
    id: DateTime.now().millisecondsSinceEpoch.remainder(0x0fffffff),
    title: title,
    body: body,
    notificationDetails: details,
    payload: payload,
  );
}

import 'dart:async';
import 'dart:convert';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart'
    hide NotificationSettings;
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

import 'android_platform_bridge.dart';
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

  final _notifications = FlutterLocalNotificationsPlugin();
  final _destinations = StreamController<NotificationDestination>.broadcast();
  StreamSubscription<NotificationSettings>? _settingsSubscription;
  NotificationDestination? _pendingDestination;
  bool _initialized = false;

  Stream<NotificationDestination> get destinations => _destinations.stream;

  NotificationDestination? takePendingDestination() {
    final value = _pendingDestination;
    _pendingDestination = null;
    return value;
  }

  Future<void> initialize() async {
    if (_initialized) return;
    _initialized = true;

    FirebaseMessaging.onBackgroundMessage(firebaseMessagingBackgroundHandler);
    await _initializeLocalNotifications(
      _notifications,
      onResponse: _onLocalNotificationResponse,
    );

    final settingsRepository = NotificationSettingsRepository();
    final settings = await settingsRepository.load();
    await _syncTopics(settings);
    _settingsSubscription = NotificationSettingsRepository.changes.listen(
      (value) => unawaited(_syncTopics(value)),
    );

    FirebaseMessaging.onMessage.listen(
      (message) => unawaited(_handleForegroundMessage(message)),
    );
    FirebaseMessaging.onMessageOpenedApp.listen(_handleRemoteMessageTap);

    final initialRemoteMessage = await FirebaseMessaging.instance
        .getInitialMessage();
    if (initialRemoteMessage != null) {
      _setPendingDestination(
        NotificationDestination.fromData(initialRemoteMessage.data),
      );
    }

    final localLaunch = await _notifications.getNotificationAppLaunchDetails();
    final localPayload = localLaunch?.notificationResponse?.payload;
    if ((localLaunch?.didNotificationLaunchApp ?? false) &&
        localPayload != null) {
      _setPendingDestination(_decodePayload(localPayload));
    }
  }

  Future<void> requestPermissionOnFirstLaunch() async {
    if (!await AndroidPlatformBridge.shouldRequestNotificationPermission()) {
      return;
    }
    await AndroidPlatformBridge.markNotificationPermissionPrompted();
    await requestPermissionFromSettings();
  }

  Future<void> requestPermissionFromSettings() async {
    await _notifications
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >()
        ?.requestNotificationsPermission();
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

  Future<void> _syncTopics(NotificationSettings settings) async {
    final config = AndroidPlatformBridge.runtimeConfig;
    if (config == null) return;
    await Future.wait([
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
  }

  Future<void> _syncTopic(String topic, bool subscribe) async {
    if (topic.trim().isEmpty) return;
    try {
      if (subscribe) {
        await FirebaseMessaging.instance.subscribeToTopic(topic.trim());
      } else {
        await FirebaseMessaging.instance.unsubscribeFromTopic(topic.trim());
      }
    } on FirebaseException {
      // 通信が戻った後、次回起動または設定変更時に再同期する。
    }
  }

  Future<void> dispose() async {
    await _settingsSubscription?.cancel();
    await _destinations.close();
  }
}

Future<void> _initializeLocalNotifications(
  FlutterLocalNotificationsPlugin notifications, {
  DidReceiveNotificationResponseCallback? onResponse,
}) async {
  const initializationSettings = InitializationSettings(
    android: AndroidInitializationSettings('ic_stat_name'),
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
  );
  await notifications.show(
    id: DateTime.now().millisecondsSinceEpoch.remainder(0x0fffffff),
    title: title,
    body: body,
    notificationDetails: details,
    payload: payload,
  );
}

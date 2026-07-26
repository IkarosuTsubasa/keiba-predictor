import 'dart:async';
import 'dart:ui';

import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'src/app.dart';
import 'src/services/ad_consent_service.dart';
import 'src/services/app_startup_service.dart';
import 'src/services/native_platform_bridge.dart';
import 'src/services/push_notification_service.dart';

bool _firebaseReady = false;

void main() {
  runZonedGuarded(
    () {
      WidgetsFlutterBinding.ensureInitialized();
      runApp(const _KeibaBootstrapApp());
    },
    (error, stack) => _recordUncaughtError(
      error,
      stack,
      reason: 'Uncaught top-level zone error',
    ),
  );
}

class _KeibaBootstrapApp extends StatefulWidget {
  const _KeibaBootstrapApp();

  @override
  State<_KeibaBootstrapApp> createState() => _KeibaBootstrapAppState();
}

class _KeibaBootstrapAppState extends State<_KeibaBootstrapApp> {
  AppStartupResult? _startupResult;
  AppStartupStage _stage = AppStartupStage.firebaseInitialization;

  @override
  void initState() {
    super.initState();
    // Render a Flutter frame before any plugin initialization. This prevents
    // a slow SDK or platform channel from keeping iOS on the native launch
    // screen long enough to trigger a launch watchdog termination.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      unawaited(_initialize());
    });
  }

  Future<void> _initialize() async {
    final startup = AppStartupService(
      initializeFirebase: () async {
        await Firebase.initializeApp();
      },
      onFirebaseReady: _enableCrashReporting,
      initializeNativeConfiguration: NativePlatformBridge.initialize,
      onStage: _onStageChanged,
      onFailure: _onStartupFailure,
    );
    final result = await startup.initialize();
    if (!mounted) return;

    setState(() => _startupResult = result);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      unawaited(_initializePostLaunchServices(result));
    });
  }

  Future<void> _enableCrashReporting() {
    _firebaseReady = true;
    FlutterError.onError = (details) {
      FlutterError.presentError(details);
      unawaited(_recordFlutterFatalError(details));
    };
    PlatformDispatcher.instance.onError = (error, stack) {
      unawaited(
        _recordUncaughtError(
          error,
          stack,
          reason: 'Uncaught platform dispatcher error',
        ),
      );
      return true;
    };
    unawaited(_recordStartupStage('firebase_ready'));
    return Future<void>.value();
  }

  void _onStageChanged(AppStartupStage stage) {
    if (mounted) setState(() => _stage = stage);
    unawaited(_recordStartupStage(stage.name));
  }

  void _onStartupFailure(
    AppStartupStage stage,
    Object error,
    StackTrace stack,
  ) {
    debugPrint('Startup stage ${stage.name} failed: $error');
    unawaited(
      _recordNonFatalError(
        error,
        stack,
        reason: 'Startup stage ${stage.name} failed',
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final result = _startupResult;
    if (result == null) {
      return _StartupView(stage: _stage);
    }
    return KeibaApp(
      navigatorObservers: result.firebaseReady
          ? [FirebaseAnalyticsObserver(analytics: FirebaseAnalytics.instance)]
          : const [],
    );
  }
}

class _StartupView extends StatelessWidget {
  const _StartupView({required this.stage});

  final AppStartupStage stage;

  @override
  Widget build(BuildContext context) {
    final status = switch (stage) {
      AppStartupStage.firebaseInitialization => '起動準備中',
      AppStartupStage.nativeConfiguration => 'アプリを準備しています',
      AppStartupStage.complete => 'まもなく表示します',
    };
    return MaterialApp(
      title: 'いかいもAI競馬',
      debugShowCheckedModeBanner: false,
      theme: KeibaApp.theme,
      home: Scaffold(
        backgroundColor: const Color(0xFF07111F),
        body: SafeArea(
          child: Center(
            child: Semantics(
              label: status,
              child: const Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'いかいもAI競馬',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 26,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 22),
                  SizedBox(
                    width: 28,
                    height: 28,
                    child: CircularProgressIndicator(
                      strokeWidth: 3,
                      color: Color(0xFF7EC7BD),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

Future<void> _initializePostLaunchServices(AppStartupResult startup) async {
  final tasks = <Future<void>>[
    _initializeAdConsent(),
    if (startup.firebaseReady) _initializePushNotifications(),
  ];
  await Future.wait(tasks);

  if (!startup.firebaseReady) return;
  await Future<void>.delayed(const Duration(milliseconds: 350));
  try {
    await PushNotificationService.instance.requestPermissionOnFirstLaunch();
  } catch (error, stack) {
    await _recordNonFatalError(
      error,
      stack,
      reason: 'Notification permission request failed',
    );
  }
  await _recordStartupStage('post_launch_services_ready');
}

Future<void> _initializeAdConsent() async {
  try {
    await AdConsentService.instance.initialize();
  } catch (error, stack) {
    await _recordNonFatalError(
      error,
      stack,
      reason: 'Ad consent initialization failed',
    );
  }
}

Future<void> _initializePushNotifications() async {
  try {
    await PushNotificationService.instance.initialize();
  } catch (error, stack) {
    await _recordNonFatalError(
      error,
      stack,
      reason: 'Push notification initialization failed',
    );
  }
}

Future<void> _recordStartupStage(String stage) async {
  if (!_firebaseReady) return;
  try {
    await FirebaseCrashlytics.instance.setCustomKey('startup_stage', stage);
    await FirebaseCrashlytics.instance.log('startup_stage=$stage');
  } catch (error) {
    debugPrint('Crashlytics startup stage logging failed: $error');
  }
}

Future<void> _recordFlutterFatalError(FlutterErrorDetails details) async {
  if (!_firebaseReady) return;
  try {
    await FirebaseCrashlytics.instance.recordFlutterFatalError(details);
  } catch (recordingError) {
    debugPrint('Crashlytics Flutter error recording failed: $recordingError');
  }
}

Future<void> _recordNonFatalError(
  Object error,
  StackTrace stack, {
  required String reason,
}) async {
  if (!_firebaseReady) {
    debugPrint('$reason: $error');
    return;
  }
  try {
    await FirebaseCrashlytics.instance.recordError(
      error,
      stack,
      reason: reason,
      fatal: false,
    );
  } catch (recordingError) {
    debugPrint('Crashlytics error recording failed: $recordingError');
  }
}

Future<void> _recordUncaughtError(
  Object error,
  StackTrace stack, {
  required String reason,
}) async {
  if (!_firebaseReady) {
    debugPrint('$reason: $error');
    return;
  }
  try {
    await FirebaseCrashlytics.instance.recordError(
      error,
      stack,
      reason: reason,
      fatal: true,
    );
  } catch (recordingError) {
    debugPrint('Crashlytics fatal error recording failed: $recordingError');
  }
}

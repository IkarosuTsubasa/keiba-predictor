import 'dart:async';
import 'dart:ui';

import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/material.dart';

import 'src/app.dart';
import 'src/services/ad_consent_service.dart';
import 'src/services/native_platform_bridge.dart';
import 'src/services/push_notification_service.dart';
import 'src/services/reward_ad_requirement.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await NativePlatformBridge.initialize();
  await Firebase.initializeApp();
  await RemoteConfigRewardAdRequirement.instance.initialize();

  FlutterError.onError = (details) {
    FlutterError.presentError(details);
    FirebaseCrashlytics.instance.recordFlutterFatalError(details);
  };
  PlatformDispatcher.instance.onError = (error, stack) {
    FirebaseCrashlytics.instance.recordError(error, stack, fatal: true);
    return true;
  };

  await _initializePushNotifications();
  runApp(
    KeibaApp(
      navigatorObservers: [
        FirebaseAnalyticsObserver(analytics: FirebaseAnalytics.instance),
      ],
    ),
  );
  unawaited(RemoteConfigRewardAdRequirement.instance.refresh());
  unawaited(_initializePostLaunchPrivacyAndPermissions());
}

Future<void> _initializePostLaunchPrivacyAndPermissions() async {
  try {
    await AdConsentService.instance.initialize();
  } catch (error, stack) {
    await FirebaseCrashlytics.instance.recordError(
      error,
      stack,
      reason: 'Ad consent initialization failed',
      fatal: false,
    );
  }
  await Future<void>.delayed(const Duration(milliseconds: 350));
  await _initializePushNotifications();
  await PushNotificationService.instance.requestPermissionOnFirstLaunch();
}

Future<void> _initializePushNotifications() async {
  try {
    await PushNotificationService.instance.initialize();
  } catch (error, stack) {
    await FirebaseCrashlytics.instance.recordError(
      error,
      stack,
      reason: 'Push notification initialization failed',
      fatal: false,
    );
  }
}

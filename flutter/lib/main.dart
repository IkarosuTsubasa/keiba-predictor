import 'dart:async';
import 'dart:ui';

import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/material.dart';
import 'package:google_mobile_ads/google_mobile_ads.dart';

import 'src/app.dart';
import 'src/services/android_platform_bridge.dart';
import 'src/services/push_notification_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await AndroidPlatformBridge.initialize();
  await Firebase.initializeApp();

  FlutterError.onError = (details) {
    FlutterError.presentError(details);
    FirebaseCrashlytics.instance.recordFlutterFatalError(details);
  };
  PlatformDispatcher.instance.onError = (error, stack) {
    FirebaseCrashlytics.instance.recordError(error, stack, fatal: true);
    return true;
  };

  await MobileAds.instance.initialize();
  await PushNotificationService.instance.initialize();
  runApp(
    KeibaApp(
      navigatorObservers: [
        FirebaseAnalyticsObserver(analytics: FirebaseAnalytics.instance),
      ],
    ),
  );
  unawaited(
    Future<void>.delayed(const Duration(milliseconds: 350)).then(
      (_) => PushNotificationService.instance.requestPermissionOnFirstLaunch(),
    ),
  );
}

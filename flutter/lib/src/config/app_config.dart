import 'package:flutter/foundation.dart';

import '../services/native_platform_bridge.dart';

class AppConfig {
  AppConfig._();

  static const productionBaseWebUrl = 'https://www.ikaimo-ai.com/keiba';
  static const _baseWebUrlOverride = String.fromEnvironment(
    'KEIBA_BASE_WEB_URL',
  );

  static bool get notificationsEnabled =>
      NativePlatformBridge.isSupported &&
      NativePlatformBridge.runtimeConfig != null;

  static bool get adsEnabled => notificationsEnabled;

  static String get baseWebUrl {
    if (_baseWebUrlOverride.trim().isNotEmpty) {
      return _baseWebUrlOverride.trim().replaceFirst(RegExp(r'/+$'), '');
    }
    final nativeUrl = NativePlatformBridge.runtimeConfig?.baseWebUrl.trim();
    if (nativeUrl != null && nativeUrl.isNotEmpty) {
      return nativeUrl.replaceFirst(RegExp(r'/+$'), '');
    }
    if (!kDebugMode) {
      return productionBaseWebUrl;
    }
    if (defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8000/keiba';
    }
    return 'http://localhost:8000/keiba';
  }
}

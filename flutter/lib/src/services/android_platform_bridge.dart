import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class AndroidRuntimeConfig {
  const AndroidRuntimeConfig({
    required this.baseWebUrl,
    required this.bannerAdUnitId,
    required this.nativeMoreAdUnitId,
    required this.fcmTopic,
    required this.fcmCentralTopic,
    required this.fcmLocalTopic,
    required this.userAgent,
  });

  factory AndroidRuntimeConfig.fromMap(Map<Object?, Object?> values) {
    String value(String key) => values[key]?.toString().trim() ?? '';
    return AndroidRuntimeConfig(
      baseWebUrl: value('baseWebUrl'),
      bannerAdUnitId: value('bannerAdUnitId'),
      nativeMoreAdUnitId: value('nativeMoreAdUnitId'),
      fcmTopic: value('fcmTopic'),
      fcmCentralTopic: value('fcmCentralTopic'),
      fcmLocalTopic: value('fcmLocalTopic'),
      userAgent: value('userAgent'),
    );
  }

  final String baseWebUrl;
  final String bannerAdUnitId;
  final String nativeMoreAdUnitId;
  final String fcmTopic;
  final String fcmCentralTopic;
  final String fcmLocalTopic;
  final String userAgent;
}

class AndroidPlatformBridge {
  AndroidPlatformBridge._();

  static const _channel = MethodChannel('com.ikaimo.keiba.app/platform');

  static AndroidRuntimeConfig? runtimeConfig;

  static bool get _isAndroid =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  static Future<void> initialize() async {
    if (!_isAndroid) return;
    try {
      final values = await _channel.invokeMapMethod<Object?, Object?>(
        'getRuntimeConfig',
      );
      if (values != null) {
        runtimeConfig = AndroidRuntimeConfig.fromMap(values);
      }
    } on PlatformException {
      runtimeConfig = null;
    } on MissingPluginException {
      runtimeConfig = null;
    }
  }

  static Future<Map<Object?, Object?>?> getNotificationSettings() async {
    if (!_isAndroid) return null;
    try {
      return await _channel.invokeMapMethod<Object?, Object?>(
        'getNotificationSettings',
      );
    } on PlatformException {
      return null;
    } on MissingPluginException {
      return null;
    }
  }

  static Future<void> setNotificationSettings({
    required bool enabled,
    required bool centralEnabled,
    required bool localEnabled,
  }) async {
    if (!_isAndroid) return;
    try {
      await _channel.invokeMethod<void>('setNotificationSettings', {
        'enabled': enabled,
        'central_enabled': centralEnabled,
        'local_enabled': localEnabled,
      });
    } on PlatformException {
      // Flutter 側のミラー設定は保存済みなので、次回起動時に再同期する。
    } on MissingPluginException {
      // Widget test や未対応プラットフォームでは何もしない。
    }
  }

  static Future<bool> shouldRequestNotificationPermission() async {
    if (!_isAndroid) return false;
    try {
      return await _channel.invokeMethod<bool>(
            'shouldRequestNotificationPermission',
          ) ??
          false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      return false;
    }
  }

  static Future<void> markNotificationPermissionPrompted() async {
    if (!_isAndroid) return;
    try {
      await _channel.invokeMethod<void>('markNotificationPermissionPrompted');
    } on PlatformException {
      // 権限ダイアログ自体は Android が管理するため続行する。
    } on MissingPluginException {
      // Widget test や未対応プラットフォームでは何もしない。
    }
  }
}

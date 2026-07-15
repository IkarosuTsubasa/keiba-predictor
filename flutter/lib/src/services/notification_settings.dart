import 'dart:async';

import 'package:shared_preferences/shared_preferences.dart';

import 'android_platform_bridge.dart';

enum NotificationScope { central, local, general }

class NotificationSettings {
  const NotificationSettings({
    this.enabled = true,
    this.centralEnabled = true,
    this.localEnabled = true,
  });

  final bool enabled;
  final bool centralEnabled;
  final bool localEnabled;

  bool allows(NotificationScope scope) {
    if (!enabled) return false;
    return switch (scope) {
      NotificationScope.central => centralEnabled,
      NotificationScope.local => localEnabled,
      NotificationScope.general => true,
    };
  }

  NotificationSettings copyWith({
    bool? enabled,
    bool? centralEnabled,
    bool? localEnabled,
  }) {
    return NotificationSettings(
      enabled: enabled ?? this.enabled,
      centralEnabled: centralEnabled ?? this.centralEnabled,
      localEnabled: localEnabled ?? this.localEnabled,
    );
  }

  static NotificationScope scopeFromData(Map<String, dynamic> data) {
    const keys = [
      'notification_scope',
      'scope_group',
      'scope_key',
      'race_scope',
      'source',
    ];
    for (final key in keys) {
      final value = data[key]?.toString().trim().toLowerCase() ?? '';
      if (value == 'local' || value == 'nar' || value.contains('地方')) {
        return NotificationScope.local;
      }
      if (value.startsWith('central') ||
          value == 'jra' ||
          value.contains('中央')) {
        return NotificationScope.central;
      }
    }
    return NotificationScope.general;
  }
}

class NotificationSettingsRepository {
  NotificationSettingsRepository({SharedPreferencesAsync? preferences})
    : _preferences = preferences ?? SharedPreferencesAsync();

  static const _enabledKey = 'notification.enabled';
  static const _centralKey = 'notification.central_enabled';
  static const _localKey = 'notification.local_enabled';
  static final _changes = StreamController<NotificationSettings>.broadcast();

  static Stream<NotificationSettings> get changes => _changes.stream;

  final SharedPreferencesAsync _preferences;

  Future<NotificationSettings> load() async {
    final legacy = await AndroidPlatformBridge.getNotificationSettings();
    if (legacy != null) {
      final settings = NotificationSettings(
        enabled: legacy['enabled'] as bool? ?? true,
        centralEnabled: legacy['central_enabled'] as bool? ?? true,
        localEnabled: legacy['local_enabled'] as bool? ?? true,
      );
      await _saveMirror(settings);
      return settings;
    }
    return loadMirror();
  }

  Future<NotificationSettings> loadMirror() async {
    final values = await Future.wait<bool?>([
      _preferences.getBool(_enabledKey),
      _preferences.getBool(_centralKey),
      _preferences.getBool(_localKey),
    ]);
    return NotificationSettings(
      enabled: values[0] ?? true,
      centralEnabled: values[1] ?? true,
      localEnabled: values[2] ?? true,
    );
  }

  Future<void> save(NotificationSettings settings) async {
    await _saveMirror(settings);
    await AndroidPlatformBridge.setNotificationSettings(
      enabled: settings.enabled,
      centralEnabled: settings.centralEnabled,
      localEnabled: settings.localEnabled,
    );
    _changes.add(settings);
  }

  Future<void> _saveMirror(NotificationSettings settings) async {
    await Future.wait<void>([
      _preferences.setBool(_enabledKey, settings.enabled),
      _preferences.setBool(_centralKey, settings.centralEnabled),
      _preferences.setBool(_localKey, settings.localEnabled),
    ]);
  }
}

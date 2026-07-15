import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/services/notification_settings.dart';

void main() {
  test('通知设置可以独立更新各个开关', () {
    const initial = NotificationSettings();
    final disabled = initial.copyWith(enabled: false);
    final localOnly = disabled.copyWith(
      centralEnabled: false,
      localEnabled: true,
    );

    expect(disabled.enabled, isFalse);
    expect(disabled.centralEnabled, isTrue);
    expect(localOnly.centralEnabled, isFalse);
    expect(localOnly.localEnabled, isTrue);
  });

  test('通知范围识别与原生版规则一致', () {
    expect(
      NotificationSettings.scopeFromData({'scope_group': 'central_main'}),
      NotificationScope.central,
    );
    expect(
      NotificationSettings.scopeFromData({'race_scope': '地方競馬'}),
      NotificationScope.local,
    );
    expect(
      NotificationSettings.scopeFromData({'source': 'daily_report'}),
      NotificationScope.general,
    );
  });

  test('总开关和分区开关共同控制通知', () {
    const enabled = NotificationSettings(localEnabled: false);
    const disabled = NotificationSettings(enabled: false);

    expect(enabled.allows(NotificationScope.central), isTrue);
    expect(enabled.allows(NotificationScope.local), isFalse);
    expect(disabled.allows(NotificationScope.general), isFalse);
  });
}

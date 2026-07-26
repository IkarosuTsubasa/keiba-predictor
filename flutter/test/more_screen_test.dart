import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/screens/more_screen.dart';
import 'package:ikaimo_keiba/src/services/ad_consent_service.dart';
import 'package:ikaimo_keiba/src/services/notification_settings.dart';

void main() {
  testWidgets('更多页面沿用现有日语入口文案', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: MoreScreen(showNotificationSettings: false)),
      ),
    );

    expect(find.text('プライバシーポリシー'), findsOneWidget);
    expect(find.text('利用規約'), findsOneWidget);
    expect(find.text('免責事項'), findsOneWidget);
    expect(find.text('お問い合わせ'), findsOneWidget);
    expect(find.text('通知設定'), findsNothing);
  });

  testWidgets('必要な地域では広告のプライバシー設定入口を表示する', (tester) async {
    final consentService = AdConsentService(adsEnabled: () => false);
    consentService.privacyOptionsRequired.value = true;

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: MoreScreen(
            showNotificationSettings: false,
            adConsentService: consentService,
          ),
        ),
      ),
    );

    expect(find.text('広告のプライバシー設定'), findsOneWidget);
    expect(find.text('広告に関する同意内容を確認・変更できます'), findsOneWidget);

    await tester.pumpWidget(const SizedBox.shrink());
    consentService.dispose();
  });

  testWidgets('FCMトークン入口はバージョンを7回押すまで表示しない', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: MoreScreen(showNotificationSettings: false)),
      ),
    );

    expect(find.byKey(const Key('copy-fcm-token-button')), findsNothing);
    for (var index = 0; index < 6; index += 1) {
      await tester.tap(find.byKey(const Key('app-version-label')));
    }
    await tester.pump();
    expect(find.byKey(const Key('copy-fcm-token-button')), findsNothing);

    await tester.tap(find.byKey(const Key('app-version-label')));
    await tester.pump();

    expect(find.byKey(const Key('developer-tools-card')), findsOneWidget);
    expect(find.byKey(const Key('copy-fcm-token-button')), findsOneWidget);
    expect(find.text('テスト機能を有効にしました。'), findsOneWidget);
  });

  testWidgets('端末側で通知が拒否されている場合は設定への導線を表示する', (tester) async {
    debugDefaultTargetPlatformOverride = TargetPlatform.iOS;
    addTearDown(() => debugDefaultTargetPlatformOverride = null);
    const channel = MethodChannel('com.ikaimo.keiba.app/platform');
    var openedSettings = false;
    tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(channel, (
      call,
    ) async {
      return switch (call.method) {
        'getNotificationSettings' => <String, bool>{
          'enabled': true,
          'central_enabled': true,
          'local_enabled': true,
        },
        'getNotificationAuthorizationStatus' => 'denied',
        'openNotificationSettings' => openedSettings = true,
        _ => null,
      };
    });
    addTearDown(
      () => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
        channel,
        null,
      ),
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: MoreScreen(
            showNotificationSettings: true,
            notificationSettingsStore: _MemoryNotificationSettingsStore(),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(
      find.byKey(const Key('notification-permission-warning')),
      findsOneWidget,
    );
    expect(find.text('端末側で通知がオフになっています'), findsOneWidget);

    await tester.tap(find.text('設定を開く'));
    await tester.pump();
    expect(openedSettings, isTrue);
    debugDefaultTargetPlatformOverride = null;
  });
}

class _MemoryNotificationSettingsStore implements NotificationSettingsStore {
  NotificationSettings value = const NotificationSettings();

  @override
  Future<NotificationSettings> load() async => value;

  @override
  Future<void> save(NotificationSettings settings) async {
    value = settings;
  }
}

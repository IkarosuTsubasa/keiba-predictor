import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/screens/more_screen.dart';
import 'package:ikaimo_keiba/src/services/ad_consent_service.dart';

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
}

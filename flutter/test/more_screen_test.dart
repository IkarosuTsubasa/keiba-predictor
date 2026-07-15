import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/screens/more_screen.dart';

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
}

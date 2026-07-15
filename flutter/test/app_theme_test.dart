import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/app.dart';

void main() {
  testWidgets('iOS 顶栏标题与 Android 一样从左侧 16dp 开始', (tester) async {
    debugDefaultTargetPlatformOverride = TargetPlatform.iOS;
    try {
      await tester.pumpWidget(
        MaterialApp(
          theme: KeibaApp.theme,
          home: Scaffold(
            appBar: AppBar(
              toolbarHeight: 86,
              titleSpacing: 16,
              title: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [Text('レース一覧'), Text('当日公開レースと詳細ページを表示します')],
              ),
            ),
          ),
        ),
      );

      expect(tester.getTopLeft(find.text('レース一覧')).dx, 16);
    } finally {
      debugDefaultTargetPlatformOverride = null;
    }
  });
}

import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/web/app_web.dart';

void main() {
  const baseUrl = 'https://www.ikaimo-ai.com/keiba';
  final baseUri = Uri.parse(baseUrl);

  group('AppWeb', () {
    test('应用标记会保留原有查询参数', () {
      final result = AppWeb.withAppMarker(
        Uri.parse('$baseUrl?date=2026-07-15'),
      );

      expect(result.queryParameters['date'], '2026-07-15');
      expect(result.queryParameters['app'], '1');
    });

    test('生成三个顶层页面地址', () {
      expect(
        AppWeb.topLevelUri(baseUrl, AppSection.races).toString(),
        '$baseUrl?app=1',
      );
      expect(
        AppWeb.topLevelUri(baseUrl, AppSection.history).toString(),
        '$baseUrl/history?app=1',
      );
      expect(
        AppWeb.topLevelUri(baseUrl, AppSection.reports).toString(),
        '$baseUrl/reports?app=1',
      );
    });

    test('站内相对地址会规范化并追加应用标记', () {
      final result = AppWeb.normalizeInAppUrl(
        '/keiba/reports/daily-1?date=2026-07-15',
        baseUrl,
      );

      expect(
        result.toString(),
        '$baseUrl/reports/daily-1?date=2026-07-15&app=1',
      );
    });

    test('拒绝站外地址', () {
      expect(
        AppWeb.normalizeInAppUrl('https://example.com/keiba', baseUrl),
        isNull,
      );
    });

    test('详情页标题及栏目判断正确', () {
      final raceUri = Uri.parse('$baseUrl/race/202607150101?app=1');
      final reportUri = Uri.parse('$baseUrl/reports/daily-1?app=1');

      expect(AppWeb.sectionFor(raceUri, baseUri), AppSection.races);
      expect(AppWeb.chromeFor(raceUri, baseUri).title, 'レース詳細');
      expect(AppWeb.sectionFor(reportUri, baseUri), AppSection.reports);
      expect(AppWeb.chromeFor(reportUri, baseUri).title, '日報詳細');
    });
  });
}

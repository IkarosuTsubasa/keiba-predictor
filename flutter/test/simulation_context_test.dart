import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ikaimo_keiba/src/models/simulation_context.dart';
import 'package:ikaimo_keiba/src/services/simulation_settings.dart';

void main() {
  final base = Uri.parse('https://www.ikaimo-ai.com/keiba');
  const path = '/keiba/open/simulator/202605050101';
  test('同场链接保留输入版本和种子', () {
    final parsed = SimulationContext.fromUri(
      Uri.parse(
        'https://www.ikaimo-ai.com$path?date=2026-11-01&iv=abcdef123456abcdef123456&seed=42&source=web_detail',
      ),
      base,
    )!;
    expect(parsed.raceId, '202605050101');
    expect(parsed.seed, 42);
    expect(
      parsed.embedUri(base.toString()).queryParameters['iv'],
      'abcdef123456abcdef123456',
    );
    expect(parsed.embedUri(base.toString()).path, '/keiba/simulator/embed');
  });
  test('拒绝外站、错误日期、版本和种子', () {
    for (final link in [
      'https://example.com$path?date=2026-11-01',
      '$path?date=2026-02-30',
      '$path?date=2026-11-01&seed=-1',
      '$path?date=2026-11-01&iv=../../secret',
      '$path?date=2026-11-01&mv=unknown',
      '$path?date=2026-11-01&seed=abc',
    ]) {
      expect(
        SimulationContext.fromUri(Uri.parse(link), base),
        isNull,
        reason: link,
      );
    }
    expect(
      SimulationContext.fromUri(Uri.parse('/keiba/reports/test'), base),
      isNull,
    );
  });
  test('单场方案不会成为下一场的全局偏好', () async {
    SharedPreferences.setMockInitialValues({});
    final storage = SimulationSettings();
    const first = SimulationContext(
      raceId: '202605050101',
      date: '2026-11-01',
      inputVersion: 'abcdef123456abcdef123456',
    );
    const second = SimulationContext(
      raceId: '202605050102',
      date: '2026-11-01',
      inputVersion: 'abcdef123456abcdef123456',
    );
    await storage.saveScene(first, {
      'settings': {
        'pace': 'fast',
        'variation': 1.7,
        'seed': 99,
        'speed': 8,
        'view': 'pack',
      },
      'time': 25,
    });
    expect(await storage.load('scene.${second.storageKey}'), isNull);
    expect(await storage.load('preferences'), {'speed': 8, 'view': 'pack'});
    expect((await storage.load('scene.${first.storageKey}'))!['time'], 25);
    expect((await storage.load('recent'))!['raceId'], first.raceId);
  });
}

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ikaimo_keiba/src/screens/simulation_home_screen.dart';
import 'package:ikaimo_keiba/src/services/simulation_service.dart';

class _FakeService extends SimulationService {
  @override
  Future<Map<String, dynamic>> list(String date) async => {
    'races': [
      {
        'race_id': '202605050101',
        'race_name': '発走予定の中央レース',
        'course': '東京',
        'scope_key': 'central_turf',
        'availability': 'ready',
        'race_state': 'scheduled',
      },
      {
        'race_id': '202605050102',
        'race_name': '時刻が不明のレース',
        'course': '東京',
        'scope_key': 'central_turf',
        'availability': 'ready',
        'race_state': 'unknown',
      },
      {
        'race_id': '202645110101',
        'race_name': '準備中の地方レース',
        'course': '川崎',
        'scope_key': 'local',
        'availability': 'preparing',
        'race_state': 'scheduled',
      },
    ],
  };
}

void main() {
  testWidgets('发走时间未知的比赛在全部视图中保留，地方准备中比赛可筛选', (tester) async {
    SharedPreferences.setMockInitialValues({});
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SimulationHomeScreen(active: true, service: _FakeService()),
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.textContaining('発走予定の中央レース'), findsOneWidget);
    expect(find.textContaining('2R · 時刻が不明のレース'), findsNothing);
    await tester.tap(find.text('発走予定のレースのみ'));
    await tester.pumpAndSettle();
    expect(find.textContaining('2R · 時刻が不明のレース'), findsOneWidget);
    await tester.tap(find.text('地方'));
    await tester.pumpAndSettle();
    expect(find.textContaining('発走予定の中央レース'), findsNothing);
    expect(find.textContaining('準備中の地方レース'), findsOneWidget);
    expect(find.textContaining('データ準備中'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}

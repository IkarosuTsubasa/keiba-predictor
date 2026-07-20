import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/models/custom_prediction.dart';
import 'package:ikaimo_keiba/src/screens/custom_prediction_screen.dart';
import 'package:ikaimo_keiba/src/services/custom_prediction_service.dart';
import 'package:ikaimo_keiba/src/services/custom_prediction_settings.dart';

void main() {
  testWidgets('マイ予測页面只显示保存参数后的结果', (tester) async {
    await tester.binding.setSurfaceSize(const Size(430, 1100));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final dataSource = _FakeDataSource();

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: CustomPredictionScreen(
            dataSource: dataSource,
            settingsStore: _MemorySettingsStore(
              isConfigured: true,
              parameters: const CustomPredictionParameters(abilityLevel: 2),
            ),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('保存した設定で予測中'), findsOneWidget);
    expect(find.text('設定を変更'), findsOneWidget);
    expect(find.byType(Slider), findsNothing);
    expect(find.text('日付を選ぶ'), findsNothing);
    expect(dataSource.fetchCount, 1);
    expect(find.text('99.9'), findsNothing);

    await tester.scrollUntilVisible(
      find.text('マイ予測結果'),
      300,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('マイ予測結果'), findsOneWidget);
    expect(find.text('ホースA'), findsOneWidget);
    expect(find.text('◎'), findsOneWidget);
    expect(find.text('99.9'), findsNothing);
  });

  testWidgets('首次保存设置后返回マイ予測并开始取数', (tester) async {
    await tester.binding.setSurfaceSize(const Size(430, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final dataSource = _FakeDataSource();
    final settingsStore = _MemorySettingsStore();

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: CustomPredictionScreen(
            dataSource: dataSource,
            settingsStore: settingsStore,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('予測設定がまだ保存されていません'), findsOneWidget);
    expect(find.text('設定を始める'), findsOneWidget);
    expect(find.text('マイ予測結果'), findsNothing);
    expect(dataSource.fetchCount, 0);

    await tester.tap(find.text('設定を始める'));
    await tester.pumpAndSettle();

    expect(find.text('予測設定'), findsOneWidget);
    expect(find.byType(Slider), findsNWidgets(7));
    expect(find.text('マイ予測結果'), findsNothing);

    await tester.tap(find.text('保存して予測を見る'));
    await tester.pumpAndSettle();

    expect(settingsStore.isConfigured, isTrue);
    expect(dataSource.fetchCount, 1);
    expect(find.text('保存した設定で予測中'), findsOneWidget);
    expect(find.text('マイ予測結果'), findsOneWidget);
    expect(find.byType(Slider), findsNothing);
  });

  testWidgets('当天没有比赛时显示等待提示且不能切换日期', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: CustomPredictionScreen(
            dataSource: _EmptyDataSource(),
            settingsStore: _MemorySettingsStore(isConfigured: true),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('日付を選ぶ'), findsNothing);
    expect(find.text('本日のレース情報はまだ準備できていません。しばらくお待ちください。'), findsOneWidget);
  });
}

class _FakeDataSource implements CustomPredictionDataSource {
  int fetchCount = 0;

  @override
  Future<CustomPredictionDay> fetch() async {
    fetchCount += 1;
    return CustomPredictionDay.fromApi({
      'ok': true,
      'data': {
        'target_date': '2026-07-19',
        'target_date_label': '2026年7月19日',
        'fallback_notice': '',
        'races': [
          {
            'race_id': '202601010101',
            'race_title': '東京1R',
            'race_name': 'テスト競走',
            'course': '東京',
            'distance_label': '芝1600m',
            'track_condition': '良',
            'horses': [
              {
                'horse_no': 1,
                'horse_name': 'ホースA',
                'total_score': 99.9,
                'factors': {
                  'base': 15,
                  'pedigree': 1,
                  'race_level_adjustment': 0.2,
                  'pace_adjustment': 0.5,
                  'ability': 8,
                  'recent_quality': 7,
                  'trend': 6,
                  'condition': 5,
                  'race_level': 4,
                  'pace_jockey': 3,
                  'risk': -2,
                },
                'default_weights': {
                  'pedigree': 0.2,
                  'race_level': 1,
                  'pace': 0.2,
                },
              },
              {
                'horse_no': 2,
                'horse_name': 'ホースB',
                'factors': {'base': 12},
                'default_weights': {},
              },
            ],
          },
        ],
      },
    });
  }
}

class _MemorySettingsStore implements CustomPredictionSettingsStore {
  _MemorySettingsStore({
    this.parameters = const CustomPredictionParameters(),
    this.isConfigured = false,
  });

  CustomPredictionParameters parameters;
  bool isConfigured;

  @override
  Future<CustomPredictionSettingsSnapshot> load() async {
    return CustomPredictionSettingsSnapshot(
      parameters: parameters,
      isConfigured: isConfigured,
    );
  }

  @override
  Future<void> save(CustomPredictionParameters parameters) async {
    this.parameters = parameters;
    isConfigured = true;
  }
}

class _EmptyDataSource implements CustomPredictionDataSource {
  @override
  Future<CustomPredictionDay> fetch() async {
    return const CustomPredictionDay(
      targetDate: '2026-07-20',
      targetDateLabel: '2026年7月20日',
      fallbackNotice: '',
      races: [],
    );
  }
}

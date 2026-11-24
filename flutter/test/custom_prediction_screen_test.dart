import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/models/custom_prediction.dart';
import 'package:ikaimo_keiba/src/screens/custom_prediction_screen.dart';
import 'package:ikaimo_keiba/src/services/custom_prediction_service.dart';
import 'package:ikaimo_keiba/src/services/custom_prediction_settings.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  testWidgets('My予測页面只显示保存参数后的结果', (tester) async {
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

    expect(find.text('あなたの予測スタイル'), findsOneWidget);
    expect(find.byKey(const Key('unlock-custom-prediction')), findsNothing);
    expect(find.text('自分の読みを\n勝負の印へ'), findsOneWidget);
    expect(find.text('設定を変更'), findsOneWidget);
    expect(find.byType(Slider), findsNothing);
    expect(find.text('日付を選ぶ'), findsNothing);
    expect(dataSource.fetchCount, 1);
    expect(find.text('99.9'), findsNothing);

    await tester.scrollUntilVisible(
      find.text('My予測の結論'),
      300,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('My予測の結論'), findsOneWidget);
    expect(find.text('ホースA'), findsOneWidget);
    expect(find.text('◎'), findsOneWidget);
    expect(find.text('99.9'), findsNothing);
  });

  testWidgets('首次保存设置后返回My予測并开始取数', (tester) async {
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

    expect(find.text('まずは、あなたの勝負基準を'), findsOneWidget);
    expect(find.text('勝負基準をつくる'), findsOneWidget);
    expect(find.text('My予測の結論'), findsNothing);
    expect(dataSource.fetchCount, 0);

    await tester.tap(find.text('勝負基準をつくる'));
    await tester.pumpAndSettle();

    expect(find.text('予測設定'), findsOneWidget);
    expect(find.text('現在のバランスプロフィール'), findsNothing);
    expect(find.text('My予測'), findsOneWidget);
    expect(find.byType(Slider), findsNWidgets(7));
    expect(find.text('My予測の結論'), findsNothing);

    await tester.drag(find.byType(Slider).first, const Offset(220, 0));
    await tester.pumpAndSettle();
    await tester.tap(find.text('このバランスを保存'));
    await tester.pumpAndSettle();

    expect(settingsStore.isConfigured, isTrue);
    expect(settingsStore.parameters.abilityLevel, 2);
    expect(dataSource.fetchCount, 1);
    expect(find.text('あなたの予測スタイル'), findsOneWidget);
    expect(find.text('My予測の結論'), findsOneWidget);
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
    expect(
      find.text('予測できるレース情報はまだ公開されていません。情報が揃うまで、予測バランスを整えてお待ちください。'),
      findsOneWidget,
    );
    expect(find.text('次の勝負に備えましょう'), findsOneWidget);
    expect(find.text('レース情報を更新'), findsOneWidget);
  });

  testWidgets('旧版广告已过期时仍可直接使用保存的预测设置', (tester) async {
    await tester.binding.setSurfaceSize(const Size(430, 1100));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final dataSource = _FakeDataSource();
    final preferences = _LegacyPreferences({
      'custom_prediction.configured': true,
      'custom_prediction.ability_level': 2,
      'custom_prediction.rewarded_access_until_epoch_ms': 1,
    });
    final settingsStore = CustomPredictionSettingsRepository(
      preferences: preferences,
    );

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

    expect(find.byKey(const Key('unlock-custom-prediction')), findsNothing);
    expect(find.text('My予測の結論'), findsOneWidget);
    expect(dataSource.fetchCount, 1);
    final savedSettings = await settingsStore.load();
    expect(savedSettings.parameters.abilityLevel, 2);

    await tester.pump(const Duration(hours: 1, minutes: 1));
    expect(find.text('My予測の結論'), findsOneWidget);
  });

  testWidgets('レース选择后分阶段演算并在完成后显示对应结果', (tester) async {
    await tester.binding.setSurfaceSize(const Size(430, 1100));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: CustomPredictionScreen(
            dataSource: _FakeDataSource(),
            settingsStore: _MemorySettingsStore(isConfigured: true),
            calculationDuration: () => const Duration(seconds: 4),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    final raceSelector = find.byKey(const ValueKey('東京-202601010101'));
    await tester.scrollUntilVisible(
      raceSelector,
      250,
      scrollable: find.byType(Scrollable).first,
    );
    await tester.tap(raceSelector);
    await tester.pumpAndSettle();
    await tester.tap(find.text('東京2R  テスト競走2'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));

    expect(find.text('あなたの読みを反映中'), findsOneWidget);
    expect(find.text('My予測の結論'), findsNothing);
    final calculationMessage = find.descendant(
      of: find.byKey(const Key('prediction-calculation-message')),
      matching: find.byType(Text),
    );
    final firstMessage = tester.widget<Text>(calculationMessage).data;

    await tester.pump(const Duration(milliseconds: 1100));
    await tester.pump(const Duration(milliseconds: 250));
    final secondMessage = tester.widget<Text>(calculationMessage).data;
    expect(secondMessage, isNot(firstMessage));

    await tester.pump(const Duration(milliseconds: 1100));
    await tester.pump(const Duration(milliseconds: 250));
    final thirdMessage = tester.widget<Text>(calculationMessage).data;
    expect(thirdMessage, isNot(secondMessage));

    await tester.pump(const Duration(milliseconds: 1000));
    await tester.pump(const Duration(milliseconds: 300));
    expect(find.text('あなたの読みを反映中'), findsNothing);
    expect(find.text('My予測の結論'), findsOneWidget);
    expect(find.text('ホースC'), findsOneWidget);
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
          {
            'race_id': '202601010102',
            'race_title': '東京2R',
            'race_name': 'テスト競走2',
            'course': '東京',
            'distance_label': 'ダート1400m',
            'track_condition': '稍重',
            'horses': [
              {
                'horse_no': 3,
                'horse_name': 'ホースC',
                'factors': {'base': 18},
                'default_weights': {},
              },
              {
                'horse_no': 4,
                'horse_name': 'ホースD',
                'factors': {'base': 10},
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

class _LegacyPreferences extends Fake implements SharedPreferencesAsync {
  _LegacyPreferences(this.values);

  final Map<String, Object> values;

  @override
  Future<bool?> getBool(String key) async => values[key] as bool?;

  @override
  Future<int?> getInt(String key) async => values[key] as int?;
}

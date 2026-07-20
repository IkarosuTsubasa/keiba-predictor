import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/models/custom_prediction.dart';
import 'package:ikaimo_keiba/src/screens/custom_prediction_screen.dart';
import 'package:ikaimo_keiba/src/services/custom_prediction_access.dart';
import 'package:ikaimo_keiba/src/services/custom_prediction_service.dart';
import 'package:ikaimo_keiba/src/services/custom_prediction_settings.dart';

void main() {
  final now = DateTime.utc(2026, 7, 20, 12);

  String accessUntilLabel(DateTime value) {
    final local = value.toLocal();
    final minute = local.minute.toString().padLeft(2, '0');
    return '${local.hour}:$minuteまで利用できます';
  }

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
            accessStore: _MemoryAccessStore(
              unlockedUntil: now.add(const Duration(hours: 1)),
            ),
            now: () => now,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('My予測を利用中'), findsOneWidget);
    expect(
      find.text(accessUntilLabel(now.add(const Duration(hours: 1)))),
      findsOneWidget,
    );
    expect(find.text('今日の予測を\n自分の基準で'), findsOneWidget);
    expect(find.text('設定を変更'), findsOneWidget);
    expect(find.byType(Slider), findsNothing);
    expect(find.text('日付を選ぶ'), findsNothing);
    expect(dataSource.fetchCount, 1);
    expect(find.text('99.9'), findsNothing);

    await tester.scrollUntilVisible(
      find.text('My予測結果'),
      300,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('My予測結果'), findsOneWidget);
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
            accessStore: _MemoryAccessStore(
              unlockedUntil: now.add(const Duration(hours: 1)),
            ),
            now: () => now,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('予測設定がまだ保存されていません'), findsOneWidget);
    expect(find.text('設定を始める'), findsOneWidget);
    expect(find.text('My予測結果'), findsNothing);
    expect(dataSource.fetchCount, 0);

    await tester.tap(find.text('設定を始める'));
    await tester.pumpAndSettle();

    expect(find.text('予測設定'), findsOneWidget);
    expect(find.text('現在のバランスプロフィール'), findsNothing);
    expect(find.text('My予測'), findsOneWidget);
    expect(find.byType(Slider), findsNWidgets(7));
    expect(find.text('My予測結果'), findsNothing);

    await tester.drag(find.byType(Slider).first, const Offset(220, 0));
    await tester.pumpAndSettle();
    await tester.tap(find.text('このバランスを保存'));
    await tester.pumpAndSettle();

    expect(settingsStore.isConfigured, isTrue);
    expect(settingsStore.parameters.abilityLevel, 2);
    expect(dataSource.fetchCount, 1);
    expect(find.text('My予測を利用中'), findsOneWidget);
    expect(find.text('My予測結果'), findsOneWidget);
    expect(find.byType(Slider), findsNothing);
  });

  testWidgets('当天没有比赛时显示等待提示且不能切换日期', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: CustomPredictionScreen(
            dataSource: _EmptyDataSource(),
            settingsStore: _MemorySettingsStore(isConfigured: true),
            accessStore: _MemoryAccessStore(
              unlockedUntil: now.add(const Duration(hours: 1)),
            ),
            now: () => now,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('日付を選ぶ'), findsNothing);
    expect(find.text('本日のレース情報はまだ準備できていません。しばらくお待ちください。'), findsOneWidget);
    expect(find.text('今日の予測はもうすぐ'), findsOneWidget);
    expect(find.text('最新情報を確認'), findsOneWidget);
  });

  testWidgets('リワード獲得後に1時間解锁并开始获取预测', (tester) async {
    await tester.binding.setSurfaceSize(const Size(430, 1100));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final dataSource = _FakeDataSource();
    final accessStore = _MemoryAccessStore();

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: CustomPredictionScreen(
            dataSource: dataSource,
            settingsStore: _MemorySettingsStore(isConfigured: true),
            accessStore: accessStore,
            rewardedGateway: const _FakeRewardedGateway(
              CustomPredictionRewardResult.rewarded,
            ),
            now: () => now,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('動画を見て1時間使う'), findsOneWidget);
    expect(find.text('My予測結果'), findsNothing);
    expect(dataSource.fetchCount, 0);

    await tester.tap(find.text('動画を見て1時間使う'));
    await tester.pumpAndSettle();

    expect(accessStore.unlockedUntil, now.add(const Duration(hours: 1)));
    expect(
      find.text(accessUntilLabel(now.add(const Duration(hours: 1)))),
      findsOneWidget,
    );
    expect(find.text('My予測結果'), findsOneWidget);
    expect(dataSource.fetchCount, 1);
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
            accessStore: _MemoryAccessStore(
              unlockedUntil: now.add(const Duration(hours: 1)),
            ),
            now: () => now,
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

    expect(find.text('あなたの基準で再予測中'), findsOneWidget);
    expect(find.text('レース条件を読み解いています'), findsOneWidget);
    expect(find.text('My予測結果'), findsNothing);

    await tester.pump(const Duration(milliseconds: 1100));
    expect(find.text('あなたの予測バランスを反映しています'), findsOneWidget);

    await tester.pump(const Duration(milliseconds: 1400));
    expect(find.text('各馬を比較し、最終順位を導いています'), findsOneWidget);

    await tester.pump(const Duration(milliseconds: 1300));
    await tester.pump(const Duration(milliseconds: 300));
    expect(find.text('あなたの基準で再予測中'), findsNothing);
    expect(find.text('My予測結果'), findsOneWidget);
    expect(find.text('ホースC'), findsOneWidget);
  });

  testWidgets('動画を途中で閉じた場合はアクセスを付与しない', (tester) async {
    await tester.binding.setSurfaceSize(const Size(430, 1100));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final dataSource = _FakeDataSource();
    final accessStore = _MemoryAccessStore();

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: CustomPredictionScreen(
            dataSource: dataSource,
            settingsStore: _MemorySettingsStore(isConfigured: true),
            accessStore: accessStore,
            rewardedGateway: const _FakeRewardedGateway(
              CustomPredictionRewardResult.dismissed,
            ),
            now: () => now,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.text('動画を見て1時間使う'));
    await tester.pumpAndSettle();

    expect(accessStore.unlockedUntil, isNull);
    expect(dataSource.fetchCount, 0);
    expect(find.text('動画を最後まで見ると、1時間利用できます。'), findsOneWidget);
    expect(find.text('動画を見て1時間使う'), findsOneWidget);
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

class _MemoryAccessStore implements CustomPredictionAccessStore {
  _MemoryAccessStore({this.unlockedUntil});

  DateTime? unlockedUntil;

  @override
  Future<DateTime?> loadUnlockedUntil() async => unlockedUntil;

  @override
  Future<void> saveUnlockedUntil(DateTime value) async {
    unlockedUntil = value;
  }
}

class _FakeRewardedGateway implements CustomPredictionRewardedGateway {
  const _FakeRewardedGateway(this.result);

  final CustomPredictionRewardResult result;

  @override
  Future<CustomPredictionRewardResult> showRewardedAd() async => result;
}

import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/custom_prediction.dart';
import '../services/custom_prediction_service.dart';
import '../services/custom_prediction_settings.dart';
import 'custom_prediction_settings_screen.dart';

class CustomPredictionScreen extends StatefulWidget {
  const CustomPredictionScreen({
    this.dataSource,
    this.settingsStore,
    this.calculationDuration,
    super.key,
  });

  final CustomPredictionDataSource? dataSource;
  final CustomPredictionSettingsStore? settingsStore;
  final Duration Function()? calculationDuration;

  @override
  State<CustomPredictionScreen> createState() => _CustomPredictionScreenState();
}

class _CustomPredictionScreenState extends State<CustomPredictionScreen>
    with SingleTickerProviderStateMixin {
  static const _navy = Color(0xFF07111F);
  static const _accent = Color(0xFF0D6E6E);
  static const _muted = Color(0xFF66746F);
  static const _calculationMessageSets = <List<String>>[
    ['レース条件を丁寧に読み解いています', 'あなたの予測バランスを重ねています', '各馬の差を見極め、順位を整えています'],
    ['馬場状態に隠れたヒントを探しています', 'あなたが重視する条件を一頭ずつ反映しています', '最後の決め手を照らし合わせています'],
    ['スタート後のレース展開を組み立てています', 'あなたらしい勝ち筋へ評価を寄せています', 'ゴール前の攻防までじっくり比較しています'],
    ['人気だけでは見えない材料を拾っています', 'あなたのこだわりで評価を磨いています', '意外な一頭まで最終確認しています'],
    ['距離と馬場の相性を探っています', '7つの基準を一頭ずつ組み合わせています', '本命候補のわずかな差を見極めています'],
    ['勝負を分ける一瞬を探っています', 'あなたの感覚を予測へ変換しています', '伏兵を含めて、最後の結論を導いています'],
    ['各馬の強みと不安材料を整理しています', '重視する要素から評価を組み直しています', 'あなただけの印を慎重に選んでいます'],
  ];

  late final CustomPredictionDataSource _dataSource;
  late final CustomPredictionSettingsStore _settingsStore;
  late final Duration Function() _calculationDuration;
  final math.Random _random = math.Random();
  CustomPredictionParameters _parameters = const CustomPredictionParameters();
  CustomPredictionDay? _day;
  String? _selectedCourse;
  String? _selectedRaceId;
  String? _errorMessage;
  bool _settingsLoading = true;
  bool _configured = false;
  bool _loading = false;
  bool _calculatingPrediction = false;
  int _calculationPhase = 0;
  int _calculationSerial = 0;
  int _lastCalculationMessageSetIndex = -1;
  List<String> _activeCalculationMessages = _calculationMessageSets.first;
  Timer? _calculationPhaseTimer;
  Timer? _calculationCompleteTimer;
  late final AnimationController _calculationProgressController;
  int _requestSerial = 0;

  @override
  void initState() {
    super.initState();
    _dataSource = widget.dataSource ?? const CustomPredictionService();
    _settingsStore =
        widget.settingsStore ?? CustomPredictionSettingsRepository();
    _calculationDuration =
        widget.calculationDuration ??
        () => Duration(milliseconds: 3000 + _random.nextInt(2001));
    _calculationProgressController = AnimationController(vsync: this);
    unawaited(_initialize());
  }

  @override
  void dispose() {
    _cancelPredictionCalculation();
    _calculationProgressController.dispose();
    super.dispose();
  }

  Future<void> _initialize() async {
    CustomPredictionSettingsSnapshot? savedSettings;
    try {
      savedSettings = await _settingsStore.load();
    } catch (_) {
      savedSettings = null;
    }
    if (!mounted) return;
    setState(() {
      _parameters =
          savedSettings?.parameters ?? const CustomPredictionParameters();
      _configured = savedSettings?.isConfigured ?? false;
      _settingsLoading = false;
    });
    if (_configured) {
      await _loadDay();
    }
  }

  Future<void> _loadDay() async {
    if (!_configured) return;
    _cancelPredictionCalculation();
    final requestSerial = ++_requestSerial;
    setState(() {
      _loading = true;
      _errorMessage = null;
      _calculatingPrediction = false;
    });
    try {
      final day = await _dataSource.fetch();
      if (!mounted || requestSerial != _requestSerial) return;
      final firstRace = day.races.isEmpty ? null : day.races.first;
      setState(() {
        _day = day;
        _selectedCourse = firstRace?.course.isEmpty ?? true
            ? null
            : firstRace!.course;
        _selectedRaceId = firstRace?.raceId;
        _loading = false;
      });
    } on CustomPredictionException catch (error) {
      if (!mounted || requestSerial != _requestSerial) return;
      setState(() {
        _day = null;
        _loading = false;
        _errorMessage = error.message;
      });
    } catch (_) {
      if (!mounted || requestSerial != _requestSerial) return;
      setState(() {
        _day = null;
        _loading = false;
        _errorMessage = '予測データを取得できませんでした。';
      });
    }
  }

  void _cancelPredictionCalculation() {
    _calculationPhaseTimer?.cancel();
    _calculationCompleteTimer?.cancel();
    _calculationPhaseTimer = null;
    _calculationCompleteTimer = null;
    _calculationProgressController.stop();
  }

  List<String> _pickCalculationMessages() {
    final setCount = _calculationMessageSets.length;
    var index = _random.nextInt(setCount);
    if (setCount > 1 && index == _lastCalculationMessageSetIndex) {
      index = (index + 1 + _random.nextInt(setCount - 1)) % setCount;
    }
    _lastCalculationMessageSetIndex = index;
    return _calculationMessageSets[index];
  }

  void _beginPredictionCalculation({
    required String? course,
    required String? raceId,
  }) {
    if (raceId == null || raceId.isEmpty) {
      _cancelPredictionCalculation();
      setState(() {
        _selectedCourse = course;
        _selectedRaceId = raceId;
        _calculatingPrediction = false;
      });
      return;
    }
    if (course == _selectedCourse && raceId == _selectedRaceId) return;

    _cancelPredictionCalculation();
    final requestedDuration = _calculationDuration();
    final duration = requestedDuration > Duration.zero
        ? requestedDuration
        : const Duration(milliseconds: 1);
    final calculationMessages = _pickCalculationMessages();
    final calculationSerial = ++_calculationSerial;
    final phaseInterval = Duration(
      milliseconds: math.max(
        1,
        duration.inMilliseconds ~/ calculationMessages.length,
      ),
    );
    setState(() {
      _selectedCourse = course;
      _selectedRaceId = raceId;
      _calculatingPrediction = true;
      _calculationPhase = 0;
      _activeCalculationMessages = calculationMessages;
    });
    _calculationProgressController.duration = duration;
    unawaited(_calculationProgressController.forward(from: 0));

    _calculationPhaseTimer = Timer.periodic(phaseInterval, (timer) {
      if (!mounted || calculationSerial != _calculationSerial) {
        timer.cancel();
        return;
      }
      if (_calculationPhase >= _activeCalculationMessages.length - 1) {
        timer.cancel();
        return;
      }
      setState(() => _calculationPhase += 1);
    });
    _calculationCompleteTimer = Timer(duration, () {
      if (!mounted || calculationSerial != _calculationSerial) return;
      _calculationPhaseTimer?.cancel();
      _calculationPhaseTimer = null;
      _calculationCompleteTimer = null;
      setState(() => _calculatingPrediction = false);
    });
  }

  void _selectCourse(String course) {
    final races = _day?.races
        .where((race) => race.course == course)
        .toList(growable: false);
    _beginPredictionCalculation(
      course: course,
      raceId: races == null || races.isEmpty ? null : races.first.raceId,
    );
  }

  void _selectRace(String? raceId) {
    _beginPredictionCalculation(course: _selectedCourse, raceId: raceId);
  }

  Future<void> _openSettings() async {
    final parameters = await Navigator.of(context)
        .push<CustomPredictionParameters>(
          MaterialPageRoute(
            builder: (_) =>
                CustomPredictionSettingsScreen(initialParameters: _parameters),
          ),
        );
    if (parameters == null || !mounted) return;
    try {
      await _settingsStore.save(parameters);
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('予測設定を保存できませんでした。')));
      return;
    }
    if (!mounted) return;
    setState(() {
      _parameters = parameters;
      _configured = true;
    });
    if (_day == null) await _loadDay();
  }

  List<CustomPredictionRace> get _visibleRaces {
    final races = _day?.races ?? const <CustomPredictionRace>[];
    final course = _selectedCourse;
    if (course == null) return races;
    return races.where((race) => race.course == course).toList(growable: false);
  }

  CustomPredictionRace? get _selectedRace {
    final raceId = _selectedRaceId;
    for (final race in _visibleRaces) {
      if (race.raceId == raceId) return race;
    }
    return _visibleRaces.isEmpty ? null : _visibleRaces.first;
  }

  List<(String, int)> get _settings => [
    ('地力', _parameters.abilityLevel),
    ('近走内容', _parameters.recentLevel),
    ('条件適性', _parameters.conditionLevel),
    ('相手関係', _parameters.opponentLevel),
    ('展開・騎手', _parameters.paceJockeyLevel),
    ('血統適性', _parameters.pedigreeLevel),
    ('安定性', _parameters.stabilityLevel),
  ];

  List<(String, int)> get _changedSettings =>
      _settings.where((setting) => setting.$2 != 0).toList(growable: false);

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      color: _accent,
      onRefresh: _loadDay,
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: EdgeInsets.zero,
        children: [
          _buildPredictionHero(),
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
            child: Column(
              children: [
                if (_settingsLoading)
                  _buildSettingsLoadingCard()
                else if (!_configured)
                  _buildSetupCard()
                else ...[
                  _buildSavedSettingsCard(),
                  const SizedBox(height: 14),
                  if (_loading)
                    _buildLoadingCard()
                  else if (_errorMessage != null)
                    _buildErrorCard()
                  else if (_day == null || _day!.races.isEmpty)
                    _buildEmptyCard()
                  else ...[
                    if (_day!.fallbackNotice.isNotEmpty) ...[
                      _buildNoticeCard(_day!.fallbackNotice),
                      const SizedBox(height: 12),
                    ],
                    _buildRaceSelector(),
                    const SizedBox(height: 12),
                    AnimatedSize(
                      duration: const Duration(milliseconds: 280),
                      curve: Curves.easeOutCubic,
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 260),
                        switchInCurve: Curves.easeOut,
                        switchOutCurve: Curves.easeIn,
                        child: _calculatingPrediction
                            ? _buildPredictionCalculationCard()
                            : _buildResultCard(),
                      ),
                    ),
                  ],
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPredictionHero() {
    final ready = _configured && !_settingsLoading;
    return Container(
      height: 170,
      decoration: const BoxDecoration(
        color: _navy,
        image: DecorationImage(
          image: AssetImage('assets/images/prediction_settings_hero.png'),
          fit: BoxFit.cover,
        ),
      ),
      child: Stack(
        children: [
          Positioned(
            right: 4,
            bottom: 7,
            child: Image.asset(
              'assets/images/prediction_horse.png',
              width: 134,
              height: 126,
              fit: BoxFit.contain,
            ),
          ),
          Positioned(
            left: 22,
            right: 126,
            top: 20,
            bottom: 18,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _HeroStatusBadge(label: 'My予測'),
                const Spacer(),
                Text(
                  ready ? '自分の読みを\n勝負の印へ' : '自分の読みで\n勝負を組み立てる',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    height: 1.17,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -0.45,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  ready ? '重視する材料から、狙いたい一頭を導き出します' : '7つの基準から、あなただけの印をつくります',
                  style: const TextStyle(
                    color: Color(0xFFD5E2EA),
                    fontSize: 12,
                    height: 1.45,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSettingsLoadingCard() {
    return const Card(
      child: Padding(
        padding: EdgeInsets.symmetric(vertical: 42),
        child: Column(
          children: [
            CircularProgressIndicator(color: _accent),
            SizedBox(height: 14),
            Text('あなたの予測スタイルを読み込んでいます'),
          ],
        ),
      ),
    );
  }

  Widget _buildSetupCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(22, 24, 22, 22),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                _FeatureIcon(icon: Icons.tune_rounded),
                SizedBox(width: 13),
                Expanded(
                  child: Text(
                    'まずは、あなたの勝負基準を',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 18),
            const Text(
              '地力・近走・適性など7つの要素にあなたの読みを反映し、レースごとのMy予測を組み立てます。',
              style: TextStyle(color: _muted, height: 1.55, fontSize: 14),
            ),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                key: const Key('start-custom-prediction-settings'),
                onPressed: _openSettings,
                icon: const Icon(Icons.tune_rounded),
                label: const Text('勝負基準をつくる'),
                style: FilledButton.styleFrom(
                  backgroundColor: _accent,
                  minimumSize: const Size.fromHeight(50),
                  textStyle: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w800,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(25),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSavedSettingsCard() {
    final changedSettings = _changedSettings;
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(18, 14, 18, 14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const _FeatureIcon(icon: Icons.auto_awesome_rounded),
                const SizedBox(width: 12),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'あなたの予測スタイル',
                        style: TextStyle(
                          fontSize: 17,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        '保存した基準は、次の予測にもそのまま使えます',
                        style: TextStyle(color: _muted, fontSize: 12),
                      ),
                    ],
                  ),
                ),
                OutlinedButton(
                  onPressed: _openSettings,
                  style: OutlinedButton.styleFrom(
                    foregroundColor: _accent,
                    side: const BorderSide(color: Color(0xFF9BC8C4)),
                    visualDensity: VisualDensity.compact,
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                  ),
                  child: const Text(
                    '設定を変更',
                    style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 11),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(13, 10, 13, 10),
              decoration: BoxDecoration(
                color: const Color(0xFFF1F7F6),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'あなたが重視するポイント',
                    style: TextStyle(
                      color: _muted,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 9),
                  Wrap(
                    spacing: 7,
                    runSpacing: 7,
                    children: [
                      if (changedSettings.isEmpty)
                        const _SavedSettingBadge(title: '全項目', label: '標準')
                      else
                        for (final setting in changedSettings.take(4))
                          _SavedSettingBadge(
                            title: setting.$1,
                            label: CustomPredictionParameters.labelFor(
                              setting.$2,
                            ),
                          ),
                    ],
                  ),
                  if (changedSettings.length > 4) ...[
                    const SizedBox(height: 8),
                    Text(
                      'ほか${changedSettings.length - 4}項目も設定済み',
                      style: const TextStyle(color: _muted, fontSize: 11),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingCard() {
    return const Card(
      child: Padding(
        padding: EdgeInsets.symmetric(vertical: 42),
        child: Column(
          children: [
            CircularProgressIndicator(color: _accent),
            SizedBox(height: 14),
            Text('勝負材料を集めています'),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            const Icon(Icons.cloud_off_outlined, color: _muted, size: 34),
            const SizedBox(height: 10),
            Text(_errorMessage ?? 'レース情報を取得できませんでした。'),
            const SizedBox(height: 14),
            FilledButton.icon(
              onPressed: _loadDay,
              icon: const Icon(Icons.refresh),
              label: const Text('もう一度試す'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(22, 22, 22, 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                _FeatureIcon(icon: Icons.schedule_rounded),
                SizedBox(width: 12),
                _StatusPill(label: 'レース情報を確認中'),
              ],
            ),
            const SizedBox(height: 20),
            const Text(
              '次の勝負に備えましょう',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 9),
            const Text(
              '予測できるレース情報はまだ公開されていません。情報が揃うまで、予測バランスを整えてお待ちください。',
              style: TextStyle(color: _muted, fontSize: 14, height: 1.55),
            ),
            const SizedBox(height: 14),
            const Row(
              children: [
                Icon(Icons.auto_awesome_rounded, color: _accent, size: 18),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    '公開され次第、あなたの基準でMy予測を組み立てられます',
                    style: TextStyle(color: _accent, fontSize: 12),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 18),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _loadDay,
                icon: const Icon(Icons.refresh_rounded),
                label: const Text(
                  'レース情報を更新',
                  style: TextStyle(fontWeight: FontWeight.w800),
                ),
                style: OutlinedButton.styleFrom(
                  foregroundColor: _accent,
                  minimumSize: const Size.fromHeight(48),
                  side: const BorderSide(color: Color(0xFF9BC8C4)),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNoticeCard(String message) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF7E2),
        border: Border.all(color: const Color(0xFFF0D995)),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.info_outline, color: Color(0xFF8A6812), size: 20),
          const SizedBox(width: 9),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(color: Color(0xFF6D5515), height: 1.45),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRaceSelector() {
    final day = _day!;
    final courses = <String>[];
    for (final race in day.races) {
      if (race.course.isNotEmpty && !courses.contains(race.course)) {
        courses.add(race.course);
      }
    }
    final races = _visibleRaces;
    final selectedRace = _selectedRace;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '狙うレースを選ぶ',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
            ),
            if (courses.length > 1) ...[
              const SizedBox(height: 13),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  for (final course in courses)
                    FilterChip(
                      label: Text(course),
                      selected: course == _selectedCourse,
                      onSelected: (selected) {
                        if (selected) _selectCourse(course);
                      },
                    ),
                ],
              ),
            ],
            const SizedBox(height: 14),
            DropdownButtonFormField<String>(
              key: ValueKey(
                '${_selectedCourse ?? 'all'}-${selectedRace?.raceId}',
              ),
              initialValue: selectedRace?.raceId,
              isExpanded: true,
              decoration: const InputDecoration(
                labelText: 'レースを選択',
                border: OutlineInputBorder(),
              ),
              items: [
                for (final race in races)
                  DropdownMenuItem(
                    value: race.raceId,
                    child: Text(
                      race.displayName,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
              ],
              onChanged: _selectRace,
            ),
            if (selectedRace != null) ...[
              const SizedBox(height: 11),
              Text(
                [
                  selectedRace.distanceLabel,
                  selectedRace.trackCondition,
                ].where((value) => value.isNotEmpty).join(' ・ '),
                style: const TextStyle(color: _muted, fontSize: 13),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionCalculationCard() {
    return Card(
      key: ValueKey('prediction-calculation-$_calculationSerial'),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 19),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                _FeatureIcon(icon: Icons.psychology_alt_rounded),
                SizedBox(width: 12),
                _StatusPill(label: 'My予測分析中'),
              ],
            ),
            const SizedBox(height: 17),
            const Text(
              'あなたの読みを反映中',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 8),
            AnimatedSwitcher(
              key: const Key('prediction-calculation-message'),
              duration: const Duration(milliseconds: 220),
              child: Text(
                _activeCalculationMessages[_calculationPhase],
                key: ValueKey(_calculationPhase),
                style: const TextStyle(
                  color: _muted,
                  fontSize: 14,
                  height: 1.5,
                ),
              ),
            ),
            const SizedBox(height: 18),
            AnimatedBuilder(
              animation: _calculationProgressController,
              builder: (context, _) {
                return ClipRRect(
                  borderRadius: BorderRadius.circular(5),
                  child: LinearProgressIndicator(
                    minHeight: 7,
                    value: _calculationProgressController.value,
                    backgroundColor: const Color(0xFFDCE9E8),
                    valueColor: const AlwaysStoppedAnimation(_accent),
                  ),
                );
              },
            ),
            const SizedBox(height: 13),
            Row(
              children: [
                for (
                  var index = 0;
                  index < _activeCalculationMessages.length;
                  index++
                ) ...[
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    width: index == _calculationPhase ? 18 : 7,
                    height: 7,
                    decoration: BoxDecoration(
                      color: index <= _calculationPhase
                          ? _accent
                          : const Color(0xFFCFDCDA),
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  if (index < _activeCalculationMessages.length - 1)
                    const SizedBox(width: 6),
                ],
                const SizedBox(width: 10),
                const Expanded(
                  child: Text(
                    '狙いを一頭ずつ絞り込んでいます',
                    style: TextStyle(color: _accent, fontSize: 11),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    final race = _selectedRace;
    final picks =
        race?.calculate(_parameters) ?? const <CustomPredictionPick>[];
    return Card(
      key: ValueKey('prediction-result-${race?.raceId ?? 'empty'}'),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(18, 17, 18, 18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'My予測の結論',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 4),
            const Text(
              'あなたが重視した勝負材料から導いた注目馬です',
              style: TextStyle(color: _muted, fontSize: 13),
            ),
            const SizedBox(height: 14),
            if (picks.isEmpty)
              const Text('予測できる出走馬がありません。')
            else
              for (var index = 0; index < picks.length; index++) ...[
                _PredictionPickRow(pick: picks[index]),
                if (index < picks.length - 1)
                  const Divider(height: 1, color: Color(0xFFE5EAEE)),
              ],
            const SizedBox(height: 14),
            const Text(
              'オッズや直前情報も見ながら、最後の買い目はご自身でご判断ください。',
              style: TextStyle(color: _muted, fontSize: 12, height: 1.45),
            ),
          ],
        ),
      ),
    );
  }
}

class _HeroStatusBadge extends StatelessWidget {
  const _HeroStatusBadge({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFFDDF5F1),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(
            Icons.auto_awesome_rounded,
            color: Color(0xFF08736F),
            size: 15,
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: const TextStyle(
              color: Color(0xFF08736F),
              fontSize: 11,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

class _FeatureIcon extends StatelessWidget {
  const _FeatureIcon({required this.icon});

  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 42,
      height: 42,
      alignment: Alignment.center,
      decoration: const BoxDecoration(
        color: Color(0xFFE7F4F1),
        shape: BoxShape.circle,
      ),
      child: Icon(icon, color: Color(0xFF0D6E6E), size: 21),
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF3D8),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Text(
        label,
        style: const TextStyle(
          color: Color(0xFF836311),
          fontSize: 11,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }
}

class _SavedSettingBadge extends StatelessWidget {
  const _SavedSettingBadge({required this.title, required this.label});

  final String title;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: const Color(0xFFF0F7F5),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text.rich(
        TextSpan(
          children: [
            TextSpan(text: '$title：'),
            TextSpan(
              text: label,
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
          ],
        ),
        style: const TextStyle(color: Color(0xFF0D6E6E), fontSize: 12),
      ),
    );
  }
}

class _PredictionPickRow extends StatelessWidget {
  const _PredictionPickRow({required this.pick});

  final CustomPredictionPick pick;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 11),
      child: Row(
        children: [
          SizedBox(
            width: 34,
            child: Text(
              pick.mark,
              textAlign: TextAlign.center,
              style: const TextStyle(
                color: Color(0xFF0D6E6E),
                fontSize: 23,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Container(
            width: 32,
            height: 32,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: const Color(0xFF07111F),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              '${pick.horseNo}',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              pick.horseName,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
    );
  }
}

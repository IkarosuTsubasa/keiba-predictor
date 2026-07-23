import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/custom_prediction.dart';
import '../services/custom_prediction_access.dart';
import '../services/custom_prediction_service.dart';
import '../services/custom_prediction_settings.dart';
import '../services/reward_ad_requirement.dart';
import 'custom_prediction_settings_screen.dart';

class CustomPredictionScreen extends StatefulWidget {
  const CustomPredictionScreen({
    this.dataSource,
    this.settingsStore,
    this.accessStore,
    this.rewardedGateway,
    this.rewardAdRequirement,
    this.now,
    this.calculationDuration,
    super.key,
  });

  final CustomPredictionDataSource? dataSource;
  final CustomPredictionSettingsStore? settingsStore;
  final CustomPredictionAccessStore? accessStore;
  final CustomPredictionRewardedGateway? rewardedGateway;
  final RewardAdRequirement? rewardAdRequirement;
  final DateTime Function()? now;
  final Duration Function()? calculationDuration;

  @override
  State<CustomPredictionScreen> createState() => _CustomPredictionScreenState();
}

class _CustomPredictionScreenState extends State<CustomPredictionScreen>
    with WidgetsBindingObserver, SingleTickerProviderStateMixin {
  static const _navy = Color(0xFF07111F);
  static const _accent = Color(0xFF0D6E6E);
  static const _muted = Color(0xFF66746F);
  static const _calculationMessages = [
    'レース条件を読み解いています',
    'あなたの予測バランスを反映しています',
    '各馬を比較し、最終順位を導いています',
  ];

  late final CustomPredictionDataSource _dataSource;
  late final CustomPredictionSettingsStore _settingsStore;
  late final CustomPredictionAccessStore _accessStore;
  late final CustomPredictionRewardedGateway _rewardedGateway;
  late final RewardAdRequirement _rewardAdRequirement;
  late final DateTime Function() _now;
  late final Duration Function() _calculationDuration;
  final math.Random _random = math.Random();
  CustomPredictionParameters _parameters = const CustomPredictionParameters();
  CustomPredictionDay? _day;
  String? _selectedCourse;
  String? _selectedRaceId;
  String? _errorMessage;
  bool _settingsLoading = true;
  bool _accessLoading = true;
  bool _configured = false;
  bool _loading = false;
  bool _unlocking = false;
  bool _calculatingPrediction = false;
  bool _rewardAdRequiredForSession = true;
  int _calculationPhase = 0;
  int _calculationSerial = 0;
  DateTime? _unlockedUntil;
  Timer? _accessExpiryTimer;
  Timer? _calculationPhaseTimer;
  Timer? _calculationCompleteTimer;
  late final AnimationController _calculationProgressController;
  int _requestSerial = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _dataSource = widget.dataSource ?? const CustomPredictionService();
    _settingsStore =
        widget.settingsStore ?? CustomPredictionSettingsRepository();
    _accessStore = widget.accessStore ?? CustomPredictionAccessRepository();
    _rewardedGateway =
        widget.rewardedGateway ?? GoogleCustomPredictionRewardedGateway();
    _rewardAdRequirement =
        widget.rewardAdRequirement ?? RemoteConfigRewardAdRequirement.instance;
    _now = widget.now ?? DateTime.now;
    _calculationDuration =
        widget.calculationDuration ??
        () => Duration(milliseconds: 3000 + _random.nextInt(2001));
    _calculationProgressController = AnimationController(vsync: this);
    unawaited(_initialize());
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _accessExpiryTimer?.cancel();
    _cancelPredictionCalculation();
    _calculationProgressController.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _refreshAccessState();
    }
  }

  Future<void> _initialize() async {
    final settingsFuture = _settingsStore.load();
    final accessFuture = _accessStore.loadUnlockedUntil();
    final rewardRequirementFuture = _rewardAdRequirement.refresh();
    CustomPredictionSettingsSnapshot? savedSettings;
    DateTime? unlockedUntil;
    try {
      savedSettings = await settingsFuture;
    } catch (_) {
      savedSettings = null;
    }
    try {
      unlockedUntil = await accessFuture;
    } catch (_) {
      unlockedUntil = null;
    }
    try {
      await rewardRequirementFuture;
    } catch (_) {
      // Remote Config の実装側が保持する安全な既定値を使用する。
    }
    if (!mounted) return;
    setState(() {
      _parameters =
          savedSettings?.parameters ?? const CustomPredictionParameters();
      _configured = savedSettings?.isConfigured ?? false;
      _settingsLoading = false;
      _unlockedUntil = unlockedUntil?.toUtc();
      _accessLoading = false;
      _rewardAdRequiredForSession = _rewardAdRequirement.isRequired;
    });
    _scheduleAccessExpiry();
    if (_configured && _hasAccess) {
      await _loadDay();
    }
  }

  bool get _hasAccess {
    if (_accessLoading) return false;
    if (!_rewardAdRequiredForSession) return true;
    final unlockedUntil = _unlockedUntil;
    return unlockedUntil != null && unlockedUntil.isAfter(_now().toUtc());
  }

  String get _accessUntilLabel {
    if (!_rewardAdRequiredForSession) {
      return 'リワード広告なしで利用できます';
    }
    final local = _unlockedUntil?.toLocal();
    if (local == null) return '';
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '$hour:$minuteまで利用できます';
  }

  void _scheduleAccessExpiry() {
    _accessExpiryTimer?.cancel();
    if (!_rewardAdRequiredForSession) return;
    final unlockedUntil = _unlockedUntil;
    if (unlockedUntil == null) return;
    final remaining = unlockedUntil.difference(_now().toUtc());
    if (remaining <= Duration.zero) {
      _expireAccess();
      return;
    }
    _accessExpiryTimer = Timer(remaining, _expireAccess);
  }

  void _refreshAccessState() {
    if (!mounted) return;
    if (!_rewardAdRequiredForSession) return;
    final unlockedUntil = _unlockedUntil;
    if (unlockedUntil == null) return;
    if (!unlockedUntil.isAfter(_now().toUtc())) {
      _expireAccess();
      return;
    }
    _scheduleAccessExpiry();
    setState(() {});
  }

  void _expireAccess() {
    if (!mounted) return;
    _accessExpiryTimer?.cancel();
    _cancelPredictionCalculation();
    _requestSerial += 1;
    setState(() {
      _unlockedUntil = null;
      _day = null;
      _selectedCourse = null;
      _selectedRaceId = null;
      _errorMessage = null;
      _loading = false;
      _calculatingPrediction = false;
    });
  }

  Future<void> _unlockWithRewardedAd() async {
    if (_unlocking || _hasAccess) return;
    setState(() => _unlocking = true);
    CustomPredictionRewardResult result;
    try {
      result = await _rewardedGateway.showRewardedAd();
    } catch (_) {
      result = CustomPredictionRewardResult.failed;
    }
    if (!mounted) return;

    if (result == CustomPredictionRewardResult.rewarded) {
      final unlockedUntil = _now().toUtc().add(const Duration(hours: 1));
      try {
        await _accessStore.saveUnlockedUntil(unlockedUntil);
      } catch (_) {
        // 保存に失敗しても、現在のセッションでは獲得した権利を有効にする。
      }
      if (!mounted) return;
      setState(() {
        _unlockedUntil = unlockedUntil;
        _unlocking = false;
      });
      _scheduleAccessExpiry();
      _showMessage('My予測を1時間利用できます。');
      await _loadDay();
      return;
    }

    setState(() => _unlocking = false);
    switch (result) {
      case CustomPredictionRewardResult.dismissed:
        _showMessage('動画を最後まで見ると、1時間利用できます。');
        break;
      case CustomPredictionRewardResult.unavailable:
        _showMessage('現在、動画を準備できません。しばらくしてからお試しください。');
        break;
      case CustomPredictionRewardResult.failed:
        _showMessage('動画を読み込めませんでした。もう一度お試しください。');
        break;
      case CustomPredictionRewardResult.rewarded:
        break;
    }
  }

  void _showMessage(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> _loadDay() async {
    if (!_configured || !_hasAccess) return;
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
    final calculationSerial = ++_calculationSerial;
    final phaseInterval = Duration(
      milliseconds: math.max(
        1,
        duration.inMilliseconds ~/ _calculationMessages.length,
      ),
    );
    setState(() {
      _selectedCourse = course;
      _selectedRaceId = raceId;
      _calculatingPrediction = true;
      _calculationPhase = 0;
    });
    _calculationProgressController.duration = duration;
    unawaited(_calculationProgressController.forward(from: 0));

    _calculationPhaseTimer = Timer.periodic(phaseInterval, (timer) {
      if (!mounted || calculationSerial != _calculationSerial) {
        timer.cancel();
        return;
      }
      if (_calculationPhase >= _calculationMessages.length - 1) {
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
      onRefresh: _configured && _hasAccess ? _loadDay : () async {},
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: EdgeInsets.zero,
        children: [
          _buildPredictionHero(),
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
            child: Column(
              children: [
                if (_settingsLoading || _accessLoading)
                  _buildSettingsLoadingCard()
                else if (!_configured)
                  _buildSetupCard()
                else ...[
                  _buildSavedSettingsCard(),
                  const SizedBox(height: 14),
                  if (!_hasAccess)
                    _buildLockedAccessCard()
                  else if (_loading)
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
                  ready ? '今日の予測を\n自分の基準で' : '自分だけの予測を\n始めよう',
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
                  ready ? 'あなたの感覚から、今日だけの答えを導き出します' : 'あなたの感覚を映した、特別な予測体験へ',
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
            Text('予測設定を読み込んでいます'),
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
                    '予測設定がまだ保存されていません',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 18),
            const Text(
              '最初に7つの要素を設定すると、保存した内容で本日のMy予測を作成します。',
              style: TextStyle(color: _muted, height: 1.55, fontSize: 14),
            ),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                key: const Key('start-custom-prediction-settings'),
                onPressed: _openSettings,
                icon: const Icon(Icons.tune_rounded),
                label: const Text('設定を始める'),
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
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _hasAccess ? 'My予測を利用中' : '予測バランスを保存済み',
                        style: const TextStyle(
                          fontSize: 17,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        _hasAccess ? _accessUntilLabel : 'いつでも同じ設定から再開できます',
                        style: const TextStyle(color: _muted, fontSize: 12),
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
                    '現在の予測バランス',
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

  Widget _buildLockedAccessCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(22, 22, 22, 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                _FeatureIcon(icon: Icons.play_circle_outline_rounded),
                SizedBox(width: 12),
                _StatusPill(label: '1時間アクセス'),
              ],
            ),
            const SizedBox(height: 18),
            const Text(
              '動画を見て、予測をひらく',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 9),
            const Text(
              '短い動画を最後まで見ると、My予測を1時間自由に利用できます。',
              style: TextStyle(color: _muted, fontSize: 14, height: 1.55),
            ),
            const SizedBox(height: 18),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                key: const Key('unlock-custom-prediction'),
                onPressed: _unlocking ? null : _unlockWithRewardedAd,
                icon: _unlocking
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.play_arrow_rounded),
                label: Text(
                  _unlocking ? '動画を準備しています' : '動画を見て1時間使う',
                  style: const TextStyle(fontWeight: FontWeight.w800),
                ),
                style: FilledButton.styleFrom(
                  backgroundColor: _accent,
                  foregroundColor: Colors.white,
                  disabledBackgroundColor: const Color(0xFF6AA5A1),
                  disabledForegroundColor: Colors.white,
                  minimumSize: const Size.fromHeight(50),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(25),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 10),
            const Center(
              child: Text(
                '保存した予測バランスはそのまま残ります',
                style: TextStyle(color: _muted, fontSize: 11),
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
            Text('レースデータを読み込んでいます'),
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
            Text(_errorMessage ?? '予測データを取得できませんでした。'),
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
                _StatusPill(label: '本日分を準備中'),
              ],
            ),
            const SizedBox(height: 20),
            const Text(
              '今日の予測はもうすぐ',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 9),
            const Text(
              '本日のレース情報はまだ準備できていません。しばらくお待ちください。',
              style: TextStyle(color: _muted, fontSize: 14, height: 1.55),
            ),
            const SizedBox(height: 14),
            const Row(
              children: [
                Icon(Icons.auto_awesome_rounded, color: _accent, size: 18),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    '準備が整い次第、あなただけの予測が完成します',
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
                  '最新情報を確認',
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
              '本日のレースを選択',
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
                labelText: '対象レース',
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
                _StatusPill(label: 'カスタム演算'),
              ],
            ),
            const SizedBox(height: 17),
            const Text(
              'あなたの基準で再予測中',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 8),
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 220),
              child: Text(
                _calculationMessages[_calculationPhase],
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
                  index < _calculationMessages.length;
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
                  if (index < _calculationMessages.length - 1)
                    const SizedBox(width: 6),
                ],
                const SizedBox(width: 10),
                const Expanded(
                  child: Text(
                    'まもなく、あなただけの予測が完成します',
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
              'My予測結果',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 4),
            const Text(
              'あなたの予測バランスから導き出した結果です',
              style: TextStyle(color: _muted, fontSize: 13),
            ),
            const SizedBox(height: 14),
            if (picks.isEmpty)
              const Text('表示できる出走馬がありません。')
            else
              for (var index = 0; index < picks.length; index++) ...[
                _PredictionPickRow(pick: picks[index]),
                if (index < picks.length - 1)
                  const Divider(height: 1, color: Color(0xFFE5EAEE)),
              ],
            const SizedBox(height: 14),
            const Text(
              'この結果は設定した条件による参考情報です。投票はご自身の判断で行ってください。',
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

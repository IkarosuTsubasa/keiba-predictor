import 'dart:async';

import 'package:flutter/material.dart';

import '../models/custom_prediction.dart';
import '../services/custom_prediction_service.dart';
import '../services/custom_prediction_settings.dart';
import 'custom_prediction_settings_screen.dart';

class CustomPredictionScreen extends StatefulWidget {
  const CustomPredictionScreen({
    this.dataSource,
    this.settingsStore,
    super.key,
  });

  final CustomPredictionDataSource? dataSource;
  final CustomPredictionSettingsStore? settingsStore;

  @override
  State<CustomPredictionScreen> createState() => _CustomPredictionScreenState();
}

class _CustomPredictionScreenState extends State<CustomPredictionScreen> {
  static const _accent = Color(0xFF0D6E6E);
  static const _muted = Color(0xFF66746F);

  late final CustomPredictionDataSource _dataSource;
  late final CustomPredictionSettingsStore _settingsStore;
  CustomPredictionParameters _parameters = const CustomPredictionParameters();
  CustomPredictionDay? _day;
  String? _selectedCourse;
  String? _selectedRaceId;
  String? _errorMessage;
  bool _settingsLoading = true;
  bool _configured = false;
  bool _loading = false;
  int _requestSerial = 0;

  @override
  void initState() {
    super.initState();
    _dataSource = widget.dataSource ?? const CustomPredictionService();
    _settingsStore =
        widget.settingsStore ?? CustomPredictionSettingsRepository();
    unawaited(_initialize());
  }

  Future<void> _initialize() async {
    try {
      final saved = await _settingsStore.load();
      if (!mounted) return;
      setState(() {
        _parameters = saved.parameters;
        _configured = saved.isConfigured;
        _settingsLoading = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _settingsLoading = false;
        _configured = false;
      });
    }
    if (_configured) await _loadDay();
  }

  Future<void> _loadDay() async {
    if (!_configured) return;
    final requestSerial = ++_requestSerial;
    setState(() {
      _loading = true;
      _errorMessage = null;
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

  void _selectCourse(String course) {
    final races = _day?.races
        .where((race) => race.course == course)
        .toList(growable: false);
    setState(() {
      _selectedCourse = course;
      _selectedRaceId = races == null || races.isEmpty
          ? null
          : races.first.raceId;
    });
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

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      color: _accent,
      onRefresh: _configured ? _loadDay : () async {},
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
        children: [
          if (_settingsLoading)
            _buildSettingsLoadingCard()
          else if (!_configured)
            _buildSetupCard()
          else ...[
            _buildSavedSettingsCard(),
            const SizedBox(height: 12),
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
              _buildResultCard(),
            ],
          ],
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
        padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 34),
        child: Column(
          children: [
            const Icon(Icons.tune_rounded, color: _accent, size: 38),
            const SizedBox(height: 13),
            const Text(
              '予測設定がまだ保存されていません',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 9),
            const Text(
              '最初に7つの要素を設定すると、保存した内容で本日のマイ予測を作成します。',
              textAlign: TextAlign.center,
              style: TextStyle(color: _muted, height: 1.55),
            ),
            const SizedBox(height: 18),
            FilledButton.icon(
              key: const Key('start-custom-prediction-settings'),
              onPressed: _openSettings,
              icon: const Icon(Icons.tune_rounded),
              label: const Text('設定を始める'),
              style: FilledButton.styleFrom(
                minimumSize: const Size(190, 48),
                textStyle: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSavedSettingsCard() {
    final settings = [
      ('地力', _parameters.abilityLevel),
      ('近走内容', _parameters.recentLevel),
      ('条件適性', _parameters.conditionLevel),
      ('相手関係', _parameters.opponentLevel),
      ('展開・騎手', _parameters.paceJockeyLevel),
      ('血統適性', _parameters.pedigreeLevel),
      ('安定性', _parameters.stabilityLevel),
    ];
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '保存した設定で予測中',
                        style: TextStyle(
                          fontSize: 17,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        '本日の全レースに同じ設定を使用します',
                        style: TextStyle(color: _muted, fontSize: 13),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 10),
                OutlinedButton.icon(
                  onPressed: _openSettings,
                  icon: const Icon(Icons.tune_rounded, size: 18),
                  label: const Text('設定を変更'),
                ),
              ],
            ),
            const SizedBox(height: 14),
            Wrap(
              spacing: 7,
              runSpacing: 7,
              children: [
                for (final setting in settings)
                  _SavedSettingBadge(
                    title: setting.$1,
                    label: CustomPredictionParameters.labelFor(setting.$2),
                  ),
              ],
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
    return const Card(
      child: Padding(
        padding: EdgeInsets.symmetric(horizontal: 20, vertical: 34),
        child: Column(
          children: [
            Icon(Icons.event_busy_outlined, color: _muted, size: 34),
            SizedBox(height: 10),
            Text('本日のレース情報はまだ準備できていません。しばらくお待ちください。'),
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
                      onSelected: (_) => _selectCourse(course),
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
              onChanged: (value) => setState(() => _selectedRaceId = value),
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

  Widget _buildResultCard() {
    final race = _selectedRace;
    final picks =
        race?.calculate(_parameters) ?? const <CustomPredictionPick>[];
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(18, 17, 18, 18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'マイ予測結果',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 4),
            const Text(
              '保存した設定で端末内計算した結果です',
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

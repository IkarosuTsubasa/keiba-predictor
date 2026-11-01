import 'dart:async';
import 'package:flutter/material.dart';
import '../models/simulation_context.dart';
import '../services/simulation_service.dart';
import '../services/simulation_settings.dart';
import '../services/simulation_analytics.dart';
import 'race_simulation_screen.dart';

class SimulationHomeScreen extends StatefulWidget {
  const SimulationHomeScreen({required this.active, this.service, super.key});
  final bool active;
  final SimulationService? service;
  @override
  State<SimulationHomeScreen> createState() => SimulationHomeScreenState();
}

class SimulationHomeScreenState extends State<SimulationHomeScreen> {
  late final _service = widget.service ?? SimulationService();
  final _settings = SimulationSettings();
  final _raceKey = GlobalKey<RaceSimulationScreenState>();
  List<Map<String, dynamic>> _races = [];
  SimulationContext? _selected, _recent;
  String _scope = '', _course = '';
  bool _upcoming = true, _loading = true;
  String? _error;
  int _generation = 0;
  late String _date;
  @override
  void initState() {
    super.initState();
    _date = DateTime.now()
        .toUtc()
        .add(const Duration(hours: 9))
        .toIso8601String()
        .substring(0, 10);
    unawaited(_refresh());
    unawaited(_restoreRecent());
  }

  Future<void> _restoreRecent() async {
    final saved = await _settings.load('recent');
    if (saved != null && mounted) {
      try {
        setState(() => _recent = SimulationContext.fromJson(saved));
      } on Object {
        /* Ignore old local state. */
      }
    }
  }

  Future<void> _refresh() async {
    final generation = ++_generation;
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final data = await _service.list(_date);
      if (!mounted || generation != _generation) return;
      setState(() {
        _races = (data['races'] as List).cast<Map<String, dynamic>>();
        _loading = false;
      });
    } on SimulationException catch (e) {
      if (mounted && generation == _generation) {
        setState(() {
          _error = e.message;
          _loading = false;
        });
      }
    }
  }

  void open(SimulationContext context) {
    if (!mounted) return;
    setState(() {
      _selected = context;
      _recent = context;
    });
    unawaited(_settings.save('recent', context.toJson()));
  }

  Future<bool> handleBack() async {
    if (_selected == null) return false;
    await _raceKey.currentState?.saveAndPause();
    if (mounted) setState(() => _selected = null);
    return true;
  }

  @override
  void dispose() {
    _generation++;
    _service.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_selected != null) {
      return RaceSimulationScreen(
        key: _raceKey,
        context: _selected!,
        active: widget.active,
        onBack: () => unawaited(handleBack()),
      );
    }
    final courses =
        _races
            .where(
              (r) =>
                  _scope.isEmpty ||
                  (r['scope_key'] as String? ?? '').startsWith(_scope),
            )
            .map((r) => r['course'] as String? ?? '')
            .where((v) => v.isNotEmpty)
            .toSet()
            .toList()
          ..sort();
    final filtered = _races
        .where(
          (r) =>
              (_scope.isEmpty ||
                  (r['scope_key'] as String? ?? '').startsWith(_scope)) &&
              (_course.isEmpty || r['course'] == _course) &&
              (!_upcoming || r['race_state'] == 'scheduled'),
        )
        .toList();
    return RefreshIndicator(
      onRefresh: _refresh,
      child: ListView(
        key: const PageStorageKey('simulation-list'),
        padding: const EdgeInsets.all(16),
        children: [
          Text('シミュレーション', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 8),
          TextButton.icon(
            onPressed: () async {
              final day = await showDatePicker(
                context: context,
                initialDate: DateTime.parse(_date),
                firstDate: DateTime(2025),
                lastDate: DateTime.now().add(const Duration(days: 30)),
              );
              if (day != null && mounted) {
                setState(() => _date = day.toIso8601String().substring(0, 10));
                unawaited(_refresh());
              }
            },
            icon: const Icon(Icons.calendar_today_outlined),
            label: Text(_date),
          ),
          if (_recent != null)
            Card(
              child: ListTile(
                leading: const Icon(Icons.history),
                title: const Text('前回のシミュレーションを続ける'),
                subtitle: Text('${_recent!.date} · ${_recent!.raceId}'),
                onTap: () => open(_recent!),
              ),
            ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            children: [
              for (final pair in const [
                ('', 'すべて'),
                ('central', '中央'),
                ('local', '地方'),
              ])
                ChoiceChip(
                  label: Text(pair.$2),
                  selected: _scope == pair.$1,
                  onSelected: (_) => setState(() {
                    _scope = pair.$1;
                    _course = '';
                  }),
                ),
            ],
          ),
          if (courses.isNotEmpty)
            DropdownButton<String>(
              isExpanded: true,
              value: courses.contains(_course) ? _course : '',
              items: [
                const DropdownMenuItem(value: '', child: Text('すべての競馬場')),
                for (final course in courses)
                  DropdownMenuItem(value: course, child: Text(course)),
              ],
              onChanged: (v) => setState(() => _course = v ?? ''),
            ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('発走予定のレースのみ'),
            subtitle: const Text('予定時刻が不明のレースは「すべて」で確認できます。'),
            value: _upcoming,
            onChanged: (v) => setState(() => _upcoming = v),
          ),
          if (_loading)
            const Padding(
              padding: EdgeInsets.all(24),
              child: Center(child: CircularProgressIndicator()),
            ),
          if (_error != null) Text(_error!),
          if (!_loading && _error == null && filtered.isEmpty)
            const Padding(
              padding: EdgeInsets.all(20),
              child: Text('条件に合うレースがありません。表示条件を変えてお試しください。'),
            ),
          for (final row in filtered)
            Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Card(
                child: ListTile(
                  isThreeLine: true,
                  title: Text(
                    '${row['course'] ?? ''} ${int.tryParse((row['race_id'] as String).substring(10)) ?? ''}R · ${row['race_name'] ?? ''}',
                  ),
                  subtitle: Text(
                    '${row['surface'] ?? ''} ${row['distance'] ?? '–'}m · ${row['horse_count'] ?? '–'}頭\n${row['scheduled_off_at'] ?? '発走時刻不明'}\n${_status(row['availability'])}${row['published_at'] == null ? '' : ' · 更新 ${row['published_at']}'}',
                  ),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () {
                    if (row['availability'] == 'unsupported' ||
                        row['availability'] == 'unavailable') {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text(_status(row['availability']))),
                      );
                      return;
                    }
                    if (row['availability'] == 'preparing') {
                      unawaited(_refresh());
                      return;
                    }
                    unawaited(
                      simulationEvent('sim_race_select', {
                        'race_id': row['race_id'] as String,
                        'source': 'app_tab',
                      }),
                    );
                    open(
                      SimulationContext(
                        raceId: row['race_id'] as String,
                        date: _date,
                      ),
                    );
                  },
                ),
              ),
            ),
          OutlinedButton.icon(
            onPressed: _refresh,
            icon: const Icon(Icons.refresh),
            label: const Text('再読み込み'),
          ),
        ],
      ),
    );
  }

  String _status(dynamic value) => switch (value) {
    'ready' => '試走できます',
    'partial' => '一部のデータを推定して試走できます',
    'unsupported' => 'このレースは対象外です',
    'unavailable' => '出走予定の馬が足りません',
    _ => 'データ準備中',
  };
}

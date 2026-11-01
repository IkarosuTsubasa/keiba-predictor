import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webview_flutter/webview_flutter.dart';
import '../config/app_config.dart';
import '../models/simulation_context.dart';
import '../services/simulation_service.dart';
import '../services/simulation_settings.dart';
import '../services/simulation_analytics.dart';

class RaceSimulationScreen extends StatefulWidget {
  const RaceSimulationScreen({
    required this.context,
    required this.active,
    required this.onBack,
    super.key,
  });
  final SimulationContext context;
  final bool active;
  final VoidCallback onBack;
  @override
  State<RaceSimulationScreen> createState() => RaceSimulationScreenState();
}

class RaceSimulationScreenState extends State<RaceSimulationScreen>
    with WidgetsBindingObserver {
  final _service = SimulationService();
  final _storage = SimulationSettings();
  WebViewController? _controller;
  SimulationContext? _bound;
  Map<String, dynamic>? _input, _state, _restore, _batch, _comparison;
  List<Map<String, dynamic>> _profiles = [], _courses = [];
  bool _ready = false, _loading = true;
  String? _error, _updatedVersion;
  int _sequence = 0, _generation = 0, _completed = 0;
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    unawaited(_load(widget.context));
  }

  @override
  void didUpdateWidget(covariant RaceSimulationScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.context != widget.context) {
      unawaited(_load(widget.context));
    } else if (oldWidget.active != widget.active) {
      unawaited(_send('active', {'active': widget.active}));
      if (!widget.active) unawaited(saveAndPause());
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    unawaited(
      _send('active', {
        'active': state == AppLifecycleState.resumed && widget.active,
      }),
    );
    if (state != AppLifecycleState.resumed) unawaited(saveAndPause());
  }

  Future<void> _load(SimulationContext context) async {
    if (_ready) await saveAndPause();
    final generation = ++_generation;
    setState(() {
      _loading = true;
      _ready = false;
      _controller = null;
      _error = null;
      _state = null;
      _batch = null;
      _comparison = null;
      _completed = 0;
      _updatedVersion = null;
    });
    try {
      final input = await _service.race(context);
      if (!mounted || generation != _generation) return;
      if (!['ready', 'partial'].contains(input['availability'])) {
        throw SimulationException(
          input['availability'] as String? ?? 'preparing',
        );
      }
      final bound = context.bind(input);
      final saved = context.source == 'app_tab'
          ? await _storage.load('scene.${bound.storageKey}')
          : null;
      final preferences = await _storage.load('preferences');
      if (!mounted || generation != _generation) return;
      _restore =
          saved ??
          {
            'settings': {...?preferences, 'seed': bound.seed},
          };
      final expected = bound.embedUri(AppConfig.baseWebUrl);
      final controller = WebViewController();
      await controller.setJavaScriptMode(JavaScriptMode.unrestricted);
      await controller.setBackgroundColor(const Color(0xFF172322));
      await controller.addJavaScriptChannel(
        'SimulationChannel',
        onMessageReceived: (message) {
          if (generation == _generation) _message(message.message, bound);
        },
      );
      await controller.setNavigationDelegate(
        NavigationDelegate(
          onNavigationRequest: (request) {
            final uri = Uri.tryParse(request.url);
            final allowed =
                request.isMainFrame &&
                uri != null &&
                uri.scheme == expected.scheme &&
                uri.host == expected.host &&
                uri.port == expected.port &&
                uri.path == expected.path &&
                uri.query == expected.query;
            return allowed
                ? NavigationDecision.navigate
                : NavigationDecision.prevent;
          },
          onWebResourceError: (error) {
            if (error.isForMainFrame == true &&
                mounted &&
                generation == _generation) {
              setState(() => _error = '画面を読み込めませんでした。再読み込みをお試しください。');
            }
          },
        ),
      );
      if (!mounted || generation != _generation) return;
      setState(() {
        _bound = bound;
        _input = input;
        _controller = controller;
        _loading = false;
      });
      await controller.loadRequest(expected);
    } on Object catch (error) {
      if (mounted && generation == _generation) {
        setState(() {
          _error = error is SimulationException
              ? error.message
              : 'データを取得できませんでした。もう一度お試しください。';
          _loading = false;
        });
      }
    }
  }

  void _message(String raw, SimulationContext context) {
    if (!mounted || _bound?.storageKey != context.storageKey) return;
    try {
      final message = jsonDecode(raw) as Map<String, dynamic>;
      if (message['version'] != 1 ||
          message['raceId'] != context.raceId ||
          message['date'] != context.date ||
          message['inputVersion'] != context.inputVersion ||
          message['modelVersion'] != context.modelVersion) {
        return;
      }
      final data = message['data'] as Map<String, dynamic>? ?? {};
      switch (message['type']) {
        case 'ready':
          setState(() {
            _ready = true;
            _batch = _restore?['batch'] as Map<String, dynamic>?;
            _comparison = _restore?['comparison'] as Map<String, dynamic>?;
            _profiles = (data['horses'] as List).cast<Map<String, dynamic>>();
            _courses = (data['courses'] as List).cast<Map<String, dynamic>>();
          });
          unawaited(_send('restore', _restore ?? {}));
          unawaited(_send('active', {'active': widget.active}));
          unawaited(_event('sim_app_open'));
        case 'state':
          setState(() {
            _state = data;
            _profiles = (data['profiles'] as List? ?? _profiles)
                .cast<Map<String, dynamic>>();
          });
        case 'selected':
          setState(() => _state = {...?_state, 'selected': data['horseNo']});
        case 'batchProgress':
          setState(() => _completed = data['completed'] as int);
        case 'batchComplete':
          setState(() {
            _completed = 0;
            _batch = data['tallies'] as Map<String, dynamic>;
          });
        case 'inputUpdate':
          setState(() => _updatedVersion = data['inputVersion'] as String?);
        case 'complete':
          unawaited(_firstComplete());
        case 'error':
          setState(() => _error = data['message'] as String?);
        case 'analytics':
          final name = data['name'];
          if (const [
            'sim_settings_change',
            'sim_batch_complete',
            'sim_app_start',
            'sim_app_complete',
          ].contains(name)) {
            unawaited(_event(name as String));
          }
      }
    } on Object {
      /* Ignore stale or incompatible bridge messages. */
    }
  }

  Future<void> _event(String name) => simulationEvent(name, {
    'race_id': _bound?.raceId ?? widget.context.raceId,
    'input_version': _bound?.inputVersion ?? '',
    'model_version': widget.context.modelVersion,
    'source': widget.context.source,
    'protocol_version': 1,
  });
  Future<void> _firstComplete() async {
    final prefs = await SharedPreferences.getInstance();
    if (prefs.getBool('simulation.v1.first_complete') == true) return;
    await prefs.setBool('simulation.v1.first_complete', true);
    await _event('sim_app_first_complete');
  }

  Future<void> _send(
    String type, [
    Map<String, dynamic> data = const {},
  ]) async {
    if (!_ready || _bound == null || _controller == null) return;
    final message = {
      'version': 1,
      ..._bound!.identity,
      'requestId': '${++_sequence}',
      'type': type,
      'data': data,
    };
    try {
      await _controller!.runJavaScript(
        'window.keibaSimulator?.receive(${jsonEncode(message)});',
      );
    } on Object {
      /* A reload may have invalidated the channel. */
    }
  }

  Future<void> saveAndPause() async {
    await _send('pause');
    await _send('cancelBatch');
    if (_completed > 0 && mounted) setState(() => _completed = 0);
    if (_bound != null && _state != null) {
      await _storage.saveScene(_bound!, {
        ..._state!,
        'batch': _batch,
        'comparison': _comparison,
      });
    }
  }

  void _change(Map<String, dynamic> values) {
    setState(() {
      _batch = null;
      _completed = 0;
    });
    unawaited(_send('settings', values));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _generation++;
    _service.cancel();
    unawaited(_send('pause'));
    unawaited(_send('cancelBatch'));
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final settings = _state?['settings'] as Map<String, dynamic>? ?? {};
    final time = (_state?['time'] as num? ?? 0).toDouble(),
        duration = (_state?['duration'] as num? ?? 1).toDouble();
    final selected = _state?['selected'];
    final profile = _profiles
        .where((p) => p['horse_no'] == selected)
        .firstOrNull;
    final horses = (_state?['horses'] as List? ?? [])
        .cast<Map<String, dynamic>>();
    final horseState = horses
        .where((h) => h['horse_no'] == selected)
        .firstOrNull;
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        Row(
          children: [
            IconButton(
              onPressed: widget.onBack,
              tooltip: 'レース選択に戻る',
              icon: const Icon(Icons.arrow_back),
            ),
            Expanded(
              child: Text(
                '${_input?['race_info']?['course'] ?? ''} ${int.tryParse(widget.context.raceId.substring(10)) ?? ''}R',
                style: Theme.of(context).textTheme.titleLarge,
              ),
            ),
            IconButton(
              onPressed: () => _load(widget.context),
              tooltip: '再読み込み',
              icon: const Icon(Icons.refresh),
            ),
          ],
        ),
        if (_loading)
          const Center(
            child: Padding(
              padding: EdgeInsets.all(24),
              child: CircularProgressIndicator(),
            ),
          ),
        if (_error != null)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Text(_error!),
                  TextButton(
                    onPressed: () => _load(
                      SimulationContext(
                        raceId: widget.context.raceId,
                        date: widget.context.date,
                      ),
                    ),
                    child: const Text('最新のデータを選ぶ'),
                  ),
                ],
              ),
            ),
          ),
        if (_updatedVersion != null)
          Card(
            child: ListTile(
              title: const Text('新しいデータがあります'),
              trailing: TextButton(
                onPressed: () => _load(
                  SimulationContext(
                    raceId: widget.context.raceId,
                    date: widget.context.date,
                  ),
                ),
                child: const Text('更新して試走'),
              ),
            ),
          ),
        if (_controller != null)
          SizedBox(
            height: 370,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: WebViewWidget(controller: _controller!),
            ),
          ),
        if (_controller != null && !_ready)
          const Padding(
            padding: EdgeInsets.all(12),
            child: Text('シミュレーションを準備しています。'),
          ),
        if (_ready) ...[
          Text('再生位置 ${(time / duration * 100).clamp(0, 100).round()}%'),
          Slider(
            value: time.clamp(0, duration),
            max: duration,
            onChanged: (v) => _send('seek', {'time': v}),
          ),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              FilledButton.icon(
                onPressed: () =>
                    _send(_state?['running'] == true ? 'pause' : 'play'),
                icon: Icon(
                  _state?['running'] == true ? Icons.pause : Icons.play_arrow,
                ),
                label: Text(_state?['running'] == true ? '一時停止' : '再生'),
              ),
              OutlinedButton(
                onPressed: () => _send('seek', {'time': 0}),
                child: const Text('最初に戻す'),
              ),
              OutlinedButton(
                onPressed: () => _change({
                  'seed':
                      (((settings['seed'] as int?) ?? _bound!.seed!) + 1) &
                      0xffffffff,
                }),
                child: const Text('別の展開で試走'),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _choice('再生速度', settings['speed'] ?? 4, const [
            (1, '1倍'),
            (2, '2倍'),
            (4, '4倍'),
            (8, '8倍'),
          ], (v) => _send('settings', {'speed': v})),
          _choice('表示', settings['view'] ?? 'course', const [
            ('course', '全体コース'),
            ('pack', '馬群拡大'),
          ], (v) => _send('settings', {'view': v})),
          const Divider(height: 28),
          Text('展開の設定', style: Theme.of(context).textTheme.titleMedium),
          _choice('前半のペース', settings['pace'] ?? 'average', const [
            ('slow', 'スロー'),
            ('average', 'ミドル'),
            ('fast', 'ハイ'),
          ], (v) => _change({'pace': v})),
          _choice(
            'スタート・調子のばらつき',
            settings['variation'] ?? 1,
            const [(0.4, '小さめ'), (1, '標準'), (1.7, '大きめ')],
            (v) => _change({'variation': v}),
          ),
          DropdownButtonFormField<String>(
            key: ValueKey(settings['courseProfileId']),
            initialValue: settings['courseProfileId'] as String?,
            isExpanded: true,
            decoration: const InputDecoration(labelText: '比較するコース'),
            items: [
              for (final c in _courses)
                DropdownMenuItem(
                  value: c['id'] as String,
                  child: Text(c['label'] as String),
                ),
            ],
            onChanged: (v) => _change({'courseProfileId': v}),
          ),
          const SizedBox(height: 8),
          DropdownButtonFormField<int>(
            key: ValueKey(settings['distance']),
            initialValue: settings['distance'] as int?,
            decoration: const InputDecoration(labelText: '比較する距離'),
            items: [
              for (final d in {
                800,
                1000,
                1200,
                1400,
                1600,
                1800,
                2000,
                2400,
                3000,
                if (settings['distance'] != null) settings['distance'] as int,
              })
                DropdownMenuItem(value: d, child: Text('${d}m')),
            ],
            onChanged: (v) => _change({'distance': v}),
          ),
          if (settings['distance'] != null ||
              settings['courseProfileId'] != null)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 8),
              child: Text('比較用の条件で試走しています。出走馬と近走データは元のレースのものです。'),
            ),
          if (_profiles.isNotEmpty) ...[
            const Divider(height: 28),
            DropdownButton<int>(
              isExpanded: true,
              value: _profiles.any((p) => p['horse_no'] == selected)
                  ? selected as int?
                  : null,
              hint: const Text('馬ごとの状態'),
              items: [
                for (final p in _profiles)
                  DropdownMenuItem(
                    value: p['horse_no'] as int,
                    child: Text('${p['horse_no']} ${p['horse_name']}'),
                  ),
              ],
              onChanged: (v) => _send('select', {'horseNo': v}),
            ),
            if (profile != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(14),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '脚質 ${profile['style']} · 近走 ${profile['runCount']}走',
                      ),
                      for (final item in const [
                        ('ability', '基礎能力'),
                        ('early', '先行力'),
                        ('stamina', '持続力'),
                        ('closing', '末脚'),
                      ])
                        Text(
                          '${item.$2} ${((profile[item.$1] as num) * 100).round()}',
                        ),
                      if (horseState != null)
                        Text(
                          '模擬スタミナ残量 ${((horseState['energy'] as num) * 100).round()}%',
                        ),
                      for (final assumption
                          in profile['assumptions'] as List? ?? [])
                        Text(assumption.toString()),
                      const Text('能力とスタミナは、近走からの推定値です。'),
                    ],
                  ),
                ),
              ),
          ],
          const Divider(height: 28),
          FilledButton.icon(
            onPressed: () {
              if (_completed > 0) {
                unawaited(_send('cancelBatch'));
                setState(() => _completed = 0);
              } else {
                setState(() => _completed = 1);
                unawaited(_send('batch'));
              }
            },
            icon: const Icon(Icons.bar_chart),
            label: Text(
              _completed > 0 ? '集計を中止（$_completed / 200）' : '200回の展開を試す',
            ),
          ),
          if (_batch != null) ...[
            const Text('表示値は、このモデルで200回試走したときの出現頻度です。'),
            for (final p in _profiles)
              if (_batch!.containsKey('${p['horse_no']}'))
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Text(
                    '${p['horse_no']} ${p['horse_name']} · 3着以内 ${((_batch!['${p['horse_no']}']['top3'] as num) / 2).toStringAsFixed(1)}%${_comparison == null ? '' : '（保存した条件 ${((_comparison!['${p['horse_no']}']?['top3'] as num? ?? 0) / 2).toStringAsFixed(1)}%）'}',
                  ),
                ),
            OutlinedButton(
              onPressed: () => setState(
                () => _comparison = Map<String, dynamic>.from(_batch!),
              ),
              child: const Text('この集計を比較用に残す'),
            ),
          ],
          const SizedBox(height: 16),
          Wrap(
            spacing: 8,
            children: [
              OutlinedButton(
                onPressed: () async {
                  await saveAndPause();
                  if (context.mounted) {
                    ScaffoldMessenger.of(
                      context,
                    ).showSnackBar(const SnackBar(content: Text('設定を保存しました。')));
                  }
                },
                child: const Text('設定を保存'),
              ),
              OutlinedButton(
                onPressed: () async {
                  final saved = await _storage.load(
                    'scene.${_bound!.storageKey}',
                  );
                  if (saved != null) await _send('restore', saved);
                },
                child: const Text('保存した設定に戻す'),
              ),
              TextButton(
                onPressed: () {
                  _comparison = null;
                  _batch = null;
                  unawaited(_send('reset'));
                },
                child: const Text('元の条件に戻す'),
              ),
            ],
          ),
          Text(
            '${_input?['source_as_of'] == null ? '取得時刻不明・データ公開' : 'データ取得'} ${_input?['source_as_of'] ?? _input?['published_at'] ?? ''}\n${_bound?.inputVersion ?? ''}',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 24),
        ],
      ],
    );
  }

  Widget _choice(
    String label,
    Object value,
    List<(Object, String)> choices,
    ValueChanged<Object> changed,
  ) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 8),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label),
        const SizedBox(height: 5),
        Wrap(
          spacing: 6,
          children: [
            for (final c in choices)
              ChoiceChip(
                label: Text(c.$2),
                selected: value == c.$1,
                onSelected: (_) => changed(c.$1),
              ),
          ],
        ),
      ],
    ),
  );
}

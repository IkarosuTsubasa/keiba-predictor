class SimulationContext {
  const SimulationContext({
    required this.raceId,
    required this.date,
    this.inputVersion = '',
    this.modelVersion = 'illustrative-terrain-v2',
    this.seed,
    this.source = 'app_tab',
  });
  final String raceId, date, inputVersion, modelVersion, source;
  final int? seed;

  static SimulationContext? fromUri(Uri uri, Uri base) {
    if (uri.hasScheme &&
        (uri.scheme != base.scheme ||
            uri.host != base.host ||
            uri.port != base.port)) {
      return null;
    }
    final match = RegExp(
      r'^/keiba/open/simulator/(\d{12})/?$',
    ).firstMatch(uri.path);
    if (match == null) return null;
    final q = uri.queryParameters;
    final day = q['date'] ?? '';
    if (!RegExp(r'^\d{4}-\d{2}-\d{2}$').hasMatch(day) ||
        DateTime.tryParse(day)?.toIso8601String().substring(0, 10) != day) {
      return null;
    }
    final version = q['iv'] ?? '';
    if (version.isNotEmpty && !RegExp(r'^[a-f0-9]{24}$').hasMatch(version)) {
      return null;
    }
    final model = q['mv'] ?? 'illustrative-terrain-v2';
    if (model != 'illustrative-terrain-v2') return null;
    final seed = int.tryParse(q['seed'] ?? '');
    if (q.containsKey('seed') &&
        (seed == null || seed < 0 || seed > 0xffffffff)) {
      return null;
    }
    return SimulationContext(
      raceId: match[1]!,
      date: day,
      inputVersion: version,
      modelVersion: model,
      seed: seed,
      source: q['source'] == 'app_race_detail'
          ? 'app_race_detail'
          : 'web_detail',
    );
  }

  SimulationContext bind(Map<String, dynamic> data) => SimulationContext(
    raceId: raceId,
    date: date,
    inputVersion: data['input_version'] as String,
    modelVersion: data['active_model_version'] as String,
    seed: seed ?? data['default_seed'] as int,
    source: source,
  );
  String get storageKey => '$date:$raceId:$inputVersion';
  Map<String, dynamic> get identity => {
    'raceId': raceId,
    'date': date,
    'inputVersion': inputVersion,
    'modelVersion': modelVersion,
  };
  Uri embedUri(String base) => Uri.parse('$base/simulator/embed').replace(
    queryParameters: {
      'race_id': raceId,
      'date': date,
      'iv': inputVersion,
      'mv': modelVersion,
      if (seed != null) 'seed': '$seed',
      'source': source,
      'app': '1',
    },
  );
  Map<String, dynamic> toJson() => {
    ...identity,
    'seed': seed,
    'source': source,
  };
  static SimulationContext fromJson(Map<String, dynamic> row) =>
      SimulationContext(
        raceId: row['raceId'] as String,
        date: row['date'] as String,
        inputVersion: row['inputVersion'] as String? ?? '',
        modelVersion:
            row['modelVersion'] as String? ?? 'illustrative-terrain-v2',
        seed: row['seed'] as int?,
        source: 'app_tab',
      );
}

import 'dart:math' as math;

class CustomPredictionDay {
  const CustomPredictionDay({
    required this.targetDate,
    required this.targetDateLabel,
    required this.fallbackNotice,
    required this.races,
  });

  factory CustomPredictionDay.fromApi(Map<String, dynamic> payload) {
    final data = _asMap(payload['data']);
    final races = _asList(data['races'])
        .map(_asMap)
        .where((item) => item.isNotEmpty)
        .map(CustomPredictionRace.fromJson)
        .toList(growable: false);
    return CustomPredictionDay(
      targetDate: _text(data['target_date']),
      targetDateLabel: _text(data['target_date_label']),
      fallbackNotice: _text(data['fallback_notice']),
      races: races,
    );
  }

  final String targetDate;
  final String targetDateLabel;
  final String fallbackNotice;
  final List<CustomPredictionRace> races;
}

class CustomPredictionRace {
  const CustomPredictionRace({
    required this.raceId,
    required this.raceTitle,
    required this.raceName,
    required this.course,
    required this.distanceLabel,
    required this.trackCondition,
    required this.horses,
  });

  factory CustomPredictionRace.fromJson(Map<String, dynamic> json) {
    return CustomPredictionRace(
      raceId: _text(json['race_id']),
      raceTitle: _text(json['race_title']),
      raceName: _text(json['race_name']),
      course: _text(json['course']),
      distanceLabel: _text(json['distance_label']),
      trackCondition: _text(json['track_condition']),
      horses: _asList(json['horses'])
          .map(_asMap)
          .where((item) => item.isNotEmpty)
          .map(CustomPredictionHorse.fromJson)
          .toList(growable: false),
    );
  }

  final String raceId;
  final String raceTitle;
  final String raceName;
  final String course;
  final String distanceLabel;
  final String trackCondition;
  final List<CustomPredictionHorse> horses;

  String get displayName {
    if (raceName.isEmpty) return raceTitle;
    return '$raceTitle  $raceName';
  }

  List<CustomPredictionPick> calculate(CustomPredictionParameters parameters) {
    const marks = ['◎', '○', '▲', '△', '☆'];
    if (horses.isEmpty) return const [];

    final baselineScores = horses
        .map((horse) => horse.baselineScore)
        .toList(growable: false);
    final scored = <_ScoredHorse>[];
    if (parameters.hasUniformLevels) {
      for (var index = 0; index < horses.length; index++) {
        scored.add(_ScoredHorse(horses[index], baselineScores[index]));
      }
    } else {
      final normalizedBaseline = _normalizeAcrossRace(
        baselineScores,
        minRange: 0.5,
        minDeviation: 3.0,
      );
      final factorVectors = <List<double>>[
        _normalizeAcrossRace(
          horses.map((horse) => horse.ability).toList(growable: false),
          minRange: 1.0,
          minDeviation: 2.0,
        ),
        _normalizeAcrossRace(
          horses
              .map((horse) => horse.recentQuality + 0.5 * horse.trend)
              .toList(growable: false),
          minRange: 1.0,
          minDeviation: 2.0,
        ),
        _normalizeAcrossRace(
          horses.map((horse) => horse.condition).toList(growable: false),
          minRange: 1.0,
          minDeviation: 2.0,
        ),
        _normalizeAcrossRace(
          horses
              .map((horse) => horse.raceLevel + 2.0 * horse.raceLevelAdjustment)
              .toList(growable: false),
          minRange: 0.75,
          minDeviation: 2.0,
        ),
        _normalizeAcrossRace(
          horses
              .map((horse) => horse.paceJockey + 2.0 * horse.paceAdjustment)
              .toList(growable: false),
          minRange: 0.75,
          minDeviation: 2.0,
        ),
        _normalizeAcrossRace(
          horses.map((horse) => horse.pedigree).toList(growable: false),
          minRange: 0.25,
          minDeviation: 0.5,
        ),
        _normalizeAcrossRace(
          horses.map((horse) => horse.risk).toList(growable: false),
          minRange: 1.0,
          minDeviation: 2.0,
        ),
      ];
      final levels = parameters.levels;
      final importance = levels
          .map((level) => math.pow(1.9, level).toDouble())
          .toList(growable: false);
      final importanceTotal = importance.fold<double>(
        0.0,
        (sum, value) => sum + value,
      );
      final standardShare = 1.0 / factorVectors.length;

      for (var horseIndex = 0; horseIndex < horses.length; horseIndex++) {
        var preferenceAdjustment = 0.0;
        for (
          var factorIndex = 0;
          factorIndex < factorVectors.length;
          factorIndex++
        ) {
          final personalizedShare = importance[factorIndex] / importanceTotal;
          preferenceAdjustment +=
              (personalizedShare - standardShare) *
              factorVectors[factorIndex][horseIndex];
        }
        scored.add(
          _ScoredHorse(
            horses[horseIndex],
            normalizedBaseline[horseIndex] + 1.15 * preferenceAdjustment,
          ),
        );
      }
    }
    scored.sort((left, right) {
      final scoreOrder = right.score.compareTo(left.score);
      if (scoreOrder != 0) return scoreOrder;
      return left.horse.horseNo.compareTo(right.horse.horseNo);
    });
    return [
      for (
        var index = 0;
        index < math.min(marks.length, scored.length);
        index++
      )
        CustomPredictionPick(
          mark: marks[index],
          horseNo: scored[index].horse.horseNo,
          horseName: scored[index].horse.horseName,
        ),
    ];
  }

  static List<double> _normalizeAcrossRace(
    List<double> values, {
    required double minRange,
    required double minDeviation,
  }) {
    if (values.isEmpty) return const [];
    final minimum = values.reduce(math.min);
    final maximum = values.reduce(math.max);
    if (maximum - minimum < minRange) {
      return List<double>.filled(values.length, 0.0, growable: false);
    }
    final mean =
        values.fold<double>(0.0, (sum, value) => sum + value) / values.length;
    final variance =
        values.fold<double>(
          0.0,
          (sum, value) => sum + math.pow(value - mean, 2).toDouble(),
        ) /
        values.length;
    final scale = math.max(math.sqrt(variance), minDeviation);
    return values
        .map((value) => ((value - mean) / scale).clamp(-2.0, 2.0).toDouble())
        .toList(growable: false);
  }
}

class CustomPredictionHorse {
  const CustomPredictionHorse({
    required this.horseNo,
    required this.horseName,
    required this.base,
    required this.pedigree,
    required this.raceLevelAdjustment,
    required this.paceAdjustment,
    required this.pedigreeWeight,
    required this.raceLevelWeight,
    required this.paceWeight,
    required this.ability,
    required this.recentQuality,
    required this.trend,
    required this.condition,
    required this.raceLevel,
    required this.paceJockey,
    required this.risk,
  });

  factory CustomPredictionHorse.fromJson(Map<String, dynamic> json) {
    final factors = _asMap(json['factors']);
    final weights = _asMap(json['default_weights']);
    return CustomPredictionHorse(
      horseNo: _integer(json['horse_no']),
      horseName: _text(json['horse_name']),
      base: _number(factors['base']),
      pedigree: _number(factors['pedigree']),
      raceLevelAdjustment: _number(factors['race_level_adjustment']),
      paceAdjustment: _number(factors['pace_adjustment']),
      pedigreeWeight: _number(weights['pedigree']),
      raceLevelWeight: _number(weights['race_level']),
      paceWeight: _number(weights['pace']),
      ability: _number(factors['ability']),
      recentQuality: _number(factors['recent_quality']),
      trend: _number(factors['trend']),
      condition: _number(factors['condition']),
      raceLevel: _number(factors['race_level']),
      paceJockey: _number(factors['pace_jockey']),
      risk: _number(factors['risk']),
    );
  }

  final int horseNo;
  final String horseName;
  final double base;
  final double pedigree;
  final double raceLevelAdjustment;
  final double paceAdjustment;
  final double pedigreeWeight;
  final double raceLevelWeight;
  final double paceWeight;
  final double ability;
  final double recentQuality;
  final double trend;
  final double condition;
  final double raceLevel;
  final double paceJockey;
  final double risk;

  double get baselineScore =>
      base +
      pedigree * pedigreeWeight +
      raceLevelAdjustment * raceLevelWeight +
      paceAdjustment * paceWeight;
}

class CustomPredictionParameters {
  const CustomPredictionParameters({
    this.abilityLevel = 0,
    this.recentLevel = 0,
    this.conditionLevel = 0,
    this.opponentLevel = 0,
    this.paceJockeyLevel = 0,
    this.pedigreeLevel = 0,
    this.stabilityLevel = 0,
  });

  static const minLevel = -2;
  static const maxLevel = 2;

  final int abilityLevel;
  final int recentLevel;
  final int conditionLevel;
  final int opponentLevel;
  final int paceJockeyLevel;
  final int pedigreeLevel;
  final int stabilityLevel;

  List<int> get levels => [
    abilityLevel,
    recentLevel,
    conditionLevel,
    opponentLevel,
    paceJockeyLevel,
    pedigreeLevel,
    stabilityLevel,
  ];

  bool get hasUniformLevels => levels.every((level) => level == levels.first);

  bool get isStandard =>
      abilityLevel == 0 &&
      recentLevel == 0 &&
      conditionLevel == 0 &&
      opponentLevel == 0 &&
      paceJockeyLevel == 0 &&
      pedigreeLevel == 0 &&
      stabilityLevel == 0;

  CustomPredictionParameters copyWith({
    int? abilityLevel,
    int? recentLevel,
    int? conditionLevel,
    int? opponentLevel,
    int? paceJockeyLevel,
    int? pedigreeLevel,
    int? stabilityLevel,
  }) {
    return CustomPredictionParameters(
      abilityLevel: _clampLevel(abilityLevel ?? this.abilityLevel),
      recentLevel: _clampLevel(recentLevel ?? this.recentLevel),
      conditionLevel: _clampLevel(conditionLevel ?? this.conditionLevel),
      opponentLevel: _clampLevel(opponentLevel ?? this.opponentLevel),
      paceJockeyLevel: _clampLevel(paceJockeyLevel ?? this.paceJockeyLevel),
      pedigreeLevel: _clampLevel(pedigreeLevel ?? this.pedigreeLevel),
      stabilityLevel: _clampLevel(stabilityLevel ?? this.stabilityLevel),
    );
  }

  static int _clampLevel(int value) => value.clamp(minLevel, maxLevel);

  static String labelFor(int value) {
    return switch (_clampLevel(value)) {
      -2 => '軽視',
      -1 => 'やや軽視',
      0 => '標準',
      1 => 'やや重視',
      _ => '重視',
    };
  }
}

class CustomPredictionPick {
  const CustomPredictionPick({
    required this.mark,
    required this.horseNo,
    required this.horseName,
  });

  final String mark;
  final int horseNo;
  final String horseName;
}

class _ScoredHorse {
  const _ScoredHorse(this.horse, this.score);

  final CustomPredictionHorse horse;
  final double score;
}

Map<String, dynamic> _asMap(Object? value) {
  if (value is! Map) return const {};
  return value.map((key, item) => MapEntry(key.toString(), item));
}

List<dynamic> _asList(Object? value) => value is List ? value : const [];

String _text(Object? value) => value?.toString().trim() ?? '';

double _number(Object? value) {
  if (value is num) return value.toDouble();
  return double.tryParse(value?.toString() ?? '') ?? 0.0;
}

int _integer(Object? value) {
  if (value is num) return value.toInt();
  return int.tryParse(value?.toString() ?? '') ?? 0;
}

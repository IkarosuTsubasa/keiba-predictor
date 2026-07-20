import 'package:flutter_test/flutter_test.dart';
import 'package:ikaimo_keiba/src/models/custom_prediction.dart';

void main() {
  test('标准参数保持默认因子权重且不暴露分数', () {
    final race = CustomPredictionRace.fromJson({
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
          'factors': {
            'base': 10,
            'pedigree': 0,
            'race_level_adjustment': 0,
            'pace_adjustment': 0,
            'pace_jockey': 10,
          },
          'default_weights': {'pedigree': 0.2, 'race_level': 1.0, 'pace': 0},
        },
        {
          'horse_no': 2,
          'horse_name': 'ホースB',
          'factors': {
            'base': 12,
            'pedigree': 0,
            'race_level_adjustment': 0,
            'pace_adjustment': 0,
            'pace_jockey': 0,
          },
          'default_weights': {'pedigree': 0.2, 'race_level': 1.0, 'pace': 0},
        },
      ],
    });

    final picks = race.calculate(const CustomPredictionParameters());

    expect(picks.first.mark, '◎');
    expect(picks.first.horseNo, 2);
    expect(picks.first.horseName, 'ホースB');

    final equalStrongPicks = race.calculate(
      const CustomPredictionParameters(
        abilityLevel: 2,
        recentLevel: 2,
        conditionLevel: 2,
        opponentLevel: 2,
        paceJockeyLevel: 2,
        pedigreeLevel: 2,
        stabilityLevel: 2,
      ),
    );
    expect(equalStrongPicks.first.horseNo, 2);
  });

  test('增强展开影响后在本地重新排序', () {
    final race = CustomPredictionRace.fromJson({
      'race_id': '202601010101',
      'race_title': '東京1R',
      'horses': [
        {
          'horse_no': 1,
          'horse_name': 'ホースA',
          'factors': {'base': 10, 'pace_jockey': 10},
          'default_weights': {'pace': 0},
        },
        {
          'horse_no': 2,
          'horse_name': 'ホースB',
          'factors': {'base': 11, 'pace_jockey': 0},
          'default_weights': {'pace': 0},
        },
      ],
    });

    final picks = race.calculate(
      const CustomPredictionParameters(paceJockeyLevel: 2),
    );

    expect(picks.first.horseNo, 1);
    expect(picks.map((pick) => pick.mark), ['◎', '○']);
  });

  test('鲜明偏好可以改变前五成员而不需要逐场打分', () {
    final race = CustomPredictionRace.fromJson({
      'race_id': '202601010102',
      'race_title': '東京2R',
      'horses': [
        for (var index = 0; index < 6; index++)
          {
            'horse_no': index + 1,
            'horse_name': 'ホース${index + 1}',
            'factors': {'base': 15 - index, 'ability': index == 5 ? 10 : 0},
          },
      ],
    });

    final standard = race.calculate(const CustomPredictionParameters());
    final abilityFocused = race.calculate(
      const CustomPredictionParameters(abilityLevel: 2),
    );

    expect(standard.map((pick) => pick.horseNo), [1, 2, 3, 4, 5]);
    expect(abilityFocused.map((pick) => pick.horseNo), contains(6));
  });

  test('参数档位会限制在允许范围内', () {
    final parameters = const CustomPredictionParameters().copyWith(
      abilityLevel: 10,
      stabilityLevel: -10,
    );

    expect(parameters.abilityLevel, CustomPredictionParameters.maxLevel);
    expect(parameters.stabilityLevel, CustomPredictionParameters.minLevel);
    expect(CustomPredictionParameters.labelFor(0), '標準');
  });
}

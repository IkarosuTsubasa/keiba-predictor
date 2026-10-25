"""Market-free class relationships appended to the frozen 292 turf features."""
from __future__ import annotations

import numpy as np
from keiba_llm_agent.analysis.race_class import infer_race_class_level
from keiba_llm_agent.scoring import turf_history_features as history
from keiba_llm_agent.scoring.turf_place_features import absolute_and_relative, require_turf

FEATURE_VERSION = 'central_turf_class_context_v1'
CLASS_NAMES = ('official_target', 'last_level', 'target_minus_last', 'last3_known_fraction',
    'last3_mean_level', 'target_minus_last3_mean', 'last8_max_level', 'last8_place_max_level',
    'target_minus_place_max', 'last8_at_target_count', 'last8_at_target_place_rate',
    'last8_at_target_margin_mean')
FEATURE_NAMES = (history.FEATURE_NAMES + tuple('context_'+n for n in CLASS_NAMES)
                 + tuple('context_relative_'+n for n in CLASS_NAMES))


def _difference(a, b):
    return a-b if a is not None and b is not None else None


def class_relationship_features(race):
    require_turf(race)
    rows = []
    target = infer_race_class_level(race.race_info.race_class)
    for horse in race.horses:
        runs = history.valid_history(horse, race.race_info)
        levels = [infer_race_class_level(r.race_name) for r in runs]
        last = levels[0] if levels else None
        mean = history.finite_stat(levels[:3], 'mean')
        best_place = history.finite_stat([v for r, v in zip(runs[:8], levels[:8], strict=True)
                                         if r.finish <= 3], 'max')
        comparable = [r for r, v in zip(runs[:8], levels[:8], strict=True)
                      if target is not None and v is not None and v >= target]
        rows.append([target, last, _difference(target, last),
            sum(v is not None for v in levels[:3])/len(levels[:3]) if levels else None,
            mean, _difference(target, mean), history.finite_stat(levels[:8], 'max'), best_place,
            _difference(target, best_place), len(comparable) if target is not None else None,
            np.mean([r.finish <= 3 for r in comparable]) if comparable else None,
            history.finite_stat([history.comparable_margin(r) for r in comparable], 'mean')])
    return absolute_and_relative(rows)


def build_turf_class_matrix(race, horse_scores):
    base, numbers = history.build_turf_history_matrix(race, horse_scores)
    matrix = np.column_stack((base, class_relationship_features(race)))
    if matrix.shape != (len(numbers), len(FEATURE_NAMES)) or np.isinf(matrix).any():
        raise ValueError('Invalid turf class feature matrix')
    return matrix, numbers

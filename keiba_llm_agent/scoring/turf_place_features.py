"""Market-free supplementary features for offline central-turf place models."""
from __future__ import annotations

from datetime import date
import numpy as np

from keiba_llm_agent.analysis.race_class import CENTRAL_COURSES, infer_race_class_level
from keiba_llm_agent.schemas.race_data import RaceData

META_NAMES = ('age', 'female', 'gelding', 'official_class', 'class_delta_from_last')
WINDOWS = ('last3', 'same_surface5', 'same_course_distance5')
HISTORY_STATS = ('time_count', 'speed_mean', 'speed_best', 'distance_mean',
                 'carried_weight_mean', 'body_weight_mean', 'body_weight_std',
                 'pace_balance_mean', 'closing_vs_race_mean')
HISTORY_NAMES = ('carried_weight_delta', 'last_body_weight', 'last_body_weight_change') + tuple(
    f'{window}_{name}' for window in WINDOWS for name in HISTORY_STATS)


def require_turf(race: RaceData) -> None:
    info = race.race_info
    if (info.source == 'local' or str(info.scope_key or '').startswith('local')
            or info.surface != '芝' or info.course not in CENTRAL_COURSES):
        raise ValueError('Supplementary place features are central-turf only')


def prior_runs(horse, info):
    target = date.fromisoformat(info.race_date)
    unique = {}
    for run in horse.recent_runs:
        if run.finish is None or not run.date or run.race_id == info.race_id:
            continue
        try:
            if date.fromisoformat(run.date) >= target:
                continue
        except ValueError:
            continue
        unique.setdefault(run.race_id or (run.date, run.course, run.distance), run)
    return sorted(unique.values(), key=lambda r: r.date, reverse=True)


def stat(values, operation='mean'):
    valid = [v for v in values if v is not None and np.isfinite(v)]
    if not valid:
        return np.nan
    return float({'mean': np.mean, 'max': np.max, 'std': np.std}[operation](valid))


def absolute_and_relative(rows):
    matrix = np.asarray([[np.nan if x is None else x for x in row] for row in rows], dtype=float)
    relative = np.full_like(matrix, np.nan)
    for column in range(matrix.shape[1]):
        good = np.isfinite(matrix[:, column])
        if good.any():
            values = matrix[good, column]
            relative[good, column] = (values-values.mean()) / max(float(values.std()), 1e-8)
    return np.column_stack((matrix, relative))


def history_stats(runs):
    speeds = [r.distance/r.time_seconds for r in runs if r.distance and r.time_seconds and r.time_seconds > 0]
    pace = [r.pace_first_3f-r.pace_last_3f for r in runs
            if r.pace_first_3f is not None and r.pace_last_3f is not None]
    closing = [r.final_3f-r.pace_last_3f for r in runs
               if r.final_3f is not None and r.pace_last_3f is not None]
    return [len(speeds), stat(speeds), stat(speeds, 'max'), stat([r.distance for r in runs]),
            stat([r.carried_weight for r in runs]), stat([r.body_weight for r in runs]),
            stat([r.body_weight for r in runs], 'std'), stat(pace), stat(closing)]


def build_supplementary_features(race: RaceData) -> dict[str, np.ndarray]:
    require_turf(race)
    meta = []; history = []
    target_class = infer_race_class_level(race.race_info.race_class)
    for horse in race.horses:
        runs = prior_runs(horse, race.race_info)
        last = runs[0] if runs else None
        last_class = infer_race_class_level(last.race_name) if last else None
        meta.append([horse.age, int(horse.sex == '牝') if horse.sex else None,
                     int(horse.sex == 'セ') if horse.sex else None, target_class,
                     target_class-last_class if target_class is not None and last_class is not None else None])
        # Current-race body weight is deliberately excluded: old early predictions may not have had it.
        values = [horse.carried_weight-last.carried_weight
                  if last and horse.carried_weight is not None and last.carried_weight is not None else None,
                  last.body_weight if last else None, last.body_weight_change if last else None]
        groups = [runs[:3], [r for r in runs if r.surface == race.race_info.surface][:5],
                  [r for r in runs if r.surface == race.race_info.surface and r.course == race.race_info.course
                   and r.distance and race.race_info.distance and abs(r.distance-race.race_info.distance) <= 200][:5]]
        for group in groups:
            values.extend(history_stats(group))
        history.append(values)
    return {'metadata': absolute_and_relative(meta), 'history': absolute_and_relative(history)}

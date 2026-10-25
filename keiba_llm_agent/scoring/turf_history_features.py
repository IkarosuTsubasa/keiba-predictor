"""Production feature contract matching the frozen history_direct experiment."""
from __future__ import annotations

from datetime import date
import re

import numpy as np

from keiba_llm_agent.analysis.race_class import CENTRAL_COURSES, infer_race_class_level
from keiba_llm_agent.scoring import scope_reranker as reranker
from keiba_llm_agent.scoring.recent_run_scorer import _track_condition_level
from keiba_llm_agent.scoring.turf_place_features import (
    HISTORY_NAMES, build_supplementary_features, require_turf,
)

FEATURE_VERSION = 'central_turf_history_direct_v1'
COURSES = sorted(CENTRAL_COURSES)
WINDOWS = ('last3', 'last8', 'same_surface', 'same_course_distance', 'recent365')
STATS = ('count', 'finish_mean', 'finish_best', 'finish_worst', 'finish_std', 'place_rate',
         'win_rate', 'margin_mean', 'margin_best', 'early_mean', 'early_std', 'gain_mean',
         'final3f_mean', 'class_mean', 'class_max')
CONTEXT = ('field_size', 'distance', 'going', 'class_explicit', 'class_missing', 'history_count',
           'rest_days', 'weight', 'frame_fraction', 'last_distance_delta', 'last_surface_match',
           'same_jockey', 'last_finish', 'last_margin', 'last_early', 'last_gain', 'last_final3f')
RAW_NAMES = CONTEXT + tuple('course_'+c for c in COURSES) + tuple(w+'_'+s for w in WINDOWS for s in STATS)
NATIVE_NAMES = reranker.BASE_FEATURE_NAMES + reranker.COURSE_GEOMETRY_FEATURE_NAMES + reranker.COURSE_STYLE_FEATURE_NAMES
FEATURE_NAMES = (tuple('native_'+n for n in NATIVE_NAMES)
                 + tuple('raw_'+n for n in RAW_NAMES) + tuple('raw_relative_'+n for n in RAW_NAMES)
                 + tuple('history_'+n for n in HISTORY_NAMES) + tuple('history_relative_'+n for n in HISTORY_NAMES))


def finite_stat(values, operation):
    good = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(good):
        return float('nan')
    return float({'mean': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}[operation](good))


def finish_value(run):
    if run.finish is None or not run.field_size or run.field_size <= 1:
        return float('nan')
    return float(np.clip((run.field_size-run.finish)/(run.field_size-1), 0, 1))


def early_value(run):
    if not run.corner_positions or not run.field_size or run.field_size <= 1:
        return float('nan')
    return float(np.clip((run.field_size-run.corner_positions[0])/(run.field_size-1), 0, 1))


def gain_value(run):
    if not run.corner_positions or not run.field_size or run.field_size <= 1 or run.finish is None:
        return float('nan')
    return (run.corner_positions[-1]-run.finish)/(run.field_size-1)


def comparable_margin(run):
    if run.finish == 1:
        return 0.0
    value = str(run.margin or '').strip()
    return abs(float(value)) if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value) else None


def valid_history(horse, info):
    unique = {}
    for run in horse.recent_runs:
        if run.finish is None or not run.date or run.date >= info.race_date or run.race_id == info.race_id:
            continue
        try:
            date.fromisoformat(run.date)
        except ValueError:
            continue
        unique.setdefault(run.race_id or (run.date, run.course, run.distance), run)
    return sorted(unique.values(), key=lambda r: r.date, reverse=True)


def summarize_history(runs):
    finishes = [finish_value(r) for r in runs]
    margins = [comparable_margin(r) for r in runs]
    early = [early_value(r) for r in runs]
    classes = [infer_race_class_level(r.race_name) for r in runs]
    return [len(runs), finite_stat(finishes, 'mean'), finite_stat(finishes, 'max'),
            finite_stat(finishes, 'min'), finite_stat(finishes, 'std'),
            np.mean([r.finish <= 3 for r in runs]) if runs else np.nan,
            np.mean([r.finish == 1 for r in runs]) if runs else np.nan,
            finite_stat(margins, 'mean'), finite_stat(margins, 'min'),
            finite_stat(early, 'mean'), finite_stat(early, 'std'),
            finite_stat([gain_value(r) for r in runs], 'mean'),
            finite_stat([r.final_3f for r in runs], 'mean'),
            finite_stat(classes, 'mean'), finite_stat(classes, 'max')]


def continuous_features(race):
    # Keep the historical experiment as an independent parity oracle, not a runtime dependency.
    info = race.race_info
    target = date.fromisoformat(info.race_date)
    explicit = infer_race_class_level(info.race_name)
    rows = []
    for horse in race.horses:
        runs = valid_history(horse, info)
        last = runs[0] if runs else None
        values = [len(race.horses), info.distance, _track_condition_level(info.track_condition),
            explicit, int(explicit is None), len(runs),
            (target-date.fromisoformat(last.date)).days if last else None,
            horse.carried_weight, horse.frame_no/8 if horse.frame_no else None,
            last.distance-info.distance if last and last.distance and info.distance else None,
            int(last.surface == info.surface) if last and last.surface and info.surface else None,
            int(last.jockey == horse.jockey) if last and last.jockey and horse.jockey else None,
            finish_value(last) if last else None, comparable_margin(last) if last else None,
            early_value(last) if last else None, gain_value(last) if last else None,
            last.final_3f if last else None]
        values.extend(int(info.course == course) for course in COURSES)
        groups = [runs[:3], runs[:8],
                  [r for r in runs if r.surface == info.surface and info.surface][:8],
                  [r for r in runs if r.course == info.course and r.surface == info.surface and
                   r.distance and info.distance and abs(r.distance-info.distance) <= 200][:8],
                  [r for r in runs if (target-date.fromisoformat(r.date)).days <= 365]]
        for group in groups:
            values.extend(summarize_history(group))
        rows.append([float(v) if v is not None else np.nan for v in values])
    matrix = np.asarray(rows, dtype=float)
    relative = np.full_like(matrix, np.nan)
    for j in range(matrix.shape[1]):
        valid = np.isfinite(matrix[:, j])
        if valid.any():
            values = matrix[valid, j]
            relative[valid, j] = (values-values.mean())/max(float(values.std()), 1e-8)
    return np.column_stack((matrix, relative))


def build_turf_history_matrix(race, horse_scores):
    require_turf(race)
    if not race.horses:
        raise ValueError('Empty runner roster')
    numbers = [h.horse_no for h in race.horses]
    if len(set(numbers)) != len(numbers) or sorted(numbers) != sorted(s.horse_no for s in horse_scores):
        raise ValueError('Runner/score roster mismatch')
    # Native features use the original breakdown, not assigned post-reranking display totals.
    native, ordered = reranker.build_scope_reranker_feature_matrix(race, horse_scores, 'course_geometry')
    matrix = np.column_stack((native, continuous_features(race), build_supplementary_features(race)['history']))
    if ordered != numbers or matrix.shape != (len(numbers), len(FEATURE_NAMES)) or np.isinf(matrix).any():
        raise ValueError('Invalid feature matrix')
    return matrix, numbers

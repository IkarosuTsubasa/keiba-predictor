"""Audit-only central-turf challenger. No authoritative ranking switch exists here."""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from keiba_llm_agent.analysis.race_class import CENTRAL_COURSES
from keiba_llm_agent.config.scoring_config import effective_scoring_weights, resolve_scoring_profile_config
from keiba_llm_agent.schemas.prediction import ScoringConfigSnapshot
from keiba_llm_agent.scoring.turf_history_features import FEATURE_NAMES, FEATURE_VERSION, build_turf_history_matrix

ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = 'central_turf_history_shadow_v1_20261025'
MODEL_DIRECTORY = ROOT/'keiba_llm_agent/models/turf_shadow'
SOURCE_FILES = (
    'keiba_llm_agent/scoring/turf_history_features.py',
    'keiba_llm_agent/scoring/turf_place_features.py',
    'keiba_llm_agent/scoring/turf_shadow.py',
    'keiba_llm_agent/scoring/scope_reranker.py',
    'keiba_llm_agent/scoring/recent_run_scorer.py',
    'keiba_llm_agent/scoring/analysis_adjustment_integrator.py',
    'keiba_llm_agent/config/scoring_config.py',
    'keiba_llm_agent/analysis/race_class.py',
    'keiba_llm_agent/analysis/course_geometry.py',
)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def source_hashes():
    # Text-normalized hashes survive Git's Windows/Linux newline conversion.
    return {name: hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for name in SOURCE_FILES}


def is_central_turf(race):
    info = race.race_info
    scope = str(info.scope_key or '').lower()
    return (str(info.source or '').lower() != 'local' and scope in ('', 'central', 'central_turf')
            and info.surface == '芝' and info.course in CENTRAL_COURSES)


def default_scoring_config(race):
    profile, _ = resolve_scoring_profile_config(scoring_profile='accuracy_default')
    config = ScoringConfigSnapshot.model_validate(profile.scoring_config.model_dump())
    return config.model_copy(update=effective_scoring_weights(
        config, surface=race.race_info.surface, field_size=len(race.horses))).model_dump()


@lru_cache(maxsize=2)
def _load_bundle(path_text, model_mtime, metadata_mtime):
    path = Path(path_text)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    raw = path.read_bytes()
    if metadata.get('model_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('Model checksum mismatch')
    if (metadata.get('model_version') != MODEL_VERSION or metadata.get('feature_version') != FEATURE_VERSION
            or metadata.get('mode') != 'shadow_only' or metadata.get('authoritative') is not False
            or metadata.get('training_target') != 'actual_top3_equal_runner_weight'
            or metadata.get('ranking_policy') != 'direct_top5'
            or metadata.get('feature_names') != list(FEATURE_NAMES)
            or metadata.get('feature_schema_sha256') != json_hash(list(FEATURE_NAMES))
            or metadata.get('source_hashes') != source_hashes()
            or metadata.get('sklearn_version') != sklearn.__version__):
        raise ValueError('Model/feature/runtime contract mismatch')
    model = joblib.load(io.BytesIO(raw))
    if model.n_features_in_ != len(FEATURE_NAMES) or list(model.classes_) != [0, 1]:
        raise ValueError('Model shape/classes mismatch')
    return model, metadata


def load_bundle():
    path = MODEL_DIRECTORY/(MODEL_VERSION+'.joblib')
    return _load_bundle(str(path), path.stat().st_mtime_ns, path.with_suffix('.json').stat().st_mtime_ns)


def predict_scores(model, matrix):
    with threadpool_limits(limits=2):
        probabilities = np.asarray(model.predict_proba(matrix))
    if probabilities.shape != (len(matrix), 2):
        raise ValueError('Unexpected probability shape')
    values = probabilities[:, 1]
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid place scores')
    return values


def forward_timing(race, training_end, generated_at):
    reasons = []
    if race.race_info.race_date <= training_end:
        reasons.append('race_not_after_training_end')
    start = race.race_info.scheduled_start_at
    if not start:
        reasons.append('scheduled_start_unknown')
    else:
        try:
            planned = datetime.fromisoformat(start.replace('Z', '+00:00'))
            if planned.tzinfo is None:
                reasons.append('scheduled_start_timezone_unknown')
            elif planned.astimezone(timezone(timedelta(hours=9))).date().isoformat() != race.race_info.race_date:
                reasons.append('scheduled_start_date_mismatch')
            elif datetime.fromisoformat(generated_at) >= planned:
                reasons.append('generated_at_or_after_scheduled_start')
        except ValueError:
            reasons.append('scheduled_start_invalid')
    return {'eligible_at_generation': not reasons, 'reason_codes': reasons,
            'scheduled_start_at': start,
            'requires_received_before_start': True}


def evaluate_turf_shadow(race, prediction):
    if not is_central_turf(race):
        return None
    started = time.perf_counter()
    generated_at = datetime.now(timezone.utc).isoformat(timespec='microseconds')
    record = {'schema_version': 'central_turf_shadow_v1', 'mode': 'shadow', 'authoritative': False,
              'model_version': MODEL_VERSION, 'feature_version': FEATURE_VERSION,
              'race_id': race.race_info.race_id, 'started_at': generated_at,
              'status': 'skipped', 'reason_code': '',
              'target': 'top5_actual_top3_seat_coverage',
              'baseline_ranking': [s.horse_no for s in sorted(prediction.horse_scores, key=lambda s: (-s.total_score, s.horse_no))],
              'baseline_marks': dict(prediction.marks),
              'baseline_model_version': prediction.scope_reranker_result.model_version,
              'baseline_prediction_sha256': json_hash(prediction.model_dump())}
    record['baseline_top5'] = record['baseline_ranking'][:5]
    mode = os.getenv('KEIBA_CENTRAL_TURF_SHADOW_MODE', 'shadow').strip().lower()
    if mode == 'off':
        record.update(status='disabled', reason_code='disabled')
        return record
    if mode != 'shadow':
        record.update(status='failed', reason_code='invalid_mode')
        return record
    try:
        if (prediction.scoring_profile != 'accuracy_default' or prediction.borderline_recovery_enabled
                or prediction.scoring_config.model_dump() != default_scoring_config(race)):
            record['reason_code'] = 'nondefault_scoring_config'
            return record
        if prediction.race_id != race.race_info.race_id:
            raise ValueError('Race identity mismatch')
        target = date.fromisoformat(race.race_info.race_date)
        for horse in race.horses:
            for run in horse.recent_runs:
                if not run.date or run.race_id == race.race_info.race_id or date.fromisoformat(run.date) >= target:
                    record['reason_code'] = 'invalid_history_timing'
                    return record
        # Work on detached inputs; never write back to Prediction or its score objects.
        clean = race.model_copy(deep=True)
        for horse in clean.horses:
            horse.odds = horse.popularity = None
            for run in horse.recent_runs:
                run.odds = run.popularity = None
        model, metadata = load_bundle()
        matrix, numbers = build_turf_history_matrix(clean, prediction.horse_scores)
        values = predict_scores(model, matrix)
        ranking = sorted(numbers, key=lambda n: (-float(values[numbers.index(n)]), n))
        features = [[None if np.isnan(v) else float(v) for v in row] for row in matrix]
        coverage = Counter(h.history_status or 'unknown' for h in clean.horses)
        generated_at = datetime.now(timezone.utc).isoformat(timespec='microseconds')
        record.update(status='predicted', reason_code='shadow_only', model_sha256=metadata['model_sha256'],
                      generated_at=generated_at,
                      training_end=metadata['training_end'], training_races=metadata['training_races'],
                      feature_schema_sha256=metadata['feature_schema_sha256'],
                      source_hashes=metadata['source_hashes'], input_sha256=json_hash(clean.model_dump()),
                      horse_numbers=numbers, features=features, features_sha256=json_hash(features),
                      missing_feature_values=int(np.isnan(matrix).sum()), history_status_counts=dict(coverage),
                      candidate_ranking=ranking, candidate_top5=ranking[:5],
                      candidate_scores={str(n): float(v) for n, v in zip(numbers, values, strict=True)},
                      score_semantics='uncalibrated_top3_classifier_signal',
                      timing=forward_timing(clean, metadata['training_end'], generated_at))
    except Exception as exc:
        record.update(status='failed', reason_code='shadow_evaluation_failed', error_type=type(exc).__name__)
    finally:
        record['elapsed_ms'] = round((time.perf_counter()-started)*1000, 3)
    return record

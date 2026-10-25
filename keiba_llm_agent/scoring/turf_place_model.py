"""Authoritative central-turf Top3 coverage model and inference provenance."""
from __future__ import annotations

from collections import Counter
from datetime import date
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from keiba_llm_agent.analysis.race_class import CENTRAL_COURSES
from keiba_llm_agent.scoring.turf_history_features import FEATURE_NAMES, FEATURE_VERSION, build_turf_history_matrix

ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = 'central_turf_history_place_v1_20261025'
MODEL_DIRECTORY = ROOT/'keiba_llm_agent/models/scope_reranker'
SOURCE_FILES = (
    'keiba_llm_agent/scoring/turf_history_features.py',
    'keiba_llm_agent/scoring/turf_place_features.py',
    'keiba_llm_agent/scoring/turf_place_model.py',
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
    return {name: hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for name in SOURCE_FILES}


def is_central_turf(race):
    info = race.race_info
    return (str(info.source or '').lower() != 'local'
            and not str(info.scope_key or '').lower().startswith('local')
            and info.surface == '芝' and info.course in CENTRAL_COURSES)


def validate_inputs(race):
    if not is_central_turf(race):
        raise ValueError('Active turf model requires a known central turf course')
    target = date.fromisoformat(race.race_info.race_date)
    for horse in race.horses:
        for run in horse.recent_runs:
            if not run.date or run.race_id == race.race_info.race_id or date.fromisoformat(run.date) >= target:
                raise ValueError('Invalid historical run timing for active turf model')


@lru_cache(maxsize=2)
def _load_bundle(path_text, model_mtime, metadata_mtime):
    path = Path(path_text)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    raw = path.read_bytes()
    if metadata.get('model_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('Active turf model checksum mismatch')
    if (metadata.get('model_version') != MODEL_VERSION or metadata.get('feature_version') != FEATURE_VERSION
            or metadata.get('mode') != 'active' or metadata.get('authoritative') is not True
            or metadata.get('training_target') != 'actual_top3_equal_runner_weight'
            or metadata.get('ranking_policy') != 'direct_top5'
            or metadata.get('feature_names') != list(FEATURE_NAMES)
            or metadata.get('feature_schema_sha256') != json_hash(list(FEATURE_NAMES))
            or metadata.get('source_hashes') != source_hashes()
            or metadata.get('sklearn_version') != sklearn.__version__):
        raise ValueError('Active turf model/feature/runtime contract mismatch')
    model = joblib.load(io.BytesIO(raw))
    if model.n_features_in_ != len(FEATURE_NAMES) or list(model.classes_) != [0, 1]:
        raise ValueError('Active turf model shape/classes mismatch')
    return model, metadata


def load_bundle(path=None):
    path = Path(path) if path else MODEL_DIRECTORY/(MODEL_VERSION+'.joblib')
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


def build_inference_audit(race, prediction):
    result = prediction.scope_reranker_result
    if not result.applied or result.model_version != MODEL_VERSION:
        return None
    # Reconstruct the input contract only; do not run either model a second time.
    from keiba_llm_agent.scoring.scope_reranker import _model_directory

    _, metadata = load_bundle(_model_directory()/(MODEL_VERSION+'.joblib'))
    matrix, numbers = build_turf_history_matrix(race, prediction.horse_scores)
    features = [[None if np.isnan(v) else float(v) for v in row] for row in matrix]
    return {'schema_version': 'central_turf_active_v1', 'mode': 'active', 'authoritative': True,
            'status': 'predicted', 'model_version': MODEL_VERSION, 'model_sha256': metadata['model_sha256'],
            'feature_version': FEATURE_VERSION, 'feature_schema_sha256': metadata['feature_schema_sha256'],
            'source_hashes': metadata['source_hashes'], 'training_end': metadata['training_end'],
            'training_races': metadata['training_races'], 'horse_numbers': numbers,
            'features': features, 'features_sha256': json_hash(features),
            'missing_feature_values': int(np.isnan(matrix).sum()),
            'history_status_counts': dict(Counter(h.history_status or 'unknown' for h in race.horses)),
            'ranking': list(result.reranked_ranking), 'top5': list(result.reranked_ranking[:5]),
            'model_scores': {str(n): v for n, v in result.candidate_model_scores.items()},
            'score_semantics': 'uncalibrated_top3_classifier_signal'}

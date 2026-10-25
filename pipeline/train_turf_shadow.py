"""Fit the frozen, all-full turf challenger and verify production feature parity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import experiment_turf_supplements as experiment
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.scoring.turf_history_features import FEATURE_NAMES, FEATURE_VERSION, build_turf_history_matrix
from keiba_llm_agent.scoring import turf_shadow as shadow

e = experiment.e


def train(prepared, supplements, output):
    if output.exists() or not output.is_relative_to(ROOT/'analysis_outputs'):
        raise ValueError('Use a fresh directory inside analysis_outputs')
    data = joblib.load(prepared)
    rows = data['rows']; manifest = data['manifest']
    if e.production_hashes() != manifest['production_hashes']:
        raise ValueError('Baseline production code/model changed since validation')
    for path, expected in manifest['source_hashes'].items():
        if e.digest(Path(path)) != expected:
            raise ValueError(f'Frozen experiment source changed: {path}')
    supplement_path = supplements/'supplement_manifest.json'
    if e.digest(supplement_path) != manifest['supplement_manifest_sha256']:
        raise ValueError('Supplement provenance changed')
    enriched = {r['race_id']: r for r in e.read_json(supplement_path)['races']}
    sources = {r['race_id']: r for r in manifest['source_files']}
    if len({r['race_id'] for r in rows}) != len(rows) or set(enriched) != set(sources):
        raise ValueError('Training cohort identity mismatch')
    matrices = []; replay = []
    for index, row in enumerate(rows):
        if row['scope'] != 'central_turf':
            raise ValueError('Only central turf is supported')
        source = sources[row['race_id']]
        for key in ('audit', 'prediction', 'result'):
            if e.digest(Path(source[key])) != source[key+'_sha256']:
                raise ValueError('Frozen source changed')
        entry = enriched[row['race_id']]
        if e.digest(Path(entry['enriched'])) != entry['enriched_sha256']:
            raise ValueError('Enriched source changed')
        race = experiment.sanitize(RaceData.model_validate(e.read_json(Path(entry['enriched']))))
        saved = Prediction.model_validate(e.read_json(Path(source['audit']))['outputs']['prediction'])
        prediction = e.refreshed_prediction(race, saved, 'current')
        matrix, numbers = build_turf_history_matrix(race, prediction.horse_scores)
        if numbers != row['numbers']:
            raise ValueError('Runner ordering mismatch')
        np.testing.assert_allclose(matrix, row['experiment_matrices']['history_direct'], rtol=0, atol=0, equal_nan=True)
        public_scores, config, result = e.reranker.apply_scope_reranker(race, prediction.horse_scores, enabled=True)
        public = prediction.model_copy(update={'horse_scores': public_scores, 'scope_reranker_config': config,
                                               'scope_reranker_result': result, 'marks': e.scorer.build_marks(public_scores)})
        after, _ = build_turf_history_matrix(race, public.horse_scores)
        np.testing.assert_allclose(after, matrix, rtol=0, atol=0, equal_nan=True)
        matrices.append(matrix); replay.append((race, public))
        if (index+1) % 100 == 0:
            print(f'Runtime feature/official ranking isolation parity: {index+1}', flush=True)
    x = np.concatenate(matrices)
    y, weights = experiment.a.labels_weights(rows)
    model = clone(e.templates('central_turf')[0])
    model.fit(x, y, sample_weight=weights)
    output.mkdir(parents=True)
    model_path = output/(shadow.MODEL_VERSION+'.joblib')
    with model_path.open('xb') as handle:
        joblib.dump(model, handle, compress=3)
    metadata = {
        'model_version': shadow.MODEL_VERSION, 'mode': 'shadow_only', 'authoritative': False,
        'feature_version': FEATURE_VERSION, 'feature_names': list(FEATURE_NAMES),
        'feature_schema_sha256': shadow.json_hash(list(FEATURE_NAMES)),
        'source_hashes': shadow.source_hashes(), 'model_sha256': e.digest(model_path),
        'sklearn_version': e.sklearn.__version__, 'numpy_version': np.__version__,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'training_start': min(r['date'] for r in rows), 'training_end': max(r['date'] for r in rows),
        'training_races': len(rows), 'training_runner_observations': len(x),
        'training_race_ids': sorted(r['race_id'] for r in rows),
        'training_target': 'actual_top3_equal_runner_weight', 'ranking_policy': 'direct_top5',
        'loss_weight_scale': float(weights[0]), 'estimator_parameters': model.get_params(),
        'prepared_sha256': e.digest(prepared), 'supplement_manifest_sha256': e.digest(supplement_path),
        'evidence_results_sha256': e.digest(prepared.parent/'results.json'),
        'validation': '630-race repeated date OOF plus chronological evaluation; release fit is not a new backtest',
        'data_caveat': 'Historical supplementary fields were retrospectively acquired; collect genuine pre-off shadow records next.',
    }
    with model_path.with_suffix('.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    loaded, checked = shadow._load_bundle(str(model_path), model_path.stat().st_mtime_ns,
                                         model_path.with_suffix('.json').stat().st_mtime_ns)
    from unittest.mock import patch
    total_ms = 0.0
    for (race, prediction), matrix in zip(replay, matrices, strict=True):
        before = prediction.model_dump(); inputs_before = race.model_dump()
        with patch.object(shadow, 'load_bundle', return_value=(loaded, checked)), \
             patch.dict(shadow.os.environ, {'KEIBA_CENTRAL_TURF_SHADOW_MODE': 'shadow'}):
            record = shadow.evaluate_turf_shadow(race, prediction)
        if record['status'] != 'predicted':
            raise ValueError(f'Shadow runtime rejected {race.race_info.race_id}: {record}')
        expected = shadow.predict_scores(model, matrix)
        np.testing.assert_array_equal(expected, [record['candidate_scores'][str(n)] for n in record['horse_numbers']])
        if prediction.model_dump() != before or race.model_dump() != inputs_before:
            raise ValueError('Authoritative prediction or input mutated')
        if record['timing']['eligible_at_generation']:
            raise ValueError('Training race incorrectly marked as forward test')
        total_ms += record['elapsed_ms']
    report = {'races_verified': len(rows), 'features': len(FEATURE_NAMES), 'runner_observations': len(x),
              'exact_frozen_feature_parity': True, 'post_official_rerank_feature_parity': True,
              'serialized_model_prediction_parity': True, 'official_prediction_and_inputs_unchanged': True,
              'mean_shadow_elapsed_ms': total_ms/len(rows), 'model_sha256': metadata['model_sha256'],
              'note': 'Compatibility checks on training inputs, not an accuracy estimate or prospective validation.'}
    e.save_json(output/'runtime_verification.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--supplements', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        train(args.prepared.resolve(), args.supplements.resolve(), args.output.resolve())

"""Verify release parity, without changing source audits or historical predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def other_scope_outputs():
    from keiba_llm_agent.schemas.race_data import RaceData, RaceInfo, HorseEntry, RecentRun
    from keiba_llm_agent.schemas.prediction import ScoringConfigSnapshot
    from keiba_llm_agent.config.scoring_config import resolve_scoring_profile_config
    from keiba_llm_agent.scoring import recent_run_scorer as scorer

    outputs = {}
    for scope, source, course in [('central_dirt', 'central', '東京'), ('local', 'local', '大井')]:
        for count in (4, 8, 12, 13, 16, 18):
            race = RaceData(race_info=RaceInfo(race_id=f'{scope}-{count}', race_date='2026-11-01',
                course=course, surface='ダート', distance=1600, source=source, scope_key=scope,
                track_condition='良'), horses=[HorseEntry(horse_no=i, horse_name=f'horse{i}',
                    carried_weight=54+i%4, recent_runs=[RecentRun(race_id=f'past-{i}', date='2026-10-01',
                        course=course, surface='ダート', distance=1600, finish=1+i%7, field_size=12,
                        final_3f=35+i*.2, corner_positions=[i%8+1, i%5+1], margin=str(i*.2))])
                    for i in range(1, count+1)])
            profile, _ = resolve_scoring_profile_config(scoring_profile='accuracy_default')
            with patch.object(scorer, 'build_pedigree_analyses_for_race', return_value=[]):
                prediction = scorer.build_prediction_from_recent_runs_with_scoring_config(
                    race, [], profile.scoring_profile,
                    ScoringConfigSnapshot.model_validate(profile.scoring_config.model_dump()),
                    scope_reranker_enabled=True)
            outputs[race.race_info.race_id] = prediction.model_dump()
    return outputs


def promote(source):
    from keiba_llm_agent.scoring import turf_place_model as active

    raw = source.read_bytes()
    metadata = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    if metadata['model_sha256'] != digest(source) or metadata['mode'] != 'shadow_only':
        raise ValueError('Invalid frozen challenger artifact')
    destination = active.MODEL_DIRECTORY/(active.MODEL_VERSION+'.joblib')
    metadata.update(model_version=active.MODEL_VERSION, mode='active', authoritative=True,
                    source_hashes=active.source_hashes(),
                    promoted_from={'model_version': metadata['model_version'],
                                   'model_sha256': metadata['model_sha256'],
                                   'metadata_sha256': digest(source.with_suffix('.json'))},
                    activation_policy='central_turf_only; direct ranking; no legacy model inference',
                    data_caveat='Supplementary fields were retrospectively acquired; live accuracy remains unproven.')
    with destination.open('xb') as handle:
        handle.write(raw)
    write_new(destination.with_suffix('.json'), metadata)
    active.load_bundle()


def verify(prepared, supplements, baseline, output):
    from pipeline import experiment_turf_supplements as experiment
    from keiba_llm_agent.schemas.race_data import RaceData
    from keiba_llm_agent.schemas.prediction import Prediction, ScoringConfigSnapshot
    from keiba_llm_agent.config.scoring_config import resolve_scoring_profile_config
    from keiba_llm_agent.scoring import turf_place_model as active
    from keiba_llm_agent.scoring import turf_shadow as shadow
    from keiba_llm_agent.scoring.turf_history_features import build_turf_history_matrix

    e = experiment.e
    data = joblib.load(prepared)
    rows, manifest = data['rows'], data['manifest']
    if digest(supplements) != manifest['supplement_manifest_sha256']:
        raise ValueError('Supplement manifest changed')
    enriched = {r['race_id']: r for r in e.read_json(supplements)['races']}
    sources = {r['race_id']: r for r in manifest['source_files']}
    row_ids = {r['race_id'] for r in rows}
    if len(rows) != len(row_ids) or set(sources) != row_ids or set(enriched) != row_ids:
        raise ValueError('Cohort mismatch')
    model, metadata = active.load_bundle()
    if (digest(prepared) != metadata['prepared_sha256']
            or digest(supplements) != metadata['supplement_manifest_sha256']
            or row_ids != set(metadata['training_race_ids'])):
        raise ValueError('Release training provenance mismatch')
    protected = {}
    runner_count = 0
    with patch.object(e.reranker, '_load_model', side_effect=AssertionError('Legacy model called')), \
         patch.object(shadow, 'evaluate_turf_shadow', side_effect=AssertionError('Shadow called')):
        for index, row in enumerate(rows):
            if row['scope'] != 'central_turf':
                raise ValueError('Unexpected scope')
            source = sources[row['race_id']]
            for key in ('audit', 'prediction', 'result'):
                protected[source[key]] = source[key+'_sha256']
            entry = enriched[row['race_id']]
            protected[entry['enriched']] = entry['enriched_sha256']
            for path in [source[k] for k in ('audit', 'prediction', 'result')] + [entry['enriched']]:
                if digest(path) != protected[path]:
                    raise ValueError(f'Source changed: {path}')
            race = experiment.sanitize(RaceData.model_validate(e.read_json(Path(entry['enriched']))))
            saved = Prediction.model_validate(e.read_json(Path(source['audit']))['outputs']['prediction'])
            profile, _ = resolve_scoring_profile_config(scoring_profile='accuracy_default')
            before = race.model_dump()
            with e.variant_context('current', race, saved):
                prediction = e.scorer.build_prediction_from_recent_runs_with_scoring_config(
                    race, saved.used_lessons, profile.scoring_profile,
                    ScoringConfigSnapshot.model_validate(profile.scoring_config.model_dump()),
                    scope_reranker_enabled=True)
            matrix, numbers = build_turf_history_matrix(race, prediction.horse_scores)
            np.testing.assert_allclose(matrix, row['experiment_matrices']['history_direct'], rtol=0, atol=0,
                                       equal_nan=True)
            assert numbers == row['numbers']
            values = active.predict_scores(model, matrix)
            expected = sorted(numbers, key=lambda n: (-float(values[numbers.index(n)]), n))
            assert prediction.scope_reranker_result.reranked_ranking == expected
            assert [s.horse_no for s in prediction.horse_scores] == expected
            assert list(prediction.marks.values()) == expected[:5]
            assert prediction.scope_reranker_result.model_version == active.MODEL_VERSION
            assert all(set(b.horse_numbers).issubset(expected[:5]) for b in prediction.bets)
            audit = active.build_inference_audit(race, prediction)
            assert audit['ranking'] == expected and audit['status'] == 'predicted'
            assert race.model_dump() == before
            runner_count += len(numbers)
            if (index+1) % 100 == 0:
                print(f'Exact formal ranking/marks/bets parity: {index+1}/{len(rows)}', flush=True)
    assert json.loads(json.dumps(other_scope_outputs())) == json.loads(baseline.read_text(encoding='utf-8'))
    for path, expected_hash in protected.items():
        if digest(path) != expected_hash:
            raise ValueError(f'Source modified during replay: {path}')
    report = {'races_verified': len(rows), 'runner_observations': runner_count, 'features': 292,
              'model_sha256': metadata['model_sha256'], 'model_version': active.MODEL_VERSION,
              'exact_frozen_feature_parity': True, 'formal_ranking_and_marks_parity': True,
              'bets_use_new_top5_only': True, 'old_turf_models_never_loaded': True,
              'other_scope_full_prediction_golden_cases': 12, 'protected_source_files': len(protected),
              'sources_unchanged': True,
              'note': 'Deployment parity checks on training inputs, not a new accuracy backtest.'}
    write_new(output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-baseline', type=Path)
    parser.add_argument('--promote', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--supplements', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.capture_baseline:
            write_new(args.capture_baseline, other_scope_outputs())
        elif args.promote:
            promote(args.promote)
        else:
            verify(args.prepared, args.supplements, args.baseline, args.output)

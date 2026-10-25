from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np

from keiba_llm_agent import main
from keiba_llm_agent.agents import race_analysis_agent as agent_module
from keiba_llm_agent.backtest.audit_snapshot import read_backtest_audit_payload, attach_result_to_backtest_audit
from keiba_llm_agent.scoring import turf_place_model as active, turf_shadow, scope_reranker, recent_run_scorer
from tests.test_turf_shadow import sample, FixedModel


def meta():
    return {'model_version': active.MODEL_VERSION, 'feature_version': active.FEATURE_VERSION,
            'mode': 'active', 'authoritative': True, 'training_target': 'actual_top3_equal_runner_weight',
            'ranking_policy': 'direct_top5', 'feature_names': list(active.FEATURE_NAMES),
            'feature_schema_sha256': active.json_hash(list(active.FEATURE_NAMES)),
            'source_hashes': active.source_hashes(), 'sklearn_version': active.sklearn.__version__,
            'model_sha256': 'test', 'training_end': '2026-10-25', 'training_races': 630}


def generate(race, prediction):
    with patch.object(recent_run_scorer, 'build_pedigree_analyses_for_race', return_value=[]):
        return recent_run_scorer.build_prediction_from_recent_runs_with_scoring_config(
            race, [], prediction.scoring_profile, prediction.scoring_config, scope_reranker_enabled=True)


class TurfActivationTests(unittest.TestCase):
    def test_packaged_active_model_is_same_frozen_candidate(self):
        model, metadata = active.load_bundle()
        self.assertEqual(model.n_features_in_, 292)
        self.assertEqual(metadata['training_races'], 630)
        self.assertEqual(metadata['training_end'], '2026-10-25')
        self.assertEqual(metadata['model_sha256'], '81c5e44ef12e1abfe1f5bf495bcbb91449604c3d5f23a72f5003f95a7adcb652')
        self.assertEqual(metadata['promoted_from']['model_sha256'], metadata['model_sha256'])
        self.assertEqual(scope_reranker.TURF_MODEL_VERSION, active.MODEL_VERSION)

    def test_new_model_controls_ranking_marks_bets_and_llm_inputs(self):
        race, initial = sample()
        llm = Mock()
        llm.enhance_prediction.return_value = {}
        with patch.object(active, 'load_bundle', return_value=(FixedModel(), meta())), \
             patch.object(scope_reranker, '_load_model', side_effect=AssertionError('old model')), \
             patch.object(turf_shadow, 'evaluate_turf_shadow', side_effect=AssertionError('shadow')), \
             patch.object(recent_run_scorer, 'build_pedigree_analyses_for_race', return_value=[]), \
             patch.object(agent_module, 'simulate_race', return_value=None) as simulate, \
             patch.dict(os.environ, {'KEIBA_SCOPE_RERANKER_ENABLED': '0', 'KEIBA_CENTRAL_TURF_SHADOW_MODE': 'shadow'}):
            prediction = agent_module.RaceAnalysisAgent(llm).run(race, [])
        self.assertEqual(prediction.scope_reranker_result.reranked_ranking, [6, 5, 4, 3, 2, 1])
        self.assertEqual(list(prediction.marks.values()), [6, 5, 4, 3, 2])
        self.assertTrue(prediction.bets)
        self.assertTrue(all(set(b.horse_numbers).issubset({6, 5, 4, 3, 2}) for b in prediction.bets))
        self.assertIs(llm.enhance_prediction.call_args.kwargs['prediction'], prediction)
        self.assertEqual(simulate.call_args.kwargs['marks'], prediction.marks)

    def test_no_history_still_uses_new_model_not_llm_ranking(self):
        race, _ = sample()
        for horse in race.horses:
            horse.recent_runs = []; horse.history_status = 'no_prior_runs'
        llm = Mock(); llm.enhance_prediction.return_value = {}
        with patch.object(active, 'load_bundle', return_value=(FixedModel(), meta())), \
             patch.object(recent_run_scorer, 'build_pedigree_analyses_for_race', return_value=[]), \
             patch.object(agent_module, 'simulate_race', return_value=None):
            prediction = agent_module.RaceAnalysisAgent(llm).run(race, [])
        self.assertEqual(prediction.scope_reranker_result.model_version, active.MODEL_VERSION)
        self.assertEqual(list(prediction.marks.values()), [6, 5, 4, 3, 2])
        llm.analyze_race.assert_not_called()

    def test_missing_or_corrupt_active_model_does_not_fall_back(self):
        race, initial = sample()
        for error in (FileNotFoundError('missing'), ValueError('checksum'), RuntimeError('prediction')):
            with patch.object(active, 'load_bundle', side_effect=error), \
                 patch.object(scope_reranker, '_load_model') as old:
                with self.assertRaises(type(error)):
                    generate(race, initial)
                old.assert_not_called()

    def test_invalid_history_market_and_roster_are_rejected(self):
        for when in (None, '2026-11-01', '2026-11-02'):
            race, initial = sample(); race.horses[0].recent_runs[0].date = when
            with self.assertRaises(ValueError):
                generate(race, initial)
        race, initial = sample(); initial.scoring_config.use_market_score_in_ranking = True
        with self.assertRaisesRegex(ValueError, 'market'):
            generate(race, initial)
        race, initial = sample(); race.horses[0].horse_no = 2
        with patch.object(active, 'load_bundle', return_value=(FixedModel(), meta())), self.assertRaises(ValueError):
            generate(race, initial)

    def test_nonfinite_probability_fails_and_ties_are_deterministic(self):
        race, initial = sample()
        with patch.object(active, 'load_bundle', return_value=(FixedModel(), meta())):
            for probabilities in (np.full((6, 2), np.nan), np.full((6, 2), 1.1), np.ones((6, 1))):
                with patch.object(FixedModel, 'predict_proba', return_value=probabilities), self.assertRaises(ValueError):
                    generate(race, initial)
            with patch.object(FixedModel, 'predict_proba', return_value=np.full((6, 2), .5)):
                self.assertEqual(list(generate(race, initial).marks.values()), [1, 2, 3, 4, 5])

    def test_audit_records_only_active_inference_without_running_model_again(self):
        race, initial = sample(); original = race.model_dump()
        with patch.object(active, 'load_bundle', return_value=(FixedModel(), meta())), \
             tempfile.TemporaryDirectory() as directory, patch.object(main, 'DEFAULT_BACKTEST_AUDIT_DIR', Path(directory)):
            prediction = generate(race, initial); before = prediction.model_dump()
            with patch.object(FixedModel, 'predict_proba', side_effect=AssertionError('duplicate prediction')), \
                 patch.object(turf_shadow, 'evaluate_turf_shadow', side_effect=AssertionError('shadow')):
                path = main._write_prediction_backtest_audit(race_data=race, prediction=prediction)
            payload = read_backtest_audit_payload(path)
            self.assertNotIn('shadow_predictions', payload)
            record = payload['model_inference']['central_turf_history']
            self.assertTrue(record['authoritative'])
            self.assertEqual(record['top5'], list(prediction.marks.values()))
            self.assertEqual(record['features_sha256'], active.json_hash(record['features']))
            attach_result_to_backtest_audit(path, {'race_id': race.race_info.race_id, 'result': {'1st': 6, '2nd': 2, '3rd': 1}})
            self.assertEqual(read_backtest_audit_payload(path)['model_inference'], payload['model_inference'])
            self.assertEqual(prediction.model_dump(), before)
        self.assertEqual(race.model_dump(), original)

    def test_other_scopes_never_load_active_turf(self):
        for scope, source, course in [('central_dirt', 'central', '東京'), ('local', 'local', '大井')]:
            race, initial = sample()
            race.race_info.scope_key = scope; race.race_info.source = source
            race.race_info.course = course; race.race_info.surface = 'ダート'
            with patch.object(active, 'load_bundle') as load:
                generate(race, initial)
                load.assert_not_called()

    def test_real_model_ranking_is_market_independent_after_full_scoring(self):
        race, initial = sample()
        before = generate(race, initial)
        for horse in race.horses:
            horse.odds = 1.1 if horse.horse_no % 2 else 999.0
            horse.popularity = 7-horse.horse_no
            for run in horse.recent_runs:
                run.odds = 999.0; run.popularity = 16
        after = generate(race, initial)
        self.assertEqual(before.scope_reranker_result.candidate_model_scores,
                         after.scope_reranker_result.candidate_model_scores)
        self.assertEqual(before.marks, after.marks)

    def test_audit_failure_cannot_mutate_formal_prediction_or_race(self):
        race, initial = sample()
        with patch.object(active, 'load_bundle', return_value=(FixedModel(), meta())):
            prediction = generate(race, initial)
        before = prediction.model_dump(); original = race.model_dump()
        def faulty(r, p):
            p.marks.clear(); r.horses.clear()
            raise RuntimeError('capture failure')
        with tempfile.TemporaryDirectory() as directory, patch.object(main, 'DEFAULT_BACKTEST_AUDIT_DIR', Path(directory)), \
             patch.object(active, 'build_inference_audit', side_effect=faulty):
            path = main._write_prediction_backtest_audit(race_data=race, prediction=prediction)
            payload = read_backtest_audit_payload(path)
            self.assertEqual(payload['model_inference']['central_turf_history']['status'], 'failed')
            self.assertEqual(payload['outputs']['prediction'], prediction.model_dump(mode='json'))
        self.assertEqual(prediction.model_dump(), before)
        self.assertEqual(race.model_dump(), original)

    def test_bundle_rejects_corruption_and_shadow_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.joblib'; joblib.dump(FixedModel(), path)
            metadata = meta(); metadata['model_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            for field, replacement in [(None, None), ('model_sha256', 'bad'), ('mode', 'shadow_only'),
                                       ('authoritative', False), ('source_hashes', {}), ('sklearn_version', '0.0'),
                                       ('feature_names', list(reversed(active.FEATURE_NAMES)))]:
                altered = deepcopy(metadata)
                if field:
                    altered[field] = replacement
                path.with_suffix('.json').write_text(json.dumps(altered), encoding='utf-8')
                active._load_bundle.cache_clear()
                if field:
                    with self.assertRaises(ValueError):
                        active._load_bundle(str(path), 0, 0)
                else:
                    self.assertIsInstance(active._load_bundle(str(path), 0, 0)[0], FixedModel)


if __name__ == '__main__':
    unittest.main()

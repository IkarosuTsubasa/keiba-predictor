from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from keiba_llm_agent import main
from keiba_llm_agent.backtest.audit_snapshot import read_backtest_audit_payload, attach_result_to_backtest_audit
from keiba_llm_agent.schemas.prediction import HorseScore, Prediction, ScoreBreakdown, TotalScoreBreakdown
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo, RecentRun
from keiba_llm_agent.scoring import turf_shadow as shadow
from keiba_llm_agent.scoring.turf_history_features import FEATURE_NAMES, FEATURE_VERSION, build_turf_history_matrix
from keiba_llm_agent.scoring.turf_place_features import build_supplementary_features
from keiba_llm_agent.scoring.scope_reranker import build_scope_reranker_feature_matrix
from pipeline import experiment_agent_architecture as oracle


class FixedModel:
    n_features_in_ = 292
    classes_ = np.array([0, 1])

    def predict_proba(self, matrix):
        values = np.linspace(.1, .8, len(matrix))
        return np.column_stack((1-values, values))


def sample():
    race = RaceData(race_info=RaceInfo(race_id='202605050101', race_date='2026-11-01',
        scheduled_start_at='2026-11-01T15:00:00+09:00', course='東京', surface='芝', distance=1600,
        source='central', scope_key='central_turf', track_condition='良', race_name='2勝クラス'),
        horses=[HorseEntry(horse_no=i, horse_name=f'horse{i}', horse_id=f'id{i}', carried_weight=55+i%3,
            frame_no=i, history_status='available', recent_runs=[RecentRun(race_id='202605040101',
                date='2026-10-01', course='東京', surface='芝', distance=1600, field_size=12, finish=i,
                race_name='1勝クラス', final_3f=34+i*.1, time_seconds=95+i*.2, carried_weight=55,
                body_weight=470+i, body_weight_change=-2, pace_first_3f=35, pace_last_3f=34.5,
                margin=str(i*.1), corner_positions=[i+1, i])]) for i in range(1, 7)])
    scores = [HorseScore(horse_no=h.horse_no, horse_name=h.horse_name,
                        scores=ScoreBreakdown(recent_form=5, distance_fit=5, course_fit=5,
                                              track_condition_fit=5, jockey_fit=5, odds_value=0, risk=0),
                        score_breakdown=TotalScoreBreakdown(total_score=30-h.horse_no),
                        total_score=30-h.horse_no, reason='sample') for h in race.horses]
    prediction = Prediction(race_id=race.race_info.race_id, race_info=race.race_info.model_copy(deep=True),
                            scoring_profile='accuracy_default', scoring_config=shadow.default_scoring_config(race),
                            horse_scores=scores, marks={'◎': 1, '○': 2, '▲': 3, '△': 4, '☆': 5}, summary='sample')
    return race, prediction


def metadata():
    return {'model_version': shadow.MODEL_VERSION, 'feature_version': FEATURE_VERSION,
            'mode': 'shadow_only', 'authoritative': False,
            'training_target': 'actual_top3_equal_runner_weight', 'ranking_policy': 'direct_top5',
            'feature_names': list(FEATURE_NAMES), 'feature_schema_sha256': shadow.json_hash(list(FEATURE_NAMES)),
            'source_hashes': shadow.source_hashes(), 'sklearn_version': shadow.sklearn.__version__,
            'model_sha256': 'test', 'training_end': '2026-10-25', 'training_races': 630}


class TurfShadowTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(shadow.os.environ, {'KEIBA_CENTRAL_TURF_SHADOW_MODE': 'shadow'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_historical_shadow_artifact_is_preserved_but_not_current_runtime(self):
        path = shadow.MODEL_DIRECTORY/(shadow.MODEL_VERSION+'.joblib')
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), meta['model_sha256'])
        model = joblib.load(path)
        self.assertEqual(model.n_features_in_, 292)
        self.assertEqual(meta['training_races'], 630)
        self.assertEqual(meta['training_end'], '2026-10-25')
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            shadow.load_bundle()

    def test_feature_contract_matches_independent_experiment(self):
        race, prediction = sample()
        matrix, numbers = build_turf_history_matrix(race, prediction.horse_scores)
        expected = np.column_stack((build_scope_reranker_feature_matrix(race, prediction.horse_scores, 'course_geometry')[0],
                                    oracle.raw_features(race),
                                    build_supplementary_features(race)['history']))
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(matrix.shape, (6, 292))
        self.assertEqual(len(set(FEATURE_NAMES)), 292)
        self.assertEqual(numbers, list(range(1, 7)))

    def test_shadow_keeps_official_prediction_and_inputs_untouched(self):
        race, prediction = sample(); before = prediction.model_dump(); original = race.model_dump()
        with patch.object(shadow, 'load_bundle', return_value=(FixedModel(), metadata())):
            record = shadow.evaluate_turf_shadow(race, prediction)
        self.assertEqual(record['status'], 'predicted')
        self.assertEqual(record['candidate_top5'], [6, 5, 4, 3, 2])
        self.assertEqual(record['baseline_top5'], [1, 2, 3, 4, 5])
        self.assertFalse(record['authoritative'])
        self.assertEqual(record['features_sha256'], shadow.json_hash(record['features']))
        self.assertEqual(prediction.model_dump(), before)
        self.assertEqual(race.model_dump(), original)
        json.dumps(record, allow_nan=False)

    def test_other_scopes_never_load_shadow_model(self):
        for scope, surface, source, course in [('local', 'ダート', 'local', '大井'),
                                              ('central_dirt', 'ダート', 'central', '東京'),
                                              ('local_turf', '芝', 'local', '東京')]:
            race, prediction = sample()
            race.race_info.scope_key = scope; race.race_info.surface = surface
            race.race_info.source = source; race.race_info.course = course
            with patch.object(shadow, 'load_bundle') as load:
                self.assertIsNone(shadow.evaluate_turf_shadow(race, prediction))
            load.assert_not_called()

    def test_off_and_invalid_mode_cannot_activate_challenger(self):
        for mode, expected in [('off', 'disabled'), ('active', 'failed')]:
            race, prediction = sample()
            with patch.dict(shadow.os.environ, {'KEIBA_CENTRAL_TURF_SHADOW_MODE': mode}), \
                 patch.object(shadow, 'load_bundle') as load:
                self.assertEqual(shadow.evaluate_turf_shadow(race, prediction)['status'], expected)
            load.assert_not_called()

    def test_custom_and_market_configs_are_not_shadow_training_contract(self):
        for field, value in [('pedigree_weight', .99), ('use_market_score_in_ranking', True)]:
            race, prediction = sample()
            setattr(prediction.scoring_config, field, value)
            with patch.object(shadow, 'load_bundle') as load:
                record = shadow.evaluate_turf_shadow(race, prediction)
            self.assertEqual(record['reason_code'], 'nondefault_scoring_config')
            load.assert_not_called()

    def test_future_same_day_and_missing_history_dates_are_not_evaluated(self):
        for when in ('2026-11-01', '2026-11-02', None):
            race, prediction = sample(); race.horses[0].recent_runs[0].date = when
            with patch.object(shadow, 'load_bundle') as load:
                record = shadow.evaluate_turf_shadow(race, prediction)
            self.assertEqual(record['reason_code'], 'invalid_history_timing')
            load.assert_not_called()

    def test_no_history_is_supported_and_not_mislabelled_fetch_failure(self):
        race, prediction = sample()
        for h in race.horses:
            h.recent_runs = []; h.history_status = 'no_prior_runs'
        with patch.object(shadow, 'load_bundle', return_value=(FixedModel(), metadata())):
            record = shadow.evaluate_turf_shadow(race, prediction)
        self.assertEqual(record['status'], 'predicted')
        self.assertEqual(record['history_status_counts'], {'no_prior_runs': 6})

    def test_market_and_current_body_weight_cannot_change_features(self):
        race, prediction = sample()
        before, _ = build_turf_history_matrix(race, prediction.horse_scores)
        for h in race.horses:
            h.odds = 1.1; h.popularity = 1; h.body_weight = 800; h.body_weight_change = 50
            h.horse_id = 'changed'; h.horse_name = 'changed'
            for run in h.recent_runs:
                run.odds = 999; run.popularity = 99
        after, _ = build_turf_history_matrix(race, prediction.horse_scores)
        np.testing.assert_allclose(before, after, rtol=0, atol=0, equal_nan=True)

    def test_missing_corrupt_or_failed_model_is_non_authoritative(self):
        for error in (FileNotFoundError('missing'), ValueError('bad checksum'), RuntimeError('failed')):
            race, prediction = sample(); before = prediction.model_dump()
            with patch.object(shadow, 'load_bundle', side_effect=error):
                record = shadow.evaluate_turf_shadow(race, prediction)
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(prediction.model_dump(), before)

    def test_duplicate_roster_and_nonfinite_scores_fail_closed(self):
        race, prediction = sample(); race.horses[0].horse_no = 2
        with patch.object(shadow, 'load_bundle', return_value=(FixedModel(), metadata())):
            self.assertEqual(shadow.evaluate_turf_shadow(race, prediction)['status'], 'failed')
        race, prediction = sample()
        with patch.object(FixedModel, 'predict_proba', return_value=np.full((6, 2), np.nan)), \
             patch.object(shadow, 'load_bundle', return_value=(FixedModel(), metadata())):
            self.assertEqual(shadow.evaluate_turf_shadow(race, prediction)['status'], 'failed')

    def test_timing_excludes_training_period_late_and_unknown_off_time(self):
        race, _ = sample()
        self.assertTrue(shadow.forward_timing(race, '2026-10-25', '2026-11-01T05:00:00+00:00')['eligible_at_generation'])
        self.assertFalse(shadow.forward_timing(race, '2026-11-01', '2026-11-01T05:00:00+00:00')['eligible_at_generation'])
        self.assertFalse(shadow.forward_timing(race, '2026-10-25', '2026-11-01T06:00:00+00:00')['eligible_at_generation'])
        race.race_info.scheduled_start_at = '2026-11-02T15:00:00+09:00'
        self.assertFalse(shadow.forward_timing(race, '2026-10-25', '2026-11-01T05:00:00+00:00')['eligible_at_generation'])
        race.race_info.scheduled_start_at = None
        self.assertFalse(shadow.forward_timing(race, '2026-10-25', '2026-11-01T05:00:00+00:00')['eligible_at_generation'])

    def test_bundle_hash_feature_order_and_source_contract_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.joblib'
            joblib.dump(FixedModel(), path)
            meta = metadata(); meta['model_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            for field, replacement in [(None, None), ('model_sha256', 'bad'),
                                       ('feature_names', list(reversed(FEATURE_NAMES))), ('source_hashes', {}),
                                       ('sklearn_version', '0.0'), ('ranking_policy', 'blend'), ('authoritative', True)]:
                altered = deepcopy(meta)
                if field:
                    altered[field] = replacement
                path.with_suffix('.json').write_text(json.dumps(altered), encoding='utf-8')
                shadow._load_bundle.cache_clear()
                if field:
                    with self.assertRaises(ValueError):
                        shadow._load_bundle(str(path), 0, 0)
                else:
                    self.assertIsInstance(shadow._load_bundle(str(path), 0, 0)[0], FixedModel)

    def test_main_no_longer_runs_shadow_and_keeps_prediction_untouched(self):
        race, prediction = sample(); before = prediction.model_dump()
        with tempfile.TemporaryDirectory() as directory, patch.object(main, 'DEFAULT_BACKTEST_AUDIT_DIR', Path(directory)), \
             patch.object(shadow, 'evaluate_turf_shadow') as evaluate:
            path = main._write_prediction_backtest_audit(race_data=race, prediction=prediction)
            payload = read_backtest_audit_payload(path)
            self.assertEqual(payload['outputs']['prediction'], before)
            self.assertNotIn('shadow_predictions', payload['outputs']['prediction'])
            self.assertNotIn('shadow_predictions', payload)
            evaluate.assert_not_called()
            attach_result_to_backtest_audit(path, {'race_id': race.race_info.race_id, 'result': {'1st': 6, '2nd': 2, '3rd': 1}})
            self.assertEqual(read_backtest_audit_payload(path)['outputs']['prediction'], before)
        self.assertEqual(prediction.model_dump(), before)

    def test_shadow_capture_is_not_called_even_when_it_would_fail(self):
        race, prediction = sample(); before = prediction.model_dump(); original = race.model_dump()
        def faulty(r, p):
            p.marks.clear(); r.horses.clear()
            raise RuntimeError('failure after mutation')
        with tempfile.TemporaryDirectory() as directory, patch.object(main, 'DEFAULT_BACKTEST_AUDIT_DIR', Path(directory)), \
             patch.object(shadow, 'evaluate_turf_shadow', side_effect=faulty) as evaluate:
            path = main._write_prediction_backtest_audit(race_data=race, prediction=prediction)
            payload = read_backtest_audit_payload(path)
            self.assertNotIn('shadow_predictions', payload)
            evaluate.assert_not_called()
            self.assertEqual(payload['outputs']['prediction'], before)
        self.assertEqual(prediction.model_dump(), before)
        self.assertEqual(race.model_dump(), original)


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from pipeline import experiment_central_evidence as exp
from keiba_llm_agent.schemas.prediction import HorseScore, ScoreBreakdown, TotalScoreBreakdown
from keiba_llm_agent.schemas.race_data import RaceData, RaceInfo, HorseEntry, RecentRun


class SignalModel:
    def __init__(self, values, width):
        self.values = np.asarray(values)
        self.n_features_in_ = width

    def predict_proba(self, matrix):
        return np.column_stack((1-self.values,self.values))


class CentralEvidenceTests(unittest.TestCase):
    def test_local_scope_excluded_even_when_source_missing(self):
        self.assertEqual(exp.scope_of({'scope_key':'local','surface':'\u30c0\u30fc\u30c8'}),'local')

    def info(self, surface='\u829d'):
        return RaceInfo(race_id='202605040211', race_date='2026-10-04', course='\u6771\u4eac',
                        surface=surface, distance=1800, track_condition='\u826f', source='central')

    def run_row(self, *, surface='\u829d', distance=1800, finish=1, rid='202505040211'):
        return RecentRun(race_id=rid,date='2025-10-05',course='\u6771\u4eac',surface=surface,
                         distance=distance,track_condition='\u826f',finish=finish,field_size=16,margin='0.1')

    def score(self, number, total):
        components = ScoreBreakdown(recent_form=5,distance_fit=5,course_fit=5,track_condition_fit=5,
                                    jockey_fit=5,odds_value=5,risk=-1)
        return HorseScore(horse_no=number,horse_name=str(number),scores=components,total_score=total,
                          reason='test',score_breakdown=TotalScoreBreakdown(total_score=total))

    def test_identical_conditions_keep_recent_quality(self):
        runs = [self.run_row(finish=n) for n in (2,5,8)]
        self.assertEqual(exp.context_form(runs,self.info()),exp.ORIGINAL_FORM(runs,self.info()))

    def test_cross_surface_failure_shrinks_to_neutral_not_positive(self):
        runs = [self.run_row(surface='\u30c0\u30fc\u30c8',finish=16) for _ in range(3)]
        current = exp.ORIGINAL_FORM(runs,self.info())
        changed = exp.context_form(runs,self.info())
        self.assertGreater(changed,current)
        self.assertLessEqual(changed,5)

    def test_same_conditions_bad_runs_keep_risk(self):
        runs = [self.run_row(finish=14),self.run_row(finish=15)]
        self.assertEqual(exp.context_risk(runs,1,5,self.info()),-5)

    def test_cross_surface_bad_runs_reduce_but_keep_negative_risk(self):
        runs = [self.run_row(surface='\u30c0\u30fc\u30c8',finish=14),
                self.run_row(surface='\u30c0\u30fc\u30c8',finish=15)]
        self.assertGreater(exp.context_risk(runs,1,5,self.info()),-5)
        self.assertLessEqual(exp.context_risk(runs,1,5,self.info()),-3)

    def test_unknown_distance_is_neutral_not_bonus(self):
        runs = [self.run_row(distance=1000)]
        self.assertEqual(exp.unknown_distance(runs,2400),5)
        self.assertEqual(exp.unknown_distance(runs,1000),exp.ORIGINAL_DISTANCE(runs,1000))

    def test_surface_guard_does_not_award_dirt_win_turf_course_credit(self):
        runs = [self.run_row(surface='\u30c0\u30fc\u30c8')]
        self.assertEqual(exp.surface_guard_course(runs,'\u6771\u4eac','\u829d'),1)
        self.assertEqual(exp.surface_guard_course(runs,'\u6771\u4eac','\u30c0\u30fc\u30c8'),10)

    def test_recent_short_sample_trend_kept_unchanged(self):
        runs = [self.run_row(finish=1),self.run_row(finish=12)]
        self.assertEqual(exp.context_trend(runs,self.info()),5)

    def test_evidence_finds_old_head_to_head_and_deduplicates(self):
        old = self.run_row(finish=2)
        horses = [HorseEntry(horse_no=1,horse_name='a',recent_runs=[old,old.model_copy()]),
                  HorseEntry(horse_no=2,horse_name='b',recent_runs=[self.run_row(finish=5)])]
        matrix = exp.evidence_matrix(RaceData(race_info=self.info(),horses=horses),[self.score(1,10),self.score(2,9)])
        index = exp.EVIDENCE_NAMES.index('head_finish_advantage')
        self.assertGreater(matrix[0,index],0)
        self.assertLess(matrix[1,index],0)
        self.assertEqual(matrix[0,exp.EVIDENCE_NAMES.index('relevant_effective_count')],1)

    def test_future_evidence_not_used(self):
        future = self.run_row(finish=1); future.date='2026-10-05'
        horse = HorseEntry(horse_no=1,horse_name='a',recent_runs=[future])
        matrix = exp.evidence_matrix(RaceData(race_info=self.info(),horses=[horse]),[self.score(1,10)])
        self.assertTrue(np.isnan(matrix[0,0]))
        self.assertEqual(matrix[0,exp.EVIDENCE_NAMES.index('relevant_effective_count')],0)

    def test_market_tampering_does_not_change_evidence(self):
        race = RaceData(race_info=self.info(),horses=[HorseEntry(horse_no=1,horse_name='a',
                        odds=1.1,popularity=1,recent_runs=[self.run_row()])])
        before = exp.evidence_matrix(race,[self.score(1,10)])
        race.horses[0].odds=999; race.horses[0].popularity=99
        race.horses[0].recent_runs[0].odds=1.1; race.horses[0].recent_runs[0].popularity=1
        np.testing.assert_allclose(before,exp.evidence_matrix(race,[self.score(1,10)]),equal_nan=True)

    def test_date_splits_cover_all_once_without_date_overlap(self):
        rows = [{'date':f'2026-09-{day:02d}'} for day in range(1,13) for _ in range(2)]
        seen = np.zeros(len(rows))
        for train,test in exp.date_splits(rows,42):
            self.assertFalse({rows[i]['date'] for i in train}&{rows[i]['date'] for i in test})
            seen[test] += 1
        np.testing.assert_array_equal(seen,np.ones(len(rows)))

    def training_row(self, n):
        return {'numbers':list(range(1,n+1)), 'top3':[1,2,3], 'winner':1,
                'variants':{'current':{'matrix':np.zeros((n,2)), 'winner_matrix':np.zeros((n,2))}}}

    def test_turf_training_matches_current_equal_weights_and_explicit_winner_ablation(self):
        for winner, expected in ((False,np.full(9,12/9)),
                                 (True,[8,1,1,1,1,1,1,1,1])):
            model = Mock()
            with patch.object(exp,'clone',return_value=model):
                exp.fit_model([self.training_row(9)],'current','central_turf',object(),winner=winner)
            args,kwargs = model.fit.call_args
            np.testing.assert_array_equal(args[1],[1,0,0,0,0,0,0,0,0] if winner else [1,1,1,0,0,0,0,0,0])
            np.testing.assert_allclose(kwargs['sample_weight'],expected)

    def test_dirt_training_equal_weight_preserves_production_loss_scale(self):
        model = Mock()
        with patch.object(exp,'clone',return_value=model):
            exp.fit_model([self.training_row(9),self.training_row(13)],'current','central_dirt',object())
        args,kwargs = model.fit.call_args
        self.assertEqual(len(args[1]),22)
        np.testing.assert_allclose(kwargs['sample_weight'],np.full(22,32/22))

    def test_production_ranking_parity_and_dirt_gate(self):
        for scope,surface in (('central_turf','\u829d'),('central_dirt','\u30c0\u30fc\u30c8')):
            for n in (12,13,16):
                info = self.info(surface); info.scope_key=scope
                race = RaceData(race_info=info,horses=[HorseEntry(horse_no=i,horse_name=str(i)) for i in range(1,n+1)])
                scores = [self.score(i,60-i) for i in range(1,n+1)]
                spec = exp.reranker.MODEL_SPECS[scope]
                place = np.linspace(.1,.9,n)
                winner = np.roll(np.linspace(.1,.9,n),2) if scope=='central_turf' else None
                width = exp.reranker.build_scope_reranker_feature_matrix(race,scores,spec['feature_set'])[0].shape[1]
                def load(path):
                    return SignalModel(winner,13) if path.endswith('central_turf_v1.joblib') else SignalModel(place,width)
                with patch.object(exp.reranker,'_load_model',side_effect=load), \
                     patch('keiba_llm_agent.scoring.turf_place_model.load_bundle', return_value=(SignalModel(place,width), {})), \
                     patch.object(exp.reranker,'_load_dirt_model',return_value=SignalModel(place,width)), \
                     patch.dict('os.environ',{'KEIBA_CENTRAL_DIRT_PLACE_MODE':'active'}):
                    expected = exp.reranker.apply_scope_reranker(race,scores,enabled=True)[0]
                row = {'scope':scope,'numbers':list(range(1,n+1)),
                       'variants':{'current':{'raw':list(range(1,n+1))}}}
                self.assertEqual(exp.runtime_ranking(row,'current',place,winner),[s.horse_no for s in expected])

    def test_variant_monkeypatch_restored(self):
        from keiba_llm_agent.schemas.prediction import Prediction
        saved = Prediction(race_id='test',marks={k:0 for k in ('\u25ce','\u25cb','\u25b2','\u25b3','\u2606')},
                           horse_scores=[],summary='test')
        for variant in exp.CANDIDATES:
            with exp.variant_context(variant,RaceData(race_info=self.info(),horses=[]),saved):
                pass
            self.assertIs(exp.scorer.score_recent_form,exp.ORIGINAL_FORM)
            self.assertIs(exp.scorer.score_course_fit,exp.ORIGINAL_COURSE)

    def test_winner_margin_zero_and_symbolic_margin_not_mixed(self):
        winner = self.run_row(); winner.margin='-0.4'
        self.assertEqual(exp.comparable_margin(winner),0)
        loser = self.run_row(finish=2); loser.margin='\u30af\u30d3'
        self.assertIsNone(exp.comparable_margin(loser))


if __name__ == '__main__':
    unittest.main()

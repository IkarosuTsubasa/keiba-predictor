import unittest
import numpy as np
from tests.test_turf_shadow import sample
from keiba_llm_agent.scoring import turf_class_features as c, turf_history_features as h


class TurfClassFeatureTests(unittest.TestCase):
    def test_original_information_preserved(self):
        race,prediction=sample(); race.race_info.race_class='2勝クラス'
        base,numbers=h.build_turf_history_matrix(race,prediction.horse_scores)
        current,ordered=c.build_turf_class_matrix(race,prediction.horse_scores)
        np.testing.assert_array_equal(current[:,:292],base)
        self.assertEqual(numbers,ordered); self.assertEqual(current.shape,(6,316))

    def test_relative_class_and_competitive_history(self):
        race,_=sample(); race.race_info.race_class='G2'
        horse=race.horses[0]; horse.recent_runs[0].race_name='テスト(G1)'
        row=c.class_relationship_features(race)[0]
        self.assertEqual(row[0],8); self.assertEqual(row[1],9); self.assertEqual(row[2],-1)
        self.assertEqual(row[7],9); self.assertEqual(row[8],-1); self.assertEqual(row[9],1)
        self.assertEqual(row[10],1); self.assertEqual(row[11],0)

    def test_unknown_target_or_history_keeps_missingness(self):
        race,_=sample(); row=c.class_relationship_features(race)[0]
        self.assertTrue(np.isnan(row[0])); self.assertTrue(np.isnan(row[2])); self.assertTrue(np.isnan(row[9]))
        race.race_info.race_class='G1'; race.horses[0].recent_runs[0].race_name='名称のみ'
        row=c.class_relationship_features(race)[0]
        self.assertTrue(np.isnan(row[1])); self.assertEqual(row[3],0)

    def test_market_bodyweight_future_and_duplicate_history_independence(self):
        race,_=sample(); race.race_info.race_class='2勝クラス'
        before=c.class_relationship_features(race)
        for horse in race.horses:
            horse.odds=999; horse.popularity=99; horse.body_weight=999
            run=horse.recent_runs[0]; run.odds=999; run.popularity=99
            horse.recent_runs.extend([run.model_copy(),run.model_copy(update={'date':'2099-01-01','race_id':'future','race_name':'G1'})])
        np.testing.assert_array_equal(before,c.class_relationship_features(race))

    def test_other_scopes_rejected(self):
        for source,surface in [('local','芝'),('central','ダート')]:
            race,_=sample(); race.race_info.source=source; race.race_info.surface=surface
            with self.assertRaises(ValueError): c.class_relationship_features(race)


if __name__=='__main__': unittest.main()

import unittest

import numpy as np

from pipeline import experiment_agent_architecture as a
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo, RecentRun


class ArchitectureFeatureTests(unittest.TestCase):
    def race(self):
        info = RaceInfo(race_id='202605040911',race_date='2026-10-25',race_name='test',
                        course='Tokyo',distance=1600,surface='turf')
        run = RecentRun(race_id='old',date='2026-09-01',finish=2,field_size=16,
                        distance=1600,surface='turf',course='Tokyo',margin='0.2')
        return RaceData(race_info=info,horses=[
            HorseEntry(horse_no=1,horse_name='a',recent_runs=[run]),
            HorseEntry(horse_no=2,horse_name='b',recent_runs=[])])

    def test_future_and_target_history_cannot_change_features(self):
        race = self.race(); before = a.raw_features(race)
        old = race.horses[0].recent_runs[0]
        race.horses[0].recent_runs.extend([
            old.model_copy(update={'race_id':'future','date':'2026-10-26','finish':1}),
            old.model_copy(update={'race_id':race.race_info.race_id,'date':'2026-09-02','finish':1}),
            old.model_copy(update={'race_id':'same_day','date':'2026-10-25','finish':1})])
        np.testing.assert_allclose(before,a.raw_features(race),equal_nan=True)

    def test_duplicates_do_not_double_count_history(self):
        race = self.race(); before = a.raw_features(race)
        race.horses[0].recent_runs.append(race.horses[0].recent_runs[0].model_copy())
        np.testing.assert_allclose(before,a.raw_features(race),equal_nan=True)

    def test_market_and_identity_do_not_change_features(self):
        race = self.race(); before = a.raw_features(race)
        for h in race.horses:
            h.odds=999; h.popularity=99; h.horse_name='changed'; h.horse_id='changed'
            for run in h.recent_runs:
                run.odds=1.1; run.popularity=1
        np.testing.assert_allclose(before,a.raw_features(race),equal_nan=True)

    def test_order_equivariance(self):
        race = self.race(); before = a.raw_features(race)
        race.horses.reverse()
        np.testing.assert_allclose(before[::-1],a.raw_features(race),equal_nan=True)

    def test_missing_history_is_not_observed_failure(self):
        race = self.race(); features = a.raw_features(race)
        self.assertEqual(features[1,a.RAW_NAMES.index('last3_count')],0)
        self.assertTrue(np.isnan(features[1,a.RAW_NAMES.index('last3_finish_mean')]))
        self.assertGreater(features[0,a.RAW_NAMES.index('last3_finish_mean')],0)

    def test_unknown_class_not_inferred_from_race_number(self):
        race = self.race(); features = a.raw_features(race)
        self.assertTrue(np.isnan(features[0,a.RAW_NAMES.index('class_explicit')]))
        self.assertEqual(features[0,a.RAW_NAMES.index('class_missing')],1)

    def test_equal_weight_target_gives_all_three_same_label(self):
        y,w = a.labels_weights([{'numbers':list(range(1,10)),'top3':[3,5,9]}])
        np.testing.assert_array_equal(y,[0,0,1,0,1,0,0,0,1])
        self.assertTrue(np.all(w==w[0]))

    def test_direct_selection_orders_by_place_signal(self):
        self.assertEqual(a.direct_ranking([7,3,9],np.asarray([.2,.8,.8])),[3,9,7])


if __name__=='__main__':
    unittest.main()

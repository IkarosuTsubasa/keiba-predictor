from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo, RecentRun
from pipeline.experiment_agent_place_models import (
    blend, context_matrix, metric_vector, repeated_splits,
    sanitize_race, summarize_vectors,
)


class AgentPlaceExperimentTests(unittest.TestCase):
    def test_sanitize_preserves_source_and_removes_target_and_future_runs(self):
        race = RaceData(race_info=RaceInfo(race_id="target", race_date="2026-09-07"), horses=[
            HorseEntry(horse_no=1, horse_name="A", odds=2.0, popularity=1, recent_runs=[
                RecentRun(race_id="prior", date="2026-09-01", odds=3.0, popularity=2),
                RecentRun(race_id="target", date="2026-09-07"),
                RecentRun(race_id="future", date="2026-09-08"),
                RecentRun(race_id="unknown"),
            ])])
        clean, removed = sanitize_race(race)
        self.assertEqual(removed, 2)
        self.assertEqual(len(race.horses[0].recent_runs), 4)
        self.assertEqual(race.horses[0].odds, 2.0)
        self.assertIsNone(clean.horses[0].odds)
        self.assertIsNone(clean.horses[0].recent_runs[0].odds)
        self.assertEqual([r.race_id for r in clean.horses[0].recent_runs], ["prior", "unknown"])

    def test_date_grouped_folds_cover_every_race_once_without_overlap(self):
        rows = [{"date": f"2026-08-{i // 3 + 1:02d}"} for i in range(60)]
        seen = np.zeros(len(rows), dtype=int)
        for train, test in repeated_splits(rows, 42):
            self.assertFalse({rows[i]["date"] for i in train} & {rows[i]["date"] for i in test})
            seen[test] += 1
        np.testing.assert_array_equal(seen, 1)

    def test_missing_history_is_explicit_and_market_changes_do_not_change_features(self):
        race = RaceData(race_info=RaceInfo(race_id="target", race_date="2026-09-07"),
                        horses=[HorseEntry(horse_no=1, horse_name="A", odds=2, popularity=1)])
        scores = [SimpleNamespace(horse_no=1, scores=SimpleNamespace(ability_score=5, recent_quality_score=5))]
        features = context_matrix(race, scores, np.zeros((1, 13)))
        self.assertEqual(features[0, 13], 0)
        self.assertTrue(np.isnan(features[0, 14]))
        self.assertEqual(features[0, -6], 1)
        race.horses[0].odds, race.horses[0].popularity = 100, 15
        np.testing.assert_allclose(features, context_matrix(race, scores, np.zeros((1, 13))), equal_nan=True)

    def test_podium_metrics_count_places_not_only_winner(self):
        row = {"numbers": list(range(1, 9)), "top3": [1, 6, 7], "winner": 1}
        first = metric_vector(row, list(range(1, 9)))
        second = metric_vector(row, [1, 6, 7, 2, 3, 4, 5, 8])
        result = summarize_vectors(np.asarray([first, second]))
        self.assertEqual(result["slots"], 4)
        self.assertEqual(result["all3"], 1)
        self.assertEqual(result["atleast2"], 1)
        self.assertEqual(result["top1_win"], 2)
        self.assertAlmostEqual(result["recall_pct"], 100 * 4 / 6, places=5)
        with self.assertRaises(ValueError):
            metric_vector(row, [1, 1, 2, 3, 4, 5, 6, 7])

    def test_blending_preserves_axis_and_runner_permutation(self):
        numbers = list(range(1, 9))
        ranking = blend(numbers, np.arange(8), numbers, 2, 1.0)
        self.assertEqual(ranking[:2], [1, 2])
        self.assertEqual(ranking[2:], [8, 7, 6, 5, 4, 3])


if __name__ == "__main__":
    unittest.main()

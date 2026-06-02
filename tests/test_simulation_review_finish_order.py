import unittest

from keiba_llm_agent.schemas.review import Review
from keiba_llm_agent.simulation.simulation_reviewer import review_race_simulation
from tests.test_simulation_reviewer import _prediction_with_simulation


def _review() -> Review:
    return Review.model_validate(
        {
            "race_id": "sim_review_001",
            "hit_summary": {
                "main_mark_top3": True,
                "marked_horses_top3_count": 2,
                "bet_hit": False,
                "roi": 0.0,
                "total_stake": 100,
                "total_return": 0,
            },
            "bet_results": [],
            "good_points": [],
            "bad_points": [],
            "lessons": [],
        }
    )


class SimulationReviewFinishOrderTests(unittest.TestCase):
    def test_favorable_horse_finish_four_becomes_close(self) -> None:
        prediction = _prediction_with_simulation()
        result = {
            "race_id": "sim_review_001",
            "finish_order": [
                {"horse_no": 6, "finish": 1},
                {"horse_no": 7, "finish": 2},
                {"horse_no": 8, "finish": 3},
                {"horse_no": 5, "finish": 4},
            ],
            "result": {"1st": 6, "2nd": 7, "3rd": 8},
        }
        simulation_review = review_race_simulation(prediction, result, _review())
        item = simulation_review.favorable_horses_result[0]
        self.assertEqual(item.finish, 4)
        self.assertEqual(item.status, "close")

    def test_finish_within_top3_becomes_top3_status(self) -> None:
        prediction = _prediction_with_simulation()
        result = {
            "race_id": "sim_review_001",
            "finish_order": [
                {"horse_no": 5, "finish": 1},
                {"horse_no": 6, "finish": 2},
                {"horse_no": 7, "finish": 3},
            ],
            "result": {"1st": 5, "2nd": 6, "3rd": 7},
        }
        simulation_review = review_race_simulation(prediction, result, _review())
        item = simulation_review.favorable_horses_result[0]
        self.assertEqual(item.status, "top3")
        self.assertEqual(item.result, "hit")

    def test_unknown_only_when_horse_missing_from_finish_order(self) -> None:
        prediction = _prediction_with_simulation()
        result = {
            "race_id": "sim_review_001",
            "finish_order": [
                {"horse_no": 6, "finish": 1},
                {"horse_no": 7, "finish": 2},
                {"horse_no": 8, "finish": 3},
            ],
            "result": {"1st": 6, "2nd": 7, "3rd": 8},
        }
        simulation_review = review_race_simulation(prediction, result, _review())
        item = simulation_review.favorable_horses_result[0]
        self.assertEqual(item.status, "unknown")
        self.assertEqual(item.result, "unknown")


if __name__ == "__main__":
    unittest.main()

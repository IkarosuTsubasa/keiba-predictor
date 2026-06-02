from __future__ import annotations

import unittest

from keiba_llm_agent.agents.race_review_agent import RaceReviewAgent
from keiba_llm_agent.llm_client import MockLLMClient
from keiba_llm_agent.schemas.prediction import (
    BetSuggestion,
    HorseScore,
    Prediction,
    ScoreBreakdown,
    StrategyDecision,
)
from keiba_llm_agent.schemas.race_data import RaceInfo


def build_prediction(bets: list[BetSuggestion]) -> Prediction:
    return Prediction(
        race_id="sample_001",
        race_info=RaceInfo(
            race_id="sample_001",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
        ),
        marks={"◎": 12, "○": 8, "▲": 7, "△": 0, "☆": 0},
        horse_scores=[
            HorseScore(
                horse_no=12,
                horse_name="A",
                total_score=40.0,
                reason="reason",
                scores=ScoreBreakdown(
                    recent_form=7,
                    distance_fit=7,
                    course_fit=7,
                    track_condition_fit=7,
                    jockey_fit=7,
                    odds_value=7,
                    risk=-1,
                ),
            )
        ],
        bets=bets,
        summary="summary",
        risks=[],
        used_lessons=[],
        strategy=StrategyDecision(
            bet_decision="BET",
            confidence="medium",
            participation_level="light",
            reason_codes=[],
            reason="reason",
        ),
    )


class ReviewBetResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = RaceReviewAgent(llm_client=MockLLMClient())
        self.result = {
            "race_id": "sample_001",
            "result": {"1st": 12, "2nd": 8, "3rd": 7},
            "payouts": [
                {"type": "wide", "combination": "8-12", "payout": 420},
                {"type": "place", "combination": "12", "payout": 180},
                {"type": "win", "combination": "12", "payout": 520},
            ],
        }

    def test_wide_hit_when_both_horses_in_top3(self) -> None:
        prediction = build_prediction(
            [BetSuggestion(bet_type="ワイド", horse_numbers=[8, 12], amount=100, reason="reason")]
        )
        review = self.agent.run("sample_001", self.result, prediction, race_info=prediction.race_info)
        self.assertTrue(review.bet_results[0].hit)

    def test_wide_miss_when_horses_not_in_top3(self) -> None:
        prediction = build_prediction(
            [BetSuggestion(bet_type="ワイド", horse_numbers=[16, 6], amount=100, reason="reason")]
        )
        review = self.agent.run("sample_001", self.result, prediction, race_info=prediction.race_info)
        self.assertFalse(review.bet_results[0].hit)

    def test_place_hit_when_horse_in_top3(self) -> None:
        prediction = build_prediction(
            [BetSuggestion(bet_type="複勝", horse_numbers=[12], amount=100, reason="reason")]
        )
        review = self.agent.run("sample_001", self.result, prediction, race_info=prediction.race_info)
        self.assertTrue(review.bet_results[0].hit)

    def test_win_hit_when_horse_is_first(self) -> None:
        prediction = build_prediction(
            [BetSuggestion(bet_type="単勝", horse_numbers=[12], amount=100, reason="reason")]
        )
        review = self.agent.run("sample_001", self.result, prediction, race_info=prediction.race_info)
        self.assertTrue(review.bet_results[0].hit)

    def test_return_amount_and_roi_are_calculated(self) -> None:
        prediction = build_prediction(
            [BetSuggestion(bet_type="ワイド", horse_numbers=[8, 12], amount=100, reason="reason")]
        )
        review = self.agent.run("sample_001", self.result, prediction, race_info=prediction.race_info)
        self.assertEqual(review.bet_results[0].payout, 420)
        self.assertEqual(review.bet_results[0].return_amount, 420)
        self.assertEqual(review.hit_summary.total_stake, 100)
        self.assertEqual(review.hit_summary.total_return, 420)
        self.assertEqual(review.hit_summary.roi, 4.2)

    def test_missing_payout_keeps_hit_but_return_zero_and_warning(self) -> None:
        prediction = build_prediction(
            [BetSuggestion(bet_type="ワイド", horse_numbers=[8, 12], amount=100, reason="reason")]
        )
        result_without_payout = {
            "race_id": "sample_001",
            "result": {"1st": 12, "2nd": 8, "3rd": 7},
            "payouts": [],
        }
        review = self.agent.run(
            "sample_001",
            result_without_payout,
            prediction,
            race_info=prediction.race_info,
        )
        self.assertTrue(review.bet_results[0].hit)
        self.assertEqual(review.bet_results[0].return_amount, 0)
        self.assertIn("払戻データが不足しているためROIは暫定です。", review.bad_points)


if __name__ == "__main__":
    unittest.main()

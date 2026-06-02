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


def _prediction_with_bet(bet: BetSuggestion) -> Prediction:
    return Prediction(
        race_id="sample_001",
        race_info=RaceInfo(
            race_id="sample_001",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
        ),
        marks={"◎": 3, "○": 5, "▲": 8, "△": 0, "☆": 0},
        horse_scores=[
            HorseScore(
                horse_no=3,
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
        bets=[bet],
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


def _result_with_payouts(payouts: list[dict]) -> dict:
    return {
        "race_id": "sample_001",
        "result": {"1st": 3, "2nd": 5, "3rd": 8},
        "finish_order": [
            {"finish": 1, "horse_no": 3, "horse_name": "A"},
            {"finish": 2, "horse_no": 5, "horse_name": "B"},
            {"finish": 3, "horse_no": 8, "horse_name": "C"},
        ],
        "payouts": payouts,
    }


class ReviewPayoutMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = RaceReviewAgent(llm_client=MockLLMClient())

    def test_wide_matches_unordered_combination(self) -> None:
        prediction = _prediction_with_bet(BetSuggestion(bet_type="ワイド", horse_numbers=[3, 5], amount=100, reason="reason"))
        result = _result_with_payouts([{"bet_type": "ワイド", "combination": "5-3", "payout": 520}])
        review = self.agent.run("sample_001", result, prediction, race_info=prediction.race_info)
        self.assertTrue(review.bet_results[0].hit)
        self.assertEqual(review.hit_summary.total_return, 520)

    def test_quinella_matches_unordered_combination(self) -> None:
        prediction = _prediction_with_bet(BetSuggestion(bet_type="馬連", horse_numbers=[5, 3], amount=100, reason="reason"))
        result = _result_with_payouts([{"bet_type": "馬連", "combination": "3-5", "payout": 840}])
        review = self.agent.run("sample_001", result, prediction, race_info=prediction.race_info)
        self.assertTrue(review.bet_results[0].hit)
        self.assertEqual(review.hit_summary.total_return, 840)

    def test_exacta_requires_order(self) -> None:
        prediction = _prediction_with_bet(BetSuggestion(bet_type="馬単", horse_numbers=[5, 3], amount=100, reason="reason"))
        result = _result_with_payouts([{"bet_type": "馬単", "combination": "3-5", "payout": 1420}])
        review = self.agent.run("sample_001", result, prediction, race_info=prediction.race_info)
        self.assertFalse(review.bet_results[0].hit)

    def test_trio_matches_unordered_combination(self) -> None:
        prediction = _prediction_with_bet(BetSuggestion(bet_type="三連複", horse_numbers=[8, 3, 5], amount=100, reason="reason"))
        result = _result_with_payouts([{"bet_type": "三連複", "combination": "3-5-8", "payout": 2450}])
        review = self.agent.run("sample_001", result, prediction, race_info=prediction.race_info)
        self.assertTrue(review.bet_results[0].hit)
        self.assertEqual(review.hit_summary.total_return, 2450)

    def test_trifecta_requires_order(self) -> None:
        prediction = _prediction_with_bet(BetSuggestion(bet_type="三連単", horse_numbers=[3, 5, 8], amount=100, reason="reason"))
        result = _result_with_payouts([{"bet_type": "三連単", "combination": "3-5-8", "payout": 8440}])
        review = self.agent.run("sample_001", result, prediction, race_info=prediction.race_info)
        self.assertTrue(review.bet_results[0].hit)
        self.assertEqual(review.hit_summary.total_return, 8440)

    def test_missing_payout_sets_warning(self) -> None:
        prediction = _prediction_with_bet(BetSuggestion(bet_type="ワイド", horse_numbers=[3, 5], amount=100, reason="reason"))
        result = _result_with_payouts([])
        review = self.agent.run("sample_001", result, prediction, race_info=prediction.race_info)
        self.assertTrue(review.bet_results[0].hit)
        self.assertTrue(review.payout_warning)
        self.assertIn("Bet hit but payout data missing. ROI is unreliable.", review.review_warnings)


if __name__ == "__main__":
    unittest.main()

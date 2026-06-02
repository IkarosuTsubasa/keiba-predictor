from __future__ import annotations

import unittest

from keiba_llm_agent.llm.llm_client import BaseLLMClient
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.review import Review
from keiba_llm_agent.simulation.simulation_reviewer import review_race_simulation


def _prediction_with_simulation() -> Prediction:
    return Prediction.model_validate(
        {
            "race_id": "sim_review_001",
            "race_info": {
                "race_id": "sim_review_001",
                "race_name": "シミュレーション回顧特別",
                "race_date": "2026-05-17",
                "course": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "weather": "晴",
            },
            "marks": {"◎": 5, "○": 6, "▲": 7, "△": 8, "☆": 9},
            "horse_scores": [
                {
                    "horse_no": 5,
                    "horse_name": "ファーングロット",
                    "scores": {
                        "recent_form": 8,
                        "distance_fit": 8,
                        "course_fit": 7,
                        "track_condition_fit": 7,
                        "jockey_fit": 6,
                        "odds_value": 5,
                        "risk": -2,
                    },
                    "total_score": 40.0,
                    "reason": "reason",
                }
            ],
            "bets": [{"bet_type": "ワイド", "horse_numbers": [5, 6], "amount": 100, "reason": "reason"}],
            "summary": "summary",
            "risks": [],
            "used_lessons": [],
            "pace_analyses": [
                {
                    "horse_no": 5,
                    "horse_name": "ファーングロット",
                    "running_style": "先行",
                    "early_position_score": 7.0,
                    "late_position_score": 6.0,
                    "position_stability": "安定",
                    "positive_flags": ["PACE_FIT"],
                    "risk_flags": [],
                    "overall_comment": "comment",
                },
                {
                    "horse_no": 6,
                    "horse_name": "ラヴァンダ",
                    "running_style": "先行",
                    "early_position_score": 7.0,
                    "late_position_score": 6.0,
                    "position_stability": "安定",
                    "positive_flags": ["PACE_FIT"],
                    "risk_flags": [],
                    "overall_comment": "comment",
                },
                {
                    "horse_no": 7,
                    "horse_name": "アイサンサン",
                    "running_style": "差し",
                    "early_position_score": 4.0,
                    "late_position_score": 7.5,
                    "position_stability": "安定",
                    "positive_flags": ["PACE_FIT"],
                    "risk_flags": [],
                    "overall_comment": "comment",
                },
            ],
            "race_pace_projection": {
                "projected_pace": "average",
                "front_runner_count": 1,
                "stalker_count": 3,
                "closer_count": 4,
                "pace_comment": "平均想定。",
                "favorable_styles": ["先行", "差し"],
                "risk_styles": [],
            },
            "race_simulation": {
                "race_id": "sim_review_001",
                "projected_pace": "average",
                "race_flow": "平均的な流れ。",
                "key_positions": "好位勢が主導。",
                "favorable_horses": [
                    {"horse_no": 5, "horse_name": "ファーングロット", "reason": "先行力を評価。"},
                    {"horse_no": 6, "horse_name": "ラヴァンダ", "reason": "好位維持に期待。"},
                    {"horse_no": 7, "horse_name": "アイサンサン", "reason": "差し脚に注意。"},
                ],
                "risk_horses": [
                    {"horse_no": 8, "horse_name": "カムニャック", "reason": "距離・コース適性に未知要素が残る。"}
                ],
                "win_scenario": "◎が好位抜け出し。",
                "top3_scenario": "◎○▲が上位候補。",
                "betting_scenario": "買い目はワイド5-6を100円。上位拮抗のため軽めに絞る。",
                "confidence_comment": "中位の信頼度。",
                "reasoning_summary": "平均寄り。先行〜差し勢の持続力勝負を想定。",
                "warnings": [],
            },
            "strategy": {
                "bet_decision": "BET",
                "confidence": "medium",
                "participation_level": "light",
                "reason_codes": [],
                "reason": "reason",
            },
        }
    )


def _review(bet_hit: bool = False) -> Review:
    return Review.model_validate(
        {
            "race_id": "sim_review_001",
            "hit_summary": {
                "main_mark_top3": True,
                "marked_horses_top3_count": 2,
                "bet_hit": bet_hit,
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


class _FailingLLM(BaseLLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        raise RuntimeError("boom")


class SimulationReviewerTests(unittest.TestCase):
    def test_favorable_and_risk_result_classification(self) -> None:
        prediction = _prediction_with_simulation()
        review = _review(bet_hit=True)
        result = {
            "race_id": "sim_review_001",
            "result": {"1st": 5, "2nd": 6, "3rd": 7},
            "finish_order": [
                {"horse_no": 5, "finish": 1},
                {"horse_no": 6, "finish": 2},
                {"horse_no": 7, "finish": 3},
                {"horse_no": 8, "finish": 8},
            ],
        }
        simulation_review = review_race_simulation(prediction, result, review)
        self.assertEqual(simulation_review.favorable_horses_result[0].result, "hit")
        self.assertEqual(simulation_review.risk_horses_result[0].result, "risk_materialized")
        self.assertEqual(simulation_review.scenario_hit_level, "high")
        self.assertIn("成功", simulation_review.betting_scenario_review)

    def test_low_and_medium_scenario_levels(self) -> None:
        prediction = _prediction_with_simulation()
        review = _review()
        medium_result = {
            "race_id": "sim_review_001",
            "result": {"1st": 5, "2nd": 6, "3rd": 9},
            "finish_order": [
                {"horse_no": 5, "finish": 1},
                {"horse_no": 6, "finish": 2},
                {"horse_no": 9, "finish": 3},
            ],
        }
        low_result = {
            "race_id": "sim_review_001",
            "result": {"1st": 8, "2nd": 9, "3rd": 10},
            "finish_order": [
                {"horse_no": 8, "finish": 1},
                {"horse_no": 9, "finish": 2},
                {"horse_no": 10, "finish": 3},
                {"horse_no": 5, "finish": 6},
                {"horse_no": 6, "finish": 7},
                {"horse_no": 7, "finish": 8},
            ],
        }
        self.assertEqual(review_race_simulation(prediction, medium_result, review).scenario_hit_level, "medium")
        low_review = review_race_simulation(prediction, low_result, review)
        self.assertEqual(low_review.scenario_hit_level, "low")
        self.assertEqual(low_review.favorable_horses_result[0].result, "miss")

    def test_llm_failure_does_not_block_fallback(self) -> None:
        prediction = _prediction_with_simulation()
        review = _review()
        result = {
            "race_id": "sim_review_001",
            "result": {"1st": 5, "2nd": 6, "3rd": 7},
        }
        simulation_review = review_race_simulation(prediction, result, review, llm_client=_FailingLLM())
        self.assertTrue(simulation_review.overall_comment)
        self.assertGreaterEqual(len(simulation_review.new_lessons), 1)


if __name__ == "__main__":
    unittest.main()

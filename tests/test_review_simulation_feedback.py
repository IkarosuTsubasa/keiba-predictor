from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.main import run_review
from keiba_llm_agent.reports.report_generator import generate_review_report
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.review import Review
from keiba_llm_agent.social.post_generator import build_review_post


class ReviewSimulationFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.prediction_path = self.temp_path / "prediction.json"
        self.result_path = self.temp_path / "result.json"
        self.review_path = self.temp_path / "review.json"
        self.lessons_path = self.temp_path / "lessons.json"
        self.lessons_path.write_text("[]\n", encoding="utf-8")
        prediction_payload = {
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
                },
                {
                    "horse_no": 6,
                    "horse_name": "ラヴァンダ",
                    "scores": {
                        "recent_form": 7,
                        "distance_fit": 7,
                        "course_fit": 6,
                        "track_condition_fit": 6,
                        "jockey_fit": 6,
                        "odds_value": 5,
                        "risk": -2,
                    },
                    "total_score": 39.0,
                    "reason": "reason",
                },
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
                    "running_style": "差し",
                    "early_position_score": 4.0,
                    "late_position_score": 7.0,
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
                    "late_position_score": 7.0,
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
        result_payload = {
            "race_id": "sim_review_001",
            "result": {"1st": 5, "2nd": 6, "3rd": 9},
            "finish_order": [
                {"horse_no": 5, "finish": 1},
                {"horse_no": 6, "finish": 2},
                {"horse_no": 9, "finish": 3},
                {"horse_no": 7, "finish": 6},
                {"horse_no": 8, "finish": 8},
            ],
            "payouts": [{"type": "wide", "combination": "5-6", "payout": 420}],
        }
        self.prediction_path.write_text(json.dumps(prediction_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_review_outputs_simulation_review_and_report_section(self) -> None:
        review_result, saved_path = run_review(
            race_id="sim_review_001",
            result_path=self.result_path,
            prediction_path=self.prediction_path,
            output_path=self.review_path,
            lessons_path=self.lessons_path,
        )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertIn("simulation_review", payload)
        self.assertIn("scenario_hit_level", payload["simulation_review"])
        self.assertTrue(payload["simulation_review"]["overall_comment"])

        prediction = Prediction.model_validate_json(self.prediction_path.read_text(encoding="utf-8"))
        review = Review.model_validate(payload)
        markdown = generate_review_report(prediction, review)
        self.assertIn("## レースシミュレーション回顧", markdown)
        text = build_review_post(prediction, review)
        self.assertLessEqual(len(text), 280)

        lessons_payload = json.loads(self.lessons_path.read_text(encoding="utf-8"))
        lesson_texts = [item["lesson"] for item in lessons_payload]
        self.assertEqual(len(lesson_texts), len(set(lesson_texts)))


if __name__ == "__main__":
    unittest.main()

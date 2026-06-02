from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.backtest.backtest_runner import generate_backtest_markdown, run_backtest
from keiba_llm_agent.main import run_backtest_command
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.review import Review
from keiba_llm_agent.schemas.result import ResultData


def _prediction_payload(
    race_id: str,
    race_date: str,
    scores: list[dict[str, float]],
) -> dict:
    horse_scores = []
    for item in scores:
        total = round(item["base"] + item["pedigree"] + item["race_level"] + item["pace"], 1)
        horse_scores.append(
            {
                "horse_no": item["horse_no"],
                "horse_name": f"Horse{item['horse_no']}",
                "scores": {
                    "recent_form": 7,
                    "distance_fit": 7,
                    "course_fit": 7,
                    "track_condition_fit": 7,
                    "jockey_fit": 7,
                    "odds_value": 5,
                    "risk": -3,
                },
                "base_total_score": item["base"],
                "pedigree_adjustment": {
                    "pedigree_bonus": max(item["pedigree"], 0.0),
                    "pedigree_penalty": min(item["pedigree"], 0.0),
                    "pedigree_adjustment": item["pedigree"],
                    "reason": "",
                },
                "race_level_adjustment": {"adjustment": item["race_level"], "reason": ""},
                "pace_adjustment": {"adjustment": item["pace"], "reason": ""},
                "score_breakdown": {
                    "base_total_score": item["base"],
                    "pedigree_adjustment": item["pedigree"],
                    "race_level_adjustment": item["race_level"],
                    "pace_adjustment": item["pace"],
                    "total_score": total,
                },
                "total_score": total,
                "reason": "reason",
            }
        )
    return {
        "race_id": race_id,
        "race_info": {
            "race_id": race_id,
            "race_name": f"Race {race_id}",
            "race_date": race_date,
            "course": "東京",
            "surface": "芝",
            "distance": 1600,
            "track_condition": "良",
            "weather": "晴",
        },
        "marks": {"◎": horse_scores[0]["horse_no"], "○": horse_scores[1]["horse_no"], "▲": horse_scores[2]["horse_no"], "△": 0, "☆": 0},
        "horse_scores": horse_scores,
        "bets": [{"bet_type": "ワイド", "horse_numbers": [1, 2], "amount": 100, "reason": "test"}],
        "summary": "summary",
        "risks": [],
        "used_lessons": [],
        "strategy": {
            "bet_decision": "BET",
            "confidence": "medium",
            "participation_level": "light",
            "reason_codes": ["HEURISTIC_MODEL_ONLY"],
            "reason": "reason",
        },
    }


class BacktestRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.predictions_dir = self.root / "predictions"
        self.results_dir = self.root / "results"
        self.reviews_dir = self.root / "reviews"
        self.backtests_dir = self.root / "backtests"
        self.predictions_dir.mkdir()
        self.results_dir.mkdir()
        self.reviews_dir.mkdir()
        self.backtests_dir.mkdir()

        in_range_reviewed = _prediction_payload(
            "r1",
            "2026-05-17",
            [
                {"horse_no": 1, "base": 35.0, "pedigree": 0.0, "race_level": -0.5, "pace": -0.3},
                {"horse_no": 2, "base": 34.8, "pedigree": 0.8, "race_level": 0.4, "pace": 0.3},
                {"horse_no": 3, "base": 33.0, "pedigree": 0.0, "race_level": 0.0, "pace": 0.0},
            ],
        )
        in_range_pending = _prediction_payload(
            "r2",
            "2026-05-18",
            [
                {"horse_no": 4, "base": 30.0, "pedigree": 0.0, "race_level": 0.0, "pace": 0.0},
                {"horse_no": 5, "base": 29.0, "pedigree": 0.0, "race_level": 0.0, "pace": 0.0},
                {"horse_no": 6, "base": 28.0, "pedigree": 0.0, "race_level": 0.0, "pace": 0.0},
            ],
        )
        out_of_range = _prediction_payload(
            "r3",
            "2026-06-01",
            [
                {"horse_no": 7, "base": 31.0, "pedigree": 0.0, "race_level": 0.0, "pace": 0.0},
                {"horse_no": 8, "base": 30.0, "pedigree": 0.0, "race_level": 0.0, "pace": 0.0},
                {"horse_no": 9, "base": 29.0, "pedigree": 0.0, "race_level": 0.0, "pace": 0.0},
            ],
        )
        (self.predictions_dir / "r1.json").write_text(json.dumps(in_range_reviewed, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.predictions_dir / "r2.json").write_text(json.dumps(in_range_pending, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.predictions_dir / "r3.json").write_text(json.dumps(out_of_range, ensure_ascii=False, indent=2), encoding="utf-8")

        result = ResultData.model_validate({"race_id": "r1", "result": {"1st": 2, "2nd": 1, "3rd": 3}, "payouts": []})
        review = Review.model_validate(
            {
                "race_id": "r1",
                "hit_summary": {
                    "main_mark_top3": True,
                    "marked_horses_top3_count": 3,
                    "bet_hit": True,
                    "roi": 2.0,
                    "total_stake": 100,
                    "total_return": 200,
                },
                "bet_results": [],
                "good_points": ["good"],
                "bad_points": ["bad"],
                "lessons": [],
            }
        )
        (self.results_dir / "r1.json").write_text(json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2), encoding="utf-8")
        (self.reviews_dir / "r1.json").write_text(json.dumps(review.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_run_backtest_filters_period_and_computes_metrics(self) -> None:
        report = run_backtest(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        self.assertEqual(report["race_count"], 2)
        self.assertEqual(report["reviewed_race_count"], 1)
        self.assertEqual(report["pending_race_count"], 1)
        self.assertEqual(report["total_stake"], 100)
        self.assertEqual(report["total_return"], 200)
        self.assertEqual(report["roi"], 2.0)
        self.assertEqual(report["modes"]["base_only"]["main_mark_top3_rate"], 1.0)
        self.assertEqual(report["modes"]["full_adjusted"]["avg_marked_top3_count"], 3.0)
        pending = next(detail for detail in report["race_details"] if detail["race_id"] == "r2")
        self.assertEqual(pending["status"], "pending")
        self.assertIn("result missing for race_id=r2", report["warnings"])

    def test_markdown_report_is_generated(self) -> None:
        report = run_backtest(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        markdown = generate_backtest_markdown(report)
        self.assertIn("# Backtest Report", markdown)
        self.assertIn("## Mode Comparison", markdown)
        self.assertIn("review済みレース数: 1", markdown)
        self.assertIn("pendingレース数: 1", markdown)
        self.assertIn("※指標は review済みレースのみで計算。", markdown)
        self.assertIn("r1", markdown)

    def test_command_saves_json_and_markdown(self) -> None:
        with (
            patch("keiba_llm_agent.main.DEFAULT_PREDICTIONS_DIR", self.predictions_dir),
            patch("keiba_llm_agent.main.DEFAULT_RESULTS_DIR", self.results_dir),
            patch("keiba_llm_agent.main.DEFAULT_REVIEWS_DIR", self.reviews_dir),
            patch("keiba_llm_agent.main.DEFAULT_BACKTESTS_DIR", self.backtests_dir),
        ):
            result = run_backtest_command("2026-05-01", "2026-05-31")
        self.assertTrue(Path(result["json_path"]).exists())
        self.assertTrue(Path(result["md_path"]).exists())
        self.assertEqual(result["reviewed_race_count"], 1)
        self.assertEqual(result["pending_race_count"], 1)

    def test_backtest_marks_roi_unreliable_when_payout_warning_exists(self) -> None:
        payload = json.loads((self.reviews_dir / "r1.json").read_text(encoding="utf-8"))
        payload["payout_warning"] = True
        payload["review_warnings"] = ["Bet hit but payout data missing. ROI is unreliable."]
        (self.reviews_dir / "r1.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report = run_backtest(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        markdown = generate_backtest_markdown(report)
        self.assertFalse(report["roi_reliable"])
        self.assertIn("Some payout data missing. ROI may be unreliable.", report["warnings"])
        self.assertIn("ROI: 200.0% (暫定)", markdown)


if __name__ == "__main__":
    unittest.main()

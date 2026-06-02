from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.error_analysis.missed_top3_analyzer import (
    generate_missed_top3_markdown,
    run_missed_top3_analysis,
)
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _horse_score(horse_no: int, horse_name: str, base: float, risk: int = -2) -> dict:
    return {
        "horse_no": horse_no,
        "horse_name": horse_name,
        "scores": {
            "recent_form": 7,
            "distance_fit": 7,
            "course_fit": 7,
            "track_condition_fit": 7,
            "jockey_fit": 7,
            "odds_value": 5,
            "risk": risk,
        },
        "base_total_score": base,
        "pedigree_adjustment": {
            "pedigree_bonus": 0.0,
            "pedigree_penalty": 0.0,
            "pedigree_adjustment": 0.0,
            "reason": "",
        },
        "race_level_adjustment": {"adjustment": 0.0, "reason": ""},
        "pace_adjustment": {"adjustment": 0.0, "reason": ""},
        "score_breakdown": {
            "base_total_score": base,
            "pedigree_adjustment_raw": 0.0,
            "pedigree_weight": 0.2,
            "pedigree_adjustment_weighted": 0.0,
            "race_level_adjustment_raw": 0.0,
            "race_level_weight": 1.0,
            "race_level_adjustment_weighted": 0.0,
            "pace_adjustment_raw": 0.0,
            "pace_weight": 0.0,
            "pace_adjustment_weighted": 0.0,
            "total_score": base,
        },
        "total_score": base,
        "reason": "reason",
    }


def _prediction_payload(
    race_id: str,
    race_name: str,
    race_date: str,
    horse_scores: list[dict],
    deep_analyses: list[dict],
    race_level_analyses: list[dict],
    pace_analyses: list[dict],
) -> dict:
    return {
        "race_id": race_id,
        "race_info": {
            "race_id": race_id,
            "race_name": race_name,
            "race_date": race_date,
            "course": "東京",
            "surface": "芝",
            "distance": 1600,
            "track_condition": "良",
            "weather": "晴",
        },
        "scoring_config": {
            "scoring_mode": "candidate_default",
            "pedigree_weight": 0.2,
            "race_level_weight": 1.0,
            "pace_weight": 0.0,
        },
        "marks": {
            "◎": horse_scores[0]["horse_no"],
            "○": horse_scores[1]["horse_no"],
            "▲": horse_scores[2]["horse_no"],
            "△": horse_scores[3]["horse_no"],
            "☆": horse_scores[4]["horse_no"],
        },
        "horse_scores": horse_scores,
        "bets": [],
        "summary": "summary",
        "risks": [],
        "used_lessons": [],
        "deep_analyses": deep_analyses,
        "pedigree_analyses": [],
        "race_level_analyses": race_level_analyses,
        "pace_analyses": pace_analyses,
        "strategy": {
            "bet_decision": "BET",
            "confidence": "medium",
            "participation_level": "light",
            "reason_codes": ["RACE_LEVEL_USED"],
            "reason": "reason",
        },
    }


def _race_data_payload(
    race_id: str,
    race_name: str,
    race_date: str,
    horses: list[tuple[int, str, float, int]],
) -> dict:
    return {
        "race_info": {
            "race_id": race_id,
            "race_name": race_name,
            "race_date": race_date,
            "course": "東京",
            "surface": "芝",
            "distance": 1600,
            "track_condition": "良",
            "weather": "晴",
        },
        "horses": [
            {
                "horse_no": horse_no,
                "frame_no": horse_no,
                "horse_name": horse_name,
                "odds": odds,
                "popularity": popularity,
                "recent_runs": [],
            }
            for horse_no, horse_name, odds, popularity in horses
        ],
    }


class MissedTop3AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.predictions_dir = self.root / "predictions"
        self.results_dir = self.root / "results"
        self.reviews_dir = self.root / "reviews"
        self.race_data_dir = self.root / "race_data"
        self.predictions_dir.mkdir()
        self.results_dir.mkdir()
        self.reviews_dir.mkdir()
        self.race_data_dir.mkdir()

        race1_scores = [
            _horse_score(1, "A1", 40.0),
            _horse_score(2, "A2", 39.0),
            _horse_score(3, "A3", 38.0),
            _horse_score(4, "A4", 37.0),
            _horse_score(5, "A5", 36.0),
            _horse_score(6, "A6", 35.1, risk=-4),
        ]
        race2_scores = [
            _horse_score(11, "B1", 40.0),
            _horse_score(12, "B2", 39.0),
            _horse_score(13, "B3", 38.0),
            _horse_score(14, "B4", 37.0),
            _horse_score(15, "B5", 36.0),
            _horse_score(16, "B6", 35.5),
        ]

        prediction1 = _prediction_payload(
            "r1",
            "Race1",
            "2026-05-17",
            race1_scores,
            deep_analyses=[
                {
                    "horse_no": 6,
                    "horse_name": "A6",
                    "positive_flags": ["RECENT_FORM_STRONG"],
                    "risk_flags": ["JOCKEY_CHANGE"],
                    "recent_form_summary": "",
                    "distance_analysis": "",
                    "course_analysis": "",
                    "track_condition_analysis": "",
                    "jockey_analysis": "",
                    "odds_analysis": "",
                    "overall_comment": "",
                }
            ],
            race_level_analyses=[
                {
                    "horse_no": 6,
                    "horse_name": "A6",
                    "positive_flags": ["HEAD_TO_HEAD_POSITIVE"],
                    "risk_flags": [],
                    "head_to_head_summary": "",
                    "race_level_summary": "",
                    "opponent_context_summary": "",
                    "overall_comment": "",
                    "adjustment_hint": 0.5,
                }
            ],
            pace_analyses=[
                {
                    "horse_no": 6,
                    "horse_name": "A6",
                    "running_style": "差し",
                    "early_position_score": 5.0,
                    "late_position_score": 7.0,
                    "position_stability": "安定",
                    "positive_flags": ["PACE_FIT"],
                    "risk_flags": [],
                    "overall_comment": "",
                }
            ],
        )
        prediction2 = _prediction_payload(
            "r2",
            "Race2",
            "2026-05-18",
            race2_scores,
            deep_analyses=[],
            race_level_analyses=[],
            pace_analyses=[],
        )

        (self.predictions_dir / "r1.json").write_text(json.dumps(prediction1, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.predictions_dir / "r2.json").write_text(json.dumps(prediction2, ensure_ascii=False, indent=2), encoding="utf-8")

        race_data1 = _race_data_payload(
            "r1",
            "Race1",
            "2026-05-17",
            [
                (1, "A1", 2.5, 1),
                (2, "A2", 3.0, 2),
                (3, "A3", 5.0, 3),
                (4, "A4", 8.0, 4),
                (5, "A5", 12.0, 5),
                (6, "A6", 18.5, 7),
            ],
        )
        race_data2 = _race_data_payload(
            "r2",
            "Race2",
            "2026-05-18",
            [
                (11, "B1", 2.0, 1),
                (12, "B2", 3.5, 4),
                (13, "B3", 5.0, 5),
                (14, "B4", 7.0, 6),
                (15, "B5", 10.0, 7),
                (16, "B6", 4.5, 2),
            ],
        )
        (self.race_data_dir / "r1.json").write_text(json.dumps(race_data1, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.race_data_dir / "r2.json").write_text(json.dumps(race_data2, ensure_ascii=False, indent=2), encoding="utf-8")

        for race_id, top3 in {"r1": [6, 1, 2], "r2": [11, 16, 12]}.items():
            result = ResultData.model_validate(
                {"race_id": race_id, "result": {"1st": top3[0], "2nd": top3[1], "3rd": top3[2]}, "payouts": []}
            )
            review = Review.model_validate(
                {
                    "race_id": race_id,
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
            (self.results_dir / f"{race_id}.json").write_text(
                json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.reviews_dir / f"{race_id}.json").write_text(
                json.dumps(review.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missed_top3_cases_and_categories_are_detected(self) -> None:
        report = run_missed_top3_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            scoring_mode="candidate_default",
            top_n=5,
        )
        self.assertEqual(report["summary"]["reviewed_race_count"], 2)
        self.assertEqual(report["summary"]["total_top3_horses"], 6)
        self.assertEqual(report["summary"]["captured_top3_horses"], 4)
        self.assertEqual(report["summary"]["missed_top3_horses"], 2)

        cases = {(case["race_id"], case["horse_no"]): case for case in report["missed_cases"]}
        self.assertIn(("r1", 6), cases)
        self.assertIn(("r2", 16), cases)

        race1_case = cases[("r1", 6)]
        self.assertEqual(race1_case["predicted_rank"], 6)
        self.assertEqual(race1_case["score_gap_to_top5"], 0.9)
        self.assertIn("JUST_BELOW_TOP5", race1_case["miss_categories"])
        self.assertIn("ODDS_UNDERESTIMATED", race1_case["miss_categories"])
        self.assertIn("RECENT_FORM_UNDERESTIMATED", race1_case["miss_categories"])
        self.assertIn("RACE_LEVEL_UNDERESTIMATED", race1_case["miss_categories"])
        self.assertIn("RISK_OVER_PENALIZED", race1_case["miss_categories"])

        race2_case = cases[("r2", 16)]
        self.assertIn("JUST_BELOW_TOP5", race2_case["miss_categories"])
        self.assertIn("POPULAR_BUT_MISSED", race2_case["miss_categories"])
        self.assertNotIn("RISK_OVER_PENALIZED", race2_case["miss_categories"])

        self.assertEqual(report["category_counts"]["JUST_BELOW_TOP5"], 2)
        self.assertEqual(report["category_counts"]["ODDS_UNDERESTIMATED"], 1)
        self.assertEqual(report["category_counts"]["POPULAR_BUT_MISSED"], 1)
        self.assertEqual(report["category_counts"]["RECENT_FORM_UNDERESTIMATED"], 1)
        self.assertEqual(report["category_counts"]["RACE_LEVEL_UNDERESTIMATED"], 1)
        self.assertNotIn("RISK_OVER_PENALIZED classification may still be too broad.", report["warnings"])

    def test_simulate_borderline_recovery_reports_recoverable_cases(self) -> None:
        report = run_missed_top3_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            scoring_mode="candidate_default",
            top_n=5,
            simulate_borderline_recovery=True,
        )
        self.assertEqual(report["summary"]["borderline_rank6_count"], 2)
        self.assertEqual(report["summary"]["recovery_candidate_count"], 1)
        self.assertEqual(report["summary"]["theoretically_recoverable_top3_count"], 1)
        self.assertEqual(report["summary"]["expected_avg_captured_top3_per_race_after_recovery"], 2.5)
        race1_case = next(case for case in report["missed_cases"] if case["race_id"] == "r1" and case["horse_no"] == 6)
        self.assertTrue(race1_case["recovery_candidate"])

    def test_top_n_6_reduces_missed_cases(self) -> None:
        report_top5 = run_missed_top3_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            top_n=5,
        )
        report_top6 = run_missed_top3_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            top_n=6,
        )
        self.assertGreater(report_top5["summary"]["missed_top3_horses"], report_top6["summary"]["missed_top3_horses"])
        self.assertEqual(report_top6["summary"]["missed_top3_horses"], 0)

    def test_markdown_report_is_generated(self) -> None:
        report = run_missed_top3_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            top_n=5,
        )
        markdown = generate_missed_top3_markdown(report)
        self.assertIn("# Missed Top3 Error Analysis", markdown)
        self.assertIn("## Miss Category Ranking", markdown)
        self.assertIn("## High Priority Misses", markdown)
        self.assertIn("## Race Details", markdown)


if __name__ == "__main__":
    unittest.main()

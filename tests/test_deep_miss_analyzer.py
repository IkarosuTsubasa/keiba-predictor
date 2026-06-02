from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.error_analysis.deep_miss_analyzer import (
    generate_deep_miss_markdown,
    run_deep_miss_analysis,
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
    pedigree_analyses: list[dict],
    race_level_analyses: list[dict],
    pace_analyses: list[dict],
) -> dict:
    return {
        "race_id": race_id,
        "race_info": {
            "race_id": race_id,
            "race_name": race_name,
            "race_date": race_date,
            "course": "京都",
            "surface": "芝",
            "distance": 2400,
            "track_condition": "良",
            "weather": "晴",
        },
        "scoring_profile": "accuracy_default",
        "scoring_mode": "candidate_default",
        "borderline_recovery_enabled": True,
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
        "pedigree_analyses": pedigree_analyses,
        "race_level_analyses": race_level_analyses,
        "pace_analyses": pace_analyses,
        "strategy": {
            "bet_decision": "BET",
            "confidence": "medium",
            "participation_level": "light",
            "reason_codes": ["RACE_LEVEL_USED"],
            "reason": "reason",
        },
        "borderline_recovery_result": {"recovery_applied": False, "recovery_cases": []},
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
            "course": "京都",
            "surface": "芝",
            "distance": 2400,
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


class DeepMissAnalyzerTests(unittest.TestCase):
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

        scores = [
            _horse_score(1, "H1", 50.0),
            _horse_score(2, "H2", 49.0),
            _horse_score(3, "H3", 48.0),
            _horse_score(4, "H4", 47.0),
            _horse_score(5, "H5", 46.0),
            _horse_score(6, "H6", 45.0),
            _horse_score(7, "H7", 44.0),
            _horse_score(8, "H8", 43.0),
            _horse_score(9, "H9", 42.0),
            _horse_score(10, "H10", 41.0),
            _horse_score(11, "H11", 40.0),
            _horse_score(12, "H12", 39.0),
            _horse_score(13, "H13", 38.0, risk=-5),
        ]
        deep_analyses = [
            {
                "horse_no": 7,
                "horse_name": "H7",
                "positive_flags": ["RECENT_FORM_STABLE", "DISTANCE_FIT"],
                "risk_flags": [],
                "recent_form_summary": "",
                "distance_analysis": "",
                "course_analysis": "",
                "track_condition_analysis": "",
                "jockey_analysis": "",
                "odds_analysis": "",
                "overall_comment": "",
            },
            {
                "horse_no": 10,
                "horse_name": "H10",
                "positive_flags": ["RECENT_FORM_STRONG"],
                "risk_flags": [],
                "recent_form_summary": "",
                "distance_analysis": "",
                "course_analysis": "",
                "track_condition_analysis": "",
                "jockey_analysis": "",
                "odds_analysis": "",
                "overall_comment": "",
            },
            {
                "horse_no": 13,
                "horse_name": "H13",
                "positive_flags": [],
                "risk_flags": ["RECENT_FORM_WEAK", "RECENT_FORM_DECLINING", "JOCKEY_CHANGE", "COURSE_UNKNOWN"],
                "recent_form_summary": "",
                "distance_analysis": "",
                "course_analysis": "",
                "track_condition_analysis": "",
                "jockey_analysis": "",
                "odds_analysis": "",
                "overall_comment": "",
            },
        ]
        pedigree_analyses = [
            {
                "horse_no": 13,
                "horse_name": "H13",
                "sire": "S",
                "dam": "D",
                "damsire": "DS",
                "surface_tendency": "芝",
                "distance_tendency": "中長距離",
                "track_condition_tendency": "良",
                "pace_tendency": "average",
                "traits": ["スタミナ"],
                "positive_flags": ["PEDIGREE_DISTANCE_FIT", "PEDIGREE_STAMINA_FIT"],
                "risk_flags": [],
                "overall_comment": "",
            }
        ]
        race_level_analyses = [
            {
                "horse_no": 10,
                "horse_name": "H10",
                "positive_flags": ["HEAD_TO_HEAD_POSITIVE", "UNDERVALUED_GOOD_RUN"],
                "risk_flags": [],
                "head_to_head_summary": "",
                "race_level_summary": "",
                "opponent_context_summary": "",
                "overall_comment": "",
                "adjustment_hint": 0.5,
            },
            {
                "horse_no": 13,
                "horse_name": "H13",
                "positive_flags": ["VALUE_WIN"],
                "risk_flags": [],
                "head_to_head_summary": "",
                "race_level_summary": "",
                "opponent_context_summary": "",
                "overall_comment": "",
                "adjustment_hint": 0.4,
            },
        ]
        pace_analyses = [
            {
                "horse_no": 13,
                "horse_name": "H13",
                "running_style": "差し",
                "early_position_score": 4.0,
                "late_position_score": 7.0,
                "position_stability": "安定",
                "positive_flags": ["PACE_FIT"],
                "risk_flags": [],
                "overall_comment": "",
            }
        ]
        prediction = _prediction_payload(
            "dm1",
            "DeepMissRace",
            "2026-05-23",
            scores,
            deep_analyses,
            pedigree_analyses,
            race_level_analyses,
            pace_analyses,
        )
        (self.predictions_dir / "dm1.json").write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")

        race_data = _race_data_payload(
            "dm1",
            "DeepMissRace",
            "2026-05-23",
            [
                (1, "H1", 2.1, 1),
                (2, "H2", 3.0, 2),
                (3, "H3", 4.5, 3),
                (4, "H4", 6.0, 4),
                (5, "H5", 8.0, 5),
                (6, "H6", 9.5, 6),
                (7, "H7", 4.8, 2),
                (8, "H8", 14.0, 7),
                (9, "H9", 16.0, 8),
                (10, "H10", 18.5, 7),
                (11, "H11", 25.0, 10),
                (12, "H12", 34.0, 11),
                (13, "H13", 55.0, 9),
            ],
        )
        (self.race_data_dir / "dm1.json").write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")

        result = ResultData.model_validate(
            {"race_id": "dm1", "result": {"1st": 10, "2nd": 7, "3rd": 13}, "payouts": []}
        )
        review = Review.model_validate(
            {
                "race_id": "dm1",
                "hit_summary": {
                    "main_mark_top3": False,
                    "marked_horses_top3_count": 0,
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
        (self.results_dir / "dm1.json").write_text(
            json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.reviews_dir / "dm1.json").write_text(
            json.dumps(review.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_low_rank_top3_cases_are_grouped_by_severity(self) -> None:
        report = run_deep_miss_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            scoring_mode="candidate_default",
        )
        self.assertEqual(report["summary"]["reviewed_race_count"], 1)
        self.assertEqual(report["summary"]["total_top3_horses"], 3)
        self.assertEqual(report["summary"]["captured_top3_horses"], 0)
        self.assertEqual(report["summary"]["low_rank_top3_horses"], 3)
        self.assertEqual(report["severity_counts"]["light"], 1)
        self.assertEqual(report["severity_counts"]["moderate"], 1)
        self.assertEqual(report["severity_counts"]["deep"], 1)

        cases = {(case["race_id"], case["horse_no"]): case for case in report["deep_miss_cases"]}
        self.assertEqual(cases[("dm1", 7)]["predicted_rank"], 7)
        self.assertEqual(cases[("dm1", 7)]["severity"], "light")
        self.assertIn("POPULAR_BUT_MISSED", cases[("dm1", 7)]["miss_categories"])
        self.assertIn("DISTANCE_FIT_UNDERESTIMATED", cases[("dm1", 7)]["miss_categories"])

        self.assertEqual(cases[("dm1", 10)]["predicted_rank"], 10)
        self.assertEqual(cases[("dm1", 10)]["severity"], "moderate")
        self.assertIn("RACE_LEVEL_UNDERESTIMATED", cases[("dm1", 10)]["miss_categories"])
        self.assertIn("ODDS_UNDERESTIMATED", cases[("dm1", 10)]["miss_categories"])

        self.assertEqual(cases[("dm1", 13)]["predicted_rank"], 13)
        self.assertEqual(cases[("dm1", 13)]["severity"], "deep")
        self.assertIn("PEDIGREE_FIT_UNDERESTIMATED", cases[("dm1", 13)]["miss_categories"])
        self.assertIn("PACE_FIT_UNDERESTIMATED", cases[("dm1", 13)]["miss_categories"])
        self.assertIn("RISK_OVER_PENALIZED", cases[("dm1", 13)]["miss_categories"])

    def test_rank6_case_is_not_counted_as_deep_miss(self) -> None:
        report = run_deep_miss_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            scoring_mode="candidate_default",
        )
        self.assertNotIn(6, [case["horse_no"] for case in report["deep_miss_cases"]])

    def test_markdown_report_is_generated(self) -> None:
        report = run_deep_miss_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            scoring_mode="candidate_default",
        )
        markdown = generate_deep_miss_markdown(report)
        self.assertIn("# Deep Miss Analysis", markdown)
        self.assertIn("## Severity Breakdown", markdown)
        self.assertIn("## Miss Category Ranking", markdown)
        self.assertIn("## High Priority Deep Misses", markdown)
        self.assertIn("## Race Details", markdown)


if __name__ == "__main__":
    unittest.main()

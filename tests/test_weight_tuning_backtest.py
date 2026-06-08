from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.backtest.backtest_runner import (
    _best_mode_summary,
    generate_weight_tuning_markdown,
    run_backtest_weights,
)
from keiba_llm_agent.backtest.scoring_comparator import score_for_weight_tuning_mode
from keiba_llm_agent.main import run_backtest_weights_command
from keiba_llm_agent.schemas.prediction import HorseScore, PedigreeAdjustment, ScoreAdjustment, ScoreBreakdown
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _horse_score(
    horse_no: int,
    base: float,
    pedigree: float = 0.0,
    race_level: float = 0.0,
    pace: float = 0.0,
    odds: float | None = None,
    popularity: int | None = None,
) -> dict:
    total = round(base + pedigree + race_level + pace, 1)
    return {
        "horse_no": horse_no,
        "horse_name": f"Horse{horse_no}",
        "scores": {
            "recent_form": 7,
            "distance_fit": 7,
            "course_fit": 7,
            "track_condition_fit": 7,
            "jockey_fit": 7,
            "odds_value": 5,
            "risk": -3,
        },
        "base_total_score": base,
        "pedigree_adjustment": {
            "pedigree_bonus": max(pedigree, 0.0),
            "pedigree_penalty": min(pedigree, 0.0),
            "pedigree_adjustment": pedigree,
            "reason": "",
        },
        "race_level_adjustment": {"adjustment": race_level, "reason": ""},
        "pace_adjustment": {"adjustment": pace, "reason": ""},
        "score_breakdown": {
            "base_total_score": base,
            "pedigree_adjustment": pedigree,
            "race_level_adjustment": race_level,
            "pace_adjustment": pace,
            "total_score": total,
        },
        "total_score": total,
        "reason": "reason",
        "odds": odds,
        "popularity": popularity,
    }


def _prediction_payload(
    race_id: str,
    race_date: str,
    scores: list[dict],
    *,
    deep_analyses: list[dict] | None = None,
    pedigree_analyses: list[dict] | None = None,
    race_level_analyses: list[dict] | None = None,
    pace_analyses: list[dict] | None = None,
) -> dict:
    return {
        "race_id": race_id,
        "race_info": {
            "race_id": race_id,
            "race_name": f"Race {race_id}",
            "race_date": race_date,
            "course": "京都",
            "surface": "芝",
            "distance": 2400,
            "track_condition": "良",
            "weather": "晴",
        },
        "marks": {"◎": scores[0]["horse_no"], "○": scores[1]["horse_no"], "▲": scores[2]["horse_no"], "△": scores[3]["horse_no"], "☆": scores[4]["horse_no"]},
        "horse_scores": scores,
        "deep_analyses": deep_analyses or [],
        "pedigree_analyses": pedigree_analyses or [],
        "race_level_analyses": race_level_analyses or [],
        "pace_analyses": pace_analyses or [],
        "scoring_config": {
            "scoring_mode": "candidate_default",
            "pedigree_weight": 0.2,
            "race_level_weight": 1.0,
            "pace_weight": 0.0,
        },
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


class WeightTuningBacktestTests(unittest.TestCase):
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

        race1_scores = [
            _horse_score(1, 36.0, odds=2.5, popularity=1),
            _horse_score(2, 35.5, odds=3.0, popularity=2),
            _horse_score(3, 35.0, odds=5.0, popularity=4),
            _horse_score(4, 34.5, odds=7.0, popularity=5),
            _horse_score(5, 34.0, odds=9.5, popularity=6),
            _horse_score(6, 33.5, race_level=0.4, pace=1.0, odds=8.0, popularity=3),
        ]
        race2_scores = [
            _horse_score(1, 36.0),
            _horse_score(2, 35.5),
            _horse_score(3, 35.0),
            _horse_score(4, 34.5, race_level=1.0, pace=1.0),
            _horse_score(5, 34.0),
            _horse_score(6, 33.5),
        ]
        race3_scores = [
            _horse_score(1, 36.0),
            _horse_score(2, 35.8, pedigree=0.1, race_level=0.1, pace=0.1),
            _horse_score(3, 35.0),
            _horse_score(4, 34.5),
            _horse_score(5, 34.0),
            _horse_score(6, 33.5),
        ]
        race4_scores = [
            _horse_score(1, 36.0),
            _horse_score(2, 35.0, pedigree=0.8),
            _horse_score(3, 34.0),
            _horse_score(4, 33.0),
            _horse_score(5, 32.0),
            _horse_score(6, 31.0),
        ]

        payloads = [
            _prediction_payload(
                "r1",
                "2026-05-17",
                race1_scores,
                deep_analyses=[
                    {
                        "horse_no": 6,
                        "horse_name": "Horse6",
                        "positive_flags": ["RECENT_FORM_STRONG"],
                        "risk_flags": [],
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
                        "horse_name": "Horse6",
                        "positive_flags": ["HEAD_TO_HEAD_POSITIVE"],
                        "risk_flags": [],
                        "head_to_head_summary": "",
                        "race_level_summary": "",
                        "opponent_context_summary": "",
                        "overall_comment": "",
                        "adjustment_hint": 0.4,
                    }
                ],
                pace_analyses=[
                    {
                        "horse_no": 6,
                        "horse_name": "Horse6",
                        "running_style": "差し",
                        "early_position_score": 5.0,
                        "late_position_score": 7.0,
                        "position_stability": "安定",
                        "positive_flags": ["PACE_FIT"],
                        "risk_flags": [],
                        "overall_comment": "",
                    }
                ],
            ),
            _prediction_payload("r2", "2026-05-18", race2_scores),
            _prediction_payload("r3", "2026-05-19", race3_scores),
            _prediction_payload("r4", "2026-05-20", race4_scores),
        ]
        for payload in payloads:
            (self.predictions_dir / f"{payload['race_id']}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        results = {
            "r1": [6, 1, 2],
            "r2": [1, 2, 6],
            "r3": [1, 2, 3],
        }
        for race_id, top3 in results.items():
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
                    "good_points": ["good"],
                    "bad_points": ["bad"],
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

    def test_weight_modes_and_custom_weight_are_applied(self) -> None:
        horse = HorseScore(
            horse_no=1,
            horse_name="Horse1",
            scores=ScoreBreakdown(
                recent_form=7,
                distance_fit=7,
                course_fit=7,
                track_condition_fit=7,
                jockey_fit=7,
                odds_value=5,
                risk=-3,
            ),
            base_total_score=35.0,
            pedigree_adjustment=PedigreeAdjustment(pedigree_adjustment=0.8),
            race_level_adjustment=ScoreAdjustment(adjustment=0.6),
            pace_adjustment=ScoreAdjustment(adjustment=0.4),
            total_score=36.8,
            reason="reason",
        )
        self.assertEqual(score_for_weight_tuning_mode(horse, "conservative_full"), 35.9)
        self.assertEqual(score_for_weight_tuning_mode(horse, "no_pace"), 36.4)
        self.assertEqual(score_for_weight_tuning_mode(horse, "no_race_level"), 36.2)
        self.assertEqual(score_for_weight_tuning_mode(horse, "local_candidate_default"), 36.1)
        self.assertEqual(score_for_weight_tuning_mode(horse, "custom", custom_weights=(0.25, 0.5, 1.0)), 35.9)

    def test_run_backtest_weights_computes_counts_and_heavy_races(self) -> None:
        report = run_backtest_weights(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            pedigree_weight=0.25,
            race_level_weight=0.5,
            pace_weight=1.0,
        )
        self.assertEqual(report["race_count"], 4)
        self.assertEqual(report["reviewed_race_count"], 3)
        self.assertEqual(report["pending_race_count"], 1)
        self.assertEqual(report["findings"]["better_race_counts"]["current_full"], 1)
        self.assertEqual(report["findings"]["worse_race_counts"]["current_full"], 1)
        self.assertEqual(report["findings"]["same_race_counts"]["current_full"], 1)
        self.assertEqual(report["findings"]["pace_or_race_level_heavy_races"], ["r2"])
        self.assertEqual(report["weights"]["pedigree_weight"], 0.25)
        self.assertEqual(report["weights"]["race_level_weight"], 0.5)
        self.assertEqual(report["weights"]["pace_weight"], 1.0)
        self.assertIn("candidate_default", report["modes"])
        self.assertIn("candidate_default_recovered", report["modes"])
        self.assertIn("overall_recommended_mode", report["best_mode_summary"])
        self.assertIn("overall_recommended_reason", report["best_mode_summary"])
        self.assertIn("result missing for race_id=r4", report["warnings"])

        markdown = generate_weight_tuning_markdown(report)
        self.assertIn("# Adjustment Weight Tuning Report", markdown)
        self.assertIn("## Best Mode", markdown)
        self.assertIn("current_full", markdown)
        self.assertIn("recommended reason", markdown)

    def test_run_backtest_weights_can_filter_local_scope(self) -> None:
        local_payload = _prediction_payload(
            "202644050101",
            "2026-05-21",
            [
                _horse_score(1, 36.0),
                _horse_score(2, 35.5),
                _horse_score(3, 35.0),
                _horse_score(4, 34.5),
                _horse_score(5, 34.0),
            ],
        )
        local_payload["race_info"]["course"] = "大井"
        local_payload["race_info"]["surface"] = "ダート"
        (self.predictions_dir / "202644050101.json").write_text(
            json.dumps(local_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = run_backtest_weights(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            scope_key="local",
        )

        self.assertEqual(report["scope_key"], "local")
        self.assertEqual(report["race_count"], 1)
        self.assertEqual(report["race_details"][0]["race_id"], "202644050101")
        self.assertIn("local_candidate_default", report["modes"])
        self.assertIn("local_candidate_default_recovered", report["modes"])

    def test_command_saves_weight_tuning_outputs(self) -> None:
        with (
            patch("keiba_llm_agent.main.DEFAULT_PREDICTIONS_DIR", self.predictions_dir),
            patch("keiba_llm_agent.main.DEFAULT_RESULTS_DIR", self.results_dir),
            patch("keiba_llm_agent.main.DEFAULT_REVIEWS_DIR", self.reviews_dir),
            patch("keiba_llm_agent.main.DEFAULT_BACKTESTS_DIR", self.backtests_dir),
        ):
            result = run_backtest_weights_command(
                "2026-05-01",
                "2026-05-31",
                pedigree_weight=0.5,
                race_level_weight=0.5,
                pace_weight=0.5,
            )
        self.assertTrue(Path(result["json_path"]).exists())
        self.assertTrue(Path(result["md_path"]).exists())
        self.assertEqual(result["reviewed_race_count"], 3)
        self.assertEqual(result["pending_race_count"], 1)

    def test_mode_subset_does_not_crash_when_current_full_dependencies_are_missing(self) -> None:
        report = run_backtest_weights(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            modes=["base_only"],
        )
        self.assertEqual(list(report["modes"].keys()), ["base_only"])
        self.assertEqual(report["findings"]["race_level_improved_races"], [])
        self.assertEqual(report["findings"]["pace_improved_races"], [])
        self.assertEqual(report["findings"]["pace_or_race_level_heavy_races"], [])

    def test_custom_weights_use_candidate_default_for_unspecified_values(self) -> None:
        report = run_backtest_weights(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            pedigree_weight=0.25,
        )
        self.assertEqual(report["weights"]["pedigree_weight"], 0.25)
        self.assertEqual(report["weights"]["race_level_weight"], 1.0)
        self.assertEqual(report["weights"]["pace_weight"], 0.0)
        self.assertIn("custom", report["modes"])

    def test_recommended_mode_does_not_auto_pick_current_full_when_worse_exists(self) -> None:
        summary = _best_mode_summary(
            {
                "current_full": {
                    "main_mark_top3_rate": 0.462,
                    "main_mark_win_rate": 0.231,
                    "avg_marked_top3_count": 1.31,
                    "marked_top3_rate": 0.437,
                    "top5_winner_rate": 0.615,
                    "avg_top5_top3_count": 1.69,
                    "avg_rank_of_winner": 2.6,
                    "avg_rank_of_top3_horses": 3.2,
                    "better_race_count": 1,
                    "worse_race_count": 1,
                    "same_race_count": 11,
                },
                "race_level_only": {
                    "main_mark_top3_rate": 0.385,
                    "main_mark_win_rate": 0.154,
                    "avg_marked_top3_count": 1.38,
                    "marked_top3_rate": 0.46,
                    "top5_winner_rate": 0.692,
                    "avg_top5_top3_count": 1.77,
                    "avg_rank_of_winner": 2.3,
                    "avg_rank_of_top3_horses": 3.0,
                    "better_race_count": 1,
                    "worse_race_count": 0,
                    "same_race_count": 12,
                },
            }
        )
        self.assertEqual(summary["best_main_mark_top3_rate"], "current_full")
        self.assertEqual(summary["best_avg_marked_top3_count"], "race_level_only")
        self.assertEqual(summary["best_top5_winner_rate"], "race_level_only")
        self.assertEqual(summary["overall_recommended_mode"], "race_level_only")
        self.assertIn("worse=0", summary["overall_recommended_reason"])

    def test_accuracy_recommended_mode_prefers_candidate_default_recovered_over_base_only(self) -> None:
        summary = _best_mode_summary(
            {
                "base_only": {
                    "main_mark_top3_rate": 0.452,
                    "main_mark_win_rate": 0.178,
                    "avg_marked_top3_count": 1.84,
                    "marked_top3_rate": 0.613,
                    "top5_winner_rate": 0.630,
                    "avg_top5_top3_count": 1.84,
                    "avg_rank_of_winner": 4.9,
                    "avg_rank_of_top3_horses": 5.1,
                    "better_race_count": 0,
                    "worse_race_count": 0,
                    "same_race_count": 73,
                },
                "candidate_default_recovered": {
                    "main_mark_top3_rate": 0.466,
                    "main_mark_win_rate": 0.192,
                    "avg_marked_top3_count": 1.96,
                    "marked_top3_rate": 0.653,
                    "top5_winner_rate": 0.658,
                    "avg_top5_top3_count": 1.96,
                    "avg_rank_of_winner": 4.9,
                    "avg_rank_of_top3_horses": 5.08,
                    "better_race_count": 10,
                    "worse_race_count": 1,
                    "same_race_count": 62,
                },
                "candidate_default": {
                    "main_mark_top3_rate": 0.466,
                    "main_mark_win_rate": 0.192,
                    "avg_marked_top3_count": 1.92,
                    "marked_top3_rate": 0.639,
                    "top5_winner_rate": 0.658,
                    "avg_top5_top3_count": 1.92,
                    "avg_rank_of_winner": 4.9,
                    "avg_rank_of_top3_horses": 5.09,
                    "better_race_count": 10,
                    "worse_race_count": 3,
                    "same_race_count": 60,
                },
            }
        )
        self.assertEqual(summary["safest_baseline"], "base_only")
        self.assertEqual(summary["accuracy_recommended_mode"], "candidate_default_recovered")
        self.assertEqual(summary["overall_recommended_mode"], "candidate_default_recovered")
        self.assertIn("1.84", summary["overall_recommended_reason"])
        self.assertIn("1.96", summary["overall_recommended_reason"])
        self.assertIn("63.0%", summary["overall_recommended_reason"])
        self.assertIn("65.8%", summary["overall_recommended_reason"])
        self.assertIn("better=10 / worse=1", summary["overall_recommended_reason"])

    def test_candidate_default_recovered_changes_marks_and_exposes_recovery_details(self) -> None:
        report = run_backtest_weights(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            modes=["candidate_default", "candidate_default_recovered", "base_only"],
        )
        race1_detail = next(detail for detail in report["race_details"] if detail["race_id"] == "r1")
        self.assertNotEqual(race1_detail["candidate_default_marks"], race1_detail["candidate_default_recovered_marks"])
        recovered_top5 = {
            horse_no
            for horse_no in race1_detail["candidate_default_recovered_marks"].values()
            if horse_no > 0
        }
        self.assertIn(6, recovered_top5)
        self.assertTrue(race1_detail["recovery_applied"])
        self.assertTrue(race1_detail["recovery_cases"])
        self.assertEqual(race1_detail["recovered_horse_no"], 6)
        self.assertEqual(race1_detail["recovered_from_rank"], 6)
        self.assertEqual(race1_detail["recovered_to_rank"], 5)
        self.assertGreater(
            report["modes"]["candidate_default_recovered"]["avg_marked_top3_count"],
            report["modes"]["candidate_default"]["avg_marked_top3_count"],
        )
        markdown = generate_weight_tuning_markdown(report)
        self.assertIn("safest baseline", markdown)
        self.assertIn("accuracy recommended", markdown)

    def test_base_only_is_unchanged_even_if_recovery_is_enabled(self) -> None:
        normal = run_backtest_weights(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            modes=["base_only", "candidate_default_recovered"],
            enable_borderline_recovery=False,
        )
        enabled = run_backtest_weights(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            modes=["base_only", "candidate_default_recovered"],
            enable_borderline_recovery=True,
        )
        self.assertEqual(normal["modes"]["base_only"], enabled["modes"]["base_only"])


if __name__ == "__main__":
    unittest.main()

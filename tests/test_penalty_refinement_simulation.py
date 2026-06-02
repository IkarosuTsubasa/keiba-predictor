from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.error_analysis.penalty_refinement_simulator import (
    generate_penalty_refinement_simulation_markdown,
    run_penalty_refinement_simulation,
)
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _horse_score(
    horse_no: int,
    horse_name: str,
    base: float,
    *,
    race_level_adjustment: float = 0.0,
    risk: int = -2,
    odds: float | None = None,
    popularity: int | None = None,
) -> dict:
    total = round(base + race_level_adjustment, 1)
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
        "race_level_adjustment": {"adjustment": race_level_adjustment, "reason": ""},
        "pace_adjustment": {"adjustment": 0.0, "reason": ""},
        "score_breakdown": {
            "base_total_score": base,
            "pedigree_adjustment_raw": 0.0,
            "pedigree_weight": 0.2,
            "pedigree_adjustment_weighted": 0.0,
            "race_level_adjustment_raw": race_level_adjustment,
            "race_level_weight": 1.0,
            "race_level_adjustment_weighted": race_level_adjustment,
            "pace_adjustment_raw": 0.0,
            "pace_weight": 0.0,
            "pace_adjustment_weighted": 0.0,
            "borderline_recovery_bonus": 0.0,
            "total_score": total,
            "total_score_after_recovery": total,
        },
        "odds": odds,
        "popularity": popularity,
        "total_score": total,
        "reason": "reason",
    }


def _deep_analysis(horse_no: int, horse_name: str, positives: list[str], risks: list[str]) -> dict:
    return {
        "horse_no": horse_no,
        "horse_name": horse_name,
        "positive_flags": positives,
        "risk_flags": risks,
        "recent_form_summary": "",
        "distance_analysis": "",
        "course_analysis": "",
        "track_condition_analysis": "",
        "jockey_analysis": "",
        "odds_analysis": "",
        "overall_comment": "",
    }


def _race_level_analysis(horse_no: int, horse_name: str, positives: list[str], risks: list[str], adjustment_hint: float) -> dict:
    return {
        "horse_no": horse_no,
        "horse_name": horse_name,
        "positive_flags": positives,
        "risk_flags": risks,
        "head_to_head_summary": "",
        "race_level_summary": "",
        "opponent_context_summary": "",
        "overall_comment": "",
        "adjustment_hint": adjustment_hint,
    }


def _pedigree_analysis(horse_no: int, horse_name: str, positives: list[str]) -> dict:
    return {
        "horse_no": horse_no,
        "horse_name": horse_name,
        "sire": "",
        "dam": "",
        "damsire": "",
        "surface_tendency": "",
        "distance_tendency": "",
        "track_condition_tendency": "",
        "pace_tendency": "",
        "positive_flags": positives,
        "risk_flags": [],
        "overall_comment": "",
    }


def _pace_analysis(horse_no: int, horse_name: str, positives: list[str], risks: list[str] | None = None) -> dict:
    return {
        "horse_no": horse_no,
        "horse_name": horse_name,
        "running_style": "差し",
        "early_position_score": 5.0,
        "late_position_score": 7.0,
        "position_stability": "stable",
        "positive_flags": positives,
        "risk_flags": risks or [],
        "overall_comment": "",
    }


def _prediction_payload(
    race_id: str,
    race_date: str,
    distance: int,
    horse_scores: list[dict],
    *,
    deep_analyses: list[dict],
    pedigree_analyses: list[dict] | None = None,
    race_level_analyses: list[dict] | None = None,
    pace_analyses: list[dict] | None = None,
) -> dict:
    return {
        "race_id": race_id,
        "race_info": {
            "race_id": race_id,
            "race_name": race_id,
            "race_date": race_date,
            "course": "東京",
            "surface": "芝",
            "distance": distance,
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
        "pedigree_analyses": pedigree_analyses or [],
        "race_level_analyses": race_level_analyses or [],
        "pace_analyses": pace_analyses or [],
        "strategy": {
            "bet_decision": "BET",
            "confidence": "medium",
            "participation_level": "light",
            "reason_codes": ["RACE_LEVEL_USED"],
            "reason": "reason",
        },
        "borderline_recovery_result": {"recovery_applied": False, "recovery_cases": []},
    }


def _race_data_payload(race_id: str, race_date: str, distance: int, horses: list[dict]) -> dict:
    return {
        "race_info": {
            "race_id": race_id,
            "race_name": race_id,
            "race_date": race_date,
            "course": "東京",
            "surface": "芝",
            "distance": distance,
            "track_condition": "良",
            "weather": "晴",
        },
        "horses": [
            {
                "horse_no": horse["horse_no"],
                "frame_no": horse["horse_no"],
                "horse_name": horse["horse_name"],
                "odds": horse["odds"],
                "popularity": horse["popularity"],
                "recent_runs": [],
            }
            for horse in horses
        ],
    }


def _finish_order(entries: list[tuple[int, int, str]]) -> list[dict]:
    return [{"finish": finish, "horse_no": horse_no, "horse_name": horse_name} for finish, horse_no, horse_name in entries]


class PenaltyRefinementSimulationTests(unittest.TestCase):
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

        self._write_head_to_head_better()
        self._write_head_to_head_worse()
        self._write_distance_unknown_success()
        self.original_prediction_text = (self.predictions_dir / "r1.json").read_text(encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_review(self, race_id: str) -> None:
        review = Review.model_validate(
            {
                "race_id": race_id,
                "hit_summary": {
                    "main_mark_top3": False,
                    "marked_horses_top3_count": 1,
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
        (self.reviews_dir / f"{race_id}.json").write_text(json.dumps(review.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_case(self, race_id: str, race_date: str, distance: int, scores: list[dict], prediction: dict, result_top3: tuple[int, int, int]) -> None:
        (self.predictions_dir / f"{race_id}.json").write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.race_data_dir / f"{race_id}.json").write_text(json.dumps(_race_data_payload(race_id, race_date, distance, scores), ensure_ascii=False, indent=2), encoding="utf-8")
        result = ResultData.model_validate(
            {
                "race_id": race_id,
                "result": {"1st": result_top3[0], "2nd": result_top3[1], "3rd": result_top3[2]},
                "payouts": [],
                "finish_order": _finish_order(
                    [
                        (1, result_top3[0], f"H{result_top3[0]}"),
                        (2, result_top3[1], f"H{result_top3[1]}"),
                        (3, result_top3[2], f"H{result_top3[2]}"),
                    ]
                ),
            }
        )
        (self.results_dir / f"{race_id}.json").write_text(json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_review(race_id)

    def _base_scores(self, offset: int = 0) -> list[dict]:
        return [
            _horse_score(offset + 1, f"H{offset + 1}", 50.0, odds=2.0, popularity=1),
            _horse_score(offset + 2, f"H{offset + 2}", 49.0, odds=3.0, popularity=2),
            _horse_score(offset + 3, f"H{offset + 3}", 48.0, odds=4.0, popularity=3),
            _horse_score(offset + 4, f"H{offset + 4}", 47.0, odds=5.0, popularity=4),
            _horse_score(offset + 5, f"H{offset + 5}", 46.0, odds=6.0, popularity=5),
            _horse_score(offset + 6, f"H{offset + 6}", 45.9, odds=8.0, popularity=6),
        ]

    def _write_head_to_head_better(self) -> None:
        scores = self._base_scores()
        scores.append(_horse_score(7, "H7", 46.3, race_level_adjustment=-0.5, odds=9.0, popularity=4))
        prediction = _prediction_payload(
            "r1",
            "2026-05-20",
            1800,
            scores,
            deep_analyses=[_deep_analysis(7, "H7", ["RECENT_FORM_STABLE", "DISTANCE_FIT", "COURSE_FIT", "STABLE_PERFORMER"], [])],
            pedigree_analyses=[_pedigree_analysis(7, "H7", ["PEDIGREE_DISTANCE_FIT"])],
            race_level_analyses=[_race_level_analysis(7, "H7", ["LARGE_FIELD_GOOD_RUN"], ["HEAD_TO_HEAD_NEGATIVE"], -0.5)],
            pace_analyses=[_pace_analysis(7, "H7", ["PACE_FIT"])],
        )
        self._write_case("r1", "2026-05-20", 1800, scores, prediction, (7, 1, 2))

    def _write_head_to_head_worse(self) -> None:
        scores = self._base_scores(10)
        scores.append(_horse_score(17, "H17", 46.3, race_level_adjustment=-0.5, odds=9.0, popularity=4))
        prediction = _prediction_payload(
            "r2",
            "2026-05-21",
            1800,
            scores,
            deep_analyses=[_deep_analysis(17, "H17", ["RECENT_FORM_STABLE", "DISTANCE_FIT", "COURSE_FIT", "STABLE_PERFORMER"], [])],
            pedigree_analyses=[_pedigree_analysis(17, "H17", ["PEDIGREE_DISTANCE_FIT"])],
            race_level_analyses=[_race_level_analysis(17, "H17", ["LARGE_FIELD_GOOD_RUN"], ["HEAD_TO_HEAD_NEGATIVE"], -0.5)],
            pace_analyses=[_pace_analysis(17, "H17", ["PACE_FIT"])],
        )
        self._write_case("r2", "2026-05-21", 1800, scores, prediction, (11, 12, 15))

    def _write_distance_unknown_success(self) -> None:
        scores = self._base_scores(20)
        scores.append(_horse_score(27, "H27", 45.6, odds=12.0, popularity=6))
        prediction = _prediction_payload(
            "r3",
            "2026-05-22",
            2400,
            scores,
            deep_analyses=[_deep_analysis(27, "H27", ["RECENT_FORM_STABLE", "COURSE_FIT"], ["DISTANCE_UNKNOWN"])],
            pedigree_analyses=[_pedigree_analysis(27, "H27", ["PEDIGREE_STAMINA_FIT", "PEDIGREE_DISTANCE_FIT"])],
            race_level_analyses=[_race_level_analysis(27, "H27", ["LARGE_FIELD_GOOD_RUN"], [], 0.0)],
            pace_analyses=[_pace_analysis(27, "H27", ["PACE_FIT"])],
        )
        self._write_case("r3", "2026-05-22", 2400, scores, prediction, (27, 21, 22))

    def _run(self) -> dict[str, object]:
        return run_penalty_refinement_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            baseline_mode="candidate_default_recovered",
        )

    def test_head_to_head_negative_cap_can_change_top5(self) -> None:
        report = self._run()
        rows = report["rule_results"]["head_to_head_negative_cap"]["rows"]
        row = next(row for row in rows if row["race_id"] == "r1")
        self.assertTrue(row["candidate_enters_top5"])
        self.assertEqual(row["candidate_new_rank"], 5)
        self.assertEqual(row["replaced_horse_no"], 5)
        self.assertEqual(row["delta"], 1)

    def test_worse_and_replaced_top3_are_counted(self) -> None:
        report = self._run()
        result = report["rule_results"]["head_to_head_negative_cap"]
        self.assertEqual(result["better_race_count"], 1)
        self.assertEqual(result["worse_race_count"], 1)
        self.assertEqual(result["replaced_top3_count"], 1)

    def test_distance_unknown_stamina_exception_recovers_long_distance_candidate(self) -> None:
        report = self._run()
        rows = report["rule_results"]["distance_unknown_stamina_exception"]["rows"]
        row = next(row for row in rows if row["race_id"] == "r3")
        self.assertTrue(row["candidate_enters_top5"])
        self.assertEqual(row["candidate_horse_no"], 27)
        self.assertEqual(row["delta"], 1)

    def test_prediction_files_are_not_modified(self) -> None:
        self._run()
        current_prediction_text = (self.predictions_dir / "r1.json").read_text(encoding="utf-8")
        self.assertEqual(self.original_prediction_text, current_prediction_text)

    def test_markdown_report_is_generated(self) -> None:
        report = self._run()
        markdown = generate_penalty_refinement_simulation_markdown(report)
        self.assertIn("# Penalty Refinement Simulation Report", markdown)
        self.assertIn("## Rule Comparison", markdown)
        self.assertIn("head_to_head_negative_cap", markdown)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.error_analysis.score_recalibration_simulator import (
    generate_score_recalibration_simulation_markdown,
    run_score_recalibration_simulation,
)
from keiba_llm_agent.main import run_score_recalibration_simulation_command
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _horse_score(
    horse_no: int,
    horse_name: str,
    total: float,
    *,
    risk: int = -2,
) -> dict:
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
        "base_total_score": total,
        "pedigree_adjustment": {
            "pedigree_bonus": 0.0,
            "pedigree_penalty": 0.0,
            "pedigree_adjustment": 0.0,
            "reason": "",
        },
        "race_level_adjustment": {"adjustment": 0.0, "reason": ""},
        "pace_adjustment": {"adjustment": 0.0, "reason": ""},
        "score_breakdown": {
            "base_total_score": total,
            "pedigree_adjustment_raw": 0.0,
            "pedigree_weight": 0.2,
            "pedigree_adjustment_weighted": 0.0,
            "race_level_adjustment_raw": 0.0,
            "race_level_weight": 1.0,
            "race_level_adjustment_weighted": 0.0,
            "pace_adjustment_raw": 0.0,
            "pace_weight": 0.0,
            "pace_adjustment_weighted": 0.0,
            "borderline_recovery_bonus": 0.0,
            "total_score": total,
            "total_score_after_recovery": total,
        },
        "odds": 5.0 + horse_no,
        "popularity": horse_no,
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


def _race_level_analysis(horse_no: int, horse_name: str, positives: list[str]) -> dict:
    return {
        "horse_no": horse_no,
        "horse_name": horse_name,
        "positive_flags": positives,
        "risk_flags": [],
        "head_to_head_summary": "",
        "race_level_summary": "",
        "opponent_context_summary": "",
        "overall_comment": "",
        "adjustment_hint": 0.5,
    }


def _pace_analysis(horse_no: int, horse_name: str, positives: list[str]) -> dict:
    return {
        "horse_no": horse_no,
        "horse_name": horse_name,
        "running_style": "差し",
        "early_position_score": 5.0,
        "late_position_score": 7.0,
        "position_stability": "安定",
        "positive_flags": positives,
        "risk_flags": [],
        "overall_comment": "",
    }


def _prediction_payload(
    race_id: str,
    race_date: str,
    horse_scores: list[dict],
    *,
    deep_analyses: list[dict],
    pedigree_analyses: list[dict],
    race_level_analyses: list[dict],
    pace_analyses: list[dict],
) -> dict:
    return {
        "race_id": race_id,
        "race_info": {
            "race_id": race_id,
            "race_name": race_id,
            "race_date": race_date,
            "course": "東京",
            "surface": "芝",
            "distance": 1600,
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
            "bet_decision": "SKIP",
            "confidence": "low",
            "participation_level": "none",
            "reason_codes": [],
            "reason": "reason",
        },
        "borderline_recovery_result": {"recovery_applied": False, "recovery_cases": []},
    }


def _result_payload(race_id: str, top3: tuple[int, int, int]) -> dict:
    result = ResultData.model_validate(
        {
            "race_id": race_id,
            "result": {"1st": top3[0], "2nd": top3[1], "3rd": top3[2]},
            "payouts": [],
            "finish_order": [
                {"finish": 1, "horse_no": top3[0], "horse_name": f"H{top3[0]}"},
                {"finish": 2, "horse_no": top3[1], "horse_name": f"H{top3[1]}"},
                {"finish": 3, "horse_no": top3[2], "horse_name": f"H{top3[2]}"},
            ],
        }
    )
    return result.model_dump(by_alias=True)


def _review_payload(race_id: str) -> dict:
    review = Review.model_validate(
        {
            "race_id": race_id,
            "hit_summary": {
                "main_mark_top3": False,
                "marked_horses_top3_count": 2,
                "bet_hit": False,
                "roi": 0.0,
                "total_stake": 0,
                "total_return": 0,
            },
            "bet_results": [],
            "good_points": [],
            "bad_points": [],
            "lessons": [],
        }
    )
    return review.model_dump()


class ScoreRecalibrationSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.predictions_dir = self.root / "predictions"
        self.results_dir = self.root / "results"
        self.reviews_dir = self.root / "reviews"
        for path in (self.predictions_dir, self.results_dir, self.reviews_dir):
            path.mkdir()
        self._write_positive_stack_case()
        self._write_unknown_softening_case()
        self._write_hard_negative_case()
        self.original_prediction_text = (self.predictions_dir / "r1.json").read_text(encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _base_scores(self, offset: int = 0, rank6_risk: int = -2) -> list[dict]:
        return [
            _horse_score(offset + 1, f"H{offset + 1}", 50.0),
            _horse_score(offset + 2, f"H{offset + 2}", 49.0),
            _horse_score(offset + 3, f"H{offset + 3}", 48.0),
            _horse_score(offset + 4, f"H{offset + 4}", 47.0),
            _horse_score(offset + 5, f"H{offset + 5}", 46.0),
            _horse_score(offset + 6, f"H{offset + 6}", 45.5, risk=rank6_risk),
        ]

    def _write_case(self, race_id: str, race_date: str, scores: list[dict], prediction: dict, top3: tuple[int, int, int]) -> None:
        self._write_json(self.predictions_dir / f"{race_id}.json", prediction)
        self._write_json(self.results_dir / f"{race_id}.json", _result_payload(race_id, top3))
        self._write_json(self.reviews_dir / f"{race_id}.json", _review_payload(race_id))

    def _write_positive_stack_case(self) -> None:
        scores = self._base_scores()
        prediction = _prediction_payload(
            "r1",
            "2026-05-20",
            scores,
            deep_analyses=[_deep_analysis(6, "H6", ["RECENT_FORM_STABLE", "DISTANCE_FIT", "COURSE_FIT"], [])],
            pedigree_analyses=[_pedigree_analysis(6, "H6", ["PEDIGREE_DISTANCE_FIT"])],
            race_level_analyses=[_race_level_analysis(6, "H6", ["HEAD_TO_HEAD_POSITIVE"])],
            pace_analyses=[_pace_analysis(6, "H6", ["PACE_FIT"])],
        )
        self._write_case("r1", "2026-05-20", scores, prediction, (6, 1, 2))

    def _write_unknown_softening_case(self) -> None:
        scores = self._base_scores(10)
        prediction = _prediction_payload(
            "r2",
            "2026-05-21",
            scores,
            deep_analyses=[_deep_analysis(16, "H16", ["RECENT_FORM_STABLE"], ["DISTANCE_UNKNOWN", "COURSE_UNKNOWN"])],
            pedigree_analyses=[_pedigree_analysis(16, "H16", ["PEDIGREE_STAMINA_FIT"])],
            race_level_analyses=[_race_level_analysis(16, "H16", ["LARGE_FIELD_GOOD_RUN"])],
            pace_analyses=[_pace_analysis(16, "H16", ["PACE_FIT"])],
        )
        self._write_case("r2", "2026-05-21", scores, prediction, (16, 11, 12))

    def _write_hard_negative_case(self) -> None:
        scores = self._base_scores(20, rank6_risk=-7)
        prediction = _prediction_payload(
            "r3",
            "2026-05-22",
            scores,
            deep_analyses=[_deep_analysis(26, "H26", ["RECENT_FORM_STABLE", "DISTANCE_FIT", "COURSE_FIT"], ["DATA_INCOMPLETE"])],
            pedigree_analyses=[_pedigree_analysis(26, "H26", ["PEDIGREE_DISTANCE_FIT"])],
            race_level_analyses=[_race_level_analysis(26, "H26", ["HEAD_TO_HEAD_POSITIVE"])],
            pace_analyses=[_pace_analysis(26, "H26", ["PACE_FIT"])],
        )
        self._write_case("r3", "2026-05-22", scores, prediction, (26, 21, 22))

    def _run(self) -> dict:
        return run_score_recalibration_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            baseline_mode="candidate_default",
        )

    def test_positive_stack_boost_can_lift_underestimated_horse_into_top5(self) -> None:
        report = self._run()
        rows = report["rule_results"]["positive_stack_boost"]["rows"]
        row = next(row for row in rows if row["race_id"] == "r1")
        self.assertIn(6, row["simulated_top5"])
        self.assertEqual(row["delta"], 1)

    def test_unknown_softening_uses_unknown_as_neutral_when_support_exists(self) -> None:
        report = self._run()
        rows = report["rule_results"]["unknown_softening"]["rows"]
        row = next(row for row in rows if row["race_id"] == "r2")
        self.assertIn(16, row["simulated_top5"])
        self.assertIn("UNKNOWN_SOFTEN:DISTANCE_UNKNOWN", row["entered_cases"][0]["reasons"])

    def test_hard_negative_case_is_not_lifted_by_positive_stack(self) -> None:
        report = self._run()
        rows = report["rule_results"]["positive_stack_boost"]["rows"]
        self.assertNotIn("r3", [row["race_id"] for row in rows])

    def test_prediction_files_are_not_modified(self) -> None:
        self._run()
        self.assertEqual(self.original_prediction_text, (self.predictions_dir / "r1.json").read_text(encoding="utf-8"))

    def test_markdown_report_is_generated(self) -> None:
        report = self._run()
        markdown = generate_score_recalibration_simulation_markdown(report)
        self.assertIn("# Score Recalibration Simulation Report", markdown)
        self.assertIn("## Rule Comparison", markdown)


class ScoreRecalibrationCommandTests(unittest.TestCase):
    def test_command_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            predictions_dir = root / "predictions"
            results_dir = root / "results"
            reviews_dir = root / "reviews"
            output_dir = root / "error_analysis"
            for path in (predictions_dir, results_dir, reviews_dir):
                path.mkdir()
            scores = [
                _horse_score(1, "H1", 50.0),
                _horse_score(2, "H2", 49.0),
                _horse_score(3, "H3", 48.0),
                _horse_score(4, "H4", 47.0),
                _horse_score(5, "H5", 46.0),
                _horse_score(6, "H6", 45.5),
            ]
            prediction = _prediction_payload(
                "r1",
                "2026-05-20",
                scores,
                deep_analyses=[_deep_analysis(6, "H6", ["RECENT_FORM_STABLE", "DISTANCE_FIT", "COURSE_FIT"], [])],
                pedigree_analyses=[_pedigree_analysis(6, "H6", ["PEDIGREE_DISTANCE_FIT"])],
                race_level_analyses=[_race_level_analysis(6, "H6", ["HEAD_TO_HEAD_POSITIVE"])],
                pace_analyses=[_pace_analysis(6, "H6", ["PACE_FIT"])],
            )
            (predictions_dir / "r1.json").write_text(json.dumps(prediction, ensure_ascii=False), encoding="utf-8")
            (results_dir / "r1.json").write_text(json.dumps(_result_payload("r1", (6, 1, 2)), ensure_ascii=False), encoding="utf-8")
            (reviews_dir / "r1.json").write_text(json.dumps(_review_payload("r1"), ensure_ascii=False), encoding="utf-8")

            import keiba_llm_agent.main as main_module

            original_dirs = (
                main_module.DEFAULT_PREDICTIONS_DIR,
                main_module.DEFAULT_RESULTS_DIR,
                main_module.DEFAULT_REVIEWS_DIR,
                main_module.DEFAULT_ERROR_ANALYSIS_DIR,
            )
            try:
                main_module.DEFAULT_PREDICTIONS_DIR = predictions_dir
                main_module.DEFAULT_RESULTS_DIR = results_dir
                main_module.DEFAULT_REVIEWS_DIR = reviews_dir
                main_module.DEFAULT_ERROR_ANALYSIS_DIR = output_dir
                result = run_score_recalibration_simulation_command("2026-05-01", "2026-05-31")
            finally:
                (
                    main_module.DEFAULT_PREDICTIONS_DIR,
                    main_module.DEFAULT_RESULTS_DIR,
                    main_module.DEFAULT_REVIEWS_DIR,
                    main_module.DEFAULT_ERROR_ANALYSIS_DIR,
                ) = original_dirs

            self.assertTrue(Path(result["json_path"]).exists())
            self.assertTrue(Path(result["md_path"]).exists())
            self.assertEqual(result["reviewed_race_count"], 1)


if __name__ == "__main__":
    unittest.main()

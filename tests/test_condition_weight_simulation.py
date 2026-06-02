from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.error_analysis.condition_weight_simulator import (
    generate_condition_weight_simulation_markdown,
    run_condition_weight_simulation,
)
from keiba_llm_agent.main import run_condition_weight_simulation_command
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _horse_score(
    horse_no: int,
    horse_name: str,
    base: float,
    *,
    race_level: float = 0.0,
    risk: int = -3,
) -> dict:
    total = round(base + race_level, 1)
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
        "race_level_adjustment": {"adjustment": race_level, "reason": ""},
        "pace_adjustment": {"adjustment": 0.0, "reason": ""},
        "score_breakdown": {
            "base_total_score": base,
            "pedigree_adjustment_raw": 0.0,
            "pedigree_weight": 0.2,
            "pedigree_adjustment_weighted": 0.0,
            "race_level_adjustment_raw": race_level,
            "race_level_weight": 1.0,
            "race_level_adjustment_weighted": race_level,
            "pace_adjustment_raw": 0.0,
            "pace_weight": 0.0,
            "pace_adjustment_weighted": 0.0,
            "borderline_recovery_bonus": 0.0,
            "total_score": total,
            "total_score_after_recovery": total,
        },
        "odds": 10.0 + horse_no,
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
        "adjustment_hint": 1.0,
    }


def _prediction_payload(race_id: str, race_date: str, surface: str, scores: list[dict], *, deep_analyses: list[dict] | None = None, race_level_analyses: list[dict] | None = None) -> dict:
    return {
        "race_id": race_id,
        "race_info": {
            "race_id": race_id,
            "race_name": race_id,
            "race_date": race_date,
            "course": "東京",
            "surface": surface,
            "distance": 1800,
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
        "marks": {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
        "horse_scores": scores,
        "bets": [],
        "summary": "summary",
        "risks": [],
        "used_lessons": [],
        "deep_analyses": deep_analyses or [],
        "pedigree_analyses": [],
        "race_level_analyses": race_level_analyses or [],
        "pace_analyses": [],
        "strategy": {
            "bet_decision": "SKIP",
            "confidence": "low",
            "participation_level": "none",
            "reason_codes": [],
            "reason": "reason",
        },
        "borderline_recovery_result": {"recovery_applied": False, "recovery_cases": []},
    }


def _race_data_payload(race_id: str, race_date: str, surface: str, scores: list[dict]) -> dict:
    return {
        "race_info": {
            "race_id": race_id,
            "race_name": race_id,
            "race_date": race_date,
            "course": "東京",
            "surface": surface,
            "distance": 1800,
            "track_condition": "良",
            "weather": "晴",
        },
        "horses": [
            {
                "horse_no": score["horse_no"],
                "frame_no": score["horse_no"],
                "horse_name": score["horse_name"],
                "jockey": "騎手",
                "carried_weight": 56.0,
                "odds": score["odds"],
                "popularity": score["popularity"],
                "recent_runs": [],
            }
            for score in scores
        ],
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


class ConditionWeightSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.predictions_dir = self.root / "predictions"
        self.results_dir = self.root / "results"
        self.reviews_dir = self.root / "reviews"
        self.race_data_dir = self.root / "race_data"
        for path in (self.predictions_dir, self.results_dir, self.reviews_dir, self.race_data_dir):
            path.mkdir()
        self._write_turf_race_level_case()
        self._write_strict_recovery_risk_case()
        self.original_prediction_text = (self.predictions_dir / "r1.json").read_text(encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_case(self, race_id: str, race_date: str, surface: str, scores: list[dict], prediction: dict, top3: tuple[int, int, int]) -> None:
        self._write_json(self.predictions_dir / f"{race_id}.json", prediction)
        self._write_json(self.race_data_dir / f"{race_id}.json", _race_data_payload(race_id, race_date, surface, scores))
        self._write_json(self.results_dir / f"{race_id}.json", _result_payload(race_id, top3))
        self._write_json(self.reviews_dir / f"{race_id}.json", _review_payload(race_id))

    def _write_turf_race_level_case(self) -> None:
        scores = [
            _horse_score(1, "H1", 50.0),
            _horse_score(2, "H2", 49.0),
            _horse_score(3, "H3", 48.0),
            _horse_score(4, "H4", 47.0),
            _horse_score(5, "H5", 46.0),
            _horse_score(6, "H6", 44.9, race_level=1.0),
        ]
        prediction = _prediction_payload("r1", "2026-05-20", "芝", scores)
        self._write_case("r1", "2026-05-20", "芝", scores, prediction, (6, 1, 2))

    def _write_strict_recovery_risk_case(self) -> None:
        scores = [
            _horse_score(11, "H11", 50.0),
            _horse_score(12, "H12", 49.0),
            _horse_score(13, "H13", 48.0),
            _horse_score(14, "H14", 47.0),
            _horse_score(15, "H15", 46.0),
            _horse_score(16, "H16", 45.9, risk=-7),
        ]
        prediction = _prediction_payload(
            "r2",
            "2026-05-21",
            "芝",
            scores,
            deep_analyses=[_deep_analysis(16, "H16", ["RECENT_FORM_STABLE", "DISTANCE_FIT", "COURSE_FIT"], [])],
            race_level_analyses=[_race_level_analysis(16, "H16", ["HEAD_TO_HEAD_POSITIVE"])],
        )
        self._write_case("r2", "2026-05-21", "芝", scores, prediction, (16, 11, 12))

    def _run(self) -> dict:
        return run_condition_weight_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
        )

    def test_turf_race_level_weight_can_change_top5(self) -> None:
        report = self._run()
        rows = report["scenario_results"]["turf_race_level_1_2"]["rows"]
        row = next(row for row in rows if row["race_id"] == "r1")
        self.assertIn(6, row["scenario_top5"])
        self.assertEqual(row["delta"], 1)

    def test_strict_rank6_recovery_blocks_strong_risk(self) -> None:
        report = self._run()
        rows = report["scenario_results"]["strict_rank6_recovery"]["rows"]
        row = next(row for row in rows if row["race_id"] == "r2")
        self.assertFalse(row["recovery_applied"])
        self.assertNotIn(16, row["scenario_top5"])

    def test_prediction_files_are_not_modified(self) -> None:
        self._run()
        self.assertEqual(self.original_prediction_text, (self.predictions_dir / "r1.json").read_text(encoding="utf-8"))

    def test_markdown_report_is_generated(self) -> None:
        markdown = generate_condition_weight_simulation_markdown(self._run())
        self.assertIn("# Condition Weight Simulation Report", markdown)
        self.assertIn("## Scenario Comparison", markdown)


class ConditionWeightCommandTests(unittest.TestCase):
    def test_command_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            predictions_dir = root / "predictions"
            results_dir = root / "results"
            reviews_dir = root / "reviews"
            race_data_dir = root / "race_data"
            output_dir = root / "error_analysis"
            for path in (predictions_dir, results_dir, reviews_dir, race_data_dir):
                path.mkdir()
            scores = [
                _horse_score(1, "H1", 50.0),
                _horse_score(2, "H2", 49.0),
                _horse_score(3, "H3", 48.0),
                _horse_score(4, "H4", 47.0),
                _horse_score(5, "H5", 46.0),
                _horse_score(6, "H6", 44.9, race_level=1.0),
            ]
            prediction = _prediction_payload("r1", "2026-05-20", "芝", scores)
            (predictions_dir / "r1.json").write_text(json.dumps(prediction, ensure_ascii=False), encoding="utf-8")
            (race_data_dir / "r1.json").write_text(json.dumps(_race_data_payload("r1", "2026-05-20", "芝", scores), ensure_ascii=False), encoding="utf-8")
            (results_dir / "r1.json").write_text(json.dumps(_result_payload("r1", (6, 1, 2)), ensure_ascii=False), encoding="utf-8")
            (reviews_dir / "r1.json").write_text(json.dumps(_review_payload("r1"), ensure_ascii=False), encoding="utf-8")

            import keiba_llm_agent.main as main_module

            original_dirs = (
                main_module.DEFAULT_PREDICTIONS_DIR,
                main_module.DEFAULT_RESULTS_DIR,
                main_module.DEFAULT_REVIEWS_DIR,
                main_module.DEFAULT_RACE_DATA_DIR,
                main_module.DEFAULT_ERROR_ANALYSIS_DIR,
            )
            try:
                main_module.DEFAULT_PREDICTIONS_DIR = predictions_dir
                main_module.DEFAULT_RESULTS_DIR = results_dir
                main_module.DEFAULT_REVIEWS_DIR = reviews_dir
                main_module.DEFAULT_RACE_DATA_DIR = race_data_dir
                main_module.DEFAULT_ERROR_ANALYSIS_DIR = output_dir
                result = run_condition_weight_simulation_command("2026-05-01", "2026-05-31")
            finally:
                (
                    main_module.DEFAULT_PREDICTIONS_DIR,
                    main_module.DEFAULT_RESULTS_DIR,
                    main_module.DEFAULT_REVIEWS_DIR,
                    main_module.DEFAULT_RACE_DATA_DIR,
                    main_module.DEFAULT_ERROR_ANALYSIS_DIR,
                ) = original_dirs

            self.assertTrue(Path(result["json_path"]).exists())
            self.assertTrue(Path(result["md_path"]).exists())
            self.assertEqual(result["reviewed_race_count"], 1)


if __name__ == "__main__":
    unittest.main()

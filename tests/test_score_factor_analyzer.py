from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.analysis_reports.score_factor_analyzer import (
    generate_score_factor_analysis_markdown,
    run_score_factor_analysis,
)
from keiba_llm_agent.main import run_score_factor_analysis_command
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _horse_score(
    horse_no: int,
    horse_name: str,
    total: float,
    *,
    recent_form: int,
    distance_fit: int,
    course_fit: int = 5,
    risk: int = -3,
    race_level: float = 0.0,
    pace: float = 0.0,
) -> dict:
    return {
        "horse_no": horse_no,
        "horse_name": horse_name,
        "scores": {
            "recent_form": recent_form,
            "distance_fit": distance_fit,
            "course_fit": course_fit,
            "track_condition_fit": 5,
            "jockey_fit": 5,
            "odds_value": 5,
            "risk": risk,
        },
        "base_total_score": total - race_level,
        "pedigree_adjustment": {
            "pedigree_bonus": 0.0,
            "pedigree_penalty": 0.0,
            "pedigree_adjustment": 0.0,
            "reason": "",
        },
        "race_level_adjustment": {"adjustment": race_level, "reason": ""},
        "pace_adjustment": {"adjustment": pace, "reason": ""},
        "score_breakdown": {
            "base_total_score": total - race_level,
            "pedigree_adjustment_raw": 0.0,
            "pedigree_weight": 0.2,
            "pedigree_adjustment_weighted": 0.0,
            "race_level_adjustment_raw": race_level,
            "race_level_weight": 1.0,
            "race_level_adjustment_weighted": race_level,
            "pace_adjustment_raw": pace,
            "pace_weight": 0.0,
            "pace_adjustment_weighted": 0.0,
            "borderline_recovery_bonus": 0.0,
            "total_score": total,
            "total_score_after_recovery": total,
        },
        "odds": 2.0 + horse_no,
        "popularity": horse_no,
        "total_score": total,
        "reason": "reason",
    }


def _prediction_payload(race_id: str, race_date: str, scores: list[dict]) -> dict:
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
        "marks": {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
        "horse_scores": scores,
        "bets": [],
        "summary": "summary",
        "risks": [],
        "used_lessons": [],
        "deep_analyses": [],
        "pedigree_analyses": [],
        "race_level_analyses": [],
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
                "main_mark_top3": True,
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


class ScoreFactorAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.predictions_dir = self.root / "predictions"
        self.results_dir = self.root / "results"
        self.reviews_dir = self.root / "reviews"
        for path in (self.predictions_dir, self.results_dir, self.reviews_dir):
            path.mkdir()
        self._write_race("r1", "2026-05-20", (1, 2, 6))
        self._write_race("r2", "2026-05-21", (1, 3, 6))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_race(self, race_id: str, race_date: str, top3: tuple[int, int, int]) -> None:
        scores = [
            _horse_score(1, "H1", 50.0, recent_form=9, distance_fit=9, risk=-1, race_level=0.5, pace=0.4),
            _horse_score(2, "H2", 48.0, recent_form=8, distance_fit=8, risk=-1, race_level=0.4, pace=0.3),
            _horse_score(3, "H3", 47.0, recent_form=7, distance_fit=7, risk=-2, race_level=0.2, pace=0.2),
            _horse_score(4, "H4", 46.0, recent_form=4, distance_fit=5, risk=-3, race_level=0.0, pace=0.0),
            _horse_score(5, "H5", 45.0, recent_form=3, distance_fit=4, risk=-5, race_level=-0.2, pace=-0.2),
            _horse_score(6, "H6", 30.0, recent_form=8, distance_fit=9, risk=-1, race_level=0.5, pace=0.6),
        ]
        self._write_json(self.predictions_dir / f"{race_id}.json", _prediction_payload(race_id, race_date, scores))
        self._write_json(self.results_dir / f"{race_id}.json", _result_payload(race_id, top3))
        self._write_json(self.reviews_dir / f"{race_id}.json", _review_payload(race_id))

    def test_factor_analysis_computes_group_diffs(self) -> None:
        report = run_score_factor_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        self.assertEqual(report["summary"]["reviewed_race_count"], 2)
        self.assertEqual(report["summary"]["total_top3_count"], 6)
        self.assertEqual(report["summary"]["missed_top3_count"], 2)
        factors = {row["factor"]: row for row in report["factor_comparison"]}
        self.assertGreater(factors["recent_form"]["top3_minus_non_top3"], 0)
        self.assertGreater(factors["pace_adjustment_raw"]["missed_top3_mean"], 0)

    def test_bucket_stats_include_top3_rate(self) -> None:
        report = run_score_factor_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        total_score_buckets = report["bucket_stats"]["total_score"]
        self.assertTrue(any("top3_rate" in row for row in total_score_buckets))

    def test_markdown_generation(self) -> None:
        report = run_score_factor_analysis(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        markdown = generate_score_factor_analysis_markdown(report)
        self.assertIn("# Score Factor Analysis", markdown)
        self.assertIn("## Factor Comparison", markdown)


class ScoreFactorAnalysisCommandTests(unittest.TestCase):
    def test_command_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            predictions_dir = root / "predictions"
            results_dir = root / "results"
            reviews_dir = root / "reviews"
            reports_dir = root / "analysis_reports"
            for path in (predictions_dir, results_dir, reviews_dir):
                path.mkdir()
            scores = [
                _horse_score(1, "H1", 50.0, recent_form=9, distance_fit=9, risk=-1),
                _horse_score(2, "H2", 48.0, recent_form=8, distance_fit=8, risk=-1),
                _horse_score(3, "H3", 47.0, recent_form=7, distance_fit=7, risk=-2),
                _horse_score(4, "H4", 46.0, recent_form=4, distance_fit=5, risk=-3),
                _horse_score(5, "H5", 45.0, recent_form=3, distance_fit=4, risk=-5),
            ]
            (predictions_dir / "r1.json").write_text(json.dumps(_prediction_payload("r1", "2026-05-20", scores), ensure_ascii=False), encoding="utf-8")
            (results_dir / "r1.json").write_text(json.dumps(_result_payload("r1", (1, 2, 3)), ensure_ascii=False), encoding="utf-8")
            (reviews_dir / "r1.json").write_text(json.dumps(_review_payload("r1"), ensure_ascii=False), encoding="utf-8")

            import keiba_llm_agent.main as main_module

            original_dirs = (
                main_module.DEFAULT_PREDICTIONS_DIR,
                main_module.DEFAULT_RESULTS_DIR,
                main_module.DEFAULT_REVIEWS_DIR,
                main_module.DEFAULT_ANALYSIS_REPORTS_DIR,
            )
            try:
                main_module.DEFAULT_PREDICTIONS_DIR = predictions_dir
                main_module.DEFAULT_RESULTS_DIR = results_dir
                main_module.DEFAULT_REVIEWS_DIR = reviews_dir
                main_module.DEFAULT_ANALYSIS_REPORTS_DIR = reports_dir
                result = run_score_factor_analysis_command("2026-05-01", "2026-05-31")
            finally:
                (
                    main_module.DEFAULT_PREDICTIONS_DIR,
                    main_module.DEFAULT_RESULTS_DIR,
                    main_module.DEFAULT_REVIEWS_DIR,
                    main_module.DEFAULT_ANALYSIS_REPORTS_DIR,
                ) = original_dirs

            self.assertTrue(Path(result["json_path"]).exists())
            self.assertTrue(Path(result["md_path"]).exists())
            self.assertEqual(result["summary"]["reviewed_race_count"], 1)


if __name__ == "__main__":
    unittest.main()

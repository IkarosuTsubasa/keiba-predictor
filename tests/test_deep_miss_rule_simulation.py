from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.error_analysis.deep_miss_rule_simulator import (
    generate_deep_miss_rule_simulation_markdown,
    run_deep_miss_rule_simulation,
)
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _horse_score(horse_no: int, horse_name: str, base: float, risk: int = -2, odds: float | None = None, popularity: int | None = None) -> dict:
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
            "borderline_recovery_bonus": 0.0,
            "total_score": base,
            "total_score_after_recovery": base,
        },
        "odds": odds,
        "popularity": popularity,
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
        "scoring_profile": "accuracy_default",
        "scoring_mode": "candidate_default",
        "borderline_recovery_enabled": True,
        "scoring_config": {
            "scoring_mode": "candidate_default",
            "pedigree_weight": 0.2,
            "race_level_weight": 1.0,
            "pace_weight": 0.0,
        },
        "borderline_recovery_config": {
            "enabled": True,
            "max_rank": 6,
            "max_score_gap": 1.0,
            "min_net_signal": 2,
            "max_recoveries_per_race": 1,
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
        "pace_analyses": [],
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


def _finish_order(entries: list[tuple[int, int, str, float, int]]) -> list[dict]:
    return [
        {
            "finish": finish,
            "horse_no": horse_no,
            "horse_name": horse_name,
            "odds": odds,
            "popularity": popularity,
        }
        for finish, horse_no, horse_name, odds, popularity in entries
    ]


class DeepMissRuleSimulationTests(unittest.TestCase):
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

        self._write_race_r1()
        self._write_race_r2()
        self._write_race_r3()
        self._write_race_r4()

        self.original_prediction_text = (self.predictions_dir / "r1.json").read_text(encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_shared_review(self, race_id: str) -> None:
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
        (self.reviews_dir / f"{race_id}.json").write_text(
            json.dumps(review.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_race_r1(self) -> None:
        scores = [
            _horse_score(1, "A1", 50.0, odds=2.0, popularity=1),
            _horse_score(2, "A2", 49.0, odds=3.0, popularity=2),
            _horse_score(3, "A3", 48.0, odds=4.0, popularity=3),
            _horse_score(4, "A4", 47.0, odds=6.0, popularity=4),
            _horse_score(5, "A5", 46.0, odds=8.0, popularity=5),
            _horse_score(6, "A6", 45.0, odds=11.0, popularity=6),
            _horse_score(7, "A7", 44.0, odds=9.0, popularity=2),
            _horse_score(8, "A8", 43.0, odds=12.0, popularity=4),
        ]
        prediction = _prediction_payload(
            "r1",
            "Race1",
            "2026-05-20",
            scores,
            deep_analyses=[
                {
                    "horse_no": 7,
                    "horse_name": "A7",
                    "positive_flags": ["RECENT_FORM_STABLE", "DISTANCE_FIT"],
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
                    "horse_no": 7,
                    "horse_name": "A7",
                    "positive_flags": ["HEAD_TO_HEAD_POSITIVE"],
                    "risk_flags": [],
                    "head_to_head_summary": "",
                    "race_level_summary": "",
                    "opponent_context_summary": "",
                    "overall_comment": "",
                    "adjustment_hint": 0.5,
                }
            ],
        )
        (self.predictions_dir / "r1.json").write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
        race_data = _race_data_payload(
            "r1",
            "Race1",
            "2026-05-20",
            [(horse["horse_no"], horse["horse_name"], horse["odds"], horse["popularity"]) for horse in scores],
        )
        (self.race_data_dir / "r1.json").write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")
        result = ResultData.model_validate(
            {
                "race_id": "r1",
                "result": {"1st": 7, "2nd": 1, "3rd": 2},
                "payouts": [],
                "finish_order": _finish_order(
                    [
                        (1, 7, "A7", 9.0, 2),
                        (2, 1, "A1", 2.0, 1),
                        (3, 2, "A2", 3.0, 2),
                        (4, 3, "A3", 4.0, 3),
                        (5, 4, "A4", 6.0, 4),
                        (8, 5, "A5", 8.0, 5),
                    ]
                ),
            }
        )
        (self.results_dir / "r1.json").write_text(
            json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_shared_review("r1")

    def _write_race_r2(self) -> None:
        scores = [
            _horse_score(11, "B1", 50.0, odds=2.0, popularity=1),
            _horse_score(12, "B2", 49.0, odds=3.0, popularity=2),
            _horse_score(13, "B3", 48.0, odds=4.0, popularity=3),
            _horse_score(14, "B4", 47.0, odds=5.0, popularity=4),
            _horse_score(15, "B5", 46.0, odds=8.0, popularity=5),
            _horse_score(16, "B6", 45.0, odds=10.0, popularity=6),
            _horse_score(17, "B7", 44.0, odds=9.0, popularity=2),
        ]
        prediction = _prediction_payload(
            "r2",
            "Race2",
            "2026-05-21",
            scores,
            deep_analyses=[
                {
                    "horse_no": 17,
                    "horse_name": "B7",
                    "positive_flags": ["RECENT_FORM_STABLE", "DISTANCE_FIT"],
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
                    "horse_no": 17,
                    "horse_name": "B7",
                    "positive_flags": ["HEAD_TO_HEAD_POSITIVE"],
                    "risk_flags": [],
                    "head_to_head_summary": "",
                    "race_level_summary": "",
                    "opponent_context_summary": "",
                    "overall_comment": "",
                    "adjustment_hint": 0.5,
                }
            ],
        )
        (self.predictions_dir / "r2.json").write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
        race_data = _race_data_payload(
            "r2",
            "Race2",
            "2026-05-21",
            [(horse["horse_no"], horse["horse_name"], horse["odds"], horse["popularity"]) for horse in scores],
        )
        (self.race_data_dir / "r2.json").write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")
        result = ResultData.model_validate(
            {
                "race_id": "r2",
                "result": {"1st": 11, "2nd": 12, "3rd": 15},
                "payouts": [],
                "finish_order": _finish_order(
                    [
                        (1, 11, "B1", 2.0, 1),
                        (2, 12, "B2", 3.0, 2),
                        (3, 15, "B5", 8.0, 5),
                        (7, 17, "B7", 9.0, 2),
                    ]
                ),
            }
        )
        (self.results_dir / "r2.json").write_text(
            json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_shared_review("r2")

    def _write_race_r3(self) -> None:
        scores = [
            _horse_score(21, "C1", 50.0, odds=2.0, popularity=1),
            _horse_score(22, "C2", 49.0, odds=3.0, popularity=2),
            _horse_score(23, "C3", 48.0, odds=4.0, popularity=3),
            _horse_score(24, "C4", 47.0, odds=5.0, popularity=4),
            _horse_score(25, "C5", 46.0, odds=6.0, popularity=5),
            _horse_score(26, "C6", 45.0, odds=10.0, popularity=6),
            _horse_score(27, "C13", 38.0, odds=55.0, popularity=10),
        ]
        prediction = _prediction_payload(
            "r3",
            "Race3",
            "2026-05-22",
            scores,
            deep_analyses=[
                {
                    "horse_no": 27,
                    "horse_name": "C13",
                    "positive_flags": ["RECENT_FORM_STABLE", "DISTANCE_FIT"],
                    "risk_flags": ["RECENT_FORM_WEAK", "RECENT_FORM_DECLINING", "COURSE_UNKNOWN", "JOCKEY_CHANGE"],
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
                    "horse_no": 27,
                    "horse_name": "C13",
                    "positive_flags": ["HEAD_TO_HEAD_POSITIVE"],
                    "risk_flags": [],
                    "head_to_head_summary": "",
                    "race_level_summary": "",
                    "opponent_context_summary": "",
                    "overall_comment": "",
                    "adjustment_hint": 0.5,
                }
            ],
        )
        (self.predictions_dir / "r3.json").write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
        race_data = _race_data_payload(
            "r3",
            "Race3",
            "2026-05-22",
            [(21, "C1", 2.0, 1), (22, "C2", 3.0, 2), (23, "C3", 4.0, 3), (24, "C4", 5.0, 4), (25, "C5", 6.0, 5), (26, "C6", 10.0, 6), (27, "C13", 55.0, 10)],
        )
        (self.race_data_dir / "r3.json").write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")
        result = ResultData.model_validate(
            {
                "race_id": "r3",
                "result": {"1st": 27, "2nd": 21, "3rd": 22},
                "payouts": [],
                "finish_order": _finish_order(
                    [
                        (1, 27, "C13", 55.0, 10),
                        (2, 21, "C1", 2.0, 1),
                        (3, 22, "C2", 3.0, 2),
                    ]
                ),
            }
        )
        (self.results_dir / "r3.json").write_text(
            json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_shared_review("r3")

    def _write_race_r4(self) -> None:
        scores = [
            _horse_score(31, "D1", 50.0, odds=2.0, popularity=1),
            _horse_score(32, "D2", 49.0, odds=3.0, popularity=2),
            _horse_score(33, "D3", 48.0, odds=4.0, popularity=3),
            _horse_score(34, "D4", 47.0, odds=5.0, popularity=4),
            _horse_score(35, "D5", 46.0, odds=6.0, popularity=5),
            _horse_score(36, "D6", 45.0, odds=8.0, popularity=6),
            _horse_score(37, "D7", 44.0, odds=9.0, popularity=4),
        ]
        prediction = _prediction_payload(
            "r4",
            "Race4",
            "2026-05-23",
            scores,
            deep_analyses=[
                {
                    "horse_no": 37,
                    "horse_name": "D7",
                    "positive_flags": ["RECENT_FORM_STABLE", "DISTANCE_FIT"],
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
                    "horse_no": 37,
                    "horse_name": "D7",
                    "positive_flags": ["HEAD_TO_HEAD_POSITIVE"],
                    "risk_flags": [],
                    "head_to_head_summary": "",
                    "race_level_summary": "",
                    "opponent_context_summary": "",
                    "overall_comment": "",
                    "adjustment_hint": 0.5,
                }
            ],
        )
        (self.predictions_dir / "r4.json").write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
        race_data = _race_data_payload(
            "r4",
            "Race4",
            "2026-05-23",
            [(31, "D1", 2.0, 1), (32, "D2", 3.0, 2), (33, "D3", 4.0, 3), (34, "D4", 5.0, 4), (35, "D5", 6.0, 5), (36, "D6", 8.0, 6), (37, "D7", 9.0, 4)],
        )
        (self.race_data_dir / "r4.json").write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")
        result = ResultData.model_validate(
            {
                "race_id": "r4",
                "result": {"1st": 37, "2nd": 31, "3rd": 32},
                "payouts": [],
                "finish_order": _finish_order(
                    [
                        (1, 37, "D7", 9.0, 4),
                        (2, 31, "D1", 2.0, 1),
                        (3, 32, "D2", 3.0, 2),
                    ]
                ),
            }
        )
        (self.results_dir / "r4.json").write_text(
            json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_shared_review("r4")

    def test_rank7_8_popular_racelevel_only_uses_rank7_8(self) -> None:
        report = run_deep_miss_rule_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            baseline_mode="candidate_default_recovered",
        )
        rows = report["rule_results"]["rank7_8_popular_racelevel"]["rows"]
        self.assertTrue(rows)
        self.assertTrue(all(7 <= row["predicted_rank"] <= 8 for row in rows))

    def test_popular_safety_net_does_not_choose_popularity_over_3(self) -> None:
        report = run_deep_miss_rule_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            baseline_mode="candidate_default_recovered",
        )
        rows = report["rule_results"]["popular_safety_net"]["rows"]
        self.assertNotIn("r4", [row["race_id"] for row in rows])

    def test_strong_risk_candidate_is_not_selected(self) -> None:
        report = run_deep_miss_rule_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            baseline_mode="candidate_default_recovered",
            include_rank13=True,
            max_rank=12,
        )
        all_rows = report["race_details"]
        self.assertNotIn("r3", [row["race_id"] for row in all_rows])

    def test_each_race_selects_at_most_one_candidate_per_rule(self) -> None:
        report = run_deep_miss_rule_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            baseline_mode="candidate_default_recovered",
        )
        for rule_name, result in report["rule_results"].items():
            race_ids = [row["race_id"] for row in result["rows"]]
            self.assertEqual(len(race_ids), len(set(race_ids)), msg=rule_name)

    def test_replaces_lowest_top5_horse(self) -> None:
        report = run_deep_miss_rule_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            baseline_mode="candidate_default_recovered",
        )
        row = next(row for row in report["rule_results"]["rank7_8_popular_racelevel"]["rows"] if row["race_id"] == "r1")
        self.assertEqual(row["replaced_horse_no"], 5)

    def test_better_and_worse_counts_are_computed(self) -> None:
        report = run_deep_miss_rule_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            baseline_mode="candidate_default_recovered",
        )
        result = report["rule_results"]["rank7_8_popular_racelevel"]
        self.assertEqual(result["better_race_count"], 1)
        self.assertEqual(result["worse_race_count"], 1)
        self.assertEqual(result["replaced_top3_count"], 1)
        self.assertEqual(result["safe_score"], 1 - 1 - 1)

    def test_candidate_default_recovered_baseline_is_unchanged(self) -> None:
        run_deep_miss_rule_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            baseline_mode="candidate_default_recovered",
        )
        current_text = (self.predictions_dir / "r1.json").read_text(encoding="utf-8")
        self.assertEqual(self.original_prediction_text, current_text)

    def test_markdown_report_is_generated(self) -> None:
        report = run_deep_miss_rule_simulation(
            from_date="2026-05-01",
            to_date="2026-05-31",
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
            race_data_dir=self.race_data_dir,
            baseline_mode="candidate_default_recovered",
        )
        markdown = generate_deep_miss_rule_simulation_markdown(report)
        self.assertIn("# Deep Miss Rule Simulation Report", markdown)
        self.assertIn("## Rule Comparison", markdown)
        self.assertIn("## Best Rule Candidates", markdown)
        self.assertIn("## Rule Details", markdown)
        self.assertIn("## Race Details", markdown)


if __name__ == "__main__":
    unittest.main()

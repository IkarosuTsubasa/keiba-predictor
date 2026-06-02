from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.audit.data_flow_auditor import (
    audit_data_flow_period,
    audit_single_race_data_flow,
    generate_period_audit_markdown,
    generate_single_audit_markdown,
)
from keiba_llm_agent.main import run_audit_data_flow_command, run_audit_race_data_flow_command
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _horse_score(horse_no: int, horse_name: str, total: float) -> dict:
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
            "risk": -2,
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
            "use_market_score_in_ranking": False,
            "market_signal_weight": 0.0,
        },
        "market_signal_config": {
            "use_market_score_in_ranking": False,
            "market_signal_weight": 0.0,
        },
        "borderline_recovery_config": {
            "enabled": True,
            "max_rank": 6,
            "max_score_gap": 1.0,
            "min_net_signal": 2,
            "max_recoveries_per_race": 1,
        },
        "marks": {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
        "horse_scores": scores,
        "bets": [],
        "summary": "summary",
        "risks": [],
        "used_lessons": [],
        "deep_analyses": [
            {
                "horse_no": score["horse_no"],
                "horse_name": score["horse_name"],
                "positive_flags": [],
                "risk_flags": [],
                "recent_form_summary": "",
                "distance_analysis": "",
                "course_analysis": "",
                "track_condition_analysis": "",
                "jockey_analysis": "",
                "odds_analysis": "",
                "overall_comment": "",
            }
            for score in scores
        ],
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
        "race_simulation": {
            "race_id": race_id,
            "race_flow": "平均ペース想定。",
            "key_positions": "上位勢が好位。",
            "favorable_horses": [],
            "risk_horses": [],
            "win_scenario": "本命が押し切る。",
            "top3_scenario": "上位印中心。",
            "betting_scenario": "買い目なし。",
            "confidence_comment": "低め。",
            "reasoning_summary": "平均寄り。",
        },
        "borderline_recovery_result": {"recovery_applied": False, "recovery_cases": []},
    }


def _race_data_payload(race_id: str, race_date: str, future_run: bool = False) -> dict:
    run_date = race_date if future_run else "2026-04-20"
    return {
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
        "horses": [
            {
                "horse_no": horse_no,
                "frame_no": horse_no,
                "horse_id": f"horse{horse_no}",
                "horse_name": f"H{horse_no}",
                "jockey": "騎手",
                "odds": 2.0 + horse_no,
                "popularity": horse_no,
                "recent_runs": [
                    {
                        "race_id": f"20260401010{horse_no}",
                        "date": run_date if horse_no == 1 else "2026-04-20",
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "finish": 2,
                        "field_size": 12,
                    }
                ],
            }
            for horse_no in range(1, 6)
        ],
    }


def _result_payload(race_id: str, with_finish_order: bool = True, with_payouts: bool = True) -> dict:
    result = ResultData.model_validate(
        {
            "race_id": race_id,
            "result": {"1st": 1, "2nd": 2, "3rd": 3},
            "payouts": [{"bet_type": "ワイド", "combination": "1-2", "payout": 300, "popularity": 1}] if with_payouts else [],
            "finish_order": [
                {"finish": 1, "horse_no": 1, "horse_name": "H1"},
                {"finish": 2, "horse_no": 2, "horse_name": "H2"},
                {"finish": 3, "horse_no": 3, "horse_name": "H3"},
                {"finish": 4, "horse_no": 4, "horse_name": "H4"},
                {"finish": 5, "horse_no": 5, "horse_name": "H5"},
            ] if with_finish_order else [],
        }
    )
    return result.model_dump(by_alias=True)


def _review_payload(race_id: str) -> dict:
    review = Review.model_validate(
        {
            "race_id": race_id,
            "hit_summary": {
                "main_mark_top3": True,
                "marked_horses_top3_count": 3,
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


class DataFlowAuditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.race_data_dir = self.root / "race_data"
        self.predictions_dir = self.root / "predictions"
        self.results_dir = self.root / "results"
        self.reviews_dir = self.root / "reviews"
        for path in (self.race_data_dir, self.predictions_dir, self.results_dir, self.reviews_dir):
            path.mkdir()
        self._write_race("r1", "2026-05-20")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_race(self, race_id: str, race_date: str, *, future_run: bool = False, with_finish_order: bool = True, with_payouts: bool = True) -> None:
        scores = [_horse_score(horse_no, f"H{horse_no}", 50.0 - horse_no) for horse_no in range(1, 6)]
        self._write_json(self.race_data_dir / f"{race_id}.json", _race_data_payload(race_id, race_date, future_run=future_run))
        self._write_json(self.predictions_dir / f"{race_id}.json", _prediction_payload(race_id, race_date, scores))
        self._write_json(self.results_dir / f"{race_id}.json", _result_payload(race_id, with_finish_order=with_finish_order, with_payouts=with_payouts))
        self._write_json(self.reviews_dir / f"{race_id}.json", _review_payload(race_id))

    def test_single_audit_reports_sources_and_readiness(self) -> None:
        audit = audit_single_race_data_flow(
            race_id="r1",
            race_data_dir=self.race_data_dir,
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        self.assertEqual(audit["status"], "warning")
        self.assertTrue(audit["readiness"]["backtest_ready"])
        self.assertTrue(audit["readiness"]["roi_reliable"])
        self.assertEqual(audit["data_flow"]["prediction"]["field_sources"]["marks"], "sorted by final total_score")

    def test_future_recent_run_is_flagged_as_leakage(self) -> None:
        self._write_race("r2", "2026-05-21", future_run=True)
        audit = audit_single_race_data_flow(
            race_id="r2",
            race_data_dir=self.race_data_dir,
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        codes = {issue["code"] for issue in audit["issues"]}
        self.assertIn("RECENT_RUN_DATE_LEAKAGE", codes)
        self.assertEqual(audit["status"], "error")

    def test_missing_payout_and_finish_order_are_flagged(self) -> None:
        self._write_race("r3", "2026-05-22", with_finish_order=False, with_payouts=False)
        audit = audit_single_race_data_flow(
            race_id="r3",
            race_data_dir=self.race_data_dir,
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        codes = {issue["code"] for issue in audit["issues"]}
        self.assertIn("PAYOUTS_MISSING", codes)
        self.assertIn("FINISH_ORDER_MISSING", codes)
        self.assertFalse(audit["readiness"]["roi_reliable"])

    def test_period_audit_summarizes_problem_races(self) -> None:
        self._write_race("r2", "2026-05-21", future_run=True)
        report = audit_data_flow_period(
            from_date="2026-05-01",
            to_date="2026-05-31",
            race_data_dir=self.race_data_dir,
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        self.assertEqual(report["summary"]["race_count"], 2)
        self.assertEqual(report["summary"]["backtest_ready_count"], 2)
        self.assertGreaterEqual(report["issue_counts"]["RECENT_RUN_DATE_LEAKAGE"], 1)
        markdown = generate_period_audit_markdown(report)
        self.assertIn("# Data Flow Audit Summary", markdown)

    def test_markdown_generation_for_single_audit(self) -> None:
        audit = audit_single_race_data_flow(
            race_id="r1",
            race_data_dir=self.race_data_dir,
            predictions_dir=self.predictions_dir,
            results_dir=self.results_dir,
            reviews_dir=self.reviews_dir,
        )
        markdown = generate_single_audit_markdown(audit)
        self.assertIn("# Data Flow Audit", markdown)
        self.assertIn("## Key Checks", markdown)


class DataFlowAuditCommandTests(unittest.TestCase):
    def test_command_uses_default_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            race_data_dir = root / "race_data"
            predictions_dir = root / "predictions"
            results_dir = root / "results"
            reviews_dir = root / "reviews"
            audits_dir = root / "audits"
            for path in (race_data_dir, predictions_dir, results_dir, reviews_dir):
                path.mkdir()
            scores = [_horse_score(horse_no, f"H{horse_no}", 50.0 - horse_no) for horse_no in range(1, 6)]
            (race_data_dir / "r1.json").write_text(json.dumps(_race_data_payload("r1", "2026-05-20"), ensure_ascii=False), encoding="utf-8")
            (predictions_dir / "r1.json").write_text(json.dumps(_prediction_payload("r1", "2026-05-20", scores), ensure_ascii=False), encoding="utf-8")
            (results_dir / "r1.json").write_text(json.dumps(_result_payload("r1"), ensure_ascii=False), encoding="utf-8")
            (reviews_dir / "r1.json").write_text(json.dumps(_review_payload("r1"), ensure_ascii=False), encoding="utf-8")

            import keiba_llm_agent.main as main_module

            original_dirs = (
                main_module.DEFAULT_RACE_DATA_DIR,
                main_module.DEFAULT_PREDICTIONS_DIR,
                main_module.DEFAULT_RESULTS_DIR,
                main_module.DEFAULT_REVIEWS_DIR,
                main_module.DEFAULT_AUDITS_DIR,
            )
            try:
                main_module.DEFAULT_RACE_DATA_DIR = race_data_dir
                main_module.DEFAULT_PREDICTIONS_DIR = predictions_dir
                main_module.DEFAULT_RESULTS_DIR = results_dir
                main_module.DEFAULT_REVIEWS_DIR = reviews_dir
                main_module.DEFAULT_AUDITS_DIR = audits_dir
                single = run_audit_race_data_flow_command("r1")
                period = run_audit_data_flow_command("2026-05-01", "2026-05-31")
            finally:
                (
                    main_module.DEFAULT_RACE_DATA_DIR,
                    main_module.DEFAULT_PREDICTIONS_DIR,
                    main_module.DEFAULT_RESULTS_DIR,
                    main_module.DEFAULT_REVIEWS_DIR,
                    main_module.DEFAULT_AUDITS_DIR,
                ) = original_dirs

            self.assertTrue(Path(single["json_path"]).exists())
            self.assertTrue(Path(period["json_path"]).exists())
            self.assertEqual(period["summary"]["race_count"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.backtest.audit_snapshot import read_backtest_audit_payload


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_agent_backtest_audit import build_agent_backtest_audit  # noqa: E402


def _prediction_payload() -> dict:
    return {
        "race_id": "202605020811",
        "race_info": {
            "race_id": "202605020811",
            "race_date": "2026-05-17",
            "course": "東京",
            "surface": "芝",
            "distance": 1600,
        },
        "scoring_profile": "accuracy_default",
        "scoring_mode": "candidate_default",
        "marks": {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
        "horse_scores": [
            {
                "horse_no": 1,
                "horse_name": "ホースA",
                "scores": {
                    "recent_form": 7,
                    "distance_fit": 8,
                    "course_fit": 6,
                    "track_condition_fit": 7,
                    "jockey_fit": 5,
                    "odds_value": 0,
                    "risk": -1,
                },
                "base_total_score": 33.0,
                "total_score": 34.2,
                "reason": "sample",
            }
        ],
        "summary": "sample",
    }


class BuildAgentBacktestAuditTests(unittest.TestCase):
    def test_builds_full_audit_from_existing_prediction_result_and_race_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_dir = root / "agent_predictions"
            results_dir = root / "agent_results"
            race_data_dir = root / "race_data"
            audit_dir = root / "agent_backtest_audit"
            for directory in (predictions_dir, results_dir, race_data_dir, audit_dir):
                directory.mkdir(parents=True, exist_ok=True)

            race_id = "202605020811"
            (predictions_dir / f"{race_id}.json").write_text(
                json.dumps(_prediction_payload(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (results_dir / f"{race_id}.json").write_text(
                json.dumps({"race_id": race_id, "result": {"1st": 1, "2nd": 2, "3rd": 3}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (race_data_dir / f"{race_id}.json").write_text(
                json.dumps(
                    {
                        "race_info": _prediction_payload()["race_info"],
                        "horses": [
                            {
                                "horse_no": 1,
                                "horse_name": "ホースA",
                                "recent_runs": [
                                    {
                                        "race_id": "202605010101",
                                        "date": "2026-05-01",
                                        "course": "東京",
                                        "surface": "芝",
                                        "distance": 1600,
                                        "finish": 2,
                                        "field_size": 16,
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary = build_agent_backtest_audit(
                predictions_dir=predictions_dir,
                results_dir=results_dir,
                audit_dir=audit_dir,
                race_data_dir=race_data_dir,
                include_items=True,
            )
            payload = read_backtest_audit_payload(audit_dir / f"{race_id}.json.gz")

        self.assertEqual(summary["counts"], {"full": 1})
        self.assertEqual(payload["audit_kind"], "full")
        self.assertTrue(payload["backtest_capabilities"]["can_recompute_from_recent_runs"])
        self.assertTrue(payload["backtest_capabilities"]["has_result_for_evaluation"])


if __name__ == "__main__":
    unittest.main()

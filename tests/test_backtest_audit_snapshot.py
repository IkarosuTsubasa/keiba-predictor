from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.backtest.audit_snapshot import (
    attach_result_to_backtest_audit,
    build_backtest_audit_payload,
    read_backtest_audit_payload,
    write_backtest_audit_payload,
)
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo, RecentRun


def _race_data() -> RaceData:
    return RaceData(
        race_info=RaceInfo(
            race_id="202605020811",
            race_name="サンプル",
            race_date="2026-05-17",
            course="東京",
            surface="芝",
            distance=1600,
            scope_key="central_turf",
        ),
        horses=[
            HorseEntry(
                horse_no=1,
                horse_name="ホースA",
                recent_runs=[
                    RecentRun(
                        race_id="202605010101",
                        date="2026-05-01",
                        course="東京",
                        surface="芝",
                        distance=1600,
                        track_condition="良",
                        finish=2,
                        field_size=16,
                    )
                ],
            )
        ],
    )


def _prediction() -> Prediction:
    return Prediction.model_validate(
        {
            "race_id": "202605020811",
            "race_info": _race_data().race_info.model_dump(),
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
    )


class BacktestAuditSnapshotTests(unittest.TestCase):
    def test_full_audit_keeps_raw_inputs_scores_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = build_backtest_audit_payload(
                race_data=_race_data(),
                prediction=_prediction(),
                result_payload={"race_id": "202605020811", "result": {"1st": 1, "2nd": 2, "3rd": 3}},
            )
            output_path = write_backtest_audit_payload(Path(temp_dir), payload)
            restored = read_backtest_audit_payload(output_path)

        self.assertEqual(restored["schema_version"], "agent_backtest_audit_v1")
        self.assertEqual(restored["audit_kind"], "full")
        self.assertTrue(restored["backtest_capabilities"]["can_recompute_scoring_logic"])
        self.assertTrue(restored["backtest_capabilities"]["can_recompute_from_recent_runs"])
        self.assertTrue(restored["backtest_capabilities"]["has_result_for_evaluation"])
        self.assertEqual(restored["inputs"]["race_data"]["horses"][0]["recent_runs"][0]["finish"], 2)
        self.assertEqual(restored["scoring"]["horse_scores"][0]["total_score"], 34.2)

    def test_attach_result_updates_existing_audit_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = build_backtest_audit_payload(race_data=_race_data(), prediction=_prediction())
            output_path = write_backtest_audit_payload(Path(temp_dir), payload)
            attach_result_to_backtest_audit(
                output_path,
                {"race_id": "202605020811", "result": {"1st": 1, "2nd": 2, "3rd": 3}},
            )
            restored = read_backtest_audit_payload(output_path)

        self.assertTrue(restored["backtest_capabilities"]["has_result_for_evaluation"])
        self.assertNotIn("result", restored["missing_inputs"])


if __name__ == "__main__":
    unittest.main()

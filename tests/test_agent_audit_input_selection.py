from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.backtest.audit_snapshot import read_backtest_audit_payload
from keiba_llm_agent.backtest.input_selection import (
    select_effective_backtest_audit_paths,
    select_effective_prediction_paths,
)
from pipeline.build_agent_backtest_audit import build_agent_backtest_audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


class AgentAuditInputSelectionTests(unittest.TestCase):
    def test_prediction_and_audit_selection_count_each_race_once_and_prefer_final(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            race_id = "202607190101"
            prediction_dir = root / "agent_predictions"
            audit_dir = root / "agent_backtest_audit"
            base_prediction = {
                "race_id": race_id,
                "race_info": {"race_id": race_id, "race_date": "2026-07-19"},
                "marks": {"◎": 1},
            }
            _write_json(prediction_dir / f"{race_id}.json", base_prediction)
            _write_json(
                prediction_dir / "_versions" / race_id / "early" / "task-early.json",
                {**base_prediction, "prediction_stage": "early", "marks": {"◎": 2}},
            )
            final_prediction_path = prediction_dir / "_versions" / race_id / "final" / "task-final.json"
            _write_json(
                final_prediction_path,
                {**base_prediction, "prediction_stage": "final", "marks": {"◎": 3}},
            )

            root_audit = {
                "race_id": race_id,
                "prediction_stage": "early",
                "outputs": {"prediction": base_prediction},
            }
            _write_gzip_json(audit_dir / f"{race_id}.json.gz", root_audit)
            _write_gzip_json(
                audit_dir / "_versions" / race_id / "early" / "task-early.json.gz",
                {**root_audit, "prediction_stage": "early"},
            )
            final_audit_path = audit_dir / "_versions" / race_id / "final" / "task-final.json.gz"
            _write_gzip_json(
                final_audit_path,
                {
                    **root_audit,
                    "prediction_stage": "final",
                    "outputs": {
                        "prediction": {
                            **base_prediction,
                            "prediction_stage": "final",
                            "marks": {"◎": 3},
                        }
                    },
                },
            )

            selected_predictions = select_effective_prediction_paths(prediction_dir)
            selected_audits = select_effective_backtest_audit_paths(audit_dir)

        self.assertEqual(selected_predictions, {race_id: final_prediction_path})
        self.assertEqual(selected_audits, {race_id: final_audit_path})

    def test_audit_rebuild_uses_only_final_prediction_for_same_race(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            race_id = "202607190102"
            prediction_dir = root / "agent_predictions"
            result_dir = root / "agent_results"
            audit_dir = root / "rebuilt_audit"
            base_prediction = {
                "race_id": race_id,
                "race_info": {"race_id": race_id, "race_date": "2026-07-19"},
                "marks": {"◎": 1},
                "horse_scores": [],
            }
            _write_json(prediction_dir / f"{race_id}.json", base_prediction)
            _write_json(
                prediction_dir / "_versions" / race_id / "early" / "task-early.json",
                {**base_prediction, "prediction_stage": "early", "marks": {"◎": 2}},
            )
            _write_json(
                prediction_dir / "_versions" / race_id / "final" / "task-final.json",
                {**base_prediction, "prediction_stage": "final", "marks": {"◎": 3}},
            )
            result_dir.mkdir(parents=True, exist_ok=True)

            summary = build_agent_backtest_audit(
                predictions_dir=prediction_dir,
                results_dir=result_dir,
                audit_dir=audit_dir,
            )
            payload = read_backtest_audit_payload(audit_dir / f"{race_id}.json.gz")

        self.assertEqual(summary["race_count"], 1)
        self.assertEqual(summary["source_prediction_file_count"], 3)
        self.assertEqual(summary["duplicate_prediction_versions_excluded"], 2)
        self.assertEqual(summary["selection_rule"], "one_per_race_final_over_early")
        self.assertEqual(payload["prediction_stage"], "final")
        self.assertEqual(payload["outputs"]["prediction"]["marks"], {"◎": 3})

    def test_invalid_final_files_fall_back_to_valid_early_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            race_id = "202607190103"
            prediction_dir = root / "agent_predictions"
            audit_dir = root / "agent_backtest_audit"
            early_prediction_path = prediction_dir / "_versions" / race_id / "early" / "task-early.json"
            invalid_final_prediction_path = prediction_dir / "_versions" / race_id / "final" / "task-final.json"
            early_audit_path = audit_dir / "_versions" / race_id / "early" / "task-early.json.gz"
            invalid_final_audit_path = audit_dir / "_versions" / race_id / "final" / "task-final.json.gz"
            _write_json(
                early_prediction_path,
                {"race_id": race_id, "prediction_stage": "early", "race_info": {"race_id": race_id}},
            )
            invalid_final_prediction_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_final_prediction_path.write_text("{invalid", encoding="utf-8")
            _write_gzip_json(
                early_audit_path,
                {"race_id": race_id, "prediction_stage": "early"},
            )
            invalid_final_audit_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_final_audit_path.write_bytes(b"not-gzip")

            selected_predictions = select_effective_prediction_paths(prediction_dir)
            selected_audits = select_effective_backtest_audit_paths(audit_dir)

        self.assertEqual(selected_predictions, {race_id: early_prediction_path})
        self.assertEqual(selected_audits, {race_id: early_audit_path})


if __name__ == "__main__":
    unittest.main()

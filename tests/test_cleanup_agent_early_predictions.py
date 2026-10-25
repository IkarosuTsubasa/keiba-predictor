from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.cleanup_runtime_artifacts import _cleanup_superseded_agent_early_predictions


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


class CleanupAgentEarlyPredictionsTests(unittest.TestCase):
    def test_shadow_early_prediction_and_audit_are_preserved_after_final(self) -> None:
        for status in ('predicted', 'failed', 'disabled'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/'data'
                rid = '202605030399'
                paths = []
                for stage in ('early', 'final'):
                    prediction = root/'agent_predictions'/'_versions'/rid/stage/'task.json'
                    audit = root/'agent_backtest_audit'/'_versions'/rid/stage/'task.json.gz'
                    _write_json(prediction, {'race_id': rid, 'prediction_stage': stage})
                    _write_gzip_json(audit, {'race_id': rid, 'prediction_stage': stage,
                        'shadow_predictions': {'central_turf_history': {'status': status}}})
                    paths.extend([prediction, audit])
                originals = {p: p.read_bytes() for p in paths}
                summary = _cleanup_superseded_agent_early_predictions(dry_run=False, data_dir=root)
                self.assertEqual(summary['skipped_reason_counts'], {'shadow_audit_retained': 1})
                self.assertEqual(summary['removed_prediction_files'], [])
                self.assertEqual(summary['removed_audit_files'], [])
                for path in paths:
                    self.assertEqual(path.read_bytes(), originals[path])

    def test_removes_early_files_only_after_valid_final_prediction_and_audit_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            race_id = "202607190201"
            predictions_dir = data_dir / "agent_predictions"
            audits_dir = data_dir / "agent_backtest_audit"
            results_dir = data_dir / "agent_results"
            early_prediction = {
                "race_id": race_id,
                "prediction_stage": "early",
                "race_info": {"race_id": race_id},
            }
            final_prediction = {
                "race_id": race_id,
                "prediction_stage": "final",
                "race_info": {"race_id": race_id},
            }
            early_prediction_root = predictions_dir / f"{race_id}.json"
            early_prediction_version = (
                predictions_dir / "_versions" / race_id / "early" / "task-early.json"
            )
            final_prediction_version = (
                predictions_dir / "_versions" / race_id / "final" / "task-final.json"
            )
            _write_json(early_prediction_root, early_prediction)
            _write_json(early_prediction_version, early_prediction)
            _write_json(final_prediction_version, final_prediction)

            early_audit_version = (
                audits_dir / "_versions" / race_id / "early" / "task-early.json.gz"
            )
            final_audit_version = (
                audits_dir / "_versions" / race_id / "final" / "task-final.json.gz"
            )
            final_audit_root = audits_dir / f"{race_id}.json.gz"
            _write_gzip_json(
                early_audit_version,
                {"race_id": race_id, "prediction_stage": "early"},
            )
            final_audit = {
                "race_id": race_id,
                "prediction_stage": "final",
                "outputs": {"prediction": final_prediction},
            }
            _write_gzip_json(final_audit_version, final_audit)
            _write_gzip_json(
                final_audit_root,
                {"race_id": race_id, "prediction_stage": "early"},
            )
            result_path = results_dir / f"{race_id}.json"
            _write_json(result_path, {"race_id": race_id, "result": {"1st": 1}})

            summary = _cleanup_superseded_agent_early_predictions(
                dry_run=False,
                data_dir=data_dir,
            )
            with gzip.open(final_audit_root, "rt", encoding="utf-8") as fh:
                canonical_audit_payload = json.load(fh)

            remaining_paths = {
                "canonical_prediction_root": early_prediction_root.exists(),
                "canonical_prediction_stage": json.loads(
                    early_prediction_root.read_text(encoding="utf-8")
                ).get("prediction_stage"),
                "early_prediction_version": early_prediction_version.exists(),
                "final_prediction_version": final_prediction_version.exists(),
                "early_audit_version": early_audit_version.exists(),
                "final_audit_version": final_audit_version.exists(),
                "final_audit_root": final_audit_root.exists(),
                "final_audit_root_stage": canonical_audit_payload.get("prediction_stage"),
                "result": result_path.exists(),
            }

        self.assertEqual(summary["candidate_races"], 1)
        self.assertEqual(summary["eligible_races"], 1)
        self.assertEqual(len(summary["removed_prediction_files"]), 1)
        self.assertEqual(len(summary["removed_audit_files"]), 1)
        self.assertEqual(summary["promoted_prediction_files"], [str(early_prediction_root)])
        self.assertEqual(summary["promoted_audit_files"], [str(final_audit_root)])
        self.assertGreater(summary["reclaimed_bytes"], 0)
        self.assertEqual(
            remaining_paths,
            {
                "canonical_prediction_root": True,
                "canonical_prediction_stage": "final",
                "early_prediction_version": False,
                "final_prediction_version": True,
                "early_audit_version": False,
                "final_audit_version": True,
                "final_audit_root": True,
                "final_audit_root_stage": "final",
                "result": True,
            },
        )

    def test_keeps_early_files_when_final_audit_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            race_id = "202607190202"
            predictions_dir = data_dir / "agent_predictions"
            early_path = predictions_dir / "_versions" / race_id / "early" / "task-early.json"
            final_path = predictions_dir / "_versions" / race_id / "final" / "task-final.json"
            _write_json(early_path, {"race_id": race_id, "prediction_stage": "early"})
            _write_json(final_path, {"race_id": race_id, "prediction_stage": "final"})

            summary = _cleanup_superseded_agent_early_predictions(
                dry_run=False,
                data_dir=data_dir,
            )
            early_exists = early_path.exists()
            final_exists = final_path.exists()

        self.assertEqual(summary["eligible_races"], 0)
        self.assertEqual(summary["skipped_reason_counts"], {"final_audit_missing_or_invalid": 1})
        self.assertTrue(early_exists)
        self.assertTrue(final_exists)

    def test_dry_run_reports_reclaim_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            race_id = "202607190203"
            predictions_dir = data_dir / "agent_predictions"
            audits_dir = data_dir / "agent_backtest_audit"
            early_path = predictions_dir / "_versions" / race_id / "early" / "task-early.json"
            final_path = predictions_dir / "_versions" / race_id / "final" / "task-final.json"
            final_audit_path = audits_dir / "_versions" / race_id / "final" / "task-final.json.gz"
            _write_json(early_path, {"race_id": race_id, "prediction_stage": "early"})
            _write_json(final_path, {"race_id": race_id, "prediction_stage": "final"})
            _write_gzip_json(
                final_audit_path,
                {"race_id": race_id, "prediction_stage": "final"},
            )

            summary = _cleanup_superseded_agent_early_predictions(
                dry_run=True,
                data_dir=data_dir,
            )
            early_exists = early_path.exists()

        self.assertEqual(len(summary["removed_prediction_files"]), 1)
        self.assertGreater(summary["reclaimed_bytes"], 0)
        self.assertTrue(early_exists)


if __name__ == "__main__":
    unittest.main()

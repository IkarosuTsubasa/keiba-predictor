from __future__ import annotations

import base64
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT_DIR / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import web_app  # noqa: E402
from race_job_store import create_job, get_job, update_job  # noqa: E402
from web_data import agent_predictions  # noqa: E402


def _payload(race_id: str, horse_no: int) -> dict:
    return {
        "race_id": race_id,
        "race_info": {
            "race_id": race_id,
            "race_date": "2026-07-19",
            "course": "東京",
            "surface": "芝",
            "distance": 1600,
        },
        "marks": {"◎": horse_no},
        "horse_scores": [
            {"horse_no": horse_no, "horse_name": f"テスト馬{horse_no}", "total_score": 20 + horse_no}
        ],
    }


def _callback_payload(prediction: dict) -> dict:
    content = json.dumps(prediction, ensure_ascii=False).encode("utf-8")
    return {"result": {"content_base64": base64.b64encode(content).decode("ascii")}}


class AgentPredictionStageTests(unittest.TestCase):
    def test_shadow_survives_remote_early_final_versions_and_settlement_without_public_change(self) -> None:
        self.check_inference_roundtrip('shadow_predictions', False)

    def test_active_inference_survives_remote_versions_and_settlement(self) -> None:
        self.check_inference_roundtrip('model_inference', True)

    def check_inference_roundtrip(self, field, authoritative) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'pipeline'
            base.mkdir()
            rid = '202605030399'
            stored = []
            with patch.object(web_app, 'BASE_DIR', base):
                for stage, selected in [('early', 1), ('final', 2)]:
                    prediction = _payload(rid, selected)
                    shadow = {'central_turf_history': {'status': 'predicted', 'authoritative': authoritative,
                              'candidate_top5': [6, 5, 4, 3, 2], 'stage_test_marker': stage}}
                    audit = {'race_id': rid, 'audit_kind': 'full', 'inputs': {'race_data': {'horses': []}},
                             'outputs': {'prediction': prediction}, field: shadow}
                    callback = _callback_payload(prediction)
                    callback['result']['audit_content_base64'] = base64.b64encode(
                        gzip.compress(json.dumps(audit).encode('utf-8'))).decode('ascii')
                    task = {'race_id': rid, 'task_id': 'shadow-'+stage,
                            'bundle_meta': {'prediction_stage': stage, 'final_prediction_expected': True}}
                    result = web_app._apply_remote_agent_prediction_result(task, callback)
                    public = json.loads(Path(result['prediction_path']).read_text(encoding='utf-8'))
                    self.assertEqual(public['marks'], prediction['marks'])
                    self.assertNotIn(field, public)
                    stored.append((Path(result['backtest_audit_path']), shadow))
                    callback['result']['audit_content_base64'] = base64.b64encode(
                        gzip.compress(json.dumps({**audit, field: {}}).encode('utf-8'))).decode('ascii')
                    web_app._apply_remote_agent_prediction_result(task, callback)
                agent_predictions.attach_result_to_agent_backtest_audit(
                    base, rid, {'race_id': rid, 'result': {'1st': 6, '2nd': 2, '3rd': 1}})
            self.assertNotEqual(stored[0][0], stored[1][0])
            for path, expected in stored:
                restored = agent_predictions.read_agent_backtest_audit_payload(path)
                self.assertEqual(restored[field], expected)
                self.assertEqual(restored['outcomes']['result']['result']['1st'], 6)

    def test_public_status_distinguishes_early_and_final_predictions(self) -> None:
        early = web_app._public_display_status(
            {"source_type": "agent_prediction", "prediction_stage": "early"},
            "open",
        )
        final = web_app._public_display_status(
            {"source_type": "agent_prediction", "prediction_stage": "final"},
            "open",
        )

        self.assertEqual(early["label"], "早期予測")
        self.assertEqual(final["label"], "最終予測")
        self.assertEqual(final["tone"], "final")

    def test_runtime_path_resolution_skips_invalid_final_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            race_id = "202605030300"
            early_path = agent_predictions.write_agent_prediction_version_payload(
                base_dir,
                race_id,
                "early",
                "task-early",
                _payload(race_id, 1),
            )
            invalid_final_path = agent_predictions.agent_prediction_version_path_for_race(
                base_dir,
                race_id,
                "final",
                "task-final",
            )
            invalid_final_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_final_path.write_text("{invalid", encoding="utf-8")

            resolved = agent_predictions.agent_prediction_path_for_race(base_dir, race_id)

        self.assertEqual(resolved, early_path)

    def test_callback_keeps_early_file_and_publishes_final_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            race_id = "202605030301"
            early_task = {
                "task_id": "task-early",
                "race_id": race_id,
                "task_type": "agent_prediction",
                "bundle_meta": {"prediction_stage": "early", "final_prediction_expected": True},
            }
            final_task = {
                "task_id": "task-final",
                "race_id": race_id,
                "task_type": "agent_prediction",
                "bundle_meta": {"prediction_stage": "final", "final_prediction_expected": True},
            }

            with patch.object(web_app, "BASE_DIR", base_dir):
                early_saved = web_app._apply_remote_agent_prediction_result(
                    early_task,
                    _callback_payload(_payload(race_id, 1)),
                )
                compatibility_path = Path(early_saved["compatibility_prediction_path"])
                early_compatibility_payload = json.loads(compatibility_path.read_text(encoding="utf-8"))
                final_saved = web_app._apply_remote_agent_prediction_result(
                    final_task,
                    _callback_payload(_payload(race_id, 2)),
                )
                resolved = agent_predictions.agent_prediction_path_for_race(base_dir, race_id)
                compatibility_audit = agent_predictions.read_agent_backtest_audit_payload(
                    agent_predictions.agent_backtest_audit_path_for_race(base_dir, race_id)
                )

            final_payload = json.loads(Path(final_saved["prediction_path"]).read_text(encoding="utf-8"))
            compatibility_payload = json.loads(compatibility_path.read_text(encoding="utf-8"))

        self.assertNotEqual(early_saved["prediction_path"], final_saved["prediction_path"])
        self.assertEqual(resolved, Path(final_saved["prediction_path"]))
        self.assertEqual(final_payload["prediction_stage"], "final")
        self.assertEqual(final_payload["marks"]["◎"], 2)
        self.assertEqual(compatibility_audit["prediction_stage"], "final")
        self.assertEqual(early_compatibility_payload["prediction_stage"], "early")
        self.assertEqual(compatibility_payload["prediction_stage"], "final")
        self.assertEqual(compatibility_payload["marks"]["◎"], 2)

    def test_final_success_callback_updates_metadata_without_reopening_settled_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202605030302",
                scope_key="central_turf",
                race_name="最終予測遅延成功テスト",
                race_date="2026-07-19",
                scheduled_off_time="2026-07-19T15:00:00",
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "")

            def _settled_while_final_running(row, now_text):
                row["status"] = "settled"
                row["prediction_stage"] = "early"
                row["early_prediction_ready_at"] = "2026-07-19T06:04:00"
                row["final_prediction_status"] = "running"
                row["current_v5_task_id"] = "task-final-late"
                row["settled_at"] = now_text

            update_job(base_dir, job_id, _settled_while_final_running)
            with patch.object(web_app, "BASE_DIR", base_dir):
                web_app._promote_job_after_remote_agent_prediction(
                    job_id=job_id,
                    task_id="task-final-late",
                    race_id="202605030302",
                    prediction_stage="final",
                    log_output="late final succeeded",
                )
            updated = get_job(base_dir, job_id) or {}

        self.assertEqual(updated["status"], "settled")
        self.assertEqual(updated["prediction_stage"], "final")
        self.assertEqual(updated["final_prediction_status"], "succeeded")
        self.assertEqual(updated["final_prediction_task_id"], "task-final-late")

    def test_final_success_callback_does_not_interrupt_result_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202605030304",
                scope_key="central_turf",
                race_name="結果取得中の最終予測成功テスト",
                race_date="2026-07-19",
                scheduled_off_time="2026-07-19T15:00:00",
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "")

            def _fetching_result_while_final_running(row, now_text):
                row["status"] = "fetching_agent_result"
                row["prediction_stage"] = "early"
                row["early_prediction_ready_at"] = "2026-07-19T06:04:00"
                row["final_prediction_status"] = "running"
                row["current_v5_task_id"] = "task-final-late"
                row["settling_started_at"] = now_text

            update_job(base_dir, job_id, _fetching_result_while_final_running)
            with patch.object(web_app, "BASE_DIR", base_dir):
                web_app._promote_job_after_remote_agent_prediction(
                    job_id=job_id,
                    task_id="task-final-late",
                    race_id="202605030304",
                    prediction_stage="final",
                    log_output="late final succeeded",
                )
            updated = get_job(base_dir, job_id) or {}

        self.assertEqual(updated["status"], "fetching_agent_result")
        self.assertEqual(updated["prediction_stage"], "final")
        self.assertEqual(updated["final_prediction_status"], "succeeded")

    def test_final_failure_callback_is_recorded_without_reopening_settled_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202605030303",
                scope_key="central_turf",
                race_name="最終予測遅延失敗テスト",
                race_date="2026-07-19",
                scheduled_off_time="2026-07-19T15:00:00",
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "")

            def _settled_while_final_running(row, now_text):
                row["status"] = "settled"
                row["prediction_stage"] = "early"
                row["early_prediction_ready_at"] = "2026-07-19T06:04:00"
                row["final_prediction_status"] = "running"
                row["current_v5_task_id"] = "task-final-late"
                row["settled_at"] = now_text

            update_job(base_dir, job_id, _settled_while_final_running)
            with patch.object(web_app, "BASE_DIR", base_dir):
                web_app._fail_job_after_remote_agent_prediction(
                    job_id=job_id,
                    task_id="task-final-late",
                    prediction_stage="final",
                    error_message="agent prediction timed out after 45 minutes",
                )
            updated = get_job(base_dir, job_id) or {}

        self.assertEqual(updated["status"], "settled")
        self.assertEqual(updated["prediction_stage"], "early")
        self.assertEqual(updated["final_prediction_status"], "failed")
        self.assertEqual(updated["pending_prediction_stage"], "")
        self.assertIn("timed out", updated["final_prediction_error"])

    def test_final_failure_restores_early_ready_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202605030301",
                scope_key="central_turf",
                race_name="最終予測失敗テスト",
                race_date="2026-07-19",
                scheduled_off_time="2026-07-19T15:00:00",
                target_surface="turf",
                target_distance="1600",
                target_track_condition="良",
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "")

            def _processing_final(row, now_text):
                row["status"] = "processing_final_prediction"
                row["prediction_stage"] = "early"
                row["early_prediction_ready_at"] = "2026-07-19T06:04:00"
                row["current_run_id"] = "202605030301"
                row["current_v5_task_id"] = "task-final"
                row["pending_prediction_stage"] = "final"

            update_job(base_dir, job_id, _processing_final)
            with patch.object(web_app, "BASE_DIR", base_dir):
                web_app._fail_job_after_remote_agent_prediction(
                    job_id=job_id,
                    task_id="task-final",
                    prediction_stage="final",
                    error_message="agent prediction timed out after 45 minutes",
                )
            updated = get_job(base_dir, job_id) or {}

        self.assertEqual(updated["status"], "agent_prediction_ready")
        self.assertEqual(updated["prediction_stage"], "early")
        self.assertEqual(updated["current_run_id"], "202605030301")
        self.assertEqual(updated["final_prediction_status"], "failed")


if __name__ == "__main__":
    unittest.main()

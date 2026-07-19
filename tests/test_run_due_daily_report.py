import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from web_admin import task_routes
from race_job_store import create_job, get_job, load_jobs, save_jobs, update_job
from v5_remote_tasks import create_task, find_latest_task_for_job, load_tasks, save_tasks
from keiba_llm_agent.parsers.netkeiba_result_parser import NetkeibaResultUnavailableError


class RunDueDailyReportTests(unittest.TestCase):
    def test_daily_report_skips_when_cleanup_did_not_run(self):
        called = []

        summary = task_routes._maybe_generate_daily_report_after_cleanup(
            cleanup_summary={"ran": False, "jst_date": "2026-06-21"},
            daily_report_generator=lambda date_text: called.append(date_text),
        )

        self.assertFalse(summary["attempted"])
        self.assertFalse(summary["ran"])
        self.assertEqual(summary["reason"], "cleanup_not_ran")
        self.assertEqual(called, [])

    def test_daily_report_runs_after_cleanup(self):
        called = []

        def generate(date_text=""):
            called.append(date_text)
            return {
                "slug": "20260621",
                "title": "2026年6月21日 AI予測日報",
                "target_date": "2026-06-21",
                "target_date_label": "2026年6月21日",
                "engine": "gemini",
                "engine_label": "Gemini",
                "mode": "llm",
                "public_url": "/keiba/reports/20260621",
            }

        with patch.dict(os.environ, {"RUN_DUE_DAILY_REPORT_ENABLED": "1"}):
            summary = task_routes._maybe_generate_daily_report_after_cleanup(
                cleanup_summary={"ran": True, "jst_date": "2026-06-21"},
                daily_report_generator=generate,
            )

        self.assertEqual(called, ["2026-06-21"])
        self.assertTrue(summary["attempted"])
        self.assertTrue(summary["ran"])
        self.assertEqual(summary["engine"], "gemini")
        self.assertEqual(summary["public_url"], "/keiba/reports/20260621")

    def test_daily_report_runs_once_after_cleanup_already_happened_today(self):
        called = []

        def generate(date_text=""):
            called.append(date_text)
            return {
                "slug": "20260621",
                "title": "2026年6月21日 AI予測日報",
                "target_date": "2026-06-21",
                "target_date_label": "2026年6月21日",
                "engine": "gemini",
                "engine_label": "Gemini",
                "mode": "llm",
                "public_url": "/keiba/reports/20260621",
            }

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            cleanup_summary = {
                "ran": False,
                "reason": "already_cleaned_today",
                "jst_date": "2026-06-21",
            }
            with patch.dict(os.environ, {"RUN_DUE_DAILY_REPORT_ENABLED": "1"}):
                first = task_routes._maybe_generate_daily_report_after_cleanup(
                    cleanup_summary=cleanup_summary,
                    daily_report_generator=generate,
                    base_dir=base_dir,
                )
                second = task_routes._maybe_generate_daily_report_after_cleanup(
                    cleanup_summary=cleanup_summary,
                    daily_report_generator=generate,
                    base_dir=base_dir,
                )

        self.assertEqual(called, ["2026-06-21"])
        self.assertTrue(first["ran"])
        self.assertEqual(second["reason"], "already_reported_today")
        self.assertEqual(second["public_url"], "/keiba/reports/20260621")

    def test_stale_agent_prediction_job_is_expired(self):
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202654062108",
                scope_key="local",
                race_name="テスト",
                race_date="2026-06-21",
                scheduled_off_time="2026-06-21T10:00:00",
                lead_minutes=60,
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "").strip()
            task = create_task(
                base_dir,
                job_id=job_id,
                race_id="202654062108",
                scope_key="local",
                task_type="agent_prediction",
            )
            task_id = str(task.get("task_id", "") or "").strip()

            jobs = load_jobs(base_dir)
            for row in jobs:
                if str(row.get("job_id", "") or "").strip() == job_id:
                    row["status"] = "processing_agent_prediction"
                    row["current_v5_task_id"] = task_id
                    row["updated_at"] = "2026-06-21T09:01:00"
                    row["predictor_status"] = "running"
            save_jobs(base_dir, jobs)

            tasks = load_tasks(base_dir)
            for row in tasks:
                if str(row.get("task_id", "") or "").strip() == task_id:
                    row["status"] = "dispatched"
                    row["created_at"] = "2026-06-21T09:01:00"
                    row["started_at"] = "2026-06-21T09:01:00"
                    row["updated_at"] = "2026-06-21T09:01:00"
            save_tasks(base_dir, tasks)

            with patch.dict(os.environ, {"RUN_DUE_AGENT_PREDICTION_STALE_MINUTES": "180"}):
                expired = task_routes._expire_stale_agent_prediction_jobs(
                    base_dir=base_dir,
                    load_race_jobs=load_jobs,
                    update_race_job=update_job,
                    now_dt=datetime(2026, 6, 21, 13, 0, 0),
                )

            updated_job = get_job(base_dir, job_id) or {}
            updated_task = find_latest_task_for_job(base_dir, job_id) or {}

        self.assertEqual([item.get("job_id") for item in expired], [job_id])
        self.assertEqual(updated_job.get("status"), "failed")
        self.assertIn("timed out", updated_job.get("error_message", ""))
        self.assertEqual(updated_task.get("status"), "expired")

    def test_failed_agent_prediction_is_requeued_after_backoff(self):
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202654062109",
                scope_key="local",
                race_name="テスト",
                race_date="2026-06-21",
                scheduled_off_time="2026-06-21T10:00:00",
                lead_minutes=60,
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "").strip()
            jobs = load_jobs(base_dir)
            for row in jobs:
                if str(row.get("job_id", "") or "").strip() == job_id:
                    row["status"] = "failed"
                    row["updated_at"] = "2026-06-21T12:50:00"
                    row["predictor_status"] = "failed"
                    row["error_message"] = "agent prediction timed out after 45 minutes"
            save_jobs(base_dir, jobs)

            with patch.dict(
                os.environ,
                {
                    "RUN_DUE_AGENT_PREDICTION_AUTO_RETRY_ENABLED": "1",
                    "RUN_DUE_AGENT_PREDICTION_MAX_RETRIES": "2",
                    "RUN_DUE_AGENT_PREDICTION_RETRY_DELAYS_MINUTES": "5,15",
                },
            ):
                retried = task_routes._auto_retry_failed_agent_prediction_jobs(
                    base_dir=base_dir,
                    load_race_jobs=load_jobs,
                    update_race_job=update_job,
                    now_dt=datetime(2026, 6, 21, 13, 0, 0),
                )
            updated_job = get_job(base_dir, job_id) or {}

        self.assertEqual([item.get("job_id") for item in retried], [job_id])
        self.assertEqual(updated_job.get("status"), "queued_agent_prediction")
        self.assertEqual(updated_job.get("agent_prediction_retry_count"), 1)
        self.assertEqual(updated_job.get("predictor_status"), "queued")
        self.assertIn("timed out", updated_job.get("agent_prediction_last_retry_error", ""))

    def test_failed_final_prediction_retries_without_losing_early_prediction(self):
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202654062109",
                scope_key="local",
                race_name="最終予測再試行テスト",
                race_date="2026-06-21",
                scheduled_off_time="2026-06-21T14:00:00",
                lead_minutes=60,
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "").strip()
            jobs = load_jobs(base_dir)
            for row in jobs:
                if str(row.get("job_id", "") or "").strip() == job_id:
                    row["status"] = "agent_prediction_ready"
                    row["prediction_stage"] = "early"
                    row["early_prediction_ready_at"] = "2026-06-21T06:04:00"
                    row["current_run_id"] = "202654062109"
                    row["ready_at"] = "2026-06-21T06:04:00"
                    row["final_prediction_status"] = "failed"
                    row["final_prediction_error"] = "agent prediction timed out after 45 minutes"
                    row["pending_prediction_stage"] = "final"
                    row["updated_at"] = "2026-06-21T13:20:00"
                    row["predictor_status"] = "succeeded"
            save_jobs(base_dir, jobs)

            with patch.dict(
                os.environ,
                {
                    "RUN_DUE_AGENT_PREDICTION_AUTO_RETRY_ENABLED": "1",
                    "RUN_DUE_AGENT_PREDICTION_MAX_RETRIES": "2",
                    "RUN_DUE_AGENT_PREDICTION_RETRY_DELAYS_MINUTES": "5,15",
                },
            ):
                retried = task_routes._auto_retry_failed_agent_prediction_jobs(
                    base_dir=base_dir,
                    load_race_jobs=load_jobs,
                    update_race_job=update_job,
                    now_dt=datetime(2026, 6, 21, 13, 25, 0),
                )
            updated_job = get_job(base_dir, job_id) or {}

        self.assertEqual([item.get("job_id") for item in retried], [job_id])
        self.assertEqual(updated_job.get("status"), "queued_final_prediction")
        self.assertEqual(updated_job.get("prediction_stage"), "early")
        self.assertEqual(updated_job.get("current_run_id"), "202654062109")
        self.assertEqual(updated_job.get("final_prediction_retry_count"), 1)

    def test_canceled_race_agent_prediction_is_not_retried(self):
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202654062110",
                scope_key="local",
                race_name="テスト",
                race_date="2026-06-21",
                scheduled_off_time="2026-06-21T10:00:00",
                lead_minutes=60,
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "").strip()
            jobs = load_jobs(base_dir)
            for row in jobs:
                if str(row.get("job_id", "") or "").strip() == job_id:
                    row["status"] = "failed"
                    row["updated_at"] = "2026-06-21T12:00:00"
                    row["error_message"] = "開催中止のため取り止め"
            save_jobs(base_dir, jobs)

            retried = task_routes._auto_retry_failed_agent_prediction_jobs(
                base_dir=base_dir,
                load_race_jobs=load_jobs,
                update_race_job=update_job,
                now_dt=datetime(2026, 6, 21, 13, 0, 0),
            )
            updated_job = get_job(base_dir, job_id) or {}

        self.assertEqual(retried, [])
        self.assertEqual(updated_job.get("status"), "failed")

    def test_agent_result_remains_due_fifteen_minutes_after_off_time(self):
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202655062508",
                scope_key="local",
                race_name="結果待機テスト",
                race_date="2026-06-25",
                scheduled_off_time="2026-06-25T10:00:00",
                lead_minutes=60,
                job_source="agent_prediction",
            )
            jobs = load_jobs(base_dir)
            for row in jobs:
                if str(row.get("job_id", "") or "") == str(job.get("job_id", "") or ""):
                    row["status"] = "agent_prediction_ready"
                    row["predictor_status"] = "succeeded"
            save_jobs(base_dir, jobs)

            before = task_routes._agent_result_diagnostics(
                base_dir=base_dir,
                load_race_jobs=lambda: load_jobs(base_dir),
                now_dt=datetime(2026, 6, 25, 10, 14, 59),
            )
            at_due = task_routes._agent_result_diagnostics(
                base_dir=base_dir,
                load_race_jobs=lambda: load_jobs(base_dir),
                now_dt=datetime(2026, 6, 25, 10, 15, 0),
            )

        self.assertEqual(before[0]["reason"], "wait_15min")
        self.assertEqual(at_due[0]["reason"], "eligible")

    def test_agent_result_is_still_due_while_final_prediction_is_processing(self):
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202655062508",
                scope_key="local",
                race_name="最終予測実行中の結果取得テスト",
                race_date="2026-06-25",
                scheduled_off_time="2026-06-25T10:00:00",
                job_source="agent_prediction",
            )
            jobs = load_jobs(base_dir)
            for row in jobs:
                if str(row.get("job_id", "") or "") == str(job.get("job_id", "") or ""):
                    row["status"] = "processing_final_prediction"
                    row["prediction_stage"] = "early"
                    row["early_prediction_ready_at"] = "2026-06-25T06:04:00"
                    row["final_prediction_status"] = "running"
                    row["predictor_status"] = "running"
            save_jobs(base_dir, jobs)

            at_due = task_routes._agent_result_diagnostics(
                base_dir=base_dir,
                load_race_jobs=lambda: load_jobs(base_dir),
                now_dt=datetime(2026, 6, 25, 10, 15, 0),
            )

        self.assertEqual(at_due[0]["reason"], "eligible")

    def test_unavailable_agent_result_is_terminal_without_run_due_error(self):
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202655062509",
                scope_key="local",
                race_name="テスト",
                race_date="2026-06-25",
                scheduled_off_time="2026-06-25T10:00:00",
                lead_minutes=60,
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "").strip()

            jobs = load_jobs(base_dir)
            for row in jobs:
                if str(row.get("job_id", "") or "").strip() == job_id:
                    row["status"] = "agent_prediction_ready"
                    row["predictor_status"] = "succeeded"
            save_jobs(base_dir, jobs)

            def scan_due_fixed(target_base_dir):
                return []

            def scan_due_diagnostics_fixed(target_base_dir):
                return []

            unavailable = NetkeibaResultUnavailableError(
                race_id="202655062509",
                reason="天候不良のため取り止めとなりました。",
            )
            with patch.dict(os.environ, {"PIPELINE_AUTO_DISCOVER_RACES_ENABLED": "0"}, clear=False):
                with patch.object(task_routes, "_jst_now", return_value=datetime(2026, 6, 25, 13, 0, 0)):
                    with patch.object(task_routes, "_fetch_and_save_agent_result_for_job", side_effect=unavailable):
                        summary = task_routes.run_due_jobs_once(
                            base_dir=base_dir,
                            scan_due_race_jobs=scan_due_fixed,
                            scan_due_race_job_diagnostics=scan_due_diagnostics_fixed,
                            load_race_jobs=load_jobs,
                            update_race_job=update_job,
                            compute_race_job_initial_status=lambda row: str(row.get("status", "") or ""),
                        )

            updated_job = get_job(base_dir, job_id) or {}

        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["agent_result_unavailable_count"], 1)
        self.assertEqual(summary["agent_result_unavailable_job_ids"], [job_id])
        self.assertEqual(updated_job.get("status"), "failed")
        self.assertIn("取り止め", updated_job.get("error_message", ""))
        self.assertEqual(updated_job.get("settlement_status"), "failed")

    def test_compact_run_due_summary_keeps_daily_report(self):
        compact = task_routes._compact_run_due_summary(
            {
                "daily_report": {
                    "attempted": True,
                    "ran": True,
                    "slug": "20260621",
                    "engine_label": "Gemini",
                    "public_url": "/keiba/reports/20260621",
                }
            }
        )

        self.assertTrue(compact["daily_report"]["attempted"])
        self.assertTrue(compact["daily_report"]["ran"])
        self.assertEqual(compact["daily_report"]["slug"], "20260621")
        self.assertEqual(compact["daily_report"]["engine_label"], "Gemini")


if __name__ == "__main__":
    unittest.main()

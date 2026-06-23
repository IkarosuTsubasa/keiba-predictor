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

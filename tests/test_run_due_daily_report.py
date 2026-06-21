import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from web_admin import task_routes


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

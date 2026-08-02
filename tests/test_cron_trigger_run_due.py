from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from cron_trigger_run_due import _format_http_error_detail


class CronTriggerRunDueTests(unittest.TestCase):
    def test_http_error_detail_prioritizes_actionable_errors(self) -> None:
        detail = json.dumps(
            {
                "ok": False,
                "auto_discovery": {"error_count": 12, "discovered_count": 83},
                "queued_job_ids": [f"job-{index}" for index in range(100)],
                "errors": [
                    {
                        "kind": "agent_result_fetch",
                        "job_id": "job-cancelled",
                        "error": "failed to parse top3 result from netkeiba result HTML",
                    }
                ],
            }
        )

        preview = _format_http_error_detail(detail)

        self.assertIn("agent_result_fetch", preview)
        self.assertIn("job-cancelled", preview)
        self.assertIn('"auto_discovery_error_count":12', preview)
        self.assertNotIn("job-99", preview)

    def test_http_error_detail_keeps_plain_text(self) -> None:
        self.assertEqual(_format_http_error_detail("  upstream   failed  "), "upstream failed")


if __name__ == "__main__":
    unittest.main()

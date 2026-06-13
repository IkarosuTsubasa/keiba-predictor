import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import race_discovery
from race_job_store import compute_initial_status, create_job, load_jobs, scan_due_diagnostics, scan_due_jobs, update_job
from web_admin import task_routes


CENTRAL_LIST_HTML = """
<div class="RaceList_Body RaceList_Top">
  <div class="RaceList_Box clearfix">
    <a href="../race/shutuba.html?race_id=202605030301">1R 3歳未勝利 09:55 ダ1600m 16頭</a>
    <a href="../race/shutuba.html?race_id=202605030302">2R 3歳未勝利 10:25 芝2400m 18頭</a>
  </div>
</div>
"""


LOCAL_LIST_HTML = """
<div class="RaceList_Body RaceList_Top">
  <div class="RaceList_Box clearfix">
    <a href="../race/shutuba.html?race_id=202654061301">1R C3ー5 15:35 ダ1400m 10頭</a>
    <a href="../race/shutuba.html?race_id=202665061312">12R B2ー4 20:30 ダ200m 8頭</a>
  </div>
</div>
"""


class RaceDiscoveryTests(unittest.TestCase):
    def test_central_discovery_uses_constructed_race_list_sub_url(self):
        requested = []

        def fake_fetcher(url, timeout=30):
            requested.append(url)
            if "race_list_sub.html?kaisai_date=20260613&current_group=1020260613" in url:
                return CENTRAL_LIST_HTML.encode("utf-8")
            return b"<html></html>"

        summary = race_discovery.discover_races_for_date(
            "2026-06-13",
            include_local=False,
            fetcher=fake_fetcher,
        )

        self.assertIn("https://race.netkeiba.com/top/", requested)
        self.assertEqual(summary["source_counts"]["central"], 2)
        self.assertEqual([row["race_id"] for row in summary["races"]], ["202605030301", "202605030302"])
        self.assertEqual(
            summary["races"][0]["shutuba_url"],
            "https://race.netkeiba.com/race/shutuba.html?race_id=202605030301",
        )

    def test_local_discovery_uses_kaisai_id_candidates_and_keeps_banei(self):
        def fake_fetcher(url, timeout=30):
            if "kaisai_id=2026540613" in url:
                return LOCAL_LIST_HTML.encode("utf-8")
            return b"<div class='RaceList_Body RaceList_Top'></div>"

        summary = race_discovery.discover_races_for_date(
            "2026-06-13",
            include_central=False,
            fetcher=fake_fetcher,
        )

        self.assertEqual(summary["source_counts"]["local"], 2)
        self.assertEqual([row["race_id"] for row in summary["races"]], ["202654061301", "202665061312"])
        self.assertEqual(
            summary["races"][1]["shutuba_url"],
            "https://nar.netkeiba.com/race/shutuba.html?race_id=202665061312",
        )


class RunDueAutoDiscoveryTests(unittest.TestCase):
    def run_due_once(self, base_dir):
        def scan_due_fixed(target_base_dir):
            return scan_due_jobs(target_base_dir, now_text="2026-06-13T06:00:00")

        def scan_due_diagnostics_fixed(target_base_dir):
            return scan_due_diagnostics(target_base_dir, now_text="2026-06-13T06:00:00")

        return task_routes.run_due_jobs_once(
            base_dir=base_dir,
            scan_due_race_jobs=scan_due_fixed,
            scan_due_race_job_diagnostics=scan_due_diagnostics_fixed,
            load_race_jobs=load_jobs,
            update_race_job=update_job,
            compute_race_job_initial_status=compute_initial_status,
            create_race_job=create_job,
        )

    def fake_meta(self, race_id, source="", timeout=30):
        local = str(source or "").strip().lower() == "local"
        return {
            "race_id": race_id,
            "scope_key": "local" if local else "central_dirt",
            "race_name": "自動検出テスト",
            "location": "高知" if local else "東京",
            "race_date": "2026-06-13",
            "scheduled_off_time": "2026-06-13T23:59:00",
            "target_surface": "dirt",
            "target_distance": "1400",
            "target_track_condition": "良",
            "race_number": "1R",
            "source_url": f"https://{'nar' if local else 'race'}.netkeiba.com/race/shutuba.html?race_id={race_id}",
        }

    def test_run_due_creates_agent_jobs_when_today_has_no_tasks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            discovery_payload = {
                "target_date": "2026-06-13",
                "races": [
                    {"race_id": "202605030301", "source": "central"},
                    {"race_id": "202654061301", "source": "local"},
                ],
                "source_counts": {"central": 1, "local": 1},
                "errors": [],
            }
            with patch.dict(os.environ, {"PIPELINE_AUTO_DISCOVER_AFTER_HOUR_JST": "0"}, clear=False):
                with patch.object(task_routes, "_jst_now", return_value=datetime(2026, 6, 13, 6, 0, 0)):
                    with patch.object(task_routes, "discover_races_for_date", return_value=discovery_payload):
                        with patch.object(task_routes, "fetch_race_meta", side_effect=self.fake_meta):
                            summary = self.run_due_once(base_dir)

            self.assertEqual(summary["auto_discovery"]["reason"], "created")
            self.assertEqual(summary["auto_discovery"]["created_count"], 2)
            self.assertEqual(summary["agent_dispatched_count"], 0)
            self.assertEqual(summary["errors"], [])
            jobs = load_jobs(base_dir)
            self.assertEqual(len(jobs), 2)
            self.assertEqual({job["race_id"] for job in jobs}, {"202605030301", "202654061301"})
            self.assertTrue(all(job["job_source"] == "agent_prediction" for job in jobs))

    def test_run_due_skips_discovery_when_today_agent_job_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            create_job(
                base_dir,
                race_id="202605030301",
                scope_key="central_dirt",
                race_name="既存タスク",
                location="東京",
                race_date="2026-06-13",
                scheduled_off_time="2026-06-13T23:59:00",
                target_surface="dirt",
                target_distance="1600",
                target_track_condition="良",
                job_source="agent_prediction",
            )
            with patch.dict(os.environ, {"PIPELINE_AUTO_DISCOVER_AFTER_HOUR_JST": "0"}, clear=False):
                with patch.object(task_routes, "_jst_now", return_value=datetime(2026, 6, 13, 6, 0, 0)):
                    with patch.object(task_routes, "discover_races_for_date", side_effect=AssertionError("should not discover")):
                        summary = self.run_due_once(base_dir)

            self.assertEqual(summary["auto_discovery"]["reason"], "agent_jobs_exist")
            self.assertEqual(len(load_jobs(base_dir)), 1)

    def test_run_due_retries_incomplete_discovery_and_adds_missing_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            create_job(
                base_dir,
                race_id="202605030301",
                scope_key="central_dirt",
                race_name="既存タスク",
                location="東京",
                race_date="2026-06-13",
                scheduled_off_time="2026-06-13T23:59:00",
                target_surface="dirt",
                target_distance="1600",
                target_track_condition="良",
                job_source="agent_prediction",
            )
            state_path = base_dir / "data" / "_shared" / task_routes.RUN_DUE_AUTO_DISCOVERY_STATE_FILE
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                '{"2026-06-13":{"completed":false,"discovered_count":2,"created_count":1,"reason":"no_new_jobs"}}',
                encoding="utf-8",
            )
            discovery_payload = {
                "target_date": "2026-06-13",
                "races": [
                    {"race_id": "202605030301", "source": "central"},
                    {"race_id": "202654061301", "source": "local"},
                ],
                "source_counts": {"central": 1, "local": 1},
                "errors": [],
            }
            with patch.dict(os.environ, {"PIPELINE_AUTO_DISCOVER_AFTER_HOUR_JST": "0"}, clear=False):
                with patch.object(task_routes, "_jst_now", return_value=datetime(2026, 6, 13, 6, 0, 0)):
                    with patch.object(task_routes, "discover_races_for_date", return_value=discovery_payload):
                        with patch.object(task_routes, "fetch_race_meta", side_effect=self.fake_meta):
                            summary = self.run_due_once(base_dir)

            self.assertEqual(summary["auto_discovery"]["reason"], "created")
            self.assertEqual(summary["auto_discovery"]["created_count"], 1)
            self.assertEqual(summary["auto_discovery"]["skipped_count"], 1)
            self.assertEqual({job["race_id"] for job in load_jobs(base_dir)}, {"202605030301", "202654061301"})


if __name__ == "__main__":
    unittest.main()

import json
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


def local_list_html_for_race_ids(race_ids):
    links = "\n".join(
        f'    <a href="../race/shutuba.html?race_id={race_id}">{int(str(race_id)[-2:])}R テスト</a>'
        for race_id in race_ids
    )
    return f"""
<div class="RaceList_Body RaceList_Top">
  <div class="RaceList_Box clearfix">
{links}
  </div>
</div>
"""


def local_shutuba_page_html(race_name, time_text="18:30", distance="1400", venue="川崎"):
    return f"""
<div class="RaceList_Item02">
  <h1 class="RaceName">{race_name}</h1>
  <div class="RaceData01">{time_text}発走 / ダ{distance}m (左) / 馬場:良</div>
  <div class="RaceData02"><span>{venue}</span></div>
</div>
"""


def local_canceled_shutuba_page_html(race_name, time_text="18:30", distance="1400", venue="川崎"):
    return local_shutuba_page_html(race_name, time_text, distance, venue).replace(
        "</div>\n",
        "<p>取止</p>\n</div>\n",
        1,
    )


LOCAL_INVALID_RACE_PAGE_HTML = """
<html><body><p>レース情報が見つかりません。</p></body></html>
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

    def test_local_discovery_supplements_july6_internal_missing_races(self):
        morioka_ids = [f"2026350706{race_no:02d}" for race_no in range(1, 13) if race_no != 11]
        kawasaki_ids = [f"2026450706{race_no:02d}" for race_no in range(1, 13) if race_no != 8]
        list_html = local_list_html_for_race_ids(morioka_ids + kawasaki_ids).encode("utf-8")

        def fake_fetcher(url, timeout=30):
            if "race_list_sub.html" in url:
                return list_html
            if "race_id=202635070611" in url:
                return local_shutuba_page_html("茉莉花特別(A)", "17:40", "1600", "盛岡").encode("utf-8")
            if "race_id=202645070608" in url:
                return local_shutuba_page_html("アルテミス賞(C2)", "18:30", "2000", "川崎").encode("utf-8")
            return LOCAL_INVALID_RACE_PAGE_HTML.encode("utf-8")

        summary = race_discovery.discover_races_for_date(
            "2026-07-06",
            include_central=False,
            fetcher=fake_fetcher,
        )

        race_ids = [row["race_id"] for row in summary["races"]]
        self.assertIn("202635070611", race_ids)
        self.assertIn("202645070608", race_ids)
        self.assertEqual(len([race_id for race_id in race_ids if race_id.startswith("2026350706")]), 12)
        self.assertEqual(len([race_id for race_id in race_ids if race_id.startswith("2026450706")]), 12)
        self.assertEqual(summary["source_results"][0]["supplemented_count"], 2)
        self.assertEqual(summary["errors"], [])

    def test_local_discovery_warns_for_unresolved_internal_missing_race(self):
        morioka_ids = [f"2026350706{race_no:02d}" for race_no in range(1, 13) if race_no != 11]
        list_html = local_list_html_for_race_ids(morioka_ids).encode("utf-8")

        def fake_fetcher(url, timeout=30):
            if "race_list_sub.html" in url:
                return list_html
            return LOCAL_INVALID_RACE_PAGE_HTML.encode("utf-8")

        summary = race_discovery.discover_races_for_date(
            "2026-07-06",
            include_central=False,
            fetcher=fake_fetcher,
        )

        self.assertNotIn("202635070611", [row["race_id"] for row in summary["races"]])
        self.assertEqual(summary["source_results"][0]["supplemented_count"], 0)
        self.assertEqual(summary["source_results"][0]["supplement_errors"], [])
        self.assertTrue(summary["source_results"][0]["supplement_warnings"])
        self.assertEqual(summary["warnings"][0]["stage"], "supplement")
        self.assertEqual(summary["errors"], [])

    def test_local_discovery_does_not_supplement_canceled_internal_missing_race(self):
        morioka_ids = [f"2026350706{race_no:02d}" for race_no in range(1, 13) if race_no != 11]
        list_html = local_list_html_for_race_ids(morioka_ids).encode("utf-8")

        def fake_fetcher(url, timeout=30):
            if "race_list_sub.html" in url:
                return list_html
            if "race_id=202635070611" in url:
                return local_canceled_shutuba_page_html("茉莉花特別(A)", "17:40", "1600", "盛岡").encode("utf-8")
            return LOCAL_INVALID_RACE_PAGE_HTML.encode("utf-8")

        summary = race_discovery.discover_races_for_date(
            "2026-07-06",
            include_central=False,
            fetcher=fake_fetcher,
        )

        self.assertNotIn("202635070611", [row["race_id"] for row in summary["races"]])
        self.assertEqual(summary["source_results"][0]["supplemented_count"], 0)
        self.assertTrue(summary["warnings"])
        self.assertEqual(summary["errors"], [])


class RunDueAutoDiscoveryTests(unittest.TestCase):
    def fake_dispatch(self, base_dir, task, job):
        return {
            "task_id": str((task or {}).get("task_id", "") or "").strip(),
            "workflow": "agent-prediction-remote.yml",
            "ref": "test",
            "race_id": str((job or {}).get("race_id", "") or "").strip(),
            "dispatch_http_status": 204,
        }

    def run_due_once(self, base_dir):
        def scan_due_fixed(target_base_dir):
            return scan_due_jobs(target_base_dir, now_text="2026-06-13T06:00:00")

        def scan_due_diagnostics_fixed(target_base_dir):
            return scan_due_diagnostics(target_base_dir, now_text="2026-06-13T06:00:00")

        with patch.object(task_routes, "_dispatch_agent_prediction_task", side_effect=self.fake_dispatch):
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
            self.assertEqual(summary["agent_dispatched_count"], 1)
            self.assertEqual(summary["agent_dispatch_limit"], 1)
            self.assertEqual(summary["errors"], [])
            jobs = load_jobs(base_dir)
            self.assertEqual(len(jobs), 2)
            self.assertEqual({job["race_id"] for job in jobs}, {"202605030301", "202654061301"})
            self.assertTrue(all(job["job_source"] == "agent_prediction" for job in jobs))
            self.assertTrue(all(job["agent_prediction_schedule"] == "daily_morning" for job in jobs))
            self.assertTrue(
                all(job["agent_prediction_queue_after_time"] == "2026-06-13T06:00:00" for job in jobs)
            )
            self.assertEqual(
                sorted(job["status"] for job in jobs),
                ["processing_agent_prediction", "queued_agent_prediction"],
            )

    def test_run_due_discovers_full_list_when_today_agent_job_exists(self):
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
            discovery_payload = {
                "target_date": "2026-06-13",
                "races": [{"race_id": "202605030301", "source": "central"}],
                "source_counts": {"central": 1, "local": 0},
                "errors": [],
            }
            with patch.dict(os.environ, {"PIPELINE_AUTO_DISCOVER_AFTER_HOUR_JST": "0"}, clear=False):
                with patch.object(task_routes, "_jst_now", return_value=datetime(2026, 6, 13, 6, 0, 0)):
                    with patch.object(task_routes, "discover_races_for_date", return_value=discovery_payload):
                        summary = self.run_due_once(base_dir)

            self.assertEqual(summary["auto_discovery"]["reason"], "no_new_jobs")
            self.assertEqual(summary["auto_discovery"]["skipped_count"], 1)
            self.assertEqual(summary["agent_dispatched_count"], 1)
            self.assertEqual(len(load_jobs(base_dir)), 1)

    def test_daily_morning_queue_ignores_lead_minutes_after_six(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202605030301",
                scope_key="central_dirt",
                race_name="早朝キューテスト",
                race_date="2026-06-13",
                scheduled_off_time="2026-06-13T23:59:00",
                target_surface="dirt",
                target_distance="1600",
                target_track_condition="良",
                lead_minutes=60,
                job_source="agent_prediction",
            )
            task_routes._mark_daily_agent_prediction_schedule(
                base_dir,
                update_job,
                str(job.get("job_id", "") or ""),
                "2026-06-13",
            )

            before = scan_due_jobs(base_dir, now_text="2026-06-13T05:59:00")
            at_start = scan_due_jobs(base_dir, now_text="2026-06-13T06:00:00")
            jobs = load_jobs(base_dir)

        self.assertEqual(before, [])
        self.assertEqual(len(at_start), 1)
        self.assertEqual(jobs[0]["status"], "queued_agent_prediction")
        self.assertEqual(jobs[0]["lead_minutes"], 60)

    def test_agent_prediction_queue_selects_earliest_off_time(self):
        jobs = [
            {
                "job_id": "late",
                "race_id": "202654061312",
                "scheduled_off_time": "2026-06-13T20:30:00",
                "status": "queued_agent_prediction",
            },
            {
                "job_id": "early",
                "race_id": "202605030301",
                "scheduled_off_time": "2026-06-13T09:55:00",
                "status": "queued_agent_prediction",
            },
        ]

        selected = task_routes.pick_next_agent_prediction_job_id(load_race_jobs=lambda: jobs)

        self.assertEqual(selected, "early")

    def test_final_prediction_is_queued_exactly_thirty_minutes_before_off(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            job = create_job(
                base_dir,
                race_id="202605030301",
                scope_key="central_dirt",
                race_name="最終予測テスト",
                race_date="2026-06-13",
                scheduled_off_time="2026-06-13T10:00:00",
                target_surface="dirt",
                target_distance="1600",
                target_track_condition="良",
                job_source="agent_prediction",
            )
            job_id = str(job.get("job_id", "") or "")
            task_routes._mark_daily_agent_prediction_schedule(base_dir, update_job, job_id, "2026-06-13")

            def mark_early_ready(row, now_text):
                row["status"] = "agent_prediction_ready"
                row["prediction_stage"] = "early"
                row["early_prediction_ready_at"] = "2026-06-13T06:04:00"
                row["final_prediction_status"] = "waiting"
                row["predictor_status"] = "succeeded"

            update_job(base_dir, job_id, mark_early_ready)
            before = scan_due_jobs(base_dir, now_text="2026-06-13T09:29:59")
            at_due = scan_due_jobs(base_dir, now_text="2026-06-13T09:30:00")
            updated = load_jobs(base_dir)[0]

        self.assertEqual(before, [])
        self.assertEqual(len(at_due), 1)
        self.assertEqual(updated["status"], "queued_final_prediction")
        self.assertEqual(updated["pending_prediction_stage"], "final")
        self.assertEqual(updated["final_prediction_due_time"], "2026-06-13T09:30:00")

    def test_final_prediction_queue_has_priority_over_early_queue(self):
        jobs = [
            {
                "job_id": "early",
                "race_id": "202605030301",
                "scheduled_off_time": "2026-06-13T09:55:00",
                "status": "queued_agent_prediction",
            },
            {
                "job_id": "final",
                "race_id": "202654061312",
                "scheduled_off_time": "2026-06-13T20:30:00",
                "status": "queued_final_prediction",
            },
        ]

        selected = task_routes.pick_next_agent_prediction_job_id(load_race_jobs=lambda: jobs)
        early = task_routes.pick_next_agent_prediction_job_id(
            load_race_jobs=lambda: jobs,
            prediction_stage="early",
        )

        self.assertEqual(selected, "final")
        self.assertEqual(early, "early")

    def test_minute_cron_dispatches_final_while_early_queue_waits_for_four_minute_slot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            early_job = create_job(
                base_dir,
                race_id="202605030301",
                scope_key="central_dirt",
                race_name="早期キュー",
                race_date="2026-06-13",
                scheduled_off_time="2026-06-13T20:00:00",
                target_surface="dirt",
                target_distance="1600",
                target_track_condition="良",
                job_source="agent_prediction",
            )
            final_job = create_job(
                base_dir,
                race_id="202605030302",
                scope_key="central_turf",
                race_name="最終キュー",
                race_date="2026-06-13",
                scheduled_off_time="2026-06-13T06:31:00",
                target_surface="turf",
                target_distance="1600",
                target_track_condition="良",
                job_source="agent_prediction",
            )

            update_job(
                base_dir,
                early_job["job_id"],
                lambda row, now_text: row.update({"status": "queued_agent_prediction", "pending_prediction_stage": "early"}),
            )
            update_job(
                base_dir,
                final_job["job_id"],
                lambda row, now_text: row.update(
                    {
                        "status": "queued_final_prediction",
                        "prediction_stage": "early",
                        "early_prediction_ready_at": "2026-06-13T06:00:00",
                        "final_prediction_status": "queued",
                        "pending_prediction_stage": "final",
                    }
                ),
            )

            with patch.dict(os.environ, {"PIPELINE_AUTO_DISCOVER_RACES_ENABLED": "0"}, clear=False):
                with patch.object(task_routes, "_jst_now", return_value=datetime(2026, 6, 13, 6, 1, 0)):
                    with patch.object(task_routes, "_dispatch_agent_prediction_task", side_effect=self.fake_dispatch):
                        summary = task_routes.run_due_jobs_once(
                            base_dir=base_dir,
                            scan_due_race_jobs=lambda target_base_dir: [],
                            scan_due_race_job_diagnostics=lambda target_base_dir: [],
                            load_race_jobs=load_jobs,
                            update_race_job=update_job,
                            compute_race_job_initial_status=compute_initial_status,
                            history_source="cron",
                        )
            jobs = {row["job_id"]: row for row in load_jobs(base_dir)}

        self.assertEqual(summary["agent_dispatched_count"], 1)
        self.assertEqual(summary["agent_dispatch_limit"], 0)
        self.assertEqual(summary["final_prediction_dispatch_limit"], 1)
        self.assertEqual(summary["agent_dispatch_results"][0]["prediction_stage"], "final")
        self.assertEqual(jobs[early_job["job_id"]]["status"], "queued_agent_prediction")
        self.assertEqual(jobs[final_job["job_id"]]["status"], "processing_final_prediction")

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

    def test_run_due_persists_auto_discovery_warnings_without_blocking_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            base_dir.mkdir(parents=True, exist_ok=True)
            discovery_payload = {
                "target_date": "2026-06-13",
                "races": [{"race_id": "202654061301", "source": "local"}],
                "source_counts": {"central": 0, "local": 1},
                "warnings": [
                    {
                        "source": "local",
                        "stage": "supplement",
                        "warnings": [{"race_id": "202654061302", "warning": "取止"}],
                    }
                ],
                "errors": [],
            }
            with patch.dict(os.environ, {"PIPELINE_AUTO_DISCOVER_AFTER_HOUR_JST": "0"}, clear=False):
                with patch.object(task_routes, "_jst_now", return_value=datetime(2026, 6, 13, 6, 0, 0)):
                    with patch.object(task_routes, "discover_races_for_date", return_value=discovery_payload):
                        with patch.object(task_routes, "fetch_race_meta", side_effect=self.fake_meta):
                            summary = self.run_due_once(base_dir)

            self.assertEqual(summary["auto_discovery"]["errors"], [])
            self.assertTrue(summary["auto_discovery"]["warnings"])
            state_path = base_dir / "data" / "_shared" / task_routes.RUN_DUE_AUTO_DISCOVERY_STATE_FILE
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertTrue(state["2026-06-13"]["completed"])
            self.assertEqual(state["2026-06-13"]["warning_count"], 1)


if __name__ == "__main__":
    unittest.main()

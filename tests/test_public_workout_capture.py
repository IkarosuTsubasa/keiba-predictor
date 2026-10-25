from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from keiba_llm_agent import main
from keiba_llm_agent.backtest.audit_snapshot import read_backtest_audit_payload
from keiba_llm_agent.fetchers import netkeiba_workout_fetcher as workouts
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo
from pipeline.web_data import agent_predictions


def race() -> RaceData:
    return RaceData(
        race_info=RaceInfo(race_id="202609040811", surface="芝", scope_key="central_turf",
                           scheduled_start_at="2026-10-25T15:40:00+09:00"),
        horses=[HorseEntry(horse_no=1, horse_id="2023100001", horse_name="テスト一"),
                HorseEntry(horse_no=2, horse_id="2023100002", horse_name="テスト二")],
    )


def page(grade="B") -> str:
    return f'''<link rel="canonical" href="https://race.netkeiba.com/race/oikiri.html?race_id=202609040811">
    <table class="OikiriTable">
      <tr class="HorseList"><td class="Umaban">1</td><td><a href="/horse/2023100001/">テスト一</a></td>
        <td class="Training_Critic">動き良い</td><td class="Rank_B">{grade}</td></tr>
      <tr class="HorseList"><td class="Umaban">2</td><td><a href="/horse/2023100002/">テスト二</a></td>
        <td class="Training_Critic">順調</td><td class="Rank_C">C</td></tr>
    </table><div>人気 1 オッズ 2.0</div>'''


def response(html=None, status=200):
    result = MagicMock()
    result.__enter__.return_value = result
    result.status_code = status
    result.iter_content.return_value = [(html if html is not None else page()).encode("utf-8")]
    return result


class PublicWorkoutCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"KEIBA_CAPTURE_PUBLIC_WORKOUTS": "1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_whitelist_missing_grade_and_identity_validation(self):
        result = workouts.parse_workout_page(page(), race())
        self.assertEqual(result["status"], "collected")
        self.assertEqual(result["available_grades"], 2)
        self.assertEqual(set(result["horses"]["1"]), {"horse_id", "grade", "critic"})
        missing = workouts.parse_workout_page(page("--"), race())
        self.assertEqual(missing["status"], "partial")
        self.assertIsNone(missing["horses"]["1"]["grade"])
        for html, expected in (
            (page().replace("202609040811", "202609040812"), "page_identity_unverified"),
            (page().replace("2023100001", "2023100999"), "roster_mismatch"),
            (page().replace('class="Umaban">2', 'class="Umaban">1'), "roster_mismatch"),
            (page().replace("OikiriTable", "Paywall"), "table_unavailable"),
            ("<html>Access denied</html>", "page_identity_unverified"),
        ):
            with self.subTest(expected=expected, html=html[:40]):
                parsed = workouts.parse_workout_page(html, race())
                self.assertEqual(parsed["status"], expected)
                self.assertEqual(parsed["horses"], {})

    def test_exact_raw_evidence_cache_and_append_only(self):
        start = datetime(2026, 10, 25, 6, 0, tzinfo=timezone.utc)
        with patch.object(workouts.requests, "get", return_value=response()) as get:
            with patch.object(workouts, "_now", return_value=start):
                first = workouts.capture_public_workout(race(), output_dir=self.root)
            paths = list(self.root.rglob("*.json.gz"))
            original = paths[0].read_bytes()
            with patch.object(workouts, "_now", return_value=start + timedelta(seconds=50)):
                cached = workouts.capture_public_workout(race(), output_dir=self.root)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(cached["received_at"], first["received_at"])
            self.assertIn("reused_at", cached)
            with patch.object(workouts, "_now", return_value=start + timedelta(seconds=650)):
                workouts.capture_public_workout(race(), output_dir=self.root)
            self.assertEqual(get.call_count, 2)
        self.assertEqual(len(list(self.root.rglob("*.json.gz"))), 2)
        self.assertEqual(paths[0].read_bytes(), original)
        raw = gzip.decompress(base64.b64decode(first["raw_response"]["data"]))
        self.assertEqual(raw, page().encode("utf-8"))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), first["raw_response"]["sha256"])
        self.assertTrue(first["received_before_scheduled_start"])
        self.assertIsNone(first["source_published_at"])
        self.assertFalse(first["used_for_scoring"])
        self.assertFalse(first["used_for_llm"])
        kwargs = get.call_args.kwargs
        self.assertEqual(kwargs["timeout"], (3, 6))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertNotIn("cookies", kwargs)
        self.assertNotIn("auth", kwargs)

    def test_restriction_timeout_size_and_unpublished_page_fail_open(self):
        for code in (401, 403, 429, 302, 500):
            with self.subTest(code=code), patch.object(workouts.requests, "get", return_value=response(status=code)) as get:
                result = workouts.capture_public_workout(race(), persist=False)
                self.assertEqual(result["status"], "restricted" if code in (401, 403, 429) else "http_error")
                self.assertEqual(get.call_count, 1)
        with patch.object(workouts.requests, "get", side_effect=requests.Timeout):
            result = workouts.capture_public_workout(race(), output_dir=self.root)
            self.assertEqual(result["status"], "fetch_failed")
        with patch.object(workouts.requests, "get") as get:
            self.assertEqual(workouts.capture_public_workout(race(), output_dir=self.root)["status"], "fetch_failed")
            get.assert_not_called()
        with patch.object(workouts.requests, "get", return_value=response("x" * (workouts.MAX_RESPONSE_BYTES + 1))):
            self.assertEqual(workouts.capture_public_workout(race(), persist=False)["status"], "fetch_failed")

    def test_dirt_supported_local_and_disabled_skip_network(self):
        data = race()
        data.race_info.surface = "ダート"
        data.race_info.scope_key = "central_dirt"
        with patch.object(workouts.requests, "get", return_value=response()) as get:
            self.assertEqual(workouts.capture_public_workout(data, persist=False)["status"], "collected")
            get.assert_called_once()
        data.race_info.source = "local"
        with patch.object(workouts.requests, "get") as get:
            self.assertEqual(workouts.capture_public_workout(data, persist=False)["status"], "unsupported")
            with patch.dict(os.environ, {"KEIBA_CAPTURE_PUBLIC_WORKOUTS": "0"}):
                self.assertEqual(workouts.capture_public_workout(race(), persist=False)["status"], "disabled")
            get.assert_not_called()

    def test_archive_failure_still_returns_evidence(self):
        with patch.object(workouts.requests, "get", return_value=response()), patch.object(Path, "mkdir", side_effect=OSError):
            result = workouts.capture_public_workout(race(), output_dir=self.root)
        self.assertEqual(result["status"], "collected")
        self.assertEqual(result["archive_error_type"], "OSError")
        self.assertIn("raw_response", result)

    def test_missing_start_and_post_race_are_not_pre_race_evidence(self):
        self.assertFalse(workouts._is_pre_race(race(), "2026-10-26T00:00:00+00:00"))
        data = race()
        data.race_info.scheduled_start_at = None
        self.assertIsNone(workouts._is_pre_race(data, "2026-10-25T00:00:00+00:00"))

    def test_online_entrypoints_capture_and_dry_run_does_not(self):
        data = race()
        saved = self.root / "race.json"
        observation = {"status": "restricted", "used_for_scoring": False}
        with (patch.object(main, "fetch_and_parse_netkeiba_race", return_value=data),
              patch.object(main, "enrich_race_data_with_recent_runs", return_value=data),
              patch.object(main, "save_race_data", return_value=saved),
              patch.object(main, "capture_public_workout", return_value=observation) as capture,
              patch.object(main, "run_analysis", return_value=(None, self.root / "prediction.json")) as analyze,
              patch.object(main, "DEFAULT_BACKTEST_AUDIT_DIR", self.root / "audit")):
            for call in (
                lambda: main.run_predict_race("test", skip_social=True),
                lambda: main.run_analyze_url("test"),
            ):
                call()
                self.assertEqual(analyze.call_args.kwargs["supplementary_observations"], {"public_workout": observation})
            self.assertEqual(capture.call_count, 2)
            main.run_analyze_url("test", dry_run=True)
            self.assertEqual(capture.call_count, 2)

    def test_audit_callback_and_result_preserve_evidence_without_model_inputs(self):
        data = race()
        before_race = data.model_dump()
        with patch.object(workouts.requests, "get", return_value=response()):
            observation = workouts.capture_public_workout(data, persist=False)
        prediction = Prediction(race_id=data.race_info.race_id, marks={"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5}, summary="テスト")
        before_prediction = prediction.model_dump()
        saved_race = self.root / "race.json"
        saved_race.write_text(data.model_dump_json(), encoding="utf-8")
        lessons = self.root / "lessons.json"
        lessons.write_text("[]", encoding="utf-8")
        with (patch.object(main, "RaceAnalysisAgent") as agent,
              patch.object(main, "create_llm_client"),
              patch.object(main, "DEFAULT_BACKTEST_AUDIT_DIR", self.root / "audit"),
              patch.object(main, "capture_public_workout") as capture):
            agent.return_value.run.return_value = prediction
            main.run_analysis(saved_race, self.root / "prediction.json", lessons_path=lessons,
                              supplementary_observations={"public_workout": observation})
            self.assertNotIn("supplementary_observations", agent.return_value.run.call_args.kwargs)
            self.assertEqual(agent.return_value.run.call_args.args[0].model_dump(), before_race)
            capture.assert_not_called()
        path = self.root / "audit" / f"{data.race_info.race_id}.json.gz"
        # Same base64/gzip transfer used by the GitHub Actions callback.
        transferred = base64.b64decode(base64.b64encode(path.read_bytes()))
        persisted = agent_predictions.write_agent_backtest_audit_bytes(self.root / "render", data.race_info.race_id, transferred)
        agent_predictions.attach_result_to_agent_backtest_audit(self.root / "render", data.race_info.race_id,
                                                               {"result": {"1st": 1, "2nd": 2, "3rd": 3}})
        restored = read_backtest_audit_payload(persisted)
        self.assertEqual(restored["inputs"]["supplementary_observations"]["public_workout"], observation)
        self.assertEqual(prediction.model_dump(), before_prediction)
        self.assertEqual(data.model_dump(), before_race)
        self.assertNotIn("supplementary_observations", json.loads((self.root / "prediction.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()

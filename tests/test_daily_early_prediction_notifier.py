import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import daily_early_prediction_notifier


class DailyEarlyPredictionNotifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.shared_dir = self.base_dir / "data" / "_shared"
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.target_date = "2026-07-23"
        self.race_ids = ["202605040101", "202654072301"]
        self.jobs = [
            {
                "job_id": f"job-{index}",
                "job_source": "agent_prediction",
                "race_id": race_id,
                "race_date": self.target_date,
                "agent_prediction_schedule": "daily_morning",
                "early_prediction_ready_at": "2026-07-23T07:00:00",
            }
            for index, race_id in enumerate(self.race_ids)
        ]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _load_jobs(self, base_dir):
        self.assertEqual(Path(base_dir), self.base_dir)
        return [dict(job) for job in self.jobs]

    def _write_discovery_state(self, *, completed=True):
        payload = {
            self.target_date: {
                "completed": completed,
                "discovered_count": len(self.race_ids),
                "created_race_ids": list(self.race_ids),
                "skipped_race_ids": [],
            }
        }
        (self.shared_dir / "run_due_auto_discovery_state.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_completion_waits_for_every_discovered_race(self) -> None:
        self._write_discovery_state()
        self.jobs[1]["early_prediction_ready_at"] = ""

        result = daily_early_prediction_notifier.daily_early_prediction_completion(
            self.base_dir,
            self._load_jobs,
            target_date=self.target_date,
        )

        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "early_predictions_incomplete")
        self.assertEqual(result["incomplete_race_ids"], [self.race_ids[1]])

    def test_completion_requires_successful_auto_discovery(self) -> None:
        self._write_discovery_state(completed=False)

        result = daily_early_prediction_notifier.daily_early_prediction_completion(
            self.base_dir,
            self._load_jobs,
            target_date=self.target_date,
        )

        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "auto_discovery_not_completed")

    def test_notification_is_sent_only_once_per_date(self) -> None:
        self._write_discovery_state()
        send_result = {
            "ok": True,
            "topic": "keiba-public-updates",
            "message_id": "message-1",
            "destination": "my_prediction",
        }
        with (
            patch.dict(
                os.environ,
                {"PIPELINE_DAILY_EARLY_PREDICTION_NOTIFY_ENABLED": "1"},
                clear=False,
            ),
            patch(
                "ntfy_notifier.publish_fcm_daily_early_prediction_complete_notification",
                return_value=send_result,
            ) as sender,
        ):
            first = (
                daily_early_prediction_notifier.maybe_send_daily_early_prediction_complete_notification(
                    self.base_dir,
                    self._load_jobs,
                    target_date=self.target_date,
                )
            )
            second = (
                daily_early_prediction_notifier.maybe_send_daily_early_prediction_complete_notification(
                    self.base_dir,
                    self._load_jobs,
                    target_date=self.target_date,
                )
            )

        self.assertTrue(first["ok"])
        self.assertFalse(first["skipped"])
        self.assertEqual(first["reason"], "notified")
        self.assertEqual(first["race_count"], 2)
        self.assertEqual(first["destination"], "my_prediction")
        self.assertTrue(second["ok"])
        self.assertTrue(second["skipped"])
        self.assertEqual(second["reason"], "already_notified")
        sender.assert_called_once_with(target_date=self.target_date, race_count=2)

        state_path = (
            self.shared_dir
            / daily_early_prediction_notifier.NOTIFICATION_STATE_DIR
            / f"{self.target_date}.json"
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "sent")
        self.assertEqual(state["message_id"], "message-1")

    def test_skipped_fcm_send_does_not_mark_date_as_notified(self) -> None:
        self._write_discovery_state()
        with patch(
            "ntfy_notifier.publish_fcm_daily_early_prediction_complete_notification",
            return_value={"ok": False, "skipped": True, "reason": "disabled"},
        ):
            result = (
                daily_early_prediction_notifier.maybe_send_daily_early_prediction_complete_notification(
                    self.base_dir,
                    self._load_jobs,
                    target_date=self.target_date,
                )
            )

        self.assertFalse(result["ok"])
        self.assertTrue(result["skipped"])
        self.assertEqual(result["reason"], "disabled")
        state_path = (
            self.shared_dir
            / daily_early_prediction_notifier.NOTIFICATION_STATE_DIR
            / f"{self.target_date}.json"
        )
        self.assertFalse(state_path.exists())

    def test_console_setting_overrides_environment_default(self) -> None:
        with patch.dict(
            os.environ,
            {"PIPELINE_DAILY_EARLY_PREDICTION_NOTIFY_ENABLED": "0"},
            clear=False,
        ):
            initial = (
                daily_early_prediction_notifier.load_daily_early_prediction_notification_settings(
                    self.base_dir
                )
            )
            saved = (
                daily_early_prediction_notifier.save_daily_early_prediction_notification_settings(
                    self.base_dir,
                    enabled=True,
                )
            )
            reloaded = (
                daily_early_prediction_notifier.load_daily_early_prediction_notification_settings(
                    self.base_dir
                )
            )

        self.assertFalse(initial["enabled"])
        self.assertEqual(initial["source"], "environment")
        self.assertTrue(saved["enabled"])
        self.assertTrue(reloaded["enabled"])
        self.assertEqual(reloaded["source"], "console")


if __name__ == "__main__":
    unittest.main()

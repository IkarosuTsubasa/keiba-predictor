import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import ntfy_notifier


class NtfyNotifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.prediction_dir = self.base_dir / "data" / "agent_predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_agent_prediction(self, *, race_id="202606100101", bet_decision="", confidence="low") -> None:
        payload = {
            "race_id": race_id,
            "race_info": {
                "race_id": race_id,
                "race_date": "2026-06-10",
                "course": "東京",
                "race_name": "テストステークス",
            },
            "marks": {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
            "strategy": {
                "bet_decision": bet_decision,
                "confidence": confidence,
            },
        }
        (self.prediction_dir / f"{race_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_default_high_evaluation_threshold_matches_public_label_boundary(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PIPELINE_AUTO_PREDICTION_NOTIFY_MIN_CONFIDENCE": "",
                "PIPELINE_HIGH_EVALUATION_NOTIFY_THRESHOLD": "",
            },
        ):
            self.assertEqual(ntfy_notifier.high_evaluation_notify_threshold(), 0.62)

    def test_agent_bet_decision_is_high_evaluation_even_with_low_confidence(self) -> None:
        race_id = "202606100101"
        self._write_agent_prediction(race_id=race_id, bet_decision="BET", confidence="low")

        with patch.dict(os.environ, {"KEIBA_AGENT_PREDICTIONS_DIR": ""}):
            result = ntfy_notifier.agent_prediction_notification_evaluation(self.base_dir, race_id)

        self.assertTrue(result["should_notify"])
        self.assertEqual(result["reason"], "high_evaluation_bet_decision")

    def test_agent_skip_decision_blocks_high_confidence_notification(self) -> None:
        race_id = "202606100102"
        self._write_agent_prediction(race_id=race_id, bet_decision="SKIP", confidence="high")

        with patch.dict(os.environ, {"KEIBA_AGENT_PREDICTIONS_DIR": ""}):
            result = ntfy_notifier.agent_prediction_notification_evaluation(self.base_dir, race_id)

        self.assertFalse(result["should_notify"])
        self.assertEqual(result["reason"], "agent_prediction_skip_decision")

    def test_agent_medium_confidence_reaches_default_high_evaluation_threshold_without_decision(self) -> None:
        race_id = "202606100103"
        self._write_agent_prediction(race_id=race_id, bet_decision="", confidence="medium")

        with patch.dict(
            os.environ,
            {
                "KEIBA_AGENT_PREDICTIONS_DIR": "",
                "PIPELINE_AUTO_PREDICTION_NOTIFY_MIN_CONFIDENCE": "",
                "PIPELINE_HIGH_EVALUATION_NOTIFY_THRESHOLD": "",
            },
        ):
            result = ntfy_notifier.agent_prediction_notification_evaluation(self.base_dir, race_id)

        self.assertTrue(result["should_notify"])
        self.assertEqual(result["reason"], "high_evaluation")

    def test_share_notifications_skip_before_any_channel_when_high_evaluation_not_met(self) -> None:
        evaluation = {
            "should_notify": False,
            "reason": "high_evaluation_not_met",
            "confidence_score": 0.44,
            "threshold": 0.62,
        }

        with (
            patch.object(ntfy_notifier, "prediction_notification_evaluation", return_value=evaluation),
            patch.object(ntfy_notifier, "publish_ntfy_share_notification") as ntfy_sender,
            patch.object(ntfy_notifier, "publish_fcm_prediction_notification") as fcm_sender,
        ):
            result = ntfy_notifier.publish_share_notifications("central_turf", "run1")

        self.assertTrue(result["skipped"])
        self.assertEqual(result["reason"], "high_evaluation_not_met")
        ntfy_sender.assert_not_called()
        fcm_sender.assert_not_called()

    def test_agent_notifications_can_send_fcm_when_ntfy_is_disabled(self) -> None:
        race_id = "202606100104"
        self._write_agent_prediction(race_id=race_id, bet_decision="BET", confidence="medium")

        ntfy_result = {"ok": False, "skipped": True, "reason": "disabled"}
        fcm_result = {"ok": True, "engine": "agent_prediction", "topic": "keiba-public-updates", "message_id": "m1"}
        with (
            patch.dict(os.environ, {"KEIBA_AGENT_PREDICTIONS_DIR": ""}),
            patch.object(ntfy_notifier, "publish_ntfy_agent_prediction_notification", return_value=ntfy_result),
            patch.object(ntfy_notifier, "publish_fcm_agent_prediction_notification", return_value=fcm_result),
        ):
            result = ntfy_notifier.publish_agent_prediction_notifications(self.base_dir, race_id=race_id)

        self.assertTrue(result["ok"])
        self.assertEqual(result["engine"], "agent_prediction")
        self.assertEqual(result["topic"], "keiba-public-updates")
        self.assertEqual(result["channels"]["ntfy"]["reason"], "disabled")
        self.assertEqual(result["channels"]["fcm"]["message_id"], "m1")

    def test_agent_notifications_skip_before_channels_when_high_evaluation_not_met(self) -> None:
        race_id = "202606100105"
        self._write_agent_prediction(race_id=race_id, bet_decision="SKIP", confidence="high")

        with (
            patch.dict(os.environ, {"KEIBA_AGENT_PREDICTIONS_DIR": ""}),
            patch.object(ntfy_notifier, "publish_ntfy_agent_prediction_notification") as ntfy_sender,
            patch.object(ntfy_notifier, "publish_fcm_agent_prediction_notification") as fcm_sender,
        ):
            result = ntfy_notifier.publish_agent_prediction_notifications(self.base_dir, race_id=race_id)

        self.assertTrue(result["skipped"])
        self.assertEqual(result["reason"], "agent_prediction_skip_decision")
        ntfy_sender.assert_not_called()
        fcm_sender.assert_not_called()

    def test_fcm_agent_prediction_notification_includes_public_url_data(self) -> None:
        race_id = "202606100106"
        self._write_agent_prediction(race_id=race_id, bet_decision="BET", confidence="high")
        captured = {}

        class FakeMessaging:
            class Notification:
                def __init__(self, title="", body=""):
                    self.title = title
                    self.body = body

            class AndroidConfig:
                def __init__(self, priority=""):
                    self.priority = priority

            class Message:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            @staticmethod
            def send(message, app=None):
                captured["message"] = message
                captured["app"] = app
                return "fcm-message-1"

        fake_firebase_admin = types.SimpleNamespace(messaging=FakeMessaging)
        with (
            patch.dict(
                os.environ,
                {
                    "KEIBA_AGENT_PREDICTIONS_DIR": "",
                    "PIPELINE_FCM_NOTIFY_ENABLED": "1",
                    "PIPELINE_FCM_TOPIC": "keiba-public-updates",
                },
            ),
            patch.dict(sys.modules, {"firebase_admin": fake_firebase_admin}),
            patch.object(ntfy_notifier, "_get_firebase_app", return_value="fake-app"),
        ):
            result = ntfy_notifier.publish_fcm_agent_prediction_notification(self.base_dir, race_id=race_id)

        self.assertTrue(result["ok"])
        self.assertEqual(result["message_id"], "fcm-message-1")
        message = captured["message"]
        self.assertEqual(message.kwargs["topic"], "keiba-public-updates")
        self.assertIn(f"/race/{race_id}", message.kwargs["data"]["url"])
        self.assertEqual(captured["app"], "fake-app")


if __name__ == "__main__":
    unittest.main()

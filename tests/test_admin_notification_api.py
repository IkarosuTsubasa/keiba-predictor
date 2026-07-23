import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import ntfy_notifier
import web_app


class AdminNotificationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.client = TestClient(web_app.app)
        self.headers = {"Authorization": "Bearer test-admin-token"}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_settings_are_admin_protected_and_persisted(self) -> None:
        with (
            patch.object(web_app, "BASE_DIR", self.base_dir),
            patch.dict(os.environ, {"ADMIN_TOKEN": "test-admin-token"}, clear=False),
        ):
            denied = self.client.get("/keiba/api/admin/notifications/settings")
            updated = self.client.post(
                "/keiba/api/admin/notifications/settings",
                headers=self.headers,
                json={"daily_early_prediction_enabled": False},
            )
            loaded = self.client.get(
                "/keiba/api/admin/notifications/settings",
                headers=self.headers,
            )

        self.assertEqual(denied.status_code, 403)
        self.assertEqual(updated.status_code, 200)
        self.assertFalse(updated.json()["settings"]["enabled"])
        self.assertEqual(updated.json()["settings"]["source"], "console")
        self.assertFalse(loaded.json()["settings"]["enabled"])

    def test_device_test_returns_only_masked_token(self) -> None:
        registration_token = "test-device-registration-token-1234567890"
        send_result = {
            "ok": True,
            "notification_kind": "daily_early_prediction_complete",
            "destination": "my_prediction",
            "message_id": "message-1",
            "token_masked": "test-d…567890",
        }
        with (
            patch.dict(os.environ, {"ADMIN_TOKEN": "test-admin-token"}, clear=False),
            patch.object(
                ntfy_notifier,
                "publish_fcm_device_test_notification",
                return_value=send_result,
            ) as sender,
        ):
            response = self.client.post(
                "/keiba/api/admin/notifications/test",
                headers=self.headers,
                json={
                    "registration_token": registration_token,
                    "notification_kind": "daily_early_prediction_complete",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(registration_token, response.text)
        self.assertEqual(response.json()["destination"], "my_prediction")
        sender.assert_called_once_with(
            registration_token,
            notification_kind="daily_early_prediction_complete",
        )


if __name__ == "__main__":
    unittest.main()

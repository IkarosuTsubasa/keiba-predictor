from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT_DIR / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import web_app  # noqa: E402
import web_auth  # noqa: E402


class WebSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        with web_app._RATE_LIMIT_LOCK:
            web_app._RATE_LIMIT_STATE.clear()

    def test_missing_admin_token_fails_closed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(web_auth.admin_token_enabled())
            self.assertFalse(web_auth.admin_token_valid(""))
            self.assertFalse(web_auth.admin_token_valid("anything"))

    def test_admin_token_uses_exact_match(self) -> None:
        with patch.dict(os.environ, {"ADMIN_TOKEN": "expected-token"}, clear=True):
            self.assertTrue(web_auth.admin_token_enabled())
            self.assertTrue(web_auth.admin_token_valid("expected-token"))
            self.assertFalse(web_auth.admin_token_valid("wrong-token"))

    def test_sensitive_paths_are_rate_limited(self) -> None:
        self.assertEqual(web_app._rate_limit_profile("/keiba/api/admin/auth-check", "GET"), "admin_auth")
        self.assertEqual(web_app._rate_limit_profile("/keiba/api/admin/jobs/create", "POST"), "admin_write")
        self.assertEqual(web_app._rate_limit_profile("/keiba/api/admin/jobs", "GET"), "admin_read")
        self.assertEqual(web_app._rate_limit_profile("/api/mobile/v1/races", "GET"), "mobile_api")
        self.assertEqual(web_app._rate_limit_profile("/keiba/internal/v5_tasks/1/callback", "POST"), "internal_callback")
        self.assertIsNone(web_app._rate_limit_profile("/keiba/api/public/board", "GET"))

    def test_rate_limit_blocks_after_limit(self) -> None:
        profile = "admin_auth"
        limit, _ = web_app.RATE_LIMITS[profile]
        for index in range(limit):
            allowed, _, _ = web_app._rate_limit_decision(profile, "unit-client", now=float(index))
            self.assertTrue(allowed)

        allowed, remaining, retry_after = web_app._rate_limit_decision(
            profile,
            "unit-client",
            now=float(limit),
        )
        self.assertFalse(allowed)
        self.assertEqual(remaining, 0)
        self.assertGreater(retry_after, 0)


if __name__ == "__main__":
    unittest.main()

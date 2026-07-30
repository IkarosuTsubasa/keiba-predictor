import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import web_app  # noqa: E402


class PublicBoardPerformanceTest(unittest.TestCase):
    def setUp(self):
        with web_app._PUBLIC_BOARD_PAYLOAD_CACHE_LOCK:
            web_app._PUBLIC_BOARD_PAYLOAD_CACHE.clear()
            web_app._PUBLIC_BOARD_PAYLOAD_REFRESHING.clear()

    def tearDown(self):
        with web_app._PUBLIC_BOARD_PAYLOAD_CACHE_LOCK:
            web_app._PUBLIC_BOARD_PAYLOAD_CACHE.clear()
            web_app._PUBLIC_BOARD_PAYLOAD_REFRESHING.clear()

    def test_expired_entry_returns_stale_immediately_and_schedules_refresh(self):
        key = web_app._public_board_cache_key("", "", False)
        cached_payload = {"target_date": "2026-07-30", "races": [{"race_id": "race-1"}]}
        cached_entry = {
            "created_at": time.time() - 3600,
            "payload": cached_payload,
        }

        with (
            patch.object(web_app, "_cached_public_board_entry", return_value=cached_entry),
            patch.object(web_app, "_schedule_public_board_refresh") as schedule_refresh,
            patch.object(web_app, "_build_public_board_payload_uncached") as rebuild,
        ):
            payload = web_app.build_public_board_payload()

        self.assertEqual(payload["target_date"], "2026-07-30")
        self.assertTrue(payload[web_app._PUBLIC_BOARD_STALE_MARKER])
        schedule_refresh.assert_called_once_with(
            key,
            date_text="",
            scope_key="",
            include_detail_fields=False,
        )
        rebuild.assert_not_called()

    def test_disk_cache_round_trip(self):
        key = web_app._public_board_cache_key("", "", False)
        entry = {
            "created_at": 123.5,
            "payload": {"target_date": "2026-07-30", "races": []},
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            web_app,
            "BASE_DIR",
            Path(temp_dir),
        ):
            web_app._save_public_board_disk_cache(key, entry)
            loaded = web_app._load_public_board_disk_cache(key)

        self.assertEqual(loaded, entry)

    def test_compact_payload_omits_detail_only_race_fields_and_history_rows(self):
        payload = {
            "races": [
                {
                    "race_id": "race-1",
                    "top5": [
                        {
                            "horse_no": "3",
                            "horse_name": "テストホース",
                            "support_score": 81,
                            "reason": "詳細ページだけで使う長文",
                        }
                    ],
                    "predictor_compare_cards": [{"label": "予測"}],
                    "predictor_top5": {"agent": [{"horse_no": "3"}]},
                }
            ],
            "history": {
                "agent_prediction": {
                    "periods": {
                        "days_30": {
                            "predicted_races": 12,
                            "daily_rows": [{"race_date": "2026-07-30"}],
                            "course_rows": [{"course": "大井"}],
                        }
                    }
                }
            },
        }

        compact = web_app._public_board_api_payload(
            payload,
            include_history_detail=False,
        )
        race = compact["races"][0]
        period = compact["history"]["agent_prediction"]["periods"]["days_30"]

        self.assertNotIn("predictor_compare_cards", race)
        self.assertNotIn("predictor_top5", race)
        self.assertEqual(
            race["top5"],
            [{"horse_no": "3", "horse_name": "テストホース", "support_score": 81}],
        )
        self.assertNotIn("daily_rows", period)
        self.assertNotIn("course_rows", period)
        self.assertEqual(period["predicted_races"], 12)


if __name__ == "__main__":
    unittest.main()

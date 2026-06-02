from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.fetchers import netkeiba_fetcher
from keiba_llm_agent.fetchers.netkeiba_fetcher import (
    fetch_and_parse_netkeiba_race,
    fetch_netkeiba_html,
    save_race_data,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_shutuba_sample.html"
FIXTURE_HTML = FIXTURE_PATH.read_text(encoding="utf-8")
SAMPLE_URL = "https://race.netkeiba.com/race/shutuba.html?race_id=202605180511"


class FakeResponse:
    def __init__(self, text: str) -> None:
        self._text = text
        self.apparent_encoding = "utf-8"
        self.encoding = None

    @property
    def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        return None


def build_odds_api_response(odds_map: dict[str, list[object]]) -> str:
    payload = {
        "official_datetime": "2026-05-18 12:34:56",
        "odds": {
            "1": odds_map,
            "2": {},
        },
    }
    encoded = base64.b64encode(
        zlib.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    ).decode("ascii")
    return json.dumps(
        {
            "status": "result",
            "data": encoded,
            "update_count": "0",
            "reason": "",
        },
        ensure_ascii=False,
    )


class NetkeibaFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.html_cache_dir = self.temp_path / "html_cache"
        self.race_data_dir = self.temp_path / "race_data"

        self.html_cache_patch = patch.object(netkeiba_fetcher, "HTML_CACHE_DIR", self.html_cache_dir)
        self.race_data_patch = patch.object(netkeiba_fetcher, "RACE_DATA_DIR", self.race_data_dir)
        self.html_cache_patch.start()
        self.race_data_patch.start()

    def tearDown(self) -> None:
        self.html_cache_patch.stop()
        self.race_data_patch.stop()
        self.temp_dir.cleanup()

    def test_reads_existing_cache_without_request(self) -> None:
        cache_path = self.html_cache_dir / "202605180511.html"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(FIXTURE_HTML, encoding="utf-8")

        with patch("keiba_llm_agent.fetchers.netkeiba_fetcher.requests.get") as mock_get:
            race_id, html = fetch_netkeiba_html(SAMPLE_URL)

        self.assertEqual(race_id, "202605180511")
        self.assertEqual(html, FIXTURE_HTML)
        mock_get.assert_not_called()

    def test_fetches_and_saves_cache_when_missing(self) -> None:
        fake_response = FakeResponse(FIXTURE_HTML)
        with patch(
            "keiba_llm_agent.fetchers.netkeiba_fetcher.requests.get",
            return_value=fake_response,
        ) as mock_get:
            race_id, html = fetch_netkeiba_html(SAMPLE_URL)

        self.assertEqual(race_id, "202605180511")
        self.assertEqual(html, FIXTURE_HTML)
        mock_get.assert_called_once()
        self.assertTrue((self.html_cache_dir / "202605180511.html").exists())

    def test_force_refresh_requests_even_when_cache_exists(self) -> None:
        cache_path = self.html_cache_dir / "202605180511.html"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("old", encoding="utf-8")

        fake_response = FakeResponse(FIXTURE_HTML)
        with patch(
            "keiba_llm_agent.fetchers.netkeiba_fetcher.requests.get",
            return_value=fake_response,
        ) as mock_get:
            _, html = fetch_netkeiba_html(SAMPLE_URL, force_refresh=True)

        self.assertEqual(html, FIXTURE_HTML)
        mock_get.assert_called_once()

    def test_fetch_and_parse_returns_race_data(self) -> None:
        with patch(
            "keiba_llm_agent.fetchers.netkeiba_fetcher.requests.get",
            return_value=FakeResponse(FIXTURE_HTML),
        ):
            race_data = fetch_and_parse_netkeiba_race(SAMPLE_URL)

        self.assertEqual(race_data.race_info.race_id, "202605180511")
        self.assertEqual(race_data.horses[0].horse_name, "サンプルホースA")

    def test_save_race_data_writes_json(self) -> None:
        with patch(
            "keiba_llm_agent.fetchers.netkeiba_fetcher.requests.get",
            return_value=FakeResponse(FIXTURE_HTML),
        ):
            race_data = fetch_and_parse_netkeiba_race(SAMPLE_URL)

        saved_path = save_race_data(race_data)
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["race_info"]["race_id"], "202605180511")
        self.assertTrue(saved_path.exists())

    def test_fetch_race_command_executes(self) -> None:
        with patch.object(netkeiba_fetcher, "HTML_CACHE_DIR", self.html_cache_dir), patch.object(
            netkeiba_fetcher, "RACE_DATA_DIR", self.race_data_dir
        ), patch(
            "keiba_llm_agent.fetchers.netkeiba_fetcher.requests.get",
            return_value=FakeResponse(FIXTURE_HTML),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main_module.main(["fetch-race", "--url", SAMPLE_URL])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["race_id"], "202605180511")
        self.assertTrue(Path(payload["saved_to"]).exists())

    def test_fetch_and_parse_enriches_current_odds_when_static_html_missing(self) -> None:
        html_without_odds = FIXTURE_HTML.replace(">5.8<", ">---.-<").replace(">2<", "><", 1)
        odds_api_response = build_odds_api_response(
            {
                "01": ["8.8", "0.0", 3],
                "02": ["14.2", "0.0", 6],
            }
        )

        def fake_get(url: str, *args, **kwargs):
            if "api_get_jra_odds.html" in url:
                return FakeResponse(f"({odds_api_response})")
            return FakeResponse(html_without_odds)

        with patch("keiba_llm_agent.fetchers.netkeiba_fetcher.requests.get", side_effect=fake_get):
            race_data = fetch_and_parse_netkeiba_race(SAMPLE_URL)

        self.assertEqual(race_data.horses[0].odds, 8.8)
        self.assertEqual(race_data.horses[0].popularity, 3)
        self.assertEqual(race_data.horses[1].odds, 14.2)
        self.assertEqual(race_data.horses[1].popularity, 6)

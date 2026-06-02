from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.fetchers import netkeiba_result_fetcher
from keiba_llm_agent.fetchers.netkeiba_result_fetcher import fetch_netkeiba_result_html


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_result_sample.html"
FIXTURE_HTML = FIXTURE_PATH.read_text(encoding="utf-8")
SAMPLE_URL = "https://race.netkeiba.com/race/result.html?race_id=202605020811"


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


class NetkeibaResultFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.cache_dir = self.temp_path / "result_html_cache"
        self.patch_cache = patch.object(netkeiba_result_fetcher, "RESULT_HTML_CACHE_DIR", self.cache_dir)
        self.patch_cache.start()

    def tearDown(self) -> None:
        self.patch_cache.stop()
        self.temp_dir.cleanup()

    def test_uses_cache_without_request(self) -> None:
        cache_path = self.cache_dir / "202605020811.html"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(FIXTURE_HTML, encoding="utf-8")
        with patch("keiba_llm_agent.fetchers.netkeiba_result_fetcher.requests.get") as mock_get:
            race_id, html = fetch_netkeiba_result_html(SAMPLE_URL)
        self.assertEqual(race_id, "202605020811")
        self.assertEqual(html, FIXTURE_HTML)
        mock_get.assert_not_called()

    def test_fetches_and_saves_when_cache_missing(self) -> None:
        with patch(
            "keiba_llm_agent.fetchers.netkeiba_result_fetcher.requests.get",
            return_value=FakeResponse(FIXTURE_HTML),
        ) as mock_get:
            race_id, html = fetch_netkeiba_result_html(SAMPLE_URL)
        self.assertEqual(race_id, "202605020811")
        self.assertEqual(html, FIXTURE_HTML)
        mock_get.assert_called_once()
        self.assertTrue((self.cache_dir / "202605020811.html").exists())

    def test_force_refresh_refetches(self) -> None:
        cache_path = self.cache_dir / "202605020811.html"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("old", encoding="utf-8")
        with patch(
            "keiba_llm_agent.fetchers.netkeiba_result_fetcher.requests.get",
            return_value=FakeResponse(FIXTURE_HTML),
        ) as mock_get:
            _, html = fetch_netkeiba_result_html(SAMPLE_URL, force_refresh=True)
        self.assertEqual(html, FIXTURE_HTML)
        mock_get.assert_called_once()


if __name__ == "__main__":
    unittest.main()

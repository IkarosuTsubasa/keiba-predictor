from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from requests import Response

from keiba_llm_agent.fetchers import netkeiba_horse_fetcher
from keiba_llm_agent.fetchers.netkeiba_horse_fetcher import (
    enrich_race_data_with_recent_runs,
    fetch_horse_html,
)
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_horse_sample.html"
FIXTURE_HTML = FIXTURE_PATH.read_text(encoding="utf-8")


def horse_response(html=FIXTURE_HTML, *, encoding="utf-8", declared="utf-8"):
    response = Response()
    response.status_code = 200
    response._content = html.encode(encoding)
    response.headers["Content-Type"] = "text/html" + (f"; charset={declared}" if declared else "")
    return response


class NetkeibaHorseFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.horse_cache_dir = self.temp_path / "horse_html_cache"
        self.cache_patch = patch.object(netkeiba_horse_fetcher, "HORSE_HTML_CACHE_DIR", self.horse_cache_dir)
        self.cache_patch.start()

    def tearDown(self) -> None:
        self.cache_patch.stop()
        self.temp_dir.cleanup()

    def test_fetch_horse_html_uses_cache_without_request(self) -> None:
        cache_path = self.horse_cache_dir / "2021104073.html"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(FIXTURE_HTML, encoding="utf-8")
        with patch("keiba_llm_agent.fetchers.netkeiba_horse_fetcher.requests.get") as mock_get:
            html = fetch_horse_html("2021104073")
        self.assertEqual(html, FIXTURE_HTML)
        mock_get.assert_not_called()

    def test_fetch_horse_html_fetches_and_saves_cache(self) -> None:
        for actual, declared in (
            ("utf-8", "UTF-8"),
            ("euc_jp", "EUC-JP"),
            ("utf-8", "EUC-JP"),
            ("euc_jp", None),
            ("utf-8-sig", "invalid-encoding"),
        ):
            with self.subTest(actual=actual, declared=declared), patch.object(
                netkeiba_horse_fetcher.requests, "get",
                return_value=horse_response(encoding=actual, declared=declared),
            ) as get:
                html = fetch_horse_html("2021104073", force_refresh=True)
                self.assertIn("\u65e5\u4ed8", html)
                self.assertTrue(netkeiba_horse_fetcher.has_recent_runs_table(html))
                get.assert_called_once()
                self.assertEqual((self.horse_cache_dir / "2021104073.html").read_text(encoding="utf-8"), html)

    def test_force_refresh_refetches_even_with_cache(self) -> None:
        cache_path = self.horse_cache_dir / "2021104073.html"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("old", encoding="utf-8")
        with patch(
            "keiba_llm_agent.fetchers.netkeiba_horse_fetcher.requests.get",
            return_value=horse_response(),
        ) as mock_get:
            html = fetch_horse_html("2021104073", force_refresh=True)
        self.assertEqual(html, FIXTURE_HTML)
        mock_get.assert_called_once()

    def test_invalid_cached_page_is_refetched(self) -> None:
        cache_path = self.horse_cache_dir / "2021104073.html"
        cache_path.parent.mkdir(parents=True)
        for data in (b"\xff", b"<html>no results table</html>"):
            with self.subTest(data=data), patch.object(
                netkeiba_horse_fetcher.requests, "get", return_value=horse_response(),
            ) as get:
                cache_path.write_bytes(data)
                with self.assertWarnsRegex(UserWarning, "stale horse cache"):
                    self.assertEqual(fetch_horse_html("2021104073"), FIXTURE_HTML)
                get.assert_called_once()

    def test_page_without_table_does_not_poison_cache(self) -> None:
        valid = self.horse_cache_dir / "2021104073.html"
        valid.parent.mkdir(parents=True)
        valid.write_text(FIXTURE_HTML, encoding="utf-8")
        unusable = "<html>no results table</html>"
        with patch.object(netkeiba_horse_fetcher.requests, "get", return_value=horse_response(unusable)):
            self.assertEqual(fetch_horse_html("2021104073", force_refresh=True), unusable)
            self.assertEqual(fetch_horse_html("2022100000"), unusable)
        self.assertEqual(valid.read_text(encoding="utf-8"), FIXTURE_HTML)
        self.assertFalse((self.horse_cache_dir / "2022100000.html").exists())

    def test_enrich_race_data_with_recent_runs_updates_horses(self) -> None:
        race_data = RaceData(
            race_info=RaceInfo(race_id="sample_001", race_date="2026-05-17"),
            horses=[
                HorseEntry(horse_no=1, horse_id="2021104073", horse_name="サンプルホースA"),
            ],
        )
        with patch.object(netkeiba_horse_fetcher, "fetch_horse_html", return_value=FIXTURE_HTML):
            enriched = enrich_race_data_with_recent_runs(race_data, limit=3)
        self.assertEqual(len(enriched.horses[0].recent_runs), 3)
        self.assertEqual(enriched.horses[0].recent_runs[0].course, "阪神")

    def test_enrich_race_data_without_limit_keeps_full_filtered_career(self) -> None:
        race_data = RaceData(
            race_info=RaceInfo(race_id="sample_001", race_date="2026-05-17"),
            horses=[
                HorseEntry(horse_no=1, horse_id="2021104073", horse_name="サンプルホースA"),
            ],
        )
        with patch.object(netkeiba_horse_fetcher, "fetch_horse_html", return_value=FIXTURE_HTML):
            enriched = enrich_race_data_with_recent_runs(race_data)

        self.assertGreater(len(enriched.horses[0].recent_runs), 3)
        self.assertNotIn("2026-05-17", [run.date for run in enriched.horses[0].recent_runs])

    def test_single_horse_failure_does_not_interrupt_race(self) -> None:
        race_data = RaceData(
            race_info=RaceInfo(race_id="sample_001", race_date="2026-05-17"),
            horses=[
                HorseEntry(horse_no=1, horse_id="2021104073", horse_name="A"),
                HorseEntry(horse_no=2, horse_id="2021109999", horse_name="B"),
            ],
        )

        def fake_fetch(horse_id: str, force_refresh: bool = False) -> str:
            if horse_id == "2021109999":
                raise RuntimeError("boom")
            return FIXTURE_HTML

        with patch.object(netkeiba_horse_fetcher, "fetch_horse_html", side_effect=fake_fetch):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                enriched = enrich_race_data_with_recent_runs(race_data, limit=2)

        self.assertEqual(len(enriched.horses[0].recent_runs), 2)
        self.assertEqual(enriched.horses[1].recent_runs, [])
        self.assertGreater(len(caught), 0)


if __name__ == "__main__":
    unittest.main()

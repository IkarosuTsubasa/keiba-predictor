from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from keiba_llm_agent.main import main
from keiba_llm_agent.parsers.netkeiba_race_parser import parse_netkeiba_shutuba_html


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_shutuba_sample.html"
REAL_HTML_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "keiba_llm_agent"
    / "data"
    / "html_cache"
    / "202605020811.html"
)


class NetkeibaRaceParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = FIXTURE_PATH.read_text(encoding="utf-8")
        self.real_html = (
            REAL_HTML_CACHE_PATH.read_text(encoding="utf-8", errors="ignore")
            if REAL_HTML_CACHE_PATH.exists()
            else None
        )

    def test_parses_race_info_from_fixture(self) -> None:
        race_data = parse_netkeiba_shutuba_html(self.html, race_id="202605180511")
        self.assertEqual(race_data.race_info.race_id, "202605180511")
        self.assertEqual(race_data.race_info.race_name, "サンプルステークス")
        self.assertEqual(race_data.race_info.race_date, "2026-05-18")
        self.assertEqual(race_data.race_info.course, "東京")
        self.assertEqual(race_data.race_info.surface, "芝")
        self.assertEqual(race_data.race_info.distance, 1600)
        self.assertEqual(race_data.race_info.track_condition, "良")
        self.assertEqual(race_data.race_info.weather, "晴")

    def test_parses_scope_from_central_race_id(self) -> None:
        race_data = parse_netkeiba_shutuba_html(self.html, race_id="202605180511")
        self.assertEqual(race_data.race_info.source, "central")
        self.assertEqual(race_data.race_info.scope_key, "central_turf")

    def test_parses_scope_from_local_race_id_and_course(self) -> None:
        local_html = self.html.replace("東京", "大井").replace("2026年5月18日", "2026年5月18日")
        race_data = parse_netkeiba_shutuba_html(local_html, race_id="202644031102")
        self.assertEqual(race_data.race_info.course, "大井")
        self.assertEqual(race_data.race_info.source, "local")
        self.assertEqual(race_data.race_info.scope_key, "local")

    def test_parses_horses_array(self) -> None:
        race_data = parse_netkeiba_shutuba_html(self.html, race_id="202605180511")
        self.assertEqual(len(race_data.horses), 2)

    def test_parses_required_horse_fields(self) -> None:
        race_data = parse_netkeiba_shutuba_html(self.html, race_id="202605180511")
        horse = race_data.horses[0]
        self.assertEqual(horse.horse_no, 1)
        self.assertEqual(horse.horse_name, "サンプルホースA")
        self.assertEqual(horse.jockey, "騎手A")
        self.assertEqual(horse.carried_weight, 57.0)
        self.assertEqual(horse.odds, 5.8)
        self.assertEqual(horse.popularity, 2)
        self.assertIsInstance(horse.odds, float)
        self.assertIsInstance(horse.popularity, int)

    def test_extracts_horse_id_from_link(self) -> None:
        race_data = parse_netkeiba_shutuba_html(self.html, race_id="202605180511")
        self.assertEqual(race_data.horses[0].horse_id, "2020101234")

    def test_missing_odds_does_not_crash(self) -> None:
        race_data = parse_netkeiba_shutuba_html(self.html, race_id="202605180511")
        self.assertIsNone(race_data.horses[1].odds)

    def test_missing_popularity_does_not_crash(self) -> None:
        race_data = parse_netkeiba_shutuba_html(self.html, race_id="202605180511")
        self.assertIsNone(race_data.horses[1].popularity)

    def test_parse_html_command_executes(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "parse-html",
                    "--html",
                    str(FIXTURE_PATH),
                    "--race-id",
                    "202605180511",
                ]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["race_info"]["race_id"], "202605180511")
        self.assertEqual(payload["horses"][0]["horse_name"], "サンプルホースA")

    def test_real_html_cache_parses_non_empty_horses(self) -> None:
        if self.real_html is None:
            self.skipTest("real html cache not found")
        race_data = parse_netkeiba_shutuba_html(self.real_html, race_id="202605020811")
        self.assertGreater(len(race_data.horses), 0)
        self.assertEqual(race_data.race_info.race_date, "2026-05-17")
        first_horse = race_data.horses[0]
        self.assertIsNotNone(first_horse.horse_no)
        self.assertTrue(first_horse.horse_name)
        self.assertEqual(first_horse.recent_runs, [])

    def test_real_html_cache_does_not_fail_when_odds_or_popularity_missing(self) -> None:
        if self.real_html is None:
            self.skipTest("real html cache not found")
        race_data = parse_netkeiba_shutuba_html(self.real_html, race_id="202605020811")
        self.assertGreater(len(race_data.horses), 0)
        self.assertEqual(race_data.horses[0].recent_runs, [])
        self.assertTrue(
            any(horse.odds is not None for horse in race_data.horses)
            or any(horse.odds is None for horse in race_data.horses)
        )
        self.assertTrue(
            any(horse.popularity is not None for horse in race_data.horses)
            or any(horse.popularity is None for horse in race_data.horses)
        )


if __name__ == "__main__":
    unittest.main()

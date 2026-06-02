from __future__ import annotations

import unittest
from unittest.mock import patch

from keiba_llm_agent.fetchers import netkeiba_horse_fetcher
from keiba_llm_agent.fetchers.netkeiba_horse_fetcher import enrich_race_data_with_recent_runs
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo


FIXTURE_PATH = "tests/fixtures/netkeiba_horse_sample.html"


class RecentRunsNoLeakageTests(unittest.TestCase):
    def test_recent_runs_excludes_target_race_and_applies_limit_after_filtering(self) -> None:
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            fixture_html = f.read()

        race_data = RaceData(
            race_info=RaceInfo(
                race_id="202605020811",
                race_date="2026-05-17",
                course="東京",
                surface="芝",
                distance=1600,
            ),
            horses=[HorseEntry(horse_no=1, horse_id="2021104073", horse_name="カピリナ")],
        )

        with patch.object(netkeiba_horse_fetcher, "fetch_horse_html", return_value=fixture_html):
            enriched = enrich_race_data_with_recent_runs(race_data, limit=2)

        dates = [run.date for run in enriched.horses[0].recent_runs]
        self.assertNotIn("2026-05-17", dates)
        self.assertIn("2026-04-11", dates)
        self.assertIn("2026-01-12", dates)
        self.assertEqual(len(enriched.horses[0].recent_runs), 2)


if __name__ == "__main__":
    unittest.main()

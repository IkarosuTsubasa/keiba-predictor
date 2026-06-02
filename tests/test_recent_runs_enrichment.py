from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.fetchers import netkeiba_fetcher, netkeiba_horse_fetcher
from keiba_llm_agent.fetchers.netkeiba_fetcher import save_race_data
from keiba_llm_agent.fetchers.netkeiba_horse_fetcher import enrich_race_data_with_recent_runs
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_horse_sample.html"
FIXTURE_HTML = FIXTURE_PATH.read_text(encoding="utf-8")


class RecentRunsEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.race_data_dir = self.temp_path / "race_data"
        self.race_data_patch = patch.object(netkeiba_fetcher, "RACE_DATA_DIR", self.race_data_dir)
        self.race_data_patch.start()

    def tearDown(self) -> None:
        self.race_data_patch.stop()
        self.temp_dir.cleanup()

    def test_enriched_race_data_has_recent_runs_and_saved_json_keeps_them(self) -> None:
        race_data = RaceData(
            race_info=RaceInfo(race_id="202605020811", race_date="2026-05-17"),
            horses=[HorseEntry(horse_no=1, horse_id="2021104073", horse_name="カピリナ")],
        )
        with patch.object(netkeiba_horse_fetcher, "fetch_horse_html", return_value=FIXTURE_HTML):
            enriched = enrich_race_data_with_recent_runs(race_data, limit=5)

        self.assertGreater(len(enriched.horses[0].recent_runs), 0)
        self.assertNotIn("2026-05-17", [run.date for run in enriched.horses[0].recent_runs])
        saved_path = save_race_data(enriched)
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertGreater(len(payload["horses"][0]["recent_runs"]), 0)


if __name__ == "__main__":
    unittest.main()

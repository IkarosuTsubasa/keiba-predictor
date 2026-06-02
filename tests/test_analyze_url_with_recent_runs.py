from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.fetchers import netkeiba_fetcher, netkeiba_horse_fetcher
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "netkeiba_horse_sample.html"
FIXTURE_HTML = FIXTURE_PATH.read_text(encoding="utf-8")
SAMPLE_URL = "https://race.netkeiba.com/race/shutuba.html?race_id=202605020811"


class AnalyzeUrlWithRecentRunsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.race_data_dir = self.temp_path / "race_data"
        self.predictions_dir = self.temp_path / "predictions"

        self.race_data_patch = patch.object(netkeiba_fetcher, "RACE_DATA_DIR", self.race_data_dir)
        self.predictions_patch = patch.object(main_module, "DEFAULT_PREDICTIONS_DIR", self.predictions_dir)
        self.race_data_patch.start()
        self.predictions_patch.start()

    def tearDown(self) -> None:
        self.race_data_patch.stop()
        self.predictions_patch.stop()
        self.temp_dir.cleanup()

    def test_analyze_url_with_recent_runs_saves_non_empty_recent_runs(self) -> None:
        race_data = RaceData(
            race_info=RaceInfo(
                race_id="202605020811",
                race_date="2026-05-17",
                course="東京",
                surface="芝",
                distance=1600,
            ),
            horses=[HorseEntry(horse_no=1, horse_id="2021104073", horse_name="カピリナ", jockey="騎手A")],
        )
        with patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=race_data), patch.object(
            netkeiba_horse_fetcher, "fetch_horse_html", return_value=FIXTURE_HTML
        ), patch.object(main_module, "fetch_horse_html", return_value=FIXTURE_HTML):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main_module.main(
                    [
                        "analyze-url",
                        "--url",
                        SAMPLE_URL,
                        "--with-recent-runs",
                        "--recent-run-limit",
                        "5",
                    ]
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        race_data_payload = json.loads(Path(payload["race_data_path"]).read_text(encoding="utf-8"))
        self.assertGreater(len(race_data_payload["horses"][0]["recent_runs"]), 0)
        self.assertNotIn("2026-05-17", [run["date"] for run in race_data_payload["horses"][0]["recent_runs"]])


if __name__ == "__main__":
    unittest.main()

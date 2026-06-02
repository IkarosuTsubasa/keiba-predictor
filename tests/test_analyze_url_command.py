from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.fetchers import netkeiba_fetcher
from keiba_llm_agent.schemas.race_data import RaceData, RecentRun


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA_PATH = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
SAMPLE_RACE_DATA = RaceData.from_json_file(SAMPLE_RACE_DATA_PATH)
SAMPLE_URL = "https://race.netkeiba.com/race/shutuba.html?race_id=sample_001"


class AnalyzeUrlCommandTests(unittest.TestCase):
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

    def test_analyze_url_dry_run_saves_race_data_only(self) -> None:
        with patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=SAMPLE_RACE_DATA):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main_module.main(["analyze-url", "--url", SAMPLE_URL, "--dry-run"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertTrue(Path(payload["race_data_path"]).exists())
        self.assertIsNone(payload["prediction_path"])
        self.assertFalse((self.predictions_dir / "sample_001.json").exists())

    def test_analyze_url_generates_race_data_and_prediction(self) -> None:
        with patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=SAMPLE_RACE_DATA):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main_module.main(["analyze-url", "--url", SAMPLE_URL])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        race_data_path = Path(payload["race_data_path"])
        prediction_path = Path(payload["prediction_path"])
        self.assertTrue(race_data_path.exists())
        self.assertTrue(prediction_path.exists())
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        self.assertEqual(prediction["race_id"], "sample_001")

    def test_force_refresh_is_passed_to_fetcher(self) -> None:
        with patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=SAMPLE_RACE_DATA) as mock_fetch:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                main_module.main(["analyze-url", "--url", SAMPLE_URL, "--force-refresh", "--dry-run"])

        mock_fetch.assert_called_once_with(SAMPLE_URL, force_refresh=True)

    def test_with_recent_runs_enriches_before_analysis(self) -> None:
        enriched_race_data = SAMPLE_RACE_DATA.model_copy(
            update={
                "horses": [
                    horse.model_copy(
                        update={
                            "recent_runs": [
                                RecentRun(
                                    date="2026-05-01",
                                    course="東京",
                                    surface="芝",
                                    distance=1600,
                                    track_condition="良",
                                    finish=2,
                                    field_size=16,
                                    jockey="騎手A",
                                    odds=5.8,
                                    popularity=2,
                                )
                            ]
                        }
                    )
                    for horse in SAMPLE_RACE_DATA.horses
                ]
            }
        )
        with patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=SAMPLE_RACE_DATA), patch.object(
            main_module, "enrich_race_data_with_recent_runs", return_value=enriched_race_data
        ) as mock_enrich:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main_module.main(
                    [
                        "analyze-url",
                        "--url",
                        SAMPLE_URL,
                        "--with-recent-runs",
                        "--recent-run-limit",
                        "3",
                    ]
                )

        self.assertEqual(exit_code, 0)
        mock_enrich.assert_called_once()
        payload = json.loads(buffer.getvalue())
        prediction_path = Path(payload["prediction_path"])
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        self.assertNotEqual(prediction["horse_scores"][0]["scores"]["recent_form"], 0)

    def test_analyze_url_does_not_break_existing_analysis_command(self) -> None:
        output_path = self.predictions_dir / "analysis_only.json"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main_module.main(
                [
                    "analysis",
                    "--race-data",
                    str(SAMPLE_RACE_DATA_PATH),
                    "--output",
                    str(output_path),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["race_id"], "sample_001")


if __name__ == "__main__":
    unittest.main()

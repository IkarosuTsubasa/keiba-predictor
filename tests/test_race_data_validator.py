from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.validators.race_data_validator import validate_race_data


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA_PATH = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
SAMPLE_RACE_DATA = RaceData.from_json_file(SAMPLE_RACE_DATA_PATH)


def _valid_race_data() -> RaceData:
    payload = SAMPLE_RACE_DATA.model_dump()
    payload["race_info"]["race_date"] = "2026-05-18"
    horses = []
    for index in range(5):
        horse = payload["horses"][index]
        horse["odds"] = 5.0 + index
        horse["popularity"] = index + 1
        horse["recent_runs"] = [
            {
                "race_id": f"2026000000{index}",
                "date": "2026-05-01",
                "course": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "finish": 2,
                "field_size": 16,
                "jockey": horse["jockey"],
                "odds": 6.2,
                "popularity": 2,
            }
        ]
        horses.append(horse)
    payload["horses"] = horses
    return RaceData.model_validate(payload)


class RaceDataValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.race_data_dir = self.temp_path / "race_data"
        self.race_data_dir.mkdir(parents=True, exist_ok=True)
        self.race_data_patch = patch.object(main_module, "DEFAULT_RACE_DATA_DIR", self.race_data_dir)
        self.race_data_patch.start()

    def tearDown(self) -> None:
        self.race_data_patch.stop()
        self.temp_dir.cleanup()

    def test_validate_race_data_ok(self) -> None:
        result = validate_race_data(_valid_race_data())
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["metrics"]["horse_count"], 5)
        self.assertEqual(result["metrics"]["recent_runs_missing_rate"], 0.0)

    def test_validate_race_data_warning_for_missing_odds_and_abnormal_race_id(self) -> None:
        race_data = _valid_race_data()
        for horse in race_data.horses[:3]:
            horse.odds = None
            horse.popularity = None
        race_data.horses[0].recent_runs[0].course = None
        race_data.horses[0].recent_runs[0].race_id = "2025"

        result = validate_race_data(race_data)
        self.assertEqual(result["status"], "WARNING")
        self.assertIn("odds missing rate is above 0.5", result["warnings"])
        self.assertIn("recent_runs contain course=null", result["warnings"])
        self.assertIn("recent_runs contain abnormal race_id", result["warnings"])

    def test_validate_race_data_error_for_missing_race_date(self) -> None:
        race_data = _valid_race_data()
        race_data.race_info.race_date = None
        result = validate_race_data(race_data)
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("race_info.race_date is missing", result["errors"])

    def test_validate_race_data_error_for_future_race_leakage(self) -> None:
        race_data = _valid_race_data()
        race_data.horses[0].recent_runs[0].date = "2026-05-18"

        result = validate_race_data(race_data)
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("future race leakage detected in recent_runs", result["errors"])

    def test_validate_race_data_error_when_recent_runs_are_empty_for_all_horses(self) -> None:
        race_data = _valid_race_data()
        for horse in race_data.horses:
            horse.recent_runs = []

        result = validate_race_data(race_data)
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("recent_runs are empty for all horses", result["errors"])

    def test_validate_race_data_command_reads_default_race_data_path(self) -> None:
        race_data = _valid_race_data()
        race_data_path = self.race_data_dir / "sample_001.json"
        race_data_path.write_text(
            json.dumps(race_data.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result = main_module.run_validate_race_data("sample_001")
        self.assertEqual(result["race_id"], "sample_001")
        self.assertEqual(result["status"], "OK")


if __name__ == "__main__":
    unittest.main()

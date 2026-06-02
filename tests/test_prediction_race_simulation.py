from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.main import run_analysis
from keiba_llm_agent.reports.report_generator import generate_prediction_report
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.social.post_generator import build_prediction_post


class PredictionRaceSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.race_data_path = self.temp_path / "race_data.json"
        self.prediction_path = self.temp_path / "prediction.json"
        self.lessons_path = self.temp_path / "lessons.json"
        self.lessons_path.write_text("[]\n", encoding="utf-8")
        race_data = {
            "race_info": {
                "race_id": "sample_simulation",
                "race_name": "サンプル推演特別",
                "race_date": "2026-05-17",
                "course": "京都",
                "surface": "芝",
                "distance": 2400,
                "track_condition": "良",
                "weather": "晴",
            },
            "horses": [
                {
                    "horse_no": 1,
                    "frame_no": 1,
                    "horse_id": "2022105587",
                    "horse_name": "サンプルホースA",
                    "jockey": "騎手A",
                    "carried_weight": 56.0,
                    "odds": 6.2,
                    "popularity": 3,
                    "recent_runs": [
                        {
                            "race_id": "202601010101",
                            "date": "2026-04-20",
                            "course": "京都",
                            "surface": "芝",
                            "distance": 2400,
                            "track_condition": "良",
                            "finish": 2,
                            "field_size": 16,
                            "jockey": "騎手A",
                            "odds": 6.5,
                            "popularity": 4,
                            "passing_order": "2-2-2-2",
                            "corner_positions": [2, 2, 2, 2],
                            "final_3f": 34.9,
                        }
                    ],
                },
                {
                    "horse_no": 2,
                    "frame_no": 2,
                    "horse_id": "2022105000",
                    "horse_name": "サンプルホースB",
                    "jockey": "騎手B",
                    "carried_weight": 56.0,
                    "odds": 12.4,
                    "popularity": 7,
                    "recent_runs": [
                        {
                            "race_id": "202601010102",
                            "date": "2026-04-20",
                            "course": "阪神",
                            "surface": "芝",
                            "distance": 2200,
                            "track_condition": "良",
                            "finish": 4,
                            "field_size": 15,
                            "jockey": "騎手B",
                            "odds": 10.3,
                            "popularity": 6,
                            "passing_order": "8-8-7-5",
                            "corner_positions": [8, 8, 7, 5],
                            "final_3f": 34.4,
                        }
                    ],
                },
            ],
        }
        self.race_data_path.write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prediction_outputs_race_simulation_and_report_section(self) -> None:
        prediction, saved_path = run_analysis(
            race_data_path=self.race_data_path,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        race_data = RaceData.from_json_file(self.race_data_path)

        self.assertIn("race_simulation", payload)
        self.assertIsInstance(payload["race_simulation"], dict)
        self.assertIn("reasoning_summary", payload["race_simulation"])
        self.assertIn(payload["race_simulation"]["reasoning_summary"], prediction.summary)

        markdown = generate_prediction_report(prediction, race_data=race_data)
        self.assertIn("## レースシミュレーション", markdown)

        text = build_prediction_post(prediction, race_data=race_data)
        self.assertLessEqual(len(text), 280)


if __name__ == "__main__":
    unittest.main()

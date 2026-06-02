from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.main import run_analysis
from keiba_llm_agent.reports.report_generator import generate_prediction_report
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.social.post_generator import build_prediction_post


class PredictionRaceLevelAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.race_data_path = self.temp_path / "race_data.json"
        self.prediction_path = self.temp_path / "prediction.json"
        self.lessons_path = self.temp_path / "lessons.json"
        self.lessons_path.write_text("[]\n", encoding="utf-8")
        race_data = {
            "race_info": {
                "race_id": "sample_level",
                "race_name": "サンプル特別",
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
                    "horse_id": "h1",
                    "horse_name": "サンプルホースA",
                    "jockey": "騎手A",
                    "carried_weight": 56.0,
                    "odds": 6.5,
                    "popularity": 3,
                    "recent_runs": [
                        {
                            "race_id": "202601010101",
                            "date": "2026-04-20",
                            "course": "阪神",
                            "surface": "芝",
                            "distance": 2400,
                            "track_condition": "良",
                            "finish": 2,
                            "field_size": 16,
                            "jockey": "騎手A",
                            "odds": 9.5,
                            "popularity": 8,
                        }
                    ],
                },
                {
                    "horse_no": 2,
                    "frame_no": 2,
                    "horse_id": "h2",
                    "horse_name": "サンプルホースB",
                    "jockey": "騎手B",
                    "carried_weight": 56.0,
                    "odds": 4.2,
                    "popularity": 2,
                    "recent_runs": [
                        {
                            "race_id": "202601010101",
                            "date": "2026-04-20",
                            "course": "阪神",
                            "surface": "芝",
                            "distance": 2400,
                            "track_condition": "良",
                            "finish": 5,
                            "field_size": 16,
                            "jockey": "騎手B",
                            "odds": 4.2,
                            "popularity": 2,
                        }
                    ],
                },
            ],
        }
        self.race_data_path.write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prediction_outputs_race_level_analyses_without_changing_score_formula(self) -> None:
        prediction, saved_path = run_analysis(
            race_data_path=self.race_data_path,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        race_data = RaceData.from_json_file(self.race_data_path)

        self.assertIn("race_level_analyses", payload)
        self.assertEqual(len(payload["race_level_analyses"]), len(race_data.horses))

        first_score = payload["horse_scores"][0]
        expected_total = round(
            first_score["base_total_score"]
            + first_score["pedigree_adjustment"]["pedigree_adjustment"]
            + first_score["race_level_adjustment"]["adjustment"]
            + first_score["pace_adjustment"]["adjustment"],
            1,
        )
        self.assertEqual(first_score["total_score"], expected_total)

        markdown = generate_prediction_report(prediction, race_data=race_data)
        self.assertIn("## レースレベル・相手関係分析", markdown)

        post = build_prediction_post(prediction, race_data=race_data)
        self.assertLessEqual(len(post), 280)


if __name__ == "__main__":
    unittest.main()

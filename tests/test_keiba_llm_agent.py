from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.main import run_analysis, run_review
from keiba_llm_agent.schemas.race_data import RaceData


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
SAMPLE_RESULT = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_result.json"
SAMPLE_LESSONS = ROOT_DIR / "keiba_llm_agent" / "memory" / "lessons.json"


class KeibaLLMAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.lessons_path = self.temp_path / "lessons.json"
        shutil.copyfile(SAMPLE_LESSONS, self.lessons_path)
        self.prediction_path = self.temp_path / "prediction.json"
        self.review_path = self.temp_path / "review.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sample_race_data_can_be_loaded(self) -> None:
        race_data = RaceData.from_json_file(SAMPLE_RACE_DATA)
        self.assertEqual(race_data.race_info.race_id, "sample_001")
        self.assertEqual(len(race_data.horses), 5)

    def test_analysis_generates_prediction_json(self) -> None:
        prediction, saved_path = run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        self.assertEqual(prediction.race_id, "sample_001")
        self.assertTrue(saved_path.exists())
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["race_id"], "sample_001")
        self.assertIn("marks", payload)
        self.assertLessEqual(payload["horse_scores"][0]["scores"]["risk"], 0)
        self.assertGreaterEqual(payload["horse_scores"][0]["scores"]["risk"], -10)

    def test_review_generates_review_json(self) -> None:
        run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        result, saved_path = run_review(
            race_id="sample_001",
            result_path=SAMPLE_RESULT,
            prediction_path=self.prediction_path,
            output_path=self.review_path,
            lessons_path=self.lessons_path,
        )
        self.assertTrue(saved_path.exists())
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["race_id"], "sample_001")
        self.assertIn("lessons", payload)
        self.assertEqual(result["review"].race_id, "sample_001")

    def test_lessons_json_is_appended(self) -> None:
        before = json.loads(self.lessons_path.read_text(encoding="utf-8"))
        run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        run_review(
            race_id="sample_001",
            result_path=SAMPLE_RESULT,
            prediction_path=self.prediction_path,
            output_path=self.review_path,
            lessons_path=self.lessons_path,
        )
        after = json.loads(self.lessons_path.read_text(encoding="utf-8"))
        self.assertGreater(len(after), 0)
        self.assertTrue(any(item.get("course") == "東京" for item in after))
        self.assertTrue(all("lesson_id" in item for item in after))
        self.assertTrue(all("score" in item for item in after))


if __name__ == "__main__":
    unittest.main()

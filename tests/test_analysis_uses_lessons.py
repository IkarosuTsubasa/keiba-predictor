from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.main import run_analysis


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"


class AnalysisUsesLessonsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.lessons_path = self.temp_path / "lessons.json"
        self.prediction_path = self.temp_path / "prediction.json"
        self.lessons_path.write_text(
            json.dumps(
                [
                    {
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "lesson": "同条件では近走の同距離・同コース実績を優先して評価する。",
                        "confidence": "medium",
                        "source_race_id": "202605020811",
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_analysis_writes_used_lessons_and_summary(self) -> None:
        prediction, saved_path = run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertTrue(saved_path.exists())
        self.assertGreater(len(payload["used_lessons"]), 0)
        self.assertEqual(payload["used_lessons"][0]["course"], "東京")
        self.assertIn("lessons使用数=1", payload["summary"])
        self.assertTrue(
            "過去lesson参考" in payload["summary"]
            or any("過去lesson" in score["reason"] for score in payload["horse_scores"])
        )
        self.assertGreater(len(prediction.used_lessons), 0)

    def test_analysis_excludes_same_race_lesson_to_avoid_leakage(self) -> None:
        self.lessons_path.write_text(
            json.dumps(
                [
                    {
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "lesson": "同一レース由来のlessonなので使わない。",
                        "confidence": "high",
                        "source_race_id": "sample_001",
                    },
                    {
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "lesson": "別レース由来のlessonなので使える。",
                        "confidence": "medium",
                        "source_race_id": "other_race_001",
                    },
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        prediction, saved_path = run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        used_source_ids = {lesson["source_race_id"] for lesson in payload["used_lessons"]}
        used_texts = {lesson["lesson"] for lesson in payload["used_lessons"]}
        self.assertNotIn("sample_001", used_source_ids)
        self.assertNotIn("同一レース由来のlessonなので使わない。", used_texts)
        self.assertIn("別レース由来のlessonなので使える。", used_texts)
        self.assertTrue(all(lesson.source_race_id != "sample_001" for lesson in prediction.used_lessons))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.memory.lesson_store import LessonStore
from keiba_llm_agent.schemas.race_data import RaceInfo


class LessonStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lessons_path = Path(self.temp_dir.name) / "lessons.json"
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
        self.store = LessonStore(self.lessons_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_find_relevant_lessons_exact_match(self) -> None:
        race_info = RaceInfo(
            race_id="sample_001",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
        )
        lessons = self.store.find_relevant_lessons(race_info)
        self.assertEqual(len(lessons), 1)

    def test_find_relevant_lessons_distance_within_200(self) -> None:
        race_info = RaceInfo(
            race_id="sample_002",
            course="東京",
            surface="芝",
            distance=1800,
            track_condition="良",
        )
        lessons = self.store.find_relevant_lessons(race_info)
        self.assertEqual(len(lessons), 1)

    def test_irrelevant_conditions_do_not_match(self) -> None:
        race_info = RaceInfo(
            race_id="sample_003",
            course="京都",
            surface="ダート",
            distance=1200,
            track_condition="重",
        )
        lessons = self.store.find_relevant_lessons(race_info)
        self.assertEqual(len(lessons), 0)


if __name__ == "__main__":
    unittest.main()

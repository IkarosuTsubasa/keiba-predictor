from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.memory.lesson_store import LessonStore
from keiba_llm_agent.schemas.review import LessonItem


class LessonConditionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lessons_path = Path(self.temp_dir.name) / "lessons.json"
        self.store = LessonStore(self.lessons_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_invalid_unknown_lesson_is_not_appended(self) -> None:
        invalid_lesson = LessonItem(
            course="unknown",
            surface="unknown",
            distance=0,
            track_condition="unknown",
            lesson="invalid",
            confidence="low",
            source_race_id="sample_001",
        )
        total = self.store.append_lessons([invalid_lesson])
        self.assertEqual(total, 0)
        payload = json.loads(self.lessons_path.read_text(encoding="utf-8"))
        self.assertEqual(payload, [])

    def test_valid_lesson_is_appended_and_searchable(self) -> None:
        valid_lesson = LessonItem(
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
            lesson="同距離実績を重視する。",
            confidence="high",
            source_race_id="202605020811",
        )
        total = self.store.append_lessons([valid_lesson])
        self.assertEqual(total, 1)
        matched = self.store.find_lessons(
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].source_race_id, "202605020811")


if __name__ == "__main__":
    unittest.main()

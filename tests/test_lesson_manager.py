from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.main import run_analysis
from keiba_llm_agent.memory.lesson_manager import LessonManager
from keiba_llm_agent.memory.lesson_store import LessonStore
from keiba_llm_agent.schemas.prediction import StrategyDecision
from keiba_llm_agent.schemas.race_data import RaceInfo
from keiba_llm_agent.schemas.review import HitSummary, LessonItem, Review


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"


class LessonManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lessons_path = Path(self.temp_dir.name) / "lessons.json"
        self.store = LessonStore(self.lessons_path)
        self.manager = LessonManager(self.lessons_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _lesson(self, lesson_text: str = "同条件では近走の同距離・同コース実績を優先して評価する。") -> LessonItem:
        return LessonItem(
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
            lesson=lesson_text,
            confidence="medium",
            source_race_id="race_a",
        )

    def test_same_lesson_is_not_duplicated(self) -> None:
        count1 = self.store.upsert_lessons([self._lesson()])
        count2 = self.store.upsert_lessons([self._lesson()])
        lessons = self.store.load_lessons()
        self.assertEqual(count1, 1)
        self.assertEqual(count2, 1)
        self.assertEqual(len(lessons), 1)
        self.assertIsNotNone(lessons[0].lesson_id)

    def test_old_format_lessons_are_migrated(self) -> None:
        self.lessons_path.write_text(
            json.dumps(
                [
                    {
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "lesson": "old lesson",
                        "confidence": "medium",
                        "source_race_id": "old_race",
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        lessons = self.store.load_lessons()
        self.assertEqual(len(lessons), 1)
        self.assertTrue(lessons[0].enabled)
        self.assertIsNotNone(lessons[0].lesson_id)
        migrated = json.loads(self.lessons_path.read_text(encoding="utf-8"))
        self.assertIn("used_count", migrated[0])
        self.assertIn("score", migrated[0])

    def test_disabled_lesson_not_returned_and_higher_score_first(self) -> None:
        lesson_a = self.store.normalize_lesson(self._lesson("A lesson")).model_copy(
            update={"score": 0.8, "enabled": True}
        )
        lesson_b = self.store.normalize_lesson(self._lesson("B lesson")).model_copy(
            update={"score": 0.1, "confidence": "low", "enabled": True}
        )
        lesson_c = self.store.normalize_lesson(self._lesson("C lesson")).model_copy(
            update={"score": 0.9, "enabled": False}
        )
        self.store.save_lessons([lesson_b, lesson_a, lesson_c])
        race_info = RaceInfo(
            race_id="x",
            race_name="x",
            race_date="2026-05-17",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
            weather="晴",
        )
        lessons = self.store.find_relevant_lessons(race_info)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0].lesson, "A lesson")

    def test_used_count_increases_after_analysis(self) -> None:
        self.store.upsert_lessons([self._lesson()])
        prediction, _ = run_analysis(SAMPLE_RACE_DATA, lessons_path=self.lessons_path)
        self.assertGreater(len(prediction.used_lessons), 0)
        lessons = self.store.load_lessons()
        self.assertEqual(lessons[0].used_count, 1)

    def test_review_success_and_failure_update_score(self) -> None:
        self.store.upsert_lessons([self._lesson()])
        lesson = self.store.load_lessons()[0]
        success_review = Review(
            race_id="race_a",
            hit_summary=HitSummary(
                main_mark_top3=True,
                marked_horses_top3_count=2,
                bet_hit=True,
                roi=1.2,
                total_stake=100,
                total_return=120,
            ),
            bet_results=[],
            good_points=[],
            bad_points=[],
            lessons=[],
        )
        strategy = StrategyDecision(
            bet_decision="BET",
            confidence="medium",
            participation_level="light",
            reason_codes=[],
            reason="x",
        )
        self.store.update_effectiveness([lesson], success_review, strategy)
        updated = self.store.load_lessons()[0]
        self.assertEqual(updated.success_count, 1)
        self.assertGreater(updated.score, 0.5)

        failure_review = Review(
            race_id="race_b",
            hit_summary=HitSummary(
                main_mark_top3=False,
                marked_horses_top3_count=0,
                bet_hit=False,
                roi=0.0,
                total_stake=100,
                total_return=0,
            ),
            bet_results=[],
            good_points=[],
            bad_points=[],
            lessons=[],
        )
        before_failure_score = updated.score
        self.store.update_effectiveness([updated], failure_review, strategy)
        updated_again = self.store.load_lessons()[0]
        self.assertEqual(updated_again.failure_count, 1)
        self.assertLess(updated_again.score, before_failure_score)

    def test_disable_enable_and_prune_commands(self) -> None:
        self.store.upsert_lessons([self._lesson()])
        lesson = self.store.load_lessons()[0]
        disabled = self.manager.disable_lesson(lesson.lesson_id or "")
        self.assertFalse(disabled.enabled)
        enabled = self.manager.enable_lesson(lesson.lesson_id or "")
        self.assertTrue(enabled.enabled)
        low_score = enabled.model_copy(update={"score": 0.1})
        self.store.save_lessons([low_score])
        disabled_count = self.manager.prune_lessons(0.2)
        self.assertEqual(disabled_count, 1)
        self.assertFalse(self.store.load_lessons()[0].enabled)


if __name__ == "__main__":
    unittest.main()

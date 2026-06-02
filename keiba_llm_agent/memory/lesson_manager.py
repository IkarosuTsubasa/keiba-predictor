from __future__ import annotations

from pathlib import Path

from keiba_llm_agent.memory.lesson_store import LessonStore
from keiba_llm_agent.schemas.review import LessonItem


class LessonManager:
    def __init__(self, lessons_path: str | Path) -> None:
        self.store = LessonStore(lessons_path)

    def list_lessons(self) -> list[LessonItem]:
        return self.store.list_lessons()

    def disable_lesson(self, lesson_id: str) -> LessonItem:
        return self.store.set_enabled(lesson_id, False)

    def enable_lesson(self, lesson_id: str) -> LessonItem:
        return self.store.set_enabled(lesson_id, True)

    def prune_lessons(self, min_score: float) -> int:
        return self.store.prune_lessons(min_score)

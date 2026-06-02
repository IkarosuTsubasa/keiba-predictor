from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from keiba_llm_agent.schemas.prediction import Prediction, StrategyDecision
from keiba_llm_agent.schemas.race_data import RaceInfo
from keiba_llm_agent.schemas.review import LessonItem, Review


CONFIDENCE_SCORES = {
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def current_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class LessonStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def ensure_exists(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]\n", encoding="utf-8")

    @staticmethod
    def generate_lesson_id(
        *,
        course: str,
        surface: str,
        distance: int,
        track_condition: str,
        lesson: str,
    ) -> str:
        raw = "||".join(
            [
                course.strip(),
                surface.strip(),
                str(distance),
                track_condition.strip(),
                lesson.strip(),
            ]
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"lesson_{digest}"

    @staticmethod
    def initial_score(confidence: str) -> float:
        return CONFIDENCE_SCORES.get(confidence, 0.5)

    @classmethod
    def normalize_lesson(cls, payload: LessonItem | dict) -> LessonItem:
        lesson = payload if isinstance(payload, LessonItem) else LessonItem.model_validate(payload)
        now = current_timestamp()
        lesson_id = lesson.lesson_id or cls.generate_lesson_id(
            course=lesson.course,
            surface=lesson.surface,
            distance=lesson.distance,
            track_condition=lesson.track_condition,
            lesson=lesson.lesson,
        )
        created_at = lesson.created_at or now
        updated_at = lesson.updated_at or created_at
        score = lesson.score
        if score == 0.5 and lesson.lesson_id is None:
            score = cls.initial_score(lesson.confidence)
        source_race_ids = list(dict.fromkeys((lesson.source_race_ids or []) + [lesson.source_race_id]))
        return lesson.model_copy(
            update={
                "lesson_id": lesson_id,
                "source_race_ids": source_race_ids,
                "created_at": created_at,
                "updated_at": updated_at,
                "enabled": lesson.enabled,
                "used_count": lesson.used_count,
                "success_count": lesson.success_count,
                "failure_count": lesson.failure_count,
                "score": max(0.0, min(1.0, score)),
            }
        )

    def load_lessons(self) -> list[LessonItem]:
        self.ensure_exists()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        normalized = self.deduplicate_lessons([self.normalize_lesson(item) for item in payload])
        serialized = [lesson.model_dump() for lesson in normalized]
        if payload != serialized:
            self.path.write_text(
                json.dumps(serialized, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return normalized

    def save_lessons(self, lessons: list[LessonItem]) -> None:
        normalized = self.deduplicate_lessons([self.normalize_lesson(lesson) for lesson in lessons])
        self.path.write_text(
            json.dumps([lesson.model_dump() for lesson in normalized], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def deduplicate_lessons(cls, lessons: list[LessonItem]) -> list[LessonItem]:
        deduped: dict[str, LessonItem] = {}
        order: list[str] = []
        for lesson in lessons:
            lesson_id = lesson.lesson_id or cls.generate_lesson_id(
                course=lesson.course,
                surface=lesson.surface,
                distance=lesson.distance,
                track_condition=lesson.track_condition,
                lesson=lesson.lesson,
            )
            if lesson_id not in deduped:
                deduped[lesson_id] = lesson.model_copy(update={"lesson_id": lesson_id})
                order.append(lesson_id)
                continue
            existing = deduped[lesson_id]
            merged = existing.model_copy(
                update={
                    "confidence": cls.merge_confidence(existing.confidence, lesson.confidence),
                    "enabled": existing.enabled or lesson.enabled,
                    "source_race_ids": list(
                        dict.fromkeys((existing.source_race_ids or []) + (lesson.source_race_ids or []) + [existing.source_race_id, lesson.source_race_id])
                    ),
                    "used_count": existing.used_count + lesson.used_count,
                    "success_count": existing.success_count + lesson.success_count,
                    "failure_count": existing.failure_count + lesson.failure_count,
                    "score": max(existing.score, lesson.score),
                    "updated_at": max(existing.updated_at or "", lesson.updated_at or "") or existing.updated_at or lesson.updated_at,
                }
            )
            deduped[lesson_id] = merged
        return [deduped[lesson_id] for lesson_id in order]

    @staticmethod
    def is_valid_lesson(lesson: LessonItem) -> bool:
        return (
            lesson.course not in ("", "unknown")
            and lesson.surface not in ("", "unknown")
            and lesson.track_condition not in ("", "unknown")
            and lesson.distance > 0
        )

    @staticmethod
    def merge_confidence(current: str, incoming: str) -> str:
        return current if CONFIDENCE_RANK[current] >= CONFIDENCE_RANK[incoming] else incoming

    def upsert_lessons(self, lessons: list[LessonItem | dict]) -> int:
        existing_lessons = self.load_lessons()
        existing = {lesson.lesson_id: lesson for lesson in existing_lessons}
        now = current_timestamp()
        new_lesson_ids: list[str] = []
        for raw_lesson in lessons:
            normalized = self.normalize_lesson(raw_lesson)
            if not self.is_valid_lesson(normalized):
                continue
            existing_lesson = existing.get(normalized.lesson_id)
            if existing_lesson is None:
                existing[normalized.lesson_id] = normalized.model_copy(
                    update={
                        "enabled": True,
                        "used_count": 0,
                        "success_count": 0,
                        "failure_count": 0,
                        "score": self.initial_score(normalized.confidence),
                        "created_at": normalized.created_at or now,
                        "updated_at": now,
                    }
                )
                new_lesson_ids.append(normalized.lesson_id or "")
                continue
            existing[normalized.lesson_id] = existing_lesson.model_copy(
                update={
                    "updated_at": now,
                    "confidence": self.merge_confidence(existing_lesson.confidence, normalized.confidence),
                    "source_race_ids": list(
                        dict.fromkeys(
                            (existing_lesson.source_race_ids or [])
                            + (normalized.source_race_ids or [])
                            + [existing_lesson.source_race_id, normalized.source_race_id]
                        )
                    ),
                }
            )
        existing_order = [lesson.lesson_id for lesson in existing_lessons if lesson.lesson_id in existing]
        saved_lessons = [existing[lesson_id] for lesson_id in existing_order]
        saved_lessons.extend(existing[lesson_id] for lesson_id in new_lesson_ids if lesson_id in existing)
        self.save_lessons(saved_lessons)
        return len(saved_lessons)

    def append_lessons(self, lessons: list[LessonItem]) -> int:
        return self.upsert_lessons(lessons)

    def list_lessons(self) -> list[LessonItem]:
        return sorted(
            self.load_lessons(),
            key=lambda lesson: (
                not lesson.enabled,
                -lesson.score,
                -lesson.used_count,
                lesson.lesson_id or "",
            ),
        )

    def set_enabled(self, lesson_id: str, enabled: bool) -> LessonItem:
        lessons = self.load_lessons()
        updated: list[LessonItem] = []
        matched: LessonItem | None = None
        now = current_timestamp()
        for lesson in lessons:
            if lesson.lesson_id == lesson_id:
                matched = lesson.model_copy(update={"enabled": enabled, "updated_at": now})
                updated.append(matched)
            else:
                updated.append(lesson)
        if matched is None:
            raise ValueError(f"lesson not found: lesson_id={lesson_id}")
        self.save_lessons(updated)
        return matched

    def prune_lessons(self, min_score: float) -> int:
        lessons = self.load_lessons()
        disabled_count = 0
        now = current_timestamp()
        updated: list[LessonItem] = []
        for lesson in lessons:
            if lesson.score < min_score and lesson.enabled:
                updated.append(lesson.model_copy(update={"enabled": False, "updated_at": now}))
                disabled_count += 1
            else:
                updated.append(lesson)
        self.save_lessons(updated)
        return disabled_count

    def find_lessons(
        self,
        *,
        course: str,
        surface: str,
        distance: int,
        track_condition: str,
    ) -> list[LessonItem]:
        return [
            lesson
            for lesson in self.load_lessons()
            if lesson.enabled
            and lesson.course == course
            and lesson.surface == surface
            and lesson.distance == distance
            and lesson.track_condition == track_condition
        ]

    @staticmethod
    def similarity_score(race_info: RaceInfo, lesson: LessonItem) -> int:
        score = 0
        if race_info.course and lesson.course == race_info.course:
            score += 3
        if race_info.surface and lesson.surface == race_info.surface:
            score += 2
        if race_info.distance is not None and abs(lesson.distance - race_info.distance) <= 200:
            score += 2
        if race_info.track_condition and lesson.track_condition == race_info.track_condition:
            score += 1
        return score

    @classmethod
    def filter_relevant_lessons(
        cls,
        lessons: list[LessonItem],
        race_info: RaceInfo,
        threshold: int = 4,
        top_n: int = 5,
        current_race_id: str | None = None,
    ) -> list[LessonItem]:
        scored_lessons = [
            (cls.similarity_score(race_info, lesson), lesson)
            for lesson in lessons
            if cls.is_valid_lesson(lesson)
            and lesson.enabled
            and not (lesson.confidence == "low" and lesson.score < 0.2)
            and not (
                current_race_id is not None
                and (
                    lesson.source_race_id == current_race_id
                    or current_race_id in (lesson.source_race_ids or [])
                )
            )
        ]
        relevant = [
            lesson
            for score, lesson in sorted(
                scored_lessons,
                key=lambda item: (
                    -item[0],
                    -item[1].score,
                    -item[1].success_count,
                    item[1].lesson_id or "",
                ),
            )
            if score >= threshold
        ]
        return relevant[:top_n]

    def find_relevant_lessons(
        self,
        race_info: RaceInfo,
        threshold: int = 4,
        top_n: int = 5,
        current_race_id: str | None = None,
    ) -> list[LessonItem]:
        return self.filter_relevant_lessons(
            self.load_lessons(),
            race_info,
            threshold=threshold,
            top_n=top_n,
            current_race_id=current_race_id,
        )

    def mark_lessons_used(self, lessons: list[LessonItem]) -> list[LessonItem]:
        existing = {lesson.lesson_id: lesson for lesson in self.load_lessons()}
        now = current_timestamp()
        updated_used_lessons: list[LessonItem] = []
        changed = False
        for lesson in lessons:
            normalized = self.normalize_lesson(lesson)
            existing_lesson = existing.get(normalized.lesson_id)
            if existing_lesson is None:
                continue
            updated_lesson = existing_lesson.model_copy(
                update={
                    "used_count": existing_lesson.used_count + 1,
                    "updated_at": now,
                }
            )
            existing[updated_lesson.lesson_id] = updated_lesson
            updated_used_lessons.append(updated_lesson)
            changed = True
        if changed:
            self.save_lessons(list(existing.values()))
        return updated_used_lessons

    def update_effectiveness(
        self,
        used_lessons: list[LessonItem],
        review: Review,
        strategy: StrategyDecision | None = None,
    ) -> list[LessonItem]:
        if not used_lessons:
            return []
        existing = {lesson.lesson_id: lesson for lesson in self.load_lessons()}
        now = current_timestamp()
        updated_lessons: list[LessonItem] = []
        for used_lesson in used_lessons:
            normalized = self.normalize_lesson(used_lesson)
            existing_lesson = existing.get(normalized.lesson_id)
            if existing_lesson is None:
                continue

            score = existing_lesson.score
            success_count = existing_lesson.success_count
            failure_count = existing_lesson.failure_count
            if review.hit_summary.main_mark_top3 or review.hit_summary.marked_horses_top3_count >= 2:
                success_count += 1
                score += 0.05
            if review.hit_summary.marked_horses_top3_count == 0:
                failure_count += 1
                score -= 0.08
            if review.hit_summary.bet_hit:
                score += 0.05
            if (
                strategy is not None
                and strategy.bet_decision == "BET"
                and not review.hit_summary.bet_hit
            ):
                score -= 0.03

            updated = existing_lesson.model_copy(
                update={
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "score": max(0.0, min(1.0, round(score, 4))),
                    "updated_at": now,
                }
            )
            existing[updated.lesson_id] = updated
            updated_lessons.append(updated)
        self.save_lessons(list(existing.values()))
        return updated_lessons

    def sync_prediction_used_lessons(self, prediction: Prediction) -> Prediction:
        updated_lessons = self.mark_lessons_used(prediction.used_lessons)
        if not updated_lessons:
            return prediction
        return prediction.model_copy(update={"used_lessons": updated_lessons})

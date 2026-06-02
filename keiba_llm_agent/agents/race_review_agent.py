from __future__ import annotations

from pathlib import Path

from keiba_llm_agent.agents.review_support import (
    PAYOUT_MISSING_NOTE_JA,
    calculate_review_metrics,
)
from keiba_llm_agent.llm import BaseLLMClient
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceInfo
from keiba_llm_agent.schemas.review import BetResultItem, LessonItem, Review
from keiba_llm_agent.simulation.simulation_reviewer import review_race_simulation


class RaceReviewAgent:
    def __init__(self, llm_client: BaseLLMClient, prompt_path: str | Path | None = None) -> None:
        default_prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "review_prompt.txt"
        self.llm_client = llm_client
        self.prompt_path = Path(prompt_path) if prompt_path else default_prompt_path
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

    def run(
        self,
        race_id: str,
        result: dict,
        prediction: Prediction,
        race_info: RaceInfo | dict | None = None,
    ) -> Review:
        serialized_race_info = None
        if isinstance(race_info, RaceInfo):
            serialized_race_info = race_info.model_dump()
        elif isinstance(race_info, dict):
            serialized_race_info = race_info

        result_payload = dict(result)
        if serialized_race_info and "race_info" not in result_payload:
            result_payload["race_info"] = serialized_race_info

        payload = {
            "race_id": race_id,
            "result": result_payload,
            "prediction": prediction.model_dump(),
            "race_info": serialized_race_info or (prediction.race_info.model_dump() if prediction.race_info else None),
        }
        response = self.llm_client.generate_review(self.prompt_template, payload)
        review = Review.model_validate(response)

        deterministic_metrics = calculate_review_metrics(prediction=prediction, result=result_payload)
        review.bet_results = [
            BetResultItem.model_validate(item) for item in deterministic_metrics["bet_results"]
        ]
        review.hit_summary.main_mark_top3 = deterministic_metrics["main_mark_top3"]
        review.hit_summary.marked_horses_top3_count = deterministic_metrics["marked_horses_top3_count"]
        review.hit_summary.bet_hit = deterministic_metrics["bet_hit"]
        review.hit_summary.total_stake = deterministic_metrics["total_stake"]
        review.hit_summary.total_return = deterministic_metrics["total_return"]
        review.hit_summary.roi = deterministic_metrics["roi"]
        review.payout_warning = deterministic_metrics["payout_warning"]
        review.review_warnings = deterministic_metrics["review_warnings"]
        if review.payout_warning and PAYOUT_MISSING_NOTE_JA not in review.bad_points:
            review.bad_points.append(PAYOUT_MISSING_NOTE_JA)

        enhancement = self.llm_client.enhance_review(
            prediction=prediction,
            result=result_payload,
            review=review,
        )
        if enhancement:
            good_points = enhancement.get("good_points")
            bad_points = enhancement.get("bad_points")
            lessons = enhancement.get("lessons")
            if isinstance(good_points, list) and all(isinstance(item, str) for item in good_points):
                review.good_points = good_points
            if isinstance(bad_points, list) and all(isinstance(item, str) for item in bad_points):
                review.bad_points = bad_points
            if isinstance(lessons, list):
                valid_lessons: list[LessonItem] = []
                for lesson in lessons:
                    try:
                        valid_lessons.append(LessonItem.model_validate(lesson))
                    except Exception:
                        continue
                if valid_lessons:
                    review.lessons = valid_lessons
        if review.payout_warning and PAYOUT_MISSING_NOTE_JA not in review.bad_points:
            review.bad_points.append(PAYOUT_MISSING_NOTE_JA)
        simulation_review = review_race_simulation(
            prediction=prediction,
            result=result_payload,
            review=review,
            llm_client=self.llm_client,
        )
        review.simulation_review = simulation_review
        if simulation_review.new_lessons:
            merged_lessons = review.lessons + simulation_review.new_lessons
            deduped_lessons: list[LessonItem] = []
            seen_keys: set[tuple[str, str, int, str, str]] = set()
            for lesson in merged_lessons:
                key = (
                    lesson.course,
                    lesson.surface,
                    lesson.distance,
                    lesson.track_condition,
                    lesson.lesson,
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped_lessons.append(lesson)
            review.lessons = deduped_lessons
        return review

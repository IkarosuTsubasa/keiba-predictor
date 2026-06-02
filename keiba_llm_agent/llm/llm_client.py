from __future__ import annotations

import json
import re
import warnings
from abc import ABC, abstractmethod

from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.review import LessonItem, Review


PREDICTION_ENHANCEMENT_PROMPT = """あなたは競馬予想文章の補助LLMです。
返答は必ずJSONのみで、summary / risks / commentary を含めてください。
horse_scores / marks / strategy は変更してはいけません。
summary は短い日本語要約、risks は日本語文字列配列、commentary は補足コメントにしてください。
"""

REVIEW_ENHANCEMENT_PROMPT = """あなたは競馬回顧文章の補助LLMです。
返答は必ずJSONのみで、good_points / bad_points / lessons を含めてください。
prediction / result / hit_summary / bet_results に含まれない事実を作ってはいけません。
lessons は review schema に従うJSON配列で返してください。
"""


def extract_json_object(text: str) -> dict:
    payload_text = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", payload_text, flags=re.DOTALL)
    if fenced_match:
        payload_text = fenced_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", payload_text, flags=re.DOTALL)
        if object_match:
            payload_text = object_match.group(0)
    return json.loads(payload_text)


class BaseLLMClient(ABC):
    def __init__(self, fallback_client: BaseLLMClient | None = None) -> None:
        self.fallback_client = fallback_client
        self.last_fallback_used = False

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str | None = None,
    ) -> dict:
        raise NotImplementedError

    def generate_analysis(self, prompt: str, payload: dict) -> dict:
        if self.fallback_client is None:
            raise NotImplementedError("generate_analysis is not implemented")
        return self.fallback_client.generate_analysis(prompt, payload)

    def generate_review(self, prompt: str, payload: dict) -> dict:
        if self.fallback_client is None:
            raise NotImplementedError("generate_review is not implemented")
        return self.fallback_client.generate_review(prompt, payload)

    def enhance_prediction(
        self,
        prediction: Prediction,
        race_data: RaceData,
        used_lessons: list[LessonItem],
    ) -> dict | None:
        payload = {
            "race_info": race_data.race_info.model_dump(),
            "marks": prediction.marks,
            "top5_horse_scores": [horse_score.model_dump() for horse_score in prediction.horse_scores[:5]],
            "strategy": prediction.strategy.model_dump() if prediction.strategy else None,
            "summary": prediction.summary,
            "risks": prediction.risks,
            "used_lessons": [lesson.model_dump() for lesson in used_lessons],
        }
        try:
            return self.generate_json(
                PREDICTION_ENHANCEMENT_PROMPT,
                json.dumps(payload, ensure_ascii=False, indent=2),
                schema_name="prediction_enhancement",
            )
        except Exception as exc:
            warnings.warn(f"prediction enhancement failed: {exc}", stacklevel=2)
            return None

    def enhance_review(
        self,
        prediction: Prediction,
        result: dict,
        review: Review,
    ) -> dict | None:
        payload = {
            "prediction": prediction.model_dump(),
            "result": result,
            "hit_summary": review.hit_summary.model_dump(),
            "bet_results": [bet_result.model_dump() for bet_result in review.bet_results],
            "good_points": review.good_points,
            "bad_points": review.bad_points,
            "lessons": [lesson.model_dump() for lesson in review.lessons],
        }
        try:
            return self.generate_json(
                REVIEW_ENHANCEMENT_PROMPT,
                json.dumps(payload, ensure_ascii=False, indent=2),
                schema_name="review_enhancement",
            )
        except Exception as exc:
            warnings.warn(f"review enhancement failed: {exc}", stacklevel=2)
            return None

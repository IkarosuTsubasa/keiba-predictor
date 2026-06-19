from __future__ import annotations

import json
import re
import warnings
from abc import ABC, abstractmethod

from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.review import LessonItem, Review


PREDICTION_ENHANCEMENT_PROMPT = """あなたは世界水準のプロ馬券師として、公開レース詳細に載せる予想文を仕上げる補助LLMです。
返答は必ずJSONのみで、summary / risks / commentary / top_horse_memos を含めてください。
horse_scores / marks / strategy は変更してはいけません。
summary は本命と相手関係を一言で伝える短い日本語要約にしてください。
risks は「判断理由」にそのまま表示されるため、馬券判断として意味のある材料だけを2〜3件に絞ってください。
commentary は公開用の補足コメントにしてください。
top_horse_memos は上位5頭それぞれについて [{"horse_no": 1, "memo": "..."}] の形式で返してください。
top_horse_memos の memo は1頭あたり自然な日本語1文。馬の強み、勝負所、嫌う材料を具体的に書き、採点式の説明にしないこと。
口調は冷静で鋭いプロ馬券師。断定しすぎず、ただし素人向けの説明臭さは避けてください。
禁止事項:
- ルールベース、heuristic、機械学習モデル、ML model、内部実装、データ処理、欠損、unknown、fallback などのシステム説明を書かないこと。
- オッズや人気が入力にない場合、オッズ条件、回収期待値、妙味、市場評価、人気、配当を根拠にしないこと。
- 「正式な機械学習モデルではない」のような免責文を出さないこと。
- 提供されていない事実を作らないこと。
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
            "market_data_available": any(
                horse.odds is not None or horse.popularity is not None for horse in race_data.horses
            ),
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

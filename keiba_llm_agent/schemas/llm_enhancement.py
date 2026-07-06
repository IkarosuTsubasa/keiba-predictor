from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from keiba_llm_agent.schemas.prediction import TopHorseMemo
from keiba_llm_agent.schemas.review import LessonItem


class PredictionEnhancement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_reason: str | None = None
    summary: str | None = None
    risks: list[str] = Field(default_factory=list)
    commentary: str | None = None
    top_horse_memos: list[TopHorseMemo] = Field(default_factory=list)


class ReviewEnhancement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    good_points: list[str] = Field(default_factory=list)
    bad_points: list[str] = Field(default_factory=list)
    lessons: list[LessonItem] = Field(default_factory=list)

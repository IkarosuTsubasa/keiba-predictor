from __future__ import annotations

from pydantic import BaseModel, Field


class RaceLevelAnalysis(BaseModel):
    horse_no: int
    horse_name: str
    positive_flags: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    head_to_head_summary: str
    race_level_summary: str
    opponent_context_summary: str
    overall_comment: str
    adjustment_hint: float = 0.0


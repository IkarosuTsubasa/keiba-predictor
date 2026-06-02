from __future__ import annotations

from pydantic import BaseModel, Field


class HorsePaceAnalysis(BaseModel):
    horse_no: int
    horse_name: str
    running_style: str
    early_position_score: float = 0.0
    late_position_score: float = 0.0
    position_stability: str
    positive_flags: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    overall_comment: str


class RacePaceProjection(BaseModel):
    projected_pace: str
    front_runner_count: int = 0
    stalker_count: int = 0
    closer_count: int = 0
    pace_comment: str
    favorable_styles: list[str] = Field(default_factory=list)
    risk_styles: list[str] = Field(default_factory=list)


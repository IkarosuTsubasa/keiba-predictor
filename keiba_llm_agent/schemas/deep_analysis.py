from __future__ import annotations

from pydantic import BaseModel, Field


class HorseDeepAnalysis(BaseModel):
    horse_no: int
    horse_name: str
    positive_flags: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    recent_form_summary: str
    distance_analysis: str
    course_analysis: str
    track_condition_analysis: str
    jockey_analysis: str
    odds_analysis: str
    overall_comment: str

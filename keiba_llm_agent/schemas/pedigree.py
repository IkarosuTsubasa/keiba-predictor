from __future__ import annotations

from pydantic import BaseModel, Field


class PedigreeInfo(BaseModel):
    horse_id: str
    horse_name: str | None = None
    sire: str | None = None
    dam: str | None = None
    damsire: str | None = None


class PedigreeAnalysis(BaseModel):
    horse_no: int
    horse_name: str
    sire: str | None = None
    dam: str | None = None
    damsire: str | None = None
    surface_tendency: str
    distance_tendency: str
    track_condition_tendency: str
    pace_tendency: str
    positive_flags: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    overall_comment: str

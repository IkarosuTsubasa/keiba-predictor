from __future__ import annotations

from pydantic import BaseModel, Field


class PedigreeInfo(BaseModel):
    horse_id: str
    horse_name: str | None = None
    sire: str | None = None
    sire_id: str | None = None
    dam: str | None = None
    dam_id: str | None = None
    damsire: str | None = None
    damsire_id: str | None = None
    sire_sire: str | None = None
    sire_sire_id: str | None = None


class PedigreePerformanceProfile(BaseModel):
    relation: str
    horse_id: str
    horse_name: str | None = None
    starts: int = 0
    wins: int = 0
    top3: int = 0
    surface_tendency: str = "unknown"
    distance_tendency: str = "unknown"
    track_condition_tendency: str = "unknown"
    pace_tendency: str = "unknown"
    class_power: str = "unknown"
    early_maturity: str = "unknown"
    positive_flags: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    score_hint: float = 0.0
    overall_comment: str = ""


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
    performance_profiles: list[PedigreePerformanceProfile] = Field(default_factory=list)
    performance_score_hint: float = 0.0
    overall_comment: str

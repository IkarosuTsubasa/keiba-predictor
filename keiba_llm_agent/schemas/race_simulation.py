from __future__ import annotations

from pydantic import BaseModel, Field


class FavorableHorse(BaseModel):
    horse_no: int
    horse_name: str
    reason: str


class RiskHorse(BaseModel):
    horse_no: int
    horse_name: str
    reason: str


class RaceSimulation(BaseModel):
    race_id: str
    projected_pace: str = "unknown"
    race_flow: str = ""
    key_positions: str = ""
    favorable_horses: list[FavorableHorse] = Field(default_factory=list)
    risk_horses: list[RiskHorse] = Field(default_factory=list)
    win_scenario: str = ""
    top3_scenario: str = ""
    betting_scenario: str = ""
    confidence_comment: str = ""
    reasoning_summary: str = ""
    warnings: list[str] = Field(default_factory=list)

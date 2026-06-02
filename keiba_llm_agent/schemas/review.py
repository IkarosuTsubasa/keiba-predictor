from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HitSummary(BaseModel):
    main_mark_top3: bool
    marked_horses_top3_count: int
    bet_hit: bool
    roi: float
    total_stake: int
    total_return: int


class BetResultItem(BaseModel):
    bet_type: str
    horse_numbers: list[int] = Field(default_factory=list)
    amount: int = 0
    hit: bool
    payout: int
    return_amount: int


class LessonItem(BaseModel):
    lesson_id: str | None = None
    course: str
    surface: str
    distance: int
    track_condition: str
    lesson: str
    confidence: Literal["low", "medium", "high"]
    source_race_id: str
    source_race_ids: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    enabled: bool = True
    used_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    score: float = 0.5


class FavorableHorseResult(BaseModel):
    horse_no: int
    horse_name: str
    predicted_reason: str
    finish: int | None = None
    status: Literal["top3", "close", "mid", "failed", "unknown"] = "unknown"
    result: Literal["hit", "miss", "partial", "unknown"]
    comment: str


class RiskHorseResult(BaseModel):
    horse_no: int
    horse_name: str
    predicted_risk: str
    finish: int | None = None
    status: Literal["top3", "close", "mid", "failed", "unknown"] = "unknown"
    result: Literal["risk_materialized", "risk_not_materialized", "unknown"]
    comment: str


class SimulationReview(BaseModel):
    race_id: str
    pace_prediction_review: str
    scenario_hit_level: Literal["high", "medium", "low", "unknown"]
    favorable_horses_result: list[FavorableHorseResult] = Field(default_factory=list)
    risk_horses_result: list[RiskHorseResult] = Field(default_factory=list)
    win_scenario_review: str
    top3_scenario_review: str
    betting_scenario_review: str
    good_points: list[str] = Field(default_factory=list)
    bad_points: list[str] = Field(default_factory=list)
    new_lessons: list[LessonItem] = Field(default_factory=list)
    overall_comment: str


class Review(BaseModel):
    race_id: str
    hit_summary: HitSummary
    bet_results: list[BetResultItem] = Field(default_factory=list)
    good_points: list[str] = Field(default_factory=list)
    bad_points: list[str] = Field(default_factory=list)
    lessons: list[LessonItem] = Field(default_factory=list)
    payout_warning: bool = False
    review_warnings: list[str] = Field(default_factory=list)
    simulation_review: SimulationReview | None = None

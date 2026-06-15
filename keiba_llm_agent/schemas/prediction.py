from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from keiba_llm_agent.schemas.deep_analysis import HorseDeepAnalysis
from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis
from keiba_llm_agent.schemas.race_data import RaceInfo
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis
from keiba_llm_agent.schemas.race_simulation import RaceSimulation
from keiba_llm_agent.schemas.review import LessonItem


REQUIRED_MARKS = ("◎", "○", "▲", "△", "☆")


class ScoreBreakdown(BaseModel):
    recent_form: int
    distance_fit: int
    course_fit: int
    track_condition_fit: int
    jockey_fit: int
    odds_value: int
    risk: int = Field(ge=-10, le=0)
    ability_score: int = 0
    recent_quality_score: int = 0
    trend_score: int = 5
    condition_fit_score: int = 0
    race_level_score: int = 5
    pace_jockey_score: int = 0


class PedigreeAdjustment(BaseModel):
    pedigree_bonus: float = 0.0
    pedigree_penalty: float = 0.0
    pedigree_adjustment: float = 0.0
    reason: str = ""


class ScoreAdjustment(BaseModel):
    adjustment: float = 0.0
    reason: str = ""


class ScoringConfigSnapshot(BaseModel):
    scoring_mode: str = "candidate_default"
    pedigree_weight: float = 0.2
    race_level_weight: float = 1.0
    pace_weight: float = 0.0
    conditional_weight_profile: str = "none"
    use_market_score_in_ranking: bool = False
    market_signal_weight: float = 0.0


class MarketSignalConfigSnapshot(BaseModel):
    use_market_score_in_ranking: bool = False
    market_signal_weight: float = 0.0


class BorderlineRecoveryConfigSnapshot(BaseModel):
    enabled: bool = False
    max_rank: int = 6
    max_score_gap: float = 1.0
    min_net_signal: int = 2
    max_recoveries_per_race: int = 1


class BorderlineRecoveryCase(BaseModel):
    horse_no: int
    horse_name: str
    original_rank: int
    score_gap_to_top5: float
    recovery_bonus: float
    recovery_reasons: list[str] = Field(default_factory=list)
    new_rank: int


class HorseBorderlineRecovery(BaseModel):
    applied: bool = False
    recovery_bonus: float = 0.0
    original_rank: int | None = None
    new_rank: int | None = None
    reasons: list[str] = Field(default_factory=list)


class BorderlineRecoveryResult(BaseModel):
    recovery_applied: bool = False
    recovery_cases: list[BorderlineRecoveryCase] = Field(default_factory=list)


class TotalScoreBreakdown(BaseModel):
    base_total_score: float = 0.0
    pedigree_adjustment_raw: float = 0.0
    pedigree_weight: float = 0.2
    pedigree_adjustment_weighted: float = 0.0
    race_level_adjustment_raw: float = 0.0
    race_level_weight: float = 1.0
    race_level_adjustment_weighted: float = 0.0
    pace_adjustment_raw: float = 0.0
    pace_weight: float = 0.0
    pace_adjustment_weighted: float = 0.0
    borderline_recovery_bonus: float = 0.0
    total_score: float = 0.0
    total_score_after_recovery: float = 0.0


class HorseScore(BaseModel):
    horse_no: int
    horse_name: str
    scores: ScoreBreakdown
    base_total_score: float = 0.0
    pedigree_adjustment: PedigreeAdjustment = Field(default_factory=PedigreeAdjustment)
    race_level_adjustment: ScoreAdjustment = Field(default_factory=ScoreAdjustment)
    pace_adjustment: ScoreAdjustment = Field(default_factory=ScoreAdjustment)
    score_breakdown: TotalScoreBreakdown = Field(default_factory=TotalScoreBreakdown)
    odds: float | None = None
    popularity: int | None = None
    borderline_recovery: HorseBorderlineRecovery = Field(default_factory=HorseBorderlineRecovery)
    total_score: float
    reason: str


class BetSuggestion(BaseModel):
    bet_type: str
    horse_numbers: list[int] = Field(default_factory=list)
    amount: int | None = None
    reason: str | None = None


class StrategyDecision(BaseModel):
    bet_decision: Literal["BET", "SKIP"]
    confidence: Literal["low", "medium", "high"]
    participation_level: Literal["none", "light", "normal", "strong"]
    reason_codes: list[str] = Field(default_factory=list)
    reason: str


class Prediction(BaseModel):
    race_id: str
    race_info: RaceInfo | None = None
    scoring_profile: str = "accuracy_default"
    scoring_mode: str = "candidate_default"
    borderline_recovery_enabled: bool = True
    scoring_config: ScoringConfigSnapshot = Field(default_factory=ScoringConfigSnapshot)
    market_signal_config: MarketSignalConfigSnapshot = Field(default_factory=MarketSignalConfigSnapshot)
    borderline_recovery_config: BorderlineRecoveryConfigSnapshot = Field(default_factory=BorderlineRecoveryConfigSnapshot)
    marks: dict[str, int]
    horse_scores: list[HorseScore] = Field(default_factory=list)
    bets: list[BetSuggestion] = Field(default_factory=list)
    summary: str
    commentary: str | None = None
    risks: list[str] = Field(default_factory=list)
    used_lessons: list[LessonItem] = Field(default_factory=list)
    deep_analyses: list[HorseDeepAnalysis] = Field(default_factory=list)
    pedigree_analyses: list[PedigreeAnalysis] = Field(default_factory=list)
    race_level_analyses: list[RaceLevelAnalysis] = Field(default_factory=list)
    pace_analyses: list[HorsePaceAnalysis] = Field(default_factory=list)
    race_pace_projection: RacePaceProjection | None = None
    race_simulation: RaceSimulation | None = None
    borderline_recovery_result: BorderlineRecoveryResult = Field(default_factory=BorderlineRecoveryResult)
    strategy: StrategyDecision | None = None

    @field_validator("marks")
    @classmethod
    def validate_marks(cls, value: dict[str, int]) -> dict[str, int]:
        missing = [mark for mark in REQUIRED_MARKS if mark not in value]
        if missing:
            raise ValueError(f"marks 缺少字段: {', '.join(missing)}")
        return value

from __future__ import annotations

from keiba_llm_agent.config.scoring_config import DEFAULT_SCORING_WEIGHTS
from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.prediction import ScoreAdjustment, TotalScoreBreakdown
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis

DEFAULT_PEDIGREE_SCORE_WEIGHT = DEFAULT_SCORING_WEIGHTS["pedigree_weight"]
DEFAULT_RACE_LEVEL_SCORE_WEIGHT = DEFAULT_SCORING_WEIGHTS["race_level_weight"]
DEFAULT_PACE_SCORE_WEIGHT = DEFAULT_SCORING_WEIGHTS["pace_weight"]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _format_signed(value: float) -> str:
    return f"{value:+.1f}"


def apply_adjustment_weight(value: float, weight: float) -> float:
    return round(value * weight, 1)


def calculate_race_level_adjustment(
    race_level_analysis: RaceLevelAnalysis | None,
) -> ScoreAdjustment:
    if race_level_analysis is None:
        return ScoreAdjustment(adjustment=0.0, reason="")

    adjustment = round(_clamp(race_level_analysis.adjustment_hint, -1.0, 1.0), 1)
    if adjustment > 0:
        if "HEAD_TO_HEAD_POSITIVE" in race_level_analysis.positive_flags:
            reason = f"相手関係・レースレベル面から{_format_signed(adjustment)}補正。"
        else:
            reason = f"レースレベル面から{_format_signed(adjustment)}補正。"
    elif adjustment < 0:
        if "HEAD_TO_HEAD_NEGATIVE" in race_level_analysis.risk_flags:
            reason = f"同組比較で劣勢のため{_format_signed(adjustment)}補正。"
        else:
            reason = f"相手関係・レースレベル面で{_format_signed(adjustment)}補正。"
    else:
        reason = ""
    return ScoreAdjustment(adjustment=adjustment, reason=reason)


def calculate_pace_adjustment(
    pace_analysis: HorsePaceAnalysis | None,
    race_pace_projection: RacePaceProjection | None,
) -> ScoreAdjustment:
    if pace_analysis is None or race_pace_projection is None:
        return ScoreAdjustment(adjustment=0.0, reason="")

    projected_pace = race_pace_projection.projected_pace
    positive_flags = set(pace_analysis.positive_flags)
    risk_flags = set(pace_analysis.risk_flags)
    adjustment = 0.0

    if "PACE_FIT" in positive_flags:
        adjustment += 0.4
    if "STALKER_ADVANTAGE" in positive_flags:
        adjustment += 0.3
    if "CLOSING_SPEED" in positive_flags and projected_pace in {"fast", "average"}:
        adjustment += 0.3
    if "POSITION_STABLE" in positive_flags:
        adjustment += 0.2
    if "FRONT_RUNNING_ADVANTAGE" in positive_flags and projected_pace in {"slow", "average"}:
        adjustment += 0.2

    if "PACE_MISMATCH" in risk_flags:
        adjustment -= 0.5
    if "POSITION_UNSTABLE" in risk_flags:
        adjustment -= 0.2
    if pace_analysis.running_style == "追込" and projected_pace == "slow":
        adjustment -= 0.4
    if pace_analysis.running_style == "逃げ" and projected_pace == "fast":
        adjustment -= 0.4

    adjustment = round(_clamp(adjustment, -0.8, 0.8), 1)
    if adjustment > 0:
        reason = f"展開面から{_format_signed(adjustment)}補正。"
    elif adjustment < 0:
        reason = f"展開面で{_format_signed(adjustment)}補正。"
    else:
        reason = ""
    return ScoreAdjustment(adjustment=adjustment, reason=reason)


def build_score_breakdown(
    base_total_score: float,
    pedigree_adjustment_raw: float,
    pedigree_weight: float,
    race_level_adjustment_raw: float,
    race_level_weight: float,
    pace_adjustment_raw: float,
    pace_weight: float,
) -> TotalScoreBreakdown:
    pedigree_adjustment_weighted = apply_adjustment_weight(pedigree_adjustment_raw, pedigree_weight)
    race_level_adjustment_weighted = apply_adjustment_weight(race_level_adjustment_raw, race_level_weight)
    pace_adjustment_weighted = apply_adjustment_weight(pace_adjustment_raw, pace_weight)
    total_score = round(
        base_total_score
        + pedigree_adjustment_weighted
        + race_level_adjustment_weighted
        + pace_adjustment_weighted,
        1,
    )
    return TotalScoreBreakdown(
        base_total_score=base_total_score,
        pedigree_adjustment_raw=pedigree_adjustment_raw,
        pedigree_weight=pedigree_weight,
        pedigree_adjustment_weighted=pedigree_adjustment_weighted,
        race_level_adjustment_raw=race_level_adjustment_raw,
        race_level_weight=race_level_weight,
        race_level_adjustment_weighted=race_level_adjustment_weighted,
        pace_adjustment_raw=pace_adjustment_raw,
        pace_weight=pace_weight,
        pace_adjustment_weighted=pace_adjustment_weighted,
        borderline_recovery_bonus=0.0,
        top_choice_refinement_bonus=0.0,
        total_score=total_score,
        total_score_after_recovery=total_score,
        total_score_after_refinement=total_score,
    )

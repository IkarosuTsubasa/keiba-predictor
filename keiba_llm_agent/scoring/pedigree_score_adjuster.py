from __future__ import annotations

from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis
from keiba_llm_agent.schemas.prediction import PedigreeAdjustment
from keiba_llm_agent.schemas.race_data import RaceInfo


BONUS_BY_FLAG = {
    "PEDIGREE_SURFACE_FIT": 0.5,
    "PEDIGREE_DISTANCE_FIT": 0.8,
    "PEDIGREE_STAMINA_FIT": 0.8,
    "PEDIGREE_POWER_FIT": 0.3,
    "PEDIGREE_TRACK_CONDITION_FIT": 0.5,
    "PEDIGREE_CLASS_POWER": 0.4,
    "PEDIGREE_EARLY_MATURITY": 0.4,
}

PENALTY_BY_FLAG = {
    "PEDIGREE_DISTANCE_RISK": 1.0,
    "PEDIGREE_SURFACE_UNKNOWN": 0.0,
    "PEDIGREE_DISTANCE_UNKNOWN": 0.0,
    "PEDIGREE_DATA_INCOMPLETE": 0.0,
}


def calculate_pedigree_adjustment(
    pedigree_analysis: PedigreeAnalysis,
    race_info: RaceInfo,
) -> PedigreeAdjustment:
    bonus = sum(BONUS_BY_FLAG.get(flag, 0.0) for flag in pedigree_analysis.positive_flags)
    penalty = sum(PENALTY_BY_FLAG.get(flag, 0.0) for flag in pedigree_analysis.risk_flags)
    performance_hint = getattr(pedigree_analysis, "performance_score_hint", 0.0)
    if performance_hint > 0:
        bonus += min(float(performance_hint), 1.2)
    elif performance_hint < 0:
        penalty += min(abs(float(performance_hint)), 0.8)
    adjustment = max(-1.5, min(2.0, round(bonus - penalty, 1)))

    parts: list[str] = []
    if bonus > 0:
        parts.append(f"適性加点{bonus:.1f}")
    if penalty > 0:
        parts.append(f"リスク減点-{penalty:.1f}")
    if performance_hint > 0:
        parts.append(f"祖先実績+{performance_hint:.1f}")
    elif performance_hint < 0:
        parts.append(f"祖先実績{performance_hint:.1f}")
    if not parts:
        parts.append("血統補正なし")
    if race_info.distance and any(
        flag in pedigree_analysis.positive_flags
        for flag in ("PEDIGREE_STAMINA_FIT", "PEDIGREE_DISTANCE_FIT")
    ):
        parts.append(f"{race_info.distance}m向き")

    return PedigreeAdjustment(
        pedigree_bonus=round(min(bonus, 2.0), 1),
        pedigree_penalty=round(min(penalty, 1.5), 1),
        pedigree_adjustment=adjustment,
        reason=" / ".join(parts),
    )

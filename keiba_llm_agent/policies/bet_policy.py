from __future__ import annotations

from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis
from keiba_llm_agent.schemas.prediction import (
    BetSuggestion,
    HorseScore,
    ScoringConfigSnapshot,
    StrategyDecision,
)
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.review import LessonItem


def _contains_key_risk(risks: list[str]) -> bool:
    lowered = [risk.lower() for risk in risks]
    return any("data leakage" in risk or "target race result" in risk for risk in lowered)


def evaluate_bet_strategy(
    race_data: RaceData,
    horse_scores: list[HorseScore],
    marks: dict[str, int],
    used_lessons: list[LessonItem],
    risks: list[str],
    pace_analyses: list[HorsePaceAnalysis] | None = None,
    scoring_config: ScoringConfigSnapshot | None = None,
) -> StrategyDecision:
    if not horse_scores:
        return StrategyDecision(
            bet_decision="SKIP",
            confidence="low",
            participation_level="none",
            reason_codes=["TOP_SCORE_LOW", "SKIP_LOW_CONFIDENCE"],
            reason="評価材料が不足しており、見送り。",
        )

    top = horse_scores[0]
    second = horse_scores[1] if len(horse_scores) > 1 else horse_scores[0]
    third = horse_scores[2] if len(horse_scores) > 2 else second
    score_gap_1_2 = round(top.total_score - second.total_score, 1)
    score_gap_1_3 = round(top.total_score - third.total_score, 1)
    average_top3_score = round(sum(score.total_score for score in horse_scores[:3]) / min(3, len(horse_scores)), 1)

    horse_map = {horse.horse_no: horse for horse in race_data.horses}
    top_horse = horse_map.get(top.horse_no)
    top_odds = top_horse.odds if top_horse else None
    top_popularity = top_horse.popularity if top_horse else None
    top3_horses = [horse_map.get(score.horse_no) for score in horse_scores[:3]]
    top3_odds = [horse.odds for horse in top3_horses if horse and horse.odds is not None]
    has_current_odds = any(horse.odds is not None for horse in race_data.horses)
    used_lessons_count = len(used_lessons)
    clear_axis = score_gap_1_2 >= 2.0 and score_gap_1_3 >= 4.0
    close_top_group = score_gap_1_2 <= 2.0 or score_gap_1_3 <= 3.0
    value_present = any(odds >= 10 for odds in top3_odds)
    top_risk = top.scores.risk
    top_pedigree_adjustment = top.pedigree_adjustment.pedigree_adjustment if getattr(top, "pedigree_adjustment", None) else 0.0
    top_pedigree_reason = top.pedigree_adjustment.reason if getattr(top, "pedigree_adjustment", None) else ""
    top_race_level_adjustment = top.race_level_adjustment.adjustment if getattr(top, "race_level_adjustment", None) else 0.0
    top_pace_adjustment = top.pace_adjustment.adjustment if getattr(top, "pace_adjustment", None) else 0.0
    effective_scoring_config = scoring_config or ScoringConfigSnapshot()
    pace_analysis_map = {analysis.horse_no: analysis for analysis in (pace_analyses or [])}
    top_pace_analysis = pace_analysis_map.get(top.horse_no)

    reason_codes: list[str] = []
    if top.total_score >= 40:
        reason_codes.append("TOP_SCORE_HIGH")
    elif top.total_score < 32:
        reason_codes.append("TOP_SCORE_LOW")

    if clear_axis:
        reason_codes.append("CLEAR_AXIS")
    else:
        reason_codes.append("NO_CLEAR_AXIS")

    if close_top_group:
        reason_codes.append("CLOSE_TOP_GROUP")

    if has_current_odds:
        reason_codes.append("ODDS_AVAILABLE")
    else:
        reason_codes.append("ODDS_MISSING")

    if value_present:
        reason_codes.append("VALUE_PRESENT")

    if top_risk <= -7:
        reason_codes.append("HIGH_RISK_TOP")

    if used_lessons_count > 0:
        reason_codes.append("LESSON_USED")

    if effective_scoring_config.pedigree_weight > 0:
        reason_codes.append("PEDIGREE_USED")
    if effective_scoring_config.pedigree_weight > 0 and (top_pedigree_adjustment < 0 or "リスク減点" in top_pedigree_reason):
        reason_codes.append("PEDIGREE_RISK_TOP")
    if effective_scoring_config.race_level_weight > 0:
        reason_codes.append("RACE_LEVEL_USED")
    if effective_scoring_config.pace_weight > 0:
        reason_codes.append("PACE_USED")
    if effective_scoring_config.pace_weight > 0 and top_pace_analysis and (
        "PACE_MISMATCH" in top_pace_analysis.risk_flags or top_pace_adjustment < 0
    ):
        reason_codes.append("PACE_RISK_TOP")

    if any("heuristic scoring" in risk for risk in risks):
        reason_codes.append("HEURISTIC_MODEL_ONLY")

    if top.total_score >= 40 and score_gap_1_3 >= 4 and top_risk >= -3 and top_odds is not None:
        confidence = "high"
    elif top.total_score >= 35 and average_top3_score >= 34 and top_risk >= -5:
        confidence = "medium"
    else:
        confidence = "low"

    should_skip = (
        top.total_score < 32
        or top_risk <= -7
        or not has_current_odds
        or _contains_key_risk(risks)
        or (close_top_group and confidence == "low")
    )

    if should_skip:
        reason_codes.append("SKIP_LOW_CONFIDENCE")
        return StrategyDecision(
            bet_decision="SKIP",
            confidence=confidence,
            participation_level="none",
            reason_codes=reason_codes,
            reason=(
                "上位比較で決め手が弱く、リスクまたはオッズ条件も不十分なため見送り。"
            ),
        )

    if confidence == "high":
        participation_level = "strong" if clear_axis else "normal"
    elif confidence == "medium":
        participation_level = "light" if close_top_group else "normal"
    else:
        participation_level = "light"

    top_descriptor = f"本命想定は{top.horse_name}"
    odds_descriptor = (
        f"単勝{top_odds:.1f}倍・人気{top_popularity}。"
        if top_odds is not None and top_popularity is not None
        else "現行オッズは取得済み。"
    )
    group_descriptor = "上位拮抗のため点数は絞る。" if close_top_group else "軸馬が比較的明確。"
    lesson_descriptor = "過去lessonも参考。" if used_lessons_count else ""
    return StrategyDecision(
        bet_decision="BET",
        confidence=confidence,
        participation_level=participation_level,
        reason_codes=reason_codes,
        reason=f"{top_descriptor} {odds_descriptor} {group_descriptor}{lesson_descriptor}".strip(),
    )


def build_bets_from_strategy(
    strategy: StrategyDecision,
    horse_scores: list[HorseScore],
) -> list[BetSuggestion]:
    if strategy.bet_decision == "SKIP" or not horse_scores:
        return []

    top = horse_scores[0]
    second = horse_scores[1] if len(horse_scores) > 1 else None
    third = horse_scores[2] if len(horse_scores) > 2 else None
    clear_axis = "CLEAR_AXIS" in strategy.reason_codes

    if strategy.participation_level == "light":
        if clear_axis or second is None:
            return [
                BetSuggestion(
                    bet_type="複勝",
                    horse_numbers=[top.horse_no],
                    amount=100,
                    reason="参加レベルを軽めに抑え、本命の複勝のみ。"
                )
            ]
        return [
            BetSuggestion(
                bet_type="ワイド",
                horse_numbers=[top.horse_no, second.horse_no],
                amount=100,
                reason="上位拮抗のため、軽めにワイド1点。"
            )
        ]

    if strategy.participation_level == "normal":
        bets = []
        if second is not None:
            bets.append(
                BetSuggestion(
                    bet_type="ワイド",
                    horse_numbers=[top.horse_no, second.horse_no],
                    amount=100,
                    reason="上位評価の2頭を本線にする。"
                )
            )
        if third is not None:
            bets.append(
                BetSuggestion(
                    bet_type="ワイド",
                    horse_numbers=[top.horse_no, third.horse_no],
                    amount=100,
                    reason="3番手までが圏内のため押さえる。"
                )
            )
        return bets

    if strategy.participation_level == "strong":
        bets = [
            BetSuggestion(
                bet_type="複勝",
                horse_numbers=[top.horse_no],
                amount=100,
                reason="本命評価が高く、軸として複勝を取る。"
            )
        ]
        if second is not None:
            bets.append(
                BetSuggestion(
                    bet_type="ワイド",
                    horse_numbers=[top.horse_no, second.horse_no],
                    amount=100,
                    reason="上位2頭で取りに行く。"
                )
            )
        return bets

    return []

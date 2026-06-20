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


def _race_scope_key(race_data: RaceData) -> str:
    race_info = race_data.race_info
    scope_key = str(race_info.scope_key or "").strip().lower()
    if scope_key:
        return scope_key
    if race_info.source == "local":
        return "local"
    if race_info.surface == "ダート":
        return "central_dirt"
    if race_info.surface == "芝":
        return "central_turf"
    return ""


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalized_span(value: float, lower: float, width: float) -> float:
    if width <= 0:
        return 0.0
    return _clamp((value - lower) / width, 0.0, 1.0)


MARK_ORDER = ("◎", "○", "▲", "△", "☆")


def _estimate_axis_strength_score(
    *,
    race_data: RaceData,
    top_score: float,
    score_gap_1_2: float,
    score_gap_1_3: float,
    average_top3_score: float,
    top_risk: int,
    top_odds: float | None,
    top_popularity: int | None,
) -> float:
    scope_key = _race_scope_key(race_data)
    base_by_scope = {
        "local": 0.49,
        "central_dirt": 0.46,
        "central_turf": 0.42,
    }
    score = base_by_scope.get(scope_key, 0.44)

    score += 0.075 * _normalized_span(top_score, 35.0, 10.0)
    score += 0.075 * _normalized_span(score_gap_1_3, 3.0, 8.0)
    score += 0.050 * _normalized_span(score_gap_1_2, 1.5, 4.0)
    score += 0.020 * _normalized_span(average_top3_score, 34.0, 8.0)

    if top_risk >= -1:
        score += 0.025
    elif top_risk <= -5:
        score -= 0.080
    elif top_risk <= -3:
        score -= 0.035

    if top_odds is None or top_odds <= 0:
        score -= 0.030
    elif top_odds >= 20 or (top_popularity is not None and top_popularity >= 8):
        score -= 0.060
    elif top_odds <= 4 and top_popularity is not None and top_popularity <= 2:
        score += 0.025

    if score_gap_1_2 <= 2.0 or score_gap_1_3 <= 3.0:
        score -= 0.045

    if scope_key == "central_dirt":
        score = min(score, 0.58)
    elif scope_key == "central_turf":
        cap = 0.72 if score_gap_1_2 >= 4.0 and score_gap_1_3 >= 10.0 else 0.62
        score = min(score, cap)
    elif scope_key == "local":
        score = min(score, 0.74)

    return round(_clamp(score, 0.30, 0.78), 3)


def _axis_strength_label_from_score(axis_strength_score: float) -> str:
    if axis_strength_score >= 0.66:
        return "high"
    if axis_strength_score >= 0.50:
        return "medium"
    return "low"


def _marked_scores_for_coverage(
    horse_scores: list[HorseScore],
    marks: dict[str, int],
) -> list[HorseScore]:
    score_map = {score.horse_no: score for score in horse_scores}
    marked_scores: list[HorseScore] = []
    seen: set[int] = set()

    for symbol in MARK_ORDER:
        try:
            horse_no = int(marks.get(symbol) or 0)
        except (TypeError, ValueError):
            continue
        score = score_map.get(horse_no)
        if score is None or horse_no in seen:
            continue
        marked_scores.append(score)
        seen.add(horse_no)

    for score in horse_scores:
        if len(marked_scores) >= 5:
            break
        if score.horse_no in seen:
            continue
        marked_scores.append(score)
        seen.add(score.horse_no)

    return marked_scores[:5]


def _estimate_marked_top3_pair_coverage_score(
    *,
    race_data: RaceData,
    horse_scores: list[HorseScore],
    marked_scores: list[HorseScore],
) -> float:
    if not marked_scores:
        return 0.0

    scope_key = _race_scope_key(race_data)
    score = {
        "central_dirt": 0.690,
        "central_turf": 0.620,
        "local": 0.750,
    }.get(scope_key, 0.660)

    marked_score_values = [score_item.total_score for score_item in marked_scores]
    average_marked_score = sum(marked_score_values) / len(marked_score_values)
    lowest_marked_score = min(marked_score_values)
    highest_marked_score = max(marked_score_values)
    marked_spread = highest_marked_score - lowest_marked_score

    marked_horse_nos = {score_item.horse_no for score_item in marked_scores}
    remaining_scores = [
        score_item.total_score
        for score_item in horse_scores
        if score_item.horse_no not in marked_horse_nos
    ]
    next_unmarked_score = remaining_scores[0] if remaining_scores else lowest_marked_score
    gap_5_to_6 = lowest_marked_score - next_unmarked_score

    marked_risks = [score_item.scores.risk for score_item in marked_scores]
    average_marked_risk = sum(marked_risks) / len(marked_risks)
    worst_marked_risk = min(marked_risks)

    horse_map = {horse.horse_no: horse for horse in race_data.horses}
    missing_odds_count = 0
    for score_item in marked_scores:
        horse = horse_map.get(score_item.horse_no)
        odds = horse.odds if horse and horse.odds is not None else score_item.odds
        if odds is None or odds <= 0:
            missing_odds_count += 1

    if gap_5_to_6 >= 2.0:
        score += 0.050
    elif gap_5_to_6 >= 1.0:
        score += 0.025
    elif gap_5_to_6 < 0.5:
        score -= 0.035

    if marked_spread < 3.0:
        score -= 0.080
    elif marked_spread < 6.0:
        score -= 0.030
    else:
        score += 0.055 * _normalized_span(marked_spread, 6.0, 8.0)

    score += 0.035 * _normalized_span(highest_marked_score, 35.0, 10.0)
    score += 0.020 * _normalized_span(average_marked_score, 25.0, 10.0)

    if highest_marked_score < 25.0:
        score -= 0.065
    elif highest_marked_score < 30.0:
        score -= 0.025
    elif average_marked_score < 25.0:
        score -= 0.020

    if lowest_marked_score < 15.0:
        score -= 0.025

    if average_marked_risk < -6.0:
        score -= 0.080
    elif worst_marked_risk <= -7:
        score -= 0.015
    elif average_marked_risk > -2.0:
        score += 0.015

    if missing_odds_count >= 4:
        score -= 0.040
    elif missing_odds_count >= 3:
        score -= 0.015
    elif 1 <= missing_odds_count <= 2:
        score += 0.025

    if average_marked_score > 40.0 and gap_5_to_6 < 0.5:
        score -= 0.070

    if len(marked_scores) < 5:
        score -= 0.030 * (5 - len(marked_scores))

    return round(_clamp(score, 0.46, 0.90), 3)


def _coverage_confidence_label_from_score(confidence_score: float) -> str:
    if confidence_score >= 0.85:
        return "high"
    if confidence_score >= 0.70:
        return "medium"
    return "low"


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
            confidence_score=0.0,
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
        reason_codes.append("MARKET_DATA_UNAVAILABLE")

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

    axis_strength_score = _estimate_axis_strength_score(
        race_data=race_data,
        top_score=top.total_score,
        score_gap_1_2=score_gap_1_2,
        score_gap_1_3=score_gap_1_3,
        average_top3_score=average_top3_score,
        top_risk=top_risk,
        top_odds=top_odds,
        top_popularity=top_popularity,
    )
    axis_strength = _axis_strength_label_from_score(axis_strength_score)
    marked_scores = _marked_scores_for_coverage(horse_scores, marks)
    confidence_score = _estimate_marked_top3_pair_coverage_score(
        race_data=race_data,
        horse_scores=horse_scores,
        marked_scores=marked_scores,
    )
    confidence = _coverage_confidence_label_from_score(confidence_score)

    should_skip = (
        top.total_score < 32
        or top_risk <= -7
        or _contains_key_risk(risks)
        or (close_top_group and axis_strength_score < 0.40)
    )

    if should_skip:
        reason_codes.append("SKIP_LOW_CONFIDENCE")
        return StrategyDecision(
            bet_decision="SKIP",
            confidence=confidence,
            confidence_score=confidence_score,
            participation_level="none",
            reason_codes=reason_codes,
            reason=(
                "上位の能力差が詰まっており、軸を決め切るには決定打が足りないため見送り。"
            ),
        )

    if axis_strength == "high":
        participation_level = "strong" if clear_axis else "normal"
    elif axis_strength == "medium":
        participation_level = "light" if close_top_group else "normal"
    else:
        participation_level = "light"

    top_descriptor = f"本命想定は{top.horse_name}"
    odds_descriptor = (
        f"単勝{top_odds:.1f}倍・人気{top_popularity}。"
        if top_odds is not None and top_popularity is not None
        else "市場オッズを使わず、能力・条件適性を中心に評価。"
    )
    group_descriptor = "上位拮抗のため点数は絞る。" if close_top_group else "軸馬が比較的明確。"
    lesson_descriptor = "過去lessonも参考。" if used_lessons_count else ""
    return StrategyDecision(
        bet_decision="BET",
        confidence=confidence,
        confidence_score=confidence_score,
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

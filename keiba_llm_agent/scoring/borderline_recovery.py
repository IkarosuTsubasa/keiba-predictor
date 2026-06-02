from __future__ import annotations

from typing import Any

from keiba_llm_agent.schemas.prediction import HorseScore


def _flags(analysis: object | None, field_name: str) -> set[str]:
    if analysis is None:
        return set()
    value = getattr(analysis, field_name, [])
    return set(value or [])


def _analysis_map(items: list[object]) -> dict[int, object]:
    return {getattr(item, "horse_no"): item for item in items if getattr(item, "horse_no", None) is not None}


def _ranked_horses(horse_scores: list[HorseScore]) -> list[HorseScore]:
    return sorted(horse_scores, key=lambda item: (-item.total_score, item.horse_no))


def _append_reason(target: list[str], condition: bool, reason: str, delta: int) -> int:
    if condition:
        target.append(reason)
        return delta
    return 0


def apply_top5_borderline_recovery(
    horse_scores: list[HorseScore],
    deep_analyses: list[object],
    pedigree_analyses: list[object],
    race_level_analyses: list[object],
    pace_analyses: list[object],
    race_info: object | None,
    scoring_config: object | None,
    enabled: bool = True,
) -> dict[str, Any]:
    ranked = _ranked_horses(horse_scores)
    original_top5 = [horse_score.horse_no for horse_score in ranked[:5]]
    result: dict[str, Any] = {
        "recovered_top5": original_top5,
        "recovery_applied": False,
        "recovery_cases": [],
        "adjusted_horse_scores": ranked,
    }
    if not enabled or len(ranked) < 6:
        return result

    deep_map = _analysis_map(deep_analyses)
    pedigree_map = _analysis_map(pedigree_analyses)
    race_level_map = _analysis_map(race_level_analyses)
    pace_map = _analysis_map(pace_analyses)
    top5_cutoff_score = ranked[4].total_score

    candidates: list[dict[str, Any]] = []
    for index, horse_score in enumerate(ranked, start=1):
        if index != 6:
            continue
        score_gap = round(top5_cutoff_score - horse_score.total_score, 1)
        if score_gap > 1.0:
            continue

        deep_analysis = deep_map.get(horse_score.horse_no)
        pedigree_analysis = pedigree_map.get(horse_score.horse_no)
        race_level_analysis = race_level_map.get(horse_score.horse_no)
        pace_analysis = pace_map.get(horse_score.horse_no)

        reasons: list[str] = ["JUST_BELOW_TOP5"]
        net_signal = 0

        popularity = horse_score.popularity
        odds = horse_score.odds
        deep_positive = _flags(deep_analysis, "positive_flags")
        deep_risk = _flags(deep_analysis, "risk_flags")
        pedigree_positive = _flags(pedigree_analysis, "positive_flags")
        race_level_positive = _flags(race_level_analysis, "positive_flags")
        race_level_risk = _flags(race_level_analysis, "risk_flags")
        pace_positive = _flags(pace_analysis, "positive_flags")
        positive_signal_count = 0

        delta = _append_reason(reasons, popularity is not None and popularity <= 3, "POPULAR_SAFETY_NET", 1)
        net_signal += delta
        positive_signal_count += 1 if delta > 0 else 0
        delta = _append_reason(
            reasons,
            popularity is not None and popularity <= 5 and odds is not None and odds < 10,
            "MARKET_SUPPORT",
            1,
        )
        net_signal += delta
        positive_signal_count += 1 if delta > 0 else 0
        delta = _append_reason(
            reasons,
            bool({"RECENT_FORM_STRONG", "RECENT_FORM_STABLE"} & deep_positive),
            "RECENT_FORM_SIGNAL",
            1,
        )
        net_signal += delta
        positive_signal_count += 1 if delta > 0 else 0
        delta = _append_reason(
            reasons,
            bool({"DISTANCE_FIT", "COURSE_FIT"} & deep_positive),
            "CONDITION_FIT_SIGNAL",
            1,
        )
        net_signal += delta
        positive_signal_count += 1 if delta > 0 else 0
        delta = _append_reason(
            reasons,
            bool({"PEDIGREE_DISTANCE_FIT", "PEDIGREE_STAMINA_FIT", "PEDIGREE_SURFACE_FIT"} & pedigree_positive),
            "PEDIGREE_SIGNAL",
            1,
        )
        net_signal += delta
        positive_signal_count += 1 if delta > 0 else 0
        delta = _append_reason(
            reasons,
            bool({"HEAD_TO_HEAD_POSITIVE", "LARGE_FIELD_GOOD_RUN", "UNDERVALUED_GOOD_RUN", "VALUE_WIN"} & race_level_positive),
            "RACE_LEVEL_SIGNAL",
            1,
        )
        net_signal += delta
        positive_signal_count += 1 if delta > 0 else 0
        delta = _append_reason(
            reasons,
            bool({"PACE_FIT", "STALKER_ADVANTAGE", "CLOSING_SPEED"} & pace_positive),
            "PACE_SIGNAL",
            1,
        )
        net_signal += delta
        positive_signal_count += 1 if delta > 0 else 0

        net_signal += _append_reason(reasons, len(deep_risk) >= 4, "TOO_MANY_RISKS", -1)
        net_signal += _append_reason(reasons, "HEAD_TO_HEAD_NEGATIVE" in race_level_risk, "HEAD_TO_HEAD_NEGATIVE", -1)
        net_signal += _append_reason(
            reasons,
            odds is not None and odds >= 50 and popularity is not None and popularity >= 8,
            "TOO_SPECULATIVE",
            -1,
        )
        net_signal += _append_reason(
            reasons,
            "RECENT_FORM_WEAK" in deep_risk and "RECENT_FORM_STABLE" not in deep_positive,
            "RECENT_FORM_WEAK_SIGNAL",
            -1,
        )

        strong_racelevel_signal = bool({"LARGE_FIELD_GOOD_RUN", "UNDERVALUED_GOOD_RUN", "VALUE_WIN", "EXPECTED_WIN"} & race_level_positive)
        anchor_signal = bool(
            {
                "POPULAR_SAFETY_NET",
                "MARKET_SUPPORT",
                "RECENT_FORM_SIGNAL",
                "CONDITION_FIT_SIGNAL",
                "PEDIGREE_SIGNAL",
            }
            & set(reasons)
        )
        both_distance_and_course_unknown = {"DISTANCE_UNKNOWN", "COURSE_UNKNOWN"}.issubset(deep_risk)
        if "HEAD_TO_HEAD_NEGATIVE" in race_level_risk:
            continue
        if positive_signal_count < 2:
            continue
        if not (anchor_signal or (strong_racelevel_signal and positive_signal_count >= 3)):
            continue
        if both_distance_and_course_unknown and not strong_racelevel_signal:
            continue
        if net_signal < 2:
            continue

        candidates.append(
            {
                "horse_score": horse_score,
                "original_rank": index,
                "score_gap_to_top5": score_gap,
                "recovery_bonus": round(min(score_gap + 0.1, 1.0), 1),
                "recovery_reasons": reasons,
                "net_signal": net_signal,
            }
        )

    if not candidates:
        return result

    selected = sorted(
        candidates,
        key=lambda item: (
            -item["net_signal"],
            item["score_gap_to_top5"],
            item["horse_score"].popularity if item["horse_score"].popularity is not None else 999,
        ),
    )[0]

    adjusted_scores: list[HorseScore] = []
    for horse_score in horse_scores:
        if horse_score.horse_no != selected["horse_score"].horse_no:
            adjusted_scores.append(horse_score)
            continue
        target_score = max(
            horse_score.total_score + selected["recovery_bonus"],
            top5_cutoff_score + 0.01,
        )
        bonus = round(target_score - horse_score.total_score, 2)
        updated_breakdown = horse_score.score_breakdown.model_copy(
            update={
                "borderline_recovery_bonus": bonus,
                "total_score_after_recovery": target_score,
            }
        )
        updated_recovery = horse_score.borderline_recovery.model_copy(
            update={
                "applied": True,
                "recovery_bonus": bonus,
                "original_rank": selected["original_rank"],
                "reasons": selected["recovery_reasons"],
            }
        )
        updated_reason = (
            f"{horse_score.reason} "
            f"Top5境界補正により+{bonus:.1f}。人気・相手関係・展開面から押さえ候補に浮上。"
        ).strip()
        adjusted_scores.append(
            horse_score.model_copy(
                update={
                    "total_score": target_score,
                    "score_breakdown": updated_breakdown,
                    "borderline_recovery": updated_recovery,
                    "reason": updated_reason,
                }
            )
        )

    adjusted_ranked = _ranked_horses(adjusted_scores)
    new_rank_map = {horse_score.horse_no: index + 1 for index, horse_score in enumerate(adjusted_ranked)}
    recovered_top5 = [horse_score.horse_no for horse_score in adjusted_ranked[:5]]
    selected_horse = next(horse_score for horse_score in adjusted_ranked if horse_score.horse_no == selected["horse_score"].horse_no)
    if selected_horse.horse_no in original_top5 or new_rank_map[selected_horse.horse_no] > 5 or recovered_top5 == original_top5:
        return result
    final_recovery = selected_horse.borderline_recovery.model_copy(
        update={"new_rank": new_rank_map[selected_horse.horse_no]}
    )
    final_breakdown = selected_horse.score_breakdown.model_copy(
        update={"total_score_after_recovery": selected_horse.total_score}
    )
    updated_selected = selected_horse.model_copy(
        update={
            "borderline_recovery": final_recovery,
            "score_breakdown": final_breakdown,
        }
    )
    adjusted_ranked = [
        updated_selected if horse_score.horse_no == updated_selected.horse_no else horse_score
        for horse_score in adjusted_ranked
    ]
    adjusted_ranked.sort(key=lambda item: (-item.total_score, item.horse_no))

    result["recovered_top5"] = [horse_score.horse_no for horse_score in adjusted_ranked[:5]]
    result["recovery_applied"] = True
    result["recovery_cases"] = [
        {
            "horse_no": updated_selected.horse_no,
            "horse_name": updated_selected.horse_name,
            "original_rank": selected["original_rank"],
            "score_gap_to_top5": selected["score_gap_to_top5"],
            "recovery_bonus": selected["recovery_bonus"],
            "recovery_reasons": selected["recovery_reasons"],
            "new_rank": new_rank_map[updated_selected.horse_no],
        }
    ]
    result["adjusted_horse_scores"] = adjusted_ranked
    return result

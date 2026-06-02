from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from keiba_llm_agent.backtest.scoring_comparator import (
    WEIGHT_TUNING_MODE_WEIGHTS,
    WeightTuningMode,
    result_top3_list,
)
from keiba_llm_agent.error_analysis.deep_miss_rule_simulator import (
    BASELINE_MODE_CHOICES,
    _baseline_custom_weights,
    _finish_map,
    _ordered_horse_scores,
    _resolve_baseline_mode,
    _top5_capture_count,
)
from keiba_llm_agent.error_analysis.missed_top3_analyzer import (
    _build_race_horse_map,
    _collect_predictions_in_period,
    _find_analysis,
    _load_race_data,
    _load_result,
    _load_review,
)
from keiba_llm_agent.schemas.prediction import HorseScore, Prediction
from keiba_llm_agent.schemas.result import ResultData


PENALTY_REFINEMENT_RULES: dict[str, dict[str, Any]] = {
    "head_to_head_negative_cap": {
        "description": "HEAD_TO_HEAD_NEGATIVE を強い正面シグナルがある場合だけ -0.2 まで緩和する",
        "target_floor": -0.2,
        "min_positive_stack": 4,
    },
    "head_to_head_zero_floor": {
        "description": "HEAD_TO_HEAD_NEGATIVE を強い正面シグナルがある場合だけ 0.0 まで緩和する aggressive simulation",
        "target_floor": 0.0,
        "min_positive_stack": 5,
    },
    "distance_unknown_stamina_exception": {
        "description": "長め距離で DISTANCE_UNKNOWN でも血統スタミナ・近走・コースの裏付けがあれば軽量救済する",
        "bonus": 0.5,
        "min_positive_stack": 3,
    },
    "track_condition_unknown_soften": {
        "description": "TRACK_CONDITION_UNKNOWN を強い正面シグナルがある場合だけ軽量緩和する",
        "bonus": 0.3,
        "min_positive_stack": 4,
    },
    "positive_stack_protection": {
        "description": "複数モジュールの正面シグナルが厚い馬を risk で沈めすぎないか検証する",
        "bonus": 0.6,
        "min_positive_stack": 5,
    },
    "combined_penalty_refinement": {
        "description": "上記 penalty refinement のいずれかを満たす候補に最大1.0の合成補正を試す",
        "bonus_cap": 1.0,
        "min_positive_stack": 3,
    },
}


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def _baseline_weights(
    mode: str,
    custom_weights: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    if mode == "custom":
        if custom_weights is None:
            raise ValueError("custom baseline requires custom weights")
        return custom_weights
    return WEIGHT_TUNING_MODE_WEIGHTS[mode]


def _flag_set(analysis: object | None, field_name: str) -> set[str]:
    return set(getattr(analysis, field_name, []) or [])


def _positive_signal_labels(
    *,
    deep_positive: set[str],
    pedigree_positive: set[str],
    race_level_positive: set[str],
    pace_positive: set[str],
) -> list[str]:
    labels: list[str] = []
    if {"RECENT_FORM_STRONG", "RECENT_FORM_STABLE"} & deep_positive:
        labels.append("RECENT_FORM_SIGNAL")
    if "STABLE_PERFORMER" in deep_positive:
        labels.append("STABLE_PERFORMER")
    if "DISTANCE_FIT" in deep_positive:
        labels.append("DISTANCE_FIT")
    if "COURSE_FIT" in deep_positive:
        labels.append("COURSE_FIT")
    if {"PEDIGREE_DISTANCE_FIT", "PEDIGREE_STAMINA_FIT", "PEDIGREE_SURFACE_FIT"} & pedigree_positive:
        labels.append("PEDIGREE_FIT")
    if {"HEAD_TO_HEAD_POSITIVE", "LARGE_FIELD_GOOD_RUN", "UNDERVALUED_GOOD_RUN", "VALUE_WIN", "EXPECTED_WIN"} & race_level_positive:
        labels.append("RACE_LEVEL_POSITIVE")
    if {"PACE_FIT", "STALKER_ADVANTAGE", "CLOSING_SPEED"} & pace_positive:
        labels.append("PACE_FIT")
    return labels


def _is_strong_risk(
    *,
    horse_score: HorseScore,
    deep_risk: set[str],
    pace_risk: set[str],
    predicted_rank: int,
    score_gap_to_top5: float,
) -> bool:
    return (
        horse_score.scores.risk <= -6
        or len(deep_risk) >= 4
        or len(pace_risk) >= 3
        or predicted_rank >= 13
        or score_gap_to_top5 > 6.0
        or {"RECENT_FORM_WEAK", "RECENT_FORM_DECLINING"}.issubset(deep_risk)
    )


def _build_candidate_pool(
    prediction: Prediction,
    result_data: ResultData,
    ordered_scores: list[HorseScore],
    race_horse_map: dict[int, object],
    *,
    max_rank: int,
    include_rank13: bool,
) -> list[dict[str, Any]]:
    finish_map = _finish_map(result_data)
    rank_map = {horse_score.horse_no: index + 1 for index, horse_score in enumerate(ordered_scores)}
    top5_cutoff_score = ordered_scores[4].total_score if len(ordered_scores) >= 5 else (ordered_scores[-1].total_score if ordered_scores else 0.0)
    candidates: list[dict[str, Any]] = []

    for horse_score in ordered_scores:
        predicted_rank = rank_map[horse_score.horse_no]
        if predicted_rank < 7:
            continue
        if not include_rank13 and predicted_rank >= 13:
            continue
        if predicted_rank > max_rank:
            continue

        deep_analysis = _find_analysis(prediction, "deep_analyses", horse_score.horse_no)
        pedigree_analysis = _find_analysis(prediction, "pedigree_analyses", horse_score.horse_no)
        race_level_analysis = _find_analysis(prediction, "race_level_analyses", horse_score.horse_no)
        pace_analysis = _find_analysis(prediction, "pace_analyses", horse_score.horse_no)
        deep_positive = _flag_set(deep_analysis, "positive_flags")
        deep_risk = _flag_set(deep_analysis, "risk_flags")
        pedigree_positive = _flag_set(pedigree_analysis, "positive_flags")
        pedigree_risk = _flag_set(pedigree_analysis, "risk_flags")
        race_level_positive = _flag_set(race_level_analysis, "positive_flags")
        race_level_risk = _flag_set(race_level_analysis, "risk_flags")
        pace_positive = _flag_set(pace_analysis, "positive_flags")
        pace_risk = _flag_set(pace_analysis, "risk_flags")
        signal_labels = _positive_signal_labels(
            deep_positive=deep_positive,
            pedigree_positive=pedigree_positive,
            race_level_positive=race_level_positive,
            pace_positive=pace_positive,
        )
        horse_entry = race_horse_map.get(horse_score.horse_no)
        score_gap = round(top5_cutoff_score - horse_score.total_score, 1)
        candidates.append(
            {
                "horse_no": horse_score.horse_no,
                "horse_name": horse_score.horse_name,
                "predicted_rank": predicted_rank,
                "baseline_score": horse_score.total_score,
                "top5_cutoff_score": top5_cutoff_score,
                "score_gap_to_top5": score_gap,
                "finish": finish_map.get(horse_score.horse_no),
                "odds": horse_score.odds if horse_score.odds is not None else getattr(horse_entry, "odds", None),
                "popularity": horse_score.popularity if horse_score.popularity is not None else getattr(horse_entry, "popularity", None),
                "deep_positive": sorted(deep_positive),
                "deep_risk": sorted(deep_risk),
                "pedigree_positive": sorted(pedigree_positive),
                "pedigree_risk": sorted(pedigree_risk),
                "race_level_positive": sorted(race_level_positive),
                "race_level_risk": sorted(race_level_risk),
                "pace_positive": sorted(pace_positive),
                "pace_risk": sorted(pace_risk),
                "positive_signal_labels": signal_labels,
                "positive_stack_count": len(signal_labels),
                "race_level_adjustment_raw": horse_score.race_level_adjustment.adjustment,
                "strong_risk": _is_strong_risk(
                    horse_score=horse_score,
                    deep_risk=deep_risk,
                    pace_risk=pace_risk,
                    predicted_rank=predicted_rank,
                    score_gap_to_top5=score_gap,
                ),
            }
        )
    return candidates


def _head_to_head_bonus(
    candidate: dict[str, Any],
    *,
    race_level_weight: float,
    target_floor: float,
    min_positive_stack: int,
) -> tuple[float, list[str]]:
    if candidate["strong_risk"]:
        return 0.0, []
    if "HEAD_TO_HEAD_NEGATIVE" not in candidate["race_level_risk"]:
        return 0.0, []
    if candidate["positive_stack_count"] < min_positive_stack:
        return 0.0, []
    raw_adjustment = float(candidate["race_level_adjustment_raw"])
    if raw_adjustment >= target_floor or race_level_weight <= 0:
        return 0.0, []
    bonus = round((target_floor - raw_adjustment) * race_level_weight, 2)
    return bonus, [f"HEAD_TO_HEAD_NEGATIVE_CAP_TO_{target_floor:+.1f}"]


def _distance_unknown_bonus(
    candidate: dict[str, Any],
    *,
    race_distance: int | None,
    min_positive_stack: int,
    bonus: float,
) -> tuple[float, list[str]]:
    if candidate["strong_risk"]:
        return 0.0, []
    if "DISTANCE_UNKNOWN" not in candidate["deep_risk"]:
        return 0.0, []
    if race_distance is None or race_distance < 2000:
        return 0.0, []
    if candidate["positive_stack_count"] < min_positive_stack:
        return 0.0, []
    pedigree_positive = set(candidate["pedigree_positive"])
    deep_positive = set(candidate["deep_positive"])
    has_pedigree_stamina = bool({"PEDIGREE_DISTANCE_FIT", "PEDIGREE_STAMINA_FIT"} & pedigree_positive)
    has_recent_support = bool({"RECENT_FORM_STRONG", "RECENT_FORM_STABLE"} & deep_positive)
    has_condition_support = "COURSE_FIT" in deep_positive or "RACE_LEVEL_POSITIVE" in candidate["positive_signal_labels"]
    if not (has_pedigree_stamina and has_recent_support and has_condition_support):
        return 0.0, []
    return bonus, ["DISTANCE_UNKNOWN_STAMINA_EXCEPTION"]


def _track_condition_unknown_bonus(
    candidate: dict[str, Any],
    *,
    min_positive_stack: int,
    bonus: float,
) -> tuple[float, list[str]]:
    if candidate["strong_risk"]:
        return 0.0, []
    if "TRACK_CONDITION_UNKNOWN" not in candidate["deep_risk"]:
        return 0.0, []
    if candidate["positive_stack_count"] < min_positive_stack:
        return 0.0, []
    if "RECENT_FORM_WEAK" in candidate["deep_risk"]:
        return 0.0, []
    return bonus, ["TRACK_CONDITION_UNKNOWN_SOFTEN"]


def _positive_stack_bonus(
    candidate: dict[str, Any],
    *,
    min_positive_stack: int,
    bonus: float,
) -> tuple[float, list[str]]:
    if candidate["strong_risk"]:
        return 0.0, []
    if candidate["positive_stack_count"] < min_positive_stack:
        return 0.0, []
    if candidate["score_gap_to_top5"] > 4.0:
        return 0.0, []
    if not ({"HEAD_TO_HEAD_NEGATIVE"} & set(candidate["race_level_risk"]) or {"DISTANCE_UNKNOWN", "TRACK_CONDITION_UNKNOWN"} & set(candidate["deep_risk"])):
        return 0.0, []
    return bonus, ["POSITIVE_STACK_PROTECTION"]


def _rule_bonus(
    candidate: dict[str, Any],
    rule_name: str,
    *,
    race_distance: int | None,
    race_level_weight: float,
) -> tuple[float, list[str]]:
    rule = PENALTY_REFINEMENT_RULES[rule_name]
    if rule_name == "head_to_head_negative_cap":
        return _head_to_head_bonus(
            candidate,
            race_level_weight=race_level_weight,
            target_floor=float(rule["target_floor"]),
            min_positive_stack=int(rule["min_positive_stack"]),
        )
    if rule_name == "head_to_head_zero_floor":
        return _head_to_head_bonus(
            candidate,
            race_level_weight=race_level_weight,
            target_floor=float(rule["target_floor"]),
            min_positive_stack=int(rule["min_positive_stack"]),
        )
    if rule_name == "distance_unknown_stamina_exception":
        return _distance_unknown_bonus(
            candidate,
            race_distance=race_distance,
            min_positive_stack=int(rule["min_positive_stack"]),
            bonus=float(rule["bonus"]),
        )
    if rule_name == "track_condition_unknown_soften":
        return _track_condition_unknown_bonus(
            candidate,
            min_positive_stack=int(rule["min_positive_stack"]),
            bonus=float(rule["bonus"]),
        )
    if rule_name == "positive_stack_protection":
        return _positive_stack_bonus(
            candidate,
            min_positive_stack=int(rule["min_positive_stack"]),
            bonus=float(rule["bonus"]),
        )
    if rule_name == "combined_penalty_refinement":
        bonuses: list[float] = []
        reasons: list[str] = []
        for child_rule in (
            "head_to_head_negative_cap",
            "distance_unknown_stamina_exception",
            "track_condition_unknown_soften",
            "positive_stack_protection",
        ):
            child_bonus, child_reasons = _rule_bonus(
                candidate,
                child_rule,
                race_distance=race_distance,
                race_level_weight=race_level_weight,
            )
            if child_bonus > 0:
                bonuses.append(child_bonus)
                reasons.extend(child_reasons)
        if not bonuses:
            return 0.0, []
        return round(min(float(rule["bonus_cap"]), sum(bonuses)), 2), list(dict.fromkeys(reasons))
    return 0.0, []


def _pick_candidate(
    candidates: list[dict[str, Any]],
    rule_name: str,
    *,
    race_distance: int | None,
    race_level_weight: float,
) -> dict[str, Any] | None:
    matched: list[dict[str, Any]] = []
    for candidate in candidates:
        bonus, reasons = _rule_bonus(
            candidate,
            rule_name,
            race_distance=race_distance,
            race_level_weight=race_level_weight,
        )
        if bonus <= 0:
            continue
        simulated_score = round(candidate["baseline_score"] + bonus, 2)
        matched.append(
            {
                **candidate,
                "simulation_bonus": bonus,
                "simulation_reasons": reasons,
                "simulated_score": simulated_score,
                "would_cross_cutoff": simulated_score > candidate["top5_cutoff_score"],
            }
        )
    if not matched:
        return None
    matched.sort(
        key=lambda item: (
            not item["would_cross_cutoff"],
            -item["simulated_score"],
            item["predicted_rank"],
            -item["positive_stack_count"],
            item["score_gap_to_top5"],
            item["horse_no"],
        )
    )
    return matched[0]


def _simulate_score_adjustment(
    baseline_top5: list[int],
    result_top3: list[int],
    candidate: dict[str, Any],
    ordered_scores: list[HorseScore],
) -> dict[str, Any]:
    adjusted_rows: list[tuple[int, float]] = []
    for horse_score in ordered_scores:
        score = horse_score.total_score
        if horse_score.horse_no == candidate["horse_no"]:
            score = round(score + candidate["simulation_bonus"], 2)
        adjusted_rows.append((horse_score.horse_no, score))
    adjusted_rows.sort(key=lambda item: (-item[1], item[0]))
    simulated_top5 = [horse_no for horse_no, _ in adjusted_rows[:5]]
    rank_map = {horse_no: index + 1 for index, (horse_no, _) in enumerate(adjusted_rows)}
    candidate_enters_top5 = candidate["horse_no"] in simulated_top5
    replaced = [horse_no for horse_no in baseline_top5 if horse_no not in simulated_top5]
    replaced_horse_no = replaced[0] if candidate_enters_top5 and replaced else None
    baseline_count = _top5_capture_count(baseline_top5, result_top3)
    simulated_count = _top5_capture_count(simulated_top5, result_top3)
    delta = simulated_count - baseline_count
    return {
        "simulated_top5": simulated_top5,
        "candidate_enters_top5": candidate_enters_top5,
        "candidate_new_rank": rank_map.get(candidate["horse_no"]),
        "replaced_horse_no": replaced_horse_no,
        "delta": delta,
        "candidate_in_top3": candidate["horse_no"] in result_top3,
        "replaced_in_top3": replaced_horse_no in result_top3 if replaced_horse_no is not None else False,
        "better": delta > 0,
        "worse": delta < 0,
        "simulated_capture_count": simulated_count,
    }


def _best_rule_key(rule_result: dict[str, Any]) -> tuple[float, float, int, int, int]:
    return (
        rule_result["safe_score"],
        rule_result["improvement"],
        -rule_result["worse_race_count"],
        -rule_result["replaced_top3_count"],
        rule_result["effective_candidate_count"],
    )


def run_penalty_refinement_simulation(
    *,
    from_date: str,
    to_date: str,
    predictions_dir: Path,
    results_dir: Path,
    reviews_dir: Path,
    race_data_dir: Path | None = None,
    baseline_mode: str | None = None,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    max_rank: int = 12,
    include_rank13: bool = False,
) -> dict[str, object]:
    resolved_baseline_mode = _resolve_baseline_mode(baseline_mode, scoring_mode)
    baseline_custom_weights = _baseline_custom_weights(
        resolved_baseline_mode,
        pedigree_weight,
        race_level_weight,
        pace_weight,
    )
    baseline_weights = _baseline_weights(resolved_baseline_mode, baseline_custom_weights)
    race_level_weight_value = baseline_weights[1]
    warnings: list[str] = []
    predictions, prediction_warnings = _collect_predictions_in_period(predictions_dir, from_date, to_date)
    warnings.extend(prediction_warnings)

    race_count = len(predictions)
    reviewed_race_count = 0
    total_top3_horses = 0
    baseline_captured_top3_horses = 0
    baseline_top5_winner_count = 0
    rule_state: dict[str, dict[str, Any]] = {}
    race_details: list[dict[str, Any]] = []
    signal_counter: Counter[str] = Counter()
    for rule_name in PENALTY_REFINEMENT_RULES:
        rule_state[rule_name] = {
            "qualified_candidate_count": 0,
            "effective_candidate_count": 0,
            "recovered_top3_count": 0,
            "false_recovery_count": 0,
            "better_race_count": 0,
            "worse_race_count": 0,
            "same_race_count": 0,
            "replaced_top3_count": 0,
            "simulated_captured_top3_horses": 0,
            "simulated_top5_winner_count": 0,
            "selected_ranks": [],
            "effective_bonuses": [],
            "rows": [],
        }

    for prediction in predictions:
        result_data = _load_result(results_dir / f"{prediction.race_id}.json")
        review = _load_review(reviews_dir / f"{prediction.race_id}.json")
        if result_data is None:
            warnings.append(f"result missing for race_id={prediction.race_id}")
            continue
        if review is None:
            warnings.append(f"review missing for race_id={prediction.race_id}")
            continue

        reviewed_race_count += 1
        result_top3 = result_top3_list(result_data)
        total_top3_horses += len(result_top3)
        race_data = _load_race_data((race_data_dir or Path()) / f"{prediction.race_id}.json") if race_data_dir is not None else None
        race_horse_map = _build_race_horse_map(race_data)
        ordered_scores, baseline_recovery = _ordered_horse_scores(
            prediction,
            resolved_baseline_mode,
            baseline_custom_weights,
        )
        baseline_top5 = [horse_score.horse_no for horse_score in ordered_scores[:5]]
        baseline_capture_count = _top5_capture_count(baseline_top5, result_top3)
        baseline_captured_top3_horses += baseline_capture_count
        winner = result_data.result.first
        if winner in baseline_top5:
            baseline_top5_winner_count += 1

        candidate_pool = _build_candidate_pool(
            prediction,
            result_data,
            ordered_scores,
            race_horse_map,
            max_rank=max_rank,
            include_rank13=include_rank13,
        )
        for candidate in candidate_pool:
            signal_counter.update(candidate["positive_signal_labels"])

        for rule_name in PENALTY_REFINEMENT_RULES:
            state = rule_state[rule_name]
            candidate = _pick_candidate(
                candidate_pool,
                rule_name,
                race_distance=prediction.race_info.distance if prediction.race_info else None,
                race_level_weight=race_level_weight_value,
            )
            if candidate is None:
                state["same_race_count"] += 1
                state["simulated_captured_top3_horses"] += baseline_capture_count
                if winner in baseline_top5:
                    state["simulated_top5_winner_count"] += 1
                continue

            simulation = _simulate_score_adjustment(
                baseline_top5,
                result_top3,
                candidate,
                ordered_scores,
            )
            simulated_top5 = simulation["simulated_top5"]
            state["qualified_candidate_count"] += 1
            state["selected_ranks"].append(candidate["predicted_rank"])
            state["simulated_captured_top3_horses"] += simulation["simulated_capture_count"]
            if winner in simulated_top5:
                state["simulated_top5_winner_count"] += 1
            if simulation["candidate_enters_top5"]:
                state["effective_candidate_count"] += 1
                state["effective_bonuses"].append(candidate["simulation_bonus"])
                if simulation["candidate_in_top3"]:
                    state["recovered_top3_count"] += 1
                else:
                    state["false_recovery_count"] += 1
                if simulation["replaced_in_top3"]:
                    state["replaced_top3_count"] += 1
            if simulation["better"]:
                state["better_race_count"] += 1
            elif simulation["worse"]:
                state["worse_race_count"] += 1
            else:
                state["same_race_count"] += 1

            row = {
                "race_id": prediction.race_id,
                "race_name": prediction.race_info.race_name if prediction.race_info and prediction.race_info.race_name else prediction.race_id,
                "baseline_top5": baseline_top5,
                "result_top3": result_top3,
                "baseline_capture_count": baseline_capture_count,
                "rule": rule_name,
                "candidate_horse_no": candidate["horse_no"],
                "candidate_horse_name": candidate["horse_name"],
                "predicted_rank": candidate["predicted_rank"],
                "finish": candidate["finish"],
                "baseline_score": candidate["baseline_score"],
                "simulation_bonus": candidate["simulation_bonus"],
                "simulated_score": candidate["simulated_score"],
                "top5_cutoff_score": candidate["top5_cutoff_score"],
                "score_gap_to_top5": candidate["score_gap_to_top5"],
                "positive_stack_count": candidate["positive_stack_count"],
                "positive_signal_labels": candidate["positive_signal_labels"],
                "simulation_reasons": candidate["simulation_reasons"],
                "candidate_enters_top5": simulation["candidate_enters_top5"],
                "candidate_new_rank": simulation["candidate_new_rank"],
                "simulated_top5": simulated_top5,
                "replaced_horse_no": simulation["replaced_horse_no"],
                "delta": simulation["delta"],
                "baseline_recovery_applied": bool(baseline_recovery.get("recovery_applied", False)),
            }
            state["rows"].append(row)
            race_details.append(row)

    baseline_avg = round(baseline_captured_top3_horses / reviewed_race_count, 4) if reviewed_race_count else 0.0
    baseline_capture_rate = round(baseline_captured_top3_horses / total_top3_horses, 4) if total_top3_horses else 0.0
    baseline_top5_winner_rate = round(baseline_top5_winner_count / reviewed_race_count, 4) if reviewed_race_count else 0.0

    rule_results: dict[str, dict[str, Any]] = {}
    for rule_name, state in rule_state.items():
        simulated_avg = round(state["simulated_captured_top3_horses"] / reviewed_race_count, 4) if reviewed_race_count else 0.0
        simulated_capture_rate = round(state["simulated_captured_top3_horses"] / total_top3_horses, 4) if total_top3_horses else 0.0
        improvement = round(simulated_avg - baseline_avg, 4)
        safe_score = (
            state["recovered_top3_count"]
            - state["replaced_top3_count"]
            - state["worse_race_count"]
        )
        rule_results[rule_name] = {
            "rule": rule_name,
            "description": PENALTY_REFINEMENT_RULES[rule_name]["description"],
            "race_count": race_count,
            "reviewed_race_count": reviewed_race_count,
            "baseline_avg_captured_top3_per_race": baseline_avg,
            "simulated_avg_captured_top3_per_race": simulated_avg,
            "improvement": improvement,
            "baseline_capture_rate": baseline_capture_rate,
            "simulated_capture_rate": simulated_capture_rate,
            "baseline_top5_winner_rate": baseline_top5_winner_rate,
            "simulated_top5_winner_rate": round(state["simulated_top5_winner_count"] / reviewed_race_count, 4) if reviewed_race_count else 0.0,
            "qualified_candidate_count": state["qualified_candidate_count"],
            "effective_candidate_count": state["effective_candidate_count"],
            "recovered_top3_count": state["recovered_top3_count"],
            "false_recovery_count": state["false_recovery_count"],
            "better_race_count": state["better_race_count"],
            "worse_race_count": state["worse_race_count"],
            "same_race_count": state["same_race_count"],
            "net_better_minus_worse": state["better_race_count"] - state["worse_race_count"],
            "replaced_top3_count": state["replaced_top3_count"],
            "safe_score": safe_score,
            "avg_selected_rank": _safe_mean([float(value) for value in state["selected_ranks"]]),
            "avg_effective_bonus": _safe_mean([float(value) for value in state["effective_bonuses"]]),
            "rows": state["rows"],
        }

    sorted_rules = sorted(rule_results.values(), key=_best_rule_key, reverse=True)
    recommended = [
        result["rule"]
        for result in sorted_rules
        if result["improvement"] >= 0.05
        and result["better_race_count"] > result["worse_race_count"]
        and result["replaced_top3_count"] <= result["recovered_top3_count"]
    ]
    not_recommended = [
        result["rule"]
        for result in rule_results.values()
        if result["improvement"] < 0.03
        or result["worse_race_count"] >= result["better_race_count"]
        or result["replaced_top3_count"] > result["recovered_top3_count"]
    ]

    report = {
        "period": {"from": from_date, "to": to_date},
        "analysis_config": {
            "baseline_mode": resolved_baseline_mode,
            "scoring_mode": scoring_mode,
            "pedigree_weight": pedigree_weight,
            "race_level_weight": race_level_weight,
            "pace_weight": pace_weight,
            "max_rank": max_rank,
            "include_rank13": include_rank13,
            "note": "what-if simulation only; prediction/review/result files are not modified",
        },
        "summary": {
            "race_count": race_count,
            "reviewed_race_count": reviewed_race_count,
            "baseline_avg_captured_top3_per_race": baseline_avg,
            "baseline_capture_rate": baseline_capture_rate,
            "baseline_top5_winner_rate": baseline_top5_winner_rate,
        },
        "signal_counts": dict(sorted(signal_counter.items(), key=lambda item: (-item[1], item[0]))),
        "rule_results": rule_results,
        "best_rule_candidates": {
            "best_rule": sorted_rules[0]["rule"] if sorted_rules else None,
            "recommended_rules": recommended,
            "not_recommended_rules": sorted(set(not_recommended)),
        },
        "race_details": race_details,
        "warnings": warnings,
    }
    return report


def generate_penalty_refinement_simulation_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    config = report["analysis_config"]
    rule_results = report["rule_results"]
    best = report["best_rule_candidates"]
    lines = [
        "# Penalty Refinement Simulation Report",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}",
        f"- reviewed races: {summary['reviewed_race_count']}",
        f"- baseline mode: {config['baseline_mode']}",
        f"- max rank: {config['max_rank']}",
        f"- baseline avg captured top3 per race: {summary['baseline_avg_captured_top3_per_race']:.2f}",
        "- 注意: what-if simulation のみ。prediction / review / result は変更しない。",
        "",
        "## Rule Comparison",
        "| rule | simulated avg | improvement | qualified | effective | recovered top3 | better | worse | replaced top3 | safe_score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rule_name, result in sorted(rule_results.items()):
        lines.append(
            f"| {rule_name} | {result['simulated_avg_captured_top3_per_race']:.2f} | {result['improvement']:.2f} | "
            f"{result['qualified_candidate_count']} | {result['effective_candidate_count']} | {result['recovered_top3_count']} | "
            f"{result['better_race_count']} | {result['worse_race_count']} | {result['replaced_top3_count']} | {result['safe_score']} |"
        )

    lines.extend(
        [
            "",
            "## Best Rule Candidates",
            f"- best rule: {best['best_rule'] or 'なし'}",
            f"- recommended rules: {', '.join(best['recommended_rules']) if best['recommended_rules'] else 'なし'}",
            f"- not recommended rules: {', '.join(best['not_recommended_rules']) if best['not_recommended_rules'] else 'なし'}",
            "",
            "## Signal Counts",
            "| signal | count |",
            "| --- | ---: |",
        ]
    )
    if report["signal_counts"]:
        for signal, count in report["signal_counts"].items():
            lines.append(f"| {signal} | {count} |")
    else:
        lines.append("| なし | 0 |")

    lines.extend(["", "## Rule Details"])
    for rule_name, result in sorted(rule_results.items()):
        success_case = next((row for row in result["rows"] if row["delta"] > 0), None)
        failure_case = next((row for row in result["rows"] if row["delta"] < 0), None)
        lines.extend(
            [
                f"### {rule_name}",
                f"- 条件: {result['description']}",
                f"- qualified candidates: {result['qualified_candidate_count']}",
                f"- effective candidates: {result['effective_candidate_count']}",
                f"- recovered_top3_count: {result['recovered_top3_count']}",
                f"- false_recovery_count: {result['false_recovery_count']}",
                f"- replaced_top3_count: {result['replaced_top3_count']}",
                f"- safe_score: {result['safe_score']}",
                f"- 典型成功case: {success_case['race_id']} / {success_case['candidate_horse_no']} / delta={success_case['delta']}" if success_case else "- 典型成功case: なし",
                f"- 典型失败case: {failure_case['race_id']} / {failure_case['candidate_horse_no']} / delta={failure_case['delta']}" if failure_case else "- 典型失败case: なし",
                "",
            ]
        )

    lines.extend(
        [
            "## Race Details",
            "| race_id | baseline_top5 | result_top3 | rule | candidate | rank | finish | bonus | new_rank | replaced | delta | reasons |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    if report["race_details"]:
        for row in report["race_details"]:
            lines.append(
                f"| {row['race_id']} | {'→'.join(str(item) for item in row['baseline_top5'])} | "
                f"{'→'.join(str(item) for item in row['result_top3'])} | {row['rule']} | "
                f"{row['candidate_horse_no']} {row['candidate_horse_name']} | {row['predicted_rank']} | "
                f"{row['finish'] if row['finish'] is not None else '-'} | {row['simulation_bonus']:.2f} | "
                f"{row['candidate_new_rank'] if row['candidate_new_rank'] is not None else '-'} | "
                f"{row['replaced_horse_no'] or '-'} | {row['delta']} | {', '.join(row['simulation_reasons'])} |"
            )
    else:
        lines.append("| なし | - | - | - | - | - | - | - | - | - | - | - |")

    findings: list[str] = []
    if any(result["improvement"] >= 0.05 and result["worse_race_count"] <= 2 for result in rule_results.values()):
        findings.append("一部の penalty refinement は次段階で正式候補にできる可能性がある。")
    if rule_results.get("head_to_head_negative_cap", {}).get("improvement", 0.0) > 0:
        findings.append("HEAD_TO_HEAD_NEGATIVE は強い正面シグナルが重なる場合、現行ペナルティがやや強い可能性。")
    if rule_results.get("distance_unknown_stamina_exception", {}).get("improvement", 0.0) > 0:
        findings.append("長距離の DISTANCE_UNKNOWN は、血統スタミナと近走安定がある場合に過小評価の可能性。")
    if any(result["replaced_top3_count"] > 0 for result in rule_results.values()):
        findings.append("replaced_top3_count が出ており、補正を正式導入する場合は rank/bonus 上限を厳しくする必要。")
    if all(result["improvement"] < 0.05 for result in rule_results.values()):
        findings.append("現時点では improvement が限定的で、default scoring へ直接入れる根拠は弱い。")
    lines.extend(["", "## Findings"])
    lines.extend(f"- {finding}" for finding in findings) if findings else lines.append("- 明確な改善ルールはまだ限定的。")

    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def save_penalty_refinement_simulation_json(report: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_penalty_refinement_simulation_markdown(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

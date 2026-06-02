from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from keiba_llm_agent.backtest.scoring_comparator import (
    WEIGHT_TUNING_MODE_WEIGHTS,
    result_top3_list,
)
from keiba_llm_agent.error_analysis.deep_miss_rule_simulator import (
    _baseline_custom_weights,
    _ordered_horse_scores,
    _resolve_baseline_mode,
    _top5_capture_count,
)
from keiba_llm_agent.error_analysis.missed_top3_analyzer import (
    _collect_predictions_in_period,
    _find_analysis,
    _load_result,
    _load_review,
)
from keiba_llm_agent.schemas.prediction import HorseScore, Prediction


RECALIBRATION_RULES: dict[str, dict[str, Any]] = {
    "unknown_softening": {
        "description": "未知リスクを明確なマイナスと同等に扱いすぎないか検証する",
        "bonus_per_unknown": 0.3,
        "bonus_cap": 0.8,
        "min_positive_signals": 3,
    },
    "positive_stack_boost": {
        "description": "複数モジュールで正面シグナルが重なる馬を軽く押し上げる",
        "bonus": 0.7,
        "strong_bonus": 1.0,
        "min_positive_signals": 4,
        "strong_positive_signals": 5,
    },
    "risk_cap_positive_stack": {
        "description": "強い正面シグナルがある馬をriskで沈めすぎないか検証する",
        "bonus": 0.8,
        "min_positive_signals": 3,
    },
    "conditional_pace_support": {
        "description": "pace単独ではなく、近走またはrace_levelと噛み合う時だけ小幅反映する",
        "bonus": 0.4,
    },
    "race_level_recent_synergy": {
        "description": "race_level positive と近走安定が同時にある馬を小幅補正する",
        "bonus": 0.5,
    },
    "combined_recalibration": {
        "description": "上記の校正候補を合成し、1頭あたり最大1.5までに制限する",
        "bonus_cap": 1.5,
    },
}

UNKNOWN_RISK_FLAGS = {"DISTANCE_UNKNOWN", "COURSE_UNKNOWN", "TRACK_CONDITION_UNKNOWN"}
RECENT_POSITIVE_FLAGS = {"RECENT_FORM_STRONG", "RECENT_FORM_STABLE"}
CONDITION_POSITIVE_FLAGS = {"DISTANCE_FIT", "COURSE_FIT", "TRACK_CONDITION_FIT"}
JOCKEY_POSITIVE_FLAGS = {"JOCKEY_CONTINUITY"}
PEDIGREE_POSITIVE_FLAGS = {
    "PEDIGREE_DISTANCE_FIT",
    "PEDIGREE_STAMINA_FIT",
    "PEDIGREE_SURFACE_FIT",
}
RACE_LEVEL_POSITIVE_FLAGS = {
    "HEAD_TO_HEAD_POSITIVE",
    "LARGE_FIELD_GOOD_RUN",
    "UNDERVALUED_GOOD_RUN",
    "VALUE_WIN",
    "EXPECTED_WIN",
}
PACE_POSITIVE_FLAGS = {"PACE_FIT", "STALKER_ADVANTAGE", "CLOSING_SPEED", "POSITION_STABLE"}
HARD_NEGATIVE_FLAGS = {"DATA_INCOMPLETE", "RECENT_FORM_WEAK"}


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _flag_set(analysis: object | None, field_name: str) -> set[str]:
    return set(getattr(analysis, field_name, []) or [])


def _baseline_weights(mode: str, custom_weights: tuple[float, float, float] | None) -> tuple[float, float, float]:
    if mode == "custom":
        if custom_weights is None:
            raise ValueError("custom baseline requires custom weights")
        return custom_weights
    return WEIGHT_TUNING_MODE_WEIGHTS[mode]


def _positive_signal_labels(
    *,
    deep_positive: set[str],
    pedigree_positive: set[str],
    race_level_positive: set[str],
    pace_positive: set[str],
) -> list[str]:
    labels: list[str] = []
    if RECENT_POSITIVE_FLAGS & deep_positive:
        labels.append("RECENT_FORM_SIGNAL")
    if "STABLE_PERFORMER" in deep_positive:
        labels.append("STABLE_PERFORMER")
    if "DISTANCE_FIT" in deep_positive:
        labels.append("DISTANCE_FIT")
    if "COURSE_FIT" in deep_positive:
        labels.append("COURSE_FIT")
    if "TRACK_CONDITION_FIT" in deep_positive:
        labels.append("TRACK_CONDITION_FIT")
    if JOCKEY_POSITIVE_FLAGS & deep_positive:
        labels.append("JOCKEY_CONTINUITY")
    if PEDIGREE_POSITIVE_FLAGS & pedigree_positive:
        labels.append("PEDIGREE_FIT")
    if RACE_LEVEL_POSITIVE_FLAGS & race_level_positive:
        labels.append("RACE_LEVEL_POSITIVE")
    if PACE_POSITIVE_FLAGS & pace_positive:
        labels.append("PACE_FIT")
    return labels


def _horse_context(prediction: Prediction, horse_score: HorseScore, rank: int, top5_cutoff: float) -> dict[str, Any]:
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
    positive_labels = _positive_signal_labels(
        deep_positive=deep_positive,
        pedigree_positive=pedigree_positive,
        race_level_positive=race_level_positive,
        pace_positive=pace_positive,
    )
    all_risk_flags = deep_risk | pedigree_risk | race_level_risk | pace_risk
    hard_negative = (
        "DATA_INCOMPLETE" in all_risk_flags
        or horse_score.scores.risk <= -7
        or ({"RECENT_FORM_WEAK", "RECENT_FORM_DECLINING"} <= all_risk_flags and "RECENT_FORM_STABLE" not in deep_positive)
    )
    return {
        "horse_no": horse_score.horse_no,
        "horse_name": horse_score.horse_name,
        "rank": rank,
        "baseline_score": horse_score.total_score,
        "top5_cutoff_score": top5_cutoff,
        "score_gap_to_top5": round(top5_cutoff - horse_score.total_score, 1),
        "deep_positive": deep_positive,
        "deep_risk": deep_risk,
        "pedigree_positive": pedigree_positive,
        "pedigree_risk": pedigree_risk,
        "race_level_positive": race_level_positive,
        "race_level_risk": race_level_risk,
        "pace_positive": pace_positive,
        "pace_risk": pace_risk,
        "all_risk_flags": all_risk_flags,
        "positive_signal_labels": positive_labels,
        "positive_signal_count": len(positive_labels),
        "hard_negative": hard_negative,
        "risk_score": horse_score.scores.risk,
    }


def _unknown_softening_bonus(context: dict[str, Any]) -> tuple[float, list[str]]:
    rule = RECALIBRATION_RULES["unknown_softening"]
    unknown_flags = sorted(UNKNOWN_RISK_FLAGS & context["deep_risk"])
    if not unknown_flags or context["hard_negative"]:
        return 0.0, []
    if context["positive_signal_count"] < int(rule["min_positive_signals"]):
        return 0.0, []
    bonus = min(float(rule["bonus_cap"]), float(rule["bonus_per_unknown"]) * len(unknown_flags))
    reasons = [f"UNKNOWN_SOFTEN:{flag}" for flag in unknown_flags]
    return round(bonus, 2), reasons


def _positive_stack_bonus(context: dict[str, Any]) -> tuple[float, list[str]]:
    rule = RECALIBRATION_RULES["positive_stack_boost"]
    if context["hard_negative"]:
        return 0.0, []
    signal_count = context["positive_signal_count"]
    if signal_count < int(rule["min_positive_signals"]):
        return 0.0, []
    bonus = float(rule["strong_bonus"] if signal_count >= int(rule["strong_positive_signals"]) else rule["bonus"])
    return round(bonus, 2), ["POSITIVE_STACK_BOOST"]


def _risk_cap_bonus(context: dict[str, Any]) -> tuple[float, list[str]]:
    rule = RECALIBRATION_RULES["risk_cap_positive_stack"]
    if "DATA_INCOMPLETE" in context["all_risk_flags"]:
        return 0.0, []
    strong_risk = (
        context["risk_score"] <= -4
        or len(context["deep_risk"]) >= 3
        or len(context["pace_risk"]) >= 2
    )
    if not strong_risk:
        return 0.0, []
    if context["positive_signal_count"] < int(rule["min_positive_signals"]):
        return 0.0, []
    return float(rule["bonus"]), ["RISK_CAP_POSITIVE_STACK"]


def _conditional_pace_bonus(context: dict[str, Any]) -> tuple[float, list[str]]:
    rule = RECALIBRATION_RULES["conditional_pace_support"]
    if context["hard_negative"] or "PACE_MISMATCH" in context["pace_risk"]:
        return 0.0, []
    if not (PACE_POSITIVE_FLAGS & context["pace_positive"]):
        return 0.0, []
    has_core_support = bool(RECENT_POSITIVE_FLAGS & context["deep_positive"]) or bool(RACE_LEVEL_POSITIVE_FLAGS & context["race_level_positive"])
    if not has_core_support:
        return 0.0, []
    return float(rule["bonus"]), ["CONDITIONAL_PACE_SUPPORT"]


def _race_level_recent_synergy_bonus(context: dict[str, Any]) -> tuple[float, list[str]]:
    rule = RECALIBRATION_RULES["race_level_recent_synergy"]
    if context["hard_negative"]:
        return 0.0, []
    has_race_level = bool(RACE_LEVEL_POSITIVE_FLAGS & context["race_level_positive"])
    has_recent = bool(RECENT_POSITIVE_FLAGS & context["deep_positive"])
    has_condition_support = bool(CONDITION_POSITIVE_FLAGS & context["deep_positive"]) or bool(PEDIGREE_POSITIVE_FLAGS & context["pedigree_positive"])
    if not (has_race_level and has_recent and has_condition_support):
        return 0.0, []
    return float(rule["bonus"]), ["RACE_LEVEL_RECENT_SYNERGY"]


def _rule_bonus(context: dict[str, Any], rule_name: str) -> tuple[float, list[str]]:
    if rule_name == "unknown_softening":
        return _unknown_softening_bonus(context)
    if rule_name == "positive_stack_boost":
        return _positive_stack_bonus(context)
    if rule_name == "risk_cap_positive_stack":
        return _risk_cap_bonus(context)
    if rule_name == "conditional_pace_support":
        return _conditional_pace_bonus(context)
    if rule_name == "race_level_recent_synergy":
        return _race_level_recent_synergy_bonus(context)
    if rule_name == "combined_recalibration":
        bonus_total = 0.0
        reasons: list[str] = []
        for child_rule in (
            "unknown_softening",
            "positive_stack_boost",
            "risk_cap_positive_stack",
            "conditional_pace_support",
            "race_level_recent_synergy",
        ):
            bonus, child_reasons = _rule_bonus(context, child_rule)
            if bonus > 0:
                bonus_total += bonus
                reasons.extend(child_reasons)
        if bonus_total <= 0:
            return 0.0, []
        cap = float(RECALIBRATION_RULES["combined_recalibration"]["bonus_cap"])
        return round(min(cap, bonus_total), 2), list(dict.fromkeys(reasons))
    return 0.0, []


def _simulate_rule(
    *,
    prediction: Prediction,
    ordered_scores: list[HorseScore],
    result_top3: list[int],
    rule_name: str,
) -> dict[str, Any]:
    baseline_top5 = [horse_score.horse_no for horse_score in ordered_scores[:5]]
    baseline_capture_count = _top5_capture_count(baseline_top5, result_top3)
    top5_cutoff = ordered_scores[4].total_score if len(ordered_scores) >= 5 else (ordered_scores[-1].total_score if ordered_scores else 0.0)
    baseline_rank_map = {horse_score.horse_no: index + 1 for index, horse_score in enumerate(ordered_scores)}
    bonus_cases: list[dict[str, Any]] = []
    scored_rows: list[tuple[HorseScore, float]] = []

    for horse_score in ordered_scores:
        context = _horse_context(
            prediction,
            horse_score,
            baseline_rank_map[horse_score.horse_no],
            top5_cutoff,
        )
        bonus, reasons = _rule_bonus(context, rule_name)
        simulated_score = round(horse_score.total_score + bonus, 2)
        scored_rows.append((horse_score, simulated_score))
        if bonus > 0:
            bonus_cases.append(
                {
                    "horse_no": horse_score.horse_no,
                    "horse_name": horse_score.horse_name,
                    "baseline_rank": context["rank"],
                    "baseline_score": horse_score.total_score,
                    "bonus": bonus,
                    "simulated_score": simulated_score,
                    "score_gap_to_top5": context["score_gap_to_top5"],
                    "positive_signal_count": context["positive_signal_count"],
                    "positive_signal_labels": context["positive_signal_labels"],
                    "risk_score": context["risk_score"],
                    "reasons": reasons,
                }
            )

    scored_rows.sort(key=lambda item: (-item[1], item[0].horse_no))
    simulated_top5 = [horse_score.horse_no for horse_score, _ in scored_rows[:5]]
    simulated_rank_map = {horse_score.horse_no: index + 1 for index, (horse_score, _) in enumerate(scored_rows)}
    simulated_capture_count = _top5_capture_count(simulated_top5, result_top3)
    new_entries = [horse_no for horse_no in simulated_top5 if horse_no not in baseline_top5]
    removed_entries = [horse_no for horse_no in baseline_top5 if horse_no not in simulated_top5]
    recovered_top3 = [horse_no for horse_no in new_entries if horse_no in result_top3]
    false_entries = [horse_no for horse_no in new_entries if horse_no not in result_top3]
    replaced_top3 = [horse_no for horse_no in removed_entries if horse_no in result_top3]

    bonus_case_map = {case["horse_no"]: case for case in bonus_cases}
    entered_cases: list[dict[str, Any]] = []
    for horse_no in new_entries:
        case = bonus_case_map.get(horse_no, {"horse_no": horse_no})
        entered_cases.append({**case, "new_rank": simulated_rank_map.get(horse_no)})

    return {
        "baseline_top5": baseline_top5,
        "simulated_top5": simulated_top5,
        "baseline_capture_count": baseline_capture_count,
        "simulated_capture_count": simulated_capture_count,
        "delta": simulated_capture_count - baseline_capture_count,
        "bonus_cases": bonus_cases,
        "new_entries": new_entries,
        "removed_entries": removed_entries,
        "entered_cases": entered_cases,
        "recovered_top3": recovered_top3,
        "false_entries": false_entries,
        "replaced_top3": replaced_top3,
        "top5_changed": simulated_top5 != baseline_top5,
    }


def _best_rule_key(rule_result: dict[str, Any]) -> tuple[float, float, int, int, int]:
    return (
        rule_result["safe_score"],
        rule_result["improvement"],
        -rule_result["worse_race_count"],
        -rule_result["replaced_top3_count"],
        rule_result["top5_changed_race_count"],
    )


def run_score_recalibration_simulation(
    *,
    from_date: str,
    to_date: str,
    predictions_dir: Path,
    results_dir: Path,
    reviews_dir: Path,
    baseline_mode: str | None = None,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
) -> dict[str, Any]:
    resolved_baseline_mode = _resolve_baseline_mode(baseline_mode, scoring_mode)
    baseline_custom_weights = _baseline_custom_weights(
        resolved_baseline_mode,
        pedigree_weight,
        race_level_weight,
        pace_weight,
    )
    baseline_weights = _baseline_weights(resolved_baseline_mode, baseline_custom_weights)
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
    for rule_name in RECALIBRATION_RULES:
        rule_state[rule_name] = {
            "candidate_bonus_count": 0,
            "top5_changed_race_count": 0,
            "recovered_top3_count": 0,
            "false_entry_count": 0,
            "better_race_count": 0,
            "worse_race_count": 0,
            "same_race_count": 0,
            "replaced_top3_count": 0,
            "simulated_captured_top3_horses": 0,
            "simulated_top5_winner_count": 0,
            "effective_bonuses": [],
            "entered_ranks": [],
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

        for rule_name in RECALIBRATION_RULES:
            state = rule_state[rule_name]
            simulation = _simulate_rule(
                prediction=prediction,
                ordered_scores=ordered_scores,
                result_top3=result_top3,
                rule_name=rule_name,
            )
            state["candidate_bonus_count"] += len(simulation["bonus_cases"])
            state["simulated_captured_top3_horses"] += simulation["simulated_capture_count"]
            if winner in simulation["simulated_top5"]:
                state["simulated_top5_winner_count"] += 1
            if simulation["top5_changed"]:
                state["top5_changed_race_count"] += 1
            state["recovered_top3_count"] += len(simulation["recovered_top3"])
            state["false_entry_count"] += len(simulation["false_entries"])
            state["replaced_top3_count"] += len(simulation["replaced_top3"])
            if simulation["delta"] > 0:
                state["better_race_count"] += 1
            elif simulation["delta"] < 0:
                state["worse_race_count"] += 1
            else:
                state["same_race_count"] += 1
            for case in simulation["entered_cases"]:
                if "bonus" in case:
                    state["effective_bonuses"].append(float(case["bonus"]))
                if "baseline_rank" in case:
                    state["entered_ranks"].append(float(case["baseline_rank"]))

            if simulation["top5_changed"] or simulation["recovered_top3"] or simulation["replaced_top3"]:
                row = {
                    "race_id": prediction.race_id,
                    "race_name": prediction.race_info.race_name if prediction.race_info and prediction.race_info.race_name else prediction.race_id,
                    "rule": rule_name,
                    "baseline_top5": simulation["baseline_top5"],
                    "simulated_top5": simulation["simulated_top5"],
                    "result_top3": result_top3,
                    "baseline_capture_count": simulation["baseline_capture_count"],
                    "simulated_capture_count": simulation["simulated_capture_count"],
                    "delta": simulation["delta"],
                    "entered_cases": simulation["entered_cases"],
                    "new_entries": simulation["new_entries"],
                    "removed_entries": simulation["removed_entries"],
                    "recovered_top3": simulation["recovered_top3"],
                    "false_entries": simulation["false_entries"],
                    "replaced_top3": simulation["replaced_top3"],
                    "baseline_recovery_applied": bool(baseline_recovery.get("recovery_applied", False)),
                }
                state["rows"].append(row)
                race_details.append(row)

    baseline_avg = round(baseline_captured_top3_horses / reviewed_race_count, 4) if reviewed_race_count else 0.0
    baseline_capture_rate = _safe_rate(baseline_captured_top3_horses, total_top3_horses)
    baseline_top5_winner_rate = _safe_rate(baseline_top5_winner_count, reviewed_race_count)

    rule_results: dict[str, dict[str, Any]] = {}
    for rule_name, state in rule_state.items():
        simulated_avg = round(state["simulated_captured_top3_horses"] / reviewed_race_count, 4) if reviewed_race_count else 0.0
        simulated_capture_rate = _safe_rate(state["simulated_captured_top3_horses"], total_top3_horses)
        improvement = round(simulated_avg - baseline_avg, 4)
        safe_score = state["recovered_top3_count"] - state["replaced_top3_count"] - state["worse_race_count"]
        rule_results[rule_name] = {
            "rule": rule_name,
            "description": RECALIBRATION_RULES[rule_name]["description"],
            "race_count": race_count,
            "reviewed_race_count": reviewed_race_count,
            "baseline_avg_captured_top3_per_race": baseline_avg,
            "simulated_avg_captured_top3_per_race": simulated_avg,
            "improvement": improvement,
            "baseline_capture_rate": baseline_capture_rate,
            "simulated_capture_rate": simulated_capture_rate,
            "baseline_top5_winner_rate": baseline_top5_winner_rate,
            "simulated_top5_winner_rate": _safe_rate(state["simulated_top5_winner_count"], reviewed_race_count),
            "candidate_bonus_count": state["candidate_bonus_count"],
            "top5_changed_race_count": state["top5_changed_race_count"],
            "recovered_top3_count": state["recovered_top3_count"],
            "false_entry_count": state["false_entry_count"],
            "better_race_count": state["better_race_count"],
            "worse_race_count": state["worse_race_count"],
            "same_race_count": state["same_race_count"],
            "net_better_minus_worse": state["better_race_count"] - state["worse_race_count"],
            "replaced_top3_count": state["replaced_top3_count"],
            "safe_score": safe_score,
            "avg_effective_bonus": _safe_mean([float(value) for value in state["effective_bonuses"]]),
            "avg_entered_baseline_rank": _safe_mean([float(value) for value in state["entered_ranks"]]),
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
    risky = [
        result["rule"]
        for result in rule_results.values()
        if result["worse_race_count"] >= result["better_race_count"]
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
            "baseline_weights": {
                "pedigree_weight": baseline_weights[0],
                "race_level_weight": baseline_weights[1],
                "pace_weight": baseline_weights[2],
            },
            "note": "what-if simulation only; prediction/review/result files are not modified",
        },
        "summary": {
            "race_count": race_count,
            "reviewed_race_count": reviewed_race_count,
            "baseline_avg_captured_top3_per_race": baseline_avg,
            "baseline_capture_rate": baseline_capture_rate,
            "baseline_top5_winner_rate": baseline_top5_winner_rate,
        },
        "rule_results": rule_results,
        "best_rule_candidates": {
            "best_rule": sorted_rules[0]["rule"] if sorted_rules else None,
            "recommended_rules": recommended,
            "risky_rules": sorted(set(risky)),
        },
        "race_details": race_details,
        "warnings": warnings,
    }
    report["findings"] = _build_findings(report)
    return report


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def _build_findings(report: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    rule_results = report["rule_results"]
    best = report["best_rule_candidates"]
    if best["recommended_rules"]:
        findings.append("改善候補あり。recommended_rules は正式 scoring へ入れる前に追加期間で再検証する価値がある。")
    if rule_results.get("unknown_softening", {}).get("improvement", 0.0) > 0:
        findings.append("UNKNOWN 系リスクは、正面シグナルが厚い場合に過小評価を生んでいる可能性。")
    if rule_results.get("risk_cap_positive_stack", {}).get("improvement", 0.0) > 0:
        findings.append("risk penalty は一部で強すぎる可能性があり、positive stack 時のcap検証余地。")
    if rule_results.get("conditional_pace_support", {}).get("improvement", 0.0) > 0:
        findings.append("pace は単独weightではなく、近走/race_levelとの条件付き補正なら使える可能性。")
    if rule_results.get("combined_recalibration", {}).get("worse_race_count", 0) > 3:
        findings.append("combined_recalibration は副作用が出やすい。導入するなら子ルール単位で分解するべき。")
    if all(result["improvement"] < 0.05 for result in rule_results.values()):
        findings.append("現時点では improvement が限定的。正式scoringへ入れるには根拠不足。")
    return findings or ["明確な改善ルールはまだ限定的。"]


def generate_score_recalibration_simulation_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    config = report["analysis_config"]
    rule_results = report["rule_results"]
    best = report["best_rule_candidates"]
    lines = [
        "# Score Recalibration Simulation Report",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}",
        f"- reviewed races: {summary['reviewed_race_count']}",
        f"- baseline mode: {config['baseline_mode']}",
        f"- baseline avg captured top3 per race: {summary['baseline_avg_captured_top3_per_race']:.2f}",
        f"- baseline capture rate: {_format_rate(summary['baseline_capture_rate'])}",
        "- 注意: what-if simulation のみ。prediction / review / result は変更しない。",
        "",
        "## Rule Comparison",
        "| rule | simulated avg | improvement | Top5 change | recovered top3 | false entry | better | worse | replaced top3 | safe_score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rule_name, result in sorted(rule_results.items()):
        lines.append(
            f"| {rule_name} | {result['simulated_avg_captured_top3_per_race']:.2f} | {result['improvement']:.2f} | "
            f"{result['top5_changed_race_count']} | {result['recovered_top3_count']} | {result['false_entry_count']} | "
            f"{result['better_race_count']} | {result['worse_race_count']} | {result['replaced_top3_count']} | {result['safe_score']} |"
        )

    lines.extend(
        [
            "",
            "## Best Rule Candidates",
            f"- best rule: {best['best_rule'] or 'なし'}",
            f"- recommended rules: {', '.join(best['recommended_rules']) if best['recommended_rules'] else 'なし'}",
            f"- risky rules: {', '.join(best['risky_rules']) if best['risky_rules'] else 'なし'}",
            "",
            "## Rule Details",
        ]
    )
    for rule_name, result in sorted(rule_results.items()):
        success_case = next((row for row in result["rows"] if row["delta"] > 0), None)
        failure_case = next((row for row in result["rows"] if row["delta"] < 0), None)
        lines.extend(
            [
                f"### {rule_name}",
                f"- 条件: {result['description']}",
                f"- bonus candidates: {result['candidate_bonus_count']}",
                f"- Top5 changed races: {result['top5_changed_race_count']}",
                f"- recovered_top3_count: {result['recovered_top3_count']}",
                f"- false_entry_count: {result['false_entry_count']}",
                f"- avg_effective_bonus: {result['avg_effective_bonus']:.2f}",
                f"- 典型成功case: {success_case['race_id']} / delta={success_case['delta']} / entry={success_case['new_entries']}" if success_case else "- 典型成功case: なし",
                f"- 典型失敗case: {failure_case['race_id']} / delta={failure_case['delta']} / entry={failure_case['new_entries']}" if failure_case else "- 典型失敗case: なし",
                "",
            ]
        )

    lines.extend(
        [
            "## Race Details",
            "| race_id | rule | baseline_top5 | simulated_top5 | result_top3 | delta | new_entries | removed | recovered | replaced_top3 |",
            "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    if report["race_details"]:
        for row in report["race_details"]:
            lines.append(
                f"| {row['race_id']} | {row['rule']} | {'→'.join(str(item) for item in row['baseline_top5'])} | "
                f"{'→'.join(str(item) for item in row['simulated_top5'])} | {'→'.join(str(item) for item in row['result_top3'])} | "
                f"{row['delta']} | {'→'.join(str(item) for item in row['new_entries']) or '-'} | "
                f"{'→'.join(str(item) for item in row['removed_entries']) or '-'} | "
                f"{'→'.join(str(item) for item in row['recovered_top3']) or '-'} | "
                f"{'→'.join(str(item) for item in row['replaced_top3']) or '-'} |"
            )
    else:
        lines.append("| なし | - | - | - | - | - | - | - | - | - |")

    lines.extend(["", "## Findings"])
    lines.extend(f"- {finding}" for finding in report.get("findings", []))
    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def save_score_recalibration_simulation_json(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_score_recalibration_simulation_markdown(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

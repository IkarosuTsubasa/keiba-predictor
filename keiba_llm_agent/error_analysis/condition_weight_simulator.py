from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from keiba_llm_agent.backtest.scoring_comparator import MARK_LABELS, count_marked_top3, result_top3_list
from keiba_llm_agent.error_analysis.missed_top3_analyzer import (
    _collect_predictions_in_period,
    _find_analysis,
    _load_race_data,
    _load_result,
    _load_review,
)
from keiba_llm_agent.scoring.borderline_recovery import apply_top5_borderline_recovery
from keiba_llm_agent.schemas.prediction import HorseScore, Prediction


DEFAULT_FORMULA_WEIGHTS = {
    "recent_form_weight": 1.5,
    "distance_weight": 1.2,
    "course_weight": 1.0,
    "track_condition_weight": 0.8,
    "jockey_weight": 0.8,
    "risk_scale": 1.0,
    "pedigree_weight": 0.2,
    "race_level_weight": 1.0,
    "pace_weight": 0.0,
}


SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {
        "description": "candidate_default + current borderline recovery",
        "recovery_profile": "current",
    },
    "turf_race_level_1_2": {
        "description": "芝のみ race_level_weight=1.2",
        "surface_overrides": {"芝": {"race_level_weight": 1.2}},
        "recovery_profile": "current",
    },
    "large_field_race_level_1_2": {
        "description": "14頭以上のみ race_level_weight=1.2",
        "large_field_overrides": {"race_level_weight": 1.2},
        "recovery_profile": "current",
    },
    "turf_or_large_race_level_1_2": {
        "description": "芝または14頭以上で race_level_weight=1.2",
        "surface_overrides": {"芝": {"race_level_weight": 1.2}},
        "large_field_overrides": {"race_level_weight": 1.2},
        "recovery_profile": "current",
    },
    "dirt_pedigree_0_3_turf_large_race_level_1_2": {
        "description": "ダートpedigree=0.3、芝/14頭以上race_level=1.2",
        "surface_overrides": {"ダート": {"pedigree_weight": 0.3}, "芝": {"race_level_weight": 1.2}},
        "large_field_overrides": {"race_level_weight": 1.2},
        "recovery_profile": "current",
    },
    "track_weight_0_6": {
        "description": "track_condition_weight=0.6",
        "formula_overrides": {"track_condition_weight": 0.6},
        "recovery_profile": "current",
    },
    "jockey_weight_0_6": {
        "description": "jockey_weight=0.6",
        "formula_overrides": {"jockey_weight": 0.6},
        "recovery_profile": "current",
    },
    "track_jockey_0_6": {
        "description": "track_condition_weight=0.6、jockey_weight=0.6",
        "formula_overrides": {"track_condition_weight": 0.6, "jockey_weight": 0.6},
        "recovery_profile": "current",
    },
    "risk_scale_1_25": {
        "description": "risk penalty を1.25倍",
        "formula_overrides": {"risk_scale": 1.25},
        "recovery_profile": "current",
    },
    "strict_rank6_recovery": {
        "description": "rank6限定。非market正面シグナル>=3、強risk除外で境界補正",
        "recovery_profile": "strict_rank6",
    },
    "no_borderline_recovery": {
        "description": "borderline recovery を使わない比較用",
        "recovery_profile": "none",
    },
}


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def _flag_set(analysis: object | None, field_name: str) -> set[str]:
    return set(getattr(analysis, field_name, []) or [])


def _field_size(prediction: Prediction, race_data: object | None) -> int:
    horses = getattr(race_data, "horses", None)
    if horses:
        return len(horses)
    return len(prediction.horse_scores)


def _scenario_weights(prediction: Prediction, field_size: int, scenario: dict[str, Any]) -> dict[str, float]:
    weights = dict(DEFAULT_FORMULA_WEIGHTS)
    weights.update(scenario.get("formula_overrides", {}))
    surface = prediction.race_info.surface if prediction.race_info else None
    surface_overrides = scenario.get("surface_overrides", {})
    if surface in surface_overrides:
        weights.update(surface_overrides[surface])
    if field_size >= 14:
        weights.update(scenario.get("large_field_overrides", {}))
    return weights


def _recompute_score(horse_score: HorseScore, weights: dict[str, float]) -> float:
    scores = horse_score.scores
    base_total = horse_score.base_total_score
    base_total += (weights["recent_form_weight"] - DEFAULT_FORMULA_WEIGHTS["recent_form_weight"]) * scores.recent_form
    base_total += (weights["distance_weight"] - DEFAULT_FORMULA_WEIGHTS["distance_weight"]) * scores.distance_fit
    base_total += (weights["course_weight"] - DEFAULT_FORMULA_WEIGHTS["course_weight"]) * scores.course_fit
    base_total += (weights["track_condition_weight"] - DEFAULT_FORMULA_WEIGHTS["track_condition_weight"]) * scores.track_condition_fit
    base_total += (weights["jockey_weight"] - DEFAULT_FORMULA_WEIGHTS["jockey_weight"]) * scores.jockey_fit
    base_total += (weights["risk_scale"] - DEFAULT_FORMULA_WEIGHTS["risk_scale"]) * scores.risk
    return round(
        base_total
        + horse_score.pedigree_adjustment.pedigree_adjustment * weights["pedigree_weight"]
        + horse_score.race_level_adjustment.adjustment * weights["race_level_weight"]
        + horse_score.pace_adjustment.adjustment * weights["pace_weight"],
        2,
    )


def _score_horses_for_scenario(
    prediction: Prediction,
    field_size: int,
    scenario: dict[str, Any],
) -> list[HorseScore]:
    weights = _scenario_weights(prediction, field_size, scenario)
    scored: list[HorseScore] = []
    for horse_score in prediction.horse_scores:
        total_score = _recompute_score(horse_score, weights)
        updated_breakdown = horse_score.score_breakdown.model_copy(
            update={
                "base_total_score": round(
                    total_score
                    - horse_score.pedigree_adjustment.pedigree_adjustment * weights["pedigree_weight"]
                    - horse_score.race_level_adjustment.adjustment * weights["race_level_weight"]
                    - horse_score.pace_adjustment.adjustment * weights["pace_weight"],
                    2,
                ),
                "pedigree_weight": weights["pedigree_weight"],
                "pedigree_adjustment_weighted": round(horse_score.pedigree_adjustment.pedigree_adjustment * weights["pedigree_weight"], 2),
                "race_level_weight": weights["race_level_weight"],
                "race_level_adjustment_weighted": round(horse_score.race_level_adjustment.adjustment * weights["race_level_weight"], 2),
                "pace_weight": weights["pace_weight"],
                "pace_adjustment_weighted": round(horse_score.pace_adjustment.adjustment * weights["pace_weight"], 2),
                "borderline_recovery_bonus": 0.0,
                "total_score": total_score,
                "total_score_after_recovery": total_score,
            }
        )
        scored.append(
            horse_score.model_copy(
                update={
                    "total_score": total_score,
                    "score_breakdown": updated_breakdown,
                    "borderline_recovery": horse_score.borderline_recovery.model_copy(
                        update={"applied": False, "recovery_bonus": 0.0, "original_rank": None, "new_rank": None, "reasons": []}
                    ),
                }
            )
        )
    return scored


def _positive_signal_count(prediction: Prediction, horse_score: HorseScore) -> tuple[int, int, list[str], set[str], set[str]]:
    deep = _find_analysis(prediction, "deep_analyses", horse_score.horse_no)
    pedigree = _find_analysis(prediction, "pedigree_analyses", horse_score.horse_no)
    race_level = _find_analysis(prediction, "race_level_analyses", horse_score.horse_no)
    pace = _find_analysis(prediction, "pace_analyses", horse_score.horse_no)
    deep_positive = _flag_set(deep, "positive_flags")
    deep_risk = _flag_set(deep, "risk_flags")
    pedigree_positive = _flag_set(pedigree, "positive_flags")
    race_level_positive = _flag_set(race_level, "positive_flags")
    race_level_risk = _flag_set(race_level, "risk_flags")
    pace_positive = _flag_set(pace, "positive_flags")
    labels: list[str] = []
    if {"RECENT_FORM_STRONG", "RECENT_FORM_STABLE"} & deep_positive:
        labels.append("RECENT_FORM_SIGNAL")
    if {"DISTANCE_FIT", "COURSE_FIT"} & deep_positive:
        labels.append("CONDITION_FIT_SIGNAL")
    if {"PEDIGREE_DISTANCE_FIT", "PEDIGREE_STAMINA_FIT", "PEDIGREE_SURFACE_FIT"} & pedigree_positive:
        labels.append("PEDIGREE_SIGNAL")
    if {"HEAD_TO_HEAD_POSITIVE", "LARGE_FIELD_GOOD_RUN", "UNDERVALUED_GOOD_RUN", "VALUE_WIN"} & race_level_positive:
        labels.append("RACE_LEVEL_SIGNAL")
    if {"PACE_FIT", "STALKER_ADVANTAGE", "CLOSING_SPEED"} & pace_positive:
        labels.append("PACE_SIGNAL")
    market_count = 0
    if horse_score.popularity is not None and horse_score.popularity <= 3:
        market_count += 1
    if horse_score.popularity is not None and horse_score.popularity <= 5 and horse_score.odds is not None and horse_score.odds < 10:
        market_count += 1
    return len(labels), market_count, labels, deep_risk, race_level_risk


def _apply_strict_rank6_recovery(prediction: Prediction, scored_horses: list[HorseScore]) -> tuple[list[HorseScore], dict[str, Any]]:
    ranked = sorted(scored_horses, key=lambda item: (-item.total_score, item.horse_no))
    result = {"recovery_applied": False, "recovery_cases": [], "recovered_top5": [horse.horse_no for horse in ranked[:5]]}
    if len(ranked) < 6:
        return ranked, result
    rank5 = ranked[4]
    rank6 = ranked[5]
    score_gap = round(rank5.total_score - rank6.total_score, 1)
    if score_gap > 1.0:
        return ranked, result
    nonmarket_count, market_count, labels, deep_risk, race_level_risk = _positive_signal_count(prediction, rank6)
    if rank6.scores.risk <= -7:
        return ranked, result
    if "HEAD_TO_HEAD_NEGATIVE" in race_level_risk:
        return ranked, result
    if "DATA_INCOMPLETE" in deep_risk:
        return ranked, result
    if "RECENT_FORM_WEAK" in deep_risk and "RECENT_FORM_STABLE" not in labels:
        return ranked, result
    if not (nonmarket_count >= 3 or (nonmarket_count >= 2 and market_count >= 1)):
        return ranked, result
    target_score = rank5.total_score + 0.01
    bonus = round(target_score - rank6.total_score, 2)
    adjusted: list[HorseScore] = []
    for horse_score in scored_horses:
        if horse_score.horse_no != rank6.horse_no:
            adjusted.append(horse_score)
            continue
        updated_breakdown = horse_score.score_breakdown.model_copy(
            update={
                "borderline_recovery_bonus": bonus,
                "total_score_after_recovery": target_score,
            }
        )
        adjusted.append(
            horse_score.model_copy(
                update={
                    "total_score": target_score,
                    "score_breakdown": updated_breakdown,
                }
            )
        )
    adjusted_ranked = sorted(adjusted, key=lambda item: (-item.total_score, item.horse_no))
    new_rank = next(index + 1 for index, horse_score in enumerate(adjusted_ranked) if horse_score.horse_no == rank6.horse_no)
    if new_rank > 5:
        return ranked, result
    result = {
        "recovery_applied": True,
        "recovery_cases": [
            {
                "horse_no": rank6.horse_no,
                "horse_name": rank6.horse_name,
                "original_rank": 6,
                "new_rank": new_rank,
                "score_gap_to_top5": score_gap,
                "recovery_bonus": bonus,
                "recovery_reasons": ["STRICT_RANK6", *labels],
            }
        ],
        "recovered_top5": [horse.horse_no for horse in adjusted_ranked[:5]],
    }
    return adjusted_ranked, result


def _rank_for_scenario(
    prediction: Prediction,
    field_size: int,
    scenario: dict[str, Any],
) -> tuple[list[HorseScore], dict[str, Any]]:
    scored_horses = _score_horses_for_scenario(prediction, field_size, scenario)
    recovery_profile = scenario.get("recovery_profile", "current")
    if recovery_profile == "none":
        ranked = sorted(scored_horses, key=lambda item: (-item.total_score, item.horse_no))
        return ranked, {"recovery_applied": False, "recovery_cases": [], "recovered_top5": [horse.horse_no for horse in ranked[:5]]}
    if recovery_profile == "strict_rank6":
        return _apply_strict_rank6_recovery(prediction, scored_horses)
    recovery_result = apply_top5_borderline_recovery(
        scored_horses,
        prediction.deep_analyses,
        prediction.pedigree_analyses,
        prediction.race_level_analyses,
        prediction.pace_analyses,
        prediction.race_info,
        prediction.scoring_config,
        enabled=True,
    )
    adjusted = recovery_result["adjusted_horse_scores"]
    return sorted(adjusted, key=lambda item: (-item.total_score, item.horse_no)), recovery_result


def _marks_from_ranked(ranked: list[HorseScore]) -> dict[str, int]:
    marks = {label: 0 for label in MARK_LABELS}
    for index, label in enumerate(MARK_LABELS):
        if index < len(ranked):
            marks[label] = ranked[index].horse_no
    return marks


def _evaluate_ranked(ranked: list[HorseScore], result_top3: list[int], winner: int) -> dict[str, Any]:
    marks = _marks_from_ranked(ranked)
    top5 = [horse.horse_no for horse in ranked[:5]]
    main_mark = marks["◎"]
    return {
        "marks": marks,
        "top5": top5,
        "main_mark_top3": main_mark in result_top3,
        "main_mark_win": main_mark == winner,
        "marked_top3_count": count_marked_top3(marks, result_top3),
        "top5_contains_winner": winner in top5,
        "top5_top3_count": len(set(top5).intersection(result_top3)),
    }


def _scenario_result_template() -> dict[str, Any]:
    return {
        "marked_top3_sum": 0,
        "main_mark_top3_count": 0,
        "main_mark_win_count": 0,
        "top5_winner_count": 0,
        "better_race_count": 0,
        "worse_race_count": 0,
        "same_race_count": 0,
        "top5_changed_race_count": 0,
        "recovery_applied_count": 0,
        "recovered_top3_count": 0,
        "replaced_top3_count": 0,
        "rows": [],
        "counts": [],
    }


def _best_scenario_key(result: dict[str, Any]) -> tuple[float, float, int, int, float]:
    return (
        result["avg_marked_top3_count"],
        result["top5_winner_rate"],
        result["better_race_count"] - result["worse_race_count"],
        -result["worse_race_count"],
        result["main_mark_top3_rate"],
    )


def run_condition_weight_simulation(
    *,
    from_date: str,
    to_date: str,
    predictions_dir: Path,
    results_dir: Path,
    reviews_dir: Path,
    race_data_dir: Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    predictions, prediction_warnings = _collect_predictions_in_period(predictions_dir, from_date, to_date)
    warnings.extend(prediction_warnings)
    scenario_state = {name: _scenario_result_template() for name in SCENARIOS}
    race_details: list[dict[str, Any]] = []
    race_count = len(predictions)
    reviewed_race_count = 0
    total_top3_horses = 0

    for prediction in predictions:
        result_data = _load_result(results_dir / f"{prediction.race_id}.json")
        review = _load_review(reviews_dir / f"{prediction.race_id}.json")
        if result_data is None:
            warnings.append(f"result missing for race_id={prediction.race_id}")
            continue
        if review is None:
            warnings.append(f"review missing for race_id={prediction.race_id}")
            continue
        race_data = _load_race_data((race_data_dir or Path()) / f"{prediction.race_id}.json") if race_data_dir is not None else None
        field_size = _field_size(prediction, race_data)
        reviewed_race_count += 1
        result_top3 = result_top3_list(result_data)
        total_top3_horses += len(result_top3)
        winner = result_data.result.first

        per_race: dict[str, Any] = {
            "race_id": prediction.race_id,
            "race_name": prediction.race_info.race_name if prediction.race_info and prediction.race_info.race_name else prediction.race_id,
            "course": prediction.race_info.course if prediction.race_info else None,
            "surface": prediction.race_info.surface if prediction.race_info else None,
            "distance": prediction.race_info.distance if prediction.race_info else None,
            "field_size": field_size,
            "result_top3": result_top3,
        }
        baseline_metrics: dict[str, Any] | None = None
        baseline_top5: list[int] | None = None
        for scenario_name, scenario in SCENARIOS.items():
            ranked, recovery = _rank_for_scenario(prediction, field_size, scenario)
            metrics = _evaluate_ranked(ranked, result_top3, winner)
            state = scenario_state[scenario_name]
            state["marked_top3_sum"] += metrics["marked_top3_count"]
            state["main_mark_top3_count"] += int(metrics["main_mark_top3"])
            state["main_mark_win_count"] += int(metrics["main_mark_win"])
            state["top5_winner_count"] += int(metrics["top5_contains_winner"])
            state["counts"].append(metrics["marked_top3_count"])
            if recovery.get("recovery_applied"):
                state["recovery_applied_count"] += 1

            if scenario_name == "baseline":
                baseline_metrics = metrics
                baseline_top5 = metrics["top5"]
                per_race["baseline_top5"] = metrics["top5"]
                per_race["baseline_marked_top3_count"] = metrics["marked_top3_count"]
                continue

            if baseline_metrics is None or baseline_top5 is None:
                raise RuntimeError("baseline scenario must be evaluated first")
            delta = metrics["marked_top3_count"] - baseline_metrics["marked_top3_count"]
            if delta > 0:
                state["better_race_count"] += 1
            elif delta < 0:
                state["worse_race_count"] += 1
            else:
                state["same_race_count"] += 1
            if metrics["top5"] != baseline_top5:
                state["top5_changed_race_count"] += 1
            new_entries = [horse_no for horse_no in metrics["top5"] if horse_no not in baseline_top5]
            removed_entries = [horse_no for horse_no in baseline_top5 if horse_no not in metrics["top5"]]
            recovered_top3 = [horse_no for horse_no in new_entries if horse_no in result_top3]
            replaced_top3 = [horse_no for horse_no in removed_entries if horse_no in result_top3]
            state["recovered_top3_count"] += len(recovered_top3)
            state["replaced_top3_count"] += len(replaced_top3)
            if delta != 0 or new_entries:
                row = {
                    "race_id": prediction.race_id,
                    "scenario": scenario_name,
                    "baseline_top5": baseline_top5,
                    "scenario_top5": metrics["top5"],
                    "result_top3": result_top3,
                    "delta": delta,
                    "new_entries": new_entries,
                    "removed_entries": removed_entries,
                    "recovered_top3": recovered_top3,
                    "replaced_top3": replaced_top3,
                    "recovery_applied": bool(recovery.get("recovery_applied", False)),
                    "recovery_cases": recovery.get("recovery_cases", []),
                }
                state["rows"].append(row)
                race_details.append(row)
            per_race[f"{scenario_name}_top5"] = metrics["top5"]
            per_race[f"{scenario_name}_marked_top3_count"] = metrics["marked_top3_count"]
            per_race[f"{scenario_name}_delta"] = delta

    scenario_results: dict[str, dict[str, Any]] = {}
    baseline_avg = 0.0
    baseline_counts = scenario_state["baseline"]["counts"]
    for scenario_name, state in scenario_state.items():
        avg_marked = round(state["marked_top3_sum"] / reviewed_race_count, 4) if reviewed_race_count else 0.0
        if scenario_name == "baseline":
            baseline_avg = avg_marked
        scenario_results[scenario_name] = {
            "scenario": scenario_name,
            "description": SCENARIOS[scenario_name]["description"],
            "reviewed_race_count": reviewed_race_count,
            "avg_marked_top3_count": avg_marked,
            "improvement": round(avg_marked - baseline_avg, 4) if scenario_name != "baseline" else 0.0,
            "capture_rate": _safe_rate(state["marked_top3_sum"], total_top3_horses),
            "main_mark_top3_rate": _safe_rate(state["main_mark_top3_count"], reviewed_race_count),
            "main_mark_win_rate": _safe_rate(state["main_mark_win_count"], reviewed_race_count),
            "top5_winner_rate": _safe_rate(state["top5_winner_count"], reviewed_race_count),
            "better_race_count": state["better_race_count"],
            "worse_race_count": state["worse_race_count"],
            "same_race_count": state["same_race_count"] if scenario_name != "baseline" else reviewed_race_count,
            "top5_changed_race_count": state["top5_changed_race_count"],
            "recovery_applied_count": state["recovery_applied_count"],
            "recovered_top3_count": state["recovered_top3_count"],
            "replaced_top3_count": state["replaced_top3_count"],
            "safe_score": state["recovered_top3_count"] - state["replaced_top3_count"] - state["worse_race_count"],
            "avg_count_delta_vs_baseline": round(_safe_mean([float(a - b) for a, b in zip(state["counts"], baseline_counts)]), 4) if scenario_name != "baseline" else 0.0,
            "rows": state["rows"],
        }

    ranked_scenarios = [
        result for name, result in scenario_results.items() if name != "baseline"
    ]
    ranked_scenarios.sort(key=_best_scenario_key, reverse=True)
    recommended = [
        result["scenario"]
        for result in ranked_scenarios
        if result["improvement"] >= 0.01
        and result["better_race_count"] > result["worse_race_count"]
        and result["worse_race_count"] <= 3
        and result["top5_winner_rate"] >= scenario_results["baseline"]["top5_winner_rate"]
    ]
    report = {
        "period": {"from": from_date, "to": to_date},
        "analysis_config": {
            "note": "what-if simulation only; prediction/review/result files are not modified",
            "baseline": "candidate_default + current borderline recovery",
            "scenarios": SCENARIOS,
        },
        "summary": {
            "race_count": race_count,
            "reviewed_race_count": reviewed_race_count,
            "total_top3_horses": total_top3_horses,
            "baseline_avg_marked_top3_count": scenario_results["baseline"]["avg_marked_top3_count"],
            "baseline_capture_rate": scenario_results["baseline"]["capture_rate"],
            "baseline_top5_winner_rate": scenario_results["baseline"]["top5_winner_rate"],
        },
        "scenario_results": scenario_results,
        "best_scenario_summary": {
            "best_avg_marked_top3_count": ranked_scenarios[0]["scenario"] if ranked_scenarios else None,
            "recommended_scenarios": recommended,
            "recommended_reason": _recommended_reason(scenario_results, recommended),
        },
        "race_details": race_details,
        "warnings": warnings,
    }
    report["findings"] = _build_findings(report)
    return report


def _recommended_reason(scenario_results: dict[str, dict[str, Any]], recommended: list[str]) -> str:
    if not recommended:
        return "現時点では default scoring に入れるだけの安定改善は限定的。"
    scenario = scenario_results[recommended[0]]
    baseline = scenario_results["baseline"]
    return (
        f"{recommended[0]} は印内Top3平均を {baseline['avg_marked_top3_count']:.2f} → "
        f"{scenario['avg_marked_top3_count']:.2f} に改善し、better={scenario['better_race_count']} / "
        f"worse={scenario['worse_race_count']}。"
    )


def _build_findings(report: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    results = report["scenario_results"]
    recommended = report["best_scenario_summary"]["recommended_scenarios"]
    if recommended:
        findings.append("条件型weightに小幅な改善候補あり。追加期間での再検証対象。")
    turf = results.get("turf_race_level_1_2")
    if turf and turf["improvement"] > 0:
        findings.append("芝では race_level を少し強める余地がある。ただし改善幅は小さい。")
    strict = results.get("strict_rank6_recovery")
    baseline = results.get("baseline")
    if strict and baseline and strict["avg_marked_top3_count"] < baseline["avg_marked_top3_count"]:
        findings.append("rank6 recovery を厳しくしすぎると回収力が落ちる。現行recoveryは一定の価値がある。")
    if all(result["improvement"] < 0.02 for name, result in results.items() if name != "baseline"):
        findings.append("単独シナリオの改善はまだ小さい。正式導入は慎重にすべき。")
    return findings or ["明確な改善候補は限定的。"]


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_condition_weight_simulation_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    best = report["best_scenario_summary"]
    lines = [
        "# Condition Weight Simulation Report",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}",
        f"- reviewed races: {summary['reviewed_race_count']}",
        f"- baseline avg marked top3: {summary['baseline_avg_marked_top3_count']:.2f}",
        f"- baseline capture rate: {_format_rate(summary['baseline_capture_rate'])}",
        f"- baseline Top5 winner率: {_format_rate(summary['baseline_top5_winner_rate'])}",
        "- 注意: what-if simulation のみ。prediction / review / result は変更しない。",
        "",
        "## Scenario Comparison",
        "| scenario | avg | improvement | main Top3率 | Top5 winner率 | better | worse | Top5 changed | recovered | replaced | safe_score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario_name, result in sorted(report["scenario_results"].items()):
        lines.append(
            f"| {scenario_name} | {result['avg_marked_top3_count']:.2f} | {result['improvement']:.2f} | "
            f"{_format_rate(result['main_mark_top3_rate'])} | {_format_rate(result['top5_winner_rate'])} | "
            f"{result['better_race_count']} | {result['worse_race_count']} | {result['top5_changed_race_count']} | "
            f"{result['recovered_top3_count']} | {result['replaced_top3_count']} | {result['safe_score']} |"
        )

    lines.extend(
        [
            "",
            "## Best Scenario",
            f"- best avg marked top3: {best['best_avg_marked_top3_count'] or 'なし'}",
            f"- recommended scenarios: {', '.join(best['recommended_scenarios']) if best['recommended_scenarios'] else 'なし'}",
            f"- recommended reason: {best['recommended_reason']}",
            "",
            "## Scenario Details",
        ]
    )
    for scenario_name, result in sorted(report["scenario_results"].items()):
        success = next((row for row in result["rows"] if row["delta"] > 0), None)
        failure = next((row for row in result["rows"] if row["delta"] < 0), None)
        lines.extend(
            [
                f"### {scenario_name}",
                f"- 条件: {result['description']}",
                f"- Top5 changed: {result['top5_changed_race_count']}",
                f"- recovery_applied_count: {result['recovery_applied_count']}",
                f"- 典型成功case: {success['race_id']} / delta={success['delta']} / new={success['new_entries']}" if success else "- 典型成功case: なし",
                f"- 典型失敗case: {failure['race_id']} / delta={failure['delta']} / new={failure['new_entries']}" if failure else "- 典型失敗case: なし",
                "",
            ]
        )

    lines.extend(
        [
            "## Race Details",
            "| race_id | scenario | baseline_top5 | scenario_top5 | result_top3 | delta | new | removed | recovered | replaced |",
            "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    if report["race_details"]:
        for row in report["race_details"]:
            lines.append(
                f"| {row['race_id']} | {row['scenario']} | {'→'.join(str(item) for item in row['baseline_top5'])} | "
                f"{'→'.join(str(item) for item in row['scenario_top5'])} | {'→'.join(str(item) for item in row['result_top3'])} | "
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


def save_condition_weight_simulation_json(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_condition_weight_simulation_markdown(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from keiba_llm_agent.backtest.scoring_comparator import (
    WeightTuningMode,
    calculate_weighted_score,
    rank_prediction_for_weight_tuning_mode_with_recovery,
    result_top3_list,
)
from keiba_llm_agent.error_analysis.missed_top3_analyzer import (
    _build_miss_categories,
    _build_race_horse_map,
    _collect_predictions_in_period,
    _find_analysis,
    _format_rate,
    _load_race_data,
    _load_result,
    _load_review,
)
from keiba_llm_agent.schemas.prediction import HorseScore, Prediction
from keiba_llm_agent.schemas.result import ResultData


BASELINE_MODE_CHOICES = [
    "base_only",
    "pedigree_only",
    "race_level_only",
    "pace_only",
    "current_full",
    "candidate_default",
    "candidate_default_recovered",
    "custom",
]

POSITIVE_RULE_CATEGORIES = {
    "RACE_LEVEL_UNDERESTIMATED",
    "RECENT_FORM_UNDERESTIMATED",
    "DISTANCE_FIT_UNDERESTIMATED",
    "COURSE_FIT_UNDERESTIMATED",
    "PEDIGREE_FIT_UNDERESTIMATED",
    "PACE_FIT_UNDERESTIMATED",
    "POPULAR_BUT_MISSED",
}

CONSERVATIVE_RULES = {
    "rank7_8_popular_racelevel",
    "rank7_8_multi_signal",
    "conservative_rank7_only",
}

RULE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "rank7_8_popular_racelevel": {
        "description": "predicted_rank 7-8、人気上位、race_level/近走シグナルあり、オッズ<=15、strong_risk除外",
        "rank_range": (7, 8),
        "odds_max": 15.0,
        "popularity_max": 3,
    },
    "rank7_8_multi_signal": {
        "description": "predicted_rank 7-8、positive missed categories>=3、オッズ<=20、strong_risk除外",
        "rank_range": (7, 8),
        "odds_max": 20.0,
        "min_positive": 3,
    },
    "popular_safety_net": {
        "description": "predicted_rank 7-10、人気<=3、オッズ<=10、positive>=2、strong_risk除外",
        "rank_range": (7, 10),
        "odds_max": 10.0,
        "popularity_max": 3,
        "min_positive": 2,
    },
    "race_level_safety_net": {
        "description": "predicted_rank 7-10、RACE_LEVEL_UNDERESTIMATED、positive>=3、オッズ<=20、strong_risk除外",
        "rank_range": (7, 10),
        "odds_max": 20.0,
        "min_positive": 3,
    },
    "odds_undervalued_limited": {
        "description": "predicted_rank 7-10、ODDS_UNDERESTIMATED、オッズ8-30、positive>=3、strong_risk除外（watchlist用 / default rankingには非推奨）",
        "rank_range": (7, 10),
        "odds_min": 8.0,
        "odds_max": 30.0,
        "min_positive": 3,
    },
    "conservative_rank7_only": {
        "description": "predicted_rank=7、人気<=5、オッズ<=15、positive>=2、strong_risk除外",
        "rank_range": (7, 7),
        "odds_max": 15.0,
        "popularity_max": 5,
        "min_positive": 2,
    },
}


def _resolve_baseline_mode(
    baseline_mode: str | None,
    scoring_mode: str | None,
) -> str:
    if baseline_mode:
        return baseline_mode
    if scoring_mode:
        return scoring_mode
    return "candidate_default_recovered"


def _baseline_custom_weights(
    mode: str,
    pedigree_weight: float | None,
    race_level_weight: float | None,
    pace_weight: float | None,
) -> tuple[float, float, float] | None:
    if mode != "custom":
        return None
    return (
        0.2 if pedigree_weight is None else pedigree_weight,
        1.0 if race_level_weight is None else race_level_weight,
        0.0 if pace_weight is None else pace_weight,
    )


def _ordered_horse_scores(
    prediction: Prediction,
    mode: str,
    custom_weights: tuple[float, float, float] | None,
) -> tuple[list[HorseScore], dict[str, Any]]:
    ranked, recovery_result = rank_prediction_for_weight_tuning_mode_with_recovery(
        prediction,
        mode=mode,  # type: ignore[arg-type]
        custom_weights=custom_weights,
        enable_borderline_recovery=False,
    )
    return ranked, recovery_result


def _finish_map(result_data: ResultData) -> dict[int, int]:
    if result_data.finish_order:
        return {item.horse_no: item.finish for item in result_data.finish_order}
    return {
        result_data.result.first: 1,
        result_data.result.second: 2,
        result_data.result.third: 3,
    }


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def _top5_capture_count(top5: list[int], result_top3: list[int]) -> int:
    return len(set(top5).intersection(result_top3))


def _candidate_positive_count(categories: list[str]) -> int:
    return sum(1 for category in categories if category in POSITIVE_RULE_CATEGORIES)


def _candidate_has_strong_risk(
    *,
    categories: list[str],
    total_risk_count: int,
    predicted_rank: int,
    popularity: int | None,
    odds: float | None,
    score_gap_to_top5: float | None,
) -> bool:
    return (
        ("RISK_OVER_PENALIZED" in categories and total_risk_count >= 4)
        or (popularity is not None and popularity >= 10 and odds is not None and odds >= 50.0)
        or predicted_rank >= 13
        or (score_gap_to_top5 is not None and score_gap_to_top5 > 6.0)
    )


def _candidate_note(candidate: dict[str, Any]) -> str:
    positives = ", ".join(candidate["positive_categories"]) if candidate["positive_categories"] else "positiveなし"
    return f"{positives} / gap={candidate['score_gap_to_top5']}"


def _build_candidate_pool(
    prediction: Prediction,
    result_data: ResultData,
    ranked_scores: list[HorseScore],
    race_horse_map: dict[int, object],
    *,
    max_rank: int,
    include_rank13: bool,
) -> list[dict[str, Any]]:
    finish_map = _finish_map(result_data)
    horse_score_map = {horse_score.horse_no: horse_score for horse_score in prediction.horse_scores}
    rank_map = {horse_score.horse_no: index + 1 for index, horse_score in enumerate(ranked_scores)}
    top5_cutoff_score = ranked_scores[4].total_score if len(ranked_scores) >= 5 else (ranked_scores[-1].total_score if ranked_scores else 0.0)
    candidates: list[dict[str, Any]] = []

    for horse_score in ranked_scores:
        predicted_rank = rank_map[horse_score.horse_no]
        if predicted_rank < 7:
            continue
        if not include_rank13 and predicted_rank >= 13:
            continue
        if predicted_rank > max_rank:
            continue

        horse_entry = race_horse_map.get(horse_score.horse_no)
        popularity = horse_score.popularity if horse_score.popularity is not None else getattr(horse_entry, "popularity", None)
        odds = horse_score.odds if horse_score.odds is not None else getattr(horse_entry, "odds", None)
        finish = finish_map.get(horse_score.horse_no)
        deep_analysis = _find_analysis(prediction, "deep_analyses", horse_score.horse_no)
        pedigree_analysis = _find_analysis(prediction, "pedigree_analyses", horse_score.horse_no)
        race_level_analysis = _find_analysis(prediction, "race_level_analyses", horse_score.horse_no)
        pace_analysis = _find_analysis(prediction, "pace_analyses", horse_score.horse_no)
        categories = _build_miss_categories(
            predicted_rank=predicted_rank,
            score_gap_to_top5=round(top5_cutoff_score - horse_score.total_score, 1),
            popularity=popularity,
            finish=finish if finish is not None else 99,
            deep_analysis=deep_analysis,
            pedigree_analysis=pedigree_analysis,
            race_level_analysis=race_level_analysis,
            pace_analysis=pace_analysis,
            top_n=5,
            data_missing=False,
            horse_score=horse_score_map.get(horse_score.horse_no),
        )
        deep_risk = set(getattr(deep_analysis, "risk_flags", []))
        pedigree_risk = set(getattr(pedigree_analysis, "risk_flags", []))
        race_level_risk = set(getattr(race_level_analysis, "risk_flags", []))
        pace_risk = set(getattr(pace_analysis, "risk_flags", []))
        total_risk_count = len(deep_risk | pedigree_risk | race_level_risk | pace_risk)
        positive_categories = [category for category in categories if category in POSITIVE_RULE_CATEGORIES]
        score_gap = round(top5_cutoff_score - horse_score.total_score, 1)
        candidates.append(
            {
                "horse_no": horse_score.horse_no,
                "horse_name": horse_score.horse_name,
                "predicted_rank": predicted_rank,
                "finish": finish,
                "odds": odds,
                "popularity": popularity,
                "score_gap_to_top5": score_gap,
                "categories": categories,
                "positive_categories": positive_categories,
                "positive_count": _candidate_positive_count(categories),
                "strong_risk": _candidate_has_strong_risk(
                    categories=categories,
                    total_risk_count=total_risk_count,
                    predicted_rank=predicted_rank,
                    popularity=popularity,
                    odds=odds,
                    score_gap_to_top5=score_gap,
                ),
                "total_risk_count": total_risk_count,
            }
        )
    return candidates


def _matches_rule(
    candidate: dict[str, Any],
    rule_name: str,
    *,
    min_positive_signals_override: int | None = None,
) -> bool:
    rule = RULE_DEFINITIONS[rule_name]
    min_rank, max_rank = rule["rank_range"]
    if not (min_rank <= candidate["predicted_rank"] <= max_rank):
        return False
    if candidate["strong_risk"]:
        return False
    if "popularity_max" in rule:
        if candidate["popularity"] is None or candidate["popularity"] > rule["popularity_max"]:
            return False
    if "odds_min" in rule:
        if candidate["odds"] is None or candidate["odds"] < rule["odds_min"]:
            return False
    if "odds_max" in rule:
        if candidate["odds"] is None or candidate["odds"] > rule["odds_max"]:
            return False

    positive_count_required = min_positive_signals_override if min_positive_signals_override is not None else rule.get("min_positive", 0)
    if candidate["positive_count"] < positive_count_required:
        return False

    category_set = set(candidate["categories"])
    if rule_name == "rank7_8_popular_racelevel":
        return bool({"RACE_LEVEL_UNDERESTIMATED", "RECENT_FORM_UNDERESTIMATED"} & category_set)
    if rule_name == "rank7_8_multi_signal":
        return True
    if rule_name == "popular_safety_net":
        return True
    if rule_name == "race_level_safety_net":
        return "RACE_LEVEL_UNDERESTIMATED" in category_set
    if rule_name == "odds_undervalued_limited":
        return "ODDS_UNDERESTIMATED" in category_set
    if rule_name == "conservative_rank7_only":
        return True
    return False


def _pick_candidate(
    candidates: list[dict[str, Any]],
    rule_name: str,
    *,
    min_positive_signals_override: int | None = None,
) -> dict[str, Any] | None:
    matched = [
        candidate
        for candidate in candidates
        if _matches_rule(candidate, rule_name, min_positive_signals_override=min_positive_signals_override)
    ]
    if not matched:
        return None
    matched.sort(
        key=lambda candidate: (
            candidate["predicted_rank"],
            -candidate["positive_count"],
            candidate["popularity"] if candidate["popularity"] is not None else 999,
            candidate["odds"] if candidate["odds"] is not None else 999.0,
            candidate["score_gap_to_top5"] if candidate["score_gap_to_top5"] is not None else 999.0,
            candidate["horse_no"],
        )
    )
    return matched[0]


def _simulate_replacement(
    baseline_top5: list[int],
    result_top3: list[int],
    candidate: dict[str, Any],
    ordered_scores: list[HorseScore],
) -> dict[str, Any]:
    if len(ordered_scores) < 5:
        return {
            "simulated_top5": baseline_top5,
            "replaced_horse_no": None,
            "delta": 0,
            "candidate_in_top3": False,
            "replaced_in_top3": False,
            "better": False,
            "worse": False,
        }
    replaced = ordered_scores[4]
    replaced_horse_no = replaced.horse_no
    if candidate["horse_no"] in baseline_top5:
        return {
            "simulated_top5": baseline_top5,
            "replaced_horse_no": None,
            "delta": 0,
            "candidate_in_top3": candidate["horse_no"] in result_top3,
            "replaced_in_top3": False,
            "better": False,
            "worse": False,
        }
    simulated_top5 = list(baseline_top5[:4]) + [candidate["horse_no"]]
    baseline_count = _top5_capture_count(baseline_top5, result_top3)
    simulated_count = _top5_capture_count(simulated_top5, result_top3)
    delta = simulated_count - baseline_count
    return {
        "simulated_top5": simulated_top5,
        "replaced_horse_no": replaced_horse_no,
        "delta": delta,
        "candidate_in_top3": candidate["horse_no"] in result_top3,
        "replaced_in_top3": replaced_horse_no in result_top3,
        "better": delta > 0,
        "worse": delta < 0,
    }


def _best_rule_key(rule_result: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        rule_result["safe_score"],
        rule_result["improvement"],
        -rule_result["worse_race_count"],
        -rule_result["replaced_top3_count"],
    )


def run_deep_miss_rule_simulation(
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
    max_rank: int = 10,
    include_rank13: bool = False,
    min_positive_signals: int | None = None,
) -> dict[str, object]:
    resolved_baseline_mode = _resolve_baseline_mode(baseline_mode, scoring_mode)
    baseline_custom_weights = _baseline_custom_weights(
        resolved_baseline_mode,
        pedigree_weight,
        race_level_weight,
        pace_weight,
    )
    warnings: list[str] = []
    predictions, prediction_warnings = _collect_predictions_in_period(predictions_dir, from_date, to_date)
    warnings.extend(prediction_warnings)

    race_count = len(predictions)
    reviewed_race_count = 0
    total_top3_horses = 0
    baseline_captured_top3_horses = 0
    baseline_top5_winner_count = 0
    baseline_avg_values: list[int] = []
    baseline_mode_metrics_by_race: dict[str, dict[str, Any]] = {}
    rule_rows: list[dict[str, Any]] = []
    rule_state: dict[str, dict[str, Any]] = {}
    for rule_name in RULE_DEFINITIONS:
        rule_state[rule_name] = {
            "selected_candidate_count": 0,
            "recovered_top3_count": 0,
            "false_recovery_count": 0,
            "better_race_count": 0,
            "worse_race_count": 0,
            "same_race_count": 0,
            "replaced_top3_count": 0,
            "recovered_ranks": [],
            "recovered_finish_distribution": {"finish1": 0, "finish2": 0, "finish3": 0},
            "simulated_captured_top3_horses": 0,
            "simulated_top5_winner_count": 0,
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
        baseline_avg_values.append(baseline_capture_count)
        winner = result_data.result.first
        if winner in baseline_top5:
            baseline_top5_winner_count += 1

        baseline_mode_metrics_by_race[prediction.race_id] = {
            "baseline_top5": baseline_top5,
            "result_top3": result_top3,
            "ordered_scores": ordered_scores,
            "recovery_applied": bool(baseline_recovery.get("recovery_applied", False)),
        }

        candidate_pool = _build_candidate_pool(
            prediction,
            result_data,
            ordered_scores,
            race_horse_map,
            max_rank=max_rank,
            include_rank13=include_rank13,
        )
        for rule_name in RULE_DEFINITIONS:
            state = rule_state[rule_name]
            candidate = _pick_candidate(
                candidate_pool,
                rule_name,
                min_positive_signals_override=min_positive_signals,
            )
            if candidate is None:
                state["same_race_count"] += 1
                state["simulated_captured_top3_horses"] += baseline_capture_count
                if winner in baseline_top5:
                    state["simulated_top5_winner_count"] += 1
                continue

            simulation = _simulate_replacement(
                baseline_top5,
                result_top3,
                candidate,
                ordered_scores,
            )
            simulated_top5 = simulation["simulated_top5"]
            simulated_capture_count = _top5_capture_count(simulated_top5, result_top3)
            state["selected_candidate_count"] += 1
            state["simulated_captured_top3_horses"] += simulated_capture_count
            if winner in simulated_top5:
                state["simulated_top5_winner_count"] += 1
            state["recovered_ranks"].append(candidate["predicted_rank"])
            if simulation["candidate_in_top3"]:
                state["recovered_top3_count"] += 1
                if candidate["finish"] == 1:
                    state["recovered_finish_distribution"]["finish1"] += 1
                elif candidate["finish"] == 2:
                    state["recovered_finish_distribution"]["finish2"] += 1
                elif candidate["finish"] == 3:
                    state["recovered_finish_distribution"]["finish3"] += 1
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
                "rule": rule_name,
                "candidate_horse_no": candidate["horse_no"],
                "candidate_horse_name": candidate["horse_name"],
                "predicted_rank": candidate["predicted_rank"],
                "finish": candidate["finish"],
                "replaced_horse_no": simulation["replaced_horse_no"],
                "delta": simulation["delta"],
                "note": _candidate_note(candidate),
            }
            state["rows"].append(row)
            rule_rows.append(row)

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
            "description": RULE_DEFINITIONS[rule_name]["description"],
            "race_count": race_count,
            "reviewed_race_count": reviewed_race_count,
            "baseline_avg_captured_top3_per_race": baseline_avg,
            "simulated_avg_captured_top3_per_race": simulated_avg,
            "improvement": improvement,
            "baseline_capture_rate": baseline_capture_rate,
            "simulated_capture_rate": simulated_capture_rate,
            "baseline_top5_winner_rate": baseline_top5_winner_rate,
            "simulated_top5_winner_rate": round(state["simulated_top5_winner_count"] / reviewed_race_count, 4) if reviewed_race_count else 0.0,
            "selected_candidate_count": state["selected_candidate_count"],
            "recovered_top3_count": state["recovered_top3_count"],
            "false_recovery_count": state["false_recovery_count"],
            "better_race_count": state["better_race_count"],
            "worse_race_count": state["worse_race_count"],
            "same_race_count": state["same_race_count"],
            "net_better_minus_worse": state["better_race_count"] - state["worse_race_count"],
            "avg_recovered_rank": _safe_mean(state["recovered_ranks"]),
            "recovered_finish_distribution": state["recovered_finish_distribution"],
            "replaced_top3_count": state["replaced_top3_count"],
            "safe_score": safe_score,
            "rows": state["rows"],
        }

    conservative_rules = [rule_results[name] for name in CONSERVATIVE_RULES]
    aggressive_rules = [rule_results[name] for name in RULE_DEFINITIONS if name not in CONSERVATIVE_RULES]
    best_conservative = max(conservative_rules, key=_best_rule_key) if conservative_rules else None
    best_aggressive = max(aggressive_rules, key=_best_rule_key) if aggressive_rules else None
    accuracy_candidates = [
        result
        for result in rule_results.values()
        if result["improvement"] >= 0.08 and result["worse_race_count"] <= 2 and result["better_race_count"] > result["worse_race_count"]
    ]
    not_recommended = [
        result["rule"]
        for result in rule_results.values()
        if result["improvement"] < 0.05 or result["worse_race_count"] >= result["better_race_count"] or result["replaced_top3_count"] >= result["recovered_top3_count"]
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
            "min_positive_signals": min_positive_signals,
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
            "best_conservative_rule": best_conservative["rule"] if best_conservative else None,
            "best_aggressive_rule": best_aggressive["rule"] if best_aggressive else None,
            "accuracy_recommended_rules": [result["rule"] for result in sorted(accuracy_candidates, key=_best_rule_key, reverse=True)],
            "not_recommended_rules": sorted(set(not_recommended)),
        },
        "race_details": rule_rows,
        "warnings": warnings,
    }
    return report


def generate_deep_miss_rule_simulation_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    config = report["analysis_config"]
    rule_results = report["rule_results"]
    best = report["best_rule_candidates"]
    lines = [
        "# Deep Miss Rule Simulation Report",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}",
        f"- reviewed races: {summary['reviewed_race_count']}",
        f"- baseline mode: {config['baseline_mode']}",
        f"- baseline avg captured top3 per race: {summary['baseline_avg_captured_top3_per_race']:.2f}",
        "",
        "## Rule Comparison",
        "| rule | simulated avg | improvement | recovered top3 | better | worse | replaced top3 | safe_score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rule_name, result in sorted(rule_results.items()):
        lines.append(
            f"| {rule_name} | {result['simulated_avg_captured_top3_per_race']:.2f} | {result['improvement']:.2f} | "
            f"{result['recovered_top3_count']} | {result['better_race_count']} | {result['worse_race_count']} | "
            f"{result['replaced_top3_count']} | {result['safe_score']} |"
        )

    lines.extend(
        [
            "",
            "## Best Rule Candidates",
            f"- best conservative rule: {best['best_conservative_rule'] or 'なし'}",
            f"- best aggressive rule: {best['best_aggressive_rule'] or 'なし'}",
            f"- accuracy recommended: {', '.join(best['accuracy_recommended_rules']) if best['accuracy_recommended_rules'] else 'なし'}",
            f"- not recommended rules: {', '.join(best['not_recommended_rules']) if best['not_recommended_rules'] else 'なし'}",
            "",
            "## Rule Details",
        ]
    )
    for rule_name, result in sorted(rule_results.items()):
        rows = result["rows"]
        success_case = next((row for row in rows if row["finish"] in {1, 2, 3} and row["delta"] > 0), None)
        failure_case = next((row for row in rows if row["delta"] < 0), None)
        lines.extend(
            [
                f"### {rule_name}",
                f"- 条件: {result['description']}",
                "- 位置づけ: watchlist用（default rankingには非推奨）" if rule_name == "odds_undervalued_limited" else "- 位置づけ: simulation候補",
                f"- 命中候補数: {result['selected_candidate_count']}",
                f"- 実際救回Top3数: {result['recovered_top3_count']}",
                f"- 误救数: {result['false_recovery_count']}",
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
            "| race_id | baseline_top5 | result_top3 | rule | candidate | predicted_rank | finish | replaced | delta | note |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    if report["race_details"]:
        for row in report["race_details"]:
            lines.append(
                f"| {row['race_id']} | {'→'.join(str(item) for item in row['baseline_top5'])} | "
                f"{'→'.join(str(item) for item in row['result_top3'])} | {row['rule']} | "
                f"{row['candidate_horse_no']} {row['candidate_horse_name']} | {row['predicted_rank']} | "
                f"{row['finish'] if row['finish'] is not None else '-'} | {row['replaced_horse_no'] or '-'} | {row['delta']} | {row['note']} |"
            )
    else:
        lines.append("| なし | - | - | - | - | - | - | - | - | - |")

    findings: list[str] = []
    for rule_name, result in rule_results.items():
        if rule_name.startswith("rank7_8") and result["improvement"] >= 0.05:
            findings.append("rank7-8 系ルールに改善余地があり、次段階では rank7-8 限定 safety net を検討できる。")
            break
    if any(rule_results[name]["improvement"] >= 0.05 for name in ("popular_safety_net", "rank7_8_popular_racelevel")):
        findings.append("人気条件付きルールに効果があり、favorite safety check の設計余地。")
    if any(rule_results[name]["improvement"] >= 0.05 for name in ("race_level_safety_net", "rank7_8_popular_racelevel")):
        findings.append("race_level 条件付きルールが効くなら、race_level positive flag の分解検証が次候補。")
    if any(result["replaced_top3_count"] > 0 for result in rule_results.values()):
        findings.append("replaced_top3_count が出ており、低排位救援は誤置換リスクを伴う。")
    if all(result["improvement"] < 0.05 for result in rule_results.values()):
        findings.append("improvement が 0.05 未満のルールが多く、現時点では default 反映に値しない。")
    lines.extend(["", "## Findings"])
    if findings:
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.append("- 明確な有効ルールはまだ限定的で、追加サンプルの蓄積が必要。")

    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])

    return "\n".join(lines) + "\n"


def save_deep_miss_rule_simulation_json(report: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_deep_miss_rule_simulation_markdown(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

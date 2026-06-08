from __future__ import annotations

import json
from pathlib import Path

from keiba_llm_agent.config.scoring_config import DEFAULT_SCORING_WEIGHTS
from keiba_llm_agent.backtest.scoring_comparator import (
    BacktestMode,
    WeightTuningMode,
    build_marks_for_mode,
    build_marks_for_prediction_weight_tuning_mode,
    build_marks_for_weight_tuning_mode,
    compare_modes_improvement,
    evaluate_mode_against_result,
    evaluate_weight_tuning_mode_against_result,
    result_top3_list,
)
from keiba_llm_agent.parsers.netkeiba_race_parser import infer_scope_key, infer_source_from_race_id
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


DEFAULT_MODES: tuple[BacktestMode, ...] = ("base_only", "pedigree_only", "full_adjusted")
VALID_SCOPE_KEYS = {"central", "central_turf", "central_dirt", "local"}
DEFAULT_WEIGHT_TUNING_MODES: tuple[WeightTuningMode, ...] = (
    "base_only",
    "current_full",
    "conservative_full",
    "no_pace",
    "no_race_level",
    "race_level_only",
    "pace_only",
    "pedigree_only",
    "candidate_default",
    "candidate_default_recovered",
)
LOCAL_WEIGHT_TUNING_MODES: tuple[WeightTuningMode, ...] = (
    "local_candidate_default",
    "local_candidate_default_recovered",
)


def _load_prediction(path: Path) -> Prediction:
    return Prediction.model_validate_json(path.read_text(encoding="utf-8"))


def _load_result(path: Path) -> ResultData | None:
    if not path.exists():
        return None
    return ResultData.model_validate_json(path.read_text(encoding="utf-8"))


def _load_review(path: Path) -> Review | None:
    if not path.exists():
        return None
    return Review.model_validate_json(path.read_text(encoding="utf-8"))


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def _horse_no_text(marks: dict[str, int], mark: str) -> str:
    return str(marks.get(mark, 0))


def _result_text(result_data: ResultData | None) -> str:
    if result_data is None:
        return "pending"
    top3 = result_top3_list(result_data)
    return "→".join(str(item) for item in top3)


def _normalize_scope_key(scope_key: str | None) -> str | None:
    text = str(scope_key or "").strip().lower()
    if not text or text in {"all", "any"}:
        return None
    if text not in VALID_SCOPE_KEYS:
        raise ValueError(f"invalid scope_key: {scope_key}")
    return text


def _prediction_scope_key(prediction: Prediction) -> str | None:
    race_info = prediction.race_info
    explicit_scope = str(getattr(race_info, "scope_key", "") or "").strip().lower() if race_info else ""
    if explicit_scope:
        return explicit_scope

    explicit_source = str(getattr(race_info, "source", "") or "").strip().lower() if race_info else ""
    if explicit_source == "local":
        return "local"

    inferred_scope = infer_scope_key(
        prediction.race_id,
        surface=getattr(race_info, "surface", None) if race_info else None,
        course=getattr(race_info, "course", None) if race_info else None,
    )
    if inferred_scope:
        return inferred_scope

    inferred_source = infer_source_from_race_id(prediction.race_id)
    if inferred_source == "local":
        return "local"
    if explicit_source == "central" or inferred_source == "central":
        return "central"
    return None


def _prediction_matches_scope(prediction: Prediction, scope_key: str | None) -> bool:
    normalized_scope = _normalize_scope_key(scope_key)
    if normalized_scope is None:
        return True
    prediction_scope = _prediction_scope_key(prediction)
    if normalized_scope == "central":
        return prediction_scope in {"central", "central_turf", "central_dirt"}
    return prediction_scope == normalized_scope


def _collect_predictions_in_period(
    predictions_dir: Path,
    from_date: str,
    to_date: str,
    scope_key: str | None = None,
) -> tuple[list[Prediction], list[str]]:
    predictions: list[Prediction] = []
    warnings: list[str] = []
    normalized_scope = _normalize_scope_key(scope_key)
    if not predictions_dir.exists():
        return predictions, warnings

    for path in sorted(predictions_dir.glob("*.json")):
        prediction = _load_prediction(path)
        race_info = prediction.race_info
        if race_info is None or not race_info.race_date:
            warnings.append(f"prediction skipped because race_info.race_date is missing: race_id={prediction.race_id}")
            continue
        if from_date <= race_info.race_date <= to_date and _prediction_matches_scope(prediction, normalized_scope):
            predictions.append(prediction)
    return predictions, warnings


def _empty_mode_metrics() -> dict[str, float]:
    return {
        "main_mark_top3_rate": 0.0,
        "main_mark_win_rate": 0.0,
        "avg_marked_top3_count": 0.0,
        "marked_top3_rate": 0.0,
        "top5_winner_rate": 0.0,
        "avg_top5_top3_count": 0.0,
        "avg_rank_of_winner": 0.0,
        "avg_rank_of_top3_horses": 0.0,
    }


def _finalize_mode_metrics(counters: dict[str, float], reviewed_race_count: int) -> dict[str, float]:
    if reviewed_race_count <= 0:
        return _empty_mode_metrics()
    return {
        "main_mark_top3_rate": counters["main_mark_top3_hits"] / reviewed_race_count,
        "main_mark_win_rate": counters["main_mark_win_hits"] / reviewed_race_count,
        "avg_marked_top3_count": counters["marked_top3_total"] / reviewed_race_count,
        "marked_top3_rate": counters["marked_top3_total"] / (reviewed_race_count * 3),
        "top5_winner_rate": counters["top5_winner_hits"] / reviewed_race_count,
        "avg_top5_top3_count": counters["top5_top3_total"] / reviewed_race_count,
        "avg_rank_of_winner": counters["winner_rank_total"] / reviewed_race_count,
        "avg_rank_of_top3_horses": counters["top3_rank_total"] / (reviewed_race_count * 3),
    }


def _new_mode_counters(active_modes: list[str]) -> dict[str, dict[str, float]]:
    return {
        mode: {
            "main_mark_top3_hits": 0.0,
            "main_mark_win_hits": 0.0,
            "marked_top3_total": 0.0,
            "top5_winner_hits": 0.0,
            "top5_top3_total": 0.0,
            "winner_rank_total": 0.0,
            "top3_rank_total": 0.0,
        }
        for mode in active_modes
    }


def _is_heavy_adjustment_race(
    base_metrics: dict[str, object],
    full_metrics: dict[str, object],
    result_top3: list[int],
) -> bool:
    if int(full_metrics["top5_top3_count"]) < int(base_metrics["top5_top3_count"]):
        return True
    base_top5 = set(int(item) for item in base_metrics["top5"])
    full_top5 = set(int(item) for item in full_metrics["top5"])
    if any(horse_no in base_top5 and horse_no not in full_top5 for horse_no in result_top3):
        return True
    if bool(base_metrics["main_mark_top3"]) and not bool(full_metrics["main_mark_top3"]):
        return True
    return False


def run_backtest(
    from_date: str,
    to_date: str,
    predictions_dir: Path,
    results_dir: Path,
    reviews_dir: Path,
    modes: list[BacktestMode] | None = None,
    min_races: int = 1,
    enable_borderline_recovery: bool = False,
    scope_key: str | None = None,
) -> dict[str, object]:
    active_modes = modes or list(DEFAULT_MODES)
    normalized_scope = _normalize_scope_key(scope_key)
    predictions, warnings = _collect_predictions_in_period(predictions_dir, from_date, to_date, scope_key=normalized_scope)
    race_count = len(predictions)
    if race_count < min_races:
        warnings.append(f"race_count below min_races: {race_count} < {min_races}")

    reviewed_race_count = 0
    total_stake = 0
    total_return = 0
    payout_warning_count = 0
    race_details: list[dict[str, object]] = []

    mode_counters = _new_mode_counters(active_modes)

    better_races: list[str] = []
    worse_races: list[str] = []
    pace_level_heavy_races: list[str] = []

    for prediction in predictions:
        result_data = _load_result(results_dir / f"{prediction.race_id}.json")
        review = _load_review(reviews_dir / f"{prediction.race_id}.json")
        is_reviewed = review is not None and result_data is not None
        if review is None:
            warnings.append(f"review missing for race_id={prediction.race_id}")
        if result_data is None:
            warnings.append(f"result missing for race_id={prediction.race_id}")

        if is_reviewed:
            reviewed_race_count += 1
            total_stake += review.hit_summary.total_stake
            total_return += review.hit_summary.total_return
            if getattr(review, "payout_warning", False):
                payout_warning_count += 1

        detail: dict[str, object] = {
            "race_id": prediction.race_id,
            "race_name": prediction.race_info.race_name if prediction.race_info else prediction.race_id,
            "status": "reviewed" if is_reviewed else "pending",
            "result_top3": result_top3_list(result_data) if result_data is not None else [],
        }
        mode_results: dict[str, dict[str, object]] = {}
        if is_reviewed:
            for mode in active_modes:
                evaluated = evaluate_mode_against_result(
                    prediction,
                    result_data,
                    mode,
                    enable_borderline_recovery=enable_borderline_recovery,
                )
                mode_results[mode] = evaluated
                detail[f"{mode}_marks"] = evaluated["marks"]
                detail[f"{mode}_marked_top3_count"] = evaluated["marked_top3_count"]
                counters = mode_counters[mode]
                counters["main_mark_top3_hits"] += 1 if evaluated["main_mark_top3"] else 0
                counters["main_mark_win_hits"] += 1 if evaluated["main_mark_win"] else 0
                counters["marked_top3_total"] += int(evaluated["marked_top3_count"])
                counters["top5_winner_hits"] += 1 if evaluated["top5_contains_winner"] else 0
                counters["top5_top3_total"] += int(evaluated["top5_top3_count"])
                counters["winner_rank_total"] += float(evaluated["winner_rank"])
                counters["top3_rank_total"] += float(sum(int(rank) for rank in evaluated["top3_ranks"]))
            if "base_only" in mode_results and "full_adjusted" in mode_results:
                improvement = compare_modes_improvement(mode_results["base_only"], mode_results["full_adjusted"])
            else:
                improvement = "same"
            detail["improvement"] = improvement
            if improvement == "better":
                better_races.append(prediction.race_id)
            elif improvement == "worse":
                worse_races.append(prediction.race_id)
            if "base_only" in mode_results and "full_adjusted" in mode_results and _is_heavy_adjustment_race(
                mode_results["base_only"],
                mode_results["full_adjusted"],
                detail["result_top3"],
            ):
                pace_level_heavy_races.append(prediction.race_id)
        else:
            for mode in active_modes:
                detail[f"{mode}_marks"] = build_marks_for_mode(prediction.horse_scores, mode)
                detail[f"{mode}_marked_top3_count"] = None
            detail["improvement"] = "pending"
        race_details.append(detail)

    pending_race_count = race_count - reviewed_race_count
    mode_metrics = {
        mode: _finalize_mode_metrics(mode_counters[mode], reviewed_race_count if reviewed_race_count > 0 else 0)
        for mode in active_modes
    }
    roi = (total_return / total_stake) if total_stake > 0 else 0.0
    roi_reliable = payout_warning_count == 0
    if not roi_reliable:
        warnings.append("Some payout data missing. ROI may be unreliable.")

    return {
        "period": {"from": from_date, "to": to_date},
        "scope_key": normalized_scope,
        "race_count": race_count,
        "reviewed_race_count": reviewed_race_count,
        "pending_race_count": pending_race_count,
        "total_stake": total_stake,
        "total_return": total_return,
        "roi": roi,
        "roi_reliable": roi_reliable,
        "modes": mode_metrics,
        "race_details": race_details,
        "findings": {
            "better_races": better_races,
            "worse_races": worse_races,
            "pace_or_race_level_heavy_races": sorted(set(pace_level_heavy_races)),
        },
        "warnings": warnings,
    }


def _mode_improvement_note(
    base_metrics: dict[str, object],
    candidate_metrics: dict[str, object],
) -> str:
    return compare_modes_improvement(base_metrics, candidate_metrics)


def _best_mode_for_race(
    mode_results: dict[str, dict[str, object]],
) -> tuple[str, str]:
    if "base_only" not in mode_results:
        return "base_only", "pending"

    def sort_key(item: tuple[str, dict[str, object]]) -> tuple[int, int, int, int, str]:
        mode_name, metrics = item
        return (
            int(metrics["main_mark_top3"]),
            int(metrics["main_mark_win"]),
            int(metrics["marked_top3_count"]),
            int(metrics["top5_top3_count"]),
            mode_name,
        )

    best_mode, _ = max(mode_results.items(), key=sort_key)
    note = _mode_improvement_note(mode_results["base_only"], mode_results[best_mode])
    return best_mode, note


def _best_mode_summary(mode_metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    if not mode_metrics:
        return {
            "best_main_mark_top3_rate": "none",
            "best_avg_marked_top3_count": "none",
            "best_top5_winner_rate": "none",
            "safest_baseline": "none",
            "accuracy_recommended_mode": "none",
            "accuracy_recommended_reason": "no reviewed races",
            "overall_recommended_mode": "none",
            "overall_recommended_reason": "no reviewed races",
        }

    def best_for(metric: str) -> str:
        return max(
            mode_metrics.items(),
            key=lambda item: (item[1][metric], item[1]["avg_marked_top3_count"], item[1]["top5_winner_rate"], item[0]),
        )[0]

    def overall_sort_key(item: tuple[str, dict[str, float]]) -> tuple[float, float, float, float, float, str]:
        mode_name, metrics = item
        return (
            -float(metrics.get("worse_race_count", 0)),
            float(metrics["avg_marked_top3_count"]),
            float(metrics["top5_winner_rate"]),
            float(metrics["avg_top5_top3_count"]),
            float(metrics["main_mark_top3_rate"]),
            mode_name,
        )

    safest_baseline = "base_only" if "base_only" in mode_metrics else max(mode_metrics.items(), key=overall_sort_key)[0]

    base_metrics = mode_metrics.get("base_only")
    accuracy_recommended_mode = safest_baseline
    accuracy_recommended_reason = (
        "base_only を安全基線として採用。改善条件を満たす上位モードはまだ限定的。"
    )
    if base_metrics is not None:
        eligible_modes: list[tuple[str, dict[str, float], float]] = []
        for mode_name, metrics in mode_metrics.items():
            if mode_name == "base_only":
                continue
            avg_gain = float(metrics["avg_marked_top3_count"]) - float(base_metrics["avg_marked_top3_count"])
            top5_not_down = float(metrics["top5_winner_rate"]) >= float(base_metrics["top5_winner_rate"])
            worse_count = int(metrics.get("worse_race_count", 0))
            sample_count = (
                int(metrics.get("better_race_count", 0))
                + worse_count
                + int(metrics.get("same_race_count", 0))
            )
            worse_limit = max(2, int(sample_count * 0.05))
            worse_ok = worse_count <= worse_limit
            better_ok = int(metrics.get("better_race_count", 0)) > worse_count
            if avg_gain >= 0.08 and top5_not_down and worse_ok and better_ok:
                eligible_modes.append((mode_name, metrics, avg_gain))

        if eligible_modes:
            accuracy_recommended_mode, accuracy_metrics, accuracy_avg_gain = max(
                eligible_modes,
                key=lambda item: (
                    item[2],
                    float(item[1]["top5_winner_rate"]) - float(base_metrics["top5_winner_rate"]),
                    float(item[1]["avg_top5_top3_count"]) - float(base_metrics["avg_top5_top3_count"]),
                    float(item[1]["main_mark_top3_rate"]) - float(base_metrics["main_mark_top3_rate"]),
                    -float(item[1].get("worse_race_count", 0)),
                    float(item[1].get("better_race_count", 0)),
                    item[0],
                ),
            )
            accuracy_recommended_reason = (
                f"印内Top3平均が {base_metrics['avg_marked_top3_count']:.2f} → "
                f"{accuracy_metrics['avg_marked_top3_count']:.2f} に改善し、"
                f"Top5 winner率も {_format_rate(base_metrics['top5_winner_rate'])} → "
                f"{_format_rate(accuracy_metrics['top5_winner_rate'])} に改善。"
                f"better={int(accuracy_metrics.get('better_race_count', 0))} / "
                f"worse={int(accuracy_metrics.get('worse_race_count', 0))} で副作用も限定的。"
            )

    if accuracy_recommended_mode != safest_baseline:
        overall_recommended_mode = accuracy_recommended_mode
        overall_recommended_reason = accuracy_recommended_reason
    else:
        overall_recommended_mode = safest_baseline
        if safest_baseline == "base_only" and base_metrics is not None:
            overall_recommended_reason = (
                f"base_only は worse=0 を維持しつつ、印内Top3平均={base_metrics['avg_marked_top3_count']:.2f}、"
                f"Top5 winner率={_format_rate(base_metrics['top5_winner_rate'])} の安全基線。"
            )
        else:
            overall_metrics = mode_metrics[overall_recommended_mode]
            overall_recommended_reason = (
                f"worse={int(overall_metrics.get('worse_race_count', 0))} を優先的に抑えつつ、"
                f"印内Top3平均={overall_metrics['avg_marked_top3_count']:.2f}、"
                f"Top5 winner率={_format_rate(overall_metrics['top5_winner_rate'])}、"
                f"Top5 Top3平均={overall_metrics['avg_top5_top3_count']:.2f} が相対的に良好。"
            )
    return {
        "best_main_mark_top3_rate": best_for("main_mark_top3_rate"),
        "best_avg_marked_top3_count": best_for("avg_marked_top3_count"),
        "best_top5_winner_rate": best_for("top5_winner_rate"),
        "safest_baseline": safest_baseline,
        "accuracy_recommended_mode": accuracy_recommended_mode,
        "accuracy_recommended_reason": accuracy_recommended_reason,
        "overall_recommended_mode": overall_recommended_mode,
        "overall_recommended_reason": overall_recommended_reason,
    }


def run_backtest_weights(
    from_date: str,
    to_date: str,
    predictions_dir: Path,
    results_dir: Path,
    reviews_dir: Path,
    modes: list[WeightTuningMode] | None = None,
    min_races: int = 1,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    enable_borderline_recovery: bool = False,
    scope_key: str | None = None,
) -> dict[str, object]:
    normalized_scope = _normalize_scope_key(scope_key)
    default_modes = (
        (*DEFAULT_WEIGHT_TUNING_MODES, *LOCAL_WEIGHT_TUNING_MODES)
        if modes is None and normalized_scope == "local"
        else DEFAULT_WEIGHT_TUNING_MODES
    )
    active_modes = list(dict.fromkeys(modes or default_modes))
    evaluation_modes = list(active_modes)
    if any(mode != "base_only" for mode in active_modes) and "base_only" not in evaluation_modes:
        evaluation_modes.append("base_only")
    custom_weights: tuple[float, float, float] | None = None
    if pedigree_weight is not None or race_level_weight is not None or pace_weight is not None:
        custom_weights = (
            DEFAULT_SCORING_WEIGHTS["pedigree_weight"] if pedigree_weight is None else pedigree_weight,
            DEFAULT_SCORING_WEIGHTS["race_level_weight"] if race_level_weight is None else race_level_weight,
            DEFAULT_SCORING_WEIGHTS["pace_weight"] if pace_weight is None else pace_weight,
        )
        if "custom" not in active_modes:
            active_modes.append("custom")
        if "custom" not in evaluation_modes:
            evaluation_modes.append("custom")

    predictions, warnings = _collect_predictions_in_period(predictions_dir, from_date, to_date, scope_key=normalized_scope)
    race_count = len(predictions)
    if race_count < min_races:
        warnings.append(f"race_count below min_races: {race_count} < {min_races}")

    reviewed_race_count = 0
    total_stake = 0
    total_return = 0
    payout_warning_count = 0
    race_details: list[dict[str, object]] = []
    mode_counters = _new_mode_counters(evaluation_modes)

    better_counts = {mode: 0 for mode in active_modes if mode != "base_only"}
    worse_counts = {mode: 0 for mode in active_modes if mode != "base_only"}
    same_counts = {mode: 0 for mode in active_modes if mode != "base_only"}
    race_level_improved: list[str] = []
    race_level_worsened: list[str] = []
    pace_improved: list[str] = []
    pace_worsened: list[str] = []
    heavy_races: list[str] = []

    for prediction in predictions:
        result_data = _load_result(results_dir / f"{prediction.race_id}.json")
        review = _load_review(reviews_dir / f"{prediction.race_id}.json")
        is_reviewed = review is not None and result_data is not None
        if review is None:
            warnings.append(f"review missing for race_id={prediction.race_id}")
        if result_data is None:
            warnings.append(f"result missing for race_id={prediction.race_id}")

        if is_reviewed:
            reviewed_race_count += 1
            total_stake += review.hit_summary.total_stake
            total_return += review.hit_summary.total_return
            if getattr(review, "payout_warning", False):
                payout_warning_count += 1

        detail: dict[str, object] = {
            "race_id": prediction.race_id,
            "race_name": prediction.race_info.race_name if prediction.race_info else prediction.race_id,
            "status": "reviewed" if is_reviewed else "pending",
            "result_top3": result_top3_list(result_data) if result_data is not None else [],
        }
        mode_results: dict[str, dict[str, object]] = {}
        if is_reviewed:
            for mode in evaluation_modes:
                evaluated = evaluate_weight_tuning_mode_against_result(
                    prediction,
                    result_data,
                    mode,
                    custom_weights=custom_weights,
                    enable_borderline_recovery=enable_borderline_recovery,
                )
                mode_results[mode] = evaluated
                if mode in active_modes:
                    detail[f"{mode}_marks"] = evaluated["marks"]
                    detail[f"{mode}_marked_top3_count"] = evaluated["marked_top3_count"]
                    if mode == "candidate_default_recovered":
                        detail["recovery_applied"] = bool(evaluated.get("recovery_applied", False))
                        detail["recovery_cases"] = evaluated.get("recovery_cases", [])
                        detail["recovered_horse_no"] = evaluated.get("recovered_horse_no")
                        detail["recovered_from_rank"] = evaluated.get("recovered_from_rank")
                        detail["recovered_to_rank"] = evaluated.get("recovered_to_rank")
                counters = mode_counters[mode]
                counters["main_mark_top3_hits"] += 1 if evaluated["main_mark_top3"] else 0
                counters["main_mark_win_hits"] += 1 if evaluated["main_mark_win"] else 0
                counters["marked_top3_total"] += int(evaluated["marked_top3_count"])
                counters["top5_winner_hits"] += 1 if evaluated["top5_contains_winner"] else 0
                counters["top5_top3_total"] += int(evaluated["top5_top3_count"])
                counters["winner_rank_total"] += float(evaluated["winner_rank"])
                counters["top3_rank_total"] += float(sum(int(rank) for rank in evaluated["top3_ranks"]))

            base_metrics = mode_results.get("base_only")
            if base_metrics is not None:
                for mode in active_modes:
                    if mode == "base_only":
                        continue
                    comparison = compare_modes_improvement(base_metrics, mode_results[mode])
                    if comparison == "better":
                        better_counts[mode] += 1
                    elif comparison == "worse":
                        worse_counts[mode] += 1
                    else:
                        same_counts[mode] += 1

            if "no_race_level" in mode_results and "current_full" in mode_results:
                race_level_effect = compare_modes_improvement(mode_results["no_race_level"], mode_results["current_full"])
                if race_level_effect == "better":
                    race_level_improved.append(prediction.race_id)
                elif race_level_effect == "worse":
                    race_level_worsened.append(prediction.race_id)

            if "no_pace" in mode_results and "current_full" in mode_results:
                pace_effect = compare_modes_improvement(mode_results["no_pace"], mode_results["current_full"])
                if pace_effect == "better":
                    pace_improved.append(prediction.race_id)
                elif pace_effect == "worse":
                    pace_worsened.append(prediction.race_id)

            if base_metrics is not None and "current_full" in mode_results:
                if _is_heavy_adjustment_race(base_metrics, mode_results["current_full"], detail["result_top3"]):
                    heavy_races.append(prediction.race_id)

            visible_mode_results = {mode: mode_results[mode] for mode in active_modes if mode in mode_results}
            best_mode, best_note = _best_mode_for_race(visible_mode_results)
            detail["best_mode"] = best_mode
            detail["best_mode_note"] = best_note
        else:
            for mode in active_modes:
                if mode == "base_only":
                    detail[f"{mode}_marks"] = build_marks_for_mode(prediction.horse_scores, "base_only")
                else:
                    detail[f"{mode}_marks"] = build_marks_for_prediction_weight_tuning_mode(
                        prediction,
                        mode,
                        custom_weights=custom_weights,
                        enable_borderline_recovery=enable_borderline_recovery,
                    )
                detail[f"{mode}_marked_top3_count"] = None
            detail["recovery_applied"] = False
            detail["recovery_cases"] = []
            detail["recovered_horse_no"] = None
            detail["recovered_from_rank"] = None
            detail["recovered_to_rank"] = None
            detail["best_mode"] = "pending"
            detail["best_mode_note"] = "pending"
        race_details.append(detail)

    pending_race_count = race_count - reviewed_race_count
    mode_metrics = {
        mode: _finalize_mode_metrics(mode_counters[mode], reviewed_race_count if reviewed_race_count > 0 else 0)
        for mode in active_modes
    }
    for mode in active_modes:
        if mode == "base_only":
            mode_metrics[mode]["better_race_count"] = 0
            mode_metrics[mode]["worse_race_count"] = 0
            mode_metrics[mode]["same_race_count"] = reviewed_race_count
            continue
        mode_metrics[mode]["better_race_count"] = better_counts[mode]
        mode_metrics[mode]["worse_race_count"] = worse_counts[mode]
        mode_metrics[mode]["same_race_count"] = same_counts[mode]
    roi = (total_return / total_stake) if total_stake > 0 else 0.0
    roi_reliable = payout_warning_count == 0
    if not roi_reliable:
        warnings.append("Some payout data missing. ROI may be unreliable.")

    return {
        "period": {"from": from_date, "to": to_date},
        "scope_key": normalized_scope,
        "race_count": race_count,
        "reviewed_race_count": reviewed_race_count,
        "pending_race_count": pending_race_count,
        "total_stake": total_stake,
        "total_return": total_return,
        "roi": roi,
        "roi_reliable": roi_reliable,
        "weights": {
            "pedigree_weight": custom_weights[0] if custom_weights is not None else None,
            "race_level_weight": custom_weights[1] if custom_weights is not None else None,
            "pace_weight": custom_weights[2] if custom_weights is not None else None,
        },
        "modes": mode_metrics,
        "race_details": race_details,
        "best_mode_summary": _best_mode_summary(mode_metrics),
        "findings": {
            "race_level_improved_races": sorted(set(race_level_improved)),
            "race_level_worsened_races": sorted(set(race_level_worsened)),
            "pace_improved_races": sorted(set(pace_improved)),
            "pace_worsened_races": sorted(set(pace_worsened)),
            "pace_or_race_level_heavy_races": sorted(set(heavy_races)),
            "better_race_counts": better_counts,
            "worse_race_counts": worse_counts,
            "same_race_counts": same_counts,
        },
        "warnings": warnings,
    }


def generate_backtest_markdown(report: dict[str, object], modes: list[BacktestMode] | None = None) -> str:
    active_modes = modes or list(report["modes"].keys())
    lines = [
        "# Backtest Report",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}",
        f"- 対象レース数: {report['race_count']}",
        f"- review済みレース数: {report['reviewed_race_count']}",
        f"- pendingレース数: {report['pending_race_count']}",
        f"- total_stake: {report['total_stake']}",
        f"- total_return: {report['total_return']}",
        f"- ROI: {(_format_rate(report['roi']) if report['total_stake'] else '0.0%')}{' (暫定)' if not report.get('roi_reliable', True) else ''}",
        "",
        "## Mode Comparison",
        "※指標は review済みレースのみで計算。",
        "| mode | ◎Top3率 | ◎Win率 | 印内Top3平均 | Top5 winner率 | Top5 Top3平均 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in active_modes:
        metrics = report["modes"][mode]
        lines.append(
            f"| {mode} | {_format_rate(metrics['main_mark_top3_rate'])} | {_format_rate(metrics['main_mark_win_rate'])} | "
            f"{metrics['avg_marked_top3_count']:.2f} | {_format_rate(metrics['top5_winner_rate'])} | {metrics['avg_top5_top3_count']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Race Details",
            "| race_id | result | base ◎ | pedigree ◎ | full ◎ | base印内Top3 | full印内Top3 | improvement |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for detail in report["race_details"]:
        lines.append(
            f"| {detail['race_id']} | {_result_text(ResultData.model_validate({'race_id': detail['race_id'], 'result': {'1st': detail['result_top3'][0], '2nd': detail['result_top3'][1], '3rd': detail['result_top3'][2]}, 'payouts': []}) if detail['result_top3'] else None)} | "
            f"{_horse_no_text(detail.get('base_only_marks', {}), '◎')} | "
            f"{_horse_no_text(detail.get('pedigree_only_marks', {}), '◎')} | "
            f"{_horse_no_text(detail.get('full_adjusted_marks', {}), '◎')} | "
            f"{detail.get('base_only_marked_top3_count', '-')} | "
            f"{detail.get('full_adjusted_marked_top3_count', '-')} | "
            f"{detail['improvement']} |"
        )

    lines.extend(["", "## Findings"])
    findings = report["findings"]
    lines.append(f"- full_adjusted が改善した race: {', '.join(findings['better_races']) if findings['better_races'] else 'なし'}")
    lines.append(f"- full_adjusted が悪化した race: {', '.join(findings['worse_races']) if findings['worse_races'] else 'なし'}")
    lines.append(
        f"- race_level / pace 補正が効きすぎた可能性がある race: "
        f"{', '.join(findings['pace_or_race_level_heavy_races']) if findings['pace_or_race_level_heavy_races'] else 'なし'}"
    )
    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def generate_weight_tuning_markdown(
    report: dict[str, object],
    modes: list[WeightTuningMode] | None = None,
) -> str:
    active_modes = modes or list(report["modes"].keys())
    lines = [
        "# Adjustment Weight Tuning Report",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}",
        f"- 対象レース数: {report['race_count']}",
        f"- review済みレース数: {report['reviewed_race_count']}",
        f"- pendingレース数: {report['pending_race_count']}",
        f"- total_stake: {report['total_stake']}",
        f"- total_return: {report['total_return']}",
        f"- ROI: {(_format_rate(report['roi']) if report['total_stake'] else '0.0%')}{' (暫定)' if not report.get('roi_reliable', True) else ''}",
        "",
        "## Mode Comparison",
        "※指標は review済みレースのみで計算。",
        "| mode | ◎Top3率 | ◎Win率 | 印内Top3平均 | Top5 winner率 | Top5 Top3平均 | better | worse | same |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    better_counts = report["findings"]["better_race_counts"]
    worse_counts = report["findings"]["worse_race_counts"]
    same_counts = report["findings"]["same_race_counts"]
    for mode in active_modes:
        metrics = report["modes"][mode]
        better = "-" if mode == "base_only" else better_counts.get(mode, 0)
        worse = "-" if mode == "base_only" else worse_counts.get(mode, 0)
        same = "-" if mode == "base_only" else same_counts.get(mode, 0)
        lines.append(
            f"| {mode} | {_format_rate(metrics['main_mark_top3_rate'])} | {_format_rate(metrics['main_mark_win_rate'])} | "
            f"{metrics['avg_marked_top3_count']:.2f} | {_format_rate(metrics['top5_winner_rate'])} | {metrics['avg_top5_top3_count']:.2f} | "
            f"{better} | {worse} | {same} |"
        )

    best_summary = report["best_mode_summary"]
    lines.extend(
        [
            "",
            "## Best Mode",
            f"- ◎Top3率 best: {best_summary['best_main_mark_top3_rate']}",
            f"- 印内Top3平均 best: {best_summary['best_avg_marked_top3_count']}",
            f"- Top5 winner率 best: {best_summary['best_top5_winner_rate']}",
            f"- safest baseline: {best_summary['safest_baseline']}",
            f"- accuracy recommended: {best_summary['accuracy_recommended_mode']}",
            f"- overall recommended mode: {best_summary['overall_recommended_mode']}",
            f"- recommended reason: {best_summary['overall_recommended_reason']}",
            "",
            "## Race Details",
            "| race_id | result | base ◎ | current ◎ | conservative ◎ | no_pace ◎ | no_raceLv ◎ | cand_def_rec ◎ | recovered horse | best mode note |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for detail in report["race_details"]:
        recovered_text = "-"
        if detail.get("recovery_applied") and detail.get("recovered_horse_no") is not None:
            recovered_text = (
                f"{detail['recovered_horse_no']}"
                f" ({detail.get('recovered_from_rank', '-')}"
                f"→{detail.get('recovered_to_rank', '-')})"
            )
        lines.append(
            f"| {detail['race_id']} | {_result_text(ResultData.model_validate({'race_id': detail['race_id'], 'result': {'1st': detail['result_top3'][0], '2nd': detail['result_top3'][1], '3rd': detail['result_top3'][2]}, 'payouts': []}) if detail['result_top3'] else None)} | "
            f"{_horse_no_text(detail.get('base_only_marks', {}), '◎')} | "
            f"{_horse_no_text(detail.get('current_full_marks', {}), '◎')} | "
            f"{_horse_no_text(detail.get('conservative_full_marks', {}), '◎')} | "
            f"{_horse_no_text(detail.get('no_pace_marks', {}), '◎')} | "
            f"{_horse_no_text(detail.get('no_race_level_marks', {}), '◎')} | "
            f"{_horse_no_text(detail.get('candidate_default_recovered_marks', {}), '◎')} | "
            f"{recovered_text} | "
            f"{detail.get('best_mode_note', '-')} |"
        )

    findings = report["findings"]
    lines.extend(
        [
            "",
            "## Findings",
            f"- race_level が改善した race: {', '.join(findings['race_level_improved_races']) if findings['race_level_improved_races'] else 'なし'}",
            f"- race_level が悪化させた race: {', '.join(findings['race_level_worsened_races']) if findings['race_level_worsened_races'] else 'なし'}",
            f"- pace が改善した race: {', '.join(findings['pace_improved_races']) if findings['pace_improved_races'] else 'なし'}",
            f"- pace が悪化させた race: {', '.join(findings['pace_worsened_races']) if findings['pace_worsened_races'] else 'なし'}",
            f"- current_full が過補正気味かどうか: {', '.join(findings['pace_or_race_level_heavy_races']) if findings['pace_or_race_level_heavy_races'] else 'なし'}",
        ]
    )
    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def save_backtest_json(report: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_backtest_markdown(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

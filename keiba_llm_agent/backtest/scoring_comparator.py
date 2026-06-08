from __future__ import annotations

from typing import Any, Literal

from keiba_llm_agent.config.scoring_config import (
    DEFAULT_CONDITIONAL_WEIGHT_PROFILE,
    DEFAULT_SCORING_MODE,
    DEFAULT_SCORING_WEIGHTS,
    LOCAL_SCORING_WEIGHTS,
    effective_scoring_weights,
)
from keiba_llm_agent.scoring.borderline_recovery import apply_top5_borderline_recovery
from keiba_llm_agent.schemas.prediction import HorseScore, Prediction
from keiba_llm_agent.schemas.result import ResultData


MARK_LABELS = ("◎", "○", "▲", "△", "☆")
BacktestMode = Literal["base_only", "pedigree_only", "full_adjusted"]
WeightTuningMode = Literal[
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
    "local_candidate_default",
    "local_candidate_default_recovered",
    "custom",
]

WEIGHT_TUNING_MODE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "base_only": (0.0, 0.0, 0.0),
    "current_full": (1.0, 1.0, 1.0),
    "conservative_full": (0.5, 0.5, 0.5),
    "no_pace": (1.0, 1.0, 0.0),
    "no_race_level": (1.0, 0.0, 1.0),
    "race_level_only": (0.0, 1.0, 0.0),
    "pace_only": (0.0, 0.0, 1.0),
    "pedigree_only": (1.0, 0.0, 0.0),
    "candidate_default": (
        DEFAULT_SCORING_WEIGHTS["pedigree_weight"],
        DEFAULT_SCORING_WEIGHTS["race_level_weight"],
        DEFAULT_SCORING_WEIGHTS["pace_weight"],
    ),
    "candidate_default_recovered": (
        DEFAULT_SCORING_WEIGHTS["pedigree_weight"],
        DEFAULT_SCORING_WEIGHTS["race_level_weight"],
        DEFAULT_SCORING_WEIGHTS["pace_weight"],
    ),
    "local_candidate_default": (
        LOCAL_SCORING_WEIGHTS["pedigree_weight"],
        LOCAL_SCORING_WEIGHTS["race_level_weight"],
        LOCAL_SCORING_WEIGHTS["pace_weight"],
    ),
    "local_candidate_default_recovered": (
        LOCAL_SCORING_WEIGHTS["pedigree_weight"],
        LOCAL_SCORING_WEIGHTS["race_level_weight"],
        LOCAL_SCORING_WEIGHTS["pace_weight"],
    ),
}


def calculate_weighted_score(
    horse_score: HorseScore,
    pedigree_weight: float = 1.0,
    race_level_weight: float = 1.0,
    pace_weight: float = 1.0,
) -> float:
    return round(
        horse_score.base_total_score
        + horse_score.pedigree_adjustment.pedigree_adjustment * pedigree_weight
        + horse_score.race_level_adjustment.adjustment * race_level_weight
        + horse_score.pace_adjustment.adjustment * pace_weight,
        1,
    )


def score_for_mode(horse_score: HorseScore, mode: BacktestMode) -> float:
    if mode == "base_only":
        return horse_score.base_total_score
    if mode == "pedigree_only":
        return calculate_weighted_score(horse_score, pedigree_weight=1.0, race_level_weight=0.0, pace_weight=0.0)
    return calculate_weighted_score(horse_score, pedigree_weight=1.0, race_level_weight=1.0, pace_weight=1.0)


def score_for_weight_tuning_mode(
    horse_score: HorseScore,
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
) -> float:
    if mode == "custom":
        if custom_weights is None:
            raise ValueError("custom mode requires custom_weights")
        return calculate_weighted_score(
            horse_score,
            pedigree_weight=custom_weights[0],
            race_level_weight=custom_weights[1],
            pace_weight=custom_weights[2],
        )
    weights = WEIGHT_TUNING_MODE_WEIGHTS[mode]
    return calculate_weighted_score(
        horse_score,
        pedigree_weight=weights[0],
        race_level_weight=weights[1],
        pace_weight=weights[2],
    )


def rank_horses_for_mode(
    horse_scores: list[HorseScore],
    mode: BacktestMode,
) -> list[HorseScore]:
    return sorted(
        horse_scores,
        key=lambda item: (-score_for_mode(item, mode), item.horse_no),
    )


def rank_horses_for_weight_tuning_mode(
    horse_scores: list[HorseScore],
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
) -> list[HorseScore]:
    return sorted(
        horse_scores,
        key=lambda item: (-score_for_weight_tuning_mode(item, mode, custom_weights=custom_weights), item.horse_no),
    )


def build_marks_for_mode(
    horse_scores: list[HorseScore],
    mode: BacktestMode,
) -> dict[str, int]:
    ranked = rank_horses_for_mode(horse_scores, mode)
    marks = {label: 0 for label in MARK_LABELS}
    for index, horse_score in enumerate(ranked[: len(MARK_LABELS)]):
        marks[MARK_LABELS[index]] = horse_score.horse_no
    return marks


def build_marks_for_weight_tuning_mode(
    horse_scores: list[HorseScore],
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
) -> dict[str, int]:
    ranked = rank_horses_for_weight_tuning_mode(horse_scores, mode, custom_weights=custom_weights)
    marks = {label: 0 for label in MARK_LABELS}
    for index, horse_score in enumerate(ranked[: len(MARK_LABELS)]):
        marks[MARK_LABELS[index]] = horse_score.horse_no
    return marks


def extract_top5_for_mode(
    horse_scores: list[HorseScore],
    mode: BacktestMode,
) -> list[int]:
    return [horse_score.horse_no for horse_score in rank_horses_for_mode(horse_scores, mode)[:5]]


def extract_top5_for_weight_tuning_mode(
    horse_scores: list[HorseScore],
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
) -> list[int]:
    return [
        horse_score.horse_no
        for horse_score in rank_horses_for_weight_tuning_mode(horse_scores, mode, custom_weights=custom_weights)[:5]
    ]


def _rank_map(horse_scores: list[HorseScore], ordered_scores: list[HorseScore]) -> dict[int, int]:
    return {horse_score.horse_no: index + 1 for index, horse_score in enumerate(ordered_scores)}


def _mode_weights(
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    if mode == "custom":
        if custom_weights is None:
            raise ValueError("custom mode requires custom_weights")
        return custom_weights
    return WEIGHT_TUNING_MODE_WEIGHTS[mode]


def _prediction_mode_weights(
    prediction: Prediction,
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    if mode in {"local_candidate_default", "local_candidate_default_recovered"}:
        return _mode_weights(mode, custom_weights=custom_weights)
    if mode not in {"candidate_default", "candidate_default_recovered"}:
        return _mode_weights(mode, custom_weights=custom_weights)
    base_config = prediction.scoring_config.model_copy(
        update={
            "scoring_mode": DEFAULT_SCORING_MODE,
            "pedigree_weight": DEFAULT_SCORING_WEIGHTS["pedigree_weight"],
            "race_level_weight": DEFAULT_SCORING_WEIGHTS["race_level_weight"],
            "pace_weight": DEFAULT_SCORING_WEIGHTS["pace_weight"],
            "conditional_weight_profile": DEFAULT_CONDITIONAL_WEIGHT_PROFILE,
        }
    )
    weights = effective_scoring_weights(
        base_config,
        surface=prediction.race_info.surface if prediction.race_info else None,
        field_size=len(prediction.horse_scores),
    )
    return weights["pedigree_weight"], weights["race_level_weight"], weights["pace_weight"]


def _score_prediction_horses(
    prediction: Prediction,
    pedigree_weight: float,
    race_level_weight: float,
    pace_weight: float,
) -> list[HorseScore]:
    scored_horses: list[HorseScore] = []
    for horse_score in prediction.horse_scores:
        weighted_score = calculate_weighted_score(
            horse_score,
            pedigree_weight=pedigree_weight,
            race_level_weight=race_level_weight,
            pace_weight=pace_weight,
        )
        updated_breakdown = horse_score.score_breakdown.model_copy(
            update={
                "pedigree_weight": pedigree_weight,
                "pedigree_adjustment_weighted": round(horse_score.pedigree_adjustment.pedigree_adjustment * pedigree_weight, 1),
                "race_level_weight": race_level_weight,
                "race_level_adjustment_weighted": round(horse_score.race_level_adjustment.adjustment * race_level_weight, 1),
                "pace_weight": pace_weight,
                "pace_adjustment_weighted": round(horse_score.pace_adjustment.adjustment * pace_weight, 1),
                "borderline_recovery_bonus": 0.0,
                "total_score": weighted_score,
                "total_score_after_recovery": weighted_score,
            }
        )
        scored_horses.append(
            horse_score.model_copy(
                update={
                    "total_score": weighted_score,
                    "score_breakdown": updated_breakdown,
                }
            )
        )
    return scored_horses


def rank_prediction_for_weight_tuning_mode(
    prediction: Prediction,
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
    enable_borderline_recovery: bool = False,
) -> list[HorseScore]:
    ranked_scores, _ = rank_prediction_for_weight_tuning_mode_with_recovery(
        prediction,
        mode,
        custom_weights=custom_weights,
        enable_borderline_recovery=enable_borderline_recovery,
    )
    return ranked_scores


def rank_prediction_for_weight_tuning_mode_with_recovery(
    prediction: Prediction,
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
    enable_borderline_recovery: bool = False,
) -> tuple[list[HorseScore], dict[str, Any]]:
    pedigree_weight, race_level_weight, pace_weight = _prediction_mode_weights(
        prediction,
        mode,
        custom_weights=custom_weights,
    )
    scored_horses = _score_prediction_horses(
        prediction,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
    )
    recovery_result: dict[str, Any] = {
        "recovered_top5": [horse_score.horse_no for horse_score in sorted(scored_horses, key=lambda item: (-item.total_score, item.horse_no))[:5]],
        "recovery_applied": False,
        "recovery_cases": [],
        "adjusted_horse_scores": scored_horses,
    }
    should_apply_recovery = mode in {"candidate_default_recovered", "local_candidate_default_recovered"} or (
        enable_borderline_recovery and mode != "base_only"
    )
    if should_apply_recovery:
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
        scored_horses = recovery_result["adjusted_horse_scores"]
    return sorted(scored_horses, key=lambda item: (-item.total_score, item.horse_no)), recovery_result


def rank_prediction_for_mode(
    prediction: Prediction,
    mode: BacktestMode,
    enable_borderline_recovery: bool = False,
) -> list[HorseScore]:
    weights_by_mode: dict[BacktestMode, tuple[float, float, float]] = {
        "base_only": (0.0, 0.0, 0.0),
        "pedigree_only": (1.0, 0.0, 0.0),
        "full_adjusted": (1.0, 1.0, 1.0),
    }
    weights = weights_by_mode[mode]
    scored_horses = _score_prediction_horses(
        prediction,
        pedigree_weight=weights[0],
        race_level_weight=weights[1],
        pace_weight=weights[2],
    )
    if enable_borderline_recovery and mode != "base_only":
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
        scored_horses = recovery_result["adjusted_horse_scores"]
    return sorted(scored_horses, key=lambda item: (-item.total_score, item.horse_no))


def build_marks_for_prediction_weight_tuning_mode(
    prediction: Prediction,
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
    enable_borderline_recovery: bool = False,
) -> dict[str, int]:
    ranked, _ = rank_prediction_for_weight_tuning_mode_with_recovery(
        prediction,
        mode,
        custom_weights=custom_weights,
        enable_borderline_recovery=enable_borderline_recovery,
    )
    marks = {label: 0 for label in MARK_LABELS}
    for index, horse_score in enumerate(ranked[: len(MARK_LABELS)]):
        marks[MARK_LABELS[index]] = horse_score.horse_no
    return marks


def extract_topn_for_prediction_weight_tuning_mode(
    prediction: Prediction,
    mode: WeightTuningMode,
    top_n: int,
    custom_weights: tuple[float, float, float] | None = None,
    enable_borderline_recovery: bool = False,
) -> list[int]:
    ranked, _ = rank_prediction_for_weight_tuning_mode_with_recovery(
        prediction,
        mode,
        custom_weights=custom_weights,
        enable_borderline_recovery=enable_borderline_recovery,
    )
    return [
        horse_score.horse_no
        for horse_score in ranked[:top_n]
    ]


def result_top3_list(result_data: ResultData) -> list[int]:
    return [result_data.result.first, result_data.result.second, result_data.result.third]


def count_marked_top3(marks: dict[str, int], result_top3: list[int]) -> int:
    marked = {horse_no for horse_no in marks.values() if horse_no > 0}
    return len(marked.intersection(result_top3))


def evaluate_mode_against_result(
    prediction: Prediction,
    result_data: ResultData,
    mode: BacktestMode,
    enable_borderline_recovery: bool = False,
    top_n: int = 5,
) -> dict[str, object]:
    ordered_scores = rank_prediction_for_mode(
        prediction,
        mode,
        enable_borderline_recovery=enable_borderline_recovery,
    )
    marks = {label: 0 for label in MARK_LABELS}
    for index, horse_score in enumerate(ordered_scores[: len(MARK_LABELS)]):
        marks[MARK_LABELS[index]] = horse_score.horse_no
    ranked_top5 = [horse_score.horse_no for horse_score in ordered_scores[:top_n]]
    top3 = result_top3_list(result_data)
    main_mark = marks["◎"]
    marked_top3_count = count_marked_top3(marks, top3)
    winner = result_data.result.first
    rank_map = _rank_map(ordered_scores, ordered_scores)
    return {
        "marks": marks,
        "top5": ranked_top5,
        "main_mark_top3": main_mark in top3,
        "main_mark_win": main_mark == winner,
        "marked_top3_count": marked_top3_count,
        "top5_contains_winner": winner in ranked_top5,
        "top5_top3_count": len(set(ranked_top5).intersection(top3)),
        "winner_rank": rank_map.get(winner, len(rank_map) + 1),
        "top3_ranks": [rank_map.get(horse_no, len(rank_map) + 1) for horse_no in top3],
    }


def evaluate_weight_tuning_mode_against_result(
    prediction: Prediction,
    result_data: ResultData,
    mode: WeightTuningMode,
    custom_weights: tuple[float, float, float] | None = None,
    enable_borderline_recovery: bool = False,
    top_n: int = 5,
) -> dict[str, object]:
    ordered_scores, recovery_result = rank_prediction_for_weight_tuning_mode_with_recovery(
        prediction,
        mode,
        custom_weights=custom_weights,
        enable_borderline_recovery=enable_borderline_recovery,
    )
    marks = {label: 0 for label in MARK_LABELS}
    for index, horse_score in enumerate(ordered_scores[: len(MARK_LABELS)]):
        marks[MARK_LABELS[index]] = horse_score.horse_no
    ranked_top5 = [horse_score.horse_no for horse_score in ordered_scores[:top_n]]
    top3 = result_top3_list(result_data)
    main_mark = marks["◎"]
    marked_top3_count = count_marked_top3(marks, top3)
    winner = result_data.result.first
    rank_map = _rank_map(ordered_scores, ordered_scores)
    recovery_cases = list(recovery_result.get("recovery_cases", []))
    first_recovery = recovery_cases[0] if recovery_cases else None
    return {
        "marks": marks,
        "top5": ranked_top5,
        "main_mark_top3": main_mark in top3,
        "main_mark_win": main_mark == winner,
        "marked_top3_count": marked_top3_count,
        "top5_contains_winner": winner in ranked_top5,
        "top5_top3_count": len(set(ranked_top5).intersection(top3)),
        "winner_rank": rank_map.get(winner, len(rank_map) + 1),
        "top3_ranks": [rank_map.get(horse_no, len(rank_map) + 1) for horse_no in top3],
        "recovery_applied": bool(recovery_result.get("recovery_applied", False)),
        "recovery_cases": recovery_cases,
        "recovered_horse_no": (first_recovery.get("horse_no") if isinstance(first_recovery, dict) else None),
        "recovered_from_rank": (first_recovery.get("original_rank") if isinstance(first_recovery, dict) else None),
        "recovered_to_rank": (first_recovery.get("new_rank") if isinstance(first_recovery, dict) else None),
    }


def compare_modes_improvement(
    base_metrics: dict[str, object],
    full_metrics: dict[str, object],
) -> str:
    base_count = int(base_metrics["marked_top3_count"])
    full_count = int(full_metrics["marked_top3_count"])
    if full_count > base_count:
        return "better"
    if full_count < base_count:
        return "worse"
    base_main = bool(base_metrics["main_mark_top3"])
    full_main = bool(full_metrics["main_mark_top3"])
    if full_main and not base_main:
        return "better"
    if base_main and not full_main:
        return "worse"
    return "same"

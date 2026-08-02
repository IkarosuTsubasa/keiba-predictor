from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from keiba_llm_agent.analysis.course_geometry import (
    COURSE_GEOMETRY_FEATURE_NAMES,
    COURSE_STYLE_FEATURE_NAMES,
    build_course_geometry_features,
    build_course_style_features,
)
from keiba_llm_agent.analysis.local_course_geometry import (
    LOCAL_COURSE_GEOMETRY_FEATURE_NAMES,
    build_local_course_geometry_features,
)
from keiba_llm_agent.analysis.race_class import (
    infer_run_race_class_level,
    infer_target_race_class_level,
)
from keiba_llm_agent.schemas.prediction import (
    HorseScopeRerankerResult,
    HorseScore,
    ScopeRerankerConfigSnapshot,
    ScopeRerankerResult,
)
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RecentRun
from keiba_llm_agent.utils.margin import parse_margin_value


MODEL_VERSION = "scope_reranker_v4_20260803_place"
DEFAULT_BLEND_WEIGHT = 0.50
MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "models" / "scope_reranker"
MODEL_SPECS = {
    "central_turf": {
        "filename": "central_turf_place_v1.joblib",
        "feature_set": "course_geometry",
        "blend_weight": 0.5,
        "preserve_top_n": 1,
        "baseline_filename": "central_turf_v1.joblib",
        "baseline_feature_set": "base",
        "baseline_blend_weight": DEFAULT_BLEND_WEIGHT,
    },
    "local": {
        "filename": "local_place_v1.joblib",
        "feature_set": "local_course_geometry",
        "blend_weight": 1.0,
        "preserve_top_n": 2,
        "baseline_filename": "local_v1.joblib",
        "baseline_feature_set": "extended",
        "baseline_blend_weight": DEFAULT_BLEND_WEIGHT,
    },
}

BASE_FEATURE_NAMES = (
    "ability",
    "recent_quality",
    "trend",
    "condition_fit",
    "race_level",
    "pace_jockey",
    "risk",
    "legacy_support",
    "pedigree_adjustment",
    "race_level_adjustment",
    "pace_adjustment",
    "carried",
    "total",
)
RUN_FEATURE_NAMES = (
    "run_count",
    "last_finish_percentile",
    "best_finish_percentile",
    "mean_finish_percentile",
    "last_margin_closeness",
    "best_margin_closeness",
    "close_loss_count",
    "last_position_gain",
    "best_position_gain",
    "last_early_position",
    "best_early_position",
    "last_final3f_vs_history",
    "rest_days_preference",
    "distance_change_penalty",
    "same_distance_best",
    "same_course_best",
    "same_surface_best",
    "large_field_best",
    "jockey_continuity",
    "same_jockey_best",
    "class_drop",
    "surface_match_last",
)


def _model_directory() -> Path:
    configured = os.getenv("KEIBA_SCOPE_RERANKER_MODEL_DIR", "").strip()
    return Path(configured).resolve() if configured else MODEL_DIRECTORY


def _scope_key(race_data: RaceData) -> str:
    race_info = race_data.race_info
    source = str(race_info.source or "").strip().lower()
    key = str(race_info.scope_key or "").strip().lower()
    if source == "local" or key == "local" or key.startswith("local_"):
        return "local"
    if race_info.surface == "芝":
        return "central_turf"
    if race_info.surface == "ダート":
        return "central_dirt"
    return "central_other"


def _legacy_support(
    score: HorseScore,
    scope: str,
    horse: HorseEntry,
    race_data: RaceData,
) -> float:
    if not any(run.finish is not None for run in horse.recent_runs):
        return 0.0
    source = str(race_data.race_info.source or "").strip().lower()
    key = str(race_data.race_info.scope_key or "").strip().lower()
    values = score.scores
    if scope == "central_turf":
        if not (
            source == "central"
            or key == "central"
            or key.startswith("central_")
        ):
            return 0.0
        return (
            values.distance_fit * 0.3
            + values.course_fit * 0.3
            + values.jockey_fit * 0.2
        )
    if scope == "local":
        return values.course_fit * 0.4 + values.track_condition_fit * 0.3
    return 0.0


def _carried_weight_adjustment(
    horse: HorseEntry,
    horses: list[HorseEntry],
) -> float:
    if horse.carried_weight is None:
        return 0.0
    weights = [
        entry.carried_weight
        for entry in horses
        if entry.carried_weight is not None
    ]
    if len(weights) < 2:
        return 0.0
    field_average = sum(weights) / len(weights)
    diff = field_average - horse.carried_weight
    factor = 0.35 if diff > 0 else 0.25
    adjustment = diff * factor
    if horse.carried_weight <= 53.0 and field_average >= 55.0:
        adjustment += 0.2
    return round(max(-1.2, min(1.6, adjustment)), 1)


def _finish_percentile(
    finish: int | None,
    field_size: int | None,
) -> float:
    if finish is None or field_size is None or field_size <= 1:
        return 0.5
    return max(
        0.0,
        min(1.0, (field_size - finish) / (field_size - 1)),
    )


def _margin_closeness(finish: int | None, margin: str | None) -> float:
    if finish == 1:
        return 1.0
    parsed = parse_margin_value(margin)
    if parsed is None:
        return 0.0
    return max(-2.0, min(1.0, 1.0 - parsed))


def _position_gain(run: RecentRun) -> float:
    positions = run.corner_positions or []
    if run.finish is None or not positions:
        return 0.0
    return float(positions[-1] - run.finish)


def _early_position(run: RecentRun) -> float:
    positions = run.corner_positions or []
    if not positions or not run.field_size or run.field_size <= 1:
        return 0.5
    return max(
        0.0,
        min(
            1.0,
            (run.field_size - positions[0]) / (run.field_size - 1),
        ),
    )


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value or "")
    except (TypeError, ValueError):
        return None


def build_run_features(
    horse: HorseEntry,
    race_data: RaceData,
) -> dict[str, float]:
    recent_runs = [
        run
        for run in horse.recent_runs[:8]
        if run.finish is not None
        and run.field_size is not None
        and run.field_size > 1
    ][:5]
    if not recent_runs:
        return {name: 0.0 for name in RUN_FEATURE_NAMES}

    last = recent_runs[0]
    finish_values = [
        _finish_percentile(run.finish, run.field_size)
        for run in recent_runs
    ]
    margin_values = [
        _margin_closeness(run.finish, run.margin)
        for run in recent_runs
    ]
    gains = [_position_gain(run) for run in recent_runs]
    early_values = [_early_position(run) for run in recent_runs]
    final_3f_values = [
        float(run.final_3f)
        for run in recent_runs
        if run.final_3f is not None
    ]
    last_final_3f_relative = 0.0
    if last.final_3f is not None and len(final_3f_values) >= 2:
        historical = final_3f_values[1:] or final_3f_values
        last_final_3f_relative = (
            sum(historical) / len(historical) - float(last.final_3f)
        )

    race_info = race_data.race_info
    race_date = _parse_date(race_info.race_date)
    last_date = _parse_date(last.date)
    rest_days = (
        (race_date - last_date).days
        if race_date is not None and last_date is not None
        else 35
    )
    rest_days_preference = -abs(max(0, rest_days) - 35) / 35.0
    target_distance = race_info.distance
    distance_change = (
        abs(last.distance - target_distance)
        if last.distance is not None and target_distance is not None
        else 400
    )
    target_level = infer_target_race_class_level(race_info)
    prior_levels = [
        level
        for run in recent_runs
        if (level := infer_run_race_class_level(run)) is not None
    ]

    def best_for(predicate: Any) -> float:
        values = [
            _finish_percentile(run.finish, run.field_size)
            for run in recent_runs
            if predicate(run)
        ]
        return max(values) if values else 0.0

    close_loss_count = 0
    for run in recent_runs:
        parsed_margin = parse_margin_value(run.margin)
        if (
            run.finish != 1
            and parsed_margin is not None
            and parsed_margin <= 1.0
        ):
            close_loss_count += 1

    return {
        "run_count": float(len(recent_runs)),
        "last_finish_percentile": finish_values[0],
        "best_finish_percentile": max(finish_values),
        "mean_finish_percentile": sum(finish_values) / len(finish_values),
        "last_margin_closeness": margin_values[0],
        "best_margin_closeness": max(margin_values),
        "close_loss_count": float(close_loss_count),
        "last_position_gain": gains[0],
        "best_position_gain": max(gains),
        "last_early_position": early_values[0],
        "best_early_position": max(early_values),
        "last_final3f_vs_history": last_final_3f_relative,
        "rest_days_preference": rest_days_preference,
        "distance_change_penalty": -float(distance_change) / 100.0,
        "same_distance_best": best_for(
            lambda run: target_distance is not None
            and run.distance == target_distance
        ),
        "same_course_best": best_for(
            lambda run: bool(race_info.course)
            and run.course == race_info.course
        ),
        "same_surface_best": best_for(
            lambda run: bool(race_info.surface)
            and run.surface == race_info.surface
        ),
        "large_field_best": best_for(
            lambda run: bool(run.field_size) and run.field_size >= 13
        ),
        "jockey_continuity": float(
            bool(horse.jockey) and last.jockey == horse.jockey
        ),
        "same_jockey_best": best_for(
            lambda run: bool(horse.jockey)
            and run.jockey == horse.jockey
        ),
        "class_drop": (
            max(prior_levels) - target_level
            if prior_levels and target_level is not None
            else 0.0
        ),
        "surface_match_last": float(
            bool(race_info.surface)
            and last.surface == race_info.surface
        ),
    }


def _base_features(
    score: HorseScore,
    race_data: RaceData,
    horse: HorseEntry,
    scope: str,
) -> dict[str, float]:
    values = score.scores
    breakdown = score.score_breakdown
    return {
        "ability": float(values.ability_score),
        "recent_quality": float(values.recent_quality_score),
        "trend": float(values.trend_score),
        "condition_fit": float(values.condition_fit_score),
        "race_level": float(values.race_level_score),
        "pace_jockey": float(values.pace_jockey_score),
        "risk": float(values.risk),
        "legacy_support": float(
            _legacy_support(score, scope, horse, race_data)
        ),
        "pedigree_adjustment": float(
            breakdown.pedigree_adjustment_weighted
        ),
        "race_level_adjustment": float(
            breakdown.race_level_adjustment_weighted
        ),
        "pace_adjustment": float(breakdown.pace_adjustment_weighted),
        "carried": float(
            _carried_weight_adjustment(horse, race_data.horses)
        ),
        "total": float(breakdown.total_score),
    }


def _normalize(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0)
    standard_deviations[standard_deviations < 1e-8] = 1.0
    return (matrix - means) / standard_deviations


def build_scope_reranker_feature_matrix(
    race_data: RaceData,
    horse_scores: list[HorseScore],
    feature_set: str,
) -> tuple[np.ndarray, list[int]]:
    scope = _scope_key(race_data)
    scores_by_horse = {
        score.horse_no: score for score in horse_scores
    }
    horse_numbers: list[int] = []
    base_rows: list[list[float]] = []
    run_rows: list[list[float]] = []
    geometry_rows: list[list[float]] = []
    local_geometry_rows: list[list[float]] = []
    for horse in race_data.horses:
        score = scores_by_horse.get(horse.horse_no)
        if score is None:
            raise ValueError(
                f"horse {horse.horse_no} is missing from horse_scores"
            )
        base = _base_features(score, race_data, horse, scope)
        run = build_run_features(horse, race_data)
        geometry = build_course_geometry_features(
            horse, race_data.race_info
        )
        style = build_course_style_features(
            horse, race_data.race_info
        )
        local_geometry = build_local_course_geometry_features(
            horse, race_data.race_info
        )
        horse_numbers.append(score.horse_no)
        base_rows.append([base[name] for name in BASE_FEATURE_NAMES])
        run_rows.append([run[name] for name in RUN_FEATURE_NAMES])
        geometry_rows.append(
            [
                geometry[name]
                for name in COURSE_GEOMETRY_FEATURE_NAMES
            ]
            + [style[name] for name in COURSE_STYLE_FEATURE_NAMES]
        )
        local_geometry_rows.append(
            [
                local_geometry[name]
                for name in LOCAL_COURSE_GEOMETRY_FEATURE_NAMES
            ]
        )

    base_matrix = _normalize(np.asarray(base_rows, dtype=float))
    if feature_set == "base":
        return base_matrix, horse_numbers
    if feature_set == "extended":
        run_matrix = _normalize(np.asarray(run_rows, dtype=float))
        return (
            np.concatenate((base_matrix, run_matrix), axis=1),
            horse_numbers,
        )
    if feature_set == "course_geometry":
        geometry_matrix = _normalize(
            np.asarray(geometry_rows, dtype=float)
        )
        return (
            np.concatenate((base_matrix, geometry_matrix), axis=1),
            horse_numbers,
        )
    if feature_set == "local_course_geometry":
        run_matrix = _normalize(np.asarray(run_rows, dtype=float))
        local_geometry_matrix = _normalize(
            np.asarray(local_geometry_rows, dtype=float)
        )
        return (
            np.concatenate(
                (
                    base_matrix,
                    run_matrix,
                    local_geometry_matrix,
                ),
                axis=1,
            ),
            horse_numbers,
        )
    raise ValueError(f"unsupported feature_set: {feature_set}")


@lru_cache(maxsize=4)
def _load_model(path_text: str) -> Any:
    return joblib.load(path_text)


@lru_cache(maxsize=1)
def _load_metadata(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def clear_scope_reranker_caches() -> None:
    _load_model.cache_clear()
    _load_metadata.cache_clear()


def scope_reranker_config_snapshot(
    enabled: bool,
    scope: str | None = None,
) -> ScopeRerankerConfigSnapshot:
    metadata = _load_metadata(
        str(_model_directory() / "metadata.json")
    )
    model_spec = MODEL_SPECS.get(scope or "", {})
    return ScopeRerankerConfigSnapshot(
        enabled=enabled,
        model_version=str(
            metadata.get("model_version") or MODEL_VERSION
        ),
        enabled_scopes=list(MODEL_SPECS),
        blend_weight=float(
            model_spec.get("blend_weight", DEFAULT_BLEND_WEIGHT)
        ),
        preserve_top_n=int(model_spec.get("preserve_top_n", 0)),
    )


def _rank_scores(
    values: dict[int, float],
) -> dict[int, float]:
    ordered = sorted(
        values,
        key=lambda horse_no: (-values[horse_no], horse_no),
    )
    denominator = max(len(ordered) - 1, 1)
    return {
        horse_no: (len(ordered) - rank - 1) / denominator
        for rank, horse_no in enumerate(ordered)
    }


def _fallback(
    horse_scores: list[HorseScore],
    scope: str,
    model_version: str,
    reason_code: str,
) -> tuple[list[HorseScore], ScopeRerankerResult]:
    ordered = sorted(
        horse_scores,
        key=lambda item: (-item.total_score, item.horse_no),
    )
    ranking = [score.horse_no for score in ordered]
    return ordered, ScopeRerankerResult(
        applied=False,
        scope=scope,
        model_version=model_version,
        reason_code=reason_code,
        original_ranking=ranking,
        baseline_ranking=ranking,
        reranked_ranking=ranking,
    )


def _blend_model_ranking(
    base_ranking: list[int],
    model_scores: dict[int, float],
    *,
    blend_weight: float,
    preserve_top_n: int = 0,
) -> tuple[list[int], dict[int, float]]:
    protected_count = min(
        preserve_top_n,
        max(len(base_ranking) - 1, 0),
    )
    protected_ranking = base_ranking[:protected_count]
    candidate_pool = base_ranking[protected_count:]
    learned_rank_scores = _rank_scores(
        {
            horse_no: model_scores[horse_no]
            for horse_no in candidate_pool
        }
    )
    original_rank_scores = _rank_scores(
        {
            horse_no: float(len(candidate_pool) - rank)
            for rank, horse_no in enumerate(candidate_pool)
        }
    )
    blended_scores = {
        horse_no: (
            (1.0 - blend_weight) * original_rank_scores[horse_no]
            + blend_weight * learned_rank_scores[horse_no]
        )
        for horse_no in candidate_pool
    }
    overall_base_scores = {
        horse_no: (
            len(base_ranking) - rank - 1
        )
        / max(len(base_ranking) - 1, 1)
        for rank, horse_no in enumerate(base_ranking)
    }
    for horse_no in protected_ranking:
        blended_scores[horse_no] = overall_base_scores[horse_no]
    reranked = protected_ranking + sorted(
        candidate_pool,
        key=lambda horse_no: (-blended_scores[horse_no], horse_no),
    )
    return reranked, blended_scores


def apply_scope_reranker(
    race_data: RaceData,
    horse_scores: list[HorseScore],
    *,
    enabled: bool,
) -> tuple[
    list[HorseScore],
    ScopeRerankerConfigSnapshot,
    ScopeRerankerResult,
]:
    scope = _scope_key(race_data)
    config = scope_reranker_config_snapshot(enabled, scope)
    if not enabled:
        scores, result = _fallback(
            horse_scores,
            scope,
            config.model_version,
            "disabled",
        )
        return scores, config, result
    model_spec = MODEL_SPECS.get(scope)
    if model_spec is None:
        scores, result = _fallback(
            horse_scores,
            scope,
            config.model_version,
            "scope_not_enabled",
        )
        return scores, config, result
    if len(horse_scores) < 2:
        scores, result = _fallback(
            horse_scores,
            scope,
            config.model_version,
            "insufficient_horses",
        )
        return scores, config, result

    original_scores = sorted(
        horse_scores,
        key=lambda item: (-item.total_score, item.horse_no),
    )
    original_ranking = [score.horse_no for score in original_scores]
    model_path = _model_directory() / str(model_spec["filename"])
    try:
        model = _load_model(str(model_path))
        baseline_model = None
        if model_spec.get("baseline_filename"):
            baseline_model = _load_model(
                str(
                    _model_directory()
                    / str(model_spec["baseline_filename"])
                )
            )
    except Exception:
        scores, result = _fallback(
            horse_scores,
            scope,
            config.model_version,
            "model_load_failed",
        )
        return scores, config, result

    try:
        baseline_ranking = list(original_ranking)
        if baseline_model is not None:
            baseline_matrix, baseline_horse_numbers = (
                build_scope_reranker_feature_matrix(
                    race_data,
                    original_scores,
                    str(model_spec["baseline_feature_set"]),
                )
            )
            baseline_expected_features = getattr(
                baseline_model,
                "n_features_in_",
                baseline_matrix.shape[1],
            )
            if (
                int(baseline_expected_features)
                != baseline_matrix.shape[1]
            ):
                raise ValueError(
                    "baseline feature count mismatch: "
                    f"expected={baseline_expected_features}, "
                    f"actual={baseline_matrix.shape[1]}"
                )
            baseline_probabilities = baseline_model.predict_proba(
                baseline_matrix
            )[:, 1]
            baseline_scores = {
                horse_no: float(probability)
                for horse_no, probability in zip(
                    baseline_horse_numbers,
                    baseline_probabilities,
                    strict=True,
                )
            }
            baseline_ranking, _ = _blend_model_ranking(
                original_ranking,
                baseline_scores,
                blend_weight=float(
                    model_spec["baseline_blend_weight"]
                ),
            )
        matrix, horse_numbers = build_scope_reranker_feature_matrix(
            race_data,
            original_scores,
            str(model_spec["feature_set"]),
        )
        expected_features = getattr(model, "n_features_in_", matrix.shape[1])
        if int(expected_features) != matrix.shape[1]:
            raise ValueError(
                f"feature count mismatch: expected={expected_features}, "
                f"actual={matrix.shape[1]}"
            )
        probabilities = model.predict_proba(matrix)[:, 1]
    except Exception:
        scores, result = _fallback(
            horse_scores,
            scope,
            config.model_version,
            "prediction_failed",
        )
        return scores, config, result

    model_scores = {
        horse_no: float(probability)
        for horse_no, probability in zip(
            horse_numbers,
            probabilities,
            strict=True,
        )
    }
    reranked_ranking, blended_scores = _blend_model_ranking(
        baseline_ranking,
        model_scores,
        blend_weight=config.blend_weight,
        preserve_top_n=config.preserve_top_n,
    )

    original_by_horse = {
        score.horse_no: score for score in original_scores
    }
    score_values = sorted(
        (float(score.total_score) for score in original_scores),
        reverse=True,
    )
    assigned_scores: dict[int, float] = {}
    previous: float | None = None
    for rank, horse_no in enumerate(reranked_ranking):
        assigned = score_values[rank]
        if previous is not None and assigned >= previous:
            assigned = previous - 0.01
        assigned_scores[horse_no] = round(assigned, 3)
        previous = assigned_scores[horse_no]

    original_rank_map = {
        horse_no: rank
        for rank, horse_no in enumerate(original_ranking, start=1)
    }
    reranked_rank_map = {
        horse_no: rank
        for rank, horse_no in enumerate(reranked_ranking, start=1)
    }
    adjusted_scores: list[HorseScore] = []
    for horse_no in reranked_ranking:
        score = original_by_horse[horse_no]
        assigned_total = assigned_scores[horse_no]
        adjustment = round(assigned_total - score.total_score, 3)
        breakdown = score.score_breakdown.model_copy(
            update={
                "scope_reranker_adjustment": adjustment,
                "total_score_after_reranker": assigned_total,
            }
        )
        adjusted_scores.append(
            score.model_copy(
                update={
                    "scope_reranker": HorseScopeRerankerResult(
                        applied=True,
                        original_rank=original_rank_map[horse_no],
                        new_rank=reranked_rank_map[horse_no],
                        model_score=round(model_scores[horse_no], 6),
                        blended_rank_score=round(
                            blended_scores[horse_no],
                            6,
                        ),
                        score_adjustment=adjustment,
                    ),
                    "score_breakdown": breakdown,
                    "total_score": assigned_total,
                }
            )
        )

    return (
        adjusted_scores,
        config,
        ScopeRerankerResult(
            applied=True,
            scope=scope,
            model_version=config.model_version,
            reason_code="applied",
            original_ranking=original_ranking,
            baseline_ranking=baseline_ranking,
            reranked_ranking=reranked_ranking,
        ),
    )

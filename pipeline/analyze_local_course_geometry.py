from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from keiba_llm_agent.analysis.local_course_geometry import (
    LOCAL_COURSE_GEOMETRY_FEATURE_NAMES,
    LOCAL_COURSE_STYLE_FEATURE_NAMES,
    build_local_course_geometry_features,
    build_local_course_style_features,
)
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.scoring.scope_reranker import (
    BASE_FEATURE_NAMES,
    MODEL_DIRECTORY,
    RUN_FEATURE_NAMES,
    build_scope_reranker_feature_matrix,
)
from pipeline.train_scope_reranker import (
    preferred_audits,
    recompute_horse_scores,
    scope_key,
    scoring_config_for_scope,
    winner_from_audit,
)


MODEL_TRAINING_END = "2026-07-11"
MODEL_BLEND_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
MODEL_PROTECTED_PREFIXES = (0, 1, 2, 3)
SMALL_FIELD_POLICIES = ("fixed", "strict_small", "skip_small")
STATIC_BLEND_WEIGHTS = (0.04, 0.06, 0.08, 0.10, 0.12)
EXTENDED_FEATURE_COUNT = len(BASE_FEATURE_NAMES) + len(RUN_FEATURE_NAMES)
GEOMETRY_FEATURE_COUNT = (
    EXTENDED_FEATURE_COUNT + len(LOCAL_COURSE_GEOMETRY_FEATURE_NAMES)
)


@dataclass
class RaceEvaluation:
    race_id: str
    race_date: str
    course: str
    surface: str
    winner: int
    top3: set[int]
    horse_numbers: list[int]
    production_ranking: list[int]
    model_matrix: np.ndarray
    geometry_signal: dict[int, float]
    exact_course_signal: dict[int, float]
    geometry_reliability: dict[int, float]
    winner_popularity: int | None

    @property
    def field_size(self) -> int:
        return len(self.horse_numbers)


def _horse_number(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _top3_from_audit(payload: dict[str, Any]) -> set[int]:
    result_payload = (payload.get("outcomes") or {}).get("result") or {}
    direct = result_payload.get("result") or {}
    top3 = {
        horse_no
        for position in ("1st", "2nd", "3rd")
        if (horse_no := _horse_number(direct.get(position))) is not None
    }
    if top3:
        return top3
    for row in result_payload.get("finish_order") or []:
        finish = _horse_number(row.get("finish"))
        horse_no = _horse_number(row.get("horse_no"))
        if finish is not None and finish <= 3 and horse_no is not None:
            top3.add(horse_no)
    return top3


def _winner_popularity(
    payload: dict[str, Any], winner: int
) -> int | None:
    result_payload = (payload.get("outcomes") or {}).get("result") or {}
    for row in result_payload.get("finish_order") or []:
        if _horse_number(row.get("horse_no")) != winner:
            continue
        return _horse_number(row.get("popularity"))
    return None


def _normalize(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0)
    standard_deviations[standard_deviations < 1e-8] = 1.0
    return (matrix - means) / standard_deviations


def build_evaluations(
    snapshot: Path,
) -> tuple[list[RaceEvaluation], dict[str, int]]:
    audits = preferred_audits(snapshot)
    baseline_model = joblib.load(MODEL_DIRECTORY / "local_v1.joblib")
    evaluations: list[RaceEvaluation] = []
    counters = {
        "preferred_union": len(audits),
        "full_settled": 0,
        "local_full_settled": 0,
        "recompute_failed": 0,
    }
    for payload in audits.values():
        if payload.get("audit_kind") != "full":
            continue
        winner = winner_from_audit(payload)
        race_data_payload = (payload.get("inputs") or {}).get("race_data")
        prediction_payload = (payload.get("outputs") or {}).get("prediction")
        if winner is None or not race_data_payload or not prediction_payload:
            continue
        counters["full_settled"] += 1
        try:
            race_data = RaceData.model_validate(race_data_payload)
            if scope_key(race_data) != "local":
                continue
            counters["local_full_settled"] += 1
            prediction = Prediction.model_validate(prediction_payload)
            raw_scores = recompute_horse_scores(
                race_data,
                prediction,
                scoring_config_for_scope("local"),
            )
            extended_matrix, matrix_horse_numbers = (
                build_scope_reranker_feature_matrix(
                    race_data, raw_scores, "extended"
                )
            )
        except Exception:
            counters["recompute_failed"] += 1
            continue

        horse_numbers: list[int] = []
        geometry_rows: list[list[float]] = []
        geometry_signal: dict[int, float] = {}
        exact_course_signal: dict[int, float] = {}
        geometry_reliability: dict[int, float] = {}
        for horse in race_data.horses:
            geometry = build_local_course_geometry_features(
                horse, race_data.race_info
            )
            style = build_local_course_style_features(
                horse, race_data.race_info
            )
            horse_numbers.append(horse.horse_no)
            geometry_rows.append(
                [
                    geometry[name]
                    for name in LOCAL_COURSE_GEOMETRY_FEATURE_NAMES
                ]
                + [
                    style[name]
                    for name in LOCAL_COURSE_STYLE_FEATURE_NAMES
                ]
            )
            geometry_signal[horse.horse_no] = geometry[
                "local_geometry_fit_shrunken_residual"
            ]
            exact_course_signal[horse.horse_no] = geometry[
                "local_geometry_exact_course_shrunken_residual"
            ]
            geometry_reliability[horse.horse_no] = geometry[
                "local_geometry_reliability"
            ]
        if horse_numbers != matrix_horse_numbers:
            counters["recompute_failed"] += 1
            continue
        baseline_probabilities = baseline_model.predict_proba(
            extended_matrix
        )[:, 1]
        baseline_signal = {
            horse_no: float(probability)
            for horse_no, probability in zip(
                horse_numbers,
                baseline_probabilities,
                strict=True,
            )
        }
        raw_ranking = [score.horse_no for score in raw_scores]
        production_ranking = blend_ranking(
            raw_ranking, baseline_signal, 0.5
        )
        evaluations.append(
            RaceEvaluation(
                race_id=race_data.race_info.race_id,
                race_date=str(race_data.race_info.race_date or ""),
                course=str(race_data.race_info.course or ""),
                surface=str(race_data.race_info.surface or ""),
                winner=winner,
                top3=_top3_from_audit(payload),
                horse_numbers=horse_numbers,
                production_ranking=production_ranking,
                model_matrix=np.concatenate(
                    (
                        extended_matrix,
                        _normalize(np.asarray(geometry_rows, dtype=float)),
                    ),
                    axis=1,
                ),
                geometry_signal=geometry_signal,
                exact_course_signal=exact_course_signal,
                geometry_reliability=geometry_reliability,
                winner_popularity=_winner_popularity(payload, winner),
            )
        )
    evaluations.sort(key=lambda row: (row.race_date, row.race_id))
    return evaluations, counters


def _rank_scores(ranking: list[int]) -> dict[int, float]:
    denominator = max(len(ranking) - 1, 1)
    return {
        horse_no: (len(ranking) - rank - 1) / denominator
        for rank, horse_no in enumerate(ranking)
    }


def _signal_rank_scores(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    ordered = sorted(
        values,
        key=lambda horse_no: (-values[horse_no], horse_no),
    )
    denominator = max(len(ordered) - 1, 1)
    return {
        horse_no: (len(ordered) - rank - 1) / denominator
        for rank, horse_no in enumerate(ordered)
    }


def blend_ranking(
    base_ranking: list[int],
    signal: dict[int, float],
    blend_weight: float,
) -> list[int]:
    base_scores = _rank_scores(base_ranking)
    signal_scores = _signal_rank_scores(signal)
    combined = {
        horse_no: (
            (1.0 - blend_weight) * base_scores[horse_no]
            + blend_weight * signal_scores.get(horse_no, 0.5)
        )
        for horse_no in base_ranking
    }
    return sorted(
        base_ranking,
        key=lambda horse_no: (-combined[horse_no], horse_no),
    )


def metrics(
    rows: list[RaceEvaluation],
    ranking_for: Callable[[RaceEvaluation], list[int]],
) -> dict[str, float | int]:
    top1_win = 0
    top1_place = 0
    winner_top3 = 0
    winner_top5 = 0
    winner_rank_sum = 0.0
    normalized_rank_sum = 0.0
    reciprocal_rank_sum = 0.0
    expected_random_top1 = 0.0
    field_size_sum = 0
    for row in rows:
        ranking = ranking_for(row)
        rank = ranking.index(row.winner) + 1
        top1_win += ranking[0] == row.winner
        top1_place += ranking[0] in row.top3
        winner_top3 += rank <= 3
        winner_top5 += rank <= 5
        winner_rank_sum += rank
        normalized_rank_sum += (rank - 1) / max(row.field_size - 1, 1)
        reciprocal_rank_sum += 1.0 / rank
        expected_random_top1 += 1.0 / row.field_size
        field_size_sum += row.field_size
    count = len(rows)
    if not count:
        return {
            "races": 0,
            "average_field_size": 0.0,
            "top1_win": 0,
            "top1_win_rate": 0.0,
            "random_adjusted_top1_lift": 0.0,
            "top1_place": 0,
            "top1_place_rate": 0.0,
            "winner_top3": 0,
            "winner_top3_rate": 0.0,
            "winner_top5": 0,
            "winner_top5_rate": 0.0,
            "mean_winner_rank": 0.0,
            "mean_normalized_winner_rank": 0.0,
            "mean_reciprocal_rank": 0.0,
        }
    return {
        "races": count,
        "average_field_size": round(field_size_sum / count, 3),
        "top1_win": top1_win,
        "top1_win_rate": round(top1_win * 100.0 / count, 3),
        "random_adjusted_top1_lift": round(
            top1_win / expected_random_top1, 3
        ),
        "top1_place": top1_place,
        "top1_place_rate": round(top1_place * 100.0 / count, 3),
        "winner_top3": winner_top3,
        "winner_top3_rate": round(winner_top3 * 100.0 / count, 3),
        "winner_top5": winner_top5,
        "winner_top5_rate": round(winner_top5 * 100.0 / count, 3),
        "mean_winner_rank": round(winner_rank_sum / count, 4),
        "mean_normalized_winner_rank": round(
            normalized_rank_sum / count, 5
        ),
        "mean_reciprocal_rank": round(
            reciprocal_rank_sum / count, 5
        ),
    }


def metric_delta(
    baseline: dict[str, float | int],
    candidate: dict[str, float | int],
) -> dict[str, float | int]:
    keys = (
        "top1_win",
        "top1_win_rate",
        "random_adjusted_top1_lift",
        "top1_place",
        "top1_place_rate",
        "winner_top3",
        "winner_top3_rate",
        "winner_top5",
        "winner_top5_rate",
        "mean_winner_rank",
        "mean_normalized_winner_rank",
        "mean_reciprocal_rank",
    )
    return {
        key: round(float(candidate[key]) - float(baseline[key]), 5)
        for key in keys
    }


def _scope_report(
    rows: list[RaceEvaluation],
    ranking_for: Callable[[RaceEvaluation], list[int]],
) -> dict[str, Any]:
    baseline = metrics(rows, lambda row: row.production_ranking)
    candidate = metrics(rows, ranking_for)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": metric_delta(baseline, candidate),
    }


def train_model(
    rows: list[RaceEvaluation],
    feature_count: int,
) -> HistGradientBoostingClassifier:
    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for row in rows:
        positive_weight = max(row.field_size - 1, 1)
        for feature_row, horse_no in zip(
            row.model_matrix, row.horse_numbers, strict=True
        ):
            is_winner = horse_no == row.winner
            features.append(feature_row[:feature_count])
            labels.append(int(is_winner))
            weights.append(float(positive_weight if is_winner else 1.0))
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=7,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=42,
    )
    model.fit(
        np.asarray(features, dtype=float),
        np.asarray(labels, dtype=int),
        sample_weight=np.asarray(weights, dtype=float),
    )
    return model


def model_ranking(
    model: HistGradientBoostingClassifier,
    row: RaceEvaluation,
    *,
    feature_count: int,
    preserve_top_n: int,
    blend_weight: float,
    small_field_policy: str,
) -> list[int]:
    if small_field_policy == "skip_small" and row.field_size <= 8:
        return list(row.production_ranking)
    protected_count = preserve_top_n
    if small_field_policy == "strict_small" and row.field_size <= 8:
        protected_count = max(protected_count, 2)
    probabilities = model.predict_proba(
        row.model_matrix[:, :feature_count]
    )[:, 1]
    signal = {
        horse_no: float(probability)
        for horse_no, probability in zip(
            row.horse_numbers, probabilities, strict=True
        )
    }
    protected = row.production_ranking[:protected_count]
    pool = row.production_ranking[protected_count:]
    return protected + blend_ranking(
        pool,
        {horse_no: signal[horse_no] for horse_no in pool},
        blend_weight,
    )


def select_model_policy(
    model: HistGradientBoostingClassifier,
    validation: list[RaceEvaluation],
    feature_count: int,
) -> tuple[int, float, str, dict[str, Any]]:
    baseline = metrics(validation, lambda row: row.production_ranking)
    screened: dict[str, Any] = {}
    choices: list[tuple[tuple[float, ...], int, float, str]] = []
    for policy in SMALL_FIELD_POLICIES:
        for preserve_top_n in MODEL_PROTECTED_PREFIXES:
            for blend_weight in MODEL_BLEND_WEIGHTS:
                ranking_for = (
                    lambda row,
                    policy=policy,
                    preserve_top_n=preserve_top_n,
                    blend_weight=blend_weight: model_ranking(
                        model,
                        row,
                        feature_count=feature_count,
                        preserve_top_n=preserve_top_n,
                        blend_weight=blend_weight,
                        small_field_policy=policy,
                    )
                )
                candidate = metrics(validation, ranking_for)
                delta = metric_delta(baseline, candidate)
                key = (
                    f"{policy}_protect_{preserve_top_n}"
                    f"_blend_{blend_weight:.2f}"
                )
                screened[key] = {
                    "metrics": candidate,
                    "delta": delta,
                }
                no_top_damage = (
                    float(delta["top1_win"]) >= 0
                    and float(delta["top1_place"]) >= 0
                )
                choices.append(
                    (
                        (
                            float(no_top_damage),
                            float(delta["winner_top3"]),
                            float(delta["winner_top5"]),
                            -float(delta["mean_normalized_winner_rank"]),
                            float(delta["mean_reciprocal_rank"]),
                            -blend_weight,
                            -preserve_top_n,
                        ),
                        preserve_top_n,
                        blend_weight,
                        policy,
                    )
                )
    _, preserve_top_n, blend_weight, policy = max(choices)
    return preserve_top_n, blend_weight, policy, screened


def _date_range(rows: list[RaceEvaluation]) -> dict[str, str | int]:
    if not rows:
        return {"races": 0, "start": "", "end": ""}
    return {
        "races": len(rows),
        "start": rows[0].race_date,
        "end": rows[-1].race_date,
    }


def _field_buckets(
    rows: list[RaceEvaluation],
    ranking_for: Callable[[RaceEvaluation], list[int]],
) -> dict[str, Any]:
    buckets = {
        "8_or_fewer": [row for row in rows if row.field_size <= 8],
        "9_to_11": [
            row for row in rows if 9 <= row.field_size <= 11
        ],
        "12_or_more": [row for row in rows if row.field_size >= 12],
    }
    return {
        name: _scope_report(bucket, ranking_for)
        for name, bucket in buckets.items()
    }


def _by_course(
    rows: list[RaceEvaluation],
    ranking_for: Callable[[RaceEvaluation], list[int]],
) -> dict[str, Any]:
    return {
        course: _scope_report(
            [row for row in rows if row.course == course],
            ranking_for,
        )
        for course in sorted({row.course for row in rows})
    }


def _crossings(
    rows: list[RaceEvaluation],
    ranking_for: Callable[[RaceEvaluation], list[int]],
    cutoff: int,
) -> dict[str, Any]:
    gained: list[RaceEvaluation] = []
    lost: list[RaceEvaluation] = []
    for row in rows:
        baseline_rank = row.production_ranking.index(row.winner) + 1
        candidate_rank = ranking_for(row).index(row.winner) + 1
        if candidate_rank <= cutoff < baseline_rank:
            gained.append(row)
        elif baseline_rank <= cutoff < candidate_rank:
            lost.append(row)
    discordant = len(gained) + len(lost)
    tail = min(len(gained), len(lost))
    paired_p_value = (
        min(
            1.0,
            2.0
            * sum(
                math.comb(discordant, k) * (0.5**discordant)
                for k in range(tail + 1)
            ),
        )
        if discordant
        else 1.0
    )
    return {
        "gained": len(gained),
        "lost": len(lost),
        "net": len(gained) - len(lost),
        "gained_popularity_6plus": sum(
            row.winner_popularity is not None
            and row.winner_popularity >= 6
            for row in gained
        ),
        "lost_popularity_6plus": sum(
            row.winner_popularity is not None
            and row.winner_popularity >= 6
            for row in lost
        ),
        "paired_exact_p_value": round(paired_p_value, 6),
    }


def _policy_grid_report(
    model: HistGradientBoostingClassifier,
    feature_count: int,
    validation: list[RaceEvaluation],
    recent: list[RaceEvaluation],
    holdout: list[RaceEvaluation],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for policy in SMALL_FIELD_POLICIES:
        for preserve_top_n in MODEL_PROTECTED_PREFIXES:
            for blend_weight in MODEL_BLEND_WEIGHTS:
                ranking_for = (
                    lambda row,
                    policy=policy,
                    preserve_top_n=preserve_top_n,
                    blend_weight=blend_weight: model_ranking(
                        model,
                        row,
                        feature_count=feature_count,
                        preserve_top_n=preserve_top_n,
                        blend_weight=blend_weight,
                        small_field_policy=policy,
                    )
                )
                key = (
                    f"{policy}_protect_{preserve_top_n}"
                    f"_blend_{blend_weight:.2f}"
                )
                report[key] = {
                    "validation": _scope_report(validation, ranking_for),
                    "recent_oot": _scope_report(recent, ranking_for),
                    "holdout": _scope_report(holdout, ranking_for),
                }
    return report


def analyze(snapshot: Path) -> dict[str, Any]:
    rows, counters = build_evaluations(snapshot)
    if not rows:
        raise RuntimeError("no settled local full audits could be recomputed")
    max_date = max(date.fromisoformat(row.race_date) for row in rows)
    oot_start = (max_date - timedelta(days=6)).isoformat()
    training = [
        row for row in rows if row.race_date <= MODEL_TRAINING_END
    ]
    holdout = [
        row for row in rows if row.race_date > MODEL_TRAINING_END
    ]
    validation = [
        row for row in holdout if row.race_date < oot_start
    ]
    recent = [row for row in holdout if row.race_date >= oot_start]

    models = {
        "base_only": (
            train_model(training, EXTENDED_FEATURE_COUNT),
            EXTENDED_FEATURE_COUNT,
        ),
        "geometry": (
            train_model(training, GEOMETRY_FEATURE_COUNT),
            GEOMETRY_FEATURE_COUNT,
        ),
        "geometry_and_style": (
            train_model(training, int(training[0].model_matrix.shape[1])),
            int(training[0].model_matrix.shape[1]),
        ),
    }
    model_reports: dict[str, Any] = {}
    for name, (model, feature_count) in models.items():
        preserve_top_n, blend_weight, policy, screened = (
            select_model_policy(model, validation, feature_count)
        )
        ranking_for = (
            lambda row,
            model=model,
            feature_count=feature_count,
            preserve_top_n=preserve_top_n,
            blend_weight=blend_weight,
            policy=policy: model_ranking(
                model,
                row,
                feature_count=feature_count,
                preserve_top_n=preserve_top_n,
                blend_weight=blend_weight,
                small_field_policy=policy,
            )
        )
        model_reports[name] = {
            "feature_count": feature_count,
            "selected_on_validation_only": True,
            "preserve_top_n": preserve_top_n,
            "blend_weight": blend_weight,
            "small_field_policy": policy,
            "screened_validation": screened,
            "policy_robustness": _policy_grid_report(
                model,
                feature_count,
                validation,
                recent,
                holdout,
            ),
            "all": _scope_report(rows, ranking_for),
            "training_replay": _scope_report(training, ranking_for),
            "holdout": _scope_report(holdout, ranking_for),
            "validation": _scope_report(validation, ranking_for),
            "recent_oot": _scope_report(recent, ranking_for),
            "holdout_field_buckets": _field_buckets(
                holdout, ranking_for
            ),
            "validation_field_buckets": _field_buckets(
                validation, ranking_for
            ),
            "recent_field_buckets": _field_buckets(
                recent, ranking_for
            ),
            "holdout_by_course": _by_course(holdout, ranking_for),
            "validation_by_course": _by_course(
                validation, ranking_for
            ),
            "recent_by_course": _by_course(recent, ranking_for),
            "holdout_by_date": {
                race_date: _scope_report(
                    [
                        row
                        for row in holdout
                        if row.race_date == race_date
                    ],
                    ranking_for,
                )
                for race_date in sorted(
                    {row.race_date for row in holdout}
                )
            },
            "holdout_top3_crossings": _crossings(
                holdout, ranking_for, 3
            ),
            "holdout_top5_crossings": _crossings(
                holdout, ranking_for, 5
            ),
            "recent_top3_crossings": _crossings(
                recent, ranking_for, 3
            ),
            "recent_top5_crossings": _crossings(
                recent, ranking_for, 5
            ),
        }

    static_reports: dict[str, Any] = {}
    for signal_name in ("geometry", "exact_course"):
        signal_for = (
            (lambda row: row.geometry_signal)
            if signal_name == "geometry"
            else (lambda row: row.exact_course_signal)
        )
        for blend_weight in STATIC_BLEND_WEIGHTS:
            ranking_for = (
                lambda row,
                signal_for=signal_for,
                blend_weight=blend_weight: blend_ranking(
                    row.production_ranking,
                    signal_for(row),
                    blend_weight,
                )
            )
            static_reports[f"{signal_name}_{blend_weight:.2f}"] = {
                "validation": _scope_report(validation, ranking_for),
                "recent_oot": _scope_report(recent, ranking_for),
                "holdout": _scope_report(holdout, ranking_for),
            }

    horse_count = sum(row.field_size for row in rows)
    covered_horses = sum(
        reliability > 0.0
        for row in rows
        for reliability in row.geometry_reliability.values()
    )
    return {
        "snapshot": str(snapshot),
        "counts": counters,
        "evaluated": _date_range(rows),
        "training_end": MODEL_TRAINING_END,
        "oot_start": oot_start,
        "training": _date_range(training),
        "validation": _date_range(validation),
        "recent_oot": _date_range(recent),
        "holdout": _date_range(holdout),
        "coverage": {
            "races": len(rows),
            "horses": horse_count,
            "horses_with_geometry_history": covered_horses,
            "horse_coverage_pct": round(
                covered_horses * 100.0 / horse_count, 2
            ),
        },
        "production_baseline": {
            "all": metrics(rows, lambda row: row.production_ranking),
            "training": metrics(
                training, lambda row: row.production_ranking
            ),
            "validation": metrics(
                validation, lambda row: row.production_ranking
            ),
            "recent_oot": metrics(
                recent, lambda row: row.production_ranking
            ),
            "all_field_buckets": _field_buckets(
                rows, lambda row: row.production_ranking
            ),
            "all_by_course": _by_course(
                rows, lambda row: row.production_ranking
            ),
        },
        "static_candidates": static_reports,
        "model_candidates": model_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze(args.snapshot.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

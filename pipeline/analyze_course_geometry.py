from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from keiba_llm_agent.analysis.course_geometry import (
    COURSE_GEOMETRY_FEATURE_NAMES,
    COURSE_STYLE_FEATURE_NAMES,
    build_course_geometry_features,
    build_course_style_features,
)
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.scoring.scope_reranker import (
    BASE_FEATURE_NAMES,
    DEFAULT_BLEND_WEIGHT,
    apply_scope_reranker,
    build_scope_reranker_feature_matrix,
)
from pipeline.train_scope_reranker import (
    preferred_audits,
    recompute_horse_scores,
    scope_key,
    scoring_config_for_scope,
    winner_from_audit,
)


JRA_COURSES = {
    "札幌",
    "函館",
    "福島",
    "新潟",
    "東京",
    "中山",
    "中京",
    "京都",
    "阪神",
    "小倉",
}
MODEL_TRAINING_END = "2026-05-24"
STATIC_BLEND_WEIGHTS = (0.04, 0.06, 0.08, 0.10, 0.12)
MODEL_BLEND_WEIGHTS = (0.25, 0.35, 0.50, 0.75, 1.00)
MODEL_PROTECTED_PREFIXES = (1, 2, 3, 4)
GEOMETRY_MODEL_FEATURE_COUNT = (
    len(BASE_FEATURE_NAMES) + len(COURSE_GEOMETRY_FEATURE_NAMES)
)


@dataclass
class RaceEvaluation:
    race_id: str
    race_date: str
    surface: str
    course: str
    winner: int
    top3: set[int]
    horse_numbers: list[int]
    raw_ranking: list[int]
    production_ranking: list[int]
    geometry_residual: dict[int, float]
    geometry_performance: dict[int, float]
    geometry_reliability: dict[int, float]
    model_matrix: np.ndarray
    winner_popularity: int | None


def _horse_number(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def top3_from_audit(payload: dict[str, Any]) -> set[int]:
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


def _normalize(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0)
    standard_deviations[standard_deviations < 1e-8] = 1.0
    return (matrix - means) / standard_deviations


def build_evaluations(
    snapshot: Path,
) -> tuple[list[RaceEvaluation], dict[str, int]]:
    audits = preferred_audits(snapshot)
    evaluations: list[RaceEvaluation] = []
    counters = {
        "preferred_union": len(audits),
        "full_settled": 0,
        "central_jra_settled": 0,
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
            if race_data.race_info.course not in JRA_COURSES:
                continue
            scope = scope_key(race_data)
            if scope not in {"central_turf", "central_dirt"}:
                continue
            counters["central_jra_settled"] += 1
            prediction = Prediction.model_validate(prediction_payload)
            raw_scores = recompute_horse_scores(
                race_data,
                prediction,
                scoring_config_for_scope(scope),
            )
            production_scores, _, _ = apply_scope_reranker(
                race_data,
                raw_scores,
                enabled=True,
            )
            base_matrix, matrix_horse_numbers = (
                build_scope_reranker_feature_matrix(
                    race_data,
                    raw_scores,
                    "base",
                )
            )
        except Exception:
            counters["recompute_failed"] += 1
            continue

        geometry_rows: list[list[float]] = []
        geometry_residual: dict[int, float] = {}
        geometry_performance: dict[int, float] = {}
        geometry_reliability: dict[int, float] = {}
        horse_numbers: list[int] = []
        for horse in race_data.horses:
            features = build_course_geometry_features(
                horse, race_data.race_info
            )
            style_features = build_course_style_features(
                horse, race_data.race_info
            )
            horse_no = horse.horse_no
            horse_numbers.append(horse_no)
            geometry_rows.append(
                [
                    features[name]
                    for name in COURSE_GEOMETRY_FEATURE_NAMES
                ]
                + [
                    style_features[name]
                    for name in COURSE_STYLE_FEATURE_NAMES
                ]
            )
            reliability = features["geometry_reliability"]
            average_similarity = features["geometry_average_similarity"]
            geometry_residual[horse_no] = features[
                "geometry_fit_shrunken_residual"
            ]
            geometry_performance[horse_no] = (
                (
                    features["geometry_analogous_performance"]
                    - 0.5
                )
                * reliability
                * average_similarity
            )
            geometry_reliability[horse_no] = reliability
        if horse_numbers != matrix_horse_numbers:
            counters["recompute_failed"] += 1
            continue
        winner_entry = next(
            (
                horse
                for horse in race_data.horses
                if horse.horse_no == winner
            ),
            None,
        )
        geometry_matrix = _normalize(
            np.asarray(geometry_rows, dtype=float)
        )
        evaluations.append(
            RaceEvaluation(
                race_id=race_data.race_info.race_id,
                race_date=str(race_data.race_info.race_date or ""),
                surface=str(race_data.race_info.surface or ""),
                course=str(race_data.race_info.course or ""),
                winner=winner,
                top3=top3_from_audit(payload),
                horse_numbers=horse_numbers,
                raw_ranking=[
                    score.horse_no for score in raw_scores
                ],
                production_ranking=[
                    score.horse_no for score in production_scores
                ],
                geometry_residual=geometry_residual,
                geometry_performance=geometry_performance,
                geometry_reliability=geometry_reliability,
                model_matrix=np.concatenate(
                    (base_matrix, geometry_matrix), axis=1
                ),
                winner_popularity=(
                    winner_entry.popularity
                    if winner_entry is not None
                    else None
                ),
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


def _tied_signal_scores(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    ordered_values = sorted(set(values.values()))
    if len(ordered_values) == 1:
        return {horse_no: 0.5 for horse_no in values}
    value_scores = {
        value: index / (len(ordered_values) - 1)
        for index, value in enumerate(ordered_values)
    }
    return {
        horse_no: value_scores[value]
        for horse_no, value in values.items()
    }


def blend_ranking(
    base_ranking: list[int],
    signal: dict[int, float],
    blend_weight: float,
) -> list[int]:
    base_scores = _rank_scores(base_ranking)
    signal_scores = _tied_signal_scores(signal)
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
    winner_top5 = 0
    winner_rank_sum = 0
    for row in rows:
        ranking = ranking_for(row)
        top1_win += ranking[0] == row.winner
        top1_place += ranking[0] in row.top3
        winner_rank = ranking.index(row.winner) + 1
        winner_top5 += winner_rank <= 5
        winner_rank_sum += winner_rank
    count = len(rows)
    if not count:
        return {
            "races": 0,
            "top1_win": 0,
            "top1_place": 0,
            "winner_top5": 0,
            "mean_winner_rank": 0.0,
        }
    return {
        "races": count,
        "top1_win": top1_win,
        "top1_place": top1_place,
        "winner_top5": winner_top5,
        "mean_winner_rank": round(winner_rank_sum / count, 4),
    }


def metric_delta(
    baseline: dict[str, float | int],
    candidate: dict[str, float | int],
) -> dict[str, float | int]:
    return {
        key: round(float(candidate[key]) - float(baseline[key]), 4)
        for key in (
            "top1_win",
            "top1_place",
            "winner_top5",
            "mean_winner_rank",
        )
    }


def select_static_candidate(
    historical: list[RaceEvaluation],
) -> tuple[str, float, dict[str, Any]]:
    baseline = metrics(
        historical, lambda row: row.production_ranking
    )
    screened: dict[str, Any] = {}
    choices: list[tuple[tuple[float, ...], str, float]] = []
    signals = {
        "residual": lambda row: row.geometry_residual,
        "performance": lambda row: row.geometry_performance,
    }
    for signal_name, signal_for in signals.items():
        for blend_weight in STATIC_BLEND_WEIGHTS:
            candidate = metrics(
                historical,
                lambda row, signal_for=signal_for, blend_weight=blend_weight: (
                    blend_ranking(
                        row.production_ranking,
                        signal_for(row),
                        blend_weight,
                    )
                ),
            )
            delta = metric_delta(baseline, candidate)
            key = f"{signal_name}_{blend_weight:.2f}"
            screened[key] = {
                "metrics": candidate,
                "delta": delta,
            }
            choices.append(
                (
                    (
                        float(delta["top1_win"]),
                        float(delta["top1_place"]),
                        float(delta["winner_top5"]),
                        -abs(float(delta["mean_winner_rank"])),
                        -blend_weight,
                    ),
                    signal_name,
                    blend_weight,
                )
            )
    _, signal_name, blend_weight = max(choices)
    return signal_name, blend_weight, screened


def train_geometry_model(
    rows: list[RaceEvaluation],
    *,
    feature_count: int | None = None,
) -> HistGradientBoostingClassifier:
    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for row in rows:
        positive_weight = max(len(row.horse_numbers) - 1, 1)
        for feature_row, horse_no in zip(
            row.model_matrix,
            row.horse_numbers,
            strict=True,
        ):
            is_winner = horse_no == row.winner
            features.append(
                feature_row[:feature_count]
                if feature_count is not None
                else feature_row
            )
            labels.append(int(is_winner))
            weights.append(
                float(positive_weight if is_winner else 1.0)
            )
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
    blend_weight: float = DEFAULT_BLEND_WEIGHT,
    preserve_top_n: int = 0,
    feature_count: int | None = None,
) -> list[int]:
    matrix = (
        row.model_matrix[:, :feature_count]
        if feature_count is not None
        else row.model_matrix
    )
    probabilities = model.predict_proba(matrix)[:, 1]
    model_signal = {
        horse_no: float(probability)
        for horse_no, probability in zip(
            row.horse_numbers,
            probabilities,
            strict=True,
        )
    }
    protected = row.production_ranking[:preserve_top_n]
    candidate_pool = row.production_ranking[preserve_top_n:]
    if not candidate_pool:
        return list(row.production_ranking)
    return protected + blend_ranking(
        candidate_pool,
        {
            horse_no: model_signal[horse_no]
            for horse_no in candidate_pool
        },
        blend_weight,
    )


def _date_range(rows: list[RaceEvaluation]) -> dict[str, str | int]:
    if not rows:
        return {"races": 0, "start": "", "end": ""}
    return {
        "races": len(rows),
        "start": rows[0].race_date,
        "end": rows[-1].race_date,
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


def _crossings(
    rows: list[RaceEvaluation],
    ranking_for: Callable[[RaceEvaluation], list[int]],
    *,
    cutoff: int = 5,
) -> dict[str, int]:
    gained = 0
    lost = 0
    unchanged = 0
    gained_popularity: list[int] = []
    lost_popularity: list[int] = []
    for row in rows:
        baseline_rank = row.production_ranking.index(row.winner) + 1
        candidate_rank = ranking_for(row).index(row.winner) + 1
        baseline_hit = baseline_rank <= cutoff
        candidate_hit = candidate_rank <= cutoff
        if candidate_hit and not baseline_hit:
            gained += 1
            if row.winner_popularity is not None:
                gained_popularity.append(row.winner_popularity)
        elif baseline_hit and not candidate_hit:
            lost += 1
            if row.winner_popularity is not None:
                lost_popularity.append(row.winner_popularity)
        else:
            unchanged += 1
    return {
        "gained": gained,
        "lost": lost,
        "net": gained - lost,
        "unchanged": unchanged,
        "gained_popularity_6plus": sum(
            popularity >= 6 for popularity in gained_popularity
        ),
        "lost_popularity_6plus": sum(
            popularity >= 6 for popularity in lost_popularity
        ),
        "gained_mean_popularity": round(
            sum(gained_popularity) / len(gained_popularity), 3
        )
        if gained_popularity
        else 0.0,
        "lost_mean_popularity": round(
            sum(lost_popularity) / len(lost_popularity), 3
        )
        if lost_popularity
        else 0.0,
    }


def select_model_policy(
    model: HistGradientBoostingClassifier,
    validation: list[RaceEvaluation],
    *,
    feature_count: int | None = None,
) -> tuple[int, float, dict[str, Any]]:
    baseline = metrics(
        validation, lambda row: row.production_ranking
    )
    screened: dict[str, Any] = {}
    choices: list[tuple[tuple[float, ...], int, float]] = []
    for preserve_top_n in MODEL_PROTECTED_PREFIXES:
        for blend_weight in MODEL_BLEND_WEIGHTS:
            ranking_for = (
                lambda row,
                preserve_top_n=preserve_top_n,
                blend_weight=blend_weight: model_ranking(
                    model,
                    row,
                    blend_weight=blend_weight,
                    preserve_top_n=preserve_top_n,
                    feature_count=feature_count,
                )
            )
            candidate = metrics(validation, ranking_for)
            delta = metric_delta(baseline, candidate)
            key = f"protect_{preserve_top_n}_blend_{blend_weight:.2f}"
            screened[key] = {
                "metrics": candidate,
                "delta": delta,
                "top5_crossings": _crossings(
                    validation, ranking_for
                ),
            }
            choices.append(
                (
                    (
                        float(delta["winner_top5"]),
                        -float(delta["mean_winner_rank"]),
                        -blend_weight,
                        -float(preserve_top_n),
                    ),
                    preserve_top_n,
                    blend_weight,
                )
            )
    _, preserve_top_n, blend_weight = max(choices)
    return preserve_top_n, blend_weight, screened


def analyze(snapshot: Path) -> dict[str, Any]:
    rows, counters = build_evaluations(snapshot)
    if not rows:
        raise RuntimeError("no settled central full audits could be recomputed")
    max_date = max(date.fromisoformat(row.race_date) for row in rows)
    oot_start = (max_date - timedelta(days=6)).isoformat()
    historical = [row for row in rows if row.race_date < oot_start]
    recent = [row for row in rows if row.race_date >= oot_start]
    signal_name, blend_weight, screened = select_static_candidate(
        historical
    )
    signal_for = (
        (lambda row: row.geometry_residual)
        if signal_name == "residual"
        else (lambda row: row.geometry_performance)
    )
    static_ranking = lambda row: blend_ranking(
        row.production_ranking,
        signal_for(row),
        blend_weight,
    )

    turf_training = [
        row
        for row in rows
        if row.surface == "芝"
        and row.race_date <= MODEL_TRAINING_END
    ]
    geometry_model = train_geometry_model(
        turf_training,
        feature_count=GEOMETRY_MODEL_FEATURE_COUNT,
    )
    style_model = train_geometry_model(turf_training)
    base_only_model = train_geometry_model(
        turf_training,
        feature_count=len(BASE_FEATURE_NAMES),
    )
    turf_holdout = [
        row
        for row in rows
        if row.surface == "芝"
        and row.race_date > MODEL_TRAINING_END
    ]
    turf_validation = [
        row
        for row in turf_holdout
        if row.race_date < oot_start
    ]
    turf_recent = [
        row
        for row in recent
        if row.surface == "芝"
    ]
    (
        model_preserve_top_n,
        model_blend_weight,
        model_screened,
    ) = select_model_policy(
        geometry_model,
        turf_validation,
        feature_count=GEOMETRY_MODEL_FEATURE_COUNT,
    )
    (
        style_preserve_top_n,
        style_blend_weight,
        style_screened,
    ) = select_model_policy(style_model, turf_validation)
    (
        base_only_preserve_top_n,
        base_only_blend_weight,
        base_only_screened,
    ) = select_model_policy(
        base_only_model,
        turf_validation,
        feature_count=len(BASE_FEATURE_NAMES),
    )
    geometry_model_ranking = lambda row: model_ranking(
        geometry_model,
        row,
        blend_weight=model_blend_weight,
        preserve_top_n=model_preserve_top_n,
        feature_count=GEOMETRY_MODEL_FEATURE_COUNT,
    )
    style_model_ranking = lambda row: model_ranking(
        style_model,
        row,
        blend_weight=style_blend_weight,
        preserve_top_n=style_preserve_top_n,
    )
    base_only_model_ranking = lambda row: model_ranking(
        base_only_model,
        row,
        blend_weight=base_only_blend_weight,
        preserve_top_n=base_only_preserve_top_n,
        feature_count=len(BASE_FEATURE_NAMES),
    )

    coverage = {}
    for scope_name, scope_rows in {
        "all": rows,
        "turf": [row for row in rows if row.surface == "芝"],
        "dirt": [row for row in rows if row.surface == "ダート"],
    }.items():
        horses = sum(len(row.horse_numbers) for row in scope_rows)
        reliable = sum(
            value > 0.0
            for row in scope_rows
            for value in row.geometry_reliability.values()
        )
        races_with_three = sum(
            sum(
                value > 0.0
                for value in row.geometry_reliability.values()
            )
            >= 3
            for row in scope_rows
        )
        coverage[scope_name] = {
            "races": len(scope_rows),
            "horses": horses,
            "horses_with_geometry_history": reliable,
            "horse_coverage_pct": round(
                reliable * 100.0 / horses, 2
            )
            if horses
            else 0.0,
            "races_with_3plus_covered_horses": races_with_three,
        }

    monthly = {}
    for month in sorted({row.race_date[:7] for row in rows}):
        month_rows = [
            row for row in rows if row.race_date.startswith(month)
        ]
        monthly[month] = _scope_report(
            month_rows, static_ranking
        )

    return {
        "snapshot": str(snapshot),
        "counts": counters,
        "evaluated": _date_range(rows),
        "oot_start": oot_start,
        "historical": _date_range(historical),
        "recent_oot": _date_range(recent),
        "coverage": coverage,
        "static_candidate": {
            "selected_on_historical_only": True,
            "signal": signal_name,
            "blend_weight": blend_weight,
            "screened_historical": screened,
            "all": _scope_report(rows, static_ranking),
            "historical": _scope_report(
                historical, static_ranking
            ),
            "recent_oot": _scope_report(recent, static_ranking),
            "turf_all": _scope_report(
                [row for row in rows if row.surface == "芝"],
                static_ranking,
            ),
            "turf_recent_oot": _scope_report(
                [
                    row
                    for row in recent
                    if row.surface == "芝"
                ],
                static_ranking,
            ),
            "dirt_all": _scope_report(
                [row for row in rows if row.surface == "ダート"],
                static_ranking,
            ),
            "dirt_recent_oot": _scope_report(
                [
                    row
                    for row in recent
                    if row.surface == "ダート"
                ],
                static_ranking,
            ),
            "monthly": monthly,
        },
        "central_turf_model_candidate": {
            "training": _date_range(turf_training),
            "validation": _date_range(turf_validation),
            "feature_count": GEOMETRY_MODEL_FEATURE_COUNT,
            "geometry_feature_count": len(
                COURSE_GEOMETRY_FEATURE_NAMES
            ),
            "selected_on_validation_only": True,
            "preserve_top_n": model_preserve_top_n,
            "blend_weight": model_blend_weight,
            "screened_validation": model_screened,
            "holdout": _scope_report(
                turf_holdout,
                geometry_model_ranking,
            ),
            "validation_result": _scope_report(
                turf_validation,
                geometry_model_ranking,
            ),
            "recent_oot": _scope_report(
                turf_recent,
                geometry_model_ranking,
            ),
            "holdout_top5_crossings": _crossings(
                turf_holdout, geometry_model_ranking
            ),
            "validation_top5_crossings": _crossings(
                turf_validation, geometry_model_ranking
            ),
            "recent_top5_crossings": _crossings(
                turf_recent, geometry_model_ranking
            ),
            "by_race_date": {
                race_date: _scope_report(
                    [
                        row
                        for row in turf_holdout
                        if row.race_date == race_date
                    ],
                    geometry_model_ranking,
                )
                for race_date in sorted(
                    {row.race_date for row in turf_holdout}
                )
            },
            "base_only_ablation": {
                "preserve_top_n": base_only_preserve_top_n,
                "blend_weight": base_only_blend_weight,
                "screened_validation": base_only_screened,
                "validation_result": _scope_report(
                    turf_validation,
                    base_only_model_ranking,
                ),
                "recent_oot": _scope_report(
                    turf_recent,
                    base_only_model_ranking,
                ),
                "holdout": _scope_report(
                    turf_holdout,
                    base_only_model_ranking,
                ),
            },
            "course_style_candidate": {
                "feature_count": int(
                    turf_training[0].model_matrix.shape[1]
                ),
                "style_feature_count": len(
                    COURSE_STYLE_FEATURE_NAMES
                ),
                "preserve_top_n": style_preserve_top_n,
                "blend_weight": style_blend_weight,
                "screened_validation": style_screened,
                "validation_result": _scope_report(
                    turf_validation,
                    style_model_ranking,
                ),
                "recent_oot": _scope_report(
                    turf_recent,
                    style_model_ranking,
                ),
                "holdout": _scope_report(
                    turf_holdout,
                    style_model_ranking,
                ),
                "holdout_top5_crossings": _crossings(
                    turf_holdout, style_model_ranking
                ),
            },
        },
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

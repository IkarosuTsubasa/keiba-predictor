from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from keiba_llm_agent.config.scoring_config import (
    DEFAULT_CONDITIONAL_WEIGHT_PROFILE,
    DEFAULT_SCORING_MODE,
    DEFAULT_SCORING_WEIGHTS,
    LOCAL_SCORING_MODE,
    LOCAL_SCORING_WEIGHTS,
    NO_CONDITIONAL_WEIGHT_PROFILE,
    effective_scoring_weights,
)
from keiba_llm_agent.schemas.prediction import (
    PedigreeAdjustment,
    Prediction,
    ScoringConfigSnapshot,
)
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.scoring.analysis_adjustment_integrator import (
    build_score_breakdown,
    calculate_pace_adjustment,
    calculate_race_level_adjustment,
)
from keiba_llm_agent.scoring.pedigree_score_adjuster import (
    calculate_pedigree_adjustment,
)
from keiba_llm_agent.scoring.recent_run_scorer import (
    adjustment_score_scale_for_race,
    apply_central_top_choice_refinement,
    calculate_carried_weight_adjustment,
    effective_pedigree_weight_for_horse,
    score_horse_by_recent_runs,
)
from keiba_llm_agent.scoring.scope_reranker import (
    BASE_FEATURE_NAMES,
    MODEL_DIRECTORY,
    MODEL_VERSION,
    RUN_FEATURE_NAMES,
    build_scope_reranker_feature_matrix,
)


TRAINING_CONFIG = {
    "central_turf": {
        "training_end": "2026-05-24",
        "feature_set": "course_geometry",
        "filename": "central_turf_geometry_v1.joblib",
        "min_samples_leaf": 40,
        "blend_weight": 1.0,
        "preserve_top_n": 1,
        "baseline_filename": "central_turf_v1.joblib",
        "baseline_feature_set": "base",
        "baseline_blend_weight": 0.5,
    },
    "local": {
        "training_end": "2026-07-11",
        "feature_set": "local_course_geometry",
        "filename": "local_geometry_v1.joblib",
        "min_samples_leaf": 40,
        "blend_weight": 1.0,
        "preserve_top_n": 2,
        "baseline_filename": "local_v1.joblib",
        "baseline_feature_set": "extended",
        "baseline_blend_weight": 0.5,
    },
}


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def preferred_audits(snapshot: Path) -> dict[str, dict[str, Any]]:
    downloaded = {
        path.name: path
        for path in (snapshot / "agent_backtest_audit").glob("*.json.gz")
    }
    rebuilt = {
        path.name: path
        for path in (
            snapshot / "agent_backtest_audit_rebuilt"
        ).glob("*.json.gz")
    }
    paths = {**rebuilt, **downloaded}
    return {name: read_gzip_json(path) for name, path in paths.items()}


def winner_from_audit(payload: dict[str, Any]) -> int | None:
    def horse_number(value: Any) -> int | None:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    result_payload = (payload.get("outcomes") or {}).get("result") or {}
    direct = result_payload.get("result") or {}
    winner = horse_number(direct.get("1st"))
    if winner is not None:
        return winner
    for row in result_payload.get("finish_order") or []:
        if horse_number(row.get("finish")) == 1:
            winner = horse_number(row.get("horse_no"))
            if winner is not None:
                return winner
    return None


def scope_key(race_data: RaceData) -> str:
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


def scoring_config_for_scope(scope: str) -> ScoringConfigSnapshot:
    if scope == "local":
        return ScoringConfigSnapshot(
            scoring_mode=LOCAL_SCORING_MODE,
            pedigree_weight=LOCAL_SCORING_WEIGHTS["pedigree_weight"],
            race_level_weight=LOCAL_SCORING_WEIGHTS[
                "race_level_weight"
            ],
            pace_weight=LOCAL_SCORING_WEIGHTS["pace_weight"],
            conditional_weight_profile=NO_CONDITIONAL_WEIGHT_PROFILE,
            use_market_score_in_ranking=False,
            market_signal_weight=0.0,
        )
    return ScoringConfigSnapshot(
        scoring_mode=DEFAULT_SCORING_MODE,
        pedigree_weight=DEFAULT_SCORING_WEIGHTS["pedigree_weight"],
        race_level_weight=DEFAULT_SCORING_WEIGHTS[
            "race_level_weight"
        ],
        pace_weight=DEFAULT_SCORING_WEIGHTS["pace_weight"],
        conditional_weight_profile=DEFAULT_CONDITIONAL_WEIGHT_PROFILE,
        use_market_score_in_ranking=False,
        market_signal_weight=0.0,
    )


def analysis_map(items: list[Any]) -> dict[int, Any]:
    return {
        int(item.horse_no): item
        for item in items
        if isinstance(getattr(item, "horse_no", None), int)
    }


def recompute_horse_scores(
    race_data: RaceData,
    saved_prediction: Prediction,
    scoring_config: ScoringConfigSnapshot,
) -> list[Any]:
    effective_weights = effective_scoring_weights(
        scoring_config,
        surface=race_data.race_info.surface,
        field_size=len(race_data.horses),
    )
    scoring_config = scoring_config.model_copy(update=effective_weights)
    deep_by_horse = analysis_map(saved_prediction.deep_analyses)
    pedigree_by_horse = analysis_map(saved_prediction.pedigree_analyses)
    race_level_by_horse = analysis_map(
        saved_prediction.race_level_analyses
    )
    pace_by_horse = analysis_map(saved_prediction.pace_analyses)
    adjustment_scale = adjustment_score_scale_for_race(
        race_data.race_info
    )
    scores = []
    for horse in race_data.horses:
        deep = deep_by_horse.get(horse.horse_no)
        pedigree = pedigree_by_horse.get(horse.horse_no)
        race_level = race_level_by_horse.get(horse.horse_no)
        pace = pace_by_horse.get(horse.horse_no)
        base_score = score_horse_by_recent_runs(
            horse,
            race_data.race_info,
            lessons=saved_prediction.used_lessons,
            use_market_score_in_ranking=False,
            market_signal_weight=0.0,
            field_size=len(race_data.horses),
            deep_analysis=deep,
            pedigree_analysis=pedigree,
            race_level_analysis=race_level,
            pace_analysis=pace,
            race_pace_projection=saved_prediction.race_pace_projection,
        )
        pedigree_adjustment = (
            calculate_pedigree_adjustment(
                pedigree,
                race_data.race_info,
            )
            if pedigree is not None
            else PedigreeAdjustment(reason="血統補正なし")
        )
        race_level_adjustment = calculate_race_level_adjustment(
            race_level
        )
        pace_adjustment = calculate_pace_adjustment(
            pace,
            saved_prediction.race_pace_projection,
        )
        carried = calculate_carried_weight_adjustment(
            horse,
            race_data.horses,
        )
        base_total = round(base_score.total_score + carried, 1)
        pedigree_weight = effective_pedigree_weight_for_horse(
            scoring_config.pedigree_weight,
            horse,
            race_data.race_info,
        )
        breakdown = build_score_breakdown(
            base_total_score=base_total,
            pedigree_adjustment_raw=(
                pedigree_adjustment.pedigree_adjustment
            ),
            pedigree_weight=round(
                pedigree_weight * adjustment_scale,
                3,
            ),
            race_level_adjustment_raw=race_level_adjustment.adjustment,
            race_level_weight=round(
                scoring_config.race_level_weight * adjustment_scale,
                3,
            ),
            pace_adjustment_raw=pace_adjustment.adjustment,
            pace_weight=round(
                scoring_config.pace_weight * adjustment_scale,
                3,
            ),
        )
        scores.append(
            base_score.model_copy(
                update={
                    "base_total_score": base_total,
                    "pedigree_adjustment": pedigree_adjustment,
                    "race_level_adjustment": race_level_adjustment,
                    "pace_adjustment": pace_adjustment,
                    "score_breakdown": breakdown,
                    "total_score": breakdown.total_score,
                }
            )
        )
    scores.sort(key=lambda item: (-item.total_score, item.horse_no))
    refinement = apply_central_top_choice_refinement(
        scores,
        saved_prediction.deep_analyses,
        saved_prediction.pedigree_analyses,
        saved_prediction.race_level_analyses,
        saved_prediction.pace_analyses,
        race_data.race_info,
    )
    if refinement["refinement_applied"]:
        scores = refinement["adjusted_horse_scores"]
        scores.sort(key=lambda item: (-item.total_score, item.horse_no))
    return scores


def training_rows(
    audits: dict[str, dict[str, Any]],
    scope: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    race_ids: list[str] = []
    eligible_races: list[
        tuple[str, str, RaceData, Prediction, int]
    ] = []
    for payload in audits.values():
        if payload.get("audit_kind") != "full":
            continue
        race_data_payload = (payload.get("inputs") or {}).get("race_data")
        winner = winner_from_audit(payload)
        if not race_data_payload or winner is None:
            continue
        race_data = RaceData.model_validate(race_data_payload)
        if scope_key(race_data) != scope:
            continue
        race_date = str(race_data.race_info.race_date or "")
        if not race_date or race_date > str(config["training_end"]):
            continue
        prediction_payload = (payload.get("outputs") or {}).get(
            "prediction"
        )
        if not prediction_payload:
            continue
        saved_prediction = Prediction.model_validate(prediction_payload)
        eligible_races.append(
            (
                race_date,
                race_data.race_info.race_id,
                race_data,
                saved_prediction,
                winner,
            )
        )

    for _, race_id, race_data, saved_prediction, winner in sorted(
        eligible_races,
        key=lambda item: (item[0], item[1]),
    ):
        scoring_config = scoring_config_for_scope(scope)
        horse_scores = recompute_horse_scores(
            race_data,
            saved_prediction,
            scoring_config,
        )
        matrix, horse_numbers = build_scope_reranker_feature_matrix(
            race_data,
            horse_scores,
            str(config["feature_set"]),
        )
        features_by_horse = {
            horse_no: feature_row
            for feature_row, horse_no in zip(
                matrix,
                horse_numbers,
                strict=True,
            )
        }
        positive_weight = max(len(horse_numbers) - 1, 1)
        for horse in race_data.horses:
            horse_no = horse.horse_no
            feature_row = features_by_horse[horse_no]
            is_winner = horse_no == winner
            feature_rows.append(feature_row)
            labels.append(int(is_winner))
            weights.append(
                float(positive_weight if is_winner else 1.0)
            )
        race_ids.append(race_id)

    if not race_ids:
        raise RuntimeError(f"no training races found for scope={scope}")
    return (
        np.asarray(feature_rows, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(weights, dtype=float),
        sorted(race_ids),
    )


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=7,
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=2.0,
        random_state=42,
    )
    model.fit(features, labels, sample_weight=weights)
    return model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(snapshot: Path, output_dir: Path) -> dict[str, Any]:
    audits = preferred_audits(snapshot)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "scikit_learn_version": sklearn.__version__,
        "market_features_used": False,
        "source_snapshot": snapshot.name,
        "models": {},
    }
    for scope, config in TRAINING_CONFIG.items():
        features, labels, weights, race_ids = training_rows(
            audits,
            scope,
            config,
        )
        model = train_model(features, labels, weights, config)
        output_path = output_dir / str(config["filename"])
        joblib.dump(model, output_path, compress=3)
        metadata["models"][scope] = {
            "filename": output_path.name,
            "feature_set": config["feature_set"],
            "training_end": config["training_end"],
            "training_races": len(race_ids),
            "training_horses": int(features.shape[0]),
            "feature_count": int(features.shape[1]),
            "blend_weight": float(config["blend_weight"]),
            "preserve_top_n": int(config["preserve_top_n"]),
            "race_ids_sha256": hashlib.sha256(
                "\n".join(race_ids).encode("utf-8")
            ).hexdigest(),
            "artifact_sha256": sha256(output_path),
        }
        for key in (
            "baseline_filename",
            "baseline_feature_set",
            "baseline_blend_weight",
        ):
            if key in config:
                metadata["models"][scope][key] = config[key]
        baseline_filename = config.get("baseline_filename")
        if baseline_filename:
            baseline_path = output_dir / str(baseline_filename)
            if not baseline_path.exists():
                raise FileNotFoundError(
                    f"baseline model not found: {baseline_path}"
                )
            metadata["models"][scope][
                "baseline_artifact_sha256"
            ] = sha256(baseline_path)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIRECTORY,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = train(
        args.snapshot.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

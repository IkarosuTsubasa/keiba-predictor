"""Prospective workout challenger; its output is audit-only."""
from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

import joblib
import numpy as np
import sklearn

from keiba_llm_agent.analysis.race_class import infer_race_class_level
from keiba_llm_agent.scoring import turf_place_model as active
from keiba_llm_agent.scoring.turf_shadow import default_scoring_config, forward_timing
from keiba_llm_agent.scoring.turf_history_features import valid_history

MODEL_VERSION = "central_turf_workout_shadow_v1_20261026"
MODEL_DIRECTORY = active.ROOT / "keiba_llm_agent/models/workout_shadow"
WORKOUT_NAMES = ("grade", "previous_grade", "grade_change", "prior_grade_age_days")
FEATURE_NAMES = tuple(active.FEATURE_NAMES) + tuple("workout_" + n for n in WORKOUT_NAMES) + tuple(
    "workout_relative_" + n for n in WORKOUT_NAMES)
GRADE_VALUES = {"A": 3., "B": 2., "C": 1., "D": 0.}
SOURCE_FILES = active.SOURCE_FILES + (
    "keiba_llm_agent/scoring/workout_shadow.py", "keiba_llm_agent/scoring/turf_shadow.py",
    "keiba_llm_agent/fetchers/netkeiba_workout_fetcher.py",
)


def source_hashes():
    return {p: hashlib.sha256((active.ROOT / p).read_text(encoding="utf-8").encode()).hexdigest()
            for p in SOURCE_FILES}


def workout_matrix(numbers, horse_ids, target_date, grades, history):
    target = date.fromisoformat(target_date)
    rows = []
    for number, horse_id in zip(numbers, horse_ids, strict=True):
        grade = grades.get(str(number), {})
        if grade and grade.get("horse_id") != horse_id:
            raise ValueError("Workout horse identity mismatch")
        current = GRADE_VALUES.get(grade.get("grade"), np.nan)
        previous = history.get(horse_id)
        old = age = np.nan
        if previous:
            when = date.fromisoformat(previous["date"])
            if when >= target:
                raise ValueError("Workout history must be strictly earlier")
            old = GRADE_VALUES[previous["grade"]]
            age = (target - when).days
        rows.append([current, old, current-old, age])
    values = np.asarray(rows, dtype=float)
    centered = np.full_like(values, np.nan)
    for col in range(4):
        mask = np.isfinite(values[:, col])
        if mask.any():
            centered[mask, col] = values[mask, col] - values[mask, col].mean()
    return np.column_stack((values, centered))


@lru_cache(maxsize=1)
def load_bundle():
    path = MODEL_DIRECTORY / (MODEL_VERSION + ".joblib")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != meta["model_sha256"]
            or meta["model_version"] != MODEL_VERSION or meta["mode"] != "shadow_only"
            or meta["authoritative"] is not False or meta["market_features_used"] is not False
            or meta["feature_names"] != list(FEATURE_NAMES)
            or meta["feature_schema_sha256"] != active.json_hash(list(FEATURE_NAMES))
            or meta["source_hashes"] != source_hashes()
            or meta["sklearn_version"] != sklearn.__version__):
        raise ValueError("Workout shadow artifact contract mismatch")
    bundle = joblib.load(io.BytesIO(raw))
    if bundle["model"].n_features_in_ != len(FEATURE_NAMES) or list(bundle["model"].classes_) != [0, 1]:
        raise ValueError("Workout shadow model shape mismatch")
    _, baseline = active.load_bundle()
    if baseline["model_sha256"] != meta["baseline_model_sha256"]:
        raise ValueError("Workout comparison baseline changed")
    return bundle, meta


def evaluate_workout_shadow(race, prediction, observation):
    if not active.is_central_turf(race):
        return None
    record = {"schema_version": "public_workout_shadow_v1", "model_version": MODEL_VERSION,
              "mode": "shadow_only", "authoritative": False, "race_id": race.race_info.race_id,
              "status": "skipped", "baseline_model_version": prediction.scope_reranker_result.model_version,
              "baseline_top5": list(prediction.marks.values()), "market_features_used": False}
    try:
        mode = os.getenv("KEIBA_WORKOUT_SHADOW_MODE", "shadow")
        if mode != "shadow":
            record.update(status="disabled" if mode == "off" else "failed", reason_code="mode_" + mode)
            return record
        if (not prediction.scope_reranker_result.applied
                or prediction.scope_reranker_result.model_version != active.MODEL_VERSION
                or prediction.scoring_profile != "accuracy_default"
                or prediction.borderline_recovery_enabled
                or prediction.scoring_config.model_dump() != default_scoring_config(race)):
            record["reason_code"] = "baseline_contract_mismatch"
            return record
        if (not observation or observation.get("status") not in ("collected", "partial")
                or not observation.get("available_grades") or observation.get("race_id") != prediction.race_id):
            record["reason_code"] = "workout_unavailable"
            return record
        raw = gzip.decompress(base64.b64decode(observation["raw_response"]["data"]))
        if hashlib.sha256(raw).hexdigest() != observation["raw_response"]["sha256"]:
            raise ValueError("Workout response checksum mismatch")
        # Recheck the persisted source, not just a supplied grade dictionary.
        from keiba_llm_agent.fetchers.netkeiba_workout_fetcher import parse_workout_page
        parsed = parse_workout_page(raw.decode("utf-8"), race)
        if parsed.get("horses") != observation["horses"] or parsed.get("status") not in ("collected", "partial"):
            raise ValueError("Workout parsed source mismatch")
        active.validate_inputs(race)
        bundle, meta = load_bundle()
        now = datetime.now(timezone.utc).isoformat()
        timing = forward_timing(race, meta["training_end"], now)
        received = datetime.fromisoformat(observation["received_at"])
        if received.tzinfo is None or received > datetime.fromisoformat(now):
            raise ValueError("Workout timestamp invalid")
        record.update(training_end=meta["training_end"], timing=timing,
                      workout_received_at=observation["received_at"])
        if not timing["eligible_at_generation"]:
            record["reason_code"] = "not_prospective"
            return record
        clean = race.model_copy(deep=True)
        for horse in clean.horses:
            horse.odds = horse.popularity = None
            for run in horse.recent_runs:
                run.odds = run.popularity = None
        matrix, numbers = active.build_turf_history_matrix(clean, prediction.horse_scores)
        by_number = {h.horse_no: h for h in clean.horses}
        ids = [by_number[n].horse_id for n in numbers]
        extra = workout_matrix(numbers, ids, clean.race_info.race_date, observation["horses"], bundle["history"])
        matrix = np.column_stack((matrix, extra))
        scores = active.predict_scores(bundle["model"], matrix)
        baseline = list(prediction.scope_reranker_result.reranked_ranking)
        if sorted(baseline) != sorted(numbers) or baseline[:5] != record["baseline_top5"]:
            raise ValueError("Baseline ranking mismatch")
        candidate = sorted(numbers, key=lambda n: (-float(scores[numbers.index(n)]), n))
        features = [[None if np.isnan(v) else float(v) for v in row] for row in matrix]
        generated_at = datetime.now(timezone.utc).isoformat()
        record.update(status="predicted", reason_code="shadow_only", generated_at=generated_at,
                      timing=forward_timing(clean, meta["training_end"], generated_at),
                      model_sha256=meta["model_sha256"], baseline_model_sha256=meta["baseline_model_sha256"],
                      training_races=meta["training_races"], feature_schema_sha256=meta["feature_schema_sha256"],
                      history_cutoff=meta["training_end"], history_policy="frozen_training_cohort",
                      baseline_ranking=baseline, candidate_ranking=candidate, candidate_top5=candidate[:5],
                      horse_numbers=numbers, features=features, features_sha256=active.json_hash(features),
                      candidate_scores={str(n): float(v) for n, v in zip(numbers, scores, strict=True)},
                      workout_source_sha256=observation["raw_response"]["sha256"],
                      grade_coverage=observation["available_grades"],
                      strata={"empty_history": not any(valid_history(h, clean.race_info) for h in clean.horses),
                              "graded": infer_race_class_level(clean.race_info.race_class) in (7, 8, 9),
                              "field_13plus": len(numbers) >= 13, "field_16plus": len(numbers) >= 16})
    except Exception as exc:
        record.update(status="failed", reason_code="shadow_evaluation_failed", error_type=type(exc).__name__)
    return record

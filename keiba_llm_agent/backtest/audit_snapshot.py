from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData


SCHEMA_VERSION = "agent_backtest_audit_v1"


def utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in ("", None, [], {})}


def _code_version() -> dict[str, str]:
    return _clean_mapping(
        {
            "github_sha": os.environ.get("GITHUB_SHA", ""),
            "github_ref": os.environ.get("GITHUB_REF", ""),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        }
    )


def _has_recent_runs(race_data_payload: dict[str, Any] | None) -> bool:
    if not race_data_payload:
        return False
    for horse in list(race_data_payload.get("horses") or []):
        if isinstance(horse, dict) and horse.get("recent_runs"):
            return True
    return False


def backtest_capabilities(
    *,
    race_data_payload: dict[str, Any] | None,
    prediction_payload: dict[str, Any] | None,
    result_payload: dict[str, Any] | None = None,
) -> dict[str, bool]:
    return {
        "can_reweight_saved_scores": bool(prediction_payload),
        "can_recompute_scoring_logic": bool(race_data_payload),
        "can_recompute_from_recent_runs": _has_recent_runs(race_data_payload),
        "has_result_for_evaluation": bool(result_payload),
    }


def missing_inputs(
    *,
    race_data_payload: dict[str, Any] | None,
    prediction_payload: dict[str, Any] | None,
    result_payload: dict[str, Any] | None = None,
) -> list[str]:
    missing: list[str] = []
    if not prediction_payload:
        missing.append("prediction")
    if not race_data_payload:
        missing.append("race_data")
    elif not _has_recent_runs(race_data_payload):
        missing.append("recent_runs")
    if not result_payload:
        missing.append("result")
    return missing


def build_backtest_audit_payload(
    *,
    race_data: RaceData | None,
    prediction: Prediction,
    result_payload: dict[str, Any] | None = None,
    created_at: str | None = None,
    source_paths: dict[str, str] | None = None,
    source: str = "keiba_llm_agent",
) -> dict[str, Any]:
    prediction_payload = prediction.model_dump()
    race_data_payload = race_data.model_dump() if race_data is not None else None
    race_info = prediction_payload.get("race_info") or (race_data_payload or {}).get("race_info") or {}
    race_id = str(prediction_payload.get("race_id") or race_info.get("race_id") or "").strip()
    return build_backtest_audit_payload_from_dicts(
        race_id=race_id,
        race_info=race_info,
        prediction_payload=prediction_payload,
        race_data_payload=race_data_payload,
        result_payload=result_payload,
        created_at=created_at,
        source_paths=source_paths,
        source=source,
    )


def build_backtest_audit_payload_from_dicts(
    *,
    race_id: str,
    race_info: dict[str, Any] | None,
    prediction_payload: dict[str, Any] | None,
    race_data_payload: dict[str, Any] | None = None,
    result_payload: dict[str, Any] | None = None,
    created_at: str | None = None,
    source_paths: dict[str, str] | None = None,
    source: str = "keiba_llm_agent",
) -> dict[str, Any]:
    prediction_payload = dict(prediction_payload or {})
    race_data_payload = dict(race_data_payload or {}) if race_data_payload else None
    result_payload = dict(result_payload or {}) if result_payload else None
    race_info = dict(race_info or prediction_payload.get("race_info") or (race_data_payload or {}).get("race_info") or {})
    race_id = str(race_id or prediction_payload.get("race_id") or race_info.get("race_id") or "").strip()
    audit_kind = "full" if race_data_payload else "prediction_only"
    prediction_stage = str(prediction_payload.get("prediction_stage") or "early").strip().lower()
    if prediction_stage not in {"early", "final"}:
        prediction_stage = "early"
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_kind": audit_kind,
        "created_at": created_at or utc_now_text(),
        "source": source,
        "race_id": race_id,
        "prediction_stage": prediction_stage,
        "race_info": race_info,
        "source_paths": dict(source_paths or {}),
        "code_version": _code_version(),
        "backtest_capabilities": backtest_capabilities(
            race_data_payload=race_data_payload,
            prediction_payload=prediction_payload,
            result_payload=result_payload,
        ),
        "missing_inputs": missing_inputs(
            race_data_payload=race_data_payload,
            prediction_payload=prediction_payload,
            result_payload=result_payload,
        ),
        "inputs": {
            "race_data": race_data_payload,
            "lessons": list(prediction_payload.get("used_lessons") or []),
        },
        "scoring": {
            "scoring_profile": prediction_payload.get("scoring_profile"),
            "scoring_mode": prediction_payload.get("scoring_mode"),
            "scoring_config": prediction_payload.get("scoring_config") or {},
            "market_signal_config": prediction_payload.get("market_signal_config") or {},
            "borderline_recovery_config": prediction_payload.get("borderline_recovery_config") or {},
            "scope_reranker_config": prediction_payload.get("scope_reranker_config") or {},
            "horse_scores": list(prediction_payload.get("horse_scores") or []),
            "marks": dict(prediction_payload.get("marks") or {}),
        },
        "analyses": {
            "deep_analyses": list(prediction_payload.get("deep_analyses") or []),
            "pedigree_analyses": list(prediction_payload.get("pedigree_analyses") or []),
            "race_level_analyses": list(prediction_payload.get("race_level_analyses") or []),
            "pace_analyses": list(prediction_payload.get("pace_analyses") or []),
            "race_pace_projection": prediction_payload.get("race_pace_projection"),
            "race_simulation": prediction_payload.get("race_simulation"),
            "borderline_recovery_result": prediction_payload.get("borderline_recovery_result") or {},
            "scope_reranker_result": prediction_payload.get("scope_reranker_result") or {},
        },
        "outcomes": {
            "result": result_payload,
        },
        "outputs": {
            "prediction": prediction_payload,
        },
    }


def audit_path_for_race(output_dir: str | Path, race_id: str) -> Path:
    race_id_text = str(race_id or "").strip()
    if not race_id_text:
        raise ValueError("race_id is required for backtest audit path")
    return Path(output_dir) / f"{race_id_text}.json.gz"


def write_backtest_audit_payload(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    race_id = str(payload.get("race_id") or "").strip()
    output_path = audit_path_for_race(output_dir, race_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return output_path


def read_backtest_audit_payload(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path)
    if audit_path.suffix == ".gz":
        with gzip.open(audit_path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(audit_path.read_text(encoding="utf-8"))


def attach_result_to_backtest_audit(path: str | Path, result_payload: dict[str, Any]) -> Path:
    audit_path = Path(path)
    payload = read_backtest_audit_payload(audit_path)
    payload.setdefault("outcomes", {})["result"] = dict(result_payload or {})
    from pipeline.workout_shadow_metrics import attach_workout_comparison

    attach_workout_comparison(payload, result_payload)
    payload["backtest_capabilities"] = backtest_capabilities(
        race_data_payload=(payload.get("inputs") or {}).get("race_data"),
        prediction_payload=(payload.get("outputs") or {}).get("prediction"),
        result_payload=result_payload,
    )
    payload["missing_inputs"] = missing_inputs(
        race_data_payload=(payload.get("inputs") or {}).get("race_data"),
        prediction_payload=(payload.get("outputs") or {}).get("prediction"),
        result_payload=result_payload,
    )
    with gzip.open(audit_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return audit_path

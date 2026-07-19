from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable


PREDICTION_STAGE_EARLY = "early"
PREDICTION_STAGE_FINAL = "final"


def _normalize_stage(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {PREDICTION_STAGE_FINAL, "final_prediction", "finalized"}:
        return PREDICTION_STAGE_FINAL
    return PREDICTION_STAGE_EARLY


def _read_payload(path: Path, *, compressed: bool) -> dict:
    try:
        if compressed:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _version_identity(root: Path, path: Path) -> tuple[str, str] | None:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 4 or parts[0] != "_versions":
        return None
    stage = str(parts[2] or "").strip().lower()
    if stage not in {PREDICTION_STAGE_EARLY, PREDICTION_STAGE_FINAL}:
        return None
    return str(parts[1] or "").strip(), stage


def _payload_identity(payload: dict) -> tuple[str, str]:
    race_info_value = payload.get("race_info")
    race_info = dict(race_info_value) if isinstance(race_info_value, dict) else {}
    outputs_value = payload.get("outputs")
    outputs = dict(outputs_value) if isinstance(outputs_value, dict) else {}
    prediction_value = outputs.get("prediction")
    prediction = dict(prediction_value) if isinstance(prediction_value, dict) else {}
    prediction_race_info_value = prediction.get("race_info")
    prediction_race_info = (
        dict(prediction_race_info_value) if isinstance(prediction_race_info_value, dict) else {}
    )
    race_id = str(
        payload.get("race_id")
        or race_info.get("race_id")
        or prediction.get("race_id")
        or prediction_race_info.get("race_id")
        or ""
    ).strip()
    stage = _normalize_stage(payload.get("prediction_stage") or prediction.get("prediction_stage"))
    return race_id, stage


def _root_race_id(path: Path, suffix: str) -> str:
    name = path.name
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else path.stem


def _candidate_paths(root: Path, suffix: str) -> list[Path]:
    if not root.exists():
        return []
    paths = list(root.glob(f"*{suffix}"))
    paths.extend((root / "_versions").glob(f"*/{PREDICTION_STAGE_EARLY}/*{suffix}"))
    paths.extend((root / "_versions").glob(f"*/{PREDICTION_STAGE_FINAL}/*{suffix}"))
    return sorted(set(paths))


def _select_effective_paths(
    root: Path,
    candidates: Iterable[Path],
    *,
    suffix: str,
    compressed: bool,
) -> dict[str, Path]:
    selected: dict[str, tuple[tuple[int, int, int, str], Path]] = {}
    for path in candidates:
        version_identity = _version_identity(root, path)
        payload = _read_payload(path, compressed=compressed)
        if not payload:
            continue
        payload_race_id, payload_stage = _payload_identity(payload)
        if version_identity is not None:
            version_race_id, path_stage = version_identity
            if payload_race_id and version_race_id and payload_race_id != version_race_id:
                continue
            race_id = version_race_id or payload_race_id
            stage = path_stage
            version_priority = 1
        else:
            race_id = payload_race_id or _root_race_id(path, suffix)
            stage = payload_stage
            version_priority = 0
        if not race_id:
            continue
        try:
            modified_ns = path.stat().st_mtime_ns
        except OSError:
            modified_ns = 0
        priority = (
            1 if stage == PREDICTION_STAGE_FINAL else 0,
            version_priority,
            modified_ns,
            path.name,
        )
        current = selected.get(race_id)
        if current is None or priority > current[0]:
            selected[race_id] = (priority, path)
    return {race_id: selected[race_id][1] for race_id in sorted(selected)}


def select_effective_prediction_paths(predictions_dir: str | Path) -> dict[str, Path]:
    """Return at most one prediction per race, preferring final over early."""
    root = Path(predictions_dir)
    return _select_effective_paths(
        root,
        _candidate_paths(root, ".json"),
        suffix=".json",
        compressed=False,
    )


def select_effective_prediction_path(predictions_dir: str | Path, race_id: str) -> Path:
    """Resolve one race without treating archived versions as separate samples."""
    root = Path(predictions_dir)
    race_id_text = str(race_id or "").strip()
    candidates: list[Path] = []
    root_path = root / f"{race_id_text}.json"
    if root_path.exists():
        candidates.append(root_path)
    version_root = root / "_versions" / race_id_text
    for stage in (PREDICTION_STAGE_EARLY, PREDICTION_STAGE_FINAL):
        candidates.extend(sorted((version_root / stage).glob("*.json")))
    selected = _select_effective_paths(
        root,
        candidates,
        suffix=".json",
        compressed=False,
    )
    return selected.get(race_id_text, root_path)


def select_effective_backtest_audit_paths(audit_dir: str | Path) -> dict[str, Path]:
    """Return at most one audit per race, preferring final over early."""
    root = Path(audit_dir)
    return _select_effective_paths(
        root,
        _candidate_paths(root, ".json.gz"),
        suffix=".json.gz",
        compressed=True,
    )

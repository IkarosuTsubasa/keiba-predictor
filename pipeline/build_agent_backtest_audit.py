from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from keiba_llm_agent.backtest.audit_snapshot import (  # noqa: E402
    build_backtest_audit_payload,
    build_backtest_audit_payload_from_dicts,
    write_backtest_audit_payload,
)
from keiba_llm_agent.schemas.prediction import Prediction  # noqa: E402
from keiba_llm_agent.schemas.race_data import RaceData  # noqa: E402
from web_data.agent_predictions import (  # noqa: E402
    agent_backtest_audit_write_dir,
    agent_prediction_write_dir,
    agent_result_write_dir,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _race_id_from_prediction_payload(path: Path, payload: dict) -> str:
    race_info = dict(payload.get("race_info") or {})
    return str(payload.get("race_id") or race_info.get("race_id") or path.stem).strip()


def _race_data_candidates(race_id: str, explicit_dir: Path | None = None) -> list[Path]:
    dirs = []
    if explicit_dir is not None:
        dirs.append(explicit_dir)
    dirs.extend(
        [
            REPO_ROOT / "keiba_llm_agent" / "data" / "race_data",
            REPO_ROOT / "data" / "race_data",
            PIPELINE_DIR / "data" / "race_data",
        ]
    )
    out = []
    seen = set()
    for directory in dirs:
        path = directory / f"{race_id}.json"
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _load_optional_race_data(race_id: str, explicit_dir: Path | None = None) -> tuple[RaceData | None, Path | None]:
    for path in _race_data_candidates(race_id, explicit_dir=explicit_dir):
        if not path.exists():
            continue
        try:
            return RaceData.from_json_file(path), path
        except Exception:
            continue
    return None, None


def _load_optional_result_payload(race_id: str, results_dir: Path) -> tuple[dict | None, Path | None]:
    path = results_dir / f"{race_id}.json"
    if not path.exists():
        return None, None
    try:
        return _read_json(path), path
    except Exception:
        return None, None


def _build_one_audit(
    prediction_path: Path,
    *,
    results_dir: Path,
    audit_dir: Path,
    race_data_dir: Path | None = None,
) -> dict:
    prediction_payload = _read_json(prediction_path)
    race_id = _race_id_from_prediction_payload(prediction_path, prediction_payload)
    result_payload, result_path = _load_optional_result_payload(race_id, results_dir)
    race_data, race_data_path = _load_optional_race_data(race_id, explicit_dir=race_data_dir)
    source_paths = {"prediction_path": str(prediction_path)}
    if result_path is not None:
        source_paths["result_path"] = str(result_path)
    if race_data_path is not None:
        source_paths["race_data_path"] = str(race_data_path)

    try:
        prediction = Prediction.model_validate(prediction_payload)
        payload = build_backtest_audit_payload(
            race_data=race_data,
            prediction=prediction,
            result_payload=result_payload,
            source_paths=source_paths,
            source="pipeline_backfill",
        )
    except Exception:
        race_info = dict(prediction_payload.get("race_info") or {})
        payload = build_backtest_audit_payload_from_dicts(
            race_id=race_id,
            race_info=race_info,
            prediction_payload=prediction_payload,
            race_data_payload=race_data.model_dump() if race_data is not None else None,
            result_payload=result_payload,
            source_paths=source_paths,
            source="pipeline_backfill",
        )

    output_path = write_backtest_audit_payload(audit_dir, payload)
    return {
        "race_id": race_id,
        "audit_kind": payload.get("audit_kind", ""),
        "has_result": bool(result_payload),
        "has_race_data": race_data is not None,
        "output_path": str(output_path),
    }


def build_agent_backtest_audit(
    *,
    base_dir: Path = PIPELINE_DIR,
    predictions_dir: Path | None = None,
    results_dir: Path | None = None,
    audit_dir: Path | None = None,
    race_data_dir: Path | None = None,
    include_items: bool = False,
) -> dict:
    predictions_dir = predictions_dir or agent_prediction_write_dir(base_dir)
    results_dir = results_dir or agent_result_write_dir(base_dir)
    audit_dir = audit_dir or agent_backtest_audit_write_dir(base_dir)
    rows = []
    for prediction_path in sorted(predictions_dir.glob("*.json")):
        try:
            rows.append(
                _build_one_audit(
                    prediction_path,
                    results_dir=results_dir,
                    audit_dir=audit_dir,
                    race_data_dir=race_data_dir,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "race_id": prediction_path.stem,
                    "audit_kind": "error",
                    "has_result": False,
                    "has_race_data": False,
                    "output_path": "",
                    "error": str(exc),
                }
            )
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("audit_kind") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "predictions_dir": str(predictions_dir),
        "results_dir": str(results_dir),
        "audit_dir": str(audit_dir),
        "race_count": len(rows),
        "counts": counts,
    }
    if include_items:
        summary["items"] = rows
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Agent prediction backtest audit bundles.")
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--race-data-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    summary = build_agent_backtest_audit(
        predictions_dir=args.predictions_dir,
        results_dir=args.results_dir,
        audit_dir=args.audit_dir,
        race_data_dir=args.race_data_dir,
        include_items=args.verbose,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from predictor_catalog import list_predictors
from race_job_runner import (
    _log_preview,
    _run_subprocess,
    normalize_track_condition_label,
    surface_cli_token,
    validate_prediction_output,
)


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


def _normalize_name(value):
    return "".join(str(value or "").split())


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv_rows(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames or []))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _pick_first_field(fieldnames, candidates):
    for name in candidates:
        if name in fieldnames:
            return name
    return ""


def _parse_mismatch_values(message: str, key: str):
    text = str(message or "").strip()
    marker = f"{key}="
    start = text.find(marker)
    if start < 0:
        return []
    tail = text[start + len(marker) :]
    tail = tail.split(";", 1)[0].strip()
    return [item.strip() for item in tail.split(",") if item.strip()]


def _should_tolerate_validation_mismatch(message: str, reconcile_result: dict):
    text = str(message or "").strip()
    if not text.startswith("odds/predictions horse_name mismatch:"):
        return False, []
    missing_names = _parse_mismatch_values(text, "missing_horse_name")
    extra_names = _parse_mismatch_values(text, "extra_horse_name")
    if not missing_names or extra_names:
        return False, missing_names
    allowed_names = {
        _normalize_name(name)
        for name in list((reconcile_result or {}).get("extra_in_odds") or [])
        if _normalize_name(name)
    }
    if not allowed_names:
        return False, missing_names
    if any(_normalize_name(name) not in allowed_names for name in missing_names):
        return False, missing_names
    return True, missing_names


def reconcile_workspace_entries(workspace: Path):
    odds_path = workspace / "odds.csv"
    shutuba_path = workspace / "shutuba.csv"
    if not odds_path.exists() or not shutuba_path.exists():
        return {
            "status": "skipped",
            "reason": "odds.csv or shutuba.csv missing",
            "changed": False,
        }

    odds_rows, odds_fields = _read_csv_rows(odds_path)
    shutuba_rows, shutuba_fields = _read_csv_rows(shutuba_path)
    odds_name_field = _pick_first_field(odds_fields, ("name", "HorseName", "horse_name"))
    shutuba_name_field = _pick_first_field(shutuba_fields, ("HorseName", "horse_name", "name"))
    if not odds_name_field or not shutuba_name_field:
        return {
            "status": "skipped",
            "reason": "name columns missing",
            "changed": False,
        }

    odds_names = []
    odds_name_set = set()
    for row in odds_rows:
        name = _normalize_name(row.get(odds_name_field, ""))
        if name and name not in odds_name_set:
            odds_name_set.add(name)
            odds_names.append(name)

    shutuba_names = []
    shutuba_name_set = set()
    shutuba_display = {}
    for row in shutuba_rows:
        raw_name = str(row.get(shutuba_name_field, "") or "").strip()
        name = _normalize_name(raw_name)
        if not name:
            continue
        if name not in shutuba_name_set:
            shutuba_name_set.add(name)
            shutuba_names.append(name)
        if name not in shutuba_display and raw_name:
            shutuba_display[name] = raw_name

    if not odds_name_set or not shutuba_name_set:
        return {
            "status": "skipped",
            "reason": "no usable names",
            "changed": False,
            "odds_horses": len(odds_name_set),
            "shutuba_horses": len(shutuba_name_set),
        }

    missing_in_odds = [name for name in shutuba_names if name not in odds_name_set]
    extra_in_odds = [name for name in odds_names if name not in shutuba_name_set]

    if not missing_in_odds:
        return {
            "status": "ok",
            "changed": False,
            "odds_horses": len(odds_name_set),
            "shutuba_horses": len(shutuba_name_set),
            "extra_in_odds": [name for name in extra_in_odds[:8]],
        }

    filtered_rows = [
        row
        for row in shutuba_rows
        if _normalize_name(row.get(shutuba_name_field, "")) in odds_name_set
    ]
    filtered_name_set = {
        _normalize_name(row.get(shutuba_name_field, ""))
        for row in filtered_rows
        if _normalize_name(row.get(shutuba_name_field, ""))
    }
    if not filtered_rows or not filtered_name_set:
        return {
            "status": "failed",
            "reason": "reconciliation would remove all shutuba rows",
            "changed": False,
            "odds_horses": len(odds_name_set),
            "shutuba_horses": len(shutuba_name_set),
            "missing_in_odds": [shutuba_display.get(name, name) for name in missing_in_odds[:8]],
        }

    _write_csv_rows(shutuba_path, shutuba_fields, filtered_rows)
    return {
        "status": "filtered",
        "changed": True,
        "odds_horses": len(odds_name_set),
        "shutuba_horses_before": len(shutuba_name_set),
        "shutuba_horses_after": len(filtered_name_set),
        "removed_names": [shutuba_display.get(name, name) for name in missing_in_odds[:12]],
        "extra_in_odds": [name for name in extra_in_odds[:8]],
        "removed_rows": max(0, len(shutuba_rows) - len(filtered_rows)),
    }


def _load_meta(workspace: Path):
    meta_path = workspace / "task_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"task_meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _safe_text(value, fallback=""):
    text = str(value or "").strip()
    return text or fallback


def _emit_batch_log(event, **fields):
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": str(event or "").strip() or "unknown",
    }
    payload.update(fields)
    print("[remote_predictor] " + json.dumps(payload, ensure_ascii=False), flush=True)


def _write_batch_summary(workspace: Path, summary: dict):
    (workspace / "remote_predictor_summary.json").write_text(
        json.dumps(dict(summary or {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_batch(workspace_dir: str):
    workspace = Path(workspace_dir).resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"workspace not found: {workspace}")
    meta = _load_meta(workspace)
    scope_key = _safe_text(meta.get("scope_key"))
    surface = _safe_text(meta.get("surface"), "dirt")
    distance = _safe_text(meta.get("distance"), "1600")
    track_cond_label = normalize_track_condition_label(_safe_text(meta.get("track_condition"), "良"))
    surface_token = surface_cli_token(surface)
    target_location = _safe_text(meta.get("location"))
    race_date = _safe_text(meta.get("race_date"))
    odds_src = workspace / "odds.csv"
    if not odds_src.exists():
        raise FileNotFoundError(f"odds.csv not found: {odds_src}")

    _emit_batch_log(
        "batch_start",
        workspace=str(workspace),
        scope_key=scope_key,
        surface=surface,
        distance=distance,
        track_condition=track_cond_label,
        location=target_location,
        race_date=race_date,
    )

    process_log = []
    output_paths = {}
    completed_predictors = []

    reconcile_result = reconcile_workspace_entries(workspace)
    process_log.append(
        {
            "step": "reconcile_workspace_entries",
            "status": reconcile_result.get("status", ""),
            "details": reconcile_result,
        }
    )
    _emit_batch_log("reconcile_workspace_entries", **reconcile_result)
    if reconcile_result.get("status") == "failed":
        failure_message = str(reconcile_result.get("reason", "") or "workspace reconciliation failed")
        failure_summary = {
            "status": "failed",
            "scope_key": scope_key,
            "surface": surface,
            "distance": distance,
            "track_condition": track_cond_label,
            "completed_predictors": list(completed_predictors),
            "output_paths": dict(output_paths),
            "failed_predictor": "",
            "failed_stage": "reconcile_workspace_entries",
            "failure_message": failure_message,
            "process_log": process_log,
        }
        _write_batch_summary(workspace, failure_summary)
        _emit_batch_log(
            "batch_failed",
            failed_predictor="",
            failed_stage="reconcile_workspace_entries",
            message=failure_message,
        )
        raise RuntimeError(failure_message)

    for spec in list_predictors():
        script_name = _safe_text(spec.get("script_name"))
        latest_name = _safe_text(spec.get("latest_filename"))
        if not script_name or not latest_name:
            continue
        pred_latest_path = workspace / latest_name
        if pred_latest_path.exists():
            pred_latest_path.unlink()
        predictor_start = __import__("datetime").datetime.now().timestamp()
        _emit_batch_log(
            "predictor_start",
            predictor_id=spec["id"],
            label=_safe_text(spec.get("label")),
            script_name=script_name,
            output_name=latest_name,
        )

        if spec["id"] == "main":
            pred_code, pred_output = _run_subprocess(
                ROOT_DIR / script_name,
                cwd=workspace,
                inputs=[surface, distance, track_cond_label],
                env={
                    "SCOPE_KEY": scope_key,
                    "PREDICTOR_NO_PROMPT": "1",
                    "PREDICTOR_TARGET_SURFACE": surface,
                    "PREDICTOR_TARGET_DISTANCE": distance,
                    "PREDICTOR_TARGET_CONDITION": track_cond_label,
                },
            )
        elif spec["id"] == "v2_opus":
            pred_code, pred_output = _run_subprocess(
                ROOT_DIR / script_name,
                cwd=workspace,
                inputs=[surface_token, distance, track_cond_label],
                env={
                    "SCOPE_KEY": scope_key,
                    "PREDICTIONS_OUTPUT": str(pred_latest_path),
                    "PREDICTOR_NO_PROMPT": "1",
                    "PREDICTOR_NO_WAIT": "1",
                    "PREDICTOR_TARGET_SURFACE": surface,
                    "PREDICTOR_TARGET_DISTANCE": distance,
                    "PREDICTOR_TARGET_CONDITION": track_cond_label,
                },
            )
        elif spec["id"] == "v3_premium":
            pred_code, pred_output = _run_subprocess(
                ROOT_DIR / script_name,
                cwd=workspace,
                env={"SCOPE_KEY": scope_key},
                script_args=[
                    "--base-dir",
                    str(workspace),
                    "--output",
                    latest_name,
                    "--race-venue",
                    target_location or "",
                    "--race-surface",
                    surface,
                    "--race-distance",
                    distance or "1800",
                    "--race-going",
                    track_cond_label,
                    "--no-prompt",
                    "--no-wait",
                ],
            )
        elif spec["id"] == "v4_gemini":
            pred_code, pred_output = _run_subprocess(
                ROOT_DIR / script_name,
                cwd=workspace,
                env={
                    "SCOPE_KEY": scope_key,
                    "PREDICTIONS_OUTPUT": str(pred_latest_path),
                    "PREDICTOR_TARGET_LOCATION": target_location,
                    "PREDICTOR_TARGET_SURFACE": surface,
                    "PREDICTOR_TARGET_DISTANCE": distance or "1800",
                    "PREDICTOR_TARGET_CONDITION": track_cond_label,
                },
            )
        elif spec["id"] == "v5_stacking":
            pred_code, pred_output = _run_subprocess(
                ROOT_DIR / script_name,
                cwd=workspace,
                env={
                    "SCOPE_KEY": scope_key,
                    "PREDICTIONS_OUTPUT": str(pred_latest_path),
                    "PREDICTOR_TARGET_LOCATION": target_location,
                    "PREDICTOR_TARGET_SURFACE": surface,
                    "PREDICTOR_TARGET_DISTANCE": distance or "1800",
                    "PREDICTOR_TARGET_CONDITION": track_cond_label,
                    "PREDICTOR_TARGET_DATE": race_date,
                    "PREDICTOR_NO_PROMPT": "1",
                    "ODDS_PATH": str(workspace / "odds.csv"),
                    "FUKU_ODDS_PATH": str(workspace / "fuku_odds.csv"),
                    "WIDE_ODDS_PATH": str(workspace / "wide_odds.csv"),
                    "QUINELLA_ODDS_PATH": str(workspace / "quinella_odds.csv"),
                    "EXACTA_ODDS_PATH": str(workspace / "exacta_odds.csv"),
                    "TRIO_ODDS_PATH": str(workspace / "trio_odds.csv"),
                },
            )
        elif spec["id"] == "v6_kiwami":
            pred_code, pred_output = _run_subprocess(
                ROOT_DIR / script_name,
                cwd=workspace,
                env={
                    "SCOPE_KEY": scope_key,
                    "PREDICTIONS_OUTPUT": str(pred_latest_path),
                    "PREDICTOR_TARGET_LOCATION": target_location,
                    "PREDICTOR_TARGET_SURFACE": surface,
                    "PREDICTOR_TARGET_DISTANCE": distance or "1800",
                    "PREDICTOR_TARGET_CONDITION": track_cond_label,
                    "PREDICTOR_TARGET_DATE": race_date,
                    "PREDICTOR_NO_PROMPT": "1",
                    "ODDS_PATH": str(workspace / "odds.csv"),
                    "FUKU_ODDS_PATH": str(workspace / "fuku_odds.csv"),
                    "WIDE_ODDS_PATH": str(workspace / "wide_odds.csv"),
                    "QUINELLA_ODDS_PATH": str(workspace / "quinella_odds.csv"),
                    "EXACTA_ODDS_PATH": str(workspace / "exacta_odds.csv"),
                    "TRIO_ODDS_PATH": str(workspace / "trio_odds.csv"),
                },
            )
        else:
            continue

        _emit_batch_log(
            "predictor_subprocess_done",
            predictor_id=spec["id"],
            code=pred_code,
            output_name=latest_name,
            output_preview=_log_preview(pred_output),
        )
        process_log.append(
            {
                "step": f"predictor_{spec['id']}",
                "code": pred_code,
                "output_preview": _log_preview(pred_output),
            }
        )
        if pred_code != 0:
            failure_summary = {
                "status": "failed",
                "scope_key": scope_key,
                "surface": surface,
                "distance": distance,
                "track_condition": track_cond_label,
                "completed_predictors": list(completed_predictors),
                "output_paths": dict(output_paths),
                "failed_predictor": spec["id"],
                "failed_stage": "subprocess",
                "failure_message": f"{spec['label']} subprocess failed with exit_code={pred_code}",
                "process_log": process_log,
            }
            _write_batch_summary(workspace, failure_summary)
            _emit_batch_log(
                "predictor_failed",
                predictor_id=spec["id"],
                stage="subprocess",
                code=pred_code,
            )
            _emit_batch_log(
                "batch_failed",
                failed_predictor=spec["id"],
                failed_stage="subprocess",
                message=failure_summary["failure_message"],
            )
            raise RuntimeError(failure_summary["failure_message"])
        _emit_batch_log(
            "predictor_validate_start",
            predictor_id=spec["id"],
            output_name=latest_name,
            output_path=str(pred_latest_path),
        )
        ok, msg = validate_prediction_output(predictor_start, pred_latest_path, odds_src)
        tolerated, tolerated_missing = _should_tolerate_validation_mismatch(msg, reconcile_result)
        if not ok and tolerated:
            process_log.append(
                {
                    "step": f"predictor_{spec['id']}_validation",
                    "status": "tolerated",
                    "details": {
                        "message": msg,
                        "tolerated_missing_horses": tolerated_missing,
                    },
                }
            )
            _emit_batch_log(
                "predictor_validate_tolerated",
                predictor_id=spec["id"],
                message=msg,
                tolerated_missing_horses=tolerated_missing,
            )
            ok = True
        if not ok:
            failure_summary = {
                "status": "failed",
                "scope_key": scope_key,
                "surface": surface,
                "distance": distance,
                "track_condition": track_cond_label,
                "completed_predictors": list(completed_predictors),
                "output_paths": dict(output_paths),
                "failed_predictor": spec["id"],
                "failed_stage": "validation",
                "failure_message": msg,
                "process_log": process_log,
            }
            _write_batch_summary(workspace, failure_summary)
            _emit_batch_log(
                "predictor_failed",
                predictor_id=spec["id"],
                stage="validation",
                message=msg,
                output_name=latest_name,
                output_path=str(pred_latest_path),
            )
            _emit_batch_log(
                "batch_failed",
                failed_predictor=spec["id"],
                failed_stage="validation",
                message=msg,
            )
            raise RuntimeError(f"{spec['label']} failed during validation: {msg}")
        output_paths[spec["id"]] = str(pred_latest_path)
        completed_predictors.append(spec["id"])
        _emit_batch_log(
            "predictor_done",
            predictor_id=spec["id"],
            output_name=latest_name,
            output_path=str(pred_latest_path),
        )

    summary = {
        "status": "succeeded",
        "scope_key": scope_key,
        "surface": surface,
        "distance": distance,
        "track_condition": track_cond_label,
        "completed_predictors": list(completed_predictors),
        "output_paths": output_paths,
        "process_log": process_log,
    }
    _write_batch_summary(workspace, summary)
    _emit_batch_log(
        "batch_done",
        completed_predictors=summary["completed_predictors"],
        output_paths=summary["output_paths"],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run remote predictor batch v1-v6")
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    summary = run_batch(args.workspace)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

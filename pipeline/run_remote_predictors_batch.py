import argparse
import json
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


def _load_meta(workspace: Path):
    meta_path = workspace / "task_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"task_meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _safe_text(value, fallback=""):
    text = str(value or "").strip()
    return text or fallback


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

    process_log = []
    output_paths = {}

    for spec in list_predictors():
        script_name = _safe_text(spec.get("script_name"))
        latest_name = _safe_text(spec.get("latest_filename"))
        if not script_name or not latest_name:
            continue
        pred_latest_path = workspace / latest_name
        if pred_latest_path.exists():
            pred_latest_path.unlink()
        predictor_start = __import__("datetime").datetime.now().timestamp()

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
                    "TRIFECTA_ODDS_PATH": str(workspace / "trifecta_odds.csv"),
                },
            )
        else:
            continue

        process_log.append(
            {"step": f"predictor_{spec['id']}", "code": pred_code, "output_preview": _log_preview(pred_output)}
        )
        if pred_code != 0:
            raise RuntimeError(f"{spec['label']} failed: {pred_output}")
        ok, msg = validate_prediction_output(predictor_start, pred_latest_path, odds_src)
        if not ok:
            raise RuntimeError(f"{spec['label']} failed: {msg}\n{pred_output}")
        output_paths[spec["id"]] = str(pred_latest_path)

    summary = {
        "scope_key": scope_key,
        "surface": surface,
        "distance": distance,
        "track_condition": track_cond_label,
        "completed_predictors": [spec["id"] for spec in list_predictors() if spec["id"] in output_paths],
        "output_paths": output_paths,
        "process_log": process_log,
    }
    (workspace / "remote_predictor_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run remote predictor batch v1-v5")
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    summary = run_batch(args.workspace)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

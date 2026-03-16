import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from local_env import load_local_env
from predictor_catalog import list_predictors, snapshot_prediction_path
from race_job_store import get_job, initialize_job_step_fields, set_job_step_state, update_job
from surface_scope import get_data_dir, migrate_legacy_data


BASE_DIR = Path(__file__).resolve().parent
load_local_env(BASE_DIR, override=False)

ODDS_EXTRACT_TIMEOUT_SECONDS = 300


def strict_llm_odds_gate_enabled():
    raw = os.environ.get("PIPELINE_BLOCK_LLM_ON_ODDS_WARNING", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def expected_odds_output_names(scope_key):
    return [
        "odds.csv",
        "fuku_odds.csv",
        "wide_odds.csv",
        "quinella_odds.csv",
        "exacta_odds.csv",
        "trio_odds.csv",
        "trifecta_odds.csv",
    ]


def validate_workspace_odds_outputs(workspace_dir, scope_key):
    workspace = Path(workspace_dir)
    missing = [name for name in expected_odds_output_names(scope_key) if not (workspace / name).exists()]
    if missing:
        return False, f"incomplete odds outputs: {', '.join(missing)}"
    return True, ""


def _mark_job_processing_started(row, now_text):
    row.update(initialize_job_step_fields(row))
    row["status"] = "processing"
    row["processing_started_at"] = now_text
    row["error_message"] = ""
    row["last_process_output"] = ""
    row["last_settlement_output"] = str(row.get("last_settlement_output", "") or "")
    set_job_step_state(row, "odds", "running", now_text)
    set_job_step_state(row, "predictor", "idle")
    set_job_step_state(row, "policy", "idle")


def _mark_odds_succeeded_and_predictor_running(row, now_text):
    row.update(initialize_job_step_fields(row))
    set_job_step_state(row, "odds", "succeeded", now_text)
    set_job_step_state(row, "predictor", "running", now_text)


def _mark_predictor_succeeded_and_policy_running(row, now_text):
    row.update(initialize_job_step_fields(row))
    set_job_step_state(row, "predictor", "succeeded", now_text)
    set_job_step_state(row, "policy", "running", now_text)


def _mark_policy_succeeded_and_ready(row, now_text, run_id, summary, refreshed_job):
    row.update(initialize_job_step_fields(row))
    row["status"] = "ready"
    row["ready_at"] = now_text
    row["current_run_id"] = run_id
    row["error_message"] = ""
    row["last_process_output"] = json.dumps(summary, ensure_ascii=False, indent=2)
    row["queued_process_at"] = str((refreshed_job or {}).get("queued_process_at", "") or "")
    set_job_step_state(row, "policy", "succeeded", now_text)


def _mark_job_failed(row, now_text, message):
    row.update(initialize_job_step_fields(row))
    error_text = str(message or "")
    previous_status = str(row.get("status", "") or "").strip().lower()
    row["status"] = "failed"
    row["error_message"] = error_text
    if previous_status == "settling":
        row["last_settlement_output"] = error_text
    else:
        row["last_process_output"] = error_text
    for step_name in ("settlement", "policy", "predictor", "odds"):
        step_status = str(row.get(f"{step_name}_status", "") or "").strip().lower()
        if step_status in ("queued", "running"):
            set_job_step_state(row, step_name, "failed", now_text, error_text)
            break
    else:
        if str(row.get("settling_started_at", "") or "").strip():
            set_job_step_state(row, "settlement", "failed", now_text, error_text)
        elif str(row.get("current_run_id", "") or "").strip():
            set_job_step_state(row, "policy", "failed", now_text, error_text)
        elif str(row.get("processing_started_at", "") or "").strip():
            set_job_step_state(row, "odds", "failed", now_text, error_text)


def _mark_settlement_started(row, now_text, names):
    row.update(initialize_job_step_fields(row))
    row["status"] = "settling"
    row["settling_started_at"] = now_text
    row["actual_top1"] = names[0]
    row["actual_top2"] = names[1]
    row["actual_top3"] = names[2]
    row["error_message"] = ""
    set_job_step_state(row, "settlement", "running", now_text)


def _mark_settlement_succeeded(row, now_text, names, output):
    row.update(initialize_job_step_fields(row))
    row["status"] = "settled"
    row["settled_at"] = now_text
    row["actual_top1"] = names[0]
    row["actual_top2"] = names[1]
    row["actual_top3"] = names[2]
    row["last_settlement_output"] = output
    row["error_message"] = ""
    set_job_step_state(row, "settlement", "succeeded", now_text)


def get_env_timeout(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        value = int(float(raw))
    except ValueError:
        return int(default)
    if value <= 0:
        return int(default)
    return value


def _race_url(scope_key, race_id):
    race_id_text = "".join(ch for ch in str(race_id or "").strip() if ch.isdigit())
    if not race_id_text:
        return ""
    if str(scope_key or "").strip() == "local":
        return f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id_text}"
    return f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id_text}"


def _default_predictor_surface(scope_key):
    return "turf" if str(scope_key or "").strip() == "central_turf" else "dirt"


def _job_predictor_surface(job):
    raw = str((job or {}).get("target_surface", "") or "").strip().lower()
    if raw in ("t", "turf", "grass", "shiba", "芝"):
        return "turf"
    if raw in ("d", "dirt", "sand", "ダ"):
        return "dirt"
    return _default_predictor_surface((job or {}).get("scope_key", ""))


def _job_predictor_distance(job):
    raw = str((job or {}).get("target_distance", "") or "").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1600
    return str(value)


def _job_predictor_track(job):
    raw = str((job or {}).get("target_track_condition", "") or "").strip()
    return raw or "良"


def _job_predictor_location(job):
    return str((job or {}).get("location", "") or "").strip()


def _job_race_date(job):
    return str((job or {}).get("race_date", "") or "").strip()


def normalize_track_condition_label(value):
    raw = str(value or "").strip()
    raw_lower = raw.lower()
    mapping = {
        "good": "\u826f",
        "firm": "\u826f",
        "slightly_heavy": "\u7a0d\u91cd",
        "slightly heavy": "\u7a0d\u91cd",
        "heavy": "\u91cd",
        "bad": "\u4e0d\u826f",
    }
    return mapping.get(raw_lower, raw or "\u826f")


def surface_cli_token(surface_value):
    return "dirt" if str(surface_value or "").strip().lower() == "dirt" else "turf"


def load_name_set(path, field):
    src = Path(path)
    if not src.exists():
        return None, f"{src} not found."
    with open(src, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if field not in fieldnames:
            return None, f"{src} missing column: {field}"
        names = {str(row.get(field, "") or "").strip() for row in reader if str(row.get(field, "") or "").strip()}
    if not names:
        return None, f"{src} has no rows for {field}"
    return names, ""


def validate_odds_predictions(odds_path, pred_path):
    odds_names, err = load_name_set(odds_path, "name")
    if odds_names is None:
        return False, err
    pred_names, err = load_name_set(pred_path, "HorseName")
    if pred_names is None:
        return False, err
    matches = odds_names & pred_names
    base = min(len(odds_names), len(pred_names))
    ratio = (len(matches) / base) if base else 0.0
    if len(matches) < 3 or ratio < 0.6:
        return False, f"odds/predictions mismatch: matches={len(matches)} ratio={ratio:.2f}"
    return True, ""


def csv_has_rows(path, min_rows=1):
    src = Path(path)
    if not src.exists():
        return False
    with open(src, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        count = 0
        for _ in reader:
            count += 1
            if count >= min_rows:
                return True
    return False


def validate_prediction_output(start_ts, pred_path, odds_path):
    pred_file = Path(pred_path)
    if not pred_file.exists():
        return False, f"{pred_file.name} not generated."
    try:
        if pred_file.stat().st_mtime < start_ts - 1:
            return False, f"{pred_file.name} not updated."
    except OSError:
        return False, f"{pred_file.name} stat unavailable."
    if not csv_has_rows(pred_file):
        return False, f"{pred_file.name} has no rows."
    ok, msg = validate_odds_predictions(odds_path, pred_file)
    if not ok:
        return False, msg
    return True, ""


def _shared_workspace_dir(base_dir):
    path = Path(base_dir) / "data" / "_shared" / "job_workspaces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    fieldnames = list(row.keys())
    if path.exists():
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
        for name in existing_fields:
            if name not in fieldnames:
                fieldnames.append(name)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in existing_rows + [row]:
            writer.writerow({name: item.get(name, "") for name in fieldnames})


def _copy_if_exists(src, dest):
    src_path = Path(src)
    if not src_path.exists():
        return ""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    return str(dest_path)


def _run_subprocess(script_path, *, cwd, inputs=None, env=None, timeout_seconds=None, script_args=None):
    payload = ""
    if inputs is not None:
        payload = "\n".join(str(item) for item in list(inputs or [])) + "\n"
    run_env = os.environ.copy()
    run_env.setdefault("PYTHONIOENCODING", "utf-8")
    run_env.setdefault("PYTHONUTF8", "1")
    if env:
        run_env.update(env)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), *(list(script_args or []))],
            input=payload,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=str(cwd),
            env=run_env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        stderr_text = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        output = f"subprocess timeout after {int(timeout_seconds or 0)}s"
        if stdout_text:
            output = f"{output}\n{stdout_text}"
        if stderr_text:
            output = f"{output}\n[stderr]\n{stderr_text}"
        return 124, output.strip()
    output = (result.stdout or "").strip()
    if result.stderr:
        output = f"{output}\n[stderr]\n{result.stderr.strip()}".strip()
    return result.returncode, output


def _snapshot_outputs(base_dir, scope_key, race_id, run_id, workspace_dir):
    data_dir = get_data_dir(base_dir, scope_key)
    race_dir = data_dir / race_id
    race_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{race_id}"
    prediction_files = {
        "odds_path": ("odds.csv", f"odds_{run_id}{suffix}.csv"),
        "fuku_odds_path": ("fuku_odds.csv", f"fuku_odds_{run_id}{suffix}.csv"),
        "wide_odds_path": ("wide_odds.csv", f"wide_odds_{run_id}{suffix}.csv"),
        "quinella_odds_path": ("quinella_odds.csv", f"quinella_odds_{run_id}{suffix}.csv"),
        "exacta_odds_path": ("exacta_odds.csv", f"exacta_odds_{run_id}{suffix}.csv"),
        "trio_odds_path": ("trio_odds.csv", f"trio_odds_{run_id}{suffix}.csv"),
        "trifecta_odds_path": ("trifecta_odds.csv", f"trifecta_odds_{run_id}{suffix}.csv"),
    }
    out = {}
    for spec in list_predictors():
        field_name = str(spec.get("run_field", "") or "").strip()
        latest_name = str(spec.get("latest_filename", "") or "").strip()
        if not field_name or not latest_name:
            continue
        dest_path = snapshot_prediction_path(data_dir, race_id, run_id, spec["id"])
        out[field_name] = _copy_if_exists(Path(workspace_dir) / latest_name, dest_path)
    for field_name, (src_name, dest_name) in prediction_files.items():
        out[field_name] = _copy_if_exists(Path(workspace_dir) / src_name, race_dir / dest_name)
    return out


def _build_run_row(job, run_id, snapshot_paths):
    race_id = str(job.get("race_id", "") or "").strip()
    scope_key = str(job.get("scope_key", "") or "").strip()
    race_url = _race_url(scope_key, race_id)
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "race_url": race_url,
        "race_id": race_id,
        "history_url": "",
        "trigger_race": "",
        "scope": scope_key,
        "location": str(job.get("location", "") or "").strip(),
        "race_date": str(job.get("race_date", "") or "").strip(),
        "surface": _job_predictor_surface(job),
        "distance": _job_predictor_distance(job),
        "track_condition": _job_predictor_track(job),
        "budget_yen": "2000,5000,10000,50000",
        "style": "scheduled",
        "strategy": "scheduled",
        "strategy_reason": "race_job_runner",
        "predictor_strategy": "scheduled",
        "predictor_reason": "race_job_runner",
        "config_version": "",
        "predictions_path": snapshot_paths.get("predictions_path", ""),
        "predictions_v2_opus_path": snapshot_paths.get("predictions_v2_opus_path", ""),
        "predictions_v3_premium_path": snapshot_paths.get("predictions_v3_premium_path", ""),
        "predictions_v4_gemini_path": snapshot_paths.get("predictions_v4_gemini_path", ""),
        "predictions_v5_stacking_path": snapshot_paths.get("predictions_v5_stacking_path", ""),
        "odds_path": snapshot_paths.get("odds_path", ""),
        "wide_odds_path": snapshot_paths.get("wide_odds_path", ""),
        "fuku_odds_path": snapshot_paths.get("fuku_odds_path", ""),
        "quinella_odds_path": snapshot_paths.get("quinella_odds_path", ""),
        "exacta_odds_path": snapshot_paths.get("exacta_odds_path", ""),
        "trio_odds_path": snapshot_paths.get("trio_odds_path", ""),
        "trifecta_odds_path": snapshot_paths.get("trifecta_odds_path", ""),
        "plan_path": "",
        "gemini_policy_path": "",
        "siliconflow_policy_path": "",
        "openai_policy_path": "",
        "grok_policy_path": "",
        "tickets": "",
        "amount_yen": "",
    }


def process_race_job(base_dir, job_id, policy_engines=None):
    base_path = Path(base_dir)
    job = get_job(base_path, job_id)
    if job is None:
        raise ValueError(f"race job not found: {job_id}")
    race_id = str(job.get("race_id", "") or "").strip()
    scope_key = str(job.get("scope_key", "") or "").strip()
    if not race_id or not scope_key:
        raise ValueError("race job missing race_id or scope_key")
    artifact_map = {
        str(item.get("artifact_type", "")).strip().lower(): dict(item)
        for item in list(job.get("artifacts", []) or [])
        if isinstance(item, dict)
    }
    kachiuma_path = str((artifact_map.get("kachiuma") or {}).get("stored_path", "") or "").strip()
    shutuba_path = str((artifact_map.get("shutuba") or {}).get("stored_path", "") or "").strip()
    if not kachiuma_path or not Path(kachiuma_path).exists():
        raise ValueError("kachiuma artifact missing")
    if not shutuba_path or not Path(shutuba_path).exists():
        raise ValueError("shutuba artifact missing")

    migrate_legacy_data(base_path, scope_key)
    data_dir = get_data_dir(base_path, scope_key)
    data_dir.mkdir(parents=True, exist_ok=True)

    update_job(
        base_path,
        job_id,
        lambda row, now_text: _mark_job_processing_started(row, now_text),
    )

    summary = {
        "job_id": job_id,
        "race_id": race_id,
        "scope_key": scope_key,
        "process_log": [],
        "run_id": "",
        "policy_engines": [],
    }
    workspace_root = _shared_workspace_dir(base_path)
    with tempfile.TemporaryDirectory(prefix=f"{job_id}_", dir=str(workspace_root)) as tmp_dir:
        workspace = Path(tmp_dir)
        shutil.copy2(kachiuma_path, workspace / "kachiuma.csv")
        shutil.copy2(shutuba_path, workspace / "shutuba.csv")

        race_url = _race_url(scope_key, race_id)
        if not race_url:
            raise ValueError("failed to build race url")

        odds_code, odds_output = _run_subprocess(
            base_path.parent / "odds_extract.py",
            cwd=workspace,
            inputs=[race_url],
            env={"SCOPE_KEY": scope_key},
            timeout_seconds=get_env_timeout(
                "PIPELINE_ODDS_EXTRACT_TIMEOUT", ODDS_EXTRACT_TIMEOUT_SECONDS
            ),
        )
        summary["process_log"].append({"step": "odds_extract", "code": odds_code, "output": odds_output})
        if odds_code != 0 or not (workspace / "odds.csv").exists():
            raise RuntimeError(f"odds extraction failed: {odds_output}")
        if strict_llm_odds_gate_enabled():
            odds_ok, odds_error = validate_workspace_odds_outputs(workspace, scope_key)
            if not odds_ok:
                raise RuntimeError(f"odds extraction incomplete: {odds_error}\n{odds_output}")
        update_job(
            base_path,
            job_id,
            lambda row, now_text: _mark_odds_succeeded_and_predictor_running(row, now_text),
        )

        surface = _job_predictor_surface(job)
        distance = _job_predictor_distance(job)
        track_cond = _job_predictor_track(job)
        track_cond_label = normalize_track_condition_label(track_cond)
        surface_token = surface_cli_token(surface)
        target_location = _job_predictor_location(job)
        race_date = _job_race_date(job)
        odds_src = workspace / "odds.csv"

        for spec in list_predictors():
            script_name = str(spec.get("script_name", "") or "").strip()
            latest_name = str(spec.get("latest_filename", "") or "").strip()
            if not script_name or not latest_name:
                continue
            pred_latest_path = workspace / latest_name
            if pred_latest_path.exists():
                pred_latest_path.unlink()
            predictor_start = datetime.now().timestamp()

            if spec["id"] == "main":
                pred_code, pred_output = _run_subprocess(
                    base_path.parent / script_name,
                    cwd=workspace,
                    inputs=[surface, distance, track_cond],
                    env={
                        "SCOPE_KEY": scope_key,
                        "PREDICTOR_NO_PROMPT": "1",
                        "PREDICTOR_TARGET_SURFACE": surface,
                        "PREDICTOR_TARGET_DISTANCE": distance,
                        "PREDICTOR_TARGET_CONDITION": track_cond,
                    },
                )
            elif spec["id"] == "v2_opus":
                pred_code, pred_output = _run_subprocess(
                    base_path.parent / script_name,
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
                    base_path.parent / script_name,
                    cwd=workspace,
                    env={"SCOPE_KEY": scope_key},
                    script_args=[
                        "--base-dir",
                        str(workspace),
                        "--output",
                        latest_name,
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
                    base_path.parent / script_name,
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
                    base_path.parent / script_name,
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

            summary["process_log"].append(
                {"step": f"predictor_{spec['id']}", "code": pred_code, "output": pred_output}
            )
            if pred_code != 0:
                raise RuntimeError(f"{spec['label']} failed: {pred_output}")
            ok, msg = validate_prediction_output(predictor_start, pred_latest_path, odds_src)
            if not ok:
                raise RuntimeError(f"{spec['label']} failed: {msg}\n{pred_output}")

        update_job(
            base_path,
            job_id,
            lambda row, now_text: _mark_predictor_succeeded_and_policy_running(row, now_text),
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary["run_id"] = run_id
        snapshot_paths = _snapshot_outputs(base_path, scope_key, race_id, run_id, workspace)
        run_row = _build_run_row(job, run_id, snapshot_paths)
        _append_csv(data_dir / "runs.csv", run_row)

        engines = list(policy_engines or ("openai", "gemini", "siliconflow", "grok"))
        import web_app  # local import to avoid circular import during app bootstrap

        for engine in engines:
            result = web_app.execute_policy_buy(scope_key, dict(run_row), run_id, policy_engine=engine, policy_model="")
            summary["policy_engines"].append(
                {
                    "engine": engine,
                    "output_text": str(result.get("output_text", "") or ""),
                    "payload_path": str(result.get("payload_path", "") or ""),
                    "ticket_count": len(list(result.get("tickets", []) or [])),
                }
            )
        refreshed_job = get_job(base_path, job_id) or {}
        update_job(
            base_path,
            job_id,
            lambda row, now_text: _mark_policy_succeeded_and_ready(
                row, now_text, run_id, summary, refreshed_job
            ),
        )
        return summary


def fail_race_job(base_dir, job_id, message):
    update_job(
        base_dir,
        job_id,
        lambda row, now_text: _mark_job_failed(row, now_text, message),
    )


def settle_race_job(base_dir, job_id, actual_top3_names):
    base_path = Path(base_dir)
    job = get_job(base_path, job_id)
    if job is None:
        raise ValueError(f"race job not found: {job_id}")
    run_id = str(job.get("current_run_id", "") or "").strip()
    scope_key = str(job.get("scope_key", "") or "").strip()
    if not run_id:
        raise ValueError("race job has no current_run_id")
    names = [str(item or "").strip() for item in list(actual_top3_names or [])[:3]]
    while len(names) < 3:
        names.append("")
    if not all(names[:3]):
        raise ValueError("actual_top1/2/3 are required")

    update_job(
        base_path,
        job_id,
        lambda row, now_text: _mark_settlement_started(row, now_text, names),
    )

    code, output = _run_subprocess(
        base_path / "record_predictor_result.py",
        cwd=base_path,
        inputs=[run_id, names[0], names[1], names[2]],
        env={"SCOPE_KEY": scope_key},
    )
    if code != 0:
        raise RuntimeError(f"settlement failed: {output}")

    summary = {
        "job_id": job_id,
        "run_id": run_id,
        "scope_key": scope_key,
        "actual_top3": names,
        "output": output,
    }
    update_job(
        base_path,
        job_id,
        lambda row, now_text: _mark_settlement_succeeded(row, now_text, names, output),
    )
    return summary


__all__ = ["fail_race_job", "process_race_job", "settle_race_job"]

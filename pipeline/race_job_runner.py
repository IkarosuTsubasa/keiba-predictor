import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from local_env import load_local_env
from prediction_validation import validate_odds_predictions
from predictor_catalog import list_predictors, snapshot_prediction_path
from race_job_store import get_job, initialize_job_step_fields, set_job_step_state, update_job
from surface_scope import get_data_dir, migrate_legacy_data
from v5_remote_tasks import create_task as create_v5_remote_task
from v5_remote_tasks import update_task as update_v5_remote_task


BASE_DIR = Path(__file__).resolve().parent
load_local_env(BASE_DIR, override=False)

ODDS_EXTRACT_TIMEOUT_SECONDS = 300


def _log_preview(value, limit=600):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _log_runner_event(event, **fields):
    payload = {"ts": datetime.now().isoformat(timespec="seconds"), "event": str(event or "").strip()}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    print("[race_job_runner] " + json.dumps(payload, ensure_ascii=False), flush=True)


def strict_llm_odds_gate_enabled():
    raw = os.environ.get("PIPELINE_BLOCK_LLM_ON_ODDS_WARNING", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def v5_predictor_enabled():
    raw = os.environ.get("PIPELINE_ENABLE_V5_PREDICTOR", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def remote_v5_enabled():
    raw = os.environ.get("PIPELINE_REMOTE_V5_ENABLED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def remote_predictor_batch_enabled():
    return remote_v5_enabled()


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
    row["current_v5_task_id"] = ""
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


def _mark_waiting_v5(row, now_text, run_id, summary, task_id):
    row.update(initialize_job_step_fields(row))
    row["status"] = "waiting_v5"
    row["current_run_id"] = run_id
    row["current_v5_task_id"] = str(task_id or "").strip()
    row["error_message"] = ""
    row["last_process_output"] = json.dumps(summary, ensure_ascii=False, indent=2)
    set_job_step_state(row, "predictor", "running", now_text)
    set_job_step_state(row, "policy", "idle")


def _mark_policy_processing_started(row, now_text):
    row.update(initialize_job_step_fields(row))
    row["status"] = "processing_policy"
    row["error_message"] = ""
    set_job_step_state(row, "predictor", "succeeded", now_text)
    set_job_step_state(row, "policy", "running", now_text)


def _mark_policy_succeeded_and_ready(row, now_text, run_id, summary, refreshed_job):
    row.update(initialize_job_step_fields(row))
    row["status"] = "ready"
    row["ready_at"] = now_text
    row["current_run_id"] = run_id
    row["current_v5_task_id"] = ""
    row["error_message"] = ""
    row["last_process_output"] = json.dumps(summary, ensure_ascii=False, indent=2)
    row["queued_process_at"] = str((refreshed_job or {}).get("queued_process_at", "") or "")
    set_job_step_state(row, "predictor", "succeeded", now_text)
    set_job_step_state(row, "policy", "succeeded", now_text)


def _mark_ntfy_notified(row, now_text, run_id, engine):
    row.update(initialize_job_step_fields(row))
    row["ntfy_notify_status"] = "notified"
    row["ntfy_notify_run_id"] = str(run_id or "").strip()
    row["ntfy_notify_engine"] = str(engine or "").strip()
    row["ntfy_notified_at"] = now_text
    row["ntfy_notify_error"] = ""


def _mark_ntfy_notify_failed(row, now_text, run_id, error_text):
    row.update(initialize_job_step_fields(row))
    row["ntfy_notify_status"] = "failed"
    row["ntfy_notify_run_id"] = str(run_id or "").strip()
    row["ntfy_notified_at"] = now_text
    row["ntfy_notify_error"] = str(error_text or "").strip()


def _maybe_send_ntfy_share_notification(base_path, job_id, scope_key, run_id):
    from ntfy_notifier import publish_ntfy_share_notification

    job = get_job(base_path, job_id) or {}
    if str(job.get("ntfy_notify_status", "") or "").strip().lower() == "notified":
        if str(job.get("ntfy_notify_run_id", "") or "").strip() == str(run_id or "").strip():
            _log_runner_event("ntfy_notify_skipped", job_id=job_id, run_id=run_id, reason="already_notified")
            return None
    try:
        result = publish_ntfy_share_notification(scope_key, run_id)
    except Exception as exc:
        error_text = str(exc or "").strip()
        update_job(
            base_path,
            job_id,
            lambda row, now_text: _mark_ntfy_notify_failed(row, now_text, run_id, error_text),
        )
        _log_runner_event("ntfy_notify_failed", job_id=job_id, run_id=run_id, error=error_text)
        return None
    if result.get("skipped"):
        _log_runner_event(
            "ntfy_notify_skipped",
            job_id=job_id,
            run_id=run_id,
            reason=str(result.get("reason", "") or "skipped"),
        )
        return result
    update_job(
        base_path,
        job_id,
        lambda row, now_text: _mark_ntfy_notified(row, now_text, run_id, result.get("engine", "")),
    )
    _log_runner_event(
        "ntfy_notify_sent",
        job_id=job_id,
        run_id=run_id,
        engine=str(result.get("engine", "") or "").strip(),
        topic=str(result.get("topic", "") or "").strip(),
        message_id=str(result.get("message_id", "") or "").strip(),
    )
    return result


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


def _public_site_url():
    value = str(os.environ.get("PIPELINE_PUBLIC_SITE_URL", "https://www.ikaimo-ai.com") or "").strip()
    return (value or "https://www.ikaimo-ai.com").rstrip("/")


def _public_base_path():
    value = str(os.environ.get("PIPELINE_PUBLIC_BASE_PATH", "/keiba") or "").strip()
    if not value:
        return "/keiba"
    if not value.startswith("/"):
        value = "/" + value
    return value.rstrip("/") or "/keiba"


def _remote_v5_bundle_url(task):
    return (
        f"{_public_site_url()}{_public_base_path()}/internal/v5_tasks/"
        f"{quote(str((task or {}).get('task_id', '') or '').strip(), safe='')}"
        f"/bundle?token={quote(str((task or {}).get('bundle_token', '') or '').strip(), safe='')}"
    )


def _remote_v5_callback_url(task):
    return (
        f"{_public_site_url()}{_public_base_path()}/internal/v5_tasks/"
        f"{quote(str((task or {}).get('task_id', '') or '').strip(), safe='')}/callback"
    )


def _dispatch_remote_v5_task(base_dir, task):
    task_id = str((task or {}).get("task_id", "") or "").strip()
    owner = str(os.environ.get("GITHUB_ACTIONS_OWNER", "") or "").strip()
    repo = str(os.environ.get("GITHUB_ACTIONS_REPO", "") or "").strip()
    workflow = str(
        os.environ.get("GITHUB_ACTIONS_WORKFLOW", "predictor-v5-remote.yml") or ""
    ).strip()
    ref = str(os.environ.get("GITHUB_ACTIONS_REF", "main") or "").strip() or "main"
    token = str(os.environ.get("GITHUB_ACTIONS_TOKEN", "") or "").strip()
    if not task_id:
        raise RuntimeError("remote v5 task id missing")
    if not owner or not repo or not workflow or not token:
        raise RuntimeError("remote v5 dispatch config missing")
    update_v5_remote_task(
        base_dir,
        task_id,
        lambda row, now_text: row.update(
            {
                "status": "dispatching",
                "attempt": int(row.get("attempt", 0) or 0) + 1,
                "started_at": str(row.get("started_at", "") or now_text),
                "workflow_dispatch_ref": ref,
                "error_message": "",
            }
        ),
    )
    payload = {
        "ref": ref,
        "inputs": {
            "task_id": task_id,
            "bundle_url": _remote_v5_bundle_url(task),
            "callback_url": _remote_v5_callback_url(task),
        },
    }
    req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow}/dispatches",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "keiba-render-remote-v5",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_body = resp.read().decode("utf-8", errors="replace")
            status_code = getattr(resp, "status", 0) or 0
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        update_v5_remote_task(
            base_dir,
            task_id,
            lambda row, now_text: row.update(
                {"status": "failed", "finished_at": now_text, "error_message": detail or str(exc)}
            ),
        )
        raise RuntimeError(f"remote v5 dispatch failed: http {exc.code} {detail}".strip())
    except Exception as exc:
        update_v5_remote_task(
            base_dir,
            task_id,
            lambda row, now_text: row.update(
                {"status": "failed", "finished_at": now_text, "error_message": str(exc)}
            ),
        )
        raise RuntimeError(f"remote v5 dispatch failed: {exc}")
    update_v5_remote_task(
        base_dir,
        task_id,
        lambda row, now_text: row.update(
            {
                "status": "dispatched",
                "workflow_dispatch_ref": ref,
                "error_message": "",
                "result_summary": {"dispatch_http_status": int(status_code)},
            }
        ),
    )
    return {
        "task_id": task_id,
        "workflow": workflow,
        "ref": ref,
        "status_code": int(status_code),
        "response_preview": _log_preview(response_body),
    }


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
    cmd = [sys.executable, str(script_path), *(list(script_args or []))]
    started_at = time.monotonic()
    _log_runner_event(
        "subprocess_start",
        script=Path(script_path).name,
        cwd=str(cwd),
        timeout_seconds=int(timeout_seconds or 0) if timeout_seconds else 0,
        args=list(script_args or []),
        input_lines=len(list(inputs or [])) if inputs is not None else 0,
    )
    try:
        result = subprocess.run(
            cmd,
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
        _log_runner_event(
            "subprocess_timeout",
            script=Path(script_path).name,
            cwd=str(cwd),
            duration_ms=int((time.monotonic() - started_at) * 1000),
            output_preview=_log_preview(output),
        )
        return 124, output.strip()
    output = (result.stdout or "").strip()
    if result.stderr:
        output = f"{output}\n[stderr]\n{result.stderr.strip()}".strip()
    _log_runner_event(
        "subprocess_end",
        script=Path(script_path).name,
        cwd=str(cwd),
        code=int(result.returncode),
        duration_ms=int((time.monotonic() - started_at) * 1000),
        output_preview=_log_preview(output),
    )
    return result.returncode, output


def _snapshot_outputs(base_dir, scope_key, race_id, run_id, workspace_dir):
    data_dir = get_data_dir(base_dir, scope_key)
    race_dir = data_dir / race_id
    race_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{race_id}"
    prediction_files = {
        "shutuba_path": ("shutuba.csv", f"shutuba_{run_id}{suffix}.csv"),
        "odds_path": ("odds.csv", f"odds_{run_id}{suffix}.csv"),
        "fuku_odds_path": ("fuku_odds.csv", f"fuku_odds_{run_id}{suffix}.csv"),
        "wide_odds_path": ("wide_odds.csv", f"wide_odds_{run_id}{suffix}.csv"),
        "quinella_odds_path": ("quinella_odds.csv", f"quinella_odds_{run_id}{suffix}.csv"),
        "exacta_odds_path": ("exacta_odds.csv", f"exacta_odds_{run_id}{suffix}.csv"),
        "trio_odds_path": ("trio_odds.csv", f"trio_odds_{run_id}{suffix}.csv"),
        "trifecta_odds_path": ("trifecta_odds.csv", f"trifecta_odds_{run_id}{suffix}.csv"),
        "runner_filter_summary_path": (
            "runner_filter_summary.json",
            f"runner_filter_summary_{run_id}{suffix}.json",
        ),
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
        "shutuba_path": snapshot_paths.get("shutuba_path", ""),
        "odds_path": snapshot_paths.get("odds_path", ""),
        "wide_odds_path": snapshot_paths.get("wide_odds_path", ""),
        "fuku_odds_path": snapshot_paths.get("fuku_odds_path", ""),
        "quinella_odds_path": snapshot_paths.get("quinella_odds_path", ""),
        "exacta_odds_path": snapshot_paths.get("exacta_odds_path", ""),
        "trio_odds_path": snapshot_paths.get("trio_odds_path", ""),
        "trifecta_odds_path": snapshot_paths.get("trifecta_odds_path", ""),
        "runner_filter_summary_path": snapshot_paths.get("runner_filter_summary_path", ""),
        "plan_path": "",
        "gemini_policy_path": "",
        "deepseek_policy_path": "",
        "openai_policy_path": "",
        "grok_policy_path": "",
        "tickets": "",
        "amount_yen": "",
    }


def _run_policy_stage(base_path, job_id, scope_key, run_id, summary, policy_engines=None):
    import web_app  # local import to avoid circular import during app bootstrap

    run_row = web_app.resolve_run(run_id, scope_key)
    if run_row is None:
        raise RuntimeError(f"run row not found for run_id={run_id}")
    engines = list(policy_engines or ("openai", "gemini", "deepseek", "grok"))
    update_job(
        base_path,
        job_id,
        lambda row, now_text: _mark_policy_processing_started(row, now_text),
    )
    for engine in engines:
        _log_runner_event("policy_stage_start", job_id=job_id, run_id=run_id, engine=engine)
        result = web_app.execute_policy_buy(scope_key, dict(run_row), run_id, policy_engine=engine, policy_model="")
        summary["policy_engines"].append(
            {
                "engine": engine,
                "output_text": str(result.get("output_text", "") or ""),
                "payload_path": str(result.get("payload_path", "") or ""),
                "ticket_count": len(list(result.get("tickets", []) or [])),
            }
        )
        _log_runner_event(
            "policy_stage_done",
            job_id=job_id,
            run_id=run_id,
            engine=engine,
            ticket_count=len(list(result.get("tickets", []) or [])),
            payload_path=str(result.get("payload_path", "") or ""),
        )
    refreshed_job = get_job(base_path, job_id) or {}
    update_job(
        base_path,
        job_id,
        lambda row, now_text: _mark_policy_succeeded_and_ready(
            row, now_text, run_id, summary, refreshed_job
        ),
    )
    _maybe_send_ntfy_share_notification(base_path, job_id, scope_key, run_id)
    _log_runner_event("process_job_done", job_id=job_id, run_id=run_id, engine_count=len(engines))
    return summary


def process_race_job(base_dir, job_id, policy_engines=None):
    base_path = Path(base_dir)
    job = get_job(base_path, job_id)
    if job is None:
        raise ValueError(f"race job not found: {job_id}")
    race_id = str(job.get("race_id", "") or "").strip()
    scope_key = str(job.get("scope_key", "") or "").strip()
    if not race_id or not scope_key:
        raise ValueError("race job missing race_id or scope_key")
    status_text = str(job.get("status", "") or "").strip().lower()
    if status_text == "queued_policy":
        run_id = str(job.get("current_run_id", "") or "").strip()
        if not run_id:
            raise ValueError("race job has no current_run_id for queued_policy")
        _log_runner_event(
            "process_job_start",
            job_id=job_id,
            race_id=race_id,
            scope_key=scope_key,
            mode="policy_only",
            policy_engines=list(policy_engines or ("openai", "gemini", "deepseek", "grok")),
        )
        summary = {
            "job_id": job_id,
            "race_id": race_id,
            "scope_key": scope_key,
            "process_log": [{"step": "resume_policy_after_v5", "code": 0, "output": f"run_id={run_id}"}],
            "run_id": run_id,
            "policy_engines": [],
        }
        return _run_policy_stage(base_path, job_id, scope_key, run_id, summary, policy_engines=policy_engines)
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
    _log_runner_event(
        "process_job_start",
        job_id=job_id,
        race_id=race_id,
        scope_key=scope_key,
            policy_engines=list(policy_engines or ("openai", "gemini", "deepseek", "grok")),
    )

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
        _log_runner_event("workspace_ready", job_id=job_id, workspace=str(workspace))
        shutil.copy2(kachiuma_path, workspace / "kachiuma.csv")
        shutil.copy2(shutuba_path, workspace / "shutuba.csv")

        race_url = _race_url(scope_key, race_id)
        if not race_url:
            raise ValueError("failed to build race url")
        _log_runner_event("odds_stage_start", job_id=job_id, race_url=race_url)

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
        _log_runner_event("odds_stage_done", job_id=job_id, code=odds_code)
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

        if not remote_predictor_batch_enabled():
            for spec in list_predictors():
                if spec["id"] == "v5_stacking" and not v5_predictor_enabled():
                    skip_message = "Predictor V5 skipped by default on deployment. Set PIPELINE_ENABLE_V5_PREDICTOR=1 to enable."
                    summary["process_log"].append(
                        {"step": f"predictor_{spec['id']}", "code": -1, "output": skip_message}
                    )
                    _log_runner_event(
                        "predictor_stage_skipped",
                        job_id=job_id,
                        predictor_id=spec["id"],
                        reason=skip_message,
                    )
                    continue
                script_name = str(spec.get("script_name", "") or "").strip()
                latest_name = str(spec.get("latest_filename", "") or "").strip()
                if not script_name or not latest_name:
                    continue
                pred_latest_path = workspace / latest_name
                if pred_latest_path.exists():
                    pred_latest_path.unlink()
                predictor_start = datetime.now().timestamp()
                _log_runner_event(
                    "predictor_stage_start",
                    job_id=job_id,
                    predictor_id=spec["id"],
                    script_name=script_name,
                    output_name=latest_name,
                )

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
                _log_runner_event(
                    "predictor_stage_done",
                    job_id=job_id,
                    predictor_id=spec["id"],
                    code=pred_code,
                    prediction_path=str(pred_latest_path),
                )

        if not remote_predictor_batch_enabled():
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
        _log_runner_event("run_row_saved", job_id=job_id, run_id=run_id, runs_csv=str(data_dir / "runs.csv"))

        if remote_predictor_batch_enabled():
            bundle_files = {
                "kachiuma.csv": kachiuma_path,
                "shutuba.csv": snapshot_paths.get("shutuba_path", "") or shutuba_path,
                "odds.csv": snapshot_paths.get("odds_path", ""),
                "fuku_odds.csv": snapshot_paths.get("fuku_odds_path", ""),
                "wide_odds.csv": snapshot_paths.get("wide_odds_path", ""),
                "quinella_odds.csv": snapshot_paths.get("quinella_odds_path", ""),
                "exacta_odds.csv": snapshot_paths.get("exacta_odds_path", ""),
                "trio_odds.csv": snapshot_paths.get("trio_odds_path", ""),
                "trifecta_odds.csv": snapshot_paths.get("trifecta_odds_path", ""),
                "runner_filter_summary.json": snapshot_paths.get("runner_filter_summary_path", ""),
            }
            bundle_files = {
                str(name or "").strip(): str(path or "").strip()
                for name, path in bundle_files.items()
                if str(path or "").strip()
            }
            task = create_v5_remote_task(
                base_path,
                job_id=job_id,
                run_id=run_id,
                race_id=race_id,
                scope_key=scope_key,
                task_type="predictor_batch",
                bundle_files=bundle_files,
                bundle_meta={
                    "race_id": race_id,
                    "run_id": run_id,
                    "scope_key": scope_key,
                    "location": target_location,
                    "race_date": race_date,
                    "surface": surface,
                    "distance": distance,
                    "track_condition": track_cond_label,
                },
            )
            dispatch_info = _dispatch_remote_v5_task(base_path, task)
            summary["v5_remote_task_id"] = str(task.get("task_id", "") or "").strip()
            summary["remote_predictor_task_id"] = str(task.get("task_id", "") or "").strip()
            summary["process_log"].append(
                {
                    "step": "predictors_remote_dispatch",
                    "code": 0,
                    "output": json.dumps(dispatch_info, ensure_ascii=False),
                }
            )
            update_job(
                base_path,
                job_id,
                lambda row, now_text: _mark_waiting_v5(
                    row,
                    now_text,
                    run_id,
                    summary,
                    task.get("task_id", ""),
                ),
            )
            _log_runner_event(
                "predictor_stage_waiting_remote_batch",
                job_id=job_id,
                run_id=run_id,
                task_id=str(task.get("task_id", "") or "").strip(),
            )
            return summary

        return _run_policy_stage(base_path, job_id, scope_key, run_id, summary, policy_engines=policy_engines)


def fail_race_job(base_dir, job_id, message):
    update_job(
        base_dir,
        job_id,
        lambda row, now_text: _mark_job_failed(row, now_text, message),
    )


def _refresh_settlement_odds(scope_key, run_id):
    import web_app  # local import to avoid circular import during app bootstrap

    run_row = web_app.resolve_run(run_id, scope_key)
    if run_row is None:
        raise RuntimeError(f"settlement odds refresh failed: run not found for run_id={run_id}")
    refresh_ok, refresh_message, refresh_warnings = web_app.maybe_refresh_run_odds(
        scope_key,
        run_row,
        run_id,
        True,
    )
    detail_parts = [
        f"[odds_refresh] status={'ok' if refresh_ok else 'fail'} message={str(refresh_message or '').strip()}".strip()
    ]
    if refresh_warnings:
        detail_parts.append(
            "[odds_refresh][warnings] " + "; ".join(str(item or "").strip() for item in refresh_warnings if str(item or "").strip())
        )
    detail_text = "\n".join(part for part in detail_parts if part)
    if not refresh_ok:
        raise RuntimeError(detail_text or "settlement odds refresh failed")
    return detail_text


def settle_race_job(base_dir, job_id, actual_top3_names, official_result_payload=None):
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

    refresh_output = ""
    extra_env = {"SCOPE_KEY": scope_key}
    temp_payload_path = None
    if official_result_payload:
        official_dir = base_path / "data" / "_shared" / "official_results"
        official_dir.mkdir(parents=True, exist_ok=True)
        temp_payload_path = official_dir / f"official_result_{run_id}.json"
        temp_payload_path.write_text(json.dumps(official_result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        extra_env["OFFICIAL_RESULT_PAYLOAD_PATH"] = str(temp_payload_path)
    else:
        refresh_output = _refresh_settlement_odds(scope_key, run_id)

    code, output = _run_subprocess(
        base_path / "record_predictor_result.py",
        cwd=base_path,
        inputs=[run_id, names[0], names[1], names[2]],
        env=extra_env,
    )
    if code != 0:
        raise RuntimeError(f"settlement failed: {output}")
    output = "\n".join(part for part in (refresh_output, output) if str(part or "").strip())

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

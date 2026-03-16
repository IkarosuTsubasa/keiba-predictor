import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from race_job_store import get_job, update_job
from surface_scope import get_data_dir, migrate_legacy_data


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


def _run_subprocess(script_path, *, cwd, inputs=None, env=None, timeout_seconds=None):
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
            [sys.executable, str(script_path)],
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
        "predictions_path": ("predictions.csv", f"predictions_{run_id}{suffix}.csv"),
        "odds_path": ("odds.csv", f"odds_{run_id}{suffix}.csv"),
        "fuku_odds_path": ("fuku_odds.csv", f"fuku_odds_{run_id}{suffix}.csv"),
        "wide_odds_path": ("wide_odds.csv", f"wide_odds_{run_id}{suffix}.csv"),
        "quinella_odds_path": ("quinella_odds.csv", f"quinella_odds_{run_id}{suffix}.csv"),
        "exacta_odds_path": ("exacta_odds.csv", f"exacta_odds_{run_id}{suffix}.csv"),
        "trio_odds_path": ("trio_odds.csv", f"trio_odds_{run_id}{suffix}.csv"),
        "trifecta_odds_path": ("trifecta_odds.csv", f"trifecta_odds_{run_id}{suffix}.csv"),
    }
    out = {}
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
        "surface": "",
        "distance": "",
        "track_condition": "",
        "budget_yen": "2000,5000,10000,50000",
        "style": "scheduled",
        "strategy": "scheduled",
        "strategy_reason": "race_job_runner",
        "predictor_strategy": "scheduled",
        "predictor_reason": "race_job_runner",
        "config_version": "",
        "predictions_path": snapshot_paths.get("predictions_path", ""),
        "predictions_v2_opus_path": "",
        "predictions_v3_premium_path": "",
        "predictions_v4_gemini_path": "",
        "predictions_v5_stacking_path": "",
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
        lambda row, now_text: row.update(
            {
                "status": "processing",
                "processing_started_at": now_text,
                "error_message": "",
                "last_process_output": "",
            }
        ),
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

        pred_code, pred_output = _run_subprocess(
            base_path.parent / "predictor.py",
            cwd=workspace,
            env={
                "SCOPE_KEY": scope_key,
                "PREDICTOR_NO_PROMPT": "1",
                "PREDICTOR_TARGET_SURFACE": _job_predictor_surface(job),
                "PREDICTOR_TARGET_DISTANCE": _job_predictor_distance(job),
                "PREDICTOR_TARGET_CONDITION": _job_predictor_track(job),
            },
        )
        summary["process_log"].append({"step": "predictor", "code": pred_code, "output": pred_output})
        if pred_code != 0 or not (workspace / "predictions.csv").exists():
            raise RuntimeError(f"predictor failed: {pred_output}")

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
            lambda row, now_text: row.update(
                {
                    "status": "ready",
                    "ready_at": now_text,
                    "current_run_id": run_id,
                    "error_message": "",
                    "last_process_output": json.dumps(summary, ensure_ascii=False, indent=2),
                    "queued_process_at": str(refreshed_job.get("queued_process_at", "") or ""),
                }
            ),
        )
        return summary


def fail_race_job(base_dir, job_id, message):
    update_job(
        base_dir,
        job_id,
        lambda row, now_text: row.update(
            {
                "status": "failed",
                "error_message": str(message or ""),
                "last_process_output": str(message or ""),
            }
        ),
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
        lambda row, now_text: row.update(
            {
                "status": "settling",
                "settling_started_at": now_text,
                "actual_top1": names[0],
                "actual_top2": names[1],
                "actual_top3": names[2],
                "error_message": "",
            }
        ),
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
        lambda row, now_text: row.update(
            {
                "status": "settled",
                "settled_at": now_text,
                "actual_top1": names[0],
                "actual_top2": names[1],
                "actual_top3": names[2],
                "last_settlement_output": output,
                "error_message": "",
            }
        ),
    )
    return summary


__all__ = ["fail_race_job", "process_race_job", "settle_race_job"]

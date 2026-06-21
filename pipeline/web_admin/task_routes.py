import json
import os
import traceback
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from fastapi.responses import JSONResponse

from fetch_central_result import (
    build_result_url as build_official_result_url,
    fetch_html as fetch_official_result_html,
    parse_result_page as parse_official_result_page,
)
from race_discovery import discover_races_for_date
from race_meta_fetcher import build_shutuba_url, fetch_race_meta, infer_source_from_race_id, normalize_race_id
from race_job_store import initialize_job_step_fields as _initialize_race_job_step_fields
from race_job_store import set_job_step_state as _set_race_job_step_state
from v5_remote_tasks import create_task as create_remote_task
from v5_remote_tasks import find_latest_task_for_job, update_task as update_remote_task
from web_data.agent_predictions import agent_result_path_for_race, agent_result_write_dir
from web_admin.remote_predictors import append_job_process_log_entry

RUN_DUE_LOCK_TTL_SECONDS = 60 * 30
AUTO_SETTLE_DELAY_MINUTES = 20
AGENT_RESULT_DELAY_MINUTES = 15
JST_OFFSET = timedelta(hours=9)
RUN_DUE_CLEANUP_STATE_FILE = "run_due_cleanup_state.json"
RUN_DUE_HISTORY_FILE = "run_due_history.jsonl"
RUN_DUE_AUTO_DISCOVERY_STATE_FILE = "run_due_auto_discovery_state.json"
RUN_DUE_HISTORY_MAX_LINES = 120
ACTIVE_RUN_DUE_JOB_STATUSES = {
    "queued_morning",
    "processing_morning",
    "queued_process",
    "processing",
    "waiting_v5",
    "queued_policy",
    "processing_policy",
    "queued_agent_prediction",
    "processing_agent_prediction",
    "agent_prediction_ready",
    "fetching_agent_result",
    "queued_settle",
    "settling",
}


def _run_due_lock_path(base_dir):
    return Path(base_dir) / "data" / "_shared" / "run_due.lock"


def _run_due_cleanup_state_path(base_dir):
    return Path(base_dir) / "data" / "_shared" / RUN_DUE_CLEANUP_STATE_FILE


def _run_due_history_path(base_dir):
    return Path(base_dir) / "data" / "_shared" / RUN_DUE_HISTORY_FILE


def _run_due_auto_discovery_state_path(base_dir):
    return Path(base_dir) / "data" / "_shared" / RUN_DUE_AUTO_DISCOVERY_STATE_FILE


def _trim_run_due_history(path):
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        return
    except Exception:
        return
    if len(lines) <= RUN_DUE_HISTORY_MAX_LINES:
        return
    path.write_text("\n".join(lines[-RUN_DUE_HISTORY_MAX_LINES:]) + "\n", encoding="utf-8")


def append_run_due_history(base_dir, summary, *, source="manual", skipped=False, reason="", lock=None):
    compact = _compact_run_due_summary(summary or {})
    errors = list(compact.get("errors", []) or [])
    record = {
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source or "").strip() or "manual",
        "ok": not bool(errors),
        "skipped": bool(skipped),
        "reason": str(reason or "").strip(),
        "error_count": len(errors),
        **compact,
    }
    if isinstance(lock, dict) and lock:
        record["lock_started_at"] = str(lock.get("started_at", "") or "").strip()
        record["lock_pid"] = str(lock.get("pid", "") or "").strip()
    history_path = _run_due_history_path(base_dir)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    _trim_run_due_history(history_path)
    return record


def safe_append_run_due_history(base_dir, summary, *, source="manual", skipped=False, reason="", lock=None):
    try:
        return append_run_due_history(base_dir, summary, source=source, skipped=skipped, reason=reason, lock=lock)
    except Exception as exc:
        print(
            "[web_app] "
            + json.dumps(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "event": "run_due_history_write_error",
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return {}


def load_run_due_history(base_dir, limit=8):
    history_path = _run_due_history_path(base_dir)
    if not history_path.exists():
        return []
    try:
        lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []
    rows = []
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        if len(rows) >= int(limit or 8):
            break
    return rows


def _acquire_run_due_lock(base_dir):
    lock_path = _run_due_lock_path(base_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    now_ts = datetime.now().timestamp()
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        started_ts = float(payload.get("started_ts", 0) or 0)
        if started_ts and now_ts - started_ts < RUN_DUE_LOCK_TTL_SECONDS:
            return False, payload, lock_path
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    payload = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "started_ts": now_ts,
        "pid": os.getpid(),
    }
    try:
        with lock_path.open("x", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
    except FileExistsError:
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        return False, payload, lock_path
    return True, payload, lock_path


def _release_run_due_lock(lock_path):
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _load_run_due_cleanup_state(base_dir):
    state_path = _run_due_cleanup_state_path(base_dir)
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_run_due_cleanup_state(base_dir, payload):
    state_path = _run_due_cleanup_state_path(base_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload or {}, ensure_ascii=False, indent=2), encoding="utf-8")


def _active_run_due_jobs(load_race_jobs, base_dir):
    active_jobs = []
    for job in list(load_race_jobs(base_dir) or []):
        status = str(job.get("status", "") or "").strip().lower()
        if status not in ACTIVE_RUN_DUE_JOB_STATUSES:
            continue
        active_jobs.append(
            {
                "job_id": str(job.get("job_id", "") or "").strip(),
                "status": status,
                "race_id": str(job.get("race_id", "") or "").strip(),
                "run_id": str(job.get("current_run_id", "") or "").strip(),
            }
        )
    return active_jobs


def _maybe_run_daily_cleanup(*, base_dir, load_race_jobs):
    jst_now = _jst_now()
    jst_date = jst_now.date().isoformat()
    active_jobs = _active_run_due_jobs(load_race_jobs, base_dir)
    if active_jobs:
        return {
            "attempted": False,
            "ran": False,
            "reason": "active_jobs_remaining",
            "jst_date": jst_date,
            "active_job_count": len(active_jobs),
            "active_job_ids": [str(item.get("job_id", "") or "").strip() for item in active_jobs[:8]],
            "active_jobs": active_jobs[:8],
        }

    cleanup_state = _load_run_due_cleanup_state(base_dir)
    last_cleanup_jst_date = str(cleanup_state.get("last_cleanup_jst_date", "") or "").strip()
    if last_cleanup_jst_date == jst_date:
        return {
            "attempted": False,
            "ran": False,
            "reason": "already_cleaned_today",
            "jst_date": jst_date,
            "last_cleanup_at": str(cleanup_state.get("last_cleanup_at", "") or "").strip(),
            "last_cleanup_jst_date": last_cleanup_jst_date,
        }

    from cleanup_runtime_artifacts import BASE_DIR as cleanup_runtime_base_dir
    from cleanup_runtime_artifacts import cleanup as cleanup_runtime_artifacts

    if Path(cleanup_runtime_base_dir).resolve() != Path(base_dir).resolve():
        return {
            "attempted": False,
            "ran": False,
            "reason": "cleanup_base_mismatch",
            "jst_date": jst_date,
            "cleanup_base_dir": str(Path(cleanup_runtime_base_dir).resolve()),
            "run_due_base_dir": str(Path(base_dir).resolve()),
        }

    summary = cleanup_runtime_artifacts()
    cleanup_record = {
        "last_cleanup_at": datetime.now().isoformat(timespec="seconds"),
        "last_cleanup_jst_date": jst_date,
        "last_cleanup_totals": dict(summary.get("totals", {}) or {}),
    }
    _save_run_due_cleanup_state(base_dir, cleanup_record)
    return {
        "attempted": True,
        "ran": True,
        "reason": "",
        "jst_date": jst_date,
        "last_cleanup_at": cleanup_record["last_cleanup_at"],
        "last_cleanup_jst_date": cleanup_record["last_cleanup_jst_date"],
        "totals": cleanup_record["last_cleanup_totals"],
    }


def _env_flag(name, default=True):
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _env_int(name, default):
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _maybe_generate_daily_report_after_cleanup(*, cleanup_summary, daily_report_generator):
    cleanup = dict(cleanup_summary or {})
    jst_date = str(cleanup.get("jst_date", "") or "").strip()
    if not bool(cleanup.get("ran")):
        return {
            "attempted": False,
            "ran": False,
            "reason": "cleanup_not_ran",
            "jst_date": jst_date,
        }
    if not _env_flag("RUN_DUE_DAILY_REPORT_ENABLED", True):
        return {
            "attempted": False,
            "ran": False,
            "reason": "disabled",
            "jst_date": jst_date,
        }
    if daily_report_generator is None:
        return {
            "attempted": False,
            "ran": False,
            "reason": "generator_not_configured",
            "jst_date": jst_date,
        }

    record = daily_report_generator(date_text=jst_date)
    item = dict(record or {})
    return {
        "attempted": True,
        "ran": True,
        "reason": "",
        "jst_date": jst_date,
        "slug": str(item.get("slug", "") or "").strip(),
        "title": str(item.get("title", "") or "").strip(),
        "target_date": str(item.get("target_date", "") or "").strip(),
        "target_date_label": str(item.get("target_date_label", "") or "").strip(),
        "engine": str(item.get("engine", "") or "").strip(),
        "engine_label": str(item.get("engine_label", "") or "").strip(),
        "mode": str(item.get("mode", "") or "").strip(),
        "fallback_reason": str(item.get("fallback_reason", "") or "").strip(),
        "public_url": str(item.get("public_url", "") or "").strip(),
    }


def _load_auto_discovery_state(base_dir):
    path = _run_due_auto_discovery_state_path(base_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_auto_discovery_state(base_dir, payload):
    path = _run_due_auto_discovery_state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload or {}), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _agent_job_date(job):
    race_date = str((job or {}).get("race_date", "") or "").strip()
    if race_date:
        return race_date
    off_time = str((job or {}).get("scheduled_off_time", "") or "").strip()
    if len(off_time) >= 10:
        return off_time[:10]
    return ""


def _is_agent_prediction_row(job):
    return str((job or {}).get("job_source", "") or "").strip().lower() == "agent_prediction"


def _agent_jobs_for_date(load_race_jobs, base_dir, race_date):
    target_date = str(race_date or "").strip()
    rows = []
    for job in list(load_race_jobs(base_dir) or []):
        if not _is_agent_prediction_row(job):
            continue
        if _agent_job_date(job) == target_date:
            rows.append(dict(job or {}))
    return rows


def _find_existing_agent_job_for_race(jobs, race_id, race_date):
    race_id_text = normalize_race_id(race_id)
    target_date = str(race_date or "").strip()
    for job in list(jobs or []):
        if not _is_agent_prediction_row(job):
            continue
        if normalize_race_id((job or {}).get("race_id", "")) != race_id_text:
            continue
        if target_date and _agent_job_date(job) != target_date:
            continue
        return dict(job or {})
    return None


def _meta_payload_for_auto_discovered_race(race_id, source, race_date, timeout=30):
    payload = fetch_race_meta(race_id, source=source, timeout=timeout)
    race_date_text = str(race_date or "").strip()
    if race_date_text:
        scheduled_off_time = str(payload.get("scheduled_off_time", "") or "").strip()
        if "T" in scheduled_off_time:
            payload["scheduled_off_time"] = f"{race_date_text}T{scheduled_off_time.split('T', 1)[1]}"
        payload["race_date"] = race_date_text
    return payload


def _annotate_auto_discovered_job(base_dir, update_race_job, job_id, meta_payload, discovered_row):
    payload = dict(meta_payload or {})
    source_row = dict(discovered_row or {})

    def _apply(row, now_text):
        row["race_number"] = str(payload.get("race_number", "") or "").strip()
        row["meta_source_url"] = str(payload.get("source_url", "") or source_row.get("shutuba_url", "") or "").strip()
        row["meta_fetched_at"] = now_text
        row["meta_error"] = ""
        row["meta_retry_count"] = str(row.get("meta_retry_count", "0") or "0")

    return update_race_job(base_dir, job_id, _apply)


def _maybe_auto_discover_agent_race_jobs(
    *,
    base_dir,
    load_race_jobs,
    update_race_job,
    create_race_job=None,
    now_dt=None,
):
    enabled = _env_flag("PIPELINE_AUTO_DISCOVER_RACES_ENABLED", True)
    now = now_dt or _jst_now()
    target_date = now.date().isoformat()
    base_summary = {
        "enabled": enabled,
        "attempted": False,
        "reason": "",
        "target_date": target_date,
        "created_count": 0,
        "discovered_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "source_counts": {},
        "created_job_ids": [],
        "created_jobs": [],
        "skipped_jobs": [],
        "errors": [],
    }
    if not enabled:
        base_summary["reason"] = "disabled"
        return base_summary
    if create_race_job is None:
        base_summary["reason"] = "create_race_job_missing"
        return base_summary

    after_hour = max(0, min(23, _env_int("PIPELINE_AUTO_DISCOVER_AFTER_HOUR_JST", 6)))
    if now.hour < after_hour:
        base_summary["reason"] = "before_auto_discovery_hour"
        return base_summary

    state = _load_auto_discovery_state(base_dir)
    state_for_date = dict((state.get(target_date) or {}) if isinstance(state.get(target_date), dict) else {})
    if state_for_date.get("completed"):
        base_summary["reason"] = "already_attempted"
        base_summary["discovered_count"] = int(state_for_date.get("discovered_count", 0) or 0)
        base_summary["source_counts"] = dict(state_for_date.get("source_counts", {}) or {})
        return base_summary

    existing_today_jobs = _agent_jobs_for_date(load_race_jobs, base_dir, target_date)
    if existing_today_jobs and not state_for_date:
        base_summary["reason"] = "agent_jobs_exist"
        base_summary["skipped_count"] = len(existing_today_jobs)
        base_summary["skipped_jobs"] = [
            {
                "race_id": str(job.get("race_id", "") or "").strip(),
                "job_id": str(job.get("job_id", "") or "").strip(),
                "reason": "agent_jobs_exist",
            }
            for job in existing_today_jobs[:20]
        ]
        return base_summary

    base_summary["attempted"] = True
    try:
        discovery = discover_races_for_date(
            target_date,
            timeout=max(1, _env_int("PIPELINE_AUTO_DISCOVER_TIMEOUT_SECONDS", 30)),
        )
    except Exception as exc:
        base_summary["reason"] = "discovery_failed"
        base_summary["error_count"] = 1
        base_summary["errors"].append({"kind": "discovery", "error": str(exc)})
        return base_summary

    races = list(discovery.get("races", []) or [])
    base_summary["discovered_count"] = len(races)
    base_summary["source_counts"] = dict(discovery.get("source_counts", {}) or {})
    discovery_errors = list(discovery.get("errors", []) or [])
    if discovery_errors:
        base_summary["errors"].extend({"kind": "discovery_source", **dict(item or {})} for item in discovery_errors)

    lead_minutes = max(0, _env_int("PIPELINE_AUTO_DISCOVER_LEAD_MINUTES", 60))
    notes = str(os.environ.get("PIPELINE_AUTO_DISCOVER_JOB_NOTES", "auto_discovered") or "").strip()
    current_jobs = list(load_race_jobs(base_dir) or [])
    created = []
    skipped = []
    errors = []
    for race in races:
        race_id = normalize_race_id((race or {}).get("race_id", ""))
        if not race_id:
            continue
        existing = _find_existing_agent_job_for_race(current_jobs, race_id, target_date)
        if existing:
            skipped.append(
                {
                    "race_id": race_id,
                    "job_id": str(existing.get("job_id", "") or "").strip(),
                    "reason": "already_exists",
                }
            )
            continue
        try:
            source = str((race or {}).get("source", "") or "").strip().lower()
            meta_payload = _meta_payload_for_auto_discovered_race(
                race_id,
                source=source,
                race_date=target_date,
                timeout=max(1, _env_int("PIPELINE_AUTO_DISCOVER_META_TIMEOUT_SECONDS", 10)),
            )
            job = create_race_job(
                base_dir,
                race_id=race_id,
                scope_key=str(meta_payload.get("scope_key", "") or "").strip(),
                race_name=str(meta_payload.get("race_name", "") or "").strip(),
                location=str(meta_payload.get("location", "") or "").strip(),
                race_date=target_date,
                scheduled_off_time=str(meta_payload.get("scheduled_off_time", "") or "").strip(),
                target_surface=str(meta_payload.get("target_surface", "") or "").strip(),
                target_distance=str(meta_payload.get("target_distance", "") or "").strip(),
                target_track_condition=str(meta_payload.get("target_track_condition", "") or "").strip() or "良",
                lead_minutes=lead_minutes,
                job_source="agent_prediction",
                notes=notes,
                artifacts=[],
            )
            updated = _annotate_auto_discovered_job(
                base_dir,
                update_race_job,
                str((job or {}).get("job_id", "") or "").strip(),
                meta_payload,
                race,
            )
            job = updated or job
            current_jobs.append(dict(job or {}))
            created.append(
                {
                    "race_id": race_id,
                    "job_id": str((job or {}).get("job_id", "") or "").strip(),
                    "status": str((job or {}).get("status", "") or "").strip(),
                    "scope_key": str((job or {}).get("scope_key", "") or "").strip(),
                    "race_name": str((job or {}).get("race_name", "") or "").strip(),
                    "scheduled_off_time": str((job or {}).get("scheduled_off_time", "") or "").strip(),
                }
            )
        except Exception as exc:
            errors.append({"race_id": race_id, "error": str(exc)})

    base_summary["created_count"] = len(created)
    base_summary["created_job_ids"] = [str(item.get("job_id", "") or "").strip() for item in created]
    base_summary["created_jobs"] = created
    base_summary["skipped_count"] = len(skipped)
    base_summary["skipped_jobs"] = skipped
    base_summary["errors"].extend({"kind": "create_job", **item} for item in errors)
    base_summary["error_count"] = len(base_summary["errors"])
    if created:
        base_summary["reason"] = "created"
    elif races:
        base_summary["reason"] = "no_new_jobs"
    elif discovery_errors:
        base_summary["reason"] = "discovery_errors"
    else:
        base_summary["reason"] = "no_races"

    state[target_date] = {
        "completed": not bool(discovery_errors or errors),
        "attempted_at": datetime.now().isoformat(timespec="seconds"),
        "discovered_count": len(races),
        "created_count": len(created),
        "skipped_count": len(skipped),
        "error_count": base_summary["error_count"],
        "source_counts": base_summary["source_counts"],
        "reason": base_summary["reason"],
        "created_race_ids": [str(item.get("race_id", "") or "").strip() for item in created],
        "skipped_race_ids": [str(item.get("race_id", "") or "").strip() for item in skipped],
    }
    _save_auto_discovery_state(base_dir, state)
    return base_summary


def pick_next_process_job_id(*, load_race_jobs):
    jobs = load_race_jobs()
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        if status in ("queued_morning", "queued_process", "queued_policy"):
            return str(job.get("job_id", "") or "").strip()
    return ""


def pick_next_settle_job_id(*, load_race_jobs):
    jobs = load_race_jobs()
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        actual_top1 = str(job.get("actual_top1", "") or "").strip()
        actual_top2 = str(job.get("actual_top2", "") or "").strip()
        actual_top3 = str(job.get("actual_top3", "") or "").strip()
        if status == "queued_settle" and actual_top1 and actual_top2 and actual_top3:
            return str(job.get("job_id", "") or "").strip()
    return ""


def pick_next_agent_prediction_job_id(*, load_race_jobs):
    jobs = load_race_jobs()
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        if status == "queued_agent_prediction":
            return str(job.get("job_id", "") or "").strip()
    return ""


def _parse_dt_text(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _jst_now():
    return datetime.utcnow() + JST_OFFSET


def _official_result_source_for_scope(scope_key):
    return "local" if str(scope_key or "").strip().lower() == "local" else "central"


def _fetch_official_result_payload_for_job(job):
    race_id = str(job.get("race_id", "") or "").strip()
    scope_key = str(job.get("scope_key", "") or "").strip()
    result_url = build_official_result_url(race_id=race_id, source=_official_result_source_for_scope(scope_key))
    last_exc = None
    for _ in range(2):
        try:
            html_bytes = fetch_official_result_html(result_url, timeout=30)
            payload = parse_official_result_page(html_bytes, source_url=result_url)
            if payload.get("result_available"):
                return payload
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    return None


def _build_agent_result_url(job):
    race_id = str((job or {}).get("race_id", "") or "").strip()
    scope_key = _agent_prediction_scope_key(job)
    if not race_id:
        raise ValueError("race_id missing")
    if scope_key == "local":
        return f"https://nar.netkeiba.com/race/result.html?race_id={quote(race_id, safe='')}"
    return f"https://race.netkeiba.com/race/result.html?race_id={quote(race_id, safe='')}"


def _agent_result_output_dir(base_dir):
    path = agent_result_write_dir(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch_and_save_agent_result_for_job(base_dir, job):
    repo_root = Path(base_dir).resolve().parent
    import sys

    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    from keiba_llm_agent.fetchers.netkeiba_result_fetcher import fetch_and_parse_netkeiba_result

    result_url = _build_agent_result_url(job)
    result_data = fetch_and_parse_netkeiba_result(result_url, force_refresh=True)
    output_path = _agent_result_output_dir(base_dir) / f"{result_data.race_id}.json"
    output_path.write_text(
        json.dumps(result_data.model_dump(by_alias=True), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    top3_names = [str((item or {}).get("horse_name", "") or "").strip() for item in result_data.model_dump().get("finish_order", [])[:3]]
    top3_numbers = [
        str((item or {}).get("horse_no", "") or "").strip()
        for item in result_data.model_dump().get("finish_order", [])[:3]
    ]
    return {
        "race_id": result_data.race_id,
        "result_url": result_url,
        "result_path": str(output_path),
        "top3_names": top3_names,
        "top3_numbers": top3_numbers,
    }


def _public_site_url():
    return str(os.environ.get("PIPELINE_PUBLIC_SITE_URL", "https://www.ikaimo-ai.com") or "").strip().rstrip("/")


def _public_base_path():
    return str(os.environ.get("PUBLIC_BASE_PATH", "/keiba") or "/keiba").strip() or "/keiba"


def _agent_prediction_callback_url(task):
    return (
        f"{_public_site_url()}{_public_base_path()}/internal/v5_tasks/"
        f"{quote(str((task or {}).get('task_id', '') or '').strip(), safe='')}/callback"
    )


def _agent_prediction_race_url(job):
    race_id = str((job or {}).get("race_id", "") or "").strip()
    scope_key = _agent_prediction_scope_key(job)
    source = "local" if scope_key == "local" else ""
    return build_shutuba_url(race_id, source=source)


def _agent_prediction_scope_key(job):
    scope_key = str((job or {}).get("scope_key", "") or "").strip().lower()
    if scope_key:
        return scope_key
    source = infer_source_from_race_id(str((job or {}).get("race_id", "") or "").strip())
    return "local" if source == "local" else ""


def _agent_prediction_scoring_profile(job):
    return "local_accuracy_default" if _agent_prediction_scope_key(job) == "local" else "accuracy_default"


def _dispatch_agent_prediction_task(base_dir, task, job):
    task_id = str((task or {}).get("task_id", "") or "").strip()
    owner = str(os.environ.get("GITHUB_ACTIONS_OWNER", "") or "").strip()
    repo = str(os.environ.get("GITHUB_ACTIONS_REPO", "") or "").strip()
    workflow = str(
        os.environ.get("GITHUB_ACTIONS_AGENT_WORKFLOW", "")
        or os.environ.get("AGENT_PREDICTION_WORKFLOW", "")
        or "agent-prediction-remote.yml"
    ).strip()
    ref = str(os.environ.get("GITHUB_ACTIONS_REF", "main") or "").strip() or "main"
    token = str(os.environ.get("GITHUB_ACTIONS_TOKEN", "") or "").strip()
    race_id = str((job or {}).get("race_id", "") or "").strip()
    scope_key = _agent_prediction_scope_key(job)
    race_url = _agent_prediction_race_url(job)
    scoring_profile = _agent_prediction_scoring_profile(job)
    if not task_id:
        raise RuntimeError("agent prediction task id missing")
    if not owner or not repo or not workflow or not token:
        raise RuntimeError("agent prediction dispatch config missing")
    update_remote_task(
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
            "race_id": race_id,
            "scope_key": scope_key,
            "race_url": race_url,
            "scoring_profile": scoring_profile,
            "callback_url": _agent_prediction_callback_url(task),
        },
    }
    req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow}/dispatches",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "keiba-render-agent-prediction",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status_code = getattr(resp, "status", 0) or 0
            resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        update_remote_task(
            base_dir,
            task_id,
            lambda row, now_text: row.update(
                {"status": "failed", "finished_at": now_text, "error_message": detail or str(exc)}
            ),
        )
        raise RuntimeError(f"agent prediction dispatch failed: http {exc.code} {detail}".strip())
    except Exception as exc:
        update_remote_task(
            base_dir,
            task_id,
            lambda row, now_text: row.update(
                {"status": "failed", "finished_at": now_text, "error_message": str(exc)}
            ),
        )
        raise RuntimeError(f"agent prediction dispatch failed: {exc}")
    update_remote_task(
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
        "race_id": race_id,
        "scope_key": scope_key,
        "race_url": race_url,
        "scoring_profile": scoring_profile,
        "callback_url": _agent_prediction_callback_url(task),
        "dispatch_http_status": int(status_code),
    }


def dispatch_agent_prediction_job(*, base_dir, job_id, load_race_jobs, update_race_job, initialize_job_step_fields, set_job_step_state):
    target_job_id = str(job_id or "").strip()
    job = next(
        (item for item in load_race_jobs(base_dir) if str((item or {}).get("job_id", "") or "").strip() == target_job_id),
        None,
    )
    if not job:
        raise LookupError("job not found")
    if str((job or {}).get("job_source", "") or "").strip().lower() != "agent_prediction":
        raise ValueError("job is not agent prediction")
    status = str((job or {}).get("status", "") or "").strip().lower()
    if status not in ("queued_agent_prediction", "scheduled", "processing_agent_prediction"):
        raise ValueError(f"job status is not dispatchable: {status or '-'}")

    existing_task = find_latest_task_for_job(
        base_dir,
        target_job_id,
        statuses=("queued", "dispatching", "dispatched", "running"),
    )
    if existing_task:
        existing_task_id = str((existing_task or {}).get("task_id", "") or "").strip()
        existing_task_status = str((existing_task or {}).get("status", "") or "").strip()

        def _mark_existing_dispatched(row, now_text):
            row.update(initialize_job_step_fields(row))
            row["status"] = "processing_agent_prediction"
            row["current_v5_task_id"] = existing_task_id
            row["error_message"] = ""
            row["last_process_output"] = append_job_process_log_entry(
                row,
                "agent_prediction_existing_task",
                0,
                json.dumps(
                    {
                        "task_id": existing_task_id,
                        "task_status": existing_task_status,
                        "already_dispatched": True,
                    },
                    ensure_ascii=False,
                ),
            )
            set_job_step_state(row, "predictor", "running", now_text)

        update_race_job(base_dir, target_job_id, _mark_existing_dispatched)
        return {
            "job_id": target_job_id,
            "race_id": str((job or {}).get("race_id", "") or "").strip(),
            "task_id": existing_task_id,
            "already_dispatched": True,
        }

    race_id = str((job or {}).get("race_id", "") or "").strip()
    scope_key = _agent_prediction_scope_key(job)
    scoring_profile = _agent_prediction_scoring_profile(job)
    task = create_remote_task(
        base_dir,
        job_id=target_job_id,
        run_id="",
        race_id=race_id,
        scope_key=scope_key,
        task_type="agent_prediction",
        bundle_files={},
        bundle_meta={
            "race_id": race_id,
            "scope_key": scope_key,
            "race_name": str((job or {}).get("race_name", "") or "").strip(),
            "race_date": str((job or {}).get("race_date", "") or "").strip(),
            "scheduled_off_time": str((job or {}).get("scheduled_off_time", "") or "").strip(),
            "location": str((job or {}).get("location", "") or "").strip(),
            "race_url": _agent_prediction_race_url(job),
            "scoring_profile": scoring_profile,
        },
    )
    dispatch_info = _dispatch_agent_prediction_task(base_dir, task, job)

    def _mark_dispatched(row, now_text):
        row.update(initialize_job_step_fields(row))
        row["status"] = "processing_agent_prediction"
        row["current_v5_task_id"] = str(task.get("task_id", "") or "").strip()
        row["error_message"] = ""
        row["last_process_output"] = append_job_process_log_entry(
            row,
            "agent_prediction_dispatch",
            0,
            json.dumps(dispatch_info, ensure_ascii=False),
        )
        set_job_step_state(row, "predictor", "running", now_text)

    update_race_job(base_dir, target_job_id, _mark_dispatched)
    return {"job_id": target_job_id, "race_id": race_id, "task_id": str(task.get("task_id", "") or "").strip(), **dispatch_info}


def _job_has_required_artifacts(job):
    if str((job or {}).get("job_source", "") or "").strip().lower() == "agent_prediction":
        return True
    artifact_types = {
        str((item or {}).get("artifact_type", "") or "").strip().lower()
        for item in list((job or {}).get("artifacts", []) or [])
        if isinstance(item, dict)
    }
    return all(name in artifact_types for name in ("kachiuma", "shutuba"))


def autofill_job_input_info(*, base_dir, job_id, get_race_job, update_race_job, compute_initial_status):
    target_job_id = str(job_id or "").strip()
    if not target_job_id:
        raise ValueError("job_id required")
    job = get_race_job(base_dir, target_job_id) or {}
    if not job:
        raise LookupError("job not found")
    race_id = str(job.get("race_id", "") or "").strip()
    if not race_id:
        raise ValueError("missing race_id")
    if not _job_has_required_artifacts(job):
        raise ValueError("missing required artifacts")
    payload = fetch_race_meta(race_id, source=str(job.get("scope_key", "") or "").strip(), timeout=30)
    existing_race_date = str(job.get("race_date", "") or "").strip()
    if existing_race_date:
        scheduled_off_time = str(payload.get("scheduled_off_time", "") or "").strip()
        if "T" in scheduled_off_time:
            payload["scheduled_off_time"] = f"{existing_race_date}T{scheduled_off_time.split('T', 1)[1]}"
        payload["race_date"] = existing_race_date

    def _apply(row, now_text):
        row["race_name"] = str(row.get("race_name", "") or "").strip() or str(payload.get("race_name", "") or "").strip()
        row["race_number"] = str(payload.get("race_number", "") or "").strip()
        row["location"] = str(row.get("location", "") or "").strip() or str(payload.get("location", "") or "").strip()
        row["race_date"] = str(payload.get("race_date", "") or "").strip()
        row["scheduled_off_time"] = str(payload.get("scheduled_off_time", "") or "").strip()
        row["target_distance"] = str(payload.get("target_distance", "") or "").strip()
        row["target_track_condition"] = str(payload.get("target_track_condition", "") or "").strip() or "良"
        row["scope_key"] = str(payload.get("scope_key", "") or "").strip()
        row["target_surface"] = str(payload.get("target_surface", "") or "").strip()
        row["meta_source_url"] = str(payload.get("source_url", "") or "").strip()
        row["meta_fetched_at"] = now_text
        row["meta_error"] = ""
        try:
            retry_count = int(str(row.get("meta_retry_count", "0") or "0").strip() or "0")
        except ValueError:
            retry_count = 0
        row["meta_retry_count"] = str(retry_count)
        row["status"] = compute_initial_status(row)

    updated = update_race_job(base_dir, target_job_id, _apply)
    return updated or {}


def _autofill_waiting_input_info_jobs(*, base_dir, load_race_jobs, update_race_job, compute_initial_status):
    jobs = list(load_race_jobs(base_dir) or [])
    processed = []
    errors = []
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        job_id = str(job.get("job_id", "") or "").strip()
        race_id = str(job.get("race_id", "") or "").strip()
        if status != "waiting_input_info":
            continue
        if not job_id:
            continue
        if not race_id:
            errors.append({"job_id": job_id, "error": "missing race_id"})
            continue
        if not _job_has_required_artifacts(job):
            errors.append({"job_id": job_id, "error": "missing required artifacts"})
            continue
        try:
            updated = autofill_job_input_info(
                base_dir=base_dir,
                job_id=job_id,
                get_race_job=lambda _base_dir, _job_id: next(
                    (item for item in jobs if str(item.get("job_id", "") or "").strip() == str(_job_id or "").strip()),
                    {},
                ),
                update_race_job=update_race_job,
                compute_initial_status=compute_initial_status,
            )
            processed.append(
                {
                    "job_id": job_id,
                    "race_id": race_id,
                    "status": str((updated or {}).get("status", "") or "").strip(),
                    "scope_key": str((updated or {}).get("scope_key", "") or "").strip(),
                }
            )
        except Exception as exc:
            error_text = str(exc or "").strip()

            def _mark_error(row, now_text):
                try:
                    retry_count = int(str(row.get("meta_retry_count", "0") or "0").strip() or "0")
                except ValueError:
                    retry_count = 0
                row["meta_retry_count"] = str(retry_count + 1)
                row["meta_error"] = error_text
                row["meta_fetched_at"] = now_text

            update_race_job(base_dir, job_id, _mark_error)
            errors.append({"job_id": job_id, "error": error_text})
    return processed, errors


def _auto_settle_diagnostics(*, load_race_jobs, now_dt=None):
    current_dt = now_dt or _jst_now()
    rows = []
    for job in load_race_jobs():
        status = str(job.get("status", "") or "").strip().lower()
        if status not in ("ready", "settling"):
            continue
        job_id = str(job.get("job_id", "") or "").strip()
        race_id = str(job.get("race_id", "") or "").strip()
        run_id = str(job.get("current_run_id", "") or "").strip()
        scheduled_off_time = str(job.get("scheduled_off_time", "") or "").strip()
        reason = "eligible"
        if str(job.get("settled_at", "") or "").strip():
            reason = "already_settled"
        else:
            off_dt = _parse_dt_text(scheduled_off_time)
            if off_dt is None:
                reason = "missing_off_time"
            elif current_dt < off_dt + timedelta(minutes=AUTO_SETTLE_DELAY_MINUTES):
                reason = "wait_20min"
            elif not race_id:
                reason = "missing_race_id"
            elif not run_id:
                reason = "missing_run_id"
        rows.append(
            {
                "job_id": job_id,
                "status": status,
                "race_id": race_id,
                "run_id": run_id,
                "scheduled_off_time": scheduled_off_time,
                "reason": reason,
            }
        )
    return rows


def _agent_result_path(base_dir, race_id):
    return agent_result_path_for_race(base_dir, race_id)


def _agent_result_diagnostics(*, base_dir, load_race_jobs, now_dt=None):
    current_dt = now_dt or _jst_now()
    rows = []
    for job in load_race_jobs():
        if str(job.get("job_source", "") or "").strip().lower() != "agent_prediction":
            continue
        status = str(job.get("status", "") or "").strip().lower()
        if status not in ("agent_prediction_ready", "fetching_agent_result"):
            continue
        job_id = str(job.get("job_id", "") or "").strip()
        race_id = str(job.get("race_id", "") or "").strip()
        scheduled_off_time = str(job.get("scheduled_off_time", "") or "").strip()
        reason = "eligible"
        result_path = _agent_result_path(base_dir, race_id)
        if result_path is not None and result_path.exists():
            reason = "already_saved"
        else:
            off_dt = _parse_dt_text(scheduled_off_time)
            if off_dt is None:
                reason = "missing_off_time"
            elif current_dt < off_dt + timedelta(minutes=AGENT_RESULT_DELAY_MINUTES):
                reason = f"wait_{AGENT_RESULT_DELAY_MINUTES}min"
            elif not race_id:
                reason = "missing_race_id"
        rows.append(
            {
                "job_id": job_id,
                "status": status,
                "race_id": race_id,
                "scheduled_off_time": scheduled_off_time,
                "reason": reason,
            }
        )
    return rows


def list_agent_result_jobs(*, base_dir, load_race_jobs, now_dt=None):
    diagnostics = _agent_result_diagnostics(base_dir=base_dir, load_race_jobs=load_race_jobs, now_dt=now_dt)
    eligible_ids = {
        str(item.get("job_id", "") or "").strip()
        for item in diagnostics
        if str(item.get("reason", "") or "").strip() in ("eligible", "already_saved")
    }
    out = []
    for job in load_race_jobs():
        job_id = str(job.get("job_id", "") or "").strip()
        if job_id and job_id in eligible_ids:
            out.append(dict(job))
    return out


def list_auto_settle_jobs(*, load_race_jobs, now_dt=None):
    diagnostics = _auto_settle_diagnostics(load_race_jobs=load_race_jobs, now_dt=now_dt)
    eligible_ids = {
        str(item.get("job_id", "") or "").strip()
        for item in diagnostics
        if str(item.get("reason", "") or "").strip() == "eligible"
    }
    out = []
    for job in load_race_jobs():
        job_id = str(job.get("job_id", "") or "").strip()
        if job_id and job_id in eligible_ids:
            out.append(dict(job))
    return out


def _diagnostic_reason_counts(items):
    counter = Counter()
    for item in list(items or []):
        reason = str((item or {}).get("reason", "") or "").strip() or "unknown"
        counter[reason] += 1
    return dict(sorted(counter.items()))


def _diagnostic_sample(items, *, limit=8, preferred_reasons=None):
    preferred = {str(x).strip() for x in list(preferred_reasons or []) if str(x).strip()}
    sample = []
    for item in list(items or []):
        row = dict(item or {})
        reason = str(row.get("reason", "") or "").strip()
        if preferred and reason not in preferred:
            continue
        sample.append(
            {
                "job_id": str(row.get("job_id", "") or "").strip(),
                "status": str(row.get("status", "") or "").strip(),
                "race_id": str(row.get("race_id", "") or "").strip(),
                "run_id": str(row.get("run_id", "") or "").strip(),
                "reason": reason,
            }
        )
        if len(sample) >= limit:
            return sample
    if preferred:
        return _diagnostic_sample(items, limit=limit, preferred_reasons=None)
    return sample[:limit]


def _log_diagnostic_summary(event, items, *, sample_reasons=None):
    print(
        "[web_app] "
        + json.dumps(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "event": event,
                "count": len(list(items or [])),
                "reason_counts": _diagnostic_reason_counts(items),
                "sample": _diagnostic_sample(items, preferred_reasons=sample_reasons),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _compact_run_due_summary(summary):
    row = dict(summary or {})
    cleanup = dict(row.get("cleanup", {}) or {})
    auto_discovery = dict(row.get("auto_discovery", {}) or {})
    daily_report = dict(row.get("daily_report", {}) or {})
    return {
        "auto_discovery": {
            "enabled": bool(auto_discovery.get("enabled")),
            "attempted": bool(auto_discovery.get("attempted")),
            "created_count": int(auto_discovery.get("created_count", 0) or 0),
            "discovered_count": int(auto_discovery.get("discovered_count", 0) or 0),
            "skipped_count": int(auto_discovery.get("skipped_count", 0) or 0),
            "error_count": int(auto_discovery.get("error_count", 0) or 0),
            "reason": str(auto_discovery.get("reason", "") or "").strip(),
            "target_date": str(auto_discovery.get("target_date", "") or "").strip(),
            "source_counts": dict(auto_discovery.get("source_counts", {}) or {}),
            "created_job_ids": list(auto_discovery.get("created_job_ids", []) or [])[:8],
        },
        "autofill_count": int(row.get("autofill_count", 0) or 0),
        "queued_count": int(row.get("queued_count", 0) or 0),
        "processed_count": int(row.get("processed_count", 0) or 0),
        "agent_dispatched_count": int(row.get("agent_dispatched_count", 0) or 0),
        "settled_count": int(row.get("settled_count", 0) or 0),
        "agent_result_count": int(row.get("agent_result_count", 0) or 0),
        "scan_due_candidate_count": int(row.get("scan_due_candidate_count", 0) or 0),
        "scan_due_reason_counts": _diagnostic_reason_counts(row.get("scan_due_candidates", [])),
        "auto_settle_candidate_count": int(row.get("auto_settle_candidate_count", 0) or 0),
        "auto_settle_reason_counts": _diagnostic_reason_counts(row.get("auto_settle_candidates", [])),
        "agent_result_candidate_count": int(row.get("agent_result_candidate_count", 0) or 0),
        "agent_result_reason_counts": _diagnostic_reason_counts(row.get("agent_result_candidates", [])),
        "queued_job_ids": list(row.get("queued_job_ids", []) or [])[:8],
        "autofilled_job_ids": [str(item.get("job_id", "") or "").strip() for item in list(row.get("autofilled_jobs", []) or [])[:8]],
        "processed_job_ids": list(row.get("processed_job_ids", []) or [])[:8],
        "agent_dispatched_job_ids": list(row.get("agent_dispatched_job_ids", []) or [])[:8],
        "settled_job_ids": list(row.get("settled_job_ids", []) or [])[:8],
        "agent_result_job_ids": list(row.get("agent_result_job_ids", []) or [])[:8],
        "auto_settle_attempted": list(row.get("auto_settle_attempted", []) or [])[:8],
        "auto_settle_skipped": list(row.get("auto_settle_skipped", []) or [])[:8],
        "cleanup": {
            "attempted": bool(cleanup.get("attempted")),
            "ran": bool(cleanup.get("ran")),
            "reason": str(cleanup.get("reason", "") or "").strip(),
            "jst_date": str(cleanup.get("jst_date", "") or "").strip(),
            "last_cleanup_at": str(cleanup.get("last_cleanup_at", "") or "").strip(),
            "active_job_count": int(cleanup.get("active_job_count", 0) or 0),
            "active_job_ids": list(cleanup.get("active_job_ids", []) or [])[:8],
            "totals": dict(cleanup.get("totals", {}) or {}),
        },
        "daily_report": {
            "attempted": bool(daily_report.get("attempted")),
            "ran": bool(daily_report.get("ran")),
            "reason": str(daily_report.get("reason", "") or "").strip(),
            "jst_date": str(daily_report.get("jst_date", "") or "").strip(),
            "slug": str(daily_report.get("slug", "") or "").strip(),
            "title": str(daily_report.get("title", "") or "").strip(),
            "target_date": str(daily_report.get("target_date", "") or "").strip(),
            "target_date_label": str(daily_report.get("target_date_label", "") or "").strip(),
            "engine": str(daily_report.get("engine", "") or "").strip(),
            "engine_label": str(daily_report.get("engine_label", "") or "").strip(),
            "mode": str(daily_report.get("mode", "") or "").strip(),
            "fallback_reason": str(daily_report.get("fallback_reason", "") or "").strip(),
            "public_url": str(daily_report.get("public_url", "") or "").strip(),
            "error": str(daily_report.get("error", "") or "").strip(),
        },
        "errors": list(row.get("errors", []) or [])[:5],
    }


def run_due_jobs_once(
    *,
    base_dir,
    scan_due_race_jobs,
    scan_due_race_job_diagnostics,
    load_race_jobs,
    update_race_job,
    compute_race_job_initial_status,
    create_race_job=None,
    daily_report_generator=None,
    history_source="manual",
):
    auto_discovery_summary = _maybe_auto_discover_agent_race_jobs(
        base_dir=base_dir,
        load_race_jobs=load_race_jobs,
        update_race_job=update_race_job,
        create_race_job=create_race_job,
    )
    if auto_discovery_summary.get("attempted"):
        print(
            "[web_app] "
            + json.dumps(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "event": "run_due_auto_discovery",
                    "target_date": str(auto_discovery_summary.get("target_date", "") or "").strip(),
                    "reason": str(auto_discovery_summary.get("reason", "") or "").strip(),
                    "discovered_count": int(auto_discovery_summary.get("discovered_count", 0) or 0),
                    "created_count": int(auto_discovery_summary.get("created_count", 0) or 0),
                    "error_count": int(auto_discovery_summary.get("error_count", 0) or 0),
                    "source_counts": dict(auto_discovery_summary.get("source_counts", {}) or {}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    autofilled_jobs, autofill_errors = _autofill_waiting_input_info_jobs(
        base_dir=base_dir,
        load_race_jobs=load_race_jobs,
        update_race_job=update_race_job,
        compute_initial_status=compute_race_job_initial_status,
    )
    if autofilled_jobs:
        _log_diagnostic_summary("autofilled_waiting_jobs", autofilled_jobs)
    scan_due_candidates = scan_due_race_job_diagnostics(base_dir)
    _log_diagnostic_summary("scan_due_candidates", scan_due_candidates, sample_reasons=("eligible",))
    changed = scan_due_race_jobs(base_dir)
    process_results = []
    agent_dispatch_results = []
    settle_results = []
    agent_result_results = []
    errors = list(autofill_errors or [])
    auto_settle_skipped = []
    auto_settle_attempted = []
    auto_settle_now = _jst_now()
    auto_settle_candidates = _auto_settle_diagnostics(
        load_race_jobs=lambda: load_race_jobs(base_dir),
        now_dt=auto_settle_now,
    )
    _log_diagnostic_summary("auto_settle_candidates", auto_settle_candidates, sample_reasons=("eligible",))
    agent_result_candidates = _agent_result_diagnostics(
        base_dir=base_dir,
        load_race_jobs=lambda: load_race_jobs(base_dir),
        now_dt=auto_settle_now,
    )
    _log_diagnostic_summary("agent_result_candidates", agent_result_candidates, sample_reasons=("eligible", "already_saved"))

    while True:
        job_id = pick_next_agent_prediction_job_id(load_race_jobs=lambda: load_race_jobs(base_dir))
        if not job_id:
            break
        print(
            "[web_app] "
            + json.dumps(
                {"ts": datetime.now().isoformat(timespec="seconds"), "event": "run_due_agent_prediction_dispatch_start", "job_id": job_id},
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            result = dispatch_agent_prediction_job(
                base_dir=base_dir,
                job_id=job_id,
                load_race_jobs=load_race_jobs,
                update_race_job=update_race_job,
                initialize_job_step_fields=_initialize_race_job_step_fields,
                set_job_step_state=_set_race_job_step_state,
            )
            agent_dispatch_results.append(result)
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "run_due_agent_prediction_dispatch_done",
                        "job_id": job_id,
                        "task_id": str((result or {}).get("task_id", "") or "").strip(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            try:
                def _mark_agent_dispatch_failed(row, now_text):
                    row["status"] = "failed"
                    row["error_message"] = str(exc)
                update_race_job(base_dir, job_id, _mark_agent_dispatch_failed)
            except Exception:
                pass
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "run_due_agent_prediction_dispatch_error",
                        "job_id": job_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            errors.append({"kind": "agent_prediction_dispatch", "job_id": job_id, "error": str(exc)})

    while True:
        job_id = pick_next_process_job_id(load_race_jobs=lambda: load_race_jobs(base_dir))
        if not job_id:
            break
        print(
            "[web_app] "
            + json.dumps(
                {"ts": datetime.now().isoformat(timespec="seconds"), "event": "run_due_process_start", "job_id": job_id},
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            from race_job_runner import process_race_job

            summary = process_race_job(base_dir, job_id)
            process_results.append(summary)
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "run_due_process_done",
                        "job_id": job_id,
                        "run_id": str((summary or {}).get("run_id", "") or "").strip(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            try:
                from race_job_runner import fail_race_job

                fail_race_job(base_dir, job_id, str(exc))
            except Exception:
                pass
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "run_due_process_error",
                        "job_id": job_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            errors.append({"kind": "process", "job_id": job_id, "error": str(exc)})

    while True:
        job_id = pick_next_settle_job_id(load_race_jobs=lambda: load_race_jobs(base_dir))
        if not job_id:
            break
        job = next((item for item in load_race_jobs(base_dir) if str(item.get("job_id", "")).strip() == job_id), {})
        actual_top3 = [
            str(job.get("actual_top1", "") or "").strip(),
            str(job.get("actual_top2", "") or "").strip(),
            str(job.get("actual_top3", "") or "").strip(),
        ]
        try:
            from race_job_runner import settle_race_job

            settle_results.append(settle_race_job(base_dir, job_id, actual_top3))
        except Exception as exc:
            try:
                from race_job_runner import fail_race_job

                fail_race_job(base_dir, job_id, str(exc))
            except Exception:
                pass
            errors.append({"kind": "settle", "job_id": job_id, "error": str(exc)})

    for job in list_auto_settle_jobs(
        load_race_jobs=lambda: load_race_jobs(base_dir),
        now_dt=auto_settle_now,
    ):
        job_id = str((job or {}).get("job_id", "") or "").strip()
        if not job_id:
            continue
        try:
            auto_settle_attempted.append(job_id)
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "auto_settle_start",
                        "job_id": job_id,
                        "race_id": str((job or {}).get("race_id", "") or "").strip(),
                        "run_id": str((job or {}).get("current_run_id", "") or "").strip(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            official_payload = _fetch_official_result_payload_for_job(job)
            if not official_payload:
                auto_settle_skipped.append({"job_id": job_id, "reason": "result_not_available"})
                print(
                    "[web_app] "
                    + json.dumps(
                        {
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "event": "auto_settle_skipped",
                            "job_id": job_id,
                            "reason": "result_not_available",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            actual_top3 = [str(item.get("horse_name", "") or "").strip() for item in list(official_payload.get("top3", []) or [])[:3]]
            if len(actual_top3) < 3 or not all(actual_top3[:3]):
                auto_settle_skipped.append({"job_id": job_id, "reason": "top3_incomplete", "top3": actual_top3})
                print(
                    "[web_app] "
                    + json.dumps(
                        {
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "event": "auto_settle_skipped",
                            "job_id": job_id,
                            "reason": "top3_incomplete",
                            "top3": actual_top3,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            from race_job_runner import settle_race_job

            result = settle_race_job(
                base_dir,
                job_id,
                actual_top3,
                official_result_payload=official_payload,
            )
            settle_results.append(result)
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "auto_settle_done",
                        "job_id": job_id,
                        "run_id": str((result or {}).get("run_id", "") or "").strip(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "auto_settle_error",
                        "job_id": job_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            errors.append({"kind": "auto_settle", "job_id": job_id, "error": str(exc)})

    for job in list_agent_result_jobs(
        base_dir=base_dir,
        load_race_jobs=lambda: load_race_jobs(base_dir),
        now_dt=auto_settle_now,
    ):
        job_id = str((job or {}).get("job_id", "") or "").strip()
        if not job_id:
            continue
        race_id = str((job or {}).get("race_id", "") or "").strip()
        existing_result_path = _agent_result_path(base_dir, race_id)
        try:
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "agent_result_fetch_start",
                        "job_id": job_id,
                        "race_id": race_id,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            def _mark_fetching(row, now_text):
                row["status"] = "fetching_agent_result"
                row["settling_started_at"] = now_text
                _set_race_job_step_state(row, "settlement", "running", now_text)

            update_race_job(base_dir, job_id, _mark_fetching)

            if existing_result_path is not None and existing_result_path.exists():
                result = {
                    "race_id": race_id,
                    "result_path": str(existing_result_path),
                    "result_url": "",
                    "top3_names": [],
                    "top3_numbers": [],
                    "already_saved": True,
                }
            else:
                result = _fetch_and_save_agent_result_for_job(base_dir, job)

            def _mark_agent_result_settled(row, now_text):
                row["status"] = "settled"
                row["settled_at"] = now_text
                row["error_message"] = ""
                top3_names = list((result or {}).get("top3_names", []) or [])[:3]
                top3_numbers = list((result or {}).get("top3_numbers", []) or [])[:3]
                actual_values = top3_names if len(top3_names) >= 3 and all(top3_names[:3]) else top3_numbers
                for index, field_name in enumerate(("actual_top1", "actual_top2", "actual_top3")):
                    row[field_name] = str(actual_values[index] if index < len(actual_values) else "").strip()
                row["last_settlement_output"] = json.dumps(result or {}, ensure_ascii=False, indent=2)
                row["last_process_output"] = append_job_process_log_entry(
                    row,
                    "agent_result_fetch",
                    0,
                    json.dumps(result or {}, ensure_ascii=False),
                )
                _set_race_job_step_state(row, "settlement", "succeeded", now_text)

            update_race_job(base_dir, job_id, _mark_agent_result_settled)
            agent_result_results.append({"job_id": job_id, **dict(result or {})})
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "agent_result_fetch_done",
                        "job_id": job_id,
                        "race_id": race_id,
                        "result_path": str((result or {}).get("result_path", "") or "").strip(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            def _mark_agent_result_failed(row, now_text):
                row["status"] = "agent_prediction_ready"
                row["error_message"] = str(exc)
                _set_race_job_step_state(row, "settlement", "failed", now_text, str(exc))

            update_race_job(base_dir, job_id, _mark_agent_result_failed)
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "agent_result_fetch_error",
                        "job_id": job_id,
                        "race_id": race_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            errors.append({"kind": "agent_result_fetch", "job_id": job_id, "error": str(exc)})

    cleanup_summary = {}
    try:
        cleanup_summary = _maybe_run_daily_cleanup(
            base_dir=base_dir,
            load_race_jobs=load_race_jobs,
        )
        print(
            "[web_app] "
            + json.dumps(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "event": "run_due_cleanup",
                    "cleanup": cleanup_summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        cleanup_summary = {
            "attempted": True,
            "ran": False,
            "reason": "cleanup_failed",
            "error": str(exc),
        }
        errors.append({"kind": "cleanup", "error": str(exc)})
        print(
            "[web_app] "
            + json.dumps(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "event": "run_due_cleanup_error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    daily_report_summary = {}
    try:
        daily_report_summary = _maybe_generate_daily_report_after_cleanup(
            cleanup_summary=cleanup_summary,
            daily_report_generator=daily_report_generator,
        )
        print(
            "[web_app] "
            + json.dumps(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "event": "run_due_daily_report",
                    "daily_report": daily_report_summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        daily_report_summary = {
            "attempted": True,
            "ran": False,
            "reason": "daily_report_failed",
            "jst_date": str(cleanup_summary.get("jst_date", "") or "").strip(),
            "error": str(exc),
        }
        print(
            "[web_app] "
            + json.dumps(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "event": "run_due_daily_report_error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "auto_discovery": auto_discovery_summary,
        "autofill_count": len(autofilled_jobs),
        "autofilled_jobs": autofilled_jobs,
        "queued_count": len(changed),
        "queued_job_ids": [str(item.get("job_id", "") or "").strip() for item in changed],
        "scan_due_candidate_count": len(scan_due_candidates),
        "scan_due_candidates": scan_due_candidates,
        "scan_due_skipped": [
            item for item in scan_due_candidates if str(item.get("reason", "") or "").strip() != "eligible"
        ],
        "processed_count": len(process_results),
        "processed_job_ids": [str(item.get("job_id", "") or "").strip() for item in process_results],
        "agent_dispatched_count": len(agent_dispatch_results),
        "agent_dispatched_job_ids": [str(item.get("job_id", "") or "").strip() for item in agent_dispatch_results],
        "agent_dispatch_results": agent_dispatch_results,
        "settled_count": len(settle_results),
        "settled_job_ids": [str(item.get("job_id", "") or "").strip() for item in settle_results],
        "agent_result_count": len(agent_result_results),
        "agent_result_job_ids": [str(item.get("job_id", "") or "").strip() for item in agent_result_results],
        "agent_result_results": agent_result_results,
        "auto_settle_candidate_count": len(auto_settle_candidates),
        "auto_settle_candidates": auto_settle_candidates,
        "agent_result_candidate_count": len(agent_result_candidates),
        "agent_result_candidates": agent_result_candidates,
        "auto_settle_attempted": auto_settle_attempted,
        "auto_settle_skipped": auto_settle_skipped,
        "cleanup": cleanup_summary,
        "daily_report": daily_report_summary,
        "errors": errors,
    }
    safe_append_run_due_history(base_dir, summary, source=history_source)
    return summary


def import_history_zip(*, base_dir, archive_bytes, overwrite=False):
    data_root = Path(base_dir) / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    imported_paths = []
    with zipfile.ZipFile(BytesIO(archive_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            raw_name = str(info.filename or "").replace("\\", "/").strip("/")
            if not raw_name:
                continue
            parts = [part for part in raw_name.split("/") if part and part not in (".", "..")]
            if not parts:
                continue
            rel_parts = None
            if len(parts) >= 2 and parts[0] == "pipeline" and parts[1] == "data":
                rel_parts = parts[2:]
            elif parts[0] == "data":
                rel_parts = parts[1:]
            elif parts[0] in ("_shared", "central_dirt", "central_turf", "local"):
                rel_parts = parts
            if not rel_parts:
                skipped += 1
                continue
            dest = data_root.joinpath(*rel_parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            file_bytes = zf.read(info)
            dest.write_bytes(file_bytes)
            written += 1
            imported_paths.append(str(dest.relative_to(data_root)).replace("\\", "/"))
    return {"written": written, "skipped": skipped, "sample_paths": imported_paths[:8]}


def internal_run_due_response(
    *,
    base_dir,
    token,
    admin_token_valid,
    scan_due_race_jobs,
    scan_due_race_job_diagnostics,
    load_race_jobs,
    update_race_job,
    compute_race_job_initial_status,
    create_race_job=None,
    daily_report_generator=None,
    history_source="internal",
):
    if not admin_token_valid(token):
        return JSONResponse({"ok": False, "error": "invalid_admin_token"}, status_code=403)
    locked, lock_payload, lock_path = _acquire_run_due_lock(base_dir)
    if not locked:
        safe_append_run_due_history(
            base_dir,
            {"errors": []},
            source=history_source,
            skipped=True,
            reason="already_running",
            lock=lock_payload,
        )
        return JSONResponse(
            {
                "ok": True,
                "skipped": True,
                "reason": "already_running",
                "lock": lock_payload,
            }
        )
    try:
        summary = run_due_jobs_once(
            base_dir=base_dir,
            scan_due_race_jobs=scan_due_race_jobs,
            scan_due_race_job_diagnostics=scan_due_race_job_diagnostics,
            load_race_jobs=load_race_jobs,
            update_race_job=update_race_job,
            compute_race_job_initial_status=compute_race_job_initial_status,
            create_race_job=create_race_job,
            daily_report_generator=daily_report_generator,
            history_source=history_source,
        )
        ok = not bool(list(summary.get("errors", []) or []))
        return JSONResponse({"ok": ok, "skipped": False, **_compact_run_due_summary(summary)}, status_code=200 if ok else 500)
    finally:
        _release_run_due_lock(lock_path)

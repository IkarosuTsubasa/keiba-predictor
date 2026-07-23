import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4


JST_OFFSET = timedelta(hours=9)
AUTO_DISCOVERY_STATE_FILE = "run_due_auto_discovery_state.json"
NOTIFICATION_STATE_DIR = "daily_early_prediction_notifications"
NOTIFICATION_SETTINGS_FILE = "daily_early_prediction_notification_settings.json"
LOCK_STALE_SECONDS = 60 * 30


def _jst_now():
    return datetime.utcnow() + JST_OFFSET


def _env_flag(name, default=True):
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in ("1", "true", "yes", "on")


def _shared_dir(base_dir):
    return Path(base_dir) / "data" / "_shared"


def _auto_discovery_state_path(base_dir):
    return _shared_dir(base_dir) / AUTO_DISCOVERY_STATE_FILE


def _notification_state_dir(base_dir):
    return _shared_dir(base_dir) / NOTIFICATION_STATE_DIR


def _notification_settings_path(base_dir):
    return _shared_dir(base_dir) / NOTIFICATION_SETTINGS_FILE


def _notification_state_path(base_dir, target_date):
    return _notification_state_dir(base_dir) / f"{target_date}.json"


def _notification_lock_path(base_dir, target_date):
    return _notification_state_dir(base_dir) / f"{target_date}.lock"


def _read_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(payload or {}) if isinstance(payload, dict) else {}


def _write_json_atomic(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(
        output_path.suffix + f".{os.getpid()}.{uuid4().hex}.tmp"
    )
    temp_path.write_text(
        json.dumps(dict(payload or {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, output_path)
    return output_path


def _job_date(job):
    race_date = str((job or {}).get("race_date", "") or "").strip()
    if race_date:
        return race_date
    off_time = str((job or {}).get("scheduled_off_time", "") or "").strip()
    return off_time[:10] if len(off_time) >= 10 else ""


def _race_id(job):
    return str((job or {}).get("race_id", "") or "").strip()


def load_daily_early_prediction_notification_settings(base_dir):
    stored = _read_json(_notification_settings_path(base_dir))
    if "enabled" in stored:
        enabled = bool(stored.get("enabled"))
        source = "console"
    else:
        enabled = _env_flag("PIPELINE_DAILY_EARLY_PREDICTION_NOTIFY_ENABLED", True)
        source = "environment"
    return {
        "enabled": enabled,
        "source": source,
        "updated_at": str(stored.get("updated_at", "") or "").strip(),
    }


def save_daily_early_prediction_notification_settings(base_dir, *, enabled):
    payload = {
        "enabled": bool(enabled),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json_atomic(_notification_settings_path(base_dir), payload)
    return {
        **payload,
        "source": "console",
    }


def daily_early_prediction_completion(base_dir, load_race_jobs, target_date=""):
    date_text = str(target_date or "").strip() or _jst_now().date().isoformat()
    discovery_state = _read_json(_auto_discovery_state_path(base_dir))
    discovery = dict(discovery_state.get(date_text) or {})
    if not discovery.get("completed"):
        return {
            "ready": False,
            "reason": "auto_discovery_not_completed",
            "target_date": date_text,
        }

    discovered_count = int(discovery.get("discovered_count", 0) or 0)
    expected_race_ids = {
        str(value or "").strip()
        for key in ("created_race_ids", "skipped_race_ids")
        for value in list(discovery.get(key, []) or [])
        if str(value or "").strip()
    }
    if discovered_count <= 0:
        return {
            "ready": False,
            "reason": "no_discovered_races",
            "target_date": date_text,
            "discovered_count": discovered_count,
        }
    if len(expected_race_ids) < discovered_count:
        return {
            "ready": False,
            "reason": "discovery_race_ids_incomplete",
            "target_date": date_text,
            "discovered_count": discovered_count,
            "expected_race_count": len(expected_race_ids),
        }

    jobs_by_race = {}
    for job in list(load_race_jobs(base_dir) or []):
        if str((job or {}).get("job_source", "") or "").strip().lower() != "agent_prediction":
            continue
        if _job_date(job) != date_text:
            continue
        race_id = _race_id(job)
        if race_id in expected_race_ids:
            jobs_by_race.setdefault(race_id, []).append(dict(job or {}))

    missing_race_ids = sorted(expected_race_ids.difference(jobs_by_race))
    incomplete_race_ids = sorted(
        race_id
        for race_id in expected_race_ids
        if race_id in jobs_by_race
        and not any(
            str(job.get("early_prediction_ready_at", "") or "").strip()
            for job in jobs_by_race[race_id]
        )
    )
    if missing_race_ids:
        return {
            "ready": False,
            "reason": "race_jobs_missing",
            "target_date": date_text,
            "race_count": len(expected_race_ids),
            "missing_race_ids": missing_race_ids,
        }
    if incomplete_race_ids:
        return {
            "ready": False,
            "reason": "early_predictions_incomplete",
            "target_date": date_text,
            "race_count": len(expected_race_ids),
            "incomplete_race_ids": incomplete_race_ids,
        }
    return {
        "ready": True,
        "reason": "all_early_predictions_ready",
        "target_date": date_text,
        "race_count": len(expected_race_ids),
        "race_ids": sorted(expected_race_ids),
    }


def _acquire_date_lock(base_dir, target_date):
    lock_path = _notification_lock_path(base_dir, target_date)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        try:
            age_seconds = max(0.0, datetime.now().timestamp() - lock_path.stat().st_mtime)
        except OSError:
            age_seconds = 0.0
        if age_seconds < LOCK_STALE_SECONDS:
            return None
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return None
        try:
            handle = lock_path.open("x", encoding="utf-8")
        except FileExistsError:
            return None
    with handle:
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
            )
        )
    return lock_path


def maybe_send_daily_early_prediction_complete_notification(
    base_dir,
    load_race_jobs,
    target_date="",
):
    date_text = str(target_date or "").strip() or _jst_now().date().isoformat()
    notification_settings = load_daily_early_prediction_notification_settings(base_dir)
    if not notification_settings.get("enabled"):
        return {
            "ok": False,
            "skipped": True,
            "reason": "disabled",
            "target_date": date_text,
            "settings_source": str(notification_settings.get("source", "") or ""),
        }

    state_path = _notification_state_path(base_dir, date_text)
    existing_state = _read_json(state_path)
    if str(existing_state.get("status", "") or "").strip().lower() == "sent":
        return {
            "ok": True,
            "skipped": True,
            "reason": "already_notified",
            **existing_state,
        }

    completion = daily_early_prediction_completion(
        base_dir,
        load_race_jobs,
        target_date=date_text,
    )
    if not completion.get("ready"):
        return {
            "ok": False,
            "skipped": True,
            **completion,
        }

    lock_path = _acquire_date_lock(base_dir, date_text)
    if lock_path is None:
        return {
            "ok": False,
            "skipped": True,
            "reason": "notification_in_progress",
            "target_date": date_text,
        }

    try:
        existing_state = _read_json(state_path)
        if str(existing_state.get("status", "") or "").strip().lower() == "sent":
            return {
                "ok": True,
                "skipped": True,
                "reason": "already_notified",
                **existing_state,
            }

        completion = daily_early_prediction_completion(
            base_dir,
            load_race_jobs,
            target_date=date_text,
        )
        if not completion.get("ready"):
            return {
                "ok": False,
                "skipped": True,
                **completion,
            }

        from ntfy_notifier import publish_fcm_daily_early_prediction_complete_notification

        try:
            result = publish_fcm_daily_early_prediction_complete_notification(
                target_date=date_text,
                race_count=int(completion.get("race_count", 0) or 0),
            )
        except Exception as exc:
            failed_state = {
                "status": "failed",
                "target_date": date_text,
                "race_count": int(completion.get("race_count", 0) or 0),
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "error": str(exc or "").strip(),
            }
            _write_json_atomic(state_path, failed_state)
            return {
                "ok": False,
                "skipped": False,
                "reason": "send_failed",
                **failed_state,
            }

        if not result.get("ok") or result.get("skipped"):
            return {
                "ok": False,
                "skipped": True,
                "reason": str(result.get("reason", "") or "send_skipped"),
                "target_date": date_text,
                "race_count": int(completion.get("race_count", 0) or 0),
                "send_result": result,
            }

        sent_state = {
            "status": "sent",
            "target_date": date_text,
            "race_count": int(completion.get("race_count", 0) or 0),
            "sent_at": datetime.now().isoformat(timespec="seconds"),
            "topic": str(result.get("topic", "") or "").strip(),
            "message_id": str(result.get("message_id", "") or "").strip(),
            "destination": str(result.get("destination", "") or "").strip(),
        }
        _write_json_atomic(state_path, sent_state)
        return {
            "ok": True,
            "skipped": False,
            "reason": "notified",
            **sent_state,
        }
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

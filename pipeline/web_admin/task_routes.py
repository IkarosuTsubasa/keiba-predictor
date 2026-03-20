import json
import os
import traceback
import zipfile
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

from fastapi.responses import JSONResponse

from fetch_central_result import (
    build_result_url as build_official_result_url,
    fetch_html as fetch_official_result_html,
    parse_result_page as parse_official_result_page,
)

RUN_DUE_LOCK_TTL_SECONDS = 60 * 30
AUTO_SETTLE_DELAY_MINUTES = 20


def _run_due_lock_path(base_dir):
    return Path(base_dir) / "data" / "_shared" / "run_due.lock"


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


def pick_next_process_job_id(*, load_race_jobs):
    jobs = load_race_jobs()
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        if status in ("queued_process", "queued_policy"):
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


def _auto_settle_diagnostics(*, load_race_jobs, now_dt=None):
    current_dt = now_dt or datetime.now()
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


def run_due_jobs_once(*, base_dir, scan_due_race_jobs, load_race_jobs):
    changed = scan_due_race_jobs(base_dir)
    process_results = []
    settle_results = []
    errors = []
    auto_settle_skipped = []
    auto_settle_attempted = []
    auto_settle_now = datetime.now()
    auto_settle_candidates = _auto_settle_diagnostics(
        load_race_jobs=lambda: load_race_jobs(base_dir),
        now_dt=auto_settle_now,
    )
    print(
        "[web_app] "
        + json.dumps(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "event": "auto_settle_candidates",
                "count": len(auto_settle_candidates),
                "items": auto_settle_candidates,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

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

    return {
        "queued_count": len(changed),
        "queued_job_ids": [str(item.get("job_id", "") or "").strip() for item in changed],
        "processed_count": len(process_results),
        "processed_job_ids": [str(item.get("job_id", "") or "").strip() for item in process_results],
        "settled_count": len(settle_results),
        "settled_job_ids": [str(item.get("job_id", "") or "").strip() for item in settle_results],
        "auto_settle_candidate_count": len(auto_settle_candidates),
        "auto_settle_candidates": auto_settle_candidates,
        "auto_settle_attempted": auto_settle_attempted,
        "auto_settle_skipped": auto_settle_skipped,
        "errors": errors,
    }


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


def internal_run_due_response(*, base_dir, token, admin_token_valid, scan_due_race_jobs, load_race_jobs):
    if not admin_token_valid(token):
        return JSONResponse({"ok": False, "error": "invalid_admin_token"}, status_code=403)
    locked, lock_payload, lock_path = _acquire_run_due_lock(base_dir)
    if not locked:
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
            load_race_jobs=load_race_jobs,
        )
        ok = not bool(list(summary.get("errors", []) or []))
        return JSONResponse({"ok": ok, "skipped": False, **summary}, status_code=200 if ok else 500)
    finally:
        _release_run_due_lock(lock_path)

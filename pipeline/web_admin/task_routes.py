import json
import os
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

from fastapi.responses import JSONResponse

from fetch_central_result import (
    build_result_url as build_official_result_url,
    fetch_html as fetch_official_result_html,
    parse_result_page as parse_official_result_page,
)
from race_meta_fetcher import fetch_race_meta

RUN_DUE_LOCK_TTL_SECONDS = 60 * 30
AUTO_SETTLE_DELAY_MINUTES = 20
JST_OFFSET = timedelta(hours=9)


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


def _job_has_required_artifacts(job):
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
    return {
        "autofill_count": int(row.get("autofill_count", 0) or 0),
        "queued_count": int(row.get("queued_count", 0) or 0),
        "processed_count": int(row.get("processed_count", 0) or 0),
        "settled_count": int(row.get("settled_count", 0) or 0),
        "scan_due_candidate_count": int(row.get("scan_due_candidate_count", 0) or 0),
        "scan_due_reason_counts": _diagnostic_reason_counts(row.get("scan_due_candidates", [])),
        "auto_settle_candidate_count": int(row.get("auto_settle_candidate_count", 0) or 0),
        "auto_settle_reason_counts": _diagnostic_reason_counts(row.get("auto_settle_candidates", [])),
        "queued_job_ids": list(row.get("queued_job_ids", []) or [])[:8],
        "autofilled_job_ids": [str(item.get("job_id", "") or "").strip() for item in list(row.get("autofilled_jobs", []) or [])[:8]],
        "processed_job_ids": list(row.get("processed_job_ids", []) or [])[:8],
        "settled_job_ids": list(row.get("settled_job_ids", []) or [])[:8],
        "auto_settle_attempted": list(row.get("auto_settle_attempted", []) or [])[:8],
        "auto_settle_skipped": list(row.get("auto_settle_skipped", []) or [])[:8],
        "errors": list(row.get("errors", []) or [])[:5],
    }
def run_due_jobs_once(*, base_dir, scan_due_race_jobs, scan_due_race_job_diagnostics, load_race_jobs, update_race_job, compute_race_job_initial_status):
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
    settle_results = []
    errors = list(autofill_errors or [])
    auto_settle_skipped = []
    auto_settle_attempted = []
    auto_settle_now = _jst_now()
    auto_settle_candidates = _auto_settle_diagnostics(
        load_race_jobs=lambda: load_race_jobs(base_dir),
        now_dt=auto_settle_now,
    )
    _log_diagnostic_summary("auto_settle_candidates", auto_settle_candidates, sample_reasons=("eligible",))

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


def internal_run_due_response(*, base_dir, token, admin_token_valid, scan_due_race_jobs, scan_due_race_job_diagnostics, load_race_jobs, update_race_job, compute_race_job_initial_status):
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
            scan_due_race_job_diagnostics=scan_due_race_job_diagnostics,
            load_race_jobs=load_race_jobs,
            update_race_job=update_race_job,
            compute_race_job_initial_status=compute_race_job_initial_status,
        )
        ok = not bool(list(summary.get("errors", []) or []))
        return JSONResponse({"ok": ok, "skipped": False, **_compact_run_due_summary(summary)}, status_code=200 if ok else 500)
    finally:
        _release_run_due_lock(lock_path)

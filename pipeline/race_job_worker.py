import argparse
import json
import os
import sys
import time
from pathlib import Path

from race_job_runner import fail_race_job, process_race_job, settle_race_job
from race_job_store import get_job, load_jobs, scan_due_jobs


BASE_DIR = Path(__file__).resolve().parent


def _poll_seconds():
    raw = str(os.environ.get("WORKER_POLL_SECONDS", "30") or "30").strip()
    try:
        return max(5, int(raw))
    except ValueError:
        return 30


def pick_next_queued_job():
    jobs = load_jobs(BASE_DIR)
    for job in jobs:
        status = str(job.get("status", "")).strip().lower()
        if status == "queued_process":
            return str(job.get("job_id", "") or "").strip()
    return ""


def pick_next_settle_job():
    jobs = load_jobs(BASE_DIR)
    for job in jobs:
        status = str(job.get("status", "")).strip().lower()
        actual_top1 = str(job.get("actual_top1", "")).strip()
        actual_top2 = str(job.get("actual_top2", "")).strip()
        actual_top3 = str(job.get("actual_top3", "")).strip()
        if status == "queued_settle" and actual_top1 and actual_top2 and actual_top3:
            return str(job.get("job_id", "") or "").strip()
    return ""


def _run_next_job():
    job_id = pick_next_queued_job()
    if not job_id:
        return None
    summary = process_race_job(BASE_DIR, job_id)
    return {"kind": "process", "job_id": job_id, "summary": summary}


def _run_next_settlement():
    job_id = pick_next_settle_job()
    if not job_id:
        return None
    job = get_job(BASE_DIR, job_id) or {}
    actual_top3 = [
        str(job.get("actual_top1", "") or "").strip(),
        str(job.get("actual_top2", "") or "").strip(),
        str(job.get("actual_top3", "") or "").strip(),
    ]
    summary = settle_race_job(BASE_DIR, job_id, actual_top3)
    return {"kind": "settle", "job_id": job_id, "summary": summary}


def run_loop(once=False):
    poll_seconds = _poll_seconds()
    while True:
        changed = scan_due_jobs(BASE_DIR)
        process_result = None
        settle_result = None
        try:
            process_result = _run_next_job()
        except Exception as exc:
            failed_job_id = pick_next_queued_job()
            if failed_job_id:
                fail_race_job(BASE_DIR, failed_job_id, str(exc))
            print(json.dumps({"kind": "process_error", "error": str(exc)}, ensure_ascii=False))
        try:
            settle_result = _run_next_settlement()
        except Exception as exc:
            failed_job_id = pick_next_settle_job()
            if failed_job_id:
                fail_race_job(BASE_DIR, failed_job_id, str(exc))
            print(json.dumps({"kind": "settle_error", "error": str(exc)}, ensure_ascii=False))

        if changed:
            print(json.dumps({"kind": "scan", "queued": len(changed), "job_ids": [item.get("job_id", "") for item in changed]}, ensure_ascii=False))
        if process_result:
            print(json.dumps(process_result, ensure_ascii=False))
        if settle_result:
            print(json.dumps(settle_result, ensure_ascii=False))
        if not changed and not process_result and not settle_result:
            print(json.dumps({"kind": "idle", "poll_seconds": poll_seconds}, ensure_ascii=False))
        if once:
            return
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Local race job worker")
    parser.add_argument("command", choices=["scan", "process", "process-next", "settle", "settle-next", "loop", "loop-once"])
    parser.add_argument("--job-id", default="")
    parser.add_argument("--actual-top1", default="")
    parser.add_argument("--actual-top2", default="")
    parser.add_argument("--actual-top3", default="")
    args = parser.parse_args()

    if args.command == "loop":
        run_loop(once=False)
        return
    if args.command == "loop-once":
        run_loop(once=True)
        return

    if args.command == "scan":
        changed = scan_due_jobs(BASE_DIR)
        print(json.dumps({"queued": len(changed), "job_ids": [item.get("job_id", "") for item in changed]}, ensure_ascii=False))
        return

    if args.command == "process-next":
        job_id = pick_next_queued_job()
        if not job_id:
            print("No queued race jobs.")
            return
    elif args.command == "settle-next":
        job_id = pick_next_settle_job()
        if not job_id:
            print("No queued settlement jobs.")
            return
    else:
        job_id = str(args.job_id or "").strip()
        if not job_id:
            print("--job-id is required for this command")
            sys.exit(1)

    if get_job(BASE_DIR, job_id) is None:
        print(f"Race job not found: {job_id}")
        sys.exit(1)

    try:
        if args.command in ("process", "process-next"):
            summary = process_race_job(BASE_DIR, job_id)
        else:
            job = get_job(BASE_DIR, job_id) or {}
            actual_top3 = [
                str(args.actual_top1 or job.get("actual_top1", "") or "").strip(),
                str(args.actual_top2 or job.get("actual_top2", "") or "").strip(),
                str(args.actual_top3 or job.get("actual_top3", "") or "").strip(),
            ]
            summary = settle_race_job(BASE_DIR, job_id, actual_top3)
    except Exception as exc:
        fail_race_job(BASE_DIR, job_id, str(exc))
        print(f"Race job failed: {exc}")
        sys.exit(1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

from race_job_runner import fail_race_job, process_race_job, settle_race_job
from race_job_store import get_job, load_jobs, scan_due_jobs


BASE_DIR = Path(__file__).resolve().parent


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


def main():
    parser = argparse.ArgumentParser(description="Local race job worker")
    parser.add_argument("command", choices=["scan", "process", "process-next", "settle", "settle-next"])
    parser.add_argument("--job-id", default="")
    parser.add_argument("--actual-top1", default="")
    parser.add_argument("--actual-top2", default="")
    parser.add_argument("--actual-top3", default="")
    args = parser.parse_args()

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

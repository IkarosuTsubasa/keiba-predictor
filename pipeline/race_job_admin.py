import argparse
import json
from pathlib import Path

from race_job_store import (
    create_job,
    get_job,
    load_jobs,
    save_artifact,
    update_job,
)
from surface_scope import normalize_scope_key


BASE_DIR = Path(__file__).resolve().parent


def _read_bytes(path_text):
    path = Path(str(path_text or "").strip())
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return path, path.read_bytes()


def create_job_from_files(args):
    scope_key = normalize_scope_key(args.scope_key)
    if not scope_key:
        raise ValueError("invalid --scope-key")
    race_id = str(args.race_id or "").strip()
    if not race_id:
        raise ValueError("--race-id is required")

    job = create_job(
        BASE_DIR,
        race_id=race_id,
        scope_key=scope_key,
        race_name=args.race_name,
        location=args.location,
        race_date=args.race_date,
        scheduled_off_time=args.scheduled_off_time,
        lead_minutes=args.lead_minutes,
        notes=args.notes,
        artifacts=[],
    )

    artifact_payloads = []
    for artifact_type, source_path in (("kachiuma", args.kachiuma_path), ("shutuba", args.shutuba_path)):
        path, payload = _read_bytes(source_path)
        artifact_payloads.append(
            save_artifact(BASE_DIR, job["job_id"], artifact_type, path.name, payload)
        )

    def _attach(row, now_text):
        row["artifacts"] = artifact_payloads
        row["status"] = "scheduled"
        row["updated_at"] = now_text

    updated = update_job(BASE_DIR, job["job_id"], _attach) or job
    print(json.dumps(updated, ensure_ascii=False, indent=2))


def list_jobs(args):
    rows = load_jobs(BASE_DIR)
    if args.status:
        status_filter = str(args.status or "").strip().lower()
        rows = [row for row in rows if str(row.get("status", "")).strip().lower() == status_filter]
    if args.limit:
        rows = rows[: args.limit]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def show_job(args):
    row = get_job(BASE_DIR, args.job_id)
    if row is None:
        raise ValueError(f"job not found: {args.job_id}")
    print(json.dumps(row, ensure_ascii=False, indent=2))


def queue_settle(args):
    names = [str(args.actual_top1 or "").strip(), str(args.actual_top2 or "").strip(), str(args.actual_top3 or "").strip()]
    if not all(names):
        raise ValueError("--actual-top1/2/3 are required")

    def _mutate(row, now_text):
        row["actual_top1"] = names[0]
        row["actual_top2"] = names[1]
        row["actual_top3"] = names[2]
        row["status"] = "queued_settle"
        row["queued_settle_at"] = now_text
        row["error_message"] = ""

    updated = update_job(BASE_DIR, args.job_id, _mutate)
    if updated is None:
        raise ValueError(f"job not found: {args.job_id}")
    print(json.dumps(updated, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Race job admin CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a race job from local CSV files")
    create_parser.add_argument("--scope-key", required=True)
    create_parser.add_argument("--race-id", required=True)
    create_parser.add_argument("--race-name", default="")
    create_parser.add_argument("--location", default="")
    create_parser.add_argument("--race-date", default="")
    create_parser.add_argument("--scheduled-off-time", required=True)
    create_parser.add_argument("--lead-minutes", type=int, default=30)
    create_parser.add_argument("--notes", default="")
    create_parser.add_argument("--kachiuma-path", required=True)
    create_parser.add_argument("--shutuba-path", required=True)
    create_parser.set_defaults(func=create_job_from_files)

    list_parser = subparsers.add_parser("list", help="List race jobs")
    list_parser.add_argument("--status", default="")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.set_defaults(func=list_jobs)

    show_parser = subparsers.add_parser("show", help="Show a single race job")
    show_parser.add_argument("--job-id", required=True)
    show_parser.set_defaults(func=show_job)

    settle_parser = subparsers.add_parser("queue-settle", help="Save result names and queue settlement")
    settle_parser.add_argument("--job-id", required=True)
    settle_parser.add_argument("--actual-top1", required=True)
    settle_parser.add_argument("--actual-top2", required=True)
    settle_parser.add_argument("--actual-top3", required=True)
    settle_parser.set_defaults(func=queue_settle)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

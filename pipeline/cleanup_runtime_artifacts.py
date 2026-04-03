from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SHARED_DIR = DATA_DIR / "_shared"
ARTIFACTS_DIR = SHARED_DIR / "race_job_artifacts"
WORKSPACES_DIR = SHARED_DIR / "job_workspaces"
RACE_JOBS_PATH = SHARED_DIR / "race_jobs.json"
RACE_JOBS_ARCHIVE_PATH = SHARED_DIR / "race_jobs_archive.jsonl"
SCOPE_NAMES = ("central_dirt", "central_turf", "local")
CACHE_DIR_NAMES = (
    "policy_cache_gemini",
    "policy_cache_deepseek",
    "policy_cache_openai",
    "policy_cache_grok",
    "policy_cache_siliconflow",
)
SCOPE_ROOT_PATTERNS = (
    "replay_*",
    "eval_ranker_modes_*",
    "tune_ranker_alpha_*",
    "bet_plan_items_*.csv",
)
LEGACY_DEBUG_PATTERNS = (
    "bet_engine_v*_cfg_*.json",
)
RACE_DIR_PATTERNS = (
    "siliconflow_policy_*.json",
    "runner_filter_summary_*.json",
)
PROMPT_INPUT_PATTERNS = (
    "*_policy_input_*.json",
)
OFFLINE_RESEARCH_FILES = (
    "offline_eval.csv",
    "context_dataset.csv",
    "context_summary.csv",
    "history_races.csv",
)
REFERENCED_ARTIFACT_STATUSES = {"settled", "failed"}
ARCHIVABLE_JOB_STATUSES = {"settled", "failed", "deleted"}


def _load_jobs():
    if not RACE_JOBS_PATH.exists():
        return []
    try:
        payload = json.loads(RACE_JOBS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _save_jobs(rows):
    RACE_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RACE_JOBS_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_archive_rows(rows):
    if not rows:
        return
    RACE_JOBS_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RACE_JOBS_ARCHIVE_PATH.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _active_artifact_dirs():
    active = set()
    for job in _load_jobs():
        for artifact in list(job.get("artifacts", []) or []):
            if not isinstance(artifact, dict):
                continue
            stored_path = str(artifact.get("stored_path", "") or "").strip()
            if not stored_path:
                continue
            try:
                active.add(str(Path(stored_path).resolve().parent))
            except Exception:
                continue
    return active


def _older_than(path: Path, cutoff: datetime):
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return False
    return modified < cutoff


def _parse_dt(value: str):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _remove_tree(path: Path, dry_run: bool):
    if dry_run:
        return True
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def _remove_file(path: Path, dry_run: bool):
    if not path.exists() or not path.is_file():
        return False
    if dry_run:
        return True
    try:
        path.unlink()
    except OSError:
        return False
    return not path.exists()


def _remove_empty_dir(path: Path, dry_run: bool):
    if not path.exists() or not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return False
    except StopIteration:
        pass
    if dry_run:
        return True
    try:
        path.rmdir()
    except OSError:
        return False
    return not path.exists()


def _iter_scope_dirs():
    for name in SCOPE_NAMES:
        path = DATA_DIR / name
        if path.exists() and path.is_dir():
            yield path


def _remove_old_files(paths, cutoff: datetime, dry_run: bool):
    removed = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        if not _older_than(path, cutoff):
            continue
        if _remove_file(path, dry_run):
            removed.append(str(path))
    return removed


def _collect_scope_root_paths(scope_dir: Path):
    for pattern in SCOPE_ROOT_PATTERNS:
        yield from scope_dir.glob(pattern)


def _collect_legacy_debug_paths(scope_dir: Path):
    for pattern in LEGACY_DEBUG_PATTERNS:
        yield from scope_dir.glob(pattern)


def _collect_race_dir_paths(scope_dir: Path):
    for pattern in RACE_DIR_PATTERNS:
        yield from scope_dir.rglob(pattern)


def _collect_prompt_input_paths(scope_dir: Path):
    for pattern in PROMPT_INPUT_PATTERNS:
        yield from scope_dir.rglob(pattern)


def _job_retention_dt(job):
    row = dict(job or {})
    for key in ("settled_at", "updated_at", "ready_at", "created_at"):
        dt = _parse_dt(row.get(key, ""))
        if dt is not None:
            return dt
    return None


def _build_archive_row(job):
    row = dict(job or {})
    artifacts = [dict(item) for item in list(row.get("artifacts", []) or []) if isinstance(item, dict)]
    return {
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "job_id": str(row.get("job_id", "") or "").strip(),
        "status": str(row.get("status", "") or "").strip(),
        "scope_key": str(row.get("scope_key", "") or "").strip(),
        "race_id": str(row.get("race_id", "") or "").strip(),
        "race_date": str(row.get("race_date", "") or "").strip(),
        "location": str(row.get("location", "") or "").strip(),
        "scheduled_off_time": str(row.get("scheduled_off_time", "") or "").strip(),
        "current_run_id": str(row.get("current_run_id", "") or "").strip(),
        "actual_top1": str(row.get("actual_top1", "") or "").strip(),
        "actual_top2": str(row.get("actual_top2", "") or "").strip(),
        "actual_top3": str(row.get("actual_top3", "") or "").strip(),
        "error_message": str(row.get("error_message", "") or "").strip(),
        "created_at": str(row.get("created_at", "") or "").strip(),
        "updated_at": str(row.get("updated_at", "") or "").strip(),
        "ready_at": str(row.get("ready_at", "") or "").strip(),
        "settled_at": str(row.get("settled_at", "") or "").strip(),
        "artifact_count": len(artifacts),
    }


def _archive_old_jobs(*, job_retention_days: int, dry_run: bool):
    cutoff = datetime.now() - timedelta(days=max(1, int(job_retention_days or 3)))
    jobs = _load_jobs()
    if not jobs:
        return {
            "job_retention_days": int(job_retention_days),
            "archived_job_ids": [],
            "removed_artifact_files": [],
            "removed_artifact_dirs": [],
            "archive_path": str(RACE_JOBS_ARCHIVE_PATH),
        }

    kept_jobs = []
    archived_rows = []
    archived_job_ids = []
    removed_artifact_files = []
    removed_artifact_dirs = []
    changed = False

    for job in jobs:
        row = dict(job)
        status = str(row.get("status", "") or "").strip().lower()
        retain_dt = _job_retention_dt(row)
        if status not in ARCHIVABLE_JOB_STATUSES or retain_dt is None or retain_dt >= cutoff:
            kept_jobs.append(row)
            continue

        archived_rows.append(_build_archive_row(row))
        archived_job_ids.append(str(row.get("job_id", "") or "").strip())
        changed = True

        touched_dirs = set()
        for artifact in [dict(item) for item in list(row.get("artifacts", []) or []) if isinstance(item, dict)]:
            stored_path = str(artifact.get("stored_path", "") or "").strip()
            if not stored_path:
                continue
            path = Path(stored_path)
            touched_dirs.add(str(path.parent))
            if path.exists() and _remove_file(path, dry_run):
                removed_artifact_files.append(str(path))
        for dir_text in sorted(touched_dirs):
            if _remove_empty_dir(Path(dir_text), dry_run):
                removed_artifact_dirs.append(dir_text)

    if changed and not dry_run:
        _append_archive_rows(archived_rows)
        _save_jobs(kept_jobs)

    return {
        "job_retention_days": int(job_retention_days),
        "archived_job_ids": archived_job_ids,
        "removed_artifact_files": removed_artifact_files,
        "removed_artifact_dirs": removed_artifact_dirs,
        "archive_path": str(RACE_JOBS_ARCHIVE_PATH),
    }


def _cleanup_workspaces_and_orphan_artifacts(*, workspace_hours: int, orphan_artifact_days: int, dry_run: bool):
    now = datetime.now()
    workspace_cutoff = now - timedelta(hours=max(1, int(workspace_hours or 12)))
    artifact_cutoff = now - timedelta(days=max(1, int(orphan_artifact_days or 7)))

    removed_workspaces = []
    removed_artifacts = []

    if WORKSPACES_DIR.exists():
        for path in WORKSPACES_DIR.iterdir():
            if not path.is_dir():
                continue
            if not _older_than(path, workspace_cutoff):
                continue
            if _remove_tree(path, dry_run):
                removed_workspaces.append(str(path))

    active_artifacts = _active_artifact_dirs()
    if ARTIFACTS_DIR.exists():
        for path in ARTIFACTS_DIR.iterdir():
            if not path.is_dir():
                continue
            resolved = str(path.resolve())
            if resolved in active_artifacts:
                continue
            if not _older_than(path, artifact_cutoff):
                continue
            if _remove_tree(path, dry_run):
                removed_artifacts.append(str(path))

    return {
        "dry_run": bool(dry_run),
        "workspace_hours": int(workspace_hours),
        "orphan_artifact_days": int(orphan_artifact_days),
        "removed_workspaces": removed_workspaces,
        "removed_artifacts": removed_artifacts,
    }


def _cleanup_referenced_artifacts(*, referenced_artifact_days: int, skip_job_ids=None, dry_run: bool):
    cutoff = datetime.now() - timedelta(days=max(1, int(referenced_artifact_days or 3)))
    skipped = {str(x).strip() for x in list(skip_job_ids or []) if str(x).strip()}
    jobs = _load_jobs()
    if not jobs:
        return {
            "referenced_artifact_days": int(referenced_artifact_days),
            "removed_files": [],
            "removed_dirs": [],
            "updated_jobs": [],
        }

    removed_files = []
    removed_dirs = []
    updated_jobs = []
    out_jobs = []
    changed = False

    for job in jobs:
        row = dict(job)
        job_id = str(row.get("job_id", "") or "").strip()
        if job_id and job_id in skipped:
            out_jobs.append(row)
            continue
        status = str(row.get("status", "") or "").strip().lower()
        artifacts = [dict(item) for item in list(row.get("artifacts", []) or []) if isinstance(item, dict)]
        if status not in REFERENCED_ARTIFACT_STATUSES or not artifacts:
            out_jobs.append(row)
            continue

        kept_artifacts = []
        job_changed = False
        touched_dirs = set()
        for artifact in artifacts:
            stored_path = str(artifact.get("stored_path", "") or "").strip()
            if not stored_path:
                kept_artifacts.append(artifact)
                continue
            path = Path(stored_path)
            uploaded_at = _parse_dt(artifact.get("uploaded_at", ""))
            is_old = uploaded_at < cutoff if uploaded_at else _older_than(path, cutoff)
            if not is_old:
                kept_artifacts.append(artifact)
                continue
            touched_dirs.add(str(path.parent))
            if path.exists():
                if _remove_file(path, dry_run):
                    removed_files.append(str(path))
                    job_changed = True
                    continue
                kept_artifacts.append(artifact)
                continue
            job_changed = True

        if job_changed:
            row["artifacts"] = kept_artifacts
            updated_jobs.append(str(row.get("job_id", "") or ""))
            changed = True
            for dir_text in sorted(touched_dirs):
                if _remove_empty_dir(Path(dir_text), dry_run):
                    removed_dirs.append(dir_text)
        out_jobs.append(row)

    if changed and not dry_run:
        _save_jobs(out_jobs)

    return {
        "referenced_artifact_days": int(referenced_artifact_days),
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "updated_jobs": updated_jobs,
    }


def cleanup(
    *,
    workspace_hours: int = 12,
    orphan_artifact_days: int = 7,
    job_retention_days: int = 3,
    referenced_artifact_days: int = 3,
    cache_days: int = 14,
    experimental_days: int = 14,
    research_days: int = 30,
    include_offline_research: bool = False,
    include_legacy_debug: bool = False,
    dry_run: bool = False,
):
    now = datetime.now()
    cache_cutoff = now - timedelta(days=max(1, int(cache_days or 14)))
    experimental_cutoff = now - timedelta(days=max(1, int(experimental_days or 14)))
    research_cutoff = now - timedelta(days=max(1, int(research_days or 30)))
    prompt_input_cutoff = now - timedelta(days=max(1, int(referenced_artifact_days or 3)))

    archived_jobs_summary = _archive_old_jobs(
        job_retention_days=job_retention_days,
        dry_run=dry_run,
    )
    runtime_summary = _cleanup_workspaces_and_orphan_artifacts(
        workspace_hours=workspace_hours,
        orphan_artifact_days=orphan_artifact_days,
        dry_run=dry_run,
    )
    referenced_artifact_summary = _cleanup_referenced_artifacts(
        referenced_artifact_days=referenced_artifact_days,
        skip_job_ids=archived_jobs_summary.get("archived_job_ids", []),
        dry_run=dry_run,
    )

    removed_cache_files = []
    for cache_dir_name in CACHE_DIR_NAMES:
        cache_dir = DATA_DIR / cache_dir_name
        if not cache_dir.exists():
            continue
        removed_cache_files.extend(_remove_old_files(cache_dir.glob("*.json"), cache_cutoff, dry_run))

    removed_official_results = []
    official_results_dir = SHARED_DIR / "official_results"
    if official_results_dir.exists():
        removed_official_results.extend(
            _remove_old_files(official_results_dir.glob("official_result_*.json"), cache_cutoff, dry_run)
        )

    removed_experimental_files = []
    removed_legacy_debug_files = []
    removed_legacy_race_files = []
    removed_prompt_input_files = []
    removed_offline_research_files = []
    for scope_dir in _iter_scope_dirs():
        removed_experimental_files.extend(
            _remove_old_files(_collect_scope_root_paths(scope_dir), experimental_cutoff, dry_run)
        )
        if include_legacy_debug:
            removed_legacy_debug_files.extend(
                _remove_old_files(_collect_legacy_debug_paths(scope_dir), experimental_cutoff, dry_run)
            )
        removed_legacy_race_files.extend(
            _remove_old_files(_collect_race_dir_paths(scope_dir), experimental_cutoff, dry_run)
        )
        removed_prompt_input_files.extend(
            _remove_old_files(_collect_prompt_input_paths(scope_dir), prompt_input_cutoff, dry_run)
        )
        if include_offline_research:
            research_paths = [scope_dir / name for name in OFFLINE_RESEARCH_FILES]
            removed_offline_research_files.extend(
                _remove_old_files(research_paths, research_cutoff, dry_run)
            )

    summary = {
        "dry_run": bool(dry_run),
        "workspace_hours": int(workspace_hours),
        "orphan_artifact_days": int(orphan_artifact_days),
        "job_retention_days": int(job_retention_days),
        "referenced_artifact_days": int(referenced_artifact_days),
        "cache_days": int(cache_days),
        "experimental_days": int(experimental_days),
        "research_days": int(research_days),
        "include_offline_research": bool(include_offline_research),
        "include_legacy_debug": bool(include_legacy_debug),
        "archived_jobs": archived_jobs_summary,
        "runtime": runtime_summary,
        "referenced_artifacts": referenced_artifact_summary,
        "removed_cache_files": removed_cache_files,
        "removed_official_results": removed_official_results,
        "removed_experimental_files": removed_experimental_files,
        "removed_legacy_debug_files": removed_legacy_debug_files,
        "removed_legacy_race_files": removed_legacy_race_files,
        "removed_prompt_input_files": removed_prompt_input_files,
        "removed_offline_research_files": removed_offline_research_files,
    }
    summary["totals"] = {
        "archived_jobs": len(archived_jobs_summary.get("archived_job_ids", [])),
        "archived_job_artifact_files": len(archived_jobs_summary.get("removed_artifact_files", [])),
        "archived_job_artifact_dirs": len(archived_jobs_summary.get("removed_artifact_dirs", [])),
        "workspaces": len(runtime_summary.get("removed_workspaces", [])),
        "orphan_artifacts": len(runtime_summary.get("removed_artifacts", [])),
        "referenced_artifact_files": len(referenced_artifact_summary.get("removed_files", [])),
        "referenced_artifact_dirs": len(referenced_artifact_summary.get("removed_dirs", [])),
        "updated_jobs": len(referenced_artifact_summary.get("updated_jobs", [])),
        "cache_files": len(removed_cache_files),
        "official_results": len(removed_official_results),
        "experimental_files": len(removed_experimental_files),
        "legacy_debug_files": len(removed_legacy_debug_files),
        "legacy_race_files": len(removed_legacy_race_files),
        "prompt_input_files": len(removed_prompt_input_files),
        "offline_research_files": len(removed_offline_research_files),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean stale runtime/cache/experimental artifacts and archive old "
            "completed jobs while keeping web-facing run history and race snapshots."
        )
    )
    parser.add_argument("--workspace-hours", type=int, default=12)
    parser.add_argument("--orphan-artifact-days", type=int, default=7)
    parser.add_argument("--job-retention-days", type=int, default=3)
    parser.add_argument("--referenced-artifact-days", type=int, default=3)
    parser.add_argument("--cache-days", type=int, default=14)
    parser.add_argument("--experimental-days", type=int, default=14)
    parser.add_argument("--research-days", type=int, default=30)
    parser.add_argument("--include-offline-research", action="store_true")
    parser.add_argument("--include-legacy-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = cleanup(
        workspace_hours=args.workspace_hours,
        orphan_artifact_days=args.orphan_artifact_days,
        job_retention_days=args.job_retention_days,
        referenced_artifact_days=args.referenced_artifact_days,
        cache_days=args.cache_days,
        experimental_days=args.experimental_days,
        research_days=args.research_days,
        include_offline_research=args.include_offline_research,
        include_legacy_debug=args.include_legacy_debug,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

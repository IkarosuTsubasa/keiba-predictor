from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = BASE_DIR / "data" / "_shared"
ARTIFACTS_DIR = SHARED_DIR / "race_job_artifacts"
WORKSPACES_DIR = SHARED_DIR / "job_workspaces"
RACE_JOBS_PATH = SHARED_DIR / "race_jobs.json"


def _load_jobs():
    if not RACE_JOBS_PATH.exists():
        return []
    try:
        payload = json.loads(RACE_JOBS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


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


def _remove_tree(path: Path, dry_run: bool):
    if dry_run:
        return True
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def cleanup(*, workspace_hours: int = 12, orphan_artifact_days: int = 7, dry_run: bool = False):
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


def main():
    parser = argparse.ArgumentParser(description="Clean stale runtime workspaces and orphan artifact directories.")
    parser.add_argument("--workspace-hours", type=int, default=12)
    parser.add_argument("--orphan-artifact-days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = cleanup(
        workspace_hours=args.workspace_hours,
        orphan_artifact_days=args.orphan_artifact_days,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

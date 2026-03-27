from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from cleanup_runtime_artifacts import cleanup as cleanup_runtime_artifacts


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SHARED_DIR = DATA_DIR / "_shared"
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
OFFLINE_RESEARCH_FILES = (
    "offline_eval.csv",
    "context_dataset.csv",
    "context_summary.csv",
    "history_races.csv",
)


def _older_than(path: Path, cutoff: datetime) -> bool:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return False
    return modified < cutoff


def _remove_file(path: Path, dry_run: bool) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if dry_run:
        return True
    try:
        path.unlink()
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


def cleanup_periodic_storage(
    *,
    workspace_hours: int = 12,
    orphan_artifact_days: int = 7,
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

    runtime_summary = cleanup_runtime_artifacts(
        workspace_hours=workspace_hours,
        orphan_artifact_days=orphan_artifact_days,
        dry_run=dry_run,
    )

    removed_cache_files = []
    for cache_dir_name in CACHE_DIR_NAMES:
        cache_dir = DATA_DIR / cache_dir_name
        if not cache_dir.exists():
            continue
        removed_cache_files.extend(
            _remove_old_files(cache_dir.glob("*.json"), cache_cutoff, dry_run)
        )

    official_results_dir = SHARED_DIR / "official_results"
    removed_official_results = []
    if official_results_dir.exists():
        removed_official_results.extend(
            _remove_old_files(official_results_dir.glob("official_result_*.json"), cache_cutoff, dry_run)
        )

    removed_experimental_files = []
    removed_legacy_debug_files = []
    removed_legacy_race_files = []
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
        if include_offline_research:
            research_paths = [scope_dir / name for name in OFFLINE_RESEARCH_FILES]
            removed_offline_research_files.extend(
                _remove_old_files(research_paths, research_cutoff, dry_run)
            )

    summary = {
        "dry_run": bool(dry_run),
        "workspace_hours": int(workspace_hours),
        "orphan_artifact_days": int(orphan_artifact_days),
        "cache_days": int(cache_days),
        "experimental_days": int(experimental_days),
        "research_days": int(research_days),
        "include_offline_research": bool(include_offline_research),
        "include_legacy_debug": bool(include_legacy_debug),
        "runtime": runtime_summary,
        "removed_cache_files": removed_cache_files,
        "removed_official_results": removed_official_results,
        "removed_experimental_files": removed_experimental_files,
        "removed_legacy_debug_files": removed_legacy_debug_files,
        "removed_legacy_race_files": removed_legacy_race_files,
        "removed_offline_research_files": removed_offline_research_files,
    }
    summary["totals"] = {
        "workspaces": len(runtime_summary.get("removed_workspaces", [])),
        "orphan_artifacts": len(runtime_summary.get("removed_artifacts", [])),
        "cache_files": len(removed_cache_files),
        "official_results": len(removed_official_results),
        "experimental_files": len(removed_experimental_files),
        "legacy_debug_files": len(removed_legacy_debug_files),
        "legacy_race_files": len(removed_legacy_race_files),
        "offline_research_files": len(removed_offline_research_files),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean periodic runtime/cache/experimental artifacts while keeping "
            "web-facing run history and race snapshots."
        )
    )
    parser.add_argument("--workspace-hours", type=int, default=12)
    parser.add_argument("--orphan-artifact-days", type=int, default=7)
    parser.add_argument("--cache-days", type=int, default=14)
    parser.add_argument("--experimental-days", type=int, default=14)
    parser.add_argument("--research-days", type=int, default=30)
    parser.add_argument("--include-offline-research", action="store_true")
    parser.add_argument("--include-legacy-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = cleanup_periodic_storage(
        workspace_hours=args.workspace_hours,
        orphan_artifact_days=args.orphan_artifact_days,
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

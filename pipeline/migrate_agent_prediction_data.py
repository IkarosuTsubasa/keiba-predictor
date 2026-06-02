import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from web_data.agent_predictions import agent_prediction_write_dir, agent_result_write_dir


def _legacy_sources(kind):
    if kind == "predictions":
        return [
            REPO_ROOT / "data" / "predictions",
            REPO_ROOT / "keiba_llm_agent" / "data" / "predictions",
        ]
    if kind == "results":
        return [
            REPO_ROOT / "data" / "results",
            REPO_ROOT / "keiba_llm_agent" / "data" / "results",
        ]
    return []


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _race_id_from_json(path):
    try:
        payload = _read_json(path)
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    race_id = str(payload.get("race_id", "") or "").strip()
    if race_id:
        return race_id
    race_info = payload.get("race_info")
    if isinstance(race_info, dict):
        return str(race_info.get("race_id", "") or "").strip()
    return ""


def _iter_legacy_json_files(kind):
    seen = set()
    for directory in _legacy_sources(kind):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            resolved = path.resolve()
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)
            yield resolved


def _copy_one(source_path, target_dir, *, overwrite=False, dry_run=False):
    race_id = _race_id_from_json(source_path) or source_path.stem
    if not race_id:
        return {"source": str(source_path), "status": "invalid", "reason": "race_id missing"}
    target_path = target_dir / f"{race_id}.json"
    target_exists = target_path.exists()
    if target_exists:
        if filecmp.cmp(source_path, target_path, shallow=False):
            return {"source": str(source_path), "target": str(target_path), "status": "existing_same"}
        if not overwrite:
            return {"source": str(source_path), "target": str(target_path), "status": "skipped_existing"}
    if not dry_run:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    return {
        "source": str(source_path),
        "target": str(target_path),
        "status": "would_copy" if dry_run else ("overwritten" if target_exists else "copied"),
    }


def migrate_kind(base_dir, kind, *, overwrite=False, dry_run=False, include_items=True):
    if kind == "predictions":
        target_dir = agent_prediction_write_dir(base_dir)
    elif kind == "results":
        target_dir = agent_result_write_dir(base_dir)
    else:
        raise ValueError(f"unsupported kind: {kind}")
    rows = [
        _copy_one(path, target_dir, overwrite=overwrite, dry_run=dry_run)
        for path in _iter_legacy_json_files(kind)
    ]
    counts = {}
    for row in rows:
        status = str(row.get("status", "") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    summary = {"kind": kind, "target_dir": str(target_dir), "counts": counts}
    if include_items:
        summary["items"] = rows
    return summary


def migrate_all(*, base_dir=PIPELINE_DIR, overwrite=False, dry_run=False, include_items=True):
    summaries = [
        migrate_kind(base_dir, "predictions", overwrite=overwrite, dry_run=dry_run, include_items=include_items),
        migrate_kind(base_dir, "results", overwrite=overwrite, dry_run=dry_run, include_items=include_items),
    ]
    return {"dry_run": bool(dry_run), "overwrite": bool(overwrite), "summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description="Copy legacy keiba_llm_agent prediction/result JSON files into persistent pipeline data dirs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the migration summary without copying files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target files when content differs.")
    parser.add_argument("--verbose", action="store_true", help="Include every file in the output.")
    args = parser.parse_args()
    summary = migrate_all(overwrite=args.overwrite, dry_run=args.dry_run, include_items=args.verbose)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import csv
from pathlib import Path


def _iter_scope_dirs(data_dir: Path):
    for name in ("central_dirt", "central_turf", "local"):
        path = data_dir / name
        if path.exists() and path.is_dir():
            yield path


def _remove_files(paths):
    removed = []
    for path in paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed.append(str(path))
        except Exception:
            continue
    return removed


def _reset_runs_csv(path: Path):
    if not path.exists():
        return 0
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue
    else:
        return 0

    reset_fields = [
        "policy_path",
        "gemini_policy_path",
        "siliconflow_policy_path",
        "openai_policy_path",
        "grok_policy_path",
    ]
    touched = 0
    for row in rows:
        changed = False
        for key in reset_fields:
            if key in fieldnames and str(row.get(key, "") or "").strip():
                row[key] = ""
                changed = True
        if changed:
            touched += 1
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return touched


def reset_llm_state(base_dir):
    base_path = Path(base_dir)
    data_dir = base_path / "data"
    shared_dir = data_dir / "_shared"
    removed = {
        "ledger_files": [],
        "policy_files": [],
        "cache_files": [],
    }

    if shared_dir.exists():
        removed["ledger_files"] = _remove_files(shared_dir.glob("*_ticket_ledger.csv"))

    removed["cache_files"].extend(_remove_files((data_dir / "policy_cache_gemini").glob("*.json")))
    removed["cache_files"].extend(_remove_files((data_dir / "policy_cache_siliconflow").glob("*.json")))
    removed["cache_files"].extend(_remove_files((data_dir / "policy_cache_openai").glob("*.json")))
    removed["cache_files"].extend(_remove_files((data_dir / "policy_cache_grok").glob("*.json")))

    for scope_dir in _iter_scope_dirs(data_dir):
        removed["policy_files"].extend(_remove_files(scope_dir.rglob("*_policy_*.json")))

    runs_updated = 0
    for scope_dir in _iter_scope_dirs(data_dir):
        runs_updated += _reset_runs_csv(scope_dir / "runs.csv")

    summary = {
        "ledger_files_removed": len(removed["ledger_files"]),
        "policy_files_removed": len(removed["policy_files"]),
        "cache_files_removed": len(removed["cache_files"]),
        "runs_rows_reset": runs_updated,
        "removed_files": removed,
    }
    return summary


__all__ = ["reset_llm_state"]

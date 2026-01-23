import csv
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_ROOT = BASE_DIR / "data"

SCOPES = ("central_dirt", "central_turf", "local")

PATH_FIELDS = {
    "predictions_path": "predictions",
    "odds_path": "odds",
    "wide_odds_path": "wide_odds",
    "fuku_odds_path": "fuku_odds",
    "quinella_odds_path": "quinella_odds",
    "trifecta_odds_path": "trifecta_odds",
    "plan_path": "bet_plan",
}

CANONICAL_FIELDS = [
    "run_id",
    "timestamp",
    "race_url",
    "race_id",
    "history_url",
    "trigger_race",
    "scope",
    "surface",
    "distance",
    "budget_yen",
    "style",
    "strategy",
    "strategy_reason",
    "predictor_strategy",
    "predictor_reason",
    "config_version",
    "predictions_path",
    "odds_path",
    "wide_odds_path",
    "fuku_odds_path",
    "quinella_odds_path",
    "trifecta_odds_path",
    "plan_path",
    "tickets",
    "amount_yen",
]


def extract_race_id(url):
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        race_id = parse_qs(parsed.query).get("race_id", [""])[0]
        race_id = re.sub(r"\D", "", race_id)
        return race_id
    except Exception:
        return ""


def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with open(path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows, reader.fieldnames or []
        except UnicodeDecodeError:
            continue
    return [], []


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def resolve_path(raw):
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidates = [ROOT_DIR / path, BASE_DIR / path]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def move_path(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return False
    if dest.exists():
        return False
    shutil.move(str(src), str(dest))
    return True


def migrate_scope(scope_key):
    scope_dir = DATA_ROOT / scope_key
    if not scope_dir.exists():
        return {"scope": scope_key, "moved": 0, "skipped": 0, "missing": 0}

    runs_path = scope_dir / "runs.csv"
    if not runs_path.exists():
        return {"scope": scope_key, "moved": 0, "skipped": 0, "missing": 0}

    runs, fields = read_csv(runs_path)
    if not runs:
        return {"scope": scope_key, "moved": 0, "skipped": 0, "missing": 0}

    extra_fields = [name for name in fields if name and name not in CANONICAL_FIELDS]
    output_fields = CANONICAL_FIELDS + extra_fields

    moved = 0
    skipped = 0
    missing = 0

    for row in runs:
        row.setdefault("scope", scope_key)
        run_id = str(row.get("run_id", "")).strip()
        race_id = str(row.get("race_id", "")).strip()
        if not race_id:
            race_id = extract_race_id(str(row.get("race_url", "")).strip())
            if race_id:
                row["race_id"] = race_id
        if not race_id:
            missing += 1
            continue

        race_dir = scope_dir / race_id
        for col, prefix in PATH_FIELDS.items():
            dest = race_dir / f"{prefix}_{run_id}_{race_id}.csv"
            current = resolve_path(row.get(col))
            if dest.exists():
                row[col] = str(dest)
                if current and current.exists() and current.resolve() != dest.resolve():
                    skipped += 1
                continue

            source = None
            if current and current.exists():
                source = current
            else:
                pattern = f"{prefix}_{run_id}*.csv"
                candidates = list(scope_dir.glob(pattern))
                if candidates:
                    source = candidates[0]

            if source and source.exists():
                try:
                    if move_path(source, dest):
                        moved += 1
                    row[col] = str(dest)
                except OSError:
                    skipped += 1
            else:
                if dest.exists():
                    row[col] = str(dest)

    write_csv(runs_path, output_fields, runs)
    return {"scope": scope_key, "moved": moved, "skipped": skipped, "missing": missing}


def main():
    summaries = []
    for scope_key in SCOPES:
        summaries.append(migrate_scope(scope_key))

    print("Migration summary:")
    for item in summaries:
        print(
            f"- {item['scope']}: moved={item['moved']} skipped={item['skipped']} missing_race_id={item['missing']}"
        )


if __name__ == "__main__":
    main()

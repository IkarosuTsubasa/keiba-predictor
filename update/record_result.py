import csv
import os
import subprocess
import sys
from pathlib import Path

from surface_scope import get_data_dir, get_scope_key, migrate_legacy_data


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = None


def init_scope():
    scope_key = get_scope_key()
    migrate_legacy_data(BASE_DIR, scope_key)
    os.environ["SCOPE_KEY"] = scope_key
    global DATA_DIR
    DATA_DIR = get_data_dir(BASE_DIR, scope_key)
    return scope_key


def load_runs():
    path = DATA_DIR / "runs.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_csv_header(path, fieldnames):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = reader.fieldnames or []
        rows = list(reader)
    if existing == fieldnames:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_csv(path, fieldnames, row):
    ensure_csv_header(path, fieldnames)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prompt_value(label):
    return input(label).strip()


def main():
    init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No runs found. Run update/run_pipeline.py first.")
        sys.exit(1)

    last_run = runs[-1]["run_id"]
    run_id = prompt_value(f"Run ID [{last_run}]: ") or last_run

    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        print("Run ID not found.")
        sys.exit(1)

    strategy = run.get("strategy", "")
    if strategy:
        print(f"Strategy: {strategy}")

    profit_raw = prompt_value("Profit/Loss yen (e.g. -500 or 1200): ")
    try:
        profit_yen = int(float(profit_raw))
    except ValueError:
        print("Invalid profit value.")
        sys.exit(1)

    base_amount = 0
    try:
        base_amount = int(float(run.get("amount_yen", 0)))
    except ValueError:
        base_amount = 0
    if base_amount <= 0:
        try:
            base_amount = int(float(run.get("budget_yen", 0)))
        except ValueError:
            base_amount = 0

    roi = ""
    if base_amount > 0:
        roi = round((base_amount + profit_yen) / base_amount, 4)

    note = prompt_value("Notes (optional): ")

    row = {
        "run_id": run_id,
        "strategy": strategy,
        "profit_yen": profit_yen,
        "base_amount": base_amount,
        "roi": roi,
        "note": note,
    }
    append_csv(DATA_DIR / "results.csv", list(row.keys()), row)
    print(f"Recorded result for {run_id}")
    optimizer = BASE_DIR / "optimize_params.py"
    if optimizer.exists():
        subprocess.run([sys.executable, str(optimizer)], check=False)


if __name__ == "__main__":
    main()

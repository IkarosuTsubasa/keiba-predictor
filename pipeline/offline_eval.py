import csv
import math
import os
from datetime import datetime
from pathlib import Path

from surface_scope import get_data_dir, get_scope_key, migrate_legacy_data


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = None
RUNS_PATH = None
RESULTS_PATH = None
PRED_RESULTS_PATH = None
OUT_PATH = None


def init_scope():
    scope_key = get_scope_key()
    migrate_legacy_data(BASE_DIR, scope_key)
    os.environ["SCOPE_KEY"] = scope_key
    global DATA_DIR, RUNS_PATH, RESULTS_PATH, PRED_RESULTS_PATH, OUT_PATH
    DATA_DIR = get_data_dir(BASE_DIR, scope_key)
    RUNS_PATH = DATA_DIR / "runs.csv"
    RESULTS_PATH = DATA_DIR / "results.csv"
    PRED_RESULTS_PATH = DATA_DIR / "predictor_results.csv"
    OUT_PATH = DATA_DIR / "offline_eval.csv"
    return scope_key


def load_csv(path):
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


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def std(values):
    if not values or len(values) < 2:
        return 0.0
    avg = mean(values)
    var = sum((v - avg) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def main():
    init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_csv(RUNS_PATH)
    if not runs:
        print("No runs found.")
        return

    raw_window = input("Window size [10]: ").strip()
    try:
        window = int(raw_window) if raw_window else 10
    except ValueError:
        window = 10
    window = max(1, window)

    recent_runs = runs[-window:]
    run_ids = [row.get("run_id") for row in recent_runs if row.get("run_id")]

    results = load_csv(RESULTS_PATH)
    results_map = {row.get("run_id"): row for row in results}

    pred_results = load_csv(PRED_RESULTS_PATH)
    pred_map = {row.get("run_id"): row for row in pred_results}
    result_rows = []
    for run_id in run_ids:
        for row in results:
            if row.get("run_id") == run_id:
                result_rows.append(row)

    def summarize_profit(rows):
        rois = []
        profits = []
        win_count = 0
        loss_count = 0
        flat_count = 0
        for row in rows:
            roi = to_float(row.get("roi"))
            if roi is not None:
                rois.append(roi)
            profit = to_int(row.get("profit_yen"))
            if profit is None:
                continue
            profits.append(profit)
            if profit > 0:
                win_count += 1
            elif profit < 0:
                loss_count += 1
            else:
                flat_count += 1
        return {
            "rois": rois,
            "profits": profits,
            "win_count": win_count,
            "loss_count": loss_count,
            "flat_count": flat_count,
        }

    hit_counts = []
    top1_hits = []
    top1_in_top3 = []
    top3_exact = []
    for run_id in run_ids:
        row = pred_map.get(run_id)
        if not row:
            continue
        hit = to_int(row.get("top3_hit_count"))
        top1 = to_int(row.get("top1_hit"))
        top1_top3 = to_int(row.get("top1_in_top3"))
        exact = to_int(row.get("top3_exact"))
        if hit is not None:
            hit_counts.append(hit)
        if top1 is not None:
            top1_hits.append(top1)
        if top1_top3 is not None:
            top1_in_top3.append(top1_top3)
        if exact is not None:
            top3_exact.append(exact)

    overall = summarize_profit(result_rows)

    pred_hit_avg = mean(hit_counts)
    pred_top1_rate = mean(top1_hits)
    pred_top1_in_top3_rate = mean(top1_in_top3)
    pred_exact_rate = mean(top3_exact)

    print("\nOffline evaluation (recent runs)")
    print(f"Window: {window} | Runs: {len(run_ids)}")
    print(f"ROI samples: {len(overall['rois'])} | Predictor samples: {len(hit_counts)}")
    if overall["rois"]:
        roi_avg = mean(overall["rois"])
        roi_std = std(overall["rois"])
        roi_min = min(overall["rois"])
        roi_max = max(overall["rois"])
        profit_avg = mean(overall["profits"])
        total_profit = sum(overall["profits"]) if overall["profits"] else 0
        print(
            f"ROI avg={roi_avg:.4f}, std={roi_std:.4f}, "
            f"min={roi_min:.4f}, max={roi_max:.4f}"
        )
        print(
            f"Profit avg={profit_avg:.1f} yen, total={total_profit} yen, "
            f"win/loss/flat={overall['win_count']}/{overall['loss_count']}/{overall['flat_count']}"
        )
    else:
        print("No ROI data.")

    budget_groups = {}
    for row in result_rows:
        budget = to_int(row.get("budget_yen"))
        if budget is None:
            budget = 2000
        budget_groups.setdefault(budget, []).append(row)
    for budget in sorted(budget_groups.keys()):
        stats = summarize_profit(budget_groups[budget])
        if not stats["rois"]:
            continue
        roi_avg = mean(stats["rois"])
        roi_std = std(stats["rois"])
        total_profit = sum(stats["profits"]) if stats["profits"] else 0
        print(
            f"[Budget {budget}] ROI avg={roi_avg:.4f}, std={roi_std:.4f}, "
            f"profit_total={total_profit} yen, samples={len(stats['rois'])}"
        )

    if hit_counts:
        print(
            f"Top3 hit avg={pred_hit_avg:.2f}, "
            f"Top1 hit rate={pred_top1_rate:.2f}, "
            f"Top1 in Top3 rate={pred_top1_in_top3_rate:.2f}, "
            f"Exact Top3 rate={pred_exact_rate:.2f}"
        )
    else:
        print("No predictor data.")

    rows_to_save = []
    targets = {"all": result_rows}
    for budget, rows in budget_groups.items():
        targets[str(budget)] = rows
    for budget_key, rows in targets.items():
        stats = summarize_profit(rows)
        roi_avg = mean(stats["rois"])
        roi_std = std(stats["rois"])
        roi_min = min(stats["rois"]) if stats["rois"] else None
        roi_max = max(stats["rois"]) if stats["rois"] else None
        profit_avg = mean(stats["profits"])
        total_profit = sum(stats["profits"]) if stats["profits"] else 0
        rows_to_save.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "budget_yen": budget_key,
                "window": window,
                "runs": len(run_ids),
                "roi_samples": len(stats["rois"]),
                "roi_avg": round(roi_avg, 6) if roi_avg is not None else "",
                "roi_std": round(roi_std, 6) if roi_avg is not None else "",
                "roi_min": round(roi_min, 6) if roi_min is not None else "",
                "roi_max": round(roi_max, 6) if roi_max is not None else "",
                "profit_avg": round(profit_avg, 2) if profit_avg is not None else "",
                "profit_total": total_profit,
                "win_count": stats["win_count"],
                "loss_count": stats["loss_count"],
                "flat_count": stats["flat_count"],
                "pred_samples": len(hit_counts),
                "pred_hit_avg": round(pred_hit_avg, 3) if pred_hit_avg is not None else "",
                "pred_top1_rate": round(pred_top1_rate, 3) if pred_top1_rate is not None else "",
                "pred_top1_in_top3_rate": round(pred_top1_in_top3_rate, 3) if pred_top1_in_top3_rate is not None else "",
                "pred_exact_rate": round(pred_exact_rate, 3) if pred_exact_rate is not None else "",
            }
        )
    for row in rows_to_save:
        append_csv(OUT_PATH, list(row.keys()), row)
    print(f"Saved: {OUT_PATH}")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()

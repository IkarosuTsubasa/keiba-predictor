import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from surface_scope import (
    get_data_dir,
    get_predictor_config_path,
    get_scope_key,
    migrate_legacy_data,
)

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = None
PRED_CONFIG_PATH = None
PRED_RESULTS_PATH = None
ODDS_PATH = ROOT_DIR / "odds.csv"


def init_scope():
    scope_key = get_scope_key()
    migrate_legacy_data(BASE_DIR, scope_key)
    os.environ["SCOPE_KEY"] = scope_key
    global DATA_DIR, PRED_CONFIG_PATH, PRED_RESULTS_PATH
    DATA_DIR = get_data_dir(BASE_DIR, scope_key)
    PRED_CONFIG_PATH = get_predictor_config_path(BASE_DIR, scope_key)
    PRED_RESULTS_PATH = DATA_DIR / "predictor_results.csv"
    return scope_key


def load_runs():
    path = DATA_DIR / "runs.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_pred_path_for_run(run, run_id, race_id, last_run_id):
    candidates = []
    run_path = str(run.get("predictions_path") or "").strip()
    if run_path:
        candidates.append(Path(run_path))
    if race_id:
        candidates.append(DATA_DIR / race_id / f"predictions_{run_id}_{race_id}.csv")
    candidates.append(DATA_DIR / f"predictions_{run_id}.csv")
    for path in candidates:
        if path.exists():
            return path
    if last_run_id and run_id == last_run_id:
        latest = ROOT_DIR / "predictions.csv"
        if latest.exists():
            return latest
    return candidates[0] if candidates else None


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


def replace_rows_for_run(path, fieldnames, run_id, new_rows):
    if isinstance(new_rows, dict):
        new_rows = [new_rows]
    keep = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            keep = [row for row in reader if row.get("run_id") != run_id]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in keep:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        for row in new_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def normalize_name(value):
    return "".join(str(value or "").split())


def pick_score_column(columns):
    for key in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if key in columns:
            return key
    return ""


def prompt_horse_name(label):
    while True:
        value = input(label).strip()
        if value:
            return value
        print("Please enter a horse name.")


def prompt_actual_top3():
    name1 = prompt_horse_name("Enter 1st place horse name: ")
    name2 = prompt_horse_name("Enter 2nd place horse name: ")
    name3 = prompt_horse_name("Enter 3rd place horse name: ")
    return [name1, name2, name3]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_odds_map(odds_path):
    if not odds_path or not Path(odds_path).exists():
        return {}
    df = pd.read_csv(odds_path, encoding="utf-8-sig")
    if "name" not in df.columns or "odds" not in df.columns:
        return {}
    df["name_key"] = df["name"].apply(normalize_name)
    odds_map = {}
    for _, row in df.iterrows():
        val = safe_float(row.get("odds"))
        if val > 0:
            odds_map[str(row.get("name_key"))] = val
    return odds_map


def ndcg_at3(pred_names, actual_order):
    if not actual_order or len(actual_order) < 3:
        return 0.0
    rel_map = {
        actual_order[0]: 3,
        actual_order[1]: 2,
        actual_order[2]: 1,
    }
    dcg = 0.0
    for idx, name in enumerate(pred_names[:3]):
        rel = rel_map.get(name, 0)
        if rel <= 0:
            continue
        dcg += (2 ** rel - 1) / math.log2(idx + 2)
    idcg = 0.0
    for idx, rel in enumerate([3, 2, 1]):
        idcg += (2 ** rel - 1) / math.log2(idx + 2)
    return dcg / idcg if idcg > 0 else 0.0


def ev_score_from_odds(pred_items, odds_map):
    ev_values = []
    for name, prob in pred_items:
        odds = odds_map.get(normalize_name(name), 0)
        if odds > 0:
            ev_values.append(float(prob) * float(odds) - 1.0)
    if not ev_values:
        return 0.0
    ev_mean = sum(ev_values) / len(ev_values)
    return (math.tanh(ev_mean) + 1.0) / 2.0


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def load_predictor_state():
    try:
        cfg = json.loads(PRED_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"rank_ema": 0.5, "ev_ema": 0.5, "risk_score": 0.5}
    state = cfg.get("state", {})
    return {
        "rank_ema": clamp(float(state.get("rank_ema", 0.5)), 0.0, 1.0),
        "ev_ema": clamp(float(state.get("ev_ema", 0.5)), 0.0, 1.0),
        "risk_score": clamp(float(state.get("risk_score", 0.5)), 0.0, 1.0),
    }


def compute_confidence(pred_items, state):
    rank_ema = state.get("rank_ema", 0.5)
    ev_ema = state.get("ev_ema", 0.5)
    risk_score = state.get("risk_score", 0.5)
    validity = clamp(0.6 * rank_ema + 0.4 * ev_ema, 0.0, 1.0)
    stability = clamp(risk_score, 0.0, 1.0)
    consistency = 0.0
    if pred_items:
        probs = [safe_float(p) for _, p in pred_items]
        if len(probs) >= 3:
            gap = probs[0] - probs[2]
        elif len(probs) >= 2:
            gap = probs[0] - probs[1]
        else:
            gap = probs[0]
        consistency = clamp(gap / 0.15, 0.0, 1.0)
    confidence = math.sqrt(stability * validity) * consistency
    return {
        "confidence_score": round(clamp(confidence, 0.0, 1.0), 4),
        "stability_score": round(stability, 4),
        "validity_score": round(validity, 4),
        "consistency_score": round(consistency, 4),
        "rank_ema": round(rank_ema, 4),
        "ev_ema": round(ev_ema, 4),
        "risk_score": round(risk_score, 4),
    }


def main():
    init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No runs found. Run update/run_pipeline.py first.")
        sys.exit(1)

    last_run = runs[-1]["run_id"]
    run_id = input(f"Run ID [{last_run}]: ").strip() or last_run
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        print("Run ID not found.")
        sys.exit(1)

    race_id = str(run.get("race_id") or "").strip()
    pred_path_run = resolve_pred_path_for_run(run, run_id, race_id, last_run)
    pred_path = str(pred_path_run) if pred_path_run else ""
    pred_file = Path(pred_path) if pred_path else None
    if not pred_file or not pred_file.exists():
        if pred_path_run:
            print(f"Predictions file not found: {pred_path_run}")
        else:
            print("Predictions file not found for this run.")
        sys.exit(1)

    odds_path = run.get("odds_path") or str(DATA_DIR / f"odds_{run_id}.csv")
    if not Path(odds_path).exists():
        odds_path = str(ODDS_PATH)
    odds_map = load_odds_map(odds_path)

    df = pd.read_csv(pred_file, encoding="utf-8-sig")
    score_key = pick_score_column(df.columns)
    if "HorseName" not in df.columns or not score_key:
        print("Predictions file missing columns: HorseName / Top3Prob")
        sys.exit(1)

    pred_name_set = {
        normalize_name(n)
        for n in df["HorseName"].dropna().astype(str)
        if normalize_name(n)
    }
    pred_top = df.sort_values(score_key, ascending=False).head(3)
    pred_names_raw = pred_top["HorseName"].tolist()
    pred_items = list(zip(pred_top["HorseName"].tolist(), pred_top[score_key].tolist()))
    pred_names = [normalize_name(n) for n in pred_names_raw]
    pred_top5 = df.sort_values(score_key, ascending=False).head(5)
    pred_top5_names_raw = pred_top5["HorseName"].tolist()
    pred_top5_names = [normalize_name(n) for n in pred_top5_names_raw]

    print("\nPredicted Top3:")
    for i, name in enumerate(pred_names_raw, 1):
        print(f"{i}. {name}")

    actual_names_raw = prompt_actual_top3()
    actual_names = [normalize_name(n) for n in actual_names_raw]
    if pred_name_set:
        missing = [
            raw
            for raw, norm in zip(actual_names_raw, actual_names)
            if norm and norm not in pred_name_set
        ]
        if missing:
            print(f"[ERROR] Actual Top3 names not found in predictions: {', '.join(missing)}")
            sys.exit(1)

    hit_count = len(set(pred_names) & set(actual_names))
    top5_hit_count = len(set(pred_top5_names) & set(actual_names)) if pred_top5_names else 0
    top1_hit = 1 if pred_names and actual_names and pred_names[0] == actual_names[0] else 0
    top1_in_top3 = 1 if pred_names and pred_names[0] in actual_names else 0
    top3_exact = 1 if set(pred_names) == set(actual_names) else 0
    score = hit_count + top1_hit
    rank_score = ndcg_at3(pred_names, actual_names)
    hit_rate = hit_count / 3.0
    ev_score = ev_score_from_odds(pred_items, odds_map)
    score_total = 0.4 * rank_score + 0.4 * ev_score + 0.2 * hit_rate
    conf_state = load_predictor_state()
    conf = compute_confidence(pred_items, conf_state)

    row = {
        "run_id": run_id,
        "strategy": run.get("predictor_strategy", ""),
        "predictions_path": str(pred_file),
        "pred_top1": pred_names_raw[0] if len(pred_names_raw) > 0 else "",
        "pred_top2": pred_names_raw[1] if len(pred_names_raw) > 1 else "",
        "pred_top3": pred_names_raw[2] if len(pred_names_raw) > 2 else "",
        "actual_top1": actual_names_raw[0],
        "actual_top2": actual_names_raw[1],
        "actual_top3": actual_names_raw[2],
        "top3_hit_count": hit_count,
        "top5_hit_count": top5_hit_count,
        "top1_hit": top1_hit,
        "top1_in_top3": top1_in_top3,
        "top3_exact": top3_exact,
        "score": score,
        "rank_score": round(rank_score, 4),
        "ev_score": round(ev_score, 4),
        "hit_rate": round(hit_rate, 4),
        "score_total": round(score_total, 4),
        "confidence_score": conf["confidence_score"],
        "stability_score": conf["stability_score"],
        "validity_score": conf["validity_score"],
        "consistency_score": conf["consistency_score"],
        "rank_ema": conf["rank_ema"],
        "ev_ema": conf["ev_ema"],
        "risk_score": conf["risk_score"],
    }
    replace_rows_for_run(PRED_RESULTS_PATH, list(row.keys()), run_id, row)
    print(f"Recorded predictor result for {run_id}")
    optimizer = BASE_DIR / "optimize_predictor_params.py"
    if optimizer.exists():
        subprocess.run([sys.executable, str(optimizer)], check=False)


if __name__ == "__main__":
    main()

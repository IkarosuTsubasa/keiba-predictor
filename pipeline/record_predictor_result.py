import csv
import os
import re
import subprocess
import sys
from pathlib import Path

from surface_scope import (
    get_data_dir,
    get_predictor_config_path,
    get_scope_key,
    migrate_legacy_data,
)
from predictor_catalog import list_predictors, resolve_run_prediction_path
from gemini_portfolio import settle_run_tickets

import pandas as pd

from predictor_metrics import (
    compute_brier_score,
    compute_hit_at_k,
    compute_mrr_top3,
    compute_top3_hits_at_k,
)


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = None
PRED_CONFIG_PATH = None
PRED_RESULTS_PATH = None


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
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with open(path, "r", encoding=enc) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    return []


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
    for key in ("rank_score", "Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if key in columns:
            return key
    return ""


def infer_score_is_probability(df, score_key):
    if "score_is_probability" in df.columns:
        series = df["score_is_probability"].dropna()
        if not series.empty:
            text = str(series.iloc[0]).strip().lower()
            return text in {"1", "true", "t", "yes", "y"}
    return score_key in {"Top3Prob_model", "Top3Prob_est", "Top3Prob"}


def build_eval_frames(run_id, pred_df, horse_col, score_col, actual_names_norm):
    race_id = str(run_id or "").strip() or "__single_race__"
    actual_rank = {}
    for idx, name in enumerate(actual_names_norm, start=1):
        if name and name not in actual_rank:
            actual_rank[name] = idx

    work = pred_df[[horse_col, score_col]].copy()
    work["horse_key"] = work[horse_col].apply(normalize_name)
    work = work[work["horse_key"] != ""].copy()
    work["Top3Prob_model"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work[work["Top3Prob_model"].notna()].copy()
    work["rank"] = work["horse_key"].map(actual_rank).fillna(99).astype(int)

    pred_eval = work[["horse_key", "Top3Prob_model"]].copy()
    pred_eval["race_id"] = race_id
    result_eval = work[["horse_key", "rank"]].copy()
    result_eval["race_id"] = race_id
    return pred_eval, result_eval


def get_eval_window_races():
    raw = (
        os.environ.get("PRED_EVAL_WINDOW_RACES", "").strip()
        or os.environ.get("PRED_OPT_WINDOW", "").strip()
        or "50"
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 50
    return max(1, value)


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


def infer_run_id_from_row(run_row):
    if not run_row:
        return ""
    for key in (
        "run_id",
        "policy_path",
        "predictions_path",
        "odds_path",
        "wide_odds_path",
        "fuku_odds_path",
        "quinella_odds_path",
        "exacta_odds_path",
        "trio_odds_path",
        "trifecta_odds_path",
        "plan_path",
        "gemini_policy_path",
        "siliconflow_policy_path",
        "openai_policy_path",
        "grok_policy_path",
    ):
        text = str(run_row.get(key, "") or "").strip()
        if key == "run_id" and text:
            return text
        match = re.search(r"(\d{8}_\d{6})", text)
        if match:
            return match.group(1)
    return ""


def normalize_race_id(value):
    raw = str(value or "").strip()
    if not raw:
        return ""
    match = re.search(r"race_id=(\d+)", raw)
    if match:
        return match.group(1)
    return re.sub(r"\D", "", raw)


def main():
    init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No runs found. Run pipeline/run_pipeline.py first.")
        sys.exit(1)

    resolved_runs = []
    for row in runs:
        resolved_run_id = infer_run_id_from_row(row)
        if not resolved_run_id:
            continue
        enriched = dict(row)
        enriched["_resolved_run_id"] = resolved_run_id
        resolved_runs.append(enriched)
    if not resolved_runs:
        print("No runnable entries found in runs.csv. Missing run_id and artifact-derived fallback.")
        sys.exit(1)

    last_run = resolved_runs[-1]["_resolved_run_id"]
    run_key = input(f"Run ID / Race ID [{last_run}]: ").strip() or last_run
    run = next((r for r in resolved_runs if r.get("_resolved_run_id") == run_key), None)
    if run is None:
        race_id_key = normalize_race_id(run_key)
        if race_id_key:
            for row in reversed(resolved_runs):
                if normalize_race_id(row.get("race_id", "")) == race_id_key:
                    run = row
                    break
    if not run:
        print("Run ID / Race ID not found.")
        sys.exit(1)
    run_id = str(run.get("_resolved_run_id", "")).strip()

    race_id = str(run.get("race_id") or "").strip()
    actual_names_raw = prompt_actual_top3()
    actual_names = [normalize_name(n) for n in actual_names_raw]
    predictor_rows = []
    for spec in list_predictors():
        pred_path_run = resolve_run_prediction_path(
            run,
            DATA_DIR,
            ROOT_DIR,
            run_id=run_id,
            race_id=race_id,
            predictor_id=spec["id"],
            last_run_id=last_run,
        )
        pred_path = str(pred_path_run) if pred_path_run else ""
        pred_file = Path(pred_path) if pred_path else None
        if not pred_file or not pred_file.exists():
            print(f"[WARN] {spec['label']} predictions not found: {pred_path_run}")
            continue

        df = pd.read_csv(pred_file, encoding="utf-8-sig")
        score_key = pick_score_column(df.columns)
        if "HorseName" not in df.columns or not score_key:
            print(f"[WARN] {spec['label']} missing columns: HorseName / score")
            continue
        score_is_probability = infer_score_is_probability(df, score_key)
        pred_name_set = {
            normalize_name(n)
            for n in df["HorseName"].dropna().astype(str)
            if normalize_name(n)
        }
        missing = [
            raw
            for raw, norm in zip(actual_names_raw, actual_names)
            if norm and norm not in pred_name_set
        ]
        if missing:
            print(f"[ERROR] {spec['label']} missing actual horses in predictions: {', '.join(missing)}")
            sys.exit(1)

        pred_top = df.sort_values(score_key, ascending=False).head(3)
        pred_names_raw = pred_top["HorseName"].tolist()
        pred_names = [normalize_name(n) for n in pred_names_raw]
        pred_top5 = df.sort_values(score_key, ascending=False).head(5)
        pred_top5_names_raw = pred_top5["HorseName"].tolist()
        pred_top5_names = [normalize_name(n) for n in pred_top5_names_raw]

        print(f"\n{spec['label']} Top3:")
        for i, name in enumerate(pred_names_raw, 1):
            print(f"{i}. {name}")

        hit_count = len(set(pred_names) & set(actual_names))
        top5_hit_count = len(set(pred_top5_names) & set(actual_names)) if pred_top5_names else 0
        top1_hit = 1 if pred_names and actual_names and pred_names[0] == actual_names[0] else 0
        top1_in_top3 = 1 if pred_names and pred_names[0] in actual_names else 0
        top3_exact = 1 if set(pred_names) == set(actual_names) else 0
        score = hit_count + top1_hit
        eval_window_races = get_eval_window_races()
        pred_eval_df, result_eval_df = build_eval_frames(
            run_id=run_id,
            pred_df=df,
            horse_col="HorseName",
            score_col=score_key,
            actual_names_norm=actual_names,
        )
        hit_at_5 = compute_hit_at_k(pred_eval_df, result_eval_df, k=5)
        top3_hits_at_5 = compute_top3_hits_at_k(pred_eval_df, result_eval_df, k=5)
        mrr_top3 = compute_mrr_top3(pred_eval_df, result_eval_df, k=10)
        brier = compute_brier_score(pred_eval_df, result_eval_df) if score_is_probability else None
        sample_races = int(pred_eval_df["race_id"].nunique()) if not pred_eval_df.empty else 0
        predictor_rows.append(
            {
                "run_id": run_id,
                "predictor_id": spec["id"],
                "predictor_label": spec["label"],
                "strategy": run.get("predictor_strategy", "") if spec["id"] == "main" else spec["id"],
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
                "rank_score": "",
                "ev_score": "",
                "hit_rate": "",
                "score_total": "",
                "sample_races": sample_races,
                "eval_window_races": eval_window_races,
                "hit_at_5": round(hit_at_5, 4),
                "top3_hits_at_5": round(top3_hits_at_5, 4),
                "mrr_top3": round(mrr_top3, 4),
                "brier": round(brier, 6) if (brier is not None and pd.notna(brier)) else "",
                "confidence_score": "",
                "stability_score": "",
                "validity_score": "",
                "consistency_score": "",
                "rank_ema": "",
                "ev_ema": "",
                "risk_score": "",
            }
        )
    if not predictor_rows:
        print("No predictor outputs found for this run.")
        sys.exit(1)
    replace_rows_for_run(PRED_RESULTS_PATH, list(predictor_rows[0].keys()), run_id, predictor_rows)
    print(f"Recorded predictor results for {run_id}: {len(predictor_rows)} predictors")
    for policy_engine in ("gemini", "siliconflow", "openai", "grok"):
        settlement = settle_run_tickets(BASE_DIR, run, actual_names_raw, policy_engine=policy_engine)
        if settlement:
            print(
                "[{policy_engine}_portfolio] settled_tickets={settled_ticket_count} run_stake_yen={run_stake_yen} "
                "run_payout_yen={run_payout_yen} run_profit_yen={run_profit_yen} "
                "available_bankroll_yen={available_bankroll_yen}".format(**settlement)
            )
    optimizer = BASE_DIR / "optimize_predictor_params.py"
    if optimizer.exists():
        subprocess.run([sys.executable, str(optimizer)], check=False)


if __name__ == "__main__":
    main()

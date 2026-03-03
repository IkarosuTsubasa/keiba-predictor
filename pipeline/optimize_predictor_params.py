import csv
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from surface_scope import (
    get_data_dir,
    get_predictor_config_path,
    get_predictor_prev_path,
    get_scope_key,
    migrate_legacy_data,
)


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = None
CONFIG_PATH = None
PREV_CONFIG_PATH = None
RESULTS_PATH = None
HISTORY_PATH = None
RUNS_PATH = None
SCOPE_KEY = ""


def pause_exit(message="Press Enter to exit..."):
    if sys.stdin and sys.stdin.isatty():
        try:
            input(message)
        except EOFError:
            pass


def init_scope():
    scope_key = get_scope_key()
    migrate_legacy_data(BASE_DIR, scope_key)
    os.environ["SCOPE_KEY"] = scope_key
    global DATA_DIR, CONFIG_PATH, PREV_CONFIG_PATH, RESULTS_PATH, HISTORY_PATH, RUNS_PATH, SCOPE_KEY
    DATA_DIR = get_data_dir(BASE_DIR, scope_key)
    CONFIG_PATH = get_predictor_config_path(BASE_DIR, scope_key)
    PREV_CONFIG_PATH = get_predictor_prev_path(BASE_DIR, scope_key)
    RESULTS_PATH = DATA_DIR / "predictor_results.csv"
    HISTORY_PATH = DATA_DIR / "predictor_config_history.csv"
    RUNS_PATH = DATA_DIR / "runs.csv"
    SCOPE_KEY = scope_key


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_name(value):
    return "".join(str(value or "").split())


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


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fallback = []
    if SCOPE_KEY == "central_turf":
        fallback.extend(
            [
                BASE_DIR / "predictor_config_turf_default.json",
                BASE_DIR / "predictor_config_turf.json",
            ]
        )
    elif SCOPE_KEY == "central_dirt":
        fallback.extend(
            [
                BASE_DIR / "predictor_config_dirt_default.json",
                BASE_DIR / "predictor_config_dirt.json",
            ]
        )
    elif SCOPE_KEY == "local":
        fallback.extend(
            [
                BASE_DIR / "predictor_config_central_dirt.json",
                BASE_DIR / "predictor_config_dirt_default.json",
                BASE_DIR / "predictor_config_dirt.json",
            ]
        )
    fallback.append(BASE_DIR / "predictor_config.json")
    for legacy in fallback:
        if legacy.exists():
            data = json.loads(legacy.read_text(encoding="utf-8"))
            CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data
    return {"version": 1, "params": {}, "state": {}}


def save_config(data):
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_grid(step):
    step = clamp(safe_float(step, 0.05), 0.01, 1.0)
    values = []
    value = 0.0
    while value <= 1.0 + 1e-12:
        values.append(round(clamp(value, 0.0, 1.0), 4))
        value += step
    if values[-1] != 1.0:
        values.append(1.0)
    return sorted(set(values))


def select_eval_window(rows, eval_window_races):
    if eval_window_races <= 0:
        return list(rows)
    return list(rows[-eval_window_races:])


def pick_first(columns, candidates):
    col_set = set(columns)
    for key in candidates:
        if key in col_set:
            return key
    return ""


def build_run_map():
    rows = load_csv(RUNS_PATH)
    out = {}
    for row in rows:
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            path = str(row.get("predictions_path") or "")
            m = re.search(r"(\d{8}_\d{6})", path)
            run_id = m.group(1) if m else ""
        if run_id:
            out[run_id] = row
    return out


def resolve_pred_path(row, run_map):
    run_id = str(row.get("run_id") or "").strip()
    race_id = str(row.get("race_id") or "").strip()
    run_row = run_map.get(run_id, {})
    if not race_id:
        race_id = str(run_row.get("race_id") or "").strip()
    candidates = []
    row_path = str(row.get("predictions_path") or "").strip()
    if row_path:
        candidates.append(Path(row_path))
    run_path = str(run_row.get("predictions_path") or "").strip()
    if run_path:
        candidates.append(Path(run_path))
    if run_id and race_id:
        candidates.append(DATA_DIR / race_id / f"predictions_{run_id}_{race_id}.csv")
    if run_id:
        candidates.append(DATA_DIR / f"predictions_{run_id}.csv")
    if run_id:
        globbed = list(DATA_DIR.rglob(f"predictions_{run_id}*.csv"))
        candidates.extend(globbed[:3])
    for path in candidates:
        if path and path.exists():
            return path
    return None


def build_eval_dataset(rows, run_map):
    horse_candidates = (
        "horse_key",
        "horse_id",
        "horse_no",
        "horse_name",
        "HorseName",
        "name",
        "\u99ac\u756a",  # 馬番
        "\u99ac\u540d",  # 馬名
    )
    lr_candidates = (
        "Top3Prob_raw_lr",
        "Top3Prob_lr",
        "Top3Prob_model",
        "Top3Prob_est",
        "Top3Prob",
        "agg_score",
        "score",
    )
    lgb_candidates = ("Top3Prob_raw_lgb", "Top3Prob_lgbm")
    dataset = []

    for row in rows:
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        actual_names = [
            normalize_name(row.get("actual_top1")),
            normalize_name(row.get("actual_top2")),
            normalize_name(row.get("actual_top3")),
        ]
        if len([n for n in actual_names if n]) < 3:
            continue

        pred_path = resolve_pred_path(row, run_map)
        if not pred_path:
            continue
        try:
            df = pd.read_csv(pred_path, encoding="utf-8-sig")
        except Exception:
            continue
        if df.empty:
            continue

        horse_col = pick_first(df.columns, horse_candidates)
        lr_col = pick_first(df.columns, lr_candidates)
        lgb_col = pick_first(df.columns, lgb_candidates)
        if not horse_col or not lr_col:
            continue

        work = pd.DataFrame()
        work["horse_key"] = df[horse_col].apply(normalize_name)
        work["p_lr"] = pd.to_numeric(df[lr_col], errors="coerce")
        if lgb_col:
            work["p_lgb_raw"] = pd.to_numeric(df[lgb_col], errors="coerce")
        else:
            work["p_lgb_raw"] = np.nan
        work = work[(work["horse_key"] != "") & work["p_lr"].notna()].copy()
        if work.empty:
            continue

        actual_rank = {}
        for idx, name in enumerate(actual_names, start=1):
            if name and name not in actual_rank:
                actual_rank[name] = idx
        work["rank"] = work["horse_key"].map(actual_rank).fillna(99).astype(int)
        has_lgb = bool(work["p_lgb_raw"].notna().any())
        p_lgb_raw = work["p_lgb_raw"].to_numpy(dtype=float) if has_lgb else None
        dataset.append(
            {
                "run_id": run_id,
                "p_lr": work["p_lr"].to_numpy(dtype=float),
                "p_lgb_raw": p_lgb_raw,
                "rank": work["rank"].to_numpy(dtype=int),
            }
        )

    return dataset


def evaluate_params(dataset, alpha, beta):
    alpha = clamp(float(alpha), 0.0, 1.0)
    beta = clamp(float(beta), 0.0, 1.0)
    if not dataset:
        return {
            "sample_races": 0,
            "hit_at_5": 0.0,
            "top3_hits_at_5": 0.0,
            "mrr_top3": 0.0,
            "brier": None,
            "score": 0.0,
        }

    hit_values = []
    top3_hit_values = []
    mrr_values = []
    brier_values = []

    for item in dataset:
        p_lr = item["p_lr"]
        p_lgb_raw = item["p_lgb_raw"]
        rank = item["rank"]
        if p_lgb_raw is not None:
            p_lgb_squashed = 0.5 + (p_lgb_raw - 0.5) * beta
            p_model = alpha * p_lgb_squashed + (1.0 - alpha) * p_lr
        else:
            p_model = p_lr

        order = np.argsort(-p_model, kind="mergesort")
        top5_rank = rank[order[:5]]
        hit_values.append(1.0 if np.any(top5_rank <= 3) else 0.0)
        top3_hit_values.append(float(np.sum(top5_rank <= 3)))

        rr = 0.0
        for idx, horse_idx in enumerate(order[:10], start=1):
            if rank[horse_idx] <= 3:
                rr = 1.0 / float(idx)
                break
        mrr_values.append(rr)

        y = (rank <= 3).astype(float)
        brier_values.append(float(np.mean((p_model - y) ** 2)))

    sample = len(hit_values)
    hit_at_5 = float(sum(hit_values) / sample) if sample else 0.0
    top3_hits_at_5 = float(sum(top3_hit_values) / sample) if sample else 0.0
    mrr_top3 = float(sum(mrr_values) / sample) if sample else 0.0
    brier = float(sum(brier_values) / sample) if brier_values else None
    score = 1000.0 * hit_at_5 + 100.0 * top3_hits_at_5
    if brier is not None:
        score -= 10.0 * brier
    return {
        "sample_races": sample,
        "hit_at_5": hit_at_5,
        "top3_hits_at_5": top3_hits_at_5,
        "mrr_top3": mrr_top3,
        "brier": brier,
        "score": score,
    }


def is_better(candidate, best):
    if best is None:
        return True
    if not math.isclose(candidate["score"], best["score"], rel_tol=0.0, abs_tol=1e-12):
        return candidate["score"] > best["score"]
    if not math.isclose(candidate["hit_at_5"], best["hit_at_5"], rel_tol=0.0, abs_tol=1e-12):
        return candidate["hit_at_5"] > best["hit_at_5"]
    if not math.isclose(
        candidate["top3_hits_at_5"],
        best["top3_hits_at_5"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return candidate["top3_hits_at_5"] > best["top3_hits_at_5"]
    c_brier = candidate.get("brier")
    b_brier = best.get("brier")
    if c_brier is None and b_brier is None:
        return False
    if c_brier is None:
        return False
    if b_brier is None:
        return True
    if not math.isclose(c_brier, b_brier, rel_tol=0.0, abs_tol=1e-12):
        return c_brier < b_brier
    return False


def format_metric(value, digits=4):
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def main():
    init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = load_csv(RESULTS_PATH)
    if not results:
        print("No predictor results found.")
        pause_exit()
        return

    eval_window_races = int(
        safe_float(
            os.environ.get("PRED_EVAL_WINDOW_RACES", os.environ.get("PRED_OPT_WINDOW", 50)),
            50,
        )
    )
    eval_window_races = max(1, eval_window_races)
    min_samples = int(safe_float(os.environ.get("PRED_OPT_MIN_SAMPLES", 30), 30))
    min_samples = max(1, min_samples)
    min_improve = safe_float(os.environ.get("PRED_OPT_MIN_IMPROVE", 0.005), 0.005)
    alpha_step = safe_float(os.environ.get("PRED_OPT_BLEND_STEP", 0.05), 0.05)
    beta_step = safe_float(os.environ.get("PRED_OPT_LGB_SQUASH_STEP", 0.05), 0.05)

    recent_rows = select_eval_window(results, eval_window_races)
    run_map = build_run_map()
    dataset = build_eval_dataset(recent_rows, run_map)
    sample_races = len(dataset)

    print(
        f"Evaluation window: last {eval_window_races} runs "
        f"(usable samples={sample_races}, min_required={min_samples})"
    )
    if sample_races < min_samples:
        print(f"Not enough samples for update ({sample_races}/{min_samples}).")
        pause_exit()
        return

    config = load_config()
    params = dict(config.get("params", {}))
    baseline_alpha = clamp(safe_float(params.get("blend_alpha", 0.3), 0.3), 0.0, 1.0)
    baseline_beta = clamp(safe_float(params.get("lgb_squash_beta", 0.5), 0.5), 0.0, 1.0)
    baseline = evaluate_params(dataset, baseline_alpha, baseline_beta)
    baseline["blend_alpha"] = baseline_alpha
    baseline["lgb_squash_beta"] = baseline_beta

    alpha_grid = build_grid(alpha_step)
    beta_grid = build_grid(beta_step)
    best = None
    for alpha in alpha_grid:
        for beta in beta_grid:
            candidate = evaluate_params(dataset, alpha, beta)
            candidate["blend_alpha"] = alpha
            candidate["lgb_squash_beta"] = beta
            if is_better(candidate, best):
                best = candidate

    hit_gain = float(best["hit_at_5"] - baseline["hit_at_5"])
    top3_gain = float(best["top3_hits_at_5"] - baseline["top3_hits_at_5"])
    brier_gain = None
    if baseline["brier"] is not None and best["brier"] is not None:
        brier_gain = float(baseline["brier"] - best["brier"])

    print(
        "Baseline: "
        f"hit_at_5={format_metric(baseline['hit_at_5'])}, "
        f"top3_hits_at_5={format_metric(baseline['top3_hits_at_5'])}, "
        f"brier={format_metric(baseline['brier'], 6)}, "
        f"score={format_metric(baseline['score'])}, "
        f"alpha={baseline_alpha:.2f}, beta={baseline_beta:.2f}"
    )
    print(
        "Best grid: "
        f"hit_at_5={format_metric(best['hit_at_5'])}, "
        f"top3_hits_at_5={format_metric(best['top3_hits_at_5'])}, "
        f"brier={format_metric(best['brier'], 6)}, "
        f"score={format_metric(best['score'])}, "
        f"alpha={best['blend_alpha']:.2f}, beta={best['lgb_squash_beta']:.2f}"
    )

    same_params = (
        math.isclose(best["blend_alpha"], baseline_alpha, abs_tol=1e-12)
        and math.isclose(best["lgb_squash_beta"], baseline_beta, abs_tol=1e-12)
    )
    should_update = (hit_gain >= min_improve) and (not same_params)
    if should_update:
        prev_snapshot = json.loads(json.dumps(config))
        PREV_CONFIG_PATH.write_text(json.dumps(prev_snapshot, indent=2), encoding="utf-8")
        params["blend_alpha"] = round(float(best["blend_alpha"]), 4)
        params["lgb_squash_beta"] = round(float(best["lgb_squash_beta"]), 4)
        config["params"] = params
        config["version"] = int(config.get("version", 1)) + 1
        state = dict(config.get("state", {}))
        state["predictor_opt_updated_at"] = datetime.now().isoformat(timespec="seconds")
        state["predictor_opt_window"] = eval_window_races
        state["predictor_opt_samples"] = sample_races
        config["state"] = state
        save_config(config)
        action = "update"
        reason = f"hit_gain={hit_gain:.4f} >= min_improve={min_improve:.4f}"
        print("Updated predictor params.")
    else:
        action = "no_change"
        if same_params:
            reason = "best params same as baseline"
        else:
            reason = (
                f"hit_gain={hit_gain:.4f} < min_improve={min_improve:.4f} "
                "(tie-breaker only for ranking)"
            )
        print("No parameter changes.")

    if math.isclose(best["hit_at_5"], baseline["hit_at_5"], abs_tol=1e-12):
        if best["top3_hits_at_5"] > baseline["top3_hits_at_5"] + 1e-12:
            print("Tie-break note: hit_at_5 tied, best has higher top3_hits_at_5.")
        elif (
            math.isclose(best["top3_hits_at_5"], baseline["top3_hits_at_5"], abs_tol=1e-12)
            and baseline["brier"] is not None
            and best["brier"] is not None
            and best["brier"] < baseline["brier"] - 1e-12
        ):
            print("Tie-break note: hit_at_5/top3_hits_at_5 tied, best has lower brier.")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "reason": reason,
        "window": eval_window_races,
        "sample_races": sample_races,
        "min_samples": min_samples,
        "min_improve": round(min_improve, 6),
        "alpha_step": round(alpha_step, 4),
        "beta_step": round(beta_step, 4),
        "baseline_blend_alpha": round(baseline_alpha, 4),
        "baseline_lgb_squash_beta": round(baseline_beta, 4),
        "baseline_hit_at_5": round(baseline["hit_at_5"], 6),
        "baseline_top3_hits_at_5": round(baseline["top3_hits_at_5"], 6),
        "baseline_mrr_top3": round(baseline["mrr_top3"], 6),
        "baseline_brier": round(baseline["brier"], 8) if baseline["brier"] is not None else "",
        "baseline_score": round(baseline["score"], 6),
        "best_blend_alpha": round(best["blend_alpha"], 4),
        "best_lgb_squash_beta": round(best["lgb_squash_beta"], 4),
        "best_hit_at_5": round(best["hit_at_5"], 6),
        "best_top3_hits_at_5": round(best["top3_hits_at_5"], 6),
        "best_mrr_top3": round(best["mrr_top3"], 6),
        "best_brier": round(best["brier"], 8) if best["brier"] is not None else "",
        "best_score": round(best["score"], 6),
        "hit_gain": round(hit_gain, 6),
        "top3_hits_gain": round(top3_gain, 6),
        "brier_gain": round(brier_gain, 8) if brier_gain is not None else "",
    }
    append_csv(HISTORY_PATH, list(row.keys()), row)

    print(f"Reason: {reason}")
    print(
        "Delta: "
        f"hit_at_5={hit_gain:+.4f}, "
        f"top3_hits_at_5={top3_gain:+.4f}, "
        f"brier={(brier_gain if brier_gain is not None else 0.0):+.6f}"
    )
    pause_exit()


if __name__ == "__main__":
    main()


import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupShuffleSplit

from surface_scope import (
    get_data_dir,
    get_predictor_config_path,
    normalize_scope_key,
)


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_SCOPE = "central_dirt"

FEATURES = [
    "ti_last",
    "ti_mean3",
    "ti_mean5",
    "ti_max5",
    "ti_std5",
    "ti_decay",
    "sm_mean5",
    "dd_mean5",
    "up_min5",
    "run_first_z_mean5",
    "run_last_z_mean5",
    "run_gain_z_mean5",
    "up_z_mean5",
    "pace_diff_z_mean5",
    "trend_mean3",
    "history_count",
    "Age",
    "SexMale",
    "SexFemale",
    "SexGelding",
    "jscore_last",
    "jscore_mean5",
    "jscore_max5",
    "jscore_current",
    "ps_avg_all",
    "ps_recent_mean",
    "ps_max_recent",
    "exp_place_recent",
    "ps_decay",
    "cup_run_first_gap",
    "cup_run_last_gap",
    "cup_run_gain_gap",
    "cup_uphill_gap",
    "cup_pace_gap",
    "cup_samples",
    "TargetDistance",
]


def _normalize_name(value) -> str:
    return "".join(str(value or "").split())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def normalize_rank_score_by_race(scores: np.ndarray, race_ids: Iterable[str]) -> np.ndarray:
    series = pd.Series(scores, dtype=float)
    keys = pd.Series(list(race_ids), index=series.index).fillna("").astype(str)
    keys = keys.replace("", "__single_race__")
    out = np.zeros(len(series), dtype=float)
    for _, idx in keys.groupby(keys).groups.items():
        pos = list(idx)
        arr = series.iloc[pos].to_numpy(dtype=float)
        std = float(np.nanstd(arr))
        if (not np.isfinite(std)) or std < 1e-12:
            z = np.zeros_like(arr, dtype=float)
        else:
            mean = float(np.nanmean(arr))
            z = (arr - mean) / std
        out[pos] = _sigmoid(z)
    return out


def _parse_alpha_grid(text: str) -> List[float]:
    raw = str(text or "").strip()
    if not raw:
        raw = "0.3:0.8:0.05"
    if ":" in raw and "," not in raw:
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) == 3:
            try:
                start, end, step = (float(parts[0]), float(parts[1]), float(parts[2]))
                if step <= 0:
                    raise ValueError
                vals = []
                cur = start
                while cur <= end + 1e-12:
                    vals.append(round(float(cur), 6))
                    cur += step
                return sorted(set(vals))
            except (TypeError, ValueError):
                pass
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(round(float(token), 6))
        except ValueError:
            continue
    return sorted(set(values)) if values else [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def _resolve_seeds(seed_text: str, split_count: int) -> List[int]:
    parsed = []
    for token in str(seed_text or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            continue
    if not parsed:
        parsed = [11, 22, 33, 44, 55]
    if split_count <= 0:
        split_count = 1
    out = parsed[:split_count]
    while len(out) < split_count:
        out.append(out[-1] + 11)
    return out


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _load_ranker_params(scope_key: str) -> Dict[str, float]:
    cfg_path = get_predictor_config_path(BASE_DIR, scope_key)
    params = {}
    if cfg_path.exists():
        try:
            import json

            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            params = dict(cfg.get("params", {}))
        except Exception:
            params = {}
    return {
        "n_estimators": max(50, _safe_int(params.get("ranker_n_estimators", 300), 300)),
        "learning_rate": min(0.5, max(0.001, _safe_float(params.get("ranker_learning_rate", 0.05), 0.05))),
        "num_leaves": max(8, _safe_int(params.get("ranker_num_leaves", 31), 31)),
        "min_child_samples": max(1, _safe_int(params.get("ranker_min_child_samples", 20), 20)),
        "subsample": min(1.0, max(0.3, _safe_float(params.get("ranker_subsample", 0.8), 0.8))),
        "colsample_bytree": min(1.0, max(0.3, _safe_float(params.get("ranker_colsample_bytree", 0.8), 0.8))),
    }


def _build_eval_dataset(scope_key: str) -> Tuple[pd.DataFrame, int]:
    data_dir = get_data_dir(BASE_DIR, scope_key)
    pred_results_path = data_dir / "predictor_results.csv"
    if not pred_results_path.exists():
        raise FileNotFoundError(f"Missing predictor_results.csv: {pred_results_path}")

    pred_results = pd.read_csv(pred_results_path, encoding="utf-8")
    if pred_results.empty:
        raise ValueError(f"predictor_results.csv empty: {pred_results_path}")
    if "run_id" not in pred_results.columns:
        raise ValueError(f"predictor_results.csv missing run_id: {pred_results_path}")

    pred_results = pred_results.drop_duplicates(subset=["run_id"], keep="last")

    rows = []
    missing_pred_files = 0
    for _, row in pred_results.iterrows():
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        pred_path = str(row.get("predictions_path") or "").strip()
        pred_file = Path(pred_path) if pred_path else None
        if (pred_file is None) or (not pred_file.exists()):
            candidates = list(data_dir.rglob(f"predictions_{run_id}*.csv"))
            pred_file = candidates[0] if candidates else None
        if (pred_file is None) or (not pred_file.exists()):
            missing_pred_files += 1
            continue

        try:
            pred_df = pd.read_csv(pred_file, encoding="utf-8-sig")
        except Exception:
            missing_pred_files += 1
            continue

        if pred_df.empty or "HorseName" not in pred_df.columns or "Top3Prob_model" not in pred_df.columns:
            continue

        actual = [
            _normalize_name(row.get("actual_top1")),
            _normalize_name(row.get("actual_top2")),
            _normalize_name(row.get("actual_top3")),
        ]
        actual_rank = {name: idx + 1 for idx, name in enumerate(actual) if name}
        if len(actual_rank) < 3:
            continue

        work = pd.DataFrame(index=pred_df.index)
        work["horse_key"] = pred_df["HorseName"].map(_normalize_name)
        work["base_score"] = pd.to_numeric(pred_df["Top3Prob_model"], errors="coerce")
        for col in FEATURES:
            work[col] = pd.to_numeric(pred_df[col], errors="coerce") if col in pred_df.columns else np.nan

        work = work[(work["horse_key"] != "") & work["base_score"].notna()].copy()
        if len(work) < 5:
            continue

        work["race_id"] = run_id
        work["rank"] = work["horse_key"].map(actual_rank).fillna(99).astype(int)
        # 仅有 Top3 真值时的 relevance: 1st=5, 2nd=4, 3rd=3, else=0
        work["y_rel"] = np.where(work["rank"] <= 3, 6 - work["rank"], 0).astype(float)
        rows.append(work)

    if not rows:
        raise ValueError(f"No usable races from predictor_results: {pred_results_path}")

    all_df = pd.concat(rows, ignore_index=True)
    for col in FEATURES:
        if all_df[col].isna().all():
            all_df[col] = 0.0
    all_df[FEATURES] = all_df[FEATURES].fillna(all_df[FEATURES].median(numeric_only=True))
    return all_df, missing_pred_files


def _calc_metrics_by_score(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    hit_at_5 = []
    top3_hits_at_5 = []
    mrr_top3 = []
    for _, group in df.groupby("race_id"):
        ordered = group.sort_values(score_col, ascending=False)
        top5 = ordered.head(5)
        ranks = top5["rank"].to_numpy(dtype=int)
        hit_at_5.append(1.0 if np.any(ranks <= 3) else 0.0)
        top3_hits_at_5.append(float(np.sum(ranks <= 3)))

        rr = 0.0
        for idx, rank_val in enumerate(ordered["rank"].to_numpy(dtype=int)[:10], start=1):
            if rank_val <= 3:
                rr = 1.0 / float(idx)
                break
        mrr_top3.append(rr)

    return {
        "hit_at_5": float(np.mean(hit_at_5)) if hit_at_5 else 0.0,
        "top3_hits_at_5": float(np.mean(top3_hits_at_5)) if top3_hits_at_5 else 0.0,
        "mrr_top3": float(np.mean(mrr_top3)) if mrr_top3 else 0.0,
    }


def _aggregate(items: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for key in ("hit_at_5", "top3_hits_at_5", "mrr_top3"):
        vals = np.array([x[key] for x in items], dtype=float)
        out[f"{key}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        out[f"{key}_std"] = float(np.std(vals, ddof=0)) if len(vals) else float("nan")
    return out


def evaluate_scope(
    scope_key: str,
    split_count: int = 5,
    seed_text: str = "11,22,33,44,55",
    alpha_grid_text: str = "0.3:0.8:0.05",
    test_size: float = 0.2,
) -> Dict[str, object]:
    scope_key = normalize_scope_key(scope_key) or DEFAULT_SCOPE
    split_count = max(1, int(split_count))
    test_size = min(0.9, max(0.05, float(test_size)))
    seeds = _resolve_seeds(seed_text, split_count)
    alpha_grid = _parse_alpha_grid(alpha_grid_text)
    ranker_params = _load_ranker_params(scope_key)
    all_df, missing_pred_files = _build_eval_dataset(scope_key)

    race_count = int(all_df["race_id"].nunique())
    if race_count < 3:
        raise ValueError(f"Not enough races for grouped split: {race_count}")

    baseline_splits = []
    ranker_splits = []
    hybrid_splits = {alpha: [] for alpha in alpha_grid}

    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(all_df, groups=all_df["race_id"]))
        train_df = all_df.iloc[train_idx].copy().sort_values("race_id")
        test_df = all_df.iloc[test_idx].copy().sort_values("race_id")

        group_train = train_df.groupby("race_id").size().tolist()
        group_test = test_df.groupby("race_id").size().tolist()
        if not group_train or not group_test:
            continue

        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=ranker_params["n_estimators"],
            learning_rate=ranker_params["learning_rate"],
            num_leaves=ranker_params["num_leaves"],
            min_child_samples=ranker_params["min_child_samples"],
            subsample=ranker_params["subsample"],
            colsample_bytree=ranker_params["colsample_bytree"],
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        ranker.fit(
            train_df[FEATURES],
            train_df["y_rel"],
            group=group_train,
            eval_set=[(test_df[FEATURES], test_df["y_rel"])],
            eval_group=[group_test],
            eval_at=[1, 3, 5],
        )

        test_eval = test_df.copy()
        test_eval["rank_score_raw"] = ranker.predict(test_eval[FEATURES])
        test_eval["rank_score_norm"] = normalize_rank_score_by_race(
            test_eval["rank_score_raw"].to_numpy(dtype=float),
            test_eval["race_id"].to_numpy(),
        )

        baseline_splits.append(_calc_metrics_by_score(test_eval, "base_score"))
        ranker_splits.append(_calc_metrics_by_score(test_eval, "rank_score_norm"))

        for alpha in alpha_grid:
            test_eval["hybrid_score"] = (
                float(alpha) * test_eval["rank_score_norm"] + (1.0 - float(alpha)) * test_eval["base_score"]
            )
            hybrid_splits[alpha].append(_calc_metrics_by_score(test_eval, "hybrid_score"))

    if not baseline_splits:
        raise ValueError("No valid grouped splits for evaluation.")

    return {
        "scope": scope_key,
        "rows": int(len(all_df)),
        "races": int(all_df["race_id"].nunique()),
        "split_count": int(len(baseline_splits)),
        "seeds": seeds,
        "alpha_grid": alpha_grid,
        "missing_pred_files": int(missing_pred_files),
        "baseline": _aggregate(baseline_splits),
        "ranker": _aggregate(ranker_splits),
        "hybrid": {alpha: _aggregate(items) for alpha, items in hybrid_splits.items()},
    }


def save_eval_csv(result: Dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    common = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scope": result["scope"],
        "rows": result["rows"],
        "races": result["races"],
        "split_count": result["split_count"],
        "seeds": ",".join(str(x) for x in result["seeds"]),
        "missing_pred_files": result["missing_pred_files"],
    }
    for model_name in ("baseline", "ranker"):
        stats = result[model_name]
        rows.append(
            {
                **common,
                "model": model_name,
                "alpha": "",
                "hit_at_5_mean": stats["hit_at_5_mean"],
                "hit_at_5_std": stats["hit_at_5_std"],
                "top3_hits_at_5_mean": stats["top3_hits_at_5_mean"],
                "top3_hits_at_5_std": stats["top3_hits_at_5_std"],
                "mrr_top3_mean": stats["mrr_top3_mean"],
                "mrr_top3_std": stats["mrr_top3_std"],
            }
        )
    for alpha in result["alpha_grid"]:
        stats = result["hybrid"][alpha]
        rows.append(
            {
                **common,
                "model": "hybrid",
                "alpha": alpha,
                "hit_at_5_mean": stats["hit_at_5_mean"],
                "hit_at_5_std": stats["hit_at_5_std"],
                "top3_hits_at_5_mean": stats["top3_hits_at_5_mean"],
                "top3_hits_at_5_std": stats["top3_hits_at_5_std"],
                "mrr_top3_mean": stats["mrr_top3_mean"],
                "mrr_top3_std": stats["mrr_top3_std"],
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def _fmt_stats(stats: Dict[str, float]) -> str:
    return (
        f"hit@5={stats['hit_at_5_mean']:.4f}+/-{stats['hit_at_5_std']:.4f}, "
        f"top3@5={stats['top3_hits_at_5_mean']:.4f}+/-{stats['top3_hits_at_5_std']:.4f}, "
        f"mrr@10={stats['mrr_top3_mean']:.4f}+/-{stats['mrr_top3_std']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline/ranker/hybrid ranking modes.")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, help="Scope key: central_dirt / central_turf / local")
    parser.add_argument("--split-count", type=int, default=5, help="Number of grouped splits")
    parser.add_argument("--seeds", default="11,22,33,44,55", help="Comma-separated random seeds")
    parser.add_argument("--alpha-grid", default="0.3:0.8:0.05", help="Grid like 0.3:0.8:0.05 or csv")
    parser.add_argument("--test-size", type=float, default=0.2, help="Grouped holdout test size")
    args = parser.parse_args()

    scope_key = normalize_scope_key(args.scope) or DEFAULT_SCOPE
    result = evaluate_scope(
        scope_key=scope_key,
        split_count=args.split_count,
        seed_text=args.seeds,
        alpha_grid_text=args.alpha_grid,
        test_size=args.test_size,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = get_data_dir(BASE_DIR, scope_key) / f"eval_ranker_modes_{ts}.csv"
    save_eval_csv(result, out_path)

    print(f"[scope={scope_key}] rows={result['rows']} races={result['races']} splits={result['split_count']}")
    print("baseline:", _fmt_stats(result["baseline"]))
    print("ranker  :", _fmt_stats(result["ranker"]))
    best_alpha = None
    best_score = None
    for alpha in result["alpha_grid"]:
        s = result["hybrid"][alpha]
        score = s["hit_at_5_mean"] - 0.5 * s["hit_at_5_std"]
        if (best_score is None) or (score > best_score):
            best_alpha = alpha
            best_score = score
    if best_alpha is not None:
        print(f"hybrid(best alpha={best_alpha:.2f}):", _fmt_stats(result["hybrid"][best_alpha]))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

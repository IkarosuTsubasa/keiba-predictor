import argparse
import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from surface_scope import get_data_dir, get_predictor_config_path, normalize_scope_key


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SCOPE = "central_dirt"


def _normalize_name(value) -> str:
    return "".join(str(value or "").split())


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    s = float(np.sum(arr))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
    return arr / s


def _softmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    m = float(np.max(arr))
    ex = np.exp(arr - m)
    s = float(np.sum(ex))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
    return ex / s


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    mu = float(np.mean(arr))
    std = float(np.std(arr))
    if (not np.isfinite(std)) or std < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / std


def _parse_grid(text: str, default_text: str) -> List[float]:
    raw = str(text or "").strip()
    if not raw:
        raw = default_text
    if ":" in raw and "," not in raw:
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) == 3:
            try:
                start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
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
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(round(float(token), 6))
        except ValueError:
            continue
    return sorted(set(out))


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
    split_count = max(1, int(split_count))
    out = parsed[:split_count]
    while len(out) < split_count:
        out.append(out[-1] + 11)
    return out


def _candidate_prediction_files(data_dir: Path, run_id: str, hint_path: str) -> List[Path]:
    out = []
    if hint_path:
        p = Path(hint_path)
        out.append(p)
    out.extend(data_dir.rglob(f"predictions_{run_id}*.csv"))
    uniq = []
    seen = set()
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _pick_pred_prob_col(df: pd.DataFrame) -> str:
    for col in ("Top3Prob_model", "Top3Prob_est", "Top3Prob"):
        if col in df.columns:
            return col
    return ""


def _build_races(scope_key: str) -> Tuple[List[Dict], int]:
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
    races = []
    missing_files = 0
    for _, row in pred_results.iterrows():
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue

        actual = [
            _normalize_name(row.get("actual_top1")),
            _normalize_name(row.get("actual_top2")),
            _normalize_name(row.get("actual_top3")),
        ]
        actual_rank = {name: idx + 1 for idx, name in enumerate(actual) if name}
        if len(actual_rank) < 3:
            continue

        pred_file = None
        for cand in _candidate_prediction_files(data_dir, run_id, str(row.get("predictions_path") or "")):
            if cand.exists():
                pred_file = cand
                break
        if pred_file is None:
            missing_files += 1
            continue
        try:
            pred_df = pd.read_csv(pred_file, encoding="utf-8-sig")
        except Exception:
            missing_files += 1
            continue
        if pred_df.empty or "HorseName" not in pred_df.columns:
            continue

        prob_col = _pick_pred_prob_col(pred_df)
        if not prob_col:
            continue
        tmp = pd.DataFrame(index=pred_df.index)
        tmp["horse_key"] = pred_df["HorseName"].map(_normalize_name)
        tmp["rank_score"] = pd.to_numeric(pred_df.get("rank_score"), errors="coerce")
        tmp["top3_prob"] = pd.to_numeric(pred_df.get(prob_col), errors="coerce")
        tmp["rank_score"] = tmp["rank_score"].fillna(tmp["top3_prob"])
        tmp = tmp[(tmp["horse_key"] != "") & tmp["rank_score"].notna() & tmp["top3_prob"].notna()].copy()
        if len(tmp) < 5:
            continue

        tmp["rank"] = tmp["horse_key"].map(actual_rank).fillna(99).astype(int)
        ranks = tmp["rank"].to_numpy(dtype=int)
        if np.sum(ranks <= 3) < 3:
            continue
        races.append(
            {
                "race_id": run_id,
                "rank_score": tmp["rank_score"].to_numpy(dtype=float),
                "top3_prob": tmp["top3_prob"].to_numpy(dtype=float),
                "rank": ranks,
            }
        )
    if not races:
        raise ValueError(f"No usable races for scope={scope_key}")
    return races, missing_files


def _calc_p_win(race: Dict, temperature: float, p_mix_w: float) -> np.ndarray:
    z = _zscore(race["rank_score"])
    p_base = _softmax(float(temperature) * z)
    p_top3 = _normalize_positive(race["top3_prob"])
    p_mix = float(p_mix_w) * p_base + (1.0 - float(p_mix_w)) * p_top3
    return _normalize_positive(p_mix)


def _calc_ranking_metrics(races: Iterable[Dict], temperature: float, p_mix_w: float) -> Dict[str, float]:
    hit_at_5 = []
    top3_hits_at_5 = []
    mrr_top3 = []
    for race in races:
        p_win = _calc_p_win(race, temperature, p_mix_w)
        order = np.argsort(-p_win)
        ranks = race["rank"][order]
        top5 = ranks[:5]
        hit_at_5.append(1.0 if np.any(top5 <= 3) else 0.0)
        top3_hits_at_5.append(float(np.sum(top5 <= 3)))
        rr = 0.0
        for idx, rank_val in enumerate(ranks[:10], start=1):
            if rank_val <= 3:
                rr = 1.0 / float(idx)
                break
        mrr_top3.append(rr)
    return {
        "hit_at_5": float(np.mean(hit_at_5)) if hit_at_5 else 0.0,
        "top3_hits_at_5": float(np.mean(top3_hits_at_5)) if top3_hits_at_5 else 0.0,
        "mrr_top3": float(np.mean(mrr_top3)) if mrr_top3 else 0.0,
    }


def _calc_place_brier(
    races: Iterable[Dict],
    temperature: float,
    p_mix_w: float,
    k_place: float,
    place_bias: float,
    place_power: float,
) -> float:
    errs = []
    for race in races:
        p_win = _calc_p_win(race, temperature, p_mix_w)
        p_place = np.clip(float(place_bias) + (p_win ** float(place_power)) * float(k_place), 0.0, 1.0)
        y = (race["rank"] <= 3).astype(float)
        errs.extend(((p_place - y) ** 2).tolist())
    if not errs:
        return float("inf")
    return float(np.mean(np.asarray(errs, dtype=float)))


def _aggregate_stats(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _render_patch(config_path: Path, old_text: str, new_text: str) -> str:
    diff = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=str(config_path),
        tofile=str(config_path),
        lineterm="",
    )
    return "\n".join(diff) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Tune bet_engine_v2 rank_temperature and place params.")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, help="central_dirt / central_turf / local")
    parser.add_argument("--split-count", type=int, default=5, help="Grouped splits by race_id")
    parser.add_argument("--seeds", default="11,22,33,44,55", help="Comma-separated random seeds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Grouped holdout size")
    parser.add_argument("--temperature-grid", default="0.6:1.8:0.1", help="Grid for rank_temperature")
    parser.add_argument("--k-place-grid", default="2.2:3.8:0.1", help="Grid for K_place")
    parser.add_argument("--place-bias-grid", default="0.0,0.02,0.04,0.06", help="Grid for place_bias")
    parser.add_argument("--place-power-grid", default="0.8:1.4:0.1", help="Grid for place_power")
    parser.add_argument("--apply", action="store_true", help="Apply best params to predictor_config_<scope>.json")
    args = parser.parse_args()

    scope_key = normalize_scope_key(args.scope) or DEFAULT_SCOPE
    split_count = max(1, int(args.split_count))
    seeds = _resolve_seeds(args.seeds, split_count)
    test_size = min(0.9, max(0.05, float(args.test_size)))

    temp_grid = _parse_grid(args.temperature_grid, "0.6:1.8:0.1")
    k_place_grid = _parse_grid(args.k_place_grid, "2.2:3.8:0.1")
    bias_grid = _parse_grid(args.place_bias_grid, "0.0,0.02,0.04,0.06")
    power_grid = _parse_grid(args.place_power_grid, "0.8:1.4:0.1")
    if not temp_grid:
        raise ValueError("temperature grid is empty.")
    if not k_place_grid or not bias_grid or not power_grid:
        raise ValueError("place param grid is empty.")

    cfg_path = get_predictor_config_path(BASE_DIR, scope_key)
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    be_cfg = dict(cfg.get("bet_engine_v2", {}))
    p_mix_w = min(1.0, max(0.0, _safe_float(be_cfg.get("p_mix_w"), 0.6)))

    races, missing_files = _build_races(scope_key)
    race_ids = np.array([r["race_id"] for r in races], dtype=object)
    indices = np.arange(len(races))
    if len(np.unique(race_ids)) < 3:
        raise ValueError(f"Not enough races for grouped split: {len(np.unique(race_ids))}")

    # Phase A: tune rank_temperature by grouped split ranking metrics.
    temp_split_metrics: Dict[float, List[Dict[str, float]]] = {t: [] for t in temp_grid}
    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, test_idx = next(gss.split(indices, groups=race_ids))
        test_races = [races[int(i)] for i in test_idx]
        for temp in temp_grid:
            m = _calc_ranking_metrics(test_races, temp, p_mix_w)
            temp_split_metrics[temp].append(m)

    temp_rows = []
    best_temp = None
    best_key = None
    best_temp_stats = None
    for temp in temp_grid:
        hit_vals = [x["hit_at_5"] for x in temp_split_metrics[temp]]
        top3_vals = [x["top3_hits_at_5"] for x in temp_split_metrics[temp]]
        mrr_vals = [x["mrr_top3"] for x in temp_split_metrics[temp]]
        hit_mean, hit_std = _aggregate_stats(hit_vals)
        top3_mean, top3_std = _aggregate_stats(top3_vals)
        mrr_mean, mrr_std = _aggregate_stats(mrr_vals)
        key = (hit_mean - 0.5 * hit_std, top3_mean, mrr_mean)
        if best_temp is None or key > best_key:
            best_temp = float(temp)
            best_key = key
            best_temp_stats = {
                "hit_mean": hit_mean,
                "hit_std": hit_std,
                "top3_mean": top3_mean,
                "top3_std": top3_std,
                "mrr_mean": mrr_mean,
                "mrr_std": mrr_std,
            }
        temp_rows.append(
            {
                "scope": scope_key,
                "stage": "temperature",
                "value_1": float(temp),
                "value_2": "",
                "value_3": "",
                "hit_at_5_mean": hit_mean,
                "hit_at_5_std": hit_std,
                "top3_hits_at_5_mean": top3_mean,
                "top3_hits_at_5_std": top3_std,
                "mrr_top3_mean": mrr_mean,
                "mrr_top3_std": mrr_std,
                "brier_mean": "",
                "brier_std": "",
                "score_primary": key[0],
            }
        )

    # Phase B: tune place params with selected temperature.
    place_candidates = [(k, b, p) for k in k_place_grid for b in bias_grid for p in power_grid]
    place_split_metrics: Dict[Tuple[float, float, float], List[float]] = {x: [] for x in place_candidates}
    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, test_idx = next(gss.split(indices, groups=race_ids))
        test_races = [races[int(i)] for i in test_idx]
        for key in place_candidates:
            brier = _calc_place_brier(
                test_races,
                temperature=float(best_temp),
                p_mix_w=p_mix_w,
                k_place=float(key[0]),
                place_bias=float(key[1]),
                place_power=float(key[2]),
            )
            place_split_metrics[key].append(brier)

    best_place = None
    best_place_key = None
    for key in place_candidates:
        mean_brier, std_brier = _aggregate_stats(place_split_metrics[key])
        rank_key = (mean_brier + 0.5 * std_brier, mean_brier)
        if best_place is None or rank_key < best_place_key:
            best_place = key
            best_place_key = rank_key
        temp_rows.append(
            {
                "scope": scope_key,
                "stage": "place",
                "value_1": float(key[0]),
                "value_2": float(key[1]),
                "value_3": float(key[2]),
                "hit_at_5_mean": "",
                "hit_at_5_std": "",
                "top3_hits_at_5_mean": "",
                "top3_hits_at_5_std": "",
                "mrr_top3_mean": "",
                "mrr_top3_std": "",
                "brier_mean": mean_brier,
                "brier_std": std_brier,
                "score_primary": -(mean_brier + 0.5 * std_brier),
            }
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = get_data_dir(BASE_DIR, scope_key) / f"tune_bet_engine_v2_params_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(temp_rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(
        f"[{scope_key}] races={len(races)} splits={len(seeds)} missing_pred_files={missing_files} "
        f"p_mix_w={p_mix_w:.3f}"
    )
    print(
        f"best rank_temperature={best_temp:.3f} "
        f"(score={best_key[0]:.6f}, "
        f"hit@5={best_temp_stats['hit_mean']:.4f}+/-{best_temp_stats['hit_std']:.4f}, "
        f"top3@5={best_temp_stats['top3_mean']:.4f}+/-{best_temp_stats['top3_std']:.4f}, "
        f"mrr@10={best_temp_stats['mrr_mean']:.4f}+/-{best_temp_stats['mrr_std']:.4f})"
    )
    print(
        "best place params: "
        f"K_place={best_place[0]:.3f}, place_bias={best_place[1]:.3f}, place_power={best_place[2]:.3f}, "
        f"brier_penalized={best_place_key[0]:.6f}"
    )
    print(f"saved: {out_path}")

    old_cfg = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else "{}\n"
    node = dict(cfg.get("bet_engine_v2", {}))
    node["rank_temperature"] = round(float(best_temp), 4)
    node["K_place"] = round(float(best_place[0]), 4)
    node["place_bias"] = round(float(best_place[1]), 4)
    node["place_power"] = round(float(best_place[2]), 4)
    cfg["bet_engine_v2"] = node
    new_cfg = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"

    if args.apply:
        cfg_path.write_text(new_cfg, encoding="utf-8")
        print(f"applied: {cfg_path}")
    else:
        patch_text = _render_patch(cfg_path, old_cfg, new_cfg)
        patch_path = out_path.with_name(f"tune_bet_engine_v2_params_patch_{ts}.diff")
        patch_path.write_text(patch_text, encoding="utf-8")
        print(f"patch saved: {patch_path}")


if __name__ == "__main__":
    main()

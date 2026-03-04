import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from calibration.temperature_scaling import apply_temperature
from surface_scope import get_data_dir, get_predictor_config_path, normalize_scope_key


BASE_DIR = Path(__file__).resolve().parent


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_prob(arr):
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return x
    x = np.clip(x, 0.0, None)
    s = float(np.sum(x))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(x.shape, 1.0 / float(x.size), dtype=float)
    return x / s


def _softmax(arr):
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return x
    m = float(np.max(x))
    ex = np.exp(x - m)
    s = float(np.sum(ex))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(x.shape, 1.0 / float(x.size), dtype=float)
    return ex / s


def _zscore(arr):
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    std = float(np.std(x))
    if (not np.isfinite(std)) or std < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / std


def _norm_name(value):
    return "".join(str(value or "").split())


def _pick_prob_col(df: pd.DataFrame) -> str:
    for col in ("Top3Prob_model", "Top3Prob_est", "Top3Prob"):
        if col in df.columns:
            return col
    return ""


def _load_mix_params(scope_key: str) -> Tuple[float, float]:
    cfg_path = get_predictor_config_path(BASE_DIR, scope_key)
    if not cfg_path.exists():
        return 0.6, 1.0
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        be3 = cfg.get("bet_engine_v3", {}) if isinstance(cfg, dict) else {}
        return float(be3.get("p_mix_w", 0.6)), float(be3.get("rank_temperature", 1.0))
    except Exception:
        return 0.6, 1.0


def _load_calibration_temp(scope_key: str) -> float:
    cfg_path = get_predictor_config_path(BASE_DIR, scope_key)
    cfg_temp = 1.0
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cal = cfg.get("calibration", {}) if isinstance(cfg.get("calibration", {}), dict) else {}
            cfg_temp = max(1e-6, float(cal.get("win_temp", 1.0)))
        except Exception:
            cfg_temp = 1.0
    temp_path = BASE_DIR / "data" / "model_calibration" / f"{scope_key}_temp.json"
    if temp_path.exists():
        try:
            payload = json.loads(temp_path.read_text(encoding="utf-8"))
            cfg_temp = max(1e-6, float(payload.get("win_temp", cfg_temp)))
        except Exception:
            pass
    return cfg_temp


def build_scope_dataset(scope_key: str) -> Tuple[List[Dict], int]:
    data_dir = get_data_dir(BASE_DIR, scope_key)
    pred_results_path = data_dir / "predictor_results.csv"
    if not pred_results_path.exists():
        return [], 0
    pred_results = pd.read_csv(pred_results_path, encoding="utf-8-sig")
    if pred_results.empty or "run_id" not in pred_results.columns:
        return [], 0
    pred_results["run_id"] = pred_results["run_id"].astype(str).str.strip()
    pred_results = pred_results.drop_duplicates(subset=["run_id"], keep="last")

    p_mix_w, rank_temp = _load_mix_params(scope_key)
    win_temp = _load_calibration_temp(scope_key)
    races = []
    missing_files = 0

    for _, row in pred_results.iterrows():
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        pred_path = str(row.get("predictions_path", "")).strip()
        pred_file = Path(pred_path) if pred_path else None
        if pred_file is None or (not pred_file.exists()):
            cands = list(data_dir.rglob(f"predictions_{run_id}*.csv"))
            pred_file = cands[0] if cands else None
        if pred_file is None or (not pred_file.exists()):
            missing_files += 1
            continue

        try:
            pred_df = pd.read_csv(pred_file, encoding="utf-8-sig")
        except Exception:
            missing_files += 1
            continue
        if pred_df.empty or "HorseName" not in pred_df.columns:
            continue
        prob_col = _pick_prob_col(pred_df)
        if not prob_col:
            continue

        actual_top3 = [_norm_name(row.get("actual_top1")), _norm_name(row.get("actual_top2")), _norm_name(row.get("actual_top3"))]
        actual_top3 = [x for x in actual_top3 if x]
        if len(actual_top3) < 3:
            continue
        actual_rank = {name: idx + 1 for idx, name in enumerate(actual_top3)}

        race = pred_df.copy()
        race["horse_key"] = race["HorseName"].map(_norm_name)
        race["rank_score"] = pd.to_numeric(race.get("rank_score"), errors="coerce")
        race["top3_prob"] = pd.to_numeric(race.get(prob_col), errors="coerce")
        race["rank_score"] = race["rank_score"].fillna(race["top3_prob"])
        race = race[(race["horse_key"] != "") & race["rank_score"].notna() & race["top3_prob"].notna()].copy()
        if len(race) < 5:
            continue

        race["rank_true"] = race["horse_key"].map(actual_rank).fillna(99).astype(int)
        z = _zscore(race["rank_score"].to_numpy(dtype=float))
        p_base = _softmax(float(rank_temp) * z)
        p_top3 = _normalize_prob(race["top3_prob"].to_numpy(dtype=float))
        p_uncal = _normalize_prob(float(p_mix_w) * p_base + (1.0 - float(p_mix_w)) * p_top3)
        p_cal = _normalize_prob(apply_temperature(p_uncal, float(win_temp)))
        rank_true = race["rank_true"].to_numpy(dtype=int)
        races.append(
            {
                "race_id": run_id,
                "p_win": p_cal,
                "rank_true": rank_true,
            }
        )
    return races, missing_files


def _parse_grid(text: str, default_text: str) -> List[float]:
    raw = str(text or "").strip() or default_text
    if ":" in raw and "," not in raw:
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) == 3:
            try:
                start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
                vals = []
                cur = start
                while cur <= end + 1e-12:
                    vals.append(round(float(cur), 6))
                    cur += step
                return vals
            except Exception:
                pass
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(round(float(token), 6))
        except Exception:
            continue
    return out


def _brier_place(races: List[Dict], k_place: float) -> float:
    errs = []
    for race in races:
        p = race["p_win"]
        y = (race["rank_true"] <= 3).astype(float)
        pred = np.clip(p * float(k_place), 0.0, 1.0)
        errs.extend(((pred - y) ** 2).tolist())
    return float(np.mean(errs)) if errs else float("inf")


def _brier_pairs(races: List[Dict], c_const: float, pair_kind: str) -> float:
    errs = []
    for race in races:
        p = race["p_win"]
        ranks = race["rank_true"]
        n = len(p)
        for i in range(n):
            for j in range(i + 1, n):
                pred = np.clip(float(p[i]) * float(p[j]) * float(c_const), 0.0, 1.0)
                if pair_kind == "wide":
                    y = 1.0 if (ranks[i] <= 3 and ranks[j] <= 3) else 0.0
                else:
                    y = 1.0 if (ranks[i] <= 2 and ranks[j] <= 2) else 0.0
                errs.append((pred - y) ** 2)
    return float(np.mean(errs)) if errs else float("inf")


def _pick_best(races: List[Dict], grid: List[float], fn) -> float:
    best_v = grid[0] if grid else 1.0
    best_loss = float("inf")
    for v in grid:
        loss = float(fn(races, float(v)))
        if loss < best_loss:
            best_loss = loss
            best_v = float(v)
    return float(round(best_v, 4))


def fit_constants(scope_key: str, split_seed: int = 42, test_size: float = 0.2, k_grid_text: str = "2.0:4.0:0.05", c_grid_text: str = "0.8:2.6:0.05"):
    races, missing_files = build_scope_dataset(scope_key)
    if not races:
        return {"K_place": 3.0, "C_wide": 1.6, "C_quinella": 1.6, "n_races": 0, "missing_pred_files": missing_files}

    groups = np.array([r["race_id"] for r in races], dtype=object)
    idx = np.arange(len(races))
    if len(np.unique(groups)) >= 5:
        gss = GroupShuffleSplit(n_splits=1, test_size=min(0.5, max(0.1, float(test_size))), random_state=int(split_seed))
        _, valid_idx = next(gss.split(idx, groups=groups))
        valid_races = [races[int(i)] for i in valid_idx]
    else:
        valid_races = races

    k_grid = _parse_grid(k_grid_text, "2.0:4.0:0.05")
    c_grid = _parse_grid(c_grid_text, "0.8:2.6:0.05")
    k_best = _pick_best(valid_races, k_grid, _brier_place)
    c_wide_best = _pick_best(valid_races, c_grid, lambda rr, cc: _brier_pairs(rr, cc, "wide"))
    c_quinella_best = _pick_best(valid_races, c_grid, lambda rr, cc: _brier_pairs(rr, cc, "quinella"))
    return {
        "K_place": k_best,
        "C_wide": c_wide_best,
        "C_quinella": c_quinella_best,
        "n_races": int(len(valid_races)),
        "missing_pred_files": int(missing_files),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit K_place/C_wide/C_quinella by scope.")
    parser.add_argument("--scope", default="central_dirt", help="central_dirt / central_turf / local")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-grid", default="2.0:4.0:0.05")
    parser.add_argument("--c-grid", default="0.8:2.6:0.05")
    parser.add_argument("--apply-config", action="store_true", help="Write constants into predictor config bet_engine_v3")
    args = parser.parse_args()

    scope = normalize_scope_key(args.scope) or "central_dirt"
    fitted = fit_constants(
        scope_key=scope,
        split_seed=args.seed,
        test_size=args.test_size,
        k_grid_text=args.k_grid,
        c_grid_text=args.c_grid,
    )
    payload = {
        "K_place": float(fitted["K_place"]),
        "C_wide": float(fitted["C_wide"]),
        "C_quinella": float(fitted["C_quinella"]),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "n_races": int(fitted["n_races"]),
    }
    out_path = BASE_DIR / "data" / "bet_constants" / f"{scope}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{scope}] K_place={payload['K_place']:.4f} C_wide={payload['C_wide']:.4f} C_quinella={payload['C_quinella']:.4f}")
    print(f"n_races={payload['n_races']} missing_pred_files={fitted['missing_pred_files']} saved={out_path}")

    if args.apply_config:
        cfg_path = get_predictor_config_path(BASE_DIR, scope)
        cfg = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        be3 = cfg.get("bet_engine_v3", {}) if isinstance(cfg.get("bet_engine_v3", {}), dict) else {}
        be3["K_place"] = payload["K_place"]
        be3["C_wide"] = payload["C_wide"]
        be3["C_quinella"] = payload["C_quinella"]
        cfg["bet_engine_v3"] = be3
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated config: {cfg_path}")


if __name__ == "__main__":
    main()


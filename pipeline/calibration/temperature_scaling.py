import json
import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))

from surface_scope import get_data_dir, get_predictor_config_path, normalize_scope_key


EPS = 1e-12


def normalize_prob(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    s = float(np.sum(arr))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
    return arr / s


def apply_temperature(prob: Sequence[float], temperature: float, eps: float = EPS) -> np.ndarray:
    p = normalize_prob(prob)
    if p.size == 0:
        return p
    t = max(float(temperature), eps)
    logits = np.log(np.clip(p, eps, 1.0)) / t
    logits = logits - float(np.max(logits))
    ex = np.exp(logits)
    s = float(np.sum(ex))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(p.shape, 1.0 / float(p.size), dtype=float)
    return ex / s


def _nll_objective(prob_list: List[np.ndarray], winner_idx: List[int], temperature: float) -> float:
    losses = []
    for p, wi in zip(prob_list, winner_idx):
        if wi < 0 or wi >= len(p):
            continue
        p_cal = apply_temperature(p, temperature)
        losses.append(-np.log(max(float(p_cal[wi]), EPS)))
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def _brier_objective(prob_list: List[np.ndarray], winner_idx: List[int], temperature: float) -> float:
    losses = []
    for p, wi in zip(prob_list, winner_idx):
        if wi < 0 or wi >= len(p):
            continue
        p_cal = apply_temperature(p, temperature)
        y = np.zeros(len(p_cal), dtype=float)
        y[wi] = 1.0
        losses.append(float(np.mean((p_cal - y) ** 2)))
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def fit_temperature(
    prob_list: Iterable[Sequence[float]],
    winner_idx: Iterable[int],
    objective: str = "nll",
    t_min: float = 0.4,
    t_max: float = 2.5,
    t_step: float = 0.02,
) -> float:
    probs = [normalize_prob(p) for p in prob_list]
    wins = [int(x) for x in winner_idx]
    if not probs:
        return 1.0
    if objective == "brier":
        fn = _brier_objective
    else:
        fn = _nll_objective

    best_t = 1.0
    best_loss = float("inf")
    t = float(t_min)
    while t <= float(t_max) + 1e-12:
        loss = fn(probs, wins, t)
        if loss < best_loss:
            best_loss = loss
            best_t = t
        t += float(t_step)

    # Local refinement around best T.
    lo = max(float(t_min), best_t - 0.08)
    hi = min(float(t_max), best_t + 0.08)
    t = lo
    while t <= hi + 1e-12:
        loss = fn(probs, wins, t)
        if loss < best_loss:
            best_loss = loss
            best_t = t
        t += 0.005
    return float(round(best_t, 4))


@dataclass
class CalibrationRecord:
    win_temp: float
    updated_at: str
    n_races: int
    objective: str = "nll"


def save_temperature_record(path: Path, win_temp: float, n_races: int, objective: str = "nll") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = CalibrationRecord(
        win_temp=float(win_temp),
        updated_at=datetime.now().isoformat(timespec="seconds"),
        n_races=int(n_races),
        objective=str(objective),
    )
    path.write_text(json.dumps(payload.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_temperature(path: Path, default: float = 1.0) -> float:
    if not path.exists():
        return float(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return max(EPS, float(payload.get("win_temp", default)))
    except Exception:
        return float(default)


def _softmax(arr: Sequence[float]) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return x
    m = float(np.max(x))
    ex = np.exp(x - m)
    s = float(np.sum(ex))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(x.shape, 1.0 / float(x.size), dtype=float)
    return ex / s


def _zscore(arr: Sequence[float]) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    std = float(np.std(x))
    if (not np.isfinite(std)) or std < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / std


def _norm_name(text) -> str:
    return "".join(str(text or "").split())


def _pick_prob_col(df: pd.DataFrame) -> str:
    for col in ("Top3Prob_model", "Top3Prob_est", "Top3Prob"):
        if col in df.columns:
            return col
    return ""


def _load_bet_mix_params(scope_key: str) -> Tuple[float, float]:
    cfg_path = get_predictor_config_path(Path(__file__).resolve().parents[1], scope_key)
    if not cfg_path.exists():
        return 0.6, 1.0
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        be3 = cfg.get("bet_engine_v3", {}) if isinstance(cfg, dict) else {}
        return float(be3.get("p_mix_w", 0.6)), float(be3.get("rank_temperature", 1.0))
    except Exception:
        return 0.6, 1.0


def build_race_prob_dataset(scope_key: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = get_data_dir(base_dir, scope_key)
    pred_results_path = data_dir / "predictor_results.csv"
    if not pred_results_path.exists():
        return [], [], []
    pred_results = pd.read_csv(pred_results_path, encoding="utf-8-sig")
    if pred_results.empty or "run_id" not in pred_results.columns:
        return [], [], []
    pred_results["run_id"] = pred_results["run_id"].astype(str).str.strip()
    pred_results = pred_results.drop_duplicates(subset=["run_id"], keep="last")

    p_mix_w, rank_temp = _load_bet_mix_params(scope_key)
    probs = []
    winners = []
    race_ids = []
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
            continue

        try:
            pred_df = pd.read_csv(pred_file, encoding="utf-8-sig")
        except Exception:
            continue
        if pred_df.empty or "HorseName" not in pred_df.columns:
            continue
        prob_col = _pick_prob_col(pred_df)
        if not prob_col:
            continue

        race = pred_df.copy()
        race["horse_key"] = race["HorseName"].map(_norm_name)
        race["rank_score"] = pd.to_numeric(race.get("rank_score"), errors="coerce")
        race["top3_prob"] = pd.to_numeric(race.get(prob_col), errors="coerce")
        race["rank_score"] = race["rank_score"].fillna(race["top3_prob"])
        race = race[(race["horse_key"] != "") & race["rank_score"].notna() & race["top3_prob"].notna()].copy()
        if len(race) < 2:
            continue

        winner_name = _norm_name(row.get("actual_top1", ""))
        if not winner_name:
            continue
        horse_keys = race["horse_key"].tolist()
        if winner_name not in horse_keys:
            continue
        winner_idx = horse_keys.index(winner_name)
        z = _zscore(race["rank_score"].to_numpy(dtype=float))
        p_base = _softmax(float(rank_temp) * z)
        p_top3 = normalize_prob(race["top3_prob"].to_numpy(dtype=float))
        p_uncal = normalize_prob(float(p_mix_w) * p_base + (1.0 - float(p_mix_w)) * p_top3)
        probs.append(p_uncal)
        winners.append(int(winner_idx))
        race_ids.append(run_id)
    return probs, winners, race_ids


def fit_scope_temperature(
    scope_key: str,
    objective: str = "nll",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[float, int]:
    probs, winners, race_ids = build_race_prob_dataset(scope_key)
    if not probs:
        return 1.0, 0
    groups = np.asarray(race_ids)
    idx = np.arange(len(probs))
    if len(np.unique(groups)) >= 5:
        gss = GroupShuffleSplit(n_splits=1, test_size=min(0.5, max(0.1, float(test_size))), random_state=random_state)
        _, valid_idx = next(gss.split(idx, groups=groups))
        valid_probs = [probs[int(i)] for i in valid_idx]
        valid_winners = [winners[int(i)] for i in valid_idx]
    else:
        valid_probs = probs
        valid_winners = winners
    t = fit_temperature(valid_probs, valid_winners, objective=objective)
    return t, int(len(valid_probs))


def main():
    parser = argparse.ArgumentParser(description="Fit scope-level win temperature scaling.")
    parser.add_argument("--scope", default="central_dirt", help="central_dirt / central_turf / local")
    parser.add_argument("--objective", default="nll", choices=["nll", "brier"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--apply-config", action="store_true", help="Write calibration.win_temp into predictor config")
    args = parser.parse_args()

    scope = normalize_scope_key(args.scope) or "central_dirt"
    t, n_races = fit_scope_temperature(scope, objective=args.objective, test_size=args.test_size)
    base_dir = Path(__file__).resolve().parents[1]
    out_path = base_dir / "data" / "model_calibration" / f"{scope}_temp.json"
    save_temperature_record(out_path, win_temp=t, n_races=n_races, objective=args.objective)
    print(f"[{scope}] win_temp={t:.4f} n_races={n_races} saved={out_path}")

    if args.apply_config:
        cfg_path = get_predictor_config_path(base_dir, scope)
        cfg = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cal = cfg.get("calibration", {}) if isinstance(cfg.get("calibration", {}), dict) else {}
        cal["win_temp"] = float(t)
        cal["enabled"] = True
        cfg["calibration"] = cal
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated config: {cfg_path}")


if __name__ == "__main__":
    main()

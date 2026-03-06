import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from surface_scope import get_data_dir


EPS = 1e-9


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip01_array(values):
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, EPS, 1.0 - EPS)


def _sigmoid(x):
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def _logit(p):
    x = _clip01_array(p)
    return np.log(x / (1.0 - x))


def _normalize_prob(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    s = float(np.sum(arr))
    if s <= 0.0 or (not np.isfinite(s)):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
    return arr / s


def _softmax(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    m = float(np.max(arr))
    ex = np.exp(arr - m)
    s = float(np.sum(ex))
    if s <= 0.0 or (not np.isfinite(s)):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
    return ex / s


def _zscore(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    mu = float(np.mean(arr))
    std = float(np.std(arr))
    if (not np.isfinite(std)) or std < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / std


def _norm_name(text):
    return "".join(str(text or "").split())


def _pick_prob_col(df: pd.DataFrame) -> str:
    for col in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if col in df.columns:
            return col
    return ""


class TemperatureScaling:
    """
    Binary temperature scaling on logit:
      p_cal = sigmoid(logit(p) / T)
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(EPS, float(temperature))

    def fit(self, probs: Iterable[float], labels: Iterable[int]):
        p = _clip01_array(probs)
        y = np.asarray(labels, dtype=float)
        if p.size == 0 or y.size == 0 or p.size != y.size:
            self.temperature = 1.0
            return self

        t_grid = np.concatenate(
            [
                np.linspace(0.4, 1.4, 41),
                np.linspace(1.5, 3.0, 31),
            ]
        )
        best_t = 1.0
        best_loss = float("inf")
        lg = _logit(p)
        for t in t_grid:
            p_cal = _sigmoid(lg / max(EPS, float(t)))
            loss = -np.mean(y * np.log(np.clip(p_cal, EPS, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p_cal, EPS, 1.0)))
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_t = float(t)
        self.temperature = max(EPS, float(best_t))
        return self

    def predict(self, probs: Iterable[float]) -> np.ndarray:
        p = _clip01_array(probs)
        lg = _logit(p)
        return _sigmoid(lg / max(EPS, float(self.temperature)))


class PlattScaling:
    """
    Platt scaling:
      p_cal = sigmoid(a * logit(p) + b)
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    def fit(self, probs: Iterable[float], labels: Iterable[int], lr: float = 0.05, n_iter: int = 600, l2: float = 1e-3):
        p = _clip01_array(probs)
        y = np.asarray(labels, dtype=float)
        if p.size == 0 or y.size == 0 or p.size != y.size:
            self.a = 1.0
            self.b = 0.0
            return self

        x = _logit(p)
        a = float(self.a)
        b = float(self.b)
        n = float(len(x))
        for _ in range(max(1, int(n_iter))):
            z = a * x + b
            y_hat = _sigmoid(z)
            err = y_hat - y
            grad_a = float(np.dot(err, x) / n + l2 * a)
            grad_b = float(np.mean(err))
            a -= float(lr) * grad_a
            b -= float(lr) * grad_b
        self.a = float(a)
        self.b = float(b)
        return self

    def predict(self, probs: Iterable[float]) -> np.ndarray:
        p = _clip01_array(probs)
        x = _logit(p)
        return _sigmoid(self.a * x + self.b)


def apply_calibration(p, calibrator: Dict):
    """
    Apply calibration on scalar or array-like probabilities.
    """
    arr = np.asarray(p, dtype=float)
    if arr.size == 0:
        return arr
    cal = calibrator if isinstance(calibrator, dict) else {}
    method = str(cal.get("method", "identity")).strip().lower()

    if method == "temperature":
        model = TemperatureScaling(float(cal.get("temperature", 1.0)))
        out = model.predict(arr)
    elif method == "platt":
        model = PlattScaling(float(cal.get("a", 1.0)), float(cal.get("b", 0.0)))
        out = model.predict(arr)
    else:
        out = _clip01_array(arr)

    if np.isscalar(p):
        return float(out[0])
    return out


def _brier_score(y_true, y_prob):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.size == 0 or p.size == 0 or y.size != p.size:
        return float("inf")
    return float(np.mean((y - p) ** 2))


def _group_split(df: pd.DataFrame, group_col: str, valid_ratio: float, random_state: int):
    if df.empty:
        return df.copy(), df.copy()
    groups = [str(x) for x in df[group_col].fillna("__na__").astype(str).tolist()]
    uniq = sorted(set(groups))
    if len(uniq) <= 1:
        return df.copy(), df.iloc[0:0].copy()
    rng = np.random.RandomState(int(random_state))
    rng.shuffle(uniq)
    n_valid = max(1, int(round(float(len(uniq)) * float(valid_ratio))))
    n_valid = min(n_valid, max(1, len(uniq) - 1))
    valid_groups = set(uniq[:n_valid])
    mask = df[group_col].astype(str).isin(valid_groups)
    return df.loc[~mask].copy(), df.loc[mask].copy()


def _fit_one_target(train_df: pd.DataFrame, valid_df: pd.DataFrame, p_col: str, y_col: str) -> Dict:
    y_train = train_df[y_col].astype(int).to_numpy(dtype=int)
    p_train = train_df[p_col].astype(float).to_numpy(dtype=float)
    if len(y_train) < 20 or np.min(y_train) == np.max(y_train):
        return {"method": "identity", "brier_valid": float("nan")}

    y_valid = valid_df[y_col].astype(int).to_numpy(dtype=int) if not valid_df.empty else y_train
    p_valid = valid_df[p_col].astype(float).to_numpy(dtype=float) if not valid_df.empty else p_train

    t_model = TemperatureScaling().fit(p_train, y_train)
    p_t = t_model.predict(p_valid)
    b_t = _brier_score(y_valid, p_t)

    p_model = PlattScaling().fit(p_train, y_train)
    p_p = p_model.predict(p_valid)
    b_p = _brier_score(y_valid, p_p)

    if b_p <= b_t:
        return {
            "method": "platt",
            "a": float(p_model.a),
            "b": float(p_model.b),
            "brier_valid": float(b_p),
        }
    return {
        "method": "temperature",
        "temperature": float(t_model.temperature),
        "brier_valid": float(b_t),
    }


def _calib_dir(base_dir: Path) -> Path:
    return Path(base_dir) / "data" / "calibration"


def save_calibrators(payload: Dict, scope_key: str, base_dir: Path = None) -> Path:
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    out_dir = _calib_dir(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scope_key}_calib.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_calibrators(scope_key: str, base_dir: Path = None) -> Dict:
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    path = _calib_dir(base) / f"{scope_key}_calib.json"
    if not path.exists():
        return {
            "scope_key": str(scope_key),
            "models": {
                "win": {"method": "identity"},
                "place": {"method": "identity"},
            },
            "updated_at": "",
            "source": "fallback_identity",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid payload")
        payload.setdefault("models", {})
        payload["models"].setdefault("win", {"method": "identity"})
        payload["models"].setdefault("place", {"method": "identity"})
        payload["source"] = str(path)
        return payload
    except Exception:
        return {
            "scope_key": str(scope_key),
            "models": {
                "win": {"method": "identity"},
                "place": {"method": "identity"},
            },
            "updated_at": "",
            "source": "fallback_identity_after_load_error",
        }


def fit_calibrators(df: pd.DataFrame, scope_key: str, base_dir: Path = None, valid_ratio: float = 0.2, random_state: int = 42) -> Dict:
    """
    Fit win/place calibrators from dataframe columns:
      race_id, p_win_base, p_place_base, label_win, label_place
    Group split is done by race_id to avoid leakage.
    """
    required = {"race_id", "p_win_base", "p_place_base", "label_win", "label_place"}
    if not isinstance(df, pd.DataFrame) or df.empty or not required.issubset(set(df.columns)):
        payload = {
            "scope_key": str(scope_key),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "n_rows": int(0),
            "n_races": int(0),
            "split": {"group_col": "race_id", "valid_ratio": float(valid_ratio), "random_state": int(random_state)},
            "models": {
                "win": {"method": "identity"},
                "place": {"method": "identity"},
            },
            "status": "insufficient_data",
        }
        save_calibrators(payload, scope_key=scope_key, base_dir=base_dir)
        return payload

    work = df.copy()
    work["race_id"] = work["race_id"].fillna("").astype(str)
    work = work[work["race_id"] != ""].copy()
    for col in ("p_win_base", "p_place_base"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ("label_win", "label_place"):
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)
    work = work[work["p_win_base"].notna() & work["p_place_base"].notna()].copy()

    train_df, valid_df = _group_split(work, group_col="race_id", valid_ratio=float(valid_ratio), random_state=int(random_state))
    win_model = _fit_one_target(train_df, valid_df, p_col="p_win_base", y_col="label_win")
    place_model = _fit_one_target(train_df, valid_df, p_col="p_place_base", y_col="label_place")

    payload = {
        "scope_key": str(scope_key),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(len(work)),
        "n_races": int(work["race_id"].nunique()),
        "split": {
            "group_col": "race_id",
            "valid_ratio": float(valid_ratio),
            "random_state": int(random_state),
            "n_train_rows": int(len(train_df)),
            "n_valid_rows": int(len(valid_df)),
            "n_train_races": int(train_df["race_id"].nunique()) if not train_df.empty else 0,
            "n_valid_races": int(valid_df["race_id"].nunique()) if not valid_df.empty else 0,
        },
        "models": {
            "win": win_model,
            "place": place_model,
        },
        "status": "ok",
    }
    save_calibrators(payload, scope_key=scope_key, base_dir=base_dir)
    return payload


def build_scope_calibration_df(scope_key: str, base_dir: Path = None, config: Dict = None) -> pd.DataFrame:
    """
    Build calibration dataset from predictor_results + predictions + actual labels.
    """
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    scope_dir = get_data_dir(base, scope_key)
    pred_results_path = scope_dir / "predictor_results.csv"
    if not pred_results_path.exists():
        return pd.DataFrame()
    try:
        pred_results = pd.read_csv(pred_results_path, encoding="utf-8-sig")
    except Exception:
        pred_results = pd.read_csv(pred_results_path, encoding="utf-8")
    if pred_results.empty or "run_id" not in pred_results.columns:
        return pd.DataFrame()

    cfg = dict(config or {})
    p_mix_w = float(cfg.get("p_mix_w", 0.6))
    rank_temperature = max(EPS, float(cfg.get("rank_temperature", 1.0)))
    place_scale = float(cfg.get("place_scale", 3.0))
    p_place_blend_w = float(cfg.get("p_place_blend_w", 0.5))
    p_place_cap = float(cfg.get("p_place_cap", 0.75))

    rows = []
    pred_results["run_id"] = pred_results["run_id"].astype(str).str.strip()
    for _, row in pred_results.iterrows():
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        pred_path_text = str(row.get("predictions_path", "") or "").strip()
        pred_path = Path(pred_path_text) if pred_path_text else None
        if pred_path is None or (not pred_path.exists()):
            cands = list(scope_dir.rglob(f"predictions_{run_id}*.csv"))
            pred_path = cands[0] if cands else None
        if pred_path is None or (not pred_path.exists()):
            continue

        try:
            pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
        except Exception:
            try:
                pred_df = pd.read_csv(pred_path, encoding="utf-8")
            except Exception:
                continue
        if pred_df.empty or "HorseName" not in pred_df.columns:
            continue
        prob_col = _pick_prob_col(pred_df)
        if not prob_col:
            continue

        race = pred_df.copy()
        race["horse_key"] = race["HorseName"].map(_norm_name)
        race = race[race["horse_key"] != ""].copy()
        if race.empty:
            continue

        race["rank_score"] = pd.to_numeric(race.get("rank_score"), errors="coerce")
        race["top3_prob_raw"] = pd.to_numeric(race.get(prob_col), errors="coerce")
        race["rank_score"] = race["rank_score"].fillna(race["top3_prob_raw"])
        race["top3_prob_raw"] = race["top3_prob_raw"].fillna(0.0)
        race = race[race["rank_score"].notna()].copy()
        if len(race) < 2:
            continue

        z = _zscore(race["rank_score"].to_numpy(dtype=float))
        p_rank = _softmax(rank_temperature * z)
        p_top3_norm = _normalize_prob(race["top3_prob_raw"].to_numpy(dtype=float))
        p_win_base = _normalize_prob(p_mix_w * p_rank + (1.0 - p_mix_w) * p_top3_norm)
        p_place_base = []
        for top3_raw, p_win in zip(race["top3_prob_raw"].to_numpy(dtype=float), p_win_base):
            t = min(1.0, max(0.0, float(top3_raw)))
            m = min(1.0, max(0.0, float(p_win) * place_scale))
            v = p_place_blend_w * t + (1.0 - p_place_blend_w) * m
            p_place_base.append(min(p_place_cap, min(1.0, max(0.0, v))))
        p_place_base = np.asarray(p_place_base, dtype=float)

        top1 = _norm_name(row.get("actual_top1", ""))
        top3_names = {
            _norm_name(row.get("actual_top1", "")),
            _norm_name(row.get("actual_top2", "")),
            _norm_name(row.get("actual_top3", "")),
        }
        top3_names = {x for x in top3_names if x}
        if not top1 or len(top3_names) < 3:
            continue

        for i, r in race.reset_index(drop=True).iterrows():
            hk = str(r.get("horse_key", "")).strip()
            if not hk:
                continue
            rows.append(
                {
                    "scope_key": str(scope_key),
                    "race_id": str(run_id),
                    "horse_key": hk,
                    "p_win_base": float(p_win_base[i]),
                    "p_place_base": float(p_place_base[i]),
                    "label_win": int(1 if hk == top1 else 0),
                    "label_place": int(1 if hk in top3_names else 0),
                }
            )
    return pd.DataFrame(rows)


def fit_scope_calibrators(scope_key: str, base_dir: Path = None, config: Dict = None, valid_ratio: float = 0.2, random_state: int = 42) -> Dict:
    df = build_scope_calibration_df(scope_key=scope_key, base_dir=base_dir, config=config)
    return fit_calibrators(
        df=df,
        scope_key=scope_key,
        base_dir=base_dir,
        valid_ratio=valid_ratio,
        random_state=random_state,
    )

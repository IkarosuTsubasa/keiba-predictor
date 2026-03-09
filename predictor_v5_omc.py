#!/usr/bin/env python3
"""
predictor_v5.py - Ultimate Stacking Ensemble Horse Racing Predictor
====================================================================
Combines the best techniques from v0-v4 with significant innovations:

1.  Multi-model stacking: 3 diverse LGBMClassifiers + 1 LGBMRanker → LogisticRegression meta-learner
2.  All 4 odds types: win, place (fukusho), wide, quinella → market network centrality
3.  Enhanced 馬場指数 interaction: horse's preferred baba range, category aptitude
4.  Running style decomposition: early position, gain, final stretch (上り)
5.  Jockey-course synergy with Bayesian smoothing
6.  Horse weight momentum (change trend, optimal deviation)
7.  Class transition detection via prize money analysis
8.  Pair-wise market consensus / network centrality from wide+quinella odds
9.  Bayesian smoothing for all small-sample statistics
10. Time-series expanding-window CV with date-based splitting
11. Platt scaling calibration on OOF stacked predictions

Key domain knowledge:
- 馬場指数 (Baba Index): Track condition. Positive=slow, Negative=fast, Zero=standard
- タイム指数 (Time Index): Performance rating. Higher=better
- Adjusted ability = TimeIndex - BabaIndex (normalizes across track conditions)

Usage:
    python predictor_v5.py
    # or via pipeline with PREDICTIONS_OUTPUT env var
"""

from __future__ import annotations

import math
import os
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# UTF-8 I/O
# ---------------------------------------------------------------------------
def configure_utf8_io():
    for s in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_io()

# ---------------------------------------------------------------------------
# Scope / Config
# ---------------------------------------------------------------------------
SCOPE_ALIASES = {
    "central_turf": {"central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"},
    "central_dirt": {"central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"},
    "local": {"local", "l", "3"},
}

GOING_MAP = {"良": 0.0, "稍": 1.0, "稍重": 1.0, "重": 2.0, "不": 3.0, "不良": 3.0}
BABA_PROXY_MAP = {"良": -5.0, "稍": 5.0, "稍重": 5.0, "重": 10.0, "不": 15.0, "不良": 15.0}


def _resolve_scope() -> str:
    raw = (os.environ.get("SCOPE_KEY", "") or "").strip().lower().replace(" ", "_").replace("-", "_")
    for k, aliases in SCOPE_ALIASES.items():
        if raw in aliases:
            return k
    return "central_dirt"


@dataclass
class Config:
    scope_key: str = ""
    output_path: Path = Path("predictions.csv")
    target_surface: str = "ダ"
    target_distance: int = 1800
    target_condition: str = "稍重"
    target_location: str = "中山"
    expected_baba_index: float = 5.0

    @staticmethod
    def from_env() -> "Config":
        scope = _resolve_scope()
        out_path = Path(os.environ.get("PREDICTIONS_OUTPUT", "predictions.csv")).expanduser()
        default_surf = "芝" if scope == "central_turf" else "ダ"
        surf_env = (os.environ.get("PREDICTOR_TARGET_SURFACE", "") or "").strip()
        if surf_env:
            surface = _normalize_surface(surf_env, default_surf)
        else:
            surface = default_surf
        dist_env = (os.environ.get("PREDICTOR_TARGET_DISTANCE", "") or "").strip()
        distance = int(float(dist_env)) if dist_env else 1800
        cond_env = (os.environ.get("PREDICTOR_TARGET_CONDITION", "") or "").strip()
        condition = cond_env if cond_env else "稍重"
        loc_env = (os.environ.get("PREDICTOR_TARGET_LOCATION", "") or "").strip()
        location = loc_env if loc_env else ""
        baba_env = (os.environ.get("PREDICTOR_BABA_INDEX_PROXY", "") or "").strip()
        baba = float(baba_env) if baba_env else BABA_PROXY_MAP.get(condition, 5.0)
        return Config(
            scope_key=scope,
            output_path=out_path,
            target_surface=surface,
            target_distance=distance,
            target_condition=condition,
            target_location=location,
            expected_baba_index=baba,
        )


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------
def _pf(x) -> float:
    try:
        return float(str(x).replace(",", "").replace('"', "").strip())
    except Exception:
        return np.nan


def _pi(x) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return 0


def _normalize_surface(val: str, fallback: str = "ダ") -> str:
    t = val.strip().lower()
    if t in ("芝", "t", "turf", "grass", "shiba", "1"):
        return "芝"
    if t in ("ダ", "d", "dirt", "sand", "2"):
        return "ダ"
    return fallback


def _parse_venue(v) -> str:
    s = str(v or "").strip()
    if not s:
        return ""
    return re.sub(r"\d+", "", s).strip()


def _parse_surface_distance(v) -> Tuple[str, float]:
    s = str(v or "").strip()
    m = re.match(r"(芝|ダ|障)(\d+)", s)
    if not m:
        return "", np.nan
    return m.group(1), float(m.group(2))


def _parse_positions(v) -> List[int]:
    s = str(v or "").strip()
    if not s:
        return []
    out = []
    for part in re.split(r"[-\s]+", s):
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


def _parse_weight_change(v) -> Tuple[float, float]:
    """Parse 馬体重 like '456(+4)' -> (456.0, 4.0)"""
    s = str(v or "").strip()
    m = re.match(r"(\d+)\(([+-]?\d+)\)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), 0.0
    except Exception:
        return np.nan, 0.0


def _normalize_name(v) -> str:
    return "".join(str(v or "").split())


def _logistic(x: float) -> float:
    x = max(min(x, 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    t = max(temperature, 1e-6)
    mx = max(scores)
    ex = [math.exp((s - mx) / t) for s in scores]
    z = sum(ex)
    if z <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / z for e in ex]


# ---------------------------------------------------------------------------
# Bayesian Smoother
# ---------------------------------------------------------------------------
class BayesianSmoother:
    def __init__(self, prior_mean: float, prior_strength: int = 5):
        self.prior_mean = prior_mean
        self.k = prior_strength

    def smooth(self, observed_mean: float, n: int) -> float:
        if n <= 0:
            return self.prior_mean
        return (n * observed_mean + self.k * self.prior_mean) / (n + self.k)


# ---------------------------------------------------------------------------
# Odds Engine - processes all 4 odds types
# ---------------------------------------------------------------------------
class OddsEngine:
    def __init__(self):
        self._features: Dict[str, Dict[str, float]] = {}
        self._hno_to_name: Dict[int, str] = {}

    def load_and_compute(
        self,
        win_path: str = "odds.csv",
        place_path: str = "fuku_odds.csv",
        wide_path: str = "wide_odds.csv",
        quinella_path: str = "quinella_odds.csv",
    ) -> None:
        # Win odds (required — warn loudly if missing or malformed)
        if not Path(win_path).exists():
            print(f"[WARN] Win odds file not found: {win_path}")
        else:
            try:
                df = pd.read_csv(win_path)
                required_cols = {"horse_no", "name", "odds"}
                missing = required_cols - set(df.columns)
                if missing:
                    print(f"[WARN] {win_path} missing columns: {missing}")
                else:
                    for _, r in df.iterrows():
                        name = _normalize_name(r.get("name", ""))
                        if not name:
                            continue
                        hno = _pi(r.get("horse_no", 0))
                        odds = _pf(r.get("odds", np.nan))
                        imp = 1.0 / odds if odds and odds > 0 else 0.0
                        self._features[name] = {
                            "odds_win": odds if pd.notna(odds) else 0.0,
                            "implied_prob_win": imp,
                            "horse_no": float(hno),
                        }
                        if hno > 0:
                            self._hno_to_name[hno] = name
                    print(f"[INFO] Loaded {len(self._features)} horses from {win_path}")
            except Exception as e:
                print(f"[ERROR] Failed to read {win_path}: {e}")

        # Place odds
        if not Path(place_path).exists():
            print(f"[WARN] Place odds file not found: {place_path}")
        else:
            try:
                df = pd.read_csv(place_path)
                required_cols = {"horse_no", "odds_mid"}
                missing = required_cols - set(df.columns)
                if missing:
                    print(f"[WARN] {place_path} missing columns: {missing}")
                else:
                    matched = 0
                    for _, r in df.iterrows():
                        hno = _pi(r.get("horse_no", 0))
                        name = self._hno_to_name.get(hno, "")
                        if not name or name not in self._features:
                            continue
                        mid = _pf(r.get("odds_mid", np.nan))
                        low = _pf(r.get("odds_low", np.nan))
                        if pd.notna(mid) and mid > 0:
                            self._features[name]["implied_prob_place"] = 1.0 / mid
                            self._features[name]["place_odds_mid"] = mid
                            matched += 1
                        elif pd.notna(low) and low > 0:
                            self._features[name]["implied_prob_place"] = 1.0 / low
                            self._features[name]["place_odds_mid"] = low
                            matched += 1
                    print(f"[INFO] Matched {matched} place odds from {place_path}")
            except Exception as e:
                print(f"[ERROR] Failed to read {place_path}: {e}")

        # Wide + Quinella -> network centrality
        wide_strength: Dict[int, float] = defaultdict(float)
        wide_count: Dict[int, int] = defaultdict(int)
        quinella_strength: Dict[int, float] = defaultdict(float)
        quinella_count: Dict[int, int] = defaultdict(int)

        if not Path(wide_path).exists():
            print(f"[WARN] Wide odds file not found: {wide_path}")
        else:
            try:
                df = pd.read_csv(wide_path)
                required_cols = {"horse_no_a", "horse_no_b", "odds_mid"}
                missing = required_cols - set(df.columns)
                if missing:
                    print(f"[WARN] {wide_path} missing columns: {missing}")
                else:
                    for _, r in df.iterrows():
                        a, b = _pi(r.get("horse_no_a", 0)), _pi(r.get("horse_no_b", 0))
                        mid = _pf(r.get("odds_mid", np.nan))
                        if a and b and pd.notna(mid) and mid > 0:
                            s = 1.0 / mid
                            wide_strength[a] += s
                            wide_strength[b] += s
                            wide_count[a] += 1
                            wide_count[b] += 1
                    print(f"[INFO] Loaded {len(df)} wide pairs from {wide_path}")
            except Exception as e:
                print(f"[ERROR] Failed to read {wide_path}: {e}")

        if not Path(quinella_path).exists():
            print(f"[WARN] Quinella odds file not found: {quinella_path}")
        else:
            try:
                df = pd.read_csv(quinella_path)
                required_cols = {"horse_no_a", "horse_no_b", "odds"}
                missing = required_cols - set(df.columns)
                if missing:
                    print(f"[WARN] {quinella_path} missing columns: {missing}")
                else:
                    for _, r in df.iterrows():
                        a, b = _pi(r.get("horse_no_a", 0)), _pi(r.get("horse_no_b", 0))
                        odds = _pf(r.get("odds", np.nan))
                        if a and b and pd.notna(odds) and odds > 0:
                            s = 1.0 / odds
                            quinella_strength[a] += s
                            quinella_strength[b] += s
                            quinella_count[a] += 1
                            quinella_count[b] += 1
                    print(f"[INFO] Loaded {len(df)} quinella pairs from {quinella_path}")
            except Exception as e:
                print(f"[ERROR] Failed to read {quinella_path}: {e}")

        # Compute network centrality per horse
        all_centrality = {}
        for hno in set(list(wide_strength.keys()) + list(quinella_strength.keys())):
            ws = wide_strength.get(hno, 0.0)
            qs = quinella_strength.get(hno, 0.0)
            all_centrality[hno] = ws + qs

        max_cent = max(all_centrality.values()) if all_centrality else 1.0
        if max_cent <= 0:
            max_cent = 1.0

        max_ws = max(wide_strength.values()) if wide_strength else 1.0
        max_qs = max(quinella_strength.values()) if quinella_strength else 1.0
        if max_ws <= 0:
            max_ws = 1.0
        if max_qs <= 0:
            max_qs = 1.0

        for hno in all_centrality:
            name = self._hno_to_name.get(hno, "")
            if not name or name not in self._features:
                continue
            wc = wide_count.get(hno, 0)
            qc = quinella_count.get(hno, 0)
            self._features[name]["wide_network_strength"] = (
                (wide_strength.get(hno, 0.0) / wc) if wc > 0 else 0.0
            )
            self._features[name]["quinella_network_strength"] = (
                (quinella_strength.get(hno, 0.0) / qc) if qc > 0 else 0.0
            )
            self._features[name]["market_centrality"] = all_centrality[hno] / max_cent

        # Fill missing place/combo features
        for name, feat in self._features.items():
            if "implied_prob_place" not in feat:
                feat["implied_prob_place"] = min(0.8, feat.get("implied_prob_win", 0.0) * 1.6)
                feat["place_odds_mid"] = 0.0
            if "wide_network_strength" not in feat:
                feat["wide_network_strength"] = 0.0
            if "quinella_network_strength" not in feat:
                feat["quinella_network_strength"] = 0.0
            if "market_centrality" not in feat:
                feat["market_centrality"] = 0.0

    def get_features(self, horse_name: str) -> Dict[str, float]:
        key = _normalize_name(horse_name)
        default = {
            "odds_win": 0.0,
            "implied_prob_win": 0.0,
            "implied_prob_place": 0.0,
            "place_odds_mid": 0.0,
            "wide_network_strength": 0.0,
            "quinella_network_strength": 0.0,
            "market_centrality": 0.0,
            "horse_no": 0.0,
        }
        return self._features.get(key, default)


# ---------------------------------------------------------------------------
# Feature Engine
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # TimeIndex core (7)
    "ti_last", "ti_mean3", "ti_mean5", "ti_max5", "ti_std5", "ti_trend", "ti_decay",
    # Adjusted TI (4)
    "adj_ti_last", "adj_ti_mean5", "adj_ti_max5", "adj_ti_trend",
    # Baba interaction (5)
    "baba_pref_mean", "baba_pref_std", "baba_distance", "baba_cat_top3_rate", "baba_adj_trend",
    # Running style (5)
    "run_early_ratio", "run_position_gain", "run_final_stretch", "run_final_best", "run_style_cat",
    # Jockey (4)
    "jockey_win_global", "jockey_win_course", "jockey_top3_course", "jockey_synergy",
    # Weight momentum (5)
    "weight_last", "weight_change_last", "weight_change_mean3", "weight_change_trend", "weight_opt_dev",
    # Class transition (4)
    "prize_mean_r3", "prize_mean_prior", "class_direction", "class_level",
    # Aptitude (7)
    "ti_surface_avg", "ti_dist_avg", "ti_venue_avg", "ti_cond_avg",
    "dist_diff_optimal", "surface_exp_ratio", "dist_exp_count",
    # Form (7)
    "top3_rate_r5", "top3_rate_all", "win_rate_r5", "avg_finish5",
    "best_finish5", "finish_std5", "top3_streak",
    # Market (4) - only features available in both training and inference
    # wide_network_strength, quinella_network_strength, market_centrality
    # are excluded from ML features (always 0 in training, non-zero in inference)
    # but still computed and stored in output for downstream use
    "odds_win", "implied_prob_win", "implied_prob_place",
    "popularity_pct",
    # Context (4)
    "field_size", "draw", "weight_carried", "rest_days",
    # Horse attributes (3)
    "age", "is_female", "is_gelding",
]


class FeatureEngine:
    def __init__(self, config: Config):
        self.config = config
        self._global_jockey_win_rate = 0.0
        self._global_jockey_top3_rate = 0.0
        self._global_ti_mean = 0.0
        self._jockey_stats: Dict[str, Dict[str, float]] = {}
        self._jockey_course_stats: Dict[str, Dict[str, float]] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["TimeIndex"] = df["ﾀｲﾑ指数"].apply(_pf) if "ﾀｲﾑ指数" in df.columns else np.nan
        df["BabaIndex"] = df["馬場指数"].apply(lambda x: _pf(x) if str(x).strip() not in ("", "**") else 0.0) if "馬場指数" in df.columns else 0.0
        df["AdjTimeIndex"] = df["TimeIndex"] - df["BabaIndex"]
        df["Rank"] = df["着順"].apply(lambda x: _pi(x) if str(x).strip() not in ("", "0") else 999) if "着順" in df.columns else 999
        df["Date"] = pd.to_datetime(df["日付"], errors="coerce") if "日付" in df.columns else pd.NaT
        df["Location"] = df["開催"].apply(_parse_venue) if "開催" in df.columns else ""
        if "距離" in df.columns:
            parsed = df["距離"].apply(lambda x: pd.Series(_parse_surface_distance(x)))
            df["Surface"] = parsed[0]
            df["Distance"] = parsed[1]
        else:
            df["Surface"] = ""
            df["Distance"] = np.nan
        if "SexAge" in df.columns:
            df["Sex"] = df["SexAge"].astype(str).str[0]
            df["Age"] = df["SexAge"].astype(str).str[1:].apply(_pf)
        else:
            df["Sex"] = ""
            df["Age"] = np.nan
        df["IsWin"] = (df["Rank"] == 1).astype(int)
        df["IsTop3"] = (df["Rank"] <= 3).astype(int)
        if "馬番" in df.columns:
            df["horse_no"] = df["馬番"].apply(_pi)
        if "オッズ" in df.columns:
            df["Odds"] = df["オッズ"].apply(_pf)
        if "人気" in df.columns:
            df["Popularity"] = df["人気"].apply(_pi)
        if "頭数" in df.columns:
            df["FieldSize"] = df["頭数"].apply(_pi)
        if "枠番" in df.columns:
            df["Draw"] = df["枠番"].apply(_pi)
        if "斤量" in df.columns:
            df["WeightCarried"] = df["斤量"].apply(_pf)
        if "上り" in df.columns:
            df["FinalStretch"] = df["上り"].apply(_pf)
        if "馬体重" in df.columns:
            wt = df["馬体重"].apply(lambda x: pd.Series(_parse_weight_change(x)))
            df["HorseWeight"] = wt[0]
            df["HorseWeightChange"] = wt[1]
        if "賞金" in df.columns:
            df["Prize"] = df["賞金"].apply(_pf)
        if "通過" in df.columns:
            df["Positions"] = df["通過"].apply(_parse_positions)
        else:
            df["Positions"] = [[] for _ in range(len(df))]
        if "JockeyId" in df.columns:
            df["JockeyId"] = df["JockeyId"].fillna("0").astype(str)
        else:
            df["JockeyId"] = "0"
        if "JockeyId_current" in df.columns:
            df["JockeyId_current"] = df["JockeyId_current"].fillna("0").astype(str)
        if "HorseName" in df.columns:
            df["HorseName"] = df["HorseName"].astype(str).str.strip()
        df = df.sort_values(["HorseName", "Date"])
        return df

    def fit_global_stats(self, history_df: pd.DataFrame, cutoff_date: Optional[pd.Timestamp] = None) -> None:
        """Compute global priors for Bayesian smoothing.

        When cutoff_date is provided, only use data BEFORE that date to prevent
        data leakage during training. For inference, cutoff_date=None uses all data.
        """
        valid = history_df[history_df["Rank"] < 900]
        if cutoff_date is not None and "Date" in valid.columns:
            valid = valid[valid["Date"] < cutoff_date]
        if valid.empty:
            return
        self._global_ti_mean = float(valid["TimeIndex"].dropna().mean()) if valid["TimeIndex"].notna().any() else 0.0
        self._global_top3_rate = float(valid["IsTop3"].mean()) if len(valid) > 0 else 0.3
        # Jockey global stats
        jockey_races = valid.groupby("JockeyId").agg(
            wins=("IsWin", "sum"),
            top3s=("IsTop3", "sum"),
            total=("IsWin", "count"),
        )
        self._global_jockey_win_rate = float(jockey_races["wins"].sum() / max(jockey_races["total"].sum(), 1))
        self._global_jockey_top3_rate = float(jockey_races["top3s"].sum() / max(jockey_races["total"].sum(), 1))
        for jid, row in jockey_races.iterrows():
            self._jockey_stats[str(jid)] = {
                "wins": float(row["wins"]),
                "top3s": float(row["top3s"]),
                "total": float(row["total"]),
            }
        # Jockey-course stats
        jc = valid.groupby(["JockeyId", "Location"]).agg(
            wins=("IsWin", "sum"),
            top3s=("IsTop3", "sum"),
            total=("IsWin", "count"),
        )
        for (jid, loc), row in jc.iterrows():
            key = f"{jid}_{loc}"
            self._jockey_course_stats[key] = {
                "wins": float(row["wins"]),
                "top3s": float(row["top3s"]),
                "total": float(row["total"]),
            }

    def _precompute_cumulative_jockey_stats(self, history_df: pd.DataFrame) -> Dict[str, Any]:
        """Pre-compute expanding cumulative jockey stats by date.

        Returns a dict keyed by date (pd.Timestamp) containing snapshots of
        jockey statistics using ONLY data strictly before that date.
        This eliminates data leakage during training.
        """
        valid = history_df[history_df["Rank"] < 900].copy()
        if "Date" not in valid.columns or valid.empty:
            return {}

        valid = valid.sort_values("Date")
        dates = sorted(valid["Date"].dropna().unique())

        # Running cumulative counters
        cum_jockey: Dict[str, Dict[str, float]] = defaultdict(lambda: {"wins": 0.0, "top3s": 0.0, "total": 0.0})
        cum_jc: Dict[str, Dict[str, float]] = defaultdict(lambda: {"wins": 0.0, "top3s": 0.0, "total": 0.0})
        cum_total_wins = 0.0
        cum_total_top3s = 0.0
        cum_total_races = 0.0
        cum_ti_sum = 0.0
        cum_ti_count = 0
        cum_top3_sum = 0.0
        cum_top3_count = 0

        snapshots: Dict = {}
        prev_date = None

        for date in dates:
            # Snapshot BEFORE this date's races
            global_wr = cum_total_wins / max(cum_total_races, 1)
            global_t3r = cum_total_top3s / max(cum_total_races, 1)
            global_ti = cum_ti_sum / max(cum_ti_count, 1)
            global_top3_rate = cum_top3_sum / max(cum_top3_count, 1) if cum_top3_count > 0 else 0.3

            snapshots[date] = {
                "jockey": {k: dict(v) for k, v in cum_jockey.items()},
                "jockey_course": {k: dict(v) for k, v in cum_jc.items()},
                "global_jockey_win_rate": global_wr,
                "global_jockey_top3_rate": global_t3r,
                "global_ti_mean": global_ti,
                "global_top3_rate": global_top3_rate,
            }

            # Add this date's races to cumulative counters
            day_races = valid[valid["Date"] == date]
            for _, row in day_races.iterrows():
                jid = str(row.get("JockeyId", "0") or "0")
                loc = str(row.get("Location", "") or "")
                is_win = int(row.get("IsWin", 0))
                is_top3 = int(row.get("IsTop3", 0))
                ti = row.get("TimeIndex", np.nan)

                cum_jockey[jid]["wins"] += is_win
                cum_jockey[jid]["top3s"] += is_top3
                cum_jockey[jid]["total"] += 1

                jc_key = f"{jid}_{loc}"
                cum_jc[jc_key]["wins"] += is_win
                cum_jc[jc_key]["top3s"] += is_top3
                cum_jc[jc_key]["total"] += 1

                cum_total_wins += is_win
                cum_total_top3s += is_top3
                cum_total_races += 1
                cum_top3_sum += is_top3
                cum_top3_count += 1

                if pd.notna(ti):
                    cum_ti_sum += ti
                    cum_ti_count += 1

        return snapshots

    def _apply_jockey_snapshot(self, snapshot: Dict) -> None:
        """Temporarily replace global stats with a point-in-time snapshot."""
        self._jockey_stats = snapshot["jockey"]
        self._jockey_course_stats = snapshot["jockey_course"]
        self._global_jockey_win_rate = snapshot["global_jockey_win_rate"]
        self._global_jockey_top3_rate = snapshot["global_jockey_top3_rate"]
        self._global_ti_mean = snapshot["global_ti_mean"]
        self._global_top3_rate = snapshot["global_top3_rate"]

    def compute_horse_features(
        self,
        history: pd.DataFrame,
        target_surface: str,
        target_distance: int,
        target_location: str,
        target_baba: float,
        target_condition: str,
        target_date: Optional[pd.Timestamp] = None,
        jockey_id: str = "0",
        field_size: int = 0,
        draw: int = 0,
        weight_carried: float = 0.0,
        sex: str = "",
        age: float = 0.0,
        odds_feat: Optional[Dict[str, float]] = None,
        popularity: int = 0,
    ) -> Dict[str, float]:
        """Compute all features for one horse given its history rows."""
        feat: Dict[str, float] = {}

        # Handle empty history
        if history.empty or "Date" not in history.columns:
            for col in FEATURE_COLUMNS:
                feat[col] = 0.0
            feat["age"] = age if pd.notna(age) else 3.0
            feat["is_female"] = 1.0 if sex == "牝" else 0.0
            feat["is_gelding"] = 1.0 if sex == "セ" else 0.0
            feat["field_size"] = float(field_size)
            feat["draw"] = float(draw)
            feat["weight_carried"] = weight_carried
            feat["popularity_pct"] = popularity / max(field_size, 1) if popularity > 0 else 0.5
            if odds_feat:
                for k in ["odds_win", "implied_prob_win", "implied_prob_place",
                           "wide_network_strength", "quinella_network_strength", "market_centrality"]:
                    feat[k] = odds_feat.get(k, 0.0)
            feat["rest_days"] = 60.0
            feat["avg_finish5"] = 8.0
            feat["best_finish5"] = 8.0
            feat["finish_std5"] = 3.0
            feat["run_early_ratio"] = 0.5
            feat["run_final_stretch"] = 35.0
            feat["run_final_best"] = 35.0
            feat["run_style_cat"] = 1.0
            feat["weight_last"] = 460.0
            return feat

        # Sort by date (oldest first, most recent last)
        hist = history.sort_values("Date")
        valid = hist[hist["Rank"] < 900]
        n = len(valid)

        # ---- TimeIndex core ----
        ti_series = valid["TimeIndex"].dropna()
        ti_vals = ti_series.tolist()
        adj_series = valid["AdjTimeIndex"].dropna()
        adj_vals = adj_series.tolist()

        if ti_vals:
            r5 = ti_vals[-5:] if len(ti_vals) >= 5 else ti_vals
            r3 = ti_vals[-3:] if len(ti_vals) >= 3 else ti_vals
            feat["ti_last"] = ti_vals[-1]
            feat["ti_mean3"] = float(np.mean(r3))
            feat["ti_mean5"] = float(np.mean(r5))
            feat["ti_max5"] = float(np.max(r5))
            feat["ti_std5"] = float(np.std(r5)) if len(r5) > 1 else 0.0
            feat["ti_trend"] = ti_vals[-1] - float(np.mean(r5))
            # Exponential decay mean: most recent (last element) gets highest weight
            weights = [0.5 ** ((len(r5) - 1 - i) / 2.5) for i in range(len(r5))]
            feat["ti_decay"] = float(np.average(r5, weights=weights))
        else:
            for k in ["ti_last", "ti_mean3", "ti_mean5", "ti_max5", "ti_std5", "ti_trend", "ti_decay"]:
                feat[k] = 0.0

        if adj_vals:
            ar5 = adj_vals[-5:] if len(adj_vals) >= 5 else adj_vals
            feat["adj_ti_last"] = adj_vals[-1]
            feat["adj_ti_mean5"] = float(np.mean(ar5))
            feat["adj_ti_max5"] = float(np.max(ar5))
            feat["adj_ti_trend"] = adj_vals[-1] - float(np.mean(ar5))
        else:
            for k in ["adj_ti_last", "adj_ti_mean5", "adj_ti_max5", "adj_ti_trend"]:
                feat[k] = 0.0

        # ---- Baba interaction ----
        if n > 0 and valid["BabaIndex"].notna().any():
            baba_vals = valid["BabaIndex"].dropna().tolist()
            # Preferred baba range from best performances
            top3_mask = valid["IsTop3"] == 1
            top3_baba = valid.loc[top3_mask, "BabaIndex"].dropna().tolist()
            if top3_baba:
                feat["baba_pref_mean"] = float(np.mean(top3_baba))
                feat["baba_pref_std"] = float(np.std(top3_baba)) if len(top3_baba) > 1 else 10.0
            else:
                feat["baba_pref_mean"] = float(np.mean(baba_vals)) if baba_vals else 0.0
                feat["baba_pref_std"] = 10.0
            feat["baba_distance"] = abs(target_baba - feat["baba_pref_mean"])

            # Baba category aptitude: fast(<-5), standard(-5~5), slow(>5)
            def baba_cat(b):
                if b < -5:
                    return "fast"
                elif b > 5:
                    return "slow"
                return "standard"

            target_cat = baba_cat(target_baba)
            cat_mask = valid["BabaIndex"].apply(lambda b: baba_cat(b) == target_cat)
            cat_races = valid[cat_mask]
            cat_n = len(cat_races)
            cat_top3 = int(cat_races["IsTop3"].sum()) if cat_n > 0 else 0
            smoother = BayesianSmoother(getattr(self, "_global_top3_rate", 0.3), 5)
            feat["baba_cat_top3_rate"] = smoother.smooth(cat_top3 / max(cat_n, 1), cat_n)

            # Baba-adjusted trend
            if len(adj_vals) >= 2:
                recent_adj = adj_vals[-3:] if len(adj_vals) >= 3 else adj_vals[-2:]
                prior_adj = adj_vals[-6:-3] if len(adj_vals) >= 6 else adj_vals[:-len(recent_adj)]
                if prior_adj:
                    feat["baba_adj_trend"] = float(np.mean(recent_adj)) - float(np.mean(prior_adj))
                else:
                    feat["baba_adj_trend"] = 0.0
            else:
                feat["baba_adj_trend"] = 0.0
        else:
            for k in ["baba_pref_mean", "baba_pref_std", "baba_distance", "baba_cat_top3_rate", "baba_adj_trend"]:
                feat[k] = 0.0

        # ---- Running style ----
        if n > 0:
            recent_pos = valid.tail(5)
            early_ratios = []
            gains = []
            for _, row in recent_pos.iterrows():
                pos_list = row.get("Positions", [])
                fs = row.get("FieldSize", 0) or 1
                if pos_list and fs > 0:
                    early_ratios.append(pos_list[0] / fs)
                    gains.append((pos_list[0] - pos_list[-1]) / fs)

            feat["run_early_ratio"] = float(np.mean(early_ratios)) if early_ratios else 0.5
            feat["run_position_gain"] = float(np.mean(gains)) if gains else 0.0

            fs_vals = recent_pos["FinalStretch"].dropna().tolist() if "FinalStretch" in recent_pos.columns else []
            feat["run_final_stretch"] = float(np.mean(fs_vals)) if fs_vals else 35.0
            feat["run_final_best"] = float(np.min(fs_vals)) if fs_vals else 35.0

            # Style category: 0=front, 1=stalker, 2=closer
            er = feat["run_early_ratio"]
            if er <= 0.3:
                feat["run_style_cat"] = 0.0
            elif er <= 0.55:
                feat["run_style_cat"] = 1.0
            else:
                feat["run_style_cat"] = 2.0
        else:
            feat["run_early_ratio"] = 0.5
            feat["run_position_gain"] = 0.0
            feat["run_final_stretch"] = 35.0
            feat["run_final_best"] = 35.0
            feat["run_style_cat"] = 1.0

        # ---- Jockey-course synergy ----
        win_smoother = BayesianSmoother(self._global_jockey_win_rate, 5)
        top3_smoother = BayesianSmoother(self._global_jockey_top3_rate, 5)

        jstats = self._jockey_stats.get(jockey_id, {"wins": 0, "top3s": 0, "total": 0})
        jn = int(jstats["total"])
        feat["jockey_win_global"] = win_smoother.smooth(jstats["wins"] / max(jn, 1), jn)
        jockey_top3_global = top3_smoother.smooth(jstats["top3s"] / max(jn, 1), jn)

        jc_key = f"{jockey_id}_{target_location}"
        jcstats = self._jockey_course_stats.get(jc_key, {"wins": 0, "top3s": 0, "total": 0})
        jcn = int(jcstats["total"])
        feat["jockey_win_course"] = win_smoother.smooth(jcstats["wins"] / max(jcn, 1), jcn)
        feat["jockey_top3_course"] = top3_smoother.smooth(jcstats["top3s"] / max(jcn, 1), jcn)
        feat["jockey_synergy"] = (
            feat["jockey_win_course"] / max(feat["jockey_win_global"], 0.01)
        )

        # ---- Weight momentum ----
        if n > 0 and "HorseWeight" in valid.columns:
            wt_vals = valid["HorseWeight"].dropna()
            wc_vals = valid["HorseWeightChange"].dropna() if "HorseWeightChange" in valid.columns else pd.Series(dtype=float)
            feat["weight_last"] = float(wt_vals.iloc[-1]) if len(wt_vals) > 0 else 460.0
            feat["weight_change_last"] = float(wc_vals.iloc[-1]) if len(wc_vals) > 0 else 0.0
            r3_wc = wc_vals.tail(3).tolist() if len(wc_vals) > 0 else [0.0]
            feat["weight_change_mean3"] = float(np.mean(r3_wc))
            feat["weight_change_trend"] = feat["weight_change_last"] - feat["weight_change_mean3"]
            # Optimal weight: weight at best TI performance
            if ti_vals and len(wt_vals) > 0:
                best_idx = valid["TimeIndex"].idxmax()
                if pd.notna(best_idx) and best_idx in valid.index:
                    best_weight = valid.loc[best_idx, "HorseWeight"]
                    if pd.notna(best_weight):
                        feat["weight_opt_dev"] = abs(feat["weight_last"] - float(best_weight))
                    else:
                        feat["weight_opt_dev"] = 0.0
                else:
                    feat["weight_opt_dev"] = 0.0
            else:
                feat["weight_opt_dev"] = 0.0
        else:
            feat["weight_last"] = 460.0
            feat["weight_change_last"] = 0.0
            feat["weight_change_mean3"] = 0.0
            feat["weight_change_trend"] = 0.0
            feat["weight_opt_dev"] = 0.0

        # ---- Class transition ----
        if n > 0 and "Prize" in valid.columns:
            prize_vals = valid["Prize"].dropna().tolist()
            if prize_vals:
                r3_prize = prize_vals[-3:] if len(prize_vals) >= 3 else prize_vals
                prior_prize = prize_vals[:-3] if len(prize_vals) > 3 else []
                feat["prize_mean_r3"] = float(np.mean(r3_prize))
                feat["prize_mean_prior"] = float(np.mean(prior_prize)) if prior_prize else feat["prize_mean_r3"]
                if feat["prize_mean_prior"] > 0:
                    feat["class_direction"] = (feat["prize_mean_r3"] - feat["prize_mean_prior"]) / feat["prize_mean_prior"]
                else:
                    feat["class_direction"] = 0.0
                feat["class_level"] = math.log1p(max(feat["prize_mean_r3"], 0.0))
            else:
                feat["prize_mean_r3"] = 0.0
                feat["prize_mean_prior"] = 0.0
                feat["class_direction"] = 0.0
                feat["class_level"] = 0.0
        else:
            feat["prize_mean_r3"] = 0.0
            feat["prize_mean_prior"] = 0.0
            feat["class_direction"] = 0.0
            feat["class_level"] = 0.0

        # ---- Aptitude ----
        ti_smoother = BayesianSmoother(self._global_ti_mean, 3)

        def ctx_avg(col_name, val, value_col="TimeIndex"):
            if n == 0:
                return self._global_ti_mean
            mask = valid[col_name] == val
            matched = valid.loc[mask, value_col].dropna()
            mn = len(matched)
            obs = float(matched.mean()) if mn > 0 else self._global_ti_mean
            return ti_smoother.smooth(obs, mn)

        feat["ti_surface_avg"] = ctx_avg("Surface", target_surface)
        feat["ti_dist_avg"] = ctx_avg("Distance", float(target_distance))
        feat["ti_venue_avg"] = ctx_avg("Location", target_location)

        # Condition aptitude
        def going_cat(cond):
            gl = GOING_MAP.get(str(cond).strip(), 1.0)
            return "heavy" if gl >= 1.5 else "light"

        target_going_cat = going_cat(target_condition)
        if n > 0 and "馬場" in valid.columns:
            cond_mask = valid["馬場"].apply(lambda c: going_cat(c) == target_going_cat)
            cond_ti = valid.loc[cond_mask, "TimeIndex"].dropna()
            cn = len(cond_ti)
            feat["ti_cond_avg"] = ti_smoother.smooth(float(cond_ti.mean()) if cn > 0 else self._global_ti_mean, cn)
        else:
            feat["ti_cond_avg"] = self._global_ti_mean

        # Distance optimality
        if n > 0:
            win_dists = valid.loc[valid["IsWin"] == 1, "Distance"].dropna()
            if len(win_dists) > 0:
                optimal_dist = float(win_dists.mean())
            else:
                optimal_dist = float(valid["Distance"].dropna().mean()) if valid["Distance"].notna().any() else target_distance
            feat["dist_diff_optimal"] = abs(target_distance - optimal_dist)
        else:
            feat["dist_diff_optimal"] = 0.0

        # Surface experience
        if n > 0:
            surf_count = int((valid["Surface"] == target_surface).sum())
            feat["surface_exp_ratio"] = surf_count / max(n, 1)
            dist_close = valid["Distance"].apply(lambda d: abs(d - target_distance) <= 200 if pd.notna(d) else False)
            feat["dist_exp_count"] = float(dist_close.sum())
        else:
            feat["surface_exp_ratio"] = 0.0
            feat["dist_exp_count"] = 0.0

        # ---- Form ----
        if n > 0:
            r5 = valid.tail(5)
            feat["top3_rate_r5"] = float(r5["IsTop3"].mean())
            feat["top3_rate_all"] = float(valid["IsTop3"].mean())
            feat["win_rate_r5"] = float(r5["IsWin"].mean())
            finishes = r5["Rank"].tolist()
            feat["avg_finish5"] = float(np.mean(finishes))
            feat["best_finish5"] = float(np.min(finishes))
            feat["finish_std5"] = float(np.std(finishes)) if len(finishes) > 1 else 0.0
            # Top3 streak (count consecutive top3 from most recent)
            streak = 0
            for v in reversed(valid["IsTop3"].tolist()):
                if v == 1:
                    streak += 1
                else:
                    break
            feat["top3_streak"] = float(streak)
        else:
            feat["top3_rate_r5"] = 0.0
            feat["top3_rate_all"] = 0.0
            feat["win_rate_r5"] = 0.0
            feat["avg_finish5"] = 8.0
            feat["best_finish5"] = 8.0
            feat["finish_std5"] = 3.0
            feat["top3_streak"] = 0.0

        # ---- Market ----
        if odds_feat:
            feat["odds_win"] = odds_feat.get("odds_win", 0.0)
            feat["implied_prob_win"] = odds_feat.get("implied_prob_win", 0.0)
            feat["implied_prob_place"] = odds_feat.get("implied_prob_place", 0.0)
            feat["wide_network_strength"] = odds_feat.get("wide_network_strength", 0.0)
            feat["quinella_network_strength"] = odds_feat.get("quinella_network_strength", 0.0)
            feat["market_centrality"] = odds_feat.get("market_centrality", 0.0)
        else:
            feat["odds_win"] = 0.0
            feat["implied_prob_win"] = 0.0
            feat["implied_prob_place"] = 0.0
            feat["wide_network_strength"] = 0.0
            feat["quinella_network_strength"] = 0.0
            feat["market_centrality"] = 0.0

        feat["popularity_pct"] = popularity / max(field_size, 1) if popularity > 0 else 0.5

        # ---- Context ----
        feat["field_size"] = float(field_size)
        feat["draw"] = float(draw)
        feat["weight_carried"] = weight_carried

        # Rest days: gap between target race date and last history race
        if n > 0:
            dates = valid["Date"].dropna()
            if len(dates) >= 1:
                last_date = dates.iloc[-1]
                if target_date is not None and pd.notna(target_date):
                    feat["rest_days"] = max(0.0, (target_date - last_date).days)
                elif len(dates) >= 2:
                    # Fallback: gap between last two races
                    feat["rest_days"] = max(0.0, (last_date - dates.iloc[-2]).days)
                else:
                    feat["rest_days"] = 60.0
            else:
                feat["rest_days"] = 60.0
        else:
            feat["rest_days"] = 60.0

        # ---- Horse attributes ----
        feat["age"] = age if pd.notna(age) else 3.0
        feat["is_female"] = 1.0 if sex == "牝" else 0.0
        feat["is_gelding"] = 1.0 if sex == "セ" else 0.0

        return feat

    def create_training_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Build features for training from kachiuma.csv data.

        Uses expanding cumulative jockey stats so that each training sample
        only sees jockey/global priors from BEFORE its race date. This
        eliminates data leakage in the global prior computation.
        """
        print("[INFO] Building training features...")
        print("[INFO] Pre-computing cumulative jockey stats by date...")
        jockey_snapshots = self._precompute_cumulative_jockey_stats(history_df)

        if "race_id" not in history_df.columns:
            history_df = history_df.copy()
            r_col = history_df["R"].astype(str) if "R" in history_df.columns else "0"
            history_df["race_id"] = history_df["Date"].astype(str) + "_" + history_df["Location"] + "_" + r_col

        rows_out = []
        grouped = history_df.groupby("race_id")

        for race_id, race_group in grouped:
            horses_in_race = race_group.groupby("HorseName")

            # Find the target race date for this race_id (for snapshot lookup)
            race_dates = race_group["Date"].dropna()
            if race_dates.empty:
                continue
            target_race_date = race_dates.max()  # most recent date in the group

            # Apply point-in-time jockey snapshot for this race date
            if target_race_date in jockey_snapshots:
                self._apply_jockey_snapshot(jockey_snapshots[target_race_date])

            for horse_name, horse_rows in horses_in_race:
                horse_rows = horse_rows.sort_values("Date")
                if len(horse_rows) < 2:
                    continue

                # Target = most recent row (the race being predicted)
                target_row = horse_rows.iloc[-1]  # last row after sort by Date asc
                hist_rows = horse_rows.iloc[:-1]  # prior rows are history

                rank = target_row.get("Rank", 999)
                if rank >= 900:
                    rank_raw = _pi(target_row.get("finish_pos", 0))
                    if rank_raw > 0:
                        rank = rank_raw
                    else:
                        continue

                is_top3 = 1 if rank <= 3 else 0

                surf, dist = target_row.get("Surface", "ダ"), target_row.get("Distance", 1800)
                loc = target_row.get("Location", "")
                baba = target_row.get("BabaIndex", 0.0)
                if pd.isna(baba):
                    baba = 0.0
                condition = str(history_df.loc[target_row.name, "馬場"] if "馬場" in history_df.columns else "稍重")
                jockey_id = str(target_row.get("JockeyId", "0"))
                field_size = int(target_row.get("FieldSize", 0) or 0)
                draw_val = int(target_row.get("Draw", 0) or 0)
                wt_carried = float(target_row.get("WeightCarried", 0) or 0)
                odds_val = float(target_row.get("Odds", 0) or 0)
                pop_val = int(target_row.get("Popularity", 0) or 0)

                # Build market features from the target row's odds
                odds_feat = {
                    "odds_win": odds_val if pd.notna(odds_val) else 0.0,
                    "implied_prob_win": 1.0 / odds_val if odds_val and odds_val > 0 else 0.0,
                    "implied_prob_place": min(0.8, (1.0 / odds_val * 1.6) if odds_val and odds_val > 0 else 0.0),
                    "wide_network_strength": 0.0,
                    "quinella_network_strength": 0.0,
                    "market_centrality": 0.0,
                }

                feat = self.compute_horse_features(
                    history=hist_rows,
                    target_surface=surf if surf else "ダ",
                    target_distance=int(dist) if pd.notna(dist) else 1800,
                    target_location=loc if loc else "",
                    target_baba=baba,
                    target_condition=condition,
                    target_date=target_row.get("Date", None),
                    jockey_id=jockey_id,
                    field_size=field_size,
                    draw=draw_val,
                    weight_carried=wt_carried,
                    sex=str(target_row.get("Sex", "")),
                    age=float(target_row.get("Age", 3) or 3),
                    odds_feat=odds_feat,
                    popularity=pop_val,
                )
                feat["y"] = float(is_top3)
                feat["rank_label"] = float(rank)
                feat["race_id"] = str(race_id)
                feat["HorseName"] = str(horse_name)
                feat["Date"] = target_row.get("Date", pd.NaT)
                rows_out.append(feat)

        df = pd.DataFrame(rows_out)
        print(f"[INFO] Training samples: {len(df)}")
        return df

    def create_inference_features(
        self,
        shutuba_df: pd.DataFrame,
        odds_engine: OddsEngine,
        target_race_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Build features for inference on upcoming race.

        shutuba_df contains ONLY historical race data for each horse
        (no current race entry row). Current race info (horse_no, odds,
        jockey) comes from odds_engine and JockeyId_current column.

        Args:
            target_race_date: The scheduled date of the target race.
                Used for rest_days calculation. Defaults to today if None.
                For backtesting, pass the actual race date for reproducibility.
        """
        print("[INFO] Building inference features...")
        if target_race_date is None:
            target_race_date = pd.Timestamp.now()

        target_horses = shutuba_df["HorseName"].unique()
        field_size = len(target_horses)

        # Derive popularity ranking from odds (lower odds = more popular)
        odds_ranking: Dict[str, int] = {}
        odds_list = []
        for h in target_horses:
            of = odds_engine.get_features(str(h).strip())
            odds_list.append((str(h).strip(), of.get("odds_win", 9999.0) or 9999.0))
        odds_list.sort(key=lambda x: x[1])
        for rank, (name, _) in enumerate(odds_list, 1):
            odds_ranking[name] = rank

        shutuba_grouped = shutuba_df.groupby("HorseName")

        rows_out = []
        for horse_name in target_horses:
            horse_name = str(horse_name).strip()
            if not horse_name:
                continue

            # ALL shutuba rows for this horse are history
            try:
                horse_hist = shutuba_grouped.get_group(horse_name)
            except KeyError:
                horse_hist = pd.DataFrame()

            # Resolve current jockey from JockeyId_current (same across all rows)
            jockey_id = "0"
            if not horse_hist.empty and "JockeyId_current" in horse_hist.columns:
                jid_cur = horse_hist["JockeyId_current"].dropna()
                if len(jid_cur) > 0:
                    jockey_id = str(jid_cur.iloc[0])
            if jockey_id == "0" and not horse_hist.empty and "JockeyId" in horse_hist.columns:
                jid = horse_hist["JockeyId"].dropna()
                if len(jid) > 0:
                    jockey_id = str(jid.iloc[-1])  # most recent jockey

            # Sex/Age from most recent history row
            sex = ""
            age = 3.0
            if not horse_hist.empty:
                latest = horse_hist.sort_values("Date").iloc[-1] if "Date" in horse_hist.columns else horse_hist.iloc[0]
                sex = str(latest.get("Sex", ""))
                age_val = latest.get("Age", 3)
                age = float(age_val) if pd.notna(age_val) else 3.0

            odds_feat = odds_engine.get_features(horse_name)
            pop_val = odds_ranking.get(horse_name, field_size // 2)

            feat = self.compute_horse_features(
                history=horse_hist if not horse_hist.empty else pd.DataFrame(),
                target_surface=self.config.target_surface,
                target_distance=self.config.target_distance,
                target_location=self.config.target_location,
                target_baba=self.config.expected_baba_index,
                target_condition=self.config.target_condition,
                target_date=target_race_date,
                jockey_id=jockey_id,
                field_size=field_size,
                draw=0,  # unknown for upcoming race
                weight_carried=0.0,  # unknown for upcoming race
                sex=sex,
                age=age,
                odds_feat=odds_feat,
                popularity=pop_val,
            )
            feat["HorseName"] = horse_name
            feat["horse_no"] = int(odds_feat.get("horse_no", 0) or 0)
            feat["Odds"] = odds_feat.get("odds_win", 0.0)
            rows_out.append(feat)

        df = pd.DataFrame(rows_out)
        print(f"[INFO] Inference horses: {len(df)}")
        return df


# ---------------------------------------------------------------------------
# Time-Series CV
# ---------------------------------------------------------------------------
class TimeSeriesCV:
    def __init__(self, n_splits: int = 5, min_train_frac: float = 0.40):
        self.n_splits = n_splits
        self.min_train_frac = min_train_frac

    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Expanding-window time-series split by Date."""
        if "Date" not in df.columns:
            # Fallback: simple sequential split
            n = len(df)
            fold_size = n // (self.n_splits + 1)
            folds = []
            for i in range(self.n_splits):
                train_end = int(n * (self.min_train_frac + (1.0 - self.min_train_frac) * i / self.n_splits))
                test_start = train_end
                test_end = min(test_start + fold_size, n)
                if test_end <= test_start:
                    continue
                folds.append((np.arange(0, train_end), np.arange(test_start, test_end)))
            return folds

        dates = df["Date"].copy()
        sorted_dates = dates.sort_values().unique()
        n_dates = len(sorted_dates)

        folds = []
        min_train = max(int(n_dates * self.min_train_frac), 10)

        for i in range(self.n_splits):
            train_end_idx = min_train + int((n_dates - min_train) * i / self.n_splits)
            test_start_idx = train_end_idx
            test_end_idx = min(train_end_idx + max(int(n_dates * 0.12), 5), n_dates)

            if test_start_idx >= n_dates or test_end_idx <= test_start_idx:
                continue

            train_dates = set(sorted_dates[:train_end_idx])
            test_dates = set(sorted_dates[test_start_idx:test_end_idx])

            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(test_dates)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                folds.append((train_idx, test_idx))

        return folds


# ---------------------------------------------------------------------------
# Platt Calibrator
# ---------------------------------------------------------------------------
class PlattCalibrator:
    def __init__(self):
        self._lr: Optional[LogisticRegression] = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        if len(probs) < 30 or len(np.unique(labels)) < 2:
            return
        # Logit transform
        eps = 1e-7
        p_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
        self._lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        self._lr.fit(logits, labels)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self._lr is None:
            return probs
        eps = 1e-7
        p_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
        return self._lr.predict_proba(logits)[:, 1]

    @property
    def is_fitted(self) -> bool:
        return self._lr is not None


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------
class StackingEnsemble:
    def __init__(self):
        self.classifiers: List[lgb.LGBMClassifier] = []
        self.ranker: Optional[lgb.LGBMRanker] = None
        self.meta_learner: Optional[LogisticRegression] = None
        self.calibrator = PlattCalibrator()
        self._features = FEATURE_COLUMNS

    @staticmethod
    def _build_clf_a() -> lgb.LGBMClassifier:
        """Deep trees, low regularization."""
        return lgb.LGBMClassifier(
            objective="binary", metric="auc",
            n_estimators=1200, learning_rate=0.02,
            num_leaves=24, min_child_samples=25,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )

    @staticmethod
    def _build_clf_b() -> lgb.LGBMClassifier:
        """Shallow trees, high regularization."""
        return lgb.LGBMClassifier(
            objective="binary", metric="auc",
            n_estimators=1200, learning_rate=0.03,
            num_leaves=16, max_depth=6, min_child_samples=40,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=123, n_jobs=-1, verbose=-1,
        )

    @staticmethod
    def _build_clf_c() -> lgb.LGBMClassifier:
        """Medium depth, dart boosting for diversity."""
        return lgb.LGBMClassifier(
            objective="binary", metric="auc",
            n_estimators=800, learning_rate=0.025,
            num_leaves=20, min_child_samples=30,
            subsample=0.75, colsample_bytree=0.65,
            reg_alpha=0.3, reg_lambda=1.5,
            boosting_type="dart", drop_rate=0.1,
            random_state=7, n_jobs=-1, verbose=-1,
        )

    @staticmethod
    def _build_ranker() -> lgb.LGBMRanker:
        return lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )

    @staticmethod
    def _relevance_labels(ranks: pd.Series) -> pd.Series:
        """top5_gain: 1st->5, 2nd->4, ..., 5th->1, else->0"""
        return ranks.apply(lambda r: max(0, 6 - int(r)) if r <= 5 else 0)

    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        print("[INFO] Training Stacking Ensemble...")
        features = [f for f in self._features if f in train_df.columns]
        dropped = set(self._features) - set(features)
        if dropped:
            print(f"[WARN] Dropped features not in training data: {dropped}")
        self._features = features

        X_all = train_df[features].values
        y_all = train_df["y"].values
        n = len(train_df)

        # Phase 1: OOF predictions via time-series CV
        cv = TimeSeriesCV(n_splits=5)
        folds = cv.split(train_df)

        oof_preds = np.zeros((n, 4))  # 3 classifiers + 1 ranker
        oof_mask = np.zeros(n, dtype=bool)

        cv_aucs = []
        for fold_idx, (tr_idx, te_idx) in enumerate(folds):
            X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
            X_te, y_te = X_all[te_idx], y_all[te_idx]

            if len(np.unique(y_tr)) < 2:
                continue

            # Train 3 classifiers
            for j, builder in enumerate([self._build_clf_a, self._build_clf_b, self._build_clf_c]):
                clf = builder()
                clf.fit(X_tr, y_tr)
                oof_preds[te_idx, j] = clf.predict_proba(X_te)[:, 1]

            # Train ranker
            if "race_id" in train_df.columns:
                tr_races = train_df.iloc[tr_idx].copy().sort_values("race_id")
                tr_qids = tr_races.groupby("race_id", sort=False).size().to_numpy()
                rank_labels = self._relevance_labels(tr_races["rank_label"])
                ranker = self._build_ranker()
                try:
                    ranker.fit(tr_races[features].values, rank_labels.values, group=tr_qids)
                    raw_rank = ranker.predict(X_te)
                    # Sigmoid normalize ranker output
                    mu, sigma = raw_rank.mean(), max(raw_rank.std(), 1e-6)
                    oof_preds[te_idx, 3] = 1.0 / (1.0 + np.exp(-(raw_rank - mu) / sigma))
                except Exception:
                    oof_preds[te_idx, 3] = 0.5

            oof_mask[te_idx] = True

            try:
                auc = roc_auc_score(y_te, oof_preds[te_idx, 0])
                cv_aucs.append(auc)
                print(f"  Fold {fold_idx + 1}: AUC={auc:.4f} (n_train={len(tr_idx)}, n_test={len(te_idx)})")
            except Exception:
                pass

        # Phase 2: Train meta-learner on OOF
        oof_X = oof_preds[oof_mask]
        oof_y = y_all[oof_mask]

        if len(oof_X) >= 30 and len(np.unique(oof_y)) >= 2:
            self.meta_learner = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            self.meta_learner.fit(oof_X, oof_y)
            oof_stacked = self.meta_learner.predict_proba(oof_X)[:, 1]
            print(f"  Meta-learner AUC: {roc_auc_score(oof_y, oof_stacked):.4f}")

            # Phase 3: Calibrate
            self.calibrator.fit(oof_stacked, oof_y)
        else:
            print("[WARN] Insufficient OOF samples for meta-learner, falling back to simple average.")

        # Phase 4: Retrain all base models on FULL data
        print("[INFO] Retraining on full data...")
        self.classifiers = []
        for builder in [self._build_clf_a, self._build_clf_b, self._build_clf_c]:
            clf = builder()
            clf.fit(X_all, y_all)
            self.classifiers.append(clf)

        # Retrain ranker
        if "race_id" in train_df.columns:
            full_sorted = train_df.sort_values("race_id")
            qids = full_sorted.groupby("race_id", sort=False).size().to_numpy()
            rank_labels = self._relevance_labels(full_sorted["rank_label"])
            self.ranker = self._build_ranker()
            try:
                self.ranker.fit(full_sorted[features].values, rank_labels.values, group=qids)
            except Exception as e:
                print(f"[WARN] Ranker training failed: {e}")
                self.ranker = None

        mean_auc = float(np.mean(cv_aucs)) if cv_aucs else 0.0
        print(f"[INFO] Mean CV AUC: {mean_auc:.4f}")
        return {"mean_auc": mean_auc, "n_folds": len(folds), "n_train": n}

    def predict(self, inference_df: pd.DataFrame) -> pd.DataFrame:
        if inference_df.empty:
            return pd.DataFrame()

        features = self._features
        X = inference_df[features].fillna(0).values
        n = len(X)

        # Base predictions
        base_preds = np.zeros((n, 4))
        for j, clf in enumerate(self.classifiers):
            base_preds[:, j] = clf.predict_proba(X)[:, 1]

        if self.ranker is not None:
            raw_rank = self.ranker.predict(X)
            mu, sigma = raw_rank.mean(), max(raw_rank.std(), 1e-6)
            base_preds[:, 3] = 1.0 / (1.0 + np.exp(-(raw_rank - mu) / sigma))
        else:
            base_preds[:, 3] = base_preds[:, :3].mean(axis=1)

        # Meta-learner
        if self.meta_learner is not None:
            stacked_prob = self.meta_learner.predict_proba(base_preds)[:, 1]
        else:
            # Simple weighted average as fallback
            stacked_prob = 0.3 * base_preds[:, 0] + 0.3 * base_preds[:, 1] + 0.2 * base_preds[:, 2] + 0.2 * base_preds[:, 3]

        # Calibrate
        calibrated_prob = self.calibrator.transform(stacked_prob)

        # Race-level normalization
        total = calibrated_prob.sum()
        if total > 0:
            normalized = calibrated_prob * min(3.0, float(n)) / total
            normalized = np.clip(normalized, 0.0, 0.98)
        else:
            normalized = calibrated_prob

        # Build results
        results = inference_df.copy()
        results["Top3Prob_model"] = normalized
        results["Top3Prob"] = calibrated_prob
        results["Top3Prob_raw_lr"] = base_preds[:, 0]
        results["Top3Prob_raw_lgb"] = base_preds[:, 1]
        results["Top3Prob_cal_lr"] = base_preds[:, 2]
        results["Top3Prob_cal_lgb"] = stacked_prob
        results["Top3Prob_lr"] = base_preds[:, 0]
        results["Top3Prob_lgbm"] = base_preds[:, 1]

        # Rank score: blend of calibrated prob + ranker signal
        if self.ranker is not None:
            raw_rank = self.ranker.predict(X)
            mu, sigma = raw_rank.mean(), max(raw_rank.std(), 1e-6)
            rank_norm = (raw_rank - raw_rank.min()) / max(raw_rank.max() - raw_rank.min(), 1e-6)
            results["rank_score"] = 0.6 * normalized + 0.4 * rank_norm
            results["rank_score_raw"] = raw_rank
        else:
            results["rank_score"] = normalized
            results["rank_score_raw"] = normalized

        # Normalize rank_score to [0, 1]
        rs = results["rank_score"].values
        rs_min, rs_max = rs.min(), rs.max()
        results["rank_score_norm"] = (rs - rs_min) / max(rs_max - rs_min, 1e-6)

        # Sort by rank_score
        results = results.sort_values("rank_score", ascending=False).reset_index(drop=True)

        # Confidence / stability / validity / consistency
        scores = results["rank_score"].values
        if len(scores) >= 3:
            gap = float(scores[0]) - float(scores[2])
            confidence = min(gap / 0.18, 1.0)
            confidence = max(0.0, confidence)
        else:
            confidence = 0.0

        # Stability: based on agreement among base models on top-3 picks
        # Each model's top-3 by its own ranking, then measure overlap
        if len(results) >= 3:
            top3_sets = []
            for j in range(min(4, base_preds.shape[1])):
                model_ranking = np.argsort(-base_preds[:, j])
                top3_sets.append(set(model_ranking[:3].tolist()))
            intersection = set.intersection(*top3_sets) if top3_sets else set()
            union = set.union(*top3_sets) if top3_sets else {0, 1, 2}
            stability = len(intersection) / max(len(union), 1)
        else:
            stability = 0.5

        validity = min(1.0, confidence * 0.7 + stability * 0.3)
        consistency = stability

        results["model_mode"] = "v5_stacking"
        # Top3Prob_model is race-level normalized (sums to ~3, not 1).
        # It is a relative allocation score, not a strict probability.
        # Top3Prob (uncalibrated) is closer to a true probability.
        results["score_is_probability"] = 0
        results["confidence_score"] = round(confidence, 4)
        results["stability_score"] = round(stability, 4)
        results["validity_score"] = round(validity, 4)
        results["consistency_score"] = round(consistency, 4)
        results["rank_ema"] = 0.5
        results["ev_ema"] = 0.5
        results["risk_score"] = round(max(0.0, 1.0 - confidence), 4)

        # horse_key for pipeline
        results["horse_key"] = results.apply(
            lambda r: str(int(r["horse_no"])) if pd.notna(r.get("horse_no")) and r["horse_no"] > 0 else str(r.get("HorseName", "")), axis=1
        )
        results["race_id"] = "current"

        return results


# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------
def main():
    print("=" * 68)
    print("  PREDICTOR V5 - Ultimate Stacking Ensemble")
    print("  Multi-model stacking + 4-odds market fusion + deep features")
    print("=" * 68)

    config = Config.from_env()

    # Load data
    if not Path("kachiuma.csv").exists() or not Path("shutuba.csv").exists():
        print("[ERROR] kachiuma.csv or shutuba.csv not found.")
        return

    history_raw = pd.read_csv("kachiuma.csv")
    shutuba_raw = pd.read_csv("shutuba.csv")
    print(f"[INFO] kachiuma: {len(history_raw)} rows, shutuba: {len(shutuba_raw)} rows")

    # Feature engine
    engine = FeatureEngine(config)
    history_df = engine.preprocess(history_raw)
    shutuba_df = engine.preprocess(shutuba_raw)

    # ----------------------------------------------------------------
    # Resolve target race context from a SINGLE source of truth.
    #
    # Priority (each field independently):
    #   1. Environment variable  (PREDICTOR_TARGET_*)
    #   2. Interactive prompt     (skipped if PREDICTOR_NO_PROMPT=1)
    #   3. Abort with clear error
    #
    # No guessing from kachiuma / shutuba — the caller must know
    # what race is being predicted.
    # ----------------------------------------------------------------
    no_prompt = bool(os.environ.get("PREDICTOR_NO_PROMPT", ""))

    def _resolve_field(env_key: str, label: str, current: str, prompt_hint: str = "") -> str:
        val = (os.environ.get(env_key, "") or "").strip()
        if val:
            return val
        if current:
            return current   # already set in Config.from_env via another env var
        if no_prompt:
            return ""
        try:
            user_val = input(f"  {label} [{prompt_hint}]: ").strip()
            return user_val if user_val else ""
        except EOFError:
            return ""

    if not config.target_location:
        config.target_location = _resolve_field(
            "PREDICTOR_TARGET_LOCATION", "Location (e.g. 中山)", "", "中山")
    if not config.target_location:
        print("[ERROR] target_location is required. Set PREDICTOR_TARGET_LOCATION env var.")
        return

    surf = _resolve_field("PREDICTOR_TARGET_SURFACE", "Surface 芝/ダ", config.target_surface, config.target_surface)
    if surf:
        config.target_surface = _normalize_surface(surf, config.target_surface)

    dist_str = _resolve_field("PREDICTOR_TARGET_DISTANCE", "Distance (m)", str(config.target_distance), str(config.target_distance))
    if dist_str:
        try:
            config.target_distance = int(float(dist_str))
        except ValueError:
            pass

    cond = _resolve_field("PREDICTOR_TARGET_CONDITION", "Condition (良/稍/重/不)", config.target_condition, config.target_condition)
    if cond:
        config.target_condition = cond
        config.expected_baba_index = BABA_PROXY_MAP.get(cond, config.expected_baba_index)

    # Target race date — used for rest_days, reproducible in backtesting
    date_str = _resolve_field("PREDICTOR_TARGET_DATE", "Race date (YYYY-MM-DD)", "", "today")
    if date_str:
        try:
            target_race_date = pd.Timestamp(date_str)
        except Exception:
            target_race_date = pd.Timestamp.now()
    else:
        target_race_date = pd.Timestamp.now()

    print(f"[INFO] Target: {config.target_location} {config.target_surface}{config.target_distance} "
          f"{config.target_condition} (baba≈{config.expected_baba_index})")
    print(f"[INFO] Target race date: {target_race_date.strftime('%Y-%m-%d')}")

    # Build training data (jockey stats are computed per-race-date inside,
    # using expanding cumulative snapshots — no leakage)
    train_df = engine.create_training_features(history_df)
    if train_df.empty:
        print("[ERROR] No training data could be built.")
        return

    # Fit global stats on ALL data for inference (no leakage concern for live prediction)
    engine.fit_global_stats(history_df, cutoff_date=None)

    # Load odds
    odds_engine = OddsEngine()
    odds_engine.load_and_compute()

    # Build inference features (shutuba_df is pure history, no current race entry)
    inference_df = engine.create_inference_features(shutuba_df, odds_engine, target_race_date=target_race_date)
    if inference_df.empty:
        print("[ERROR] No inference data could be built.")
        return

    # Train ensemble
    ensemble = StackingEnsemble()
    metrics = ensemble.train(train_df)

    # Predict
    results = ensemble.predict(inference_df)

    # Print results
    print("\n" + "=" * 68)
    print("  PREDICTIONS - V5 Ultimate Stacking Ensemble")
    print("=" * 68)

    for i, (_, row) in enumerate(results.iterrows()):
        hno = row.get("horse_no", 0)
        hno_str = f"{int(hno):>2}" if hno and hno > 0 else " ?"
        name = str(row.get("HorseName", ""))
        odds_val = row.get("Odds", 0)
        score = float(row.get("rank_score", 0))
        prob = float(row.get("Top3Prob_model", 0))
        odds_str = f"odds={odds_val:6.1f}" if odds_val and odds_val > 0 else "odds=   N/A"
        marker = " ★★★" if i < 1 else " ★★" if i < 3 else " ★" if i < 5 else ""
        print(f"  {i+1:>2}. {hno_str}  {name:<18s}  {odds_str}  score={score:.4f}  Top3={prob*100:.1f}%{marker}")

    conf = float(results["confidence_score"].iloc[0]) if not results.empty else 0.0
    stab = float(results["stability_score"].iloc[0]) if not results.empty else 0.0
    print(f"\n  Confidence: {conf:.4f}  Stability: {stab:.4f}")
    print(f"  CV AUC: {metrics.get('mean_auc', 0):.4f}  Runners: {len(results)}")

    # Save
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Saved: {config.output_path}")


# Pipeline entry point
def main_pipeline_entry():
    main()


if __name__ == "__main__":
    main()

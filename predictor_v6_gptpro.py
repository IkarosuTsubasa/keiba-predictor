#!/usr/bin/env python3
"""
predictor_v6.py - market-prior + skill-residual + race-ranker predictor
=======================================================================

This version is intentionally built on top of predictor_v5_omc's feature engine,
but fixes the parts that matter most for real hit rate:

1. Training samples are built chronologically per horse (strictly pre-race history only)
2. History is deduplicated by HorseName + 日付 before any rolling logic
3. Top3 / Win / Rank are modeled as separate heads
4. Final probability is learned from OOF meta-combination, then Platt-calibrated
5. Final display score is separated from true probability
6. Combo odds (wide / quinella / exacta / trio) are used only as
   a light post-model rerank overlay unless historical snapshots exist

Usage:
    python predictor_v6.py

Expected files in working directory:
    - kachiuma.csv
    - shutuba.csv
Optional:
    - odds.csv
    - fuku_odds.csv
    - wide_odds.csv
    - quinella_odds.csv
    - exacta_odds.csv
    - trio_odds.csv
"""

from __future__ import annotations

import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from predictor_v5_omc import (
        BABA_PROXY_MAP,
        FEATURE_COLUMNS,
        Config,
        FeatureEngine,
        OddsEngine,
        PlattCalibrator,
        TimeSeriesCV,
        _normalize_surface,
        _pf,
        _pi,
        configure_utf8_io,
        resolve_lgbm_n_jobs,
    )
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "[ERROR] predictor_v6.py requires predictor_v5_omc.py in the same directory. "
        f"Import failed: {exc}"
    )

warnings.filterwarnings("ignore", category=UserWarning)
configure_utf8_io()
LGBM_N_JOBS = resolve_lgbm_n_jobs()

PROB_EPS = 1e-6


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


ENABLE_MARKET_OVERLAY = _env_truthy("PREDICTOR_ENABLE_MARKET_OVERLAY", True)
USE_MARKET_FEATURES = _env_truthy("PREDICTOR_USE_MARKET_FEATURES", False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [re.sub(r"\s+", "", str(c or "")).strip() for c in d.columns]
    return d


def neutral_market_feat() -> Dict[str, float]:
    return {
        "odds_win": 0.0,
        "implied_prob_win": 0.0,
        "implied_prob_place": 0.0,
        "wide_network_strength": 0.0,
        "quinella_network_strength": 0.0,
        "market_centrality": 0.0,
        "exacta_network_strength": 0.0,
        "exacta_network_strength_norm": 0.0,
        "trio_network_strength": 0.0,
        "trio_network_strength_norm": 0.0,
    }



def dedup_history(df: pd.DataFrame, label: str = "data") -> pd.DataFrame:
    d = df.copy()
    subset = [c for c in ["HorseName", "日付"] if c in d.columns]
    before = len(d)
    if subset:
        d = d.drop(columns=["finish_pos", "is_top3"], errors="ignore")
        d = d.drop_duplicates(subset=subset).reset_index(drop=True)
    after = len(d)
    if after != before:
        print(f"[INFO] {label} dedup: {before} -> {after} rows")
    return d



def ensure_race_id(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "race_id" in d.columns:
        d["race_id"] = d["race_id"].astype(str)
        return d

    date_col = d["Date"].dt.strftime("%Y-%m-%d") if "Date" in d.columns else ""
    loc_col = d["Location"].fillna("").astype(str) if "Location" in d.columns else ""
    if "R" in d.columns:
        r_col = d["R"].fillna(0).apply(lambda x: str(int(float(x))) if str(x).strip() not in {"", "nan"} else "0")
    else:
        r_col = "0"
    d["race_id"] = date_col.astype(str) + "_" + loc_col.astype(str) + "_" + pd.Series(r_col, index=d.index).astype(str)
    return d



def clip_prob(p: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, PROB_EPS, 1.0 - PROB_EPS)



def sigmoid(x: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-arr))



def logit(p: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = clip_prob(np.asarray(p, dtype=float))
    return np.log(arr / (1.0 - arr))



def safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)



def group_minmax(values: Sequence[float], groups: Sequence[Any]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    g = pd.Series(groups).astype(str).to_numpy()
    out = np.zeros_like(v, dtype=float)
    if len(v) == 0:
        return out
    frame = pd.DataFrame({"g": g, "v": v})
    for gid, idx in frame.groupby("g", sort=False).groups.items():
        arr = v[list(idx)]
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if not np.isfinite(lo) or not np.isfinite(hi):
            out[list(idx)] = 0.5
        elif hi - lo <= 1e-12:
            out[list(idx)] = 0.5
        else:
            out[list(idx)] = (arr - lo) / (hi - lo)
    return out



def race_level_normalize(values: Sequence[float], groups: Sequence[Any], top_k: int = 3) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    g = pd.Series(groups).astype(str).to_numpy()
    out = np.zeros_like(v, dtype=float)
    if len(v) == 0:
        return out
    frame = pd.DataFrame({"g": g, "v": v})
    for gid, idx in frame.groupby("g", sort=False).groups.items():
        arr = np.clip(v[list(idx)], 0.0, None)
        total = float(np.nansum(arr))
        k = min(top_k, len(arr))
        if total > 0:
            out[list(idx)] = arr * float(k) / total
        else:
            out[list(idx)] = float(k) / max(len(arr), 1)
    return np.clip(out, 0.0, 0.999)



def expected_calibration_error(y_true: Sequence[int], probs: Sequence[float], n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    if len(y) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (p >= bins[i]) & (p < bins[i + 1])
        else:
            mask = (p >= bins[i]) & (p <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += mask.mean() * abs(acc - conf)
    return float(ece)



def _rank_relevance(rank: float) -> float:
    try:
        r = int(float(rank))
    except Exception:
        return 0.0
    return float(max(0, 6 - r) if r <= 5 else 0)



def ndcg_at_k(relevance: Sequence[float], score: Sequence[float], k: int = 3) -> float:
    rel = np.asarray(relevance, dtype=float)
    sc = np.asarray(score, dtype=float)
    if len(rel) == 0:
        return 0.0
    order = np.argsort(-sc)[:k]
    ideal = np.argsort(-rel)[:k]

    def _dcg(vals: np.ndarray) -> float:
        if len(vals) == 0:
            return 0.0
        gains = (2.0 ** vals - 1.0)
        discounts = np.log2(np.arange(2, len(vals) + 2))
        return float(np.sum(gains / discounts))

    dcg = _dcg(rel[order])
    idcg = _dcg(rel[ideal])
    if idcg <= 0:
        return 0.0
    return dcg / idcg



def evaluate_rank_metrics(
    race_ids: Sequence[Any],
    y_top3: Sequence[int],
    y_win: Sequence[int],
    rank_label: Sequence[float],
    score: Sequence[float],
    top3_prob: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    df = pd.DataFrame(
        {
            "race_id": pd.Series(race_ids).astype(str),
            "y_top3": np.asarray(y_top3, dtype=int),
            "y_win": np.asarray(y_win, dtype=int),
            "rank_label": np.asarray(rank_label, dtype=float),
            "score": np.asarray(score, dtype=float),
        }
    )
    if top3_prob is not None:
        df["top3_prob"] = np.asarray(top3_prob, dtype=float)

    if df.empty:
        return {
            "hit1": 0.0,
            "hit3_slot": 0.0,
            "ndcg3": 0.0,
            "ece": 0.0,
            "n_races": 0.0,
        }

    hit1_list: List[float] = []
    hit3_slot_list: List[float] = []
    ndcg3_list: List[float] = []

    for _, g in df.groupby("race_id", sort=False):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        if g.empty:
            continue
        hit1_list.append(float(g.loc[0, "y_win"]))
        top_k = min(3, len(g))
        hit3_slot_list.append(float(g.head(top_k)["y_top3"].sum()) / max(top_k, 1))
        rel = g["rank_label"].apply(_rank_relevance).to_numpy(dtype=float)
        ndcg3_list.append(ndcg_at_k(rel, g["score"].to_numpy(dtype=float), k=3))

    metrics = {
        "hit1": float(np.mean(hit1_list)) if hit1_list else 0.0,
        "hit3_slot": float(np.mean(hit3_slot_list)) if hit3_slot_list else 0.0,
        "ndcg3": float(np.mean(ndcg3_list)) if ndcg3_list else 0.0,
        "n_races": float(df["race_id"].nunique()),
    }
    if top3_prob is not None and len(df) > 0:
        metrics["ece"] = expected_calibration_error(df["y_top3"], df["top3_prob"])
    else:
        metrics["ece"] = 0.0
    return metrics



def add_auxiliary_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in d.columns:
            d[col] = 0.0
    for col in [
        "y", "y_top3", "y_win", "rank_label", "horse_no", "Odds", "field_size",
        "draw", "weight_carried", "rest_days",
    ]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d["odds_win"] = safe_series(d, "odds_win", 0.0)
    d["implied_prob_win"] = safe_series(d, "implied_prob_win", 0.0)
    d["implied_prob_place"] = safe_series(d, "implied_prob_place", 0.0)
    d["popularity_pct"] = safe_series(d, "popularity_pct", 0.5)
    d["field_size"] = safe_series(d, "field_size", 0.0)
    d["rest_days"] = safe_series(d, "rest_days", 60.0)
    d["top3_rate_r5"] = safe_series(d, "top3_rate_r5", 0.0)
    d["top3_rate_all"] = safe_series(d, "top3_rate_all", 0.0)
    d["ti_mean5"] = safe_series(d, "ti_mean5", 0.0)
    d["ti_surface_avg"] = safe_series(d, "ti_surface_avg", 0.0)
    d["dist_diff_optimal"] = safe_series(d, "dist_diff_optimal", 0.0)
    d["age"] = safe_series(d, "age", 3.0)

    d["log_odds_win"] = np.log1p(np.clip(d["odds_win"].to_numpy(dtype=float), 0.0, None))
    d["market_missing"] = ((d["odds_win"] <= 0) | (d["implied_prob_win"] <= 0)).astype(float)
    d["field_size_inv"] = 1.0 / np.clip(d["field_size"].to_numpy(dtype=float), 1.0, None)
    d["rest_days_log"] = np.log1p(np.clip(d["rest_days"].to_numpy(dtype=float), 0.0, None))
    d["implied_prob_delta"] = d["implied_prob_place"] - d["implied_prob_win"]
    d["top3_form_delta"] = d["top3_rate_r5"] - d["top3_rate_all"]
    d["ti_context_edge"] = d["ti_mean5"] - d["ti_surface_avg"]
    d["dist_diff_optimal_log"] = np.log1p(np.clip(d["dist_diff_optimal"].to_numpy(dtype=float), 0.0, None))
    d["age_centered"] = d["age"] - 3.0
    d["odds_x_form"] = d["implied_prob_win"] * (0.5 + d["top3_rate_r5"])
    return d


MARKET_RAW_FEATURES = ["odds_win", "implied_prob_win", "implied_prob_place", "popularity_pct", "field_size"]
MARKET_FEATURES = MARKET_RAW_FEATURES + [
    "log_odds_win",
    "market_missing",
    "field_size_inv",
    "implied_prob_delta",
    "odds_x_form",
]
SKILL_FEATURES = [
    c for c in FEATURE_COLUMNS if c not in {"odds_win", "implied_prob_win", "implied_prob_place", "popularity_pct"}
] + [
    "rest_days_log",
    "top3_form_delta",
    "ti_context_edge",
    "dist_diff_optimal_log",
    "age_centered",
]
RANK_FEATURES = list(dict.fromkeys(SKILL_FEATURES + MARKET_RAW_FEATURES + ["log_odds_win", "market_missing", "field_size_inv"]))
META_FEATURES = [
    "market_top3",
    "skill_top3",
    "market_win",
    "skill_win",
    "ranker_norm",
    "top3_edge",
    "win_edge",
    "implied_prob_win",
    "popularity_pct",
    "field_size_inv",
]


# ---------------------------------------------------------------------------
# Feature engine v6: corrected chronological sample construction
# ---------------------------------------------------------------------------
class FeatureEngineV6(FeatureEngine):
    def _snapshot_map(self, history_df: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, Any]]:
        raw = self._precompute_cumulative_jockey_stats(history_df)
        out: Dict[pd.Timestamp, Dict[str, Any]] = {}
        for k, v in raw.items():
            try:
                out[pd.Timestamp(k)] = v
            except Exception:
                continue
        return out

    def _get_snapshot(self, snapshots: Dict[pd.Timestamp, Dict[str, Any]], target_date: Any) -> Optional[Dict[str, Any]]:
        if target_date is None or pd.isna(target_date):
            return None
        try:
            ts = pd.Timestamp(target_date)
        except Exception:
            return None
        return snapshots.get(ts)

    @staticmethod
    def _make_target_row_market_feat(target_row: pd.Series) -> Dict[str, float]:
        if not USE_MARKET_FEATURES:
            return neutral_market_feat()
        odds_val = float(target_row.get("Odds", 0) or 0)
        if not np.isfinite(odds_val):
            odds_val = 0.0
        feat = neutral_market_feat()
        feat["odds_win"] = odds_val
        feat["implied_prob_win"] = 1.0 / odds_val if odds_val > 0 else 0.0
        feat["implied_prob_place"] = min(0.8, (1.0 / odds_val) * 1.6) if odds_val > 0 else 0.0
        return feat

    def create_training_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        print("[INFO] Building training features (v6 chronological per-horse expansion)...")
        d = history_df.copy()
        d = ensure_race_id(d)
        d = d.sort_values(["HorseName", "Date", "race_id"]).reset_index(drop=True)
        snapshots = self._snapshot_map(d)

        rows_out: List[Dict[str, Any]] = []
        total_horses = int(d["HorseName"].nunique()) if "HorseName" in d.columns else 0
        for idx_h, (horse_name, h) in enumerate(d.groupby("HorseName", sort=False), 1):
            h = h.sort_values(["Date", "race_id"]).reset_index(drop=True)
            if len(h) < 2:
                continue
            if idx_h % 250 == 0:
                print(f"  [INFO] processed horses: {idx_h}/{total_horses}")

            for i in range(1, len(h)):
                target_row = h.iloc[i]
                hist_rows = h.iloc[:i]
                rank = int(target_row.get("Rank", 999) or 999)
                if rank >= 900:
                    continue

                target_date = target_row.get("Date", pd.NaT)
                snapshot = self._get_snapshot(snapshots, target_date)
                if snapshot is not None:
                    self._apply_jockey_snapshot(snapshot)
                else:
                    self.fit_global_stats(d, cutoff_date=pd.Timestamp(target_date) if pd.notna(target_date) else None)

                surf = str(target_row.get("Surface", "") or self.config.target_surface or "ダ")
                dist_val = target_row.get("Distance", self.config.target_distance)
                dist = int(float(dist_val)) if pd.notna(dist_val) else int(self.config.target_distance)
                loc = str(target_row.get("Location", "") or "")
                baba = float(target_row.get("BabaIndex", 0.0) or 0.0)
                condition = str(target_row.get("馬場", "") or self.config.target_condition)
                jockey_id = str(target_row.get("JockeyId", "0") or "0")
                field_size = int(float(target_row.get("FieldSize", 0) or 0))
                draw_val = int(float(target_row.get("Draw", 0) or 0))
                wt_carried = float(target_row.get("WeightCarried", 0.0) or 0.0)
                popularity = int(float(target_row.get("Popularity", 0) or 0)) if USE_MARKET_FEATURES else 0
                sex = str(target_row.get("Sex", "") or "")
                age = float(target_row.get("Age", 3.0) or 3.0)

                feat = self.compute_horse_features(
                    history=hist_rows,
                    target_surface=surf,
                    target_distance=dist,
                    target_location=loc,
                    target_baba=baba,
                    target_condition=condition,
                    target_date=target_date if pd.notna(target_date) else None,
                    jockey_id=jockey_id,
                    field_size=field_size,
                    draw=draw_val,
                    weight_carried=wt_carried,
                    sex=sex,
                    age=age,
                    odds_feat=self._make_target_row_market_feat(target_row),
                    popularity=popularity,
                )
                feat["y"] = int(rank <= 3)
                feat["y_top3"] = int(rank <= 3)
                feat["y_win"] = int(rank == 1)
                feat["rank_label"] = float(rank)
                feat["race_id"] = str(target_row.get("race_id", ""))
                feat["HorseName"] = str(horse_name)
                feat["Date"] = target_date
                feat["horse_no"] = float(target_row.get("horse_no", 0) or 0)
                feat["Odds"] = float(target_row.get("Odds", 0) or 0)
                rows_out.append(feat)

        out = pd.DataFrame(rows_out)
        print(f"[INFO] Training samples: {len(out)}  | races={out['race_id'].nunique() if not out.empty else 0}")
        return out

    @staticmethod
    def _extract_current_row(group: pd.DataFrame) -> Tuple[Optional[pd.Series], pd.DataFrame]:
        if group.empty:
            return None, group
        current_mask = pd.Series(False, index=group.index)
        if "Date" in group.columns:
            current_mask = current_mask | group["Date"].isna()
        if "Rank" in group.columns:
            current_mask = current_mask | (pd.to_numeric(group["Rank"], errors="coerce").fillna(999) >= 900)
        current_rows = group.loc[current_mask]
        if current_rows.empty:
            return None, group
        current_row = current_rows.iloc[-1]
        hist = group.loc[~current_mask].copy()
        return current_row, hist

    @staticmethod
    def _resolve_first_non_null(source: pd.Series, default: float = 0.0) -> float:
        vals = pd.to_numeric(source, errors="coerce").dropna()
        if len(vals) == 0:
            return default
        return float(vals.iloc[-1])

    def create_inference_features(
        self,
        shutuba_df: pd.DataFrame,
        odds_engine: OddsEngine,
        target_race_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        print("[INFO] Building inference features (v6)...")
        if target_race_date is None:
            target_race_date = pd.Timestamp.now()

        d = shutuba_df.copy()
        d = d.sort_values(["HorseName", "Date"]).reset_index(drop=True)
        grouped = {str(k).strip(): v.copy() for k, v in d.groupby("HorseName", sort=False)} if not d.empty else {}

        shutuba_names = [
            str(x).strip()
            for x in d.get("HorseName", pd.Series(dtype=str)).dropna().unique().tolist()
            if str(x).strip()
        ]
        odds_names = (
            [
                str(k).strip()
                for k in getattr(odds_engine, "_features", {}).keys()
                if str(k).strip()
            ]
            if USE_MARKET_FEATURES
            else []
        )

        current_mask = pd.Series(False, index=d.index)
        if not d.empty:
            if "Date" in d.columns:
                current_mask = current_mask | d["Date"].isna()
            if "Rank" in d.columns:
                current_mask = current_mask | (
                    pd.to_numeric(d["Rank"], errors="coerce").fillna(999) >= 900
                )
        current_names = (
            [
                str(x).strip()
                for x in d.loc[current_mask, "HorseName"].dropna().unique().tolist()
                if str(x).strip()
            ]
            if ("HorseName" in d.columns and current_mask.any())
            else []
        )
        shared_names = [name for name in odds_names if name in set(shutuba_names)]

        if current_names:
            target_horses = current_names
        elif odds_names:
            target_horses = odds_names
            if shutuba_names and not shared_names:
                raise RuntimeError(
                    "shutuba.csv and odds files appear to describe different races. "
                    "Refusing to infer with mismatched horse lists."
                )
        else:
            target_horses = shutuba_names

        if not target_horses:
            return pd.DataFrame()

        odds_ranking: Dict[str, int] = {}
        if USE_MARKET_FEATURES:
            odds_list: List[Tuple[str, float]] = []
            for horse_name in target_horses:
                of = odds_engine.get_features(horse_name)
                odds_raw = pd.to_numeric(pd.Series([of.get("odds_win", np.nan)]), errors="coerce").iloc[0]
                odds_val = float(odds_raw) if pd.notna(odds_raw) else np.nan
                odds_list.append((horse_name, odds_val))
            valid_market_count = sum(1 for _, odds_val in odds_list if np.isfinite(odds_val) and odds_val > 0)
            if valid_market_count > 0:
                odds_list.sort(key=lambda x: x[1] if np.isfinite(x[1]) and x[1] > 0 else 9999.0)
                odds_ranking = {name: rank for rank, (name, _) in enumerate(odds_list, 1)}
        field_size = len(target_horses)

        rows_out: List[Dict[str, Any]] = []
        for horse_name in target_horses:
            g = grouped.get(horse_name, pd.DataFrame())
            current_row, hist = self._extract_current_row(g) if not g.empty else (None, pd.DataFrame())
            hist = hist.sort_values(["Date"]).copy() if not hist.empty else hist

            # Current-row-aware inputs; only use true current-row values if present.
            jockey_id = "0"
            draw_val = 0
            wt_carried = 0.0
            current_field_size = field_size
            sex = ""
            age = 3.0
            display_name = horse_name
            horse_no_val = 0

            if current_row is not None:
                display_name = str(current_row.get("HorseName", horse_name) or horse_name).strip()
                jockey_id = str(current_row.get("JockeyId_current", current_row.get("JockeyId", "0")) or "0")
                draw_val = int(float(current_row.get("Draw", current_row.get("枠番", 0)) or 0))
                wt_carried = float(current_row.get("WeightCarried", current_row.get("斤量", 0.0)) or 0.0)
                current_field_size = int(float(current_row.get("FieldSize", current_row.get("頭数", field_size)) or field_size))
                sex = str(current_row.get("Sex", "") or "")
                age_val = current_row.get("Age", 3.0)
                age = float(age_val) if pd.notna(age_val) else 3.0
                horse_no_val = int(float(current_row.get("horse_no", current_row.get("馬番", 0)) or 0))
            else:
                if not g.empty and "JockeyId_current" in g.columns:
                    cur_vals = g["JockeyId_current"].dropna().astype(str)
                    if len(cur_vals) > 0:
                        jockey_id = str(cur_vals.iloc[-1])
                if jockey_id == "0" and not hist.empty and "JockeyId" in hist.columns:
                    jid = hist["JockeyId"].dropna().astype(str)
                    if len(jid) > 0:
                        jockey_id = str(jid.iloc[-1])
                if not hist.empty:
                    latest = hist.iloc[-1]
                    display_name = str(latest.get("HorseName", horse_name) or horse_name).strip()
                    sex = str(latest.get("Sex", "") or "")
                    age_val = latest.get("Age", 3.0)
                    age = float(age_val) if pd.notna(age_val) else 3.0
                    horse_no_val = int(float(latest.get("horse_no", latest.get("馬番", 0)) or 0))

            entry_ref_feat = odds_engine.get_features(display_name or horse_name)
            odds_feat = entry_ref_feat if USE_MARKET_FEATURES else neutral_market_feat()
            popularity = int(odds_ranking.get(horse_name, odds_ranking.get(display_name, 0)))

            feat = self.compute_horse_features(
                history=hist if not hist.empty else pd.DataFrame(),
                target_surface=self.config.target_surface,
                target_distance=int(self.config.target_distance),
                target_location=self.config.target_location,
                target_baba=float(self.config.expected_baba_index),
                target_condition=self.config.target_condition,
                target_date=target_race_date,
                jockey_id=jockey_id,
                field_size=current_field_size,
                draw=draw_val,
                weight_carried=wt_carried,
                sex=sex,
                age=age,
                odds_feat=odds_feat,
                popularity=popularity,
            )
            feat["HorseName"] = display_name or horse_name
            odds_horse_no = int(float(entry_ref_feat.get("horse_no", 0) or 0))
            # Prefer the current race horse number from odds/current entry.
            # If neither exists, keep horse_no empty instead of leaking a stale
            # number from the horse's prior race history.
            feat["horse_no"] = odds_horse_no if odds_horse_no > 0 else (horse_no_val if current_row is not None else 0)
            feat["Odds"] = float(entry_ref_feat.get("odds_win", 0.0) or 0.0)
            feat["race_id"] = "current"
            rows_out.append(feat)

        out = pd.DataFrame(rows_out)
        print(f"[INFO] Inference horses: {len(out)}")
        return out


# ---------------------------------------------------------------------------
# Predictor v6
# ---------------------------------------------------------------------------
class PredictorV6:
    def __init__(self):
        self.market_top3_model: Optional[LogisticRegression] = None
        self.market_win_model: Optional[LogisticRegression] = None
        self.skill_top3_models: List[lgb.LGBMClassifier] = []
        self.skill_win_models: List[lgb.LGBMClassifier] = []
        self.ranker: Optional[lgb.LGBMRanker] = None
        self.meta_top3_model: Optional[LogisticRegression] = None
        self.meta_win_model: Optional[LogisticRegression] = None
        self.top3_calibrator = PlattCalibrator()
        self.win_calibrator = PlattCalibrator()
        if USE_MARKET_FEATURES:
            self.top3_blend_weights: Dict[str, float] = {"market": 0.40, "skill": 0.45, "ranker": 0.15}
            self.win_blend_weights: Dict[str, float] = {"market": 0.45, "skill": 0.40, "ranker": 0.15}
        else:
            self.top3_blend_weights = {"market": 0.00, "skill": 0.80, "ranker": 0.20}
            self.win_blend_weights = {"market": 0.00, "skill": 0.75, "ranker": 0.25}
        self.rank_weights: Dict[str, float] = {"top3": 0.50, "win": 0.25, "rank": 0.25}
        self.overlay_enabled_: bool = bool(ENABLE_MARKET_OVERLAY and USE_MARKET_FEATURES)
        self.training_report_: Dict[str, float] = {}

    # ----- model builders -----
    @staticmethod
    def _build_market_lr() -> LogisticRegression:
        return LogisticRegression(C=0.8, max_iter=3000, solver="lbfgs")

    @staticmethod
    def _build_skill_top3_a() -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.15,
            reg_lambda=1.2,
            random_state=42,
            n_jobs=LGBM_N_JOBS,
            verbose=-1,
        )

    @staticmethod
    def _build_skill_top3_b() -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=900,
            learning_rate=0.02,
            num_leaves=20,
            max_depth=6,
            min_child_samples=35,
            subsample=0.75,
            colsample_bytree=0.70,
            reg_alpha=0.35,
            reg_lambda=2.0,
            random_state=123,
            n_jobs=LGBM_N_JOBS,
            verbose=-1,
        )

    @staticmethod
    def _build_skill_win_a() -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="binary",
            is_unbalance=True,
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=24,
            min_child_samples=18,
            subsample=0.85,
            colsample_bytree=0.78,
            reg_alpha=0.20,
            reg_lambda=1.6,
            random_state=7,
            n_jobs=LGBM_N_JOBS,
            verbose=-1,
        )

    @staticmethod
    def _build_skill_win_b() -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="binary",
            is_unbalance=True,
            n_estimators=1100,
            learning_rate=0.02,
            num_leaves=16,
            max_depth=5,
            min_child_samples=30,
            subsample=0.70,
            colsample_bytree=0.65,
            reg_alpha=0.45,
            reg_lambda=2.4,
            random_state=99,
            n_jobs=LGBM_N_JOBS,
            verbose=-1,
        )

    @staticmethod
    def _build_ranker() -> lgb.LGBMRanker:
        return lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=650,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.15,
            reg_lambda=1.2,
            random_state=42,
            n_jobs=LGBM_N_JOBS,
            verbose=-1,
        )

    @staticmethod
    def _safe_fit_logistic(X: np.ndarray, y: np.ndarray) -> Tuple[Optional[LogisticRegression], float]:
        prior = float(np.mean(y)) if len(y) > 0 else 0.0
        if len(y) < 20 or len(np.unique(y)) < 2:
            return None, prior
        model = PredictorV6._build_market_lr()
        model.fit(X, y)
        return model, prior

    @staticmethod
    def _predict_logistic(model: Optional[LogisticRegression], prior: float, X: np.ndarray) -> np.ndarray:
        if model is None:
            return np.full(X.shape[0], prior, dtype=float)
        return model.predict_proba(X)[:, 1]

    @staticmethod
    def _fit_lgb_ensemble(builders: Sequence[Any], X: np.ndarray, y: np.ndarray) -> Tuple[List[Any], float]:
        prior = float(np.mean(y)) if len(y) > 0 else 0.0
        models: List[Any] = []
        if len(y) < 20 or len(np.unique(y)) < 2:
            return models, prior
        for builder in builders:
            try:
                model = builder()
                model.fit(X, y)
                models.append(model)
            except Exception:
                continue
        return models, prior

    @staticmethod
    def _predict_lgb_ensemble(models: Sequence[Any], prior: float, X: np.ndarray) -> np.ndarray:
        if not models:
            return np.full(X.shape[0], prior, dtype=float)
        preds = [m.predict_proba(X)[:, 1] for m in models]
        return np.mean(preds, axis=0)

    @staticmethod
    def _relevance_labels(rank_series: pd.Series) -> np.ndarray:
        return np.array([max(0, 6 - int(float(v))) if float(v) <= 5 else 0 for v in rank_series], dtype=float)

    def _fit_ranker(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> Optional[lgb.LGBMRanker]:
        if df.empty or "race_id" not in df.columns or df["race_id"].nunique() < 8:
            return None
        sorted_df = df.sort_values(["Date", "race_id", "HorseName"]).copy()
        y_rank = self._relevance_labels(sorted_df["rank_label"])
        if np.nanmax(y_rank) <= 0:
            return None
        group_sizes = sorted_df.groupby("race_id", sort=False).size().to_numpy()
        try:
            model = self._build_ranker()
            model.fit(sorted_df[list(feature_cols)].fillna(0).to_numpy(dtype=float), y_rank, group=group_sizes)
            return model
        except Exception:
            return None

    def _predict_ranker(self, model: Optional[lgb.LGBMRanker], df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
        if model is None or df.empty:
            return np.full(len(df), 0.5, dtype=float)
        raw = model.predict(df[list(feature_cols)].fillna(0).to_numpy(dtype=float))
        race_ids = df.get("race_id", pd.Series(["current"] * len(df), index=df.index)).astype(str)
        return group_minmax(raw, race_ids)

    def _build_meta_frame(
        self,
        df: pd.DataFrame,
        market_top3: np.ndarray,
        skill_top3: np.ndarray,
        market_win: np.ndarray,
        skill_win: np.ndarray,
        ranker_norm: np.ndarray,
    ) -> pd.DataFrame:
        meta = pd.DataFrame(index=df.index)
        meta["market_top3"] = market_top3
        meta["skill_top3"] = skill_top3
        meta["market_win"] = market_win
        meta["skill_win"] = skill_win
        meta["ranker_norm"] = ranker_norm
        meta["top3_edge"] = skill_top3 - market_top3
        meta["win_edge"] = skill_win - market_win
        meta["implied_prob_win"] = safe_series(df, "implied_prob_win", 0.0)
        meta["popularity_pct"] = safe_series(df, "popularity_pct", 0.5)
        meta["field_size_inv"] = safe_series(df, "field_size_inv", 1.0)
        return meta

    def _predict_meta_prob(
        self,
        model: Optional[LogisticRegression],
        X_meta: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        if model is None:
            return fallback
        return model.predict_proba(X_meta)[:, 1]

    def _blend_meta_fallback(self, meta_df: pd.DataFrame, mode: str) -> np.ndarray:
        if mode == "win":
            weights = self.win_blend_weights
            market_col = "market_win"
            skill_col = "skill_win"
        else:
            weights = self.top3_blend_weights
            market_col = "market_top3"
            skill_col = "skill_top3"
        return np.clip(
            weights["market"] * meta_df[market_col].to_numpy(dtype=float)
            + weights["skill"] * meta_df[skill_col].to_numpy(dtype=float)
            + weights["ranker"] * meta_df["ranker_norm"].to_numpy(dtype=float),
            PROB_EPS,
            1.0 - PROB_EPS,
        )

    def _fit_meta_stack(
        self,
        df: pd.DataFrame,
        meta_df: pd.DataFrame,
        labels: np.ndarray,
        mode: str,
    ) -> Tuple[Optional[LogisticRegression], PlattCalibrator, np.ndarray, np.ndarray]:
        fallback = self._blend_meta_fallback(meta_df, mode)
        strict_meta_oof = np.full(len(df), np.nan, dtype=float)
        strict_true_oof = np.full(len(df), np.nan, dtype=float)
        inner_cv = TimeSeriesCV(n_splits=4, min_train_frac=0.55)
        inner_folds = inner_cv.split(df)

        for tr_idx, te_idx in inner_folds:
            X_tr = meta_df.iloc[tr_idx][META_FEATURES].fillna(0).to_numpy(dtype=float)
            X_te = meta_df.iloc[te_idx][META_FEATURES].fillna(0).to_numpy(dtype=float)
            y_tr = labels[tr_idx]

            if len(np.unique(y_tr)) >= 2 and len(y_tr) >= 30:
                fold_model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
                fold_model.fit(X_tr, y_tr)
                meta_tr = fold_model.predict_proba(X_tr)[:, 1]
                meta_te = fold_model.predict_proba(X_te)[:, 1]
            else:
                meta_tr = fallback[tr_idx]
                meta_te = fallback[te_idx]

            fold_calibrator = PlattCalibrator()
            fold_calibrator.fit(meta_tr, y_tr)
            strict_meta_oof[te_idx] = meta_te
            strict_true_oof[te_idx] = fold_calibrator.transform(meta_te)

        missing_mask = ~np.isfinite(strict_meta_oof)
        if np.any(missing_mask):
            strict_meta_oof[missing_mask] = fallback[missing_mask]
            strict_true_oof[missing_mask] = fallback[missing_mask]

        X_all = meta_df[META_FEATURES].fillna(0).to_numpy(dtype=float)
        if len(np.unique(labels)) >= 2 and len(labels) >= 30:
            final_model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
            final_model.fit(X_all, labels)
        else:
            final_model = None

        final_calibrator = PlattCalibrator()
        valid_mask = np.isfinite(strict_meta_oof) & np.isfinite(labels)
        final_calibrator.fit(strict_meta_oof[valid_mask], labels[valid_mask])
        return final_model, final_calibrator, strict_meta_oof, strict_true_oof

    @staticmethod
    def _overlay_specs() -> List[Tuple[str, float]]:
        return [
            ("market_centrality", 0.32),
            ("wide_network_strength", 0.18),
            ("quinella_network_strength", 0.14),
            ("exacta_network_strength_norm", 0.12),
            ("trio_network_strength_norm", 0.12),
        ]

    def _has_overlay_signal(self, df: pd.DataFrame) -> bool:
        for col, _ in self._overlay_specs():
            if col not in df.columns:
                continue
            arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if len(arr) > 0 and np.nanmax(np.abs(arr)) > 1e-12:
                return True
        return False

    def _tune_rank_weights(
        self,
        df: pd.DataFrame,
        race_ids: Sequence[Any],
        y_top3: Sequence[int],
        y_win: Sequence[int],
        rank_label: Sequence[float],
        top3_true: np.ndarray,
        win_true: np.ndarray,
        ranker_norm: np.ndarray,
    ) -> Dict[str, float]:
        groups = pd.Series(race_ids).astype(str).to_numpy()
        top3_component = group_minmax(race_level_normalize(top3_true, groups, top_k=3), groups)
        win_component = group_minmax(race_level_normalize(win_true, groups, top_k=1), groups)
        rank_component = group_minmax(ranker_norm, groups)
        ece = expected_calibration_error(y_top3, top3_true)

        best = {"top3": 0.50, "win": 0.25, "rank": 0.25, "objective": -1e9}
        for w_top3 in np.arange(0.40, 0.66, 0.05):
            for w_win in np.arange(0.10, 0.36, 0.05):
                w_rank = 1.0 - w_top3 - w_win
                if w_rank < 0.10 or w_rank > 0.40:
                    continue
                score = w_top3 * top3_component + w_win * win_component + w_rank * rank_component
                score_eval, overlay_strength = self._apply_market_overlay(df, score, groups)
                metrics = evaluate_rank_metrics(race_ids, y_top3, y_win, rank_label, score_eval, top3_prob=top3_true)
                objective = (
                    0.45 * metrics["hit3_slot"]
                    + 0.25 * metrics["hit1"]
                    + 0.20 * metrics["ndcg3"]
                    - 0.10 * ece
                )
                if objective > best["objective"]:
                    best = {
                        "top3": float(w_top3),
                        "win": float(w_win),
                        "rank": float(w_rank),
                        "objective": float(objective),
                        "hit1": float(metrics["hit1"]),
                        "hit3_slot": float(metrics["hit3_slot"]),
                        "ndcg3": float(metrics["ndcg3"]),
                        "ece": float(ece),
                        "overlay_strength": float(overlay_strength),
                    }
        return best

    @staticmethod
    def _confidence_from_score(score: np.ndarray) -> float:
        if len(score) < 3:
            return 0.0
        s = np.sort(score)[::-1]
        gap = float(s[0] - s[min(2, len(s) - 1)])
        return max(0.0, min(1.0, gap / 0.22))

    @staticmethod
    def _stability_from_components(
        top3_component: np.ndarray,
        win_component: np.ndarray,
        rank_component: np.ndarray,
        market_component: Optional[np.ndarray] = None,
        top_k: int = 3,
    ) -> float:
        rankings = []
        for arr in [top3_component, win_component, rank_component, market_component]:
            if arr is None or len(arr) == 0:
                continue
            rankings.append(set(np.argsort(-np.asarray(arr, dtype=float))[:top_k].tolist()))
        if len(rankings) < 2:
            return 0.5
        inter = set.intersection(*rankings)
        union = set.union(*rankings)
        return len(inter) / max(len(union), 1)

    def _apply_market_overlay(self, df: pd.DataFrame, base_score: np.ndarray, race_ids: Sequence[Any]) -> Tuple[np.ndarray, float]:
        if not self.overlay_enabled_:
            return np.asarray(base_score, dtype=float), 0.0
        specs = self._overlay_specs()
        overlay = np.zeros(len(df), dtype=float)
        active_weight = 0.0
        for col, w in specs:
            if col not in df.columns:
                continue
            arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if np.nanmax(np.abs(arr)) <= 1e-12:
                continue
            overlay += w * group_minmax(arr, race_ids)
            active_weight += w
        if active_weight <= 0:
            return np.asarray(base_score, dtype=float), 0.0
        overlay = overlay / active_weight
        strength = 0.08
        return (1.0 - strength) * np.asarray(base_score, dtype=float) + strength * overlay, strength

    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        print("[INFO] Training predictor_v6...")
        d = add_auxiliary_columns(train_df)
        for col in META_FEATURES + RANK_FEATURES + SKILL_FEATURES + MARKET_FEATURES:
            if col not in d.columns:
                d[col] = 0.0
        d = d.sort_values(["Date", "race_id", "HorseName"]).reset_index(drop=True)
        self.overlay_enabled_ = bool(ENABLE_MARKET_OVERLAY and USE_MARKET_FEATURES) and self._has_overlay_signal(d)

        y_top3 = pd.to_numeric(d["y_top3"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_win = pd.to_numeric(d["y_win"], errors="coerce").fillna(0).astype(int).to_numpy()
        race_ids = d["race_id"].astype(str).to_numpy()

        X_skill = d[SKILL_FEATURES].fillna(0).to_numpy(dtype=float)

        cv = TimeSeriesCV(n_splits=5, min_train_frac=0.45)
        folds = cv.split(d)

        oof_market_top3 = np.full(len(d), np.nan, dtype=float)
        oof_skill_top3 = np.full(len(d), np.nan, dtype=float)
        oof_market_win = np.full(len(d), np.nan, dtype=float)
        oof_skill_win = np.full(len(d), np.nan, dtype=float)
        oof_ranker = np.full(len(d), np.nan, dtype=float)

        for fold_idx, (tr_idx, te_idx) in enumerate(folds, 1):
            df_tr = d.iloc[tr_idx].copy()
            df_te = d.iloc[te_idx].copy()
            X_skill_tr, X_skill_te = X_skill[tr_idx], X_skill[te_idx]

            if USE_MARKET_FEATURES:
                X_market_tr = d.iloc[tr_idx][MARKET_FEATURES].fillna(0).to_numpy(dtype=float)
                X_market_te = d.iloc[te_idx][MARKET_FEATURES].fillna(0).to_numpy(dtype=float)
                m_top3_model, m_top3_prior = self._safe_fit_logistic(X_market_tr, y_top3[tr_idx])
                m_win_model, m_win_prior = self._safe_fit_logistic(X_market_tr, y_win[tr_idx])
                oof_market_top3[te_idx] = self._predict_logistic(m_top3_model, m_top3_prior, X_market_te)
                oof_market_win[te_idx] = self._predict_logistic(m_win_model, m_win_prior, X_market_te)
            else:
                oof_market_top3[te_idx] = 0.0
                oof_market_win[te_idx] = 0.0

            s_top3_models, s_top3_prior = self._fit_lgb_ensemble(
                [self._build_skill_top3_a, self._build_skill_top3_b], X_skill_tr, y_top3[tr_idx]
            )
            s_win_models, s_win_prior = self._fit_lgb_ensemble(
                [self._build_skill_win_a, self._build_skill_win_b], X_skill_tr, y_win[tr_idx]
            )
            oof_skill_top3[te_idx] = self._predict_lgb_ensemble(s_top3_models, s_top3_prior, X_skill_te)
            oof_skill_win[te_idx] = self._predict_lgb_ensemble(s_win_models, s_win_prior, X_skill_te)

            fold_ranker = self._fit_ranker(df_tr, RANK_FEATURES)
            oof_ranker[te_idx] = self._predict_ranker(fold_ranker, df_te, RANK_FEATURES)

            fold_meta = self._build_meta_frame(
                df_te,
                oof_market_top3[te_idx],
                oof_skill_top3[te_idx],
                oof_market_win[te_idx],
                oof_skill_win[te_idx],
                oof_ranker[te_idx],
            )
            raw_top3_fold = np.clip(
                0.40 * fold_meta["market_top3"].to_numpy(dtype=float)
                + 0.45 * fold_meta["skill_top3"].to_numpy(dtype=float)
                + 0.15 * fold_meta["ranker_norm"].to_numpy(dtype=float),
                PROB_EPS,
                1.0 - PROB_EPS,
            )
            fold_metrics = evaluate_rank_metrics(
                df_te["race_id"],
                y_top3[te_idx],
                y_win[te_idx],
                df_te["rank_label"],
                group_minmax(raw_top3_fold, df_te["race_id"]),
                top3_prob=raw_top3_fold,
            )
            print(
                "  Fold {fold}: hit1={hit1:.4f} hit3={hit3:.4f} ndcg3={ndcg3:.4f} "
                "(train={ntr}, test={nte})".format(
                    fold=fold_idx,
                    hit1=fold_metrics["hit1"],
                    hit3=fold_metrics["hit3_slot"],
                    ndcg3=fold_metrics["ndcg3"],
                    ntr=len(tr_idx),
                    nte=len(te_idx),
                )
            )

        valid_mask = (
            np.isfinite(oof_market_top3)
            & np.isfinite(oof_skill_top3)
            & np.isfinite(oof_market_win)
            & np.isfinite(oof_skill_win)
            & np.isfinite(oof_ranker)
        )
        if not np.any(valid_mask):
            raise RuntimeError("No valid OOF predictions generated. Check training data / dates.")

        d_valid = d.loc[valid_mask].copy().reset_index(drop=True)
        meta_valid = self._build_meta_frame(
            d_valid,
            oof_market_top3[valid_mask],
            oof_skill_top3[valid_mask],
            oof_market_win[valid_mask],
            oof_skill_win[valid_mask],
            oof_ranker[valid_mask],
        )
        y_top3_valid = y_top3[valid_mask]
        y_win_valid = y_win[valid_mask]
        race_valid = d_valid["race_id"].astype(str).to_numpy()
        rank_valid = d_valid["rank_label"].to_numpy(dtype=float)

        self.meta_top3_model, self.top3_calibrator, oof_top3_meta, oof_top3_true = self._fit_meta_stack(
            d_valid,
            meta_valid,
            y_top3_valid,
            mode="top3",
        )
        self.meta_win_model, self.win_calibrator, oof_win_meta, oof_win_true = self._fit_meta_stack(
            d_valid,
            meta_valid,
            y_win_valid,
            mode="win",
        )

        self.rank_weights = self._tune_rank_weights(
            d_valid,
            race_valid,
            y_top3_valid,
            y_win_valid,
            rank_valid,
            oof_top3_true,
            oof_win_true,
            oof_ranker[valid_mask],
        )

        # OOF final metrics
        top3_component = group_minmax(race_level_normalize(oof_top3_true, race_valid, top_k=3), race_valid)
        win_component = group_minmax(race_level_normalize(oof_win_true, race_valid, top_k=1), race_valid)
        rank_component = group_minmax(oof_ranker[valid_mask], race_valid)
        final_oof_score = (
            self.rank_weights["top3"] * top3_component
            + self.rank_weights["win"] * win_component
            + self.rank_weights["rank"] * rank_component
        )
        final_oof_score, oof_overlay_strength = self._apply_market_overlay(d_valid, final_oof_score, race_valid)
        oof_metrics = evaluate_rank_metrics(
            race_valid,
            y_top3_valid,
            y_win_valid,
            rank_valid,
            final_oof_score,
            top3_prob=oof_top3_true,
        )

        # Refit full models
        print("[INFO] Refitting full models...")
        if USE_MARKET_FEATURES:
            X_market = d[MARKET_FEATURES].fillna(0).to_numpy(dtype=float)
            self.market_top3_model, self.market_top3_prior_ = self._safe_fit_logistic(X_market, y_top3)
            self.market_win_model, self.market_win_prior_ = self._safe_fit_logistic(X_market, y_win)
        else:
            self.market_top3_model, self.market_top3_prior_ = None, 0.0
            self.market_win_model, self.market_win_prior_ = None, 0.0
        self.skill_top3_models, self.skill_top3_prior_ = self._fit_lgb_ensemble(
            [self._build_skill_top3_a, self._build_skill_top3_b], X_skill, y_top3
        )
        self.skill_win_models, self.skill_win_prior_ = self._fit_lgb_ensemble(
            [self._build_skill_win_a, self._build_skill_win_b], X_skill, y_win
        )
        self.ranker = self._fit_ranker(d, RANK_FEATURES)

        self.training_report_ = {
            "oof_hit1": float(oof_metrics["hit1"]),
            "oof_hit3_slot": float(oof_metrics["hit3_slot"]),
            "oof_ndcg3": float(oof_metrics["ndcg3"]),
            "oof_ece": float(oof_metrics["ece"]),
            "oof_races": float(oof_metrics["n_races"]),
            "blend_top3": float(self.rank_weights["top3"]),
            "blend_win": float(self.rank_weights["win"]),
            "blend_rank": float(self.rank_weights["rank"]),
            "overlay_enabled": float(1 if self.overlay_enabled_ else 0),
            "overlay_strength": float(oof_overlay_strength),
            "oof_top3_prob_mean": float(np.mean(oof_top3_true)),
            "oof_win_prob_mean": float(np.mean(oof_win_true)),
        }
        print(
            "[INFO] OOF metrics: hit1={hit1:.4f} hit3={hit3:.4f} ndcg3={ndcg3:.4f} ece={ece:.4f}".format(
                hit1=self.training_report_["oof_hit1"],
                hit3=self.training_report_["oof_hit3_slot"],
                ndcg3=self.training_report_["oof_ndcg3"],
                ece=self.training_report_["oof_ece"],
            )
        )
        print(
            "[INFO] Final rank weights: top3={t:.2f} win={w:.2f} rank={r:.2f}".format(
                t=self.rank_weights["top3"],
                w=self.rank_weights["win"],
                r=self.rank_weights["rank"],
            )
        )
        return self.training_report_

    def predict(self, inference_df: pd.DataFrame) -> pd.DataFrame:
        if inference_df.empty:
            return pd.DataFrame()

        d = add_auxiliary_columns(inference_df)
        for col in MARKET_FEATURES + SKILL_FEATURES + RANK_FEATURES + META_FEATURES:
            if col not in d.columns:
                d[col] = 0.0
        if "race_id" not in d.columns:
            d["race_id"] = "current"
        race_ids = d["race_id"].astype(str).to_numpy()

        X_skill = d[SKILL_FEATURES].fillna(0).to_numpy(dtype=float)

        if USE_MARKET_FEATURES:
            X_market = d[MARKET_FEATURES].fillna(0).to_numpy(dtype=float)
            market_top3 = self._predict_logistic(self.market_top3_model, getattr(self, "market_top3_prior_", 0.0), X_market)
            market_win = self._predict_logistic(self.market_win_model, getattr(self, "market_win_prior_", 0.0), X_market)
        else:
            market_top3 = np.zeros(len(d), dtype=float)
            market_win = np.zeros(len(d), dtype=float)
        skill_top3 = self._predict_lgb_ensemble(self.skill_top3_models, getattr(self, "skill_top3_prior_", 0.0), X_skill)
        skill_win = self._predict_lgb_ensemble(self.skill_win_models, getattr(self, "skill_win_prior_", 0.0), X_skill)
        ranker_norm = self._predict_ranker(self.ranker, d, RANK_FEATURES)

        meta_df = self._build_meta_frame(d, market_top3, skill_top3, market_win, skill_win, ranker_norm)
        X_meta = meta_df[META_FEATURES].fillna(0).to_numpy(dtype=float)

        top3_fallback = np.clip(
            self.top3_blend_weights["market"] * market_top3
            + self.top3_blend_weights["skill"] * skill_top3
            + self.top3_blend_weights["ranker"] * ranker_norm,
            PROB_EPS,
            1.0 - PROB_EPS,
        )
        win_fallback = np.clip(
            self.win_blend_weights["market"] * market_win
            + self.win_blend_weights["skill"] * skill_win
            + self.win_blend_weights["ranker"] * ranker_norm,
            PROB_EPS,
            1.0 - PROB_EPS,
        )
        top3_meta = self._predict_meta_prob(self.meta_top3_model, X_meta, top3_fallback)
        win_meta = self._predict_meta_prob(self.meta_win_model, X_meta, win_fallback)

        top3_true = self.top3_calibrator.transform(top3_meta)
        win_true = self.win_calibrator.transform(win_meta)

        top3_share = race_level_normalize(top3_true, race_ids, top_k=3)
        win_share = race_level_normalize(win_true, race_ids, top_k=1)
        top3_component = group_minmax(top3_share, race_ids)
        win_component = group_minmax(win_share, race_ids)
        rank_component = group_minmax(ranker_norm, race_ids)

        rank_score_base = (
            self.rank_weights["top3"] * top3_component
            + self.rank_weights["win"] * win_component
            + self.rank_weights["rank"] * rank_component
        )
        rank_score, overlay_strength = self._apply_market_overlay(d, rank_score_base, race_ids)
        rank_score_norm = group_minmax(rank_score, race_ids)

        results = d.copy()
        results["MarketTop3Prob"] = market_top3
        results["SkillTop3Prob"] = skill_top3
        results["MarketWinProb"] = market_win
        results["SkillWinProb"] = skill_win
        results["Top3Prob_true"] = top3_true
        results["Top3Score_norm"] = top3_share
        results["WinProb_true"] = win_true
        results["WinScore_norm"] = win_share
        results["ranker_score"] = rank_component
        results["rank_score_base"] = rank_score_base
        results["rank_score"] = rank_score
        results["rank_score_norm"] = rank_score_norm
        results["overlay_strength"] = overlay_strength

        # Backward-compatible columns for existing website / pipeline.
        results["Top3Prob_model"] = top3_share
        results["Top3Prob"] = top3_true
        results["Top3Prob_raw_lr"] = skill_top3
        results["Top3Prob_raw_lgb"] = market_top3
        results["Top3Prob_cal_lr"] = top3_true
        results["Top3Prob_cal_lgb"] = top3_true
        results["Top3Prob_lr"] = skill_top3
        results["Top3Prob_lgbm"] = market_top3

        results = results.sort_values(["race_id", "rank_score"], ascending=[True, False]).reset_index(drop=True)

        # Race-level diagnostics.
        conf_by_race: Dict[str, float] = {}
        stab_by_race: Dict[str, float] = {}
        for rid, g in results.groupby("race_id", sort=False):
            conf_by_race[rid] = self._confidence_from_score(g["rank_score"].to_numpy(dtype=float))
            stab_by_race[rid] = self._stability_from_components(
                group_minmax(g["Top3Score_norm"].to_numpy(dtype=float), [rid] * len(g)),
                group_minmax(g["WinScore_norm"].to_numpy(dtype=float), [rid] * len(g)),
                group_minmax(g["ranker_score"].to_numpy(dtype=float), [rid] * len(g)),
                market_component=group_minmax(g["MarketTop3Prob"].to_numpy(dtype=float), [rid] * len(g)),
            )
        results["confidence_score"] = results["race_id"].map(conf_by_race).fillna(0.0)
        results["stability_score"] = results["race_id"].map(stab_by_race).fillna(0.5)
        results["validity_score"] = np.clip(0.70 * results["confidence_score"] + 0.30 * results["stability_score"], 0.0, 1.0)
        results["consistency_score"] = results["stability_score"]
        results["risk_score"] = 1.0 - results["confidence_score"]
        results["score_is_probability"] = 0
        results["model_mode"] = "v6_skill_rank_no_odds" if not USE_MARKET_FEATURES else "v6_market_skill_rank"
        results["horse_key"] = results.apply(
            lambda r: str(int(r["horse_no"])) if pd.notna(r.get("horse_no")) and float(r.get("horse_no", 0) or 0) > 0 else str(r.get("HorseName", "")),
            axis=1,
        )
        return results


# ---------------------------------------------------------------------------
# Runtime / main
# ---------------------------------------------------------------------------
def resolve_target_context(config: Config) -> Tuple[Config, pd.Timestamp]:
    no_prompt = bool(os.environ.get("PREDICTOR_NO_PROMPT", ""))

    def _resolve_field(env_key: str, label: str, current: str, prompt_hint: str = "") -> str:
        val = (os.environ.get(env_key, "") or "").strip()
        if val:
            return val
        if current:
            return current
        if no_prompt:
            return ""
        try:
            user_val = input(f"  {label} [{prompt_hint}]: ").strip()
            return user_val if user_val else ""
        except EOFError:
            return ""

    if not config.target_location:
        config.target_location = _resolve_field("PREDICTOR_TARGET_LOCATION", "Location (e.g. 中山)", "", "中山")
    if not config.target_location:
        raise RuntimeError("target_location is required. Set PREDICTOR_TARGET_LOCATION env var.")

    surf = _resolve_field("PREDICTOR_TARGET_SURFACE", "Surface 芝/ダ", config.target_surface, config.target_surface)
    if surf:
        config.target_surface = _normalize_surface(surf, config.target_surface)

    dist_str = _resolve_field(
        "PREDICTOR_TARGET_DISTANCE", "Distance (m)", str(config.target_distance), str(config.target_distance)
    )
    if dist_str:
        try:
            config.target_distance = int(float(dist_str))
        except ValueError:
            pass

    cond = _resolve_field(
        "PREDICTOR_TARGET_CONDITION", "Condition (良/稍/重/不)", config.target_condition, config.target_condition
    )
    if cond:
        config.target_condition = cond
        config.expected_baba_index = BABA_PROXY_MAP.get(cond, config.expected_baba_index)

    date_str = _resolve_field("PREDICTOR_TARGET_DATE", "Race date (YYYY-MM-DD)", "", "today")
    if date_str:
        try:
            target_race_date = pd.Timestamp(date_str)
        except Exception:
            target_race_date = pd.Timestamp.now()
    else:
        target_race_date = pd.Timestamp.now()

    return config, target_race_date



def main() -> None:
    print("=" * 72)
    print("  PREDICTOR V6 - Market Prior + Skill Residual + Rank Fusion")
    print("=" * 72)

    if not Path("kachiuma.csv").exists() or not Path("shutuba.csv").exists():
        raise SystemExit("[ERROR] kachiuma.csv or shutuba.csv not found.")

    history_raw = normalize_columns(pd.read_csv("kachiuma.csv"))
    shutuba_raw = normalize_columns(pd.read_csv("shutuba.csv"))
    history_raw = dedup_history(history_raw, label="kachiuma")
    shutuba_raw = dedup_history(shutuba_raw, label="shutuba")
    print(f"[INFO] kachiuma: {len(history_raw)} rows, shutuba: {len(shutuba_raw)} rows")

    config = Config.from_env()
    try:
        config, target_race_date = resolve_target_context(config)
    except RuntimeError as exc:
        raise SystemExit(f"[ERROR] {exc}")

    print(
        f"[INFO] Target: {config.target_location} {config.target_surface}{config.target_distance} "
        f"{config.target_condition} (baba≈{config.expected_baba_index})"
    )
    print(f"[INFO] Target race date: {target_race_date.strftime('%Y-%m-%d')}")

    engine = FeatureEngineV6(config)
    history_df = engine.preprocess(history_raw)
    shutuba_df = engine.preprocess(shutuba_raw)
    history_df = ensure_race_id(history_df)
    shutuba_df = ensure_race_id(shutuba_df)

    train_df = engine.create_training_features(history_df)
    if train_df.empty:
        raise SystemExit("[ERROR] No training data could be built.")

    engine.fit_global_stats(history_df, cutoff_date=None)

    odds_engine = OddsEngine()
    odds_engine.load_and_compute(
        win_path=os.environ.get("ODDS_PATH", "odds.csv"),
        place_path=os.environ.get("FUKU_ODDS_PATH", "fuku_odds.csv"),
        wide_path=os.environ.get("WIDE_ODDS_PATH", "wide_odds.csv"),
        quinella_path=os.environ.get("QUINELLA_ODDS_PATH", "quinella_odds.csv"),
        exacta_path=os.environ.get("EXACTA_ODDS_PATH", "exacta_odds.csv"),
        trio_path=os.environ.get("TRIO_ODDS_PATH", "trio_odds.csv"),
    )

    inference_df = engine.create_inference_features(shutuba_df, odds_engine, target_race_date=target_race_date)
    if inference_df.empty:
        raise SystemExit("[ERROR] No inference data could be built.")

    predictor = PredictorV6()
    report = predictor.train(train_df)
    results = predictor.predict(inference_df)

    print("\n" + "=" * 72)
    print("  PREDICTIONS - V6")
    print("=" * 72)
    for i, (_, row) in enumerate(results.iterrows(), 1):
        hno = row.get("horse_no", 0)
        hno_str = f"{int(hno):>2}" if pd.notna(hno) and float(hno) > 0 else " ?"
        name = str(row.get("HorseName", ""))
        odds_val = float(row.get("Odds", 0) or 0)
        rank_score = float(row.get("rank_score", 0) or 0)
        top3_true = float(row.get("Top3Prob_true", 0) or 0)
        win_true = float(row.get("WinProb_true", 0) or 0)
        odds_str = f"odds={odds_val:6.1f}" if odds_val > 0 else "odds=   N/A"
        marker = " ★★★" if i == 1 else " ★★" if i <= 3 else " ★" if i <= 5 else ""
        print(
            f"  {i:>2}. {hno_str}  {name:<18.18s}  {odds_str}  "
            f"rank={rank_score:.4f}  Top3={top3_true*100:5.1f}%  Win={win_true*100:5.1f}%{marker}"
        )

    if not results.empty:
        conf = float(results["confidence_score"].iloc[0])
        stab = float(results["stability_score"].iloc[0])
    else:
        conf = 0.0
        stab = 0.0
    print(f"\n  Confidence: {conf:.4f}  Stability: {stab:.4f}")
    print(
        "  OOF hit1={hit1:.4f}  hit3={hit3:.4f}  ndcg3={ndcg3:.4f}  ece={ece:.4f}".format(
            hit1=report.get("oof_hit1", 0.0),
            hit3=report.get("oof_hit3_slot", 0.0),
            ndcg3=report.get("oof_ndcg3", 0.0),
            ece=report.get("oof_ece", 0.0),
        )
    )

    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Saved: {output_path}")


if __name__ == "__main__":
    main()

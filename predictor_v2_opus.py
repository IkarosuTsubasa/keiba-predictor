"""
predictor_v2.py - 高命中率赛马预测器
============================================================
核心策略:
  1. 赔率 = 最强信号 (市场汇集百万级信息量)
  2. 历史表现 + 趋势 + 跑法 = 实力全面评估
  3. 休息天数 + 体重变化 + 级别 = 状态判定
  4. LightGBM + 时序CV + Platt校准 = 最优预测

使用方法:
  SCOPE_KEY=central_turf python predictor_v2.py
  输入: kachiuma.csv, shutuba.csv, odds.csv
  输出: predictions.csv (与现有pipeline完全兼容)
============================================================
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def configure_utf8_io():
    for s in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_io()
START_TIME = datetime.now()
OUTPUT_PATH = Path(os.environ.get("PREDICTIONS_OUTPUT", "predictions.csv")).expanduser()


def resolve_lgbm_n_jobs():
    raw = str(os.environ.get("PREDICTOR_LGBM_N_JOBS", "1") or "1").strip()
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return 1


LGBM_N_JOBS = resolve_lgbm_n_jobs()

NO_WAIT = str(os.environ.get("PREDICTOR_NO_WAIT", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_truthy(name):
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}

# ============================================================
# 1. Scope Resolution
# ============================================================
SCOPE_ALIASES = {
    "central_turf": {"central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"},
    "central_dirt": {"central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"},
    "local": {"local", "l", "3"},
}


def resolve_scope():
    raw = os.environ.get("SCOPE_KEY", "").strip().lower().replace(" ", "_").replace("-", "_")
    for k, aliases in SCOPE_ALIASES.items():
        if raw in aliases:
            return k
    try:
        raw = input("Scope (central_turf/central_dirt/local) [central_dirt]: ").strip().lower()
    except EOFError:
        raw = ""
    for k, aliases in SCOPE_ALIASES.items():
        if raw in aliases:
            return k
    return "central_dirt"


SCOPE = resolve_scope()
print(f"[INFO] scope={SCOPE}")

# ============================================================
# 2. Data Loading
# ============================================================
kachiuma = pd.read_csv("kachiuma.csv")
shusso = pd.read_csv("shutuba.csv")

for df in (kachiuma, shusso):
    df.columns = [re.sub(r"\s+", "", str(c or "")).strip() for c in df.columns]

_before = len(kachiuma)
kachiuma = (
    kachiuma.drop(columns=["finish_pos", "is_top3"], errors="ignore")
    .drop_duplicates(subset=["HorseName", "日付"])
    .reset_index(drop=True)
)
if _before != len(kachiuma):
    print(f"[INFO] kachiuma dedup: {_before} -> {len(kachiuma)} rows")

odds_df = None
if os.path.exists("odds.csv"):
    odds_df = pd.read_csv("odds.csv")
    odds_df.columns = [re.sub(r"\s+", "", str(c or "")).strip() for c in odds_df.columns]
    print(f"[INFO] Loaded odds.csv: {len(odds_df)} horses")
else:
    print("[WARN] odds.csv not found. Predictions will lack market signal.")

fuku_odds_df = None
if os.path.exists("fuku_odds.csv"):
    fuku_odds_df = pd.read_csv("fuku_odds.csv")
    fuku_odds_df.columns = [re.sub(r"\s+", "", str(c or "")).strip() for c in fuku_odds_df.columns]
    print(f"[INFO] Loaded fuku_odds.csv: {len(fuku_odds_df)} horses")
else:
    print("[WARN] fuku_odds.csv not found. Place odds blend will be skipped.")


# ============================================================
# 3. Parsing Utilities
# ============================================================
def parse_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).replace(",", "").replace('"', "").strip())
    except Exception:
        return np.nan


def parse_int(x):
    if pd.isna(x):
        return np.nan
    try:
        return int(float(str(x).strip()))
    except Exception:
        return np.nan


def parse_surface_distance(s):
    if pd.isna(s):
        return ("", np.nan)
    m = re.match(r"(芝|ダ|障)(\d+)", str(s).strip())
    return (m.group(1), int(m.group(2))) if m else ("", np.nan)


def parse_sex_age(val):
    if pd.isna(val):
        return ("", np.nan)
    m = re.match(r"(牡|牝|セ)(\d+)", str(val).strip())
    return (m.group(1), int(m.group(2))) if m else ("", np.nan)


def parse_run_positions(val):
    if pd.isna(val):
        return []
    parts = re.split(r"[-\s]+", str(val).strip())
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return out


def parse_horse_weight(val):
    if pd.isna(val):
        return (np.nan, np.nan)
    s = str(val).strip()
    m = re.match(r"(\d+)\s*\(([+-]?\d+)\)", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m2 = re.match(r"(\d+)", s)
    if m2:
        return (float(m2.group(1)), np.nan)
    return (np.nan, np.nan)


def parse_track_condition(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    mapping = {"良": 0.0, "稍重": 1.0, "稍": 1.0, "重": 2.0, "不良": 3.0}
    return mapping.get(s, np.nan)


def normalize_jockey_id(val):
    if pd.isna(val):
        return ""
    s = str(val).strip().lstrip("0")
    return s or "0"


# ============================================================
# 4. Base Enrichment
# ============================================================
def enrich(df):
    d = df.copy()

    # Surface & Distance
    sd = d["距離"].apply(parse_surface_distance)
    d["Surface"] = sd.apply(lambda t: t[0])
    d["Distance"] = sd.apply(lambda t: t[1])

    # Track condition
    d["TrackCond"] = d["馬場"].apply(parse_track_condition) if "馬場" in d.columns else np.nan

    # Core stats
    d["TimeIndex"] = d["ﾀｲﾑ指数"].apply(parse_float)
    d["Uphill"] = d["上り"].apply(parse_float)

    # 馬場指数 (Track Speed Index): negative=slow track, positive=fast track
    d["BabaIndex"] = d["馬場指数"].apply(parse_float) if "馬場指数" in d.columns else np.nan

    # Adjusted TI = TI - 馬場指数: removes track speed bias, reveals pure horse ability
    d["TI_Adjusted"] = d["TimeIndex"] - d["BabaIndex"]

    # TI efficiency: TI / Uphill (higher = runs fast finish AND good time)
    up_safe = d["Uphill"].clip(lower=30.0)
    d["TI_Efficiency"] = d["TimeIndex"] / up_safe

    # Rank & FieldSize
    d["Rank"] = d["着順"].apply(parse_int) if "着順" in d.columns else np.nan
    d["FieldSize"] = d["頭数"].apply(parse_int) if "頭数" in d.columns else np.nan

    # Odds & Popularity (pre-race info, no leakage)
    d["Odds"] = d["オッズ"].apply(parse_float) if "オッズ" in d.columns else np.nan
    d["Popularity"] = d["人気"].apply(parse_int) if "人気" in d.columns else np.nan

    # Draw & Weight
    d["Draw"] = d["枠番"].apply(parse_int) if "枠番" in d.columns else np.nan
    d["HorseNo"] = d["馬番"].apply(parse_int) if "馬番" in d.columns else np.nan
    d["WeightCarried"] = d["斤量"].apply(parse_float) if "斤量" in d.columns else np.nan

    # Date
    d["Date"] = pd.to_datetime(d["日付"], errors="coerce") if "日付" in d.columns else pd.NaT

    # Sex & Age
    sa = d["SexAge"].apply(parse_sex_age)
    d["Sex"] = sa.apply(lambda t: t[0])
    d["Age"] = sa.apply(lambda t: t[1])
    d["IsFemale"] = (d["Sex"] == "牝").astype(float)
    d["IsGelding"] = (d["Sex"] == "セ").astype(float)

    # Horse weight & change (paddock info, pre-race)
    if "馬体重" in d.columns:
        hw = d["馬体重"].apply(parse_horse_weight)
        d["HorseWeight"] = hw.apply(lambda t: t[0])
        d["HorseWeightChange"] = hw.apply(lambda t: t[1])
    else:
        d["HorseWeight"] = np.nan
        d["HorseWeightChange"] = np.nan

    # Running positions
    if "通過" in d.columns:
        runs = d["通過"].apply(parse_run_positions)
        d["RunFirst"] = runs.apply(lambda r: r[0] if r else np.nan)
        d["RunLast"] = runs.apply(lambda r: r[-1] if r else np.nan)
    else:
        d["RunFirst"] = np.nan
        d["RunLast"] = np.nan

    fs_safe = d["FieldSize"].clip(lower=1)
    d["RunFirstPct"] = d["RunFirst"] / fs_safe
    d["RunLastPct"] = d["RunLast"] / fs_safe
    d["RunGain"] = d["RunFirstPct"] - d["RunLastPct"]  # positive = improved position

    # Prize money (class indicator)
    d["Prize"] = d["賞金"].apply(parse_float) if "賞金" in d.columns else np.nan

    # Race key for grouping
    venue = d["開催"].fillna("").astype(str) if "開催" in d.columns else ""
    race_no = d["R"].fillna("").astype(str) if "R" in d.columns else ""
    d["RaceKey"] = d["Date"].astype(str) + "|" + venue + "|" + race_no

    # Derived targets
    d["IsTop3"] = (d["Rank"] <= 3).astype(float)
    d["IsWin"] = (d["Rank"] == 1).astype(float)
    d["FinishPct"] = d["Rank"] / fs_safe

    return d


# ============================================================
# 5. Jockey Scoring
# ============================================================
def compute_jockey_scores(df):
    d = df.sort_values("Date").copy()
    if "JockeyId" not in d.columns:
        d["JockeyScore"] = 0.3
        return d

    d["_jid"] = d["JockeyId"].fillna("").astype(str).str.strip()
    valid = d["_jid"] != ""

    prior = 20.0
    default_rate = 0.3

    cum_count = d.loc[valid].groupby("_jid").cumcount()
    cum_top3 = d.loc[valid].groupby("_jid")["IsTop3"].cumsum()
    cum_top3 = cum_top3.groupby(d.loc[valid, "_jid"]).shift(1).fillna(0)
    d.loc[valid, "JockeyScore"] = (cum_top3 + prior * default_rate) / (cum_count + prior)
    d["JockeyScore"] = d["JockeyScore"].fillna(default_rate)

    d = d.drop(columns=["_jid"])
    return d


def build_jockey_score_map(hist_df):
    d = hist_df.copy()
    if "JockeyId" not in d.columns:
        return {}
    d["_jid"] = d["JockeyId"].fillna("").astype(str).str.strip()
    d = d[d["_jid"] != ""]
    if d.empty:
        return {}
    latest = d.sort_values("Date").groupby("_jid")["JockeyScore"].last()
    return latest.to_dict()


# ============================================================
# 6. Race Condition Input
# ============================================================
def prompt_race_condition():
    env_surface = str(os.environ.get("PREDICTOR_TARGET_SURFACE", "")).strip().lower()
    if env_surface in ("d", "dirt", "sand", "ダ", "2"):
        default_surface = "ダ"
    elif env_surface in ("t", "turf", "grass", "shiba", "芝", "1"):
        default_surface = "芝"
    elif SCOPE == "central_turf":
        default_surface = "芝"
    else:
        default_surface = "ダ"

    default_distance = 1800
    env_distance = str(os.environ.get("PREDICTOR_TARGET_DISTANCE", "")).strip()
    if env_distance:
        try:
            default_distance = int(float(env_distance))
        except Exception:
            default_distance = 1800

    default_track_text = str(os.environ.get("PREDICTOR_TARGET_CONDITION", "")).strip() or "良"
    no_prompt = _env_truthy("PREDICTOR_NO_PROMPT")

    surf_raw = ""
    dist_raw = ""
    track_raw = ""
    if not no_prompt:
        try:
            default_label = "dirt" if default_surface == "ダ" else "turf"
            surf_raw = input(f"Surface (turf/dirt) [{default_label}]: ").strip().lower()
        except EOFError:
            surf_raw = ""
        try:
            dist_raw = input(f"Distance (meters) [{default_distance}]: ").strip()
        except EOFError:
            dist_raw = ""
        try:
            track_raw = input(f"Track (良/稍重/重/不良) [{default_track_text}]: ").strip()
        except EOFError:
            track_raw = ""

    if surf_raw in ("d", "dirt", "ダ", "sand"):
        surface = "ダ"
    elif surf_raw in ("t", "turf", "grass", "shiba", "芝"):
        surface = "芝"
    else:
        surface = default_surface

    try:
        distance = int(dist_raw) if dist_raw else default_distance
    except Exception:
        distance = default_distance

    track_cond = parse_track_condition(track_raw or default_track_text)
    if not np.isfinite(track_cond):
        track_cond = 0.0

    label = "turf" if surface == "芝" else "dirt"
    cond_names = {0.0: "良", 1.0: "稍重", 2.0: "重", 3.0: "不良"}
    print(f"[INFO] Race: {label} {distance}m {cond_names.get(track_cond, '?')}")
    return surface, distance, track_cond


# ============================================================
# 7. Build Training Data
# ============================================================
def build_training_data(hist_df, target_surface, target_distance):
    """
    训练数据构建 (严格无泄漏):
    - 历史特征: groupby(HorseName).shift(1) + rolling -> 只用过去数据
    - 赛前特征: 当场赔率/人气/场地/斤量等 -> 赛前即可获得
    - 目标: 当场是否前3
    """
    d = hist_df.sort_values(["HorseName", "Date"]).copy()
    g = d.groupby("HorseName")

    d["_n"] = g.cumcount()  # 0-indexed race count

    # ---- Lag result features (shift by 1: only prior race results) ----
    lag_cols = [
        "TimeIndex", "Uphill", "IsTop3", "IsWin", "FinishPct",
        "RunFirstPct", "RunLastPct", "RunGain", "Prize",
        "TI_Adjusted", "TI_Efficiency",
    ]
    for col in lag_cols:
        if col in d.columns:
            d[f"_L_{col}"] = g[col].shift(1)
        else:
            d[f"_L_{col}"] = np.nan

    # ---- Rolling/Expanding on lagged values ----
    def _roll(col, w, fn="mean"):
        s = d[f"_L_{col}"]
        r = s.groupby(d["HorseName"]).rolling(w, min_periods=1)
        return getattr(r, fn)().reset_index(level=0, drop=True)

    def _expand(col, fn="mean"):
        s = d[f"_L_{col}"]
        r = s.groupby(d["HorseName"]).expanding(min_periods=1)
        return getattr(r, fn)().reset_index(level=0, drop=True)

    # --- Time Index ---
    d["f_ti_last"] = d["_L_TimeIndex"]
    d["f_ti_mean3"] = _roll("TimeIndex", 3)
    d["f_ti_mean5"] = _roll("TimeIndex", 5)
    d["f_ti_max5"] = _roll("TimeIndex", 5, "max")
    d["f_ti_std5"] = _roll("TimeIndex", 5, "std")
    d["f_ti_trend"] = d["f_ti_last"] - d["f_ti_mean3"]

    # --- Adjusted TI (TI - 馬場指数 = pure ability, removes track bias) ---
    d["f_ti_adj_last"] = d["_L_TI_Adjusted"]
    d["f_ti_adj_mean5"] = _roll("TI_Adjusted", 5)
    d["f_ti_adj_max5"] = _roll("TI_Adjusted", 5, "max")
    d["f_ti_adj_trend"] = d["f_ti_adj_last"] - d["f_ti_adj_mean5"]

    # --- TI Efficiency (TI / Uphill: high = strong finisher with good time) ---
    d["f_ti_eff_mean5"] = _roll("TI_Efficiency", 5)

    # --- Performance History ---
    d["f_top3_rate5"] = _roll("IsTop3", 5)
    d["f_top3_rate_all"] = _expand("IsTop3")
    d["f_win_rate5"] = _roll("IsWin", 5)
    d["f_avg_finish5"] = _roll("FinishPct", 5)
    d["f_best_finish5"] = _roll("FinishPct", 5, "min")

    # --- Finishing Speed ---
    d["f_up_mean5"] = _roll("Uphill", 5)
    d["f_up_best5"] = _roll("Uphill", 5, "min")

    # --- Running Style ---
    d["f_run_first5"] = _roll("RunFirstPct", 5)
    d["f_run_last5"] = _roll("RunLastPct", 5)
    d["f_run_gain5"] = _roll("RunGain", 5)

    # --- Rest Days ---
    d["f_rest_days"] = g["Date"].diff().dt.days

    # --- Horse Weight (paddock, pre-race) ---
    d["f_horse_weight"] = d["HorseWeight"]
    d["f_weight_change"] = d["HorseWeightChange"]

    # --- Career ---
    d["f_career_races"] = d["_n"]

    # --- Prize/Class ---
    d["f_avg_prize"] = _expand("Prize")
    d["f_max_prize"] = _expand("Prize", "max")

    # --- Surface Experience ---
    d["_surf_match"] = (d["Surface"] == target_surface).astype(float)
    d["_L_surf"] = g["_surf_match"].shift(1)
    d["f_surface_exp_ratio"] = (
        d["_L_surf"]
        .groupby(d["HorseName"])
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    d["f_surface_exp_count"] = (
        d["_L_surf"]
        .groupby(d["HorseName"])
        .expanding(min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # --- Distance Experience ---
    d["_dist_close"] = ((d["Distance"] - target_distance).abs() <= 200).astype(float)
    d["_L_dist"] = g["_dist_close"].shift(1)
    d["f_dist_exp_count"] = (
        d["_L_dist"]
        .groupby(d["HorseName"])
        .expanding(min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # --- Condition-Specific TI (赔率无法完全反映的适性信息) ---
    # TI only on matching surface (turf specialist vs dirt specialist)
    d["_ti_surf"] = d["TimeIndex"].where(d["Surface"] == target_surface, np.nan)
    d["_L_ti_surf"] = g["_ti_surf"].shift(1)
    d["f_ti_surface_mean"] = (
        d["_L_ti_surf"].groupby(d["HorseName"])
        .expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    )
    d["f_ti_surface_max"] = (
        d["_L_ti_surf"].groupby(d["HorseName"])
        .expanding(min_periods=1).max().reset_index(level=0, drop=True)
    )
    # TI only at similar distance (±200m)
    d["_ti_dist"] = d["TimeIndex"].where(
        (d["Distance"] - target_distance).abs() <= 200, np.nan
    )
    d["_L_ti_dist"] = g["_ti_dist"].shift(1)
    d["f_ti_dist_mean"] = (
        d["_L_ti_dist"].groupby(d["HorseName"])
        .expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    )

    # --- Form Momentum (连胜动量: 62.9% vs 25.3%) ---
    d["f_top3_last3"] = _roll("IsTop3", 3, "sum")   # 0~3: last 3 races top3 count

    # --- Finish Consistency (稳定性: 51.6% vs 34.0%) ---
    d["f_finish_std5"] = _roll("FinishPct", 5, "std")

    # ---- Pre-race features (NO lag, available before race) ----
    d["f_odds"] = d["Odds"]
    d["f_odds_implied"] = 1.0 / d["Odds"].clip(lower=1.0)
    d["f_popularity"] = d["Popularity"]
    d["f_popularity_pct"] = d["Popularity"] / d["FieldSize"].clip(lower=1)
    d["f_field_size"] = d["FieldSize"]
    d["f_draw"] = d["Draw"]
    d["f_weight_carried"] = d["WeightCarried"]
    d["f_age"] = d["Age"]
    d["f_is_female"] = d["IsFemale"]
    d["f_is_gelding"] = d["IsGelding"]
    d["f_jockey_score"] = d["JockeyScore"]

    # ---- Target ----
    d["y"] = d["IsTop3"]

    # ---- Filter: need >= 1 prior race ----
    d = d[d["_n"] >= 1].copy()
    d = d.dropna(subset=["y"])
    d["y"] = d["y"].astype(int)

    return d


# ============================================================
# 8. Build Prediction Profiles
# ============================================================
def build_prediction_profiles(shusso_df, cur_odds_df, target_surface, target_distance,
                              jockey_map, race_date=None):
    d = shusso_df.sort_values(["HorseName", "Date"]).copy()

    profiles = []
    for horse_name, h in d.groupby("HorseName"):
        n = len(h)
        recent = h.tail(5)
        last3 = h.tail(3)

        out = {"HorseName": horse_name}

        # --- TI ---
        ti = h["TimeIndex"].dropna()
        ti_r = recent["TimeIndex"].dropna()
        ti_l3 = last3["TimeIndex"].dropna()
        out["f_ti_last"] = float(ti.iloc[-1]) if len(ti) else np.nan
        out["f_ti_mean3"] = float(ti_l3.mean()) if len(ti_l3) else np.nan
        out["f_ti_mean5"] = float(ti_r.mean()) if len(ti_r) else np.nan
        out["f_ti_max5"] = float(ti_r.max()) if len(ti_r) else np.nan
        out["f_ti_std5"] = float(ti_r.std()) if len(ti_r) >= 2 else np.nan
        _ti_last = out["f_ti_last"]
        _ti_m3 = out["f_ti_mean3"]
        out["f_ti_trend"] = (
            _ti_last - _ti_m3
            if pd.notna(_ti_last) and np.isfinite(_ti_last) and pd.notna(_ti_m3) and np.isfinite(_ti_m3)
            else np.nan
        )

        # --- Adjusted TI (pure ability) ---
        tia = recent["TI_Adjusted"].dropna()
        out["f_ti_adj_last"] = float(tia.iloc[-1]) if len(tia) else np.nan
        out["f_ti_adj_mean5"] = float(tia.mean()) if len(tia) else np.nan
        out["f_ti_adj_max5"] = float(tia.max()) if len(tia) else np.nan
        _tia_last = out["f_ti_adj_last"]
        _tia_m5 = out["f_ti_adj_mean5"]
        out["f_ti_adj_trend"] = (
            _tia_last - _tia_m5
            if pd.notna(_tia_last) and np.isfinite(_tia_last) and pd.notna(_tia_m5) and np.isfinite(_tia_m5)
            else np.nan
        )

        # --- TI Efficiency ---
        tie = recent["TI_Efficiency"].dropna()
        out["f_ti_eff_mean5"] = float(tie.mean()) if len(tie) else np.nan

        # --- Performance ---
        top3_r = recent["IsTop3"].dropna()
        top3_all = h["IsTop3"].dropna()
        win_r = recent["IsWin"].dropna()
        _top3_prior, _win_prior, _bk = 0.33, 0.10, 5
        _n5 = len(top3_r)
        out["f_top3_rate5"] = (float(top3_r.sum()) + _top3_prior * _bk) / (_n5 + _bk) if _n5 > 0 else _top3_prior
        _nall = len(top3_all)
        out["f_top3_rate_all"] = (float(top3_all.sum()) + _top3_prior * _bk) / (_nall + _bk) if _nall > 0 else _top3_prior
        _nw5 = len(win_r)
        out["f_win_rate5"] = (float(win_r.sum()) + _win_prior * _bk) / (_nw5 + _bk) if _nw5 > 0 else _win_prior

        fp = recent["FinishPct"].dropna()
        out["f_avg_finish5"] = float(fp.mean()) if len(fp) else np.nan
        out["f_best_finish5"] = float(fp.min()) if len(fp) else np.nan

        # --- Uphill ---
        up = recent["Uphill"].dropna()
        out["f_up_mean5"] = float(up.mean()) if len(up) else np.nan
        out["f_up_best5"] = float(up.min()) if len(up) else np.nan

        # --- Running Style ---
        rf = recent["RunFirstPct"].dropna()
        rl = recent["RunLastPct"].dropna()
        rg = recent["RunGain"].dropna()
        out["f_run_first5"] = float(rf.mean()) if len(rf) else np.nan
        out["f_run_last5"] = float(rl.mean()) if len(rl) else np.nan
        out["f_run_gain5"] = float(rg.mean()) if len(rg) else np.nan

        # --- Rest Days ---
        _race_dt = race_date if race_date is not None else datetime.now()
        if n >= 1 and pd.notna(h["Date"].iloc[-1]):
            out["f_rest_days"] = (_race_dt - h["Date"].iloc[-1]).days
        else:
            out["f_rest_days"] = np.nan

        # --- Horse Weight ---
        hw = h["HorseWeight"].dropna()
        hwc = h["HorseWeightChange"].dropna()
        out["f_horse_weight"] = float(hw.iloc[-1]) if len(hw) else np.nan
        out["f_weight_change"] = float(hwc.iloc[-1]) if len(hwc) else np.nan

        # --- Career ---
        out["f_career_races"] = n

        # --- Prize/Class ---
        pr = h["Prize"].dropna()
        out["f_avg_prize"] = float(pr.mean()) if len(pr) else np.nan
        out["f_max_prize"] = float(pr.max()) if len(pr) else np.nan

        # --- Surface/Distance Experience ---
        surf_match = (h["Surface"] == target_surface).astype(float)
        out["f_surface_exp_ratio"] = float(surf_match.mean())
        out["f_surface_exp_count"] = float(surf_match.sum())
        dist_close = ((h["Distance"] - target_distance).abs() <= 200).astype(float)
        out["f_dist_exp_count"] = float(dist_close.sum())

        # --- Condition-Specific TI ---
        ti_surf = h.loc[h["Surface"] == target_surface, "TimeIndex"].dropna()
        out["f_ti_surface_mean"] = float(ti_surf.mean()) if len(ti_surf) else np.nan
        out["f_ti_surface_max"] = float(ti_surf.max()) if len(ti_surf) else np.nan
        ti_dist = h.loc[(h["Distance"] - target_distance).abs() <= 200, "TimeIndex"].dropna()
        out["f_ti_dist_mean"] = float(ti_dist.mean()) if len(ti_dist) else np.nan

        # --- Form Momentum ---
        last3_top3 = h.tail(3)["IsTop3"].dropna()
        out["f_top3_last3"] = float(last3_top3.sum()) if len(last3_top3) else np.nan

        # --- Finish Consistency ---
        out["f_finish_std5"] = float(fp.std()) if len(fp) >= 2 else np.nan

        # --- Age & Sex ---
        out["f_age"] = float(h["Age"].iloc[-1]) if h["Age"].notna().any() else np.nan
        out["f_is_female"] = float(h["IsFemale"].iloc[-1]) if h["IsFemale"].notna().any() else np.nan
        out["f_is_gelding"] = float(h["IsGelding"].iloc[-1]) if h["IsGelding"].notna().any() else np.nan

        # --- Jockey ---
        if "JockeyId_current" in h.columns and h["JockeyId_current"].notna().any():
            jid = normalize_jockey_id(h["JockeyId_current"].dropna().iloc[-1])
            out["f_jockey_score"] = jockey_map.get(jid, 0.3)
        elif "JockeyId" in h.columns and h["JockeyId"].notna().any():
            jid = normalize_jockey_id(h["JockeyId"].dropna().iloc[-1])
            out["f_jockey_score"] = jockey_map.get(jid, 0.3)
        else:
            out["f_jockey_score"] = 0.3

        # Placeholders for current-race pre-race features
        out["f_odds"] = np.nan
        out["f_odds_implied"] = np.nan
        out["f_popularity"] = np.nan
        out["f_popularity_pct"] = np.nan
        out["f_field_size"] = np.nan
        out["f_draw"] = np.nan
        out["f_weight_carried"] = np.nan
        out["horse_no"] = np.nan

        profiles.append(out)

    prof_df = pd.DataFrame(profiles)

    # ---- Merge current odds ----
    if cur_odds_df is not None and not cur_odds_df.empty:
        odds_map = {}
        for _, row in cur_odds_df.iterrows():
            name = str(row.get("name", "")).strip()
            if name:
                odds_map[name] = {
                    "odds": parse_float(row.get("odds")),
                    "horse_no": parse_int(row.get("horse_no")),
                }

        field_size = len(odds_map)

        # Derive popularity from odds rank (lower odds = more popular)
        sorted_names = sorted(
            odds_map.keys(),
            key=lambda nm: odds_map[nm]["odds"] if np.isfinite(odds_map[nm].get("odds", np.nan)) else 9999,
        )
        for rank, name in enumerate(sorted_names, 1):
            odds_map[name]["popularity"] = rank

        for i, row in prof_df.iterrows():
            name = row["HorseName"]
            if name in odds_map:
                info = odds_map[name]
                o = info["odds"]
                prof_df.at[i, "f_odds"] = o
                prof_df.at[i, "f_odds_implied"] = 1.0 / max(o, 1.0) if np.isfinite(o) else np.nan
                prof_df.at[i, "f_popularity"] = info["popularity"]
                prof_df.at[i, "f_popularity_pct"] = info["popularity"] / field_size if field_size else np.nan
                prof_df.at[i, "f_field_size"] = field_size
                prof_df.at[i, "horse_no"] = info["horse_no"]

    return prof_df


# ============================================================
# 9. Feature List
# ============================================================
FEATURES = [
    # ---- Market Signal (最强信号) ----
    "f_odds",
    "f_odds_implied",
    "f_popularity",
    "f_popularity_pct",
    "f_field_size",
    # ---- Race Context ----
    "f_draw",
    "f_weight_carried",
    # ---- Time Index (实力核心) ----
    "f_ti_last",
    "f_ti_mean3",
    "f_ti_mean5",
    "f_ti_max5",
    "f_ti_std5",
    "f_ti_trend",
    # ---- Adjusted TI = TI - 馬場指数 (纯粹能力, 消除赛场偏差) ----
    "f_ti_adj_last",
    "f_ti_adj_mean5",
    "f_ti_adj_max5",
    "f_ti_adj_trend",
    # ---- TI Efficiency = TI / 上がり (高 = 末脚快 + 整体强) ----
    "f_ti_eff_mean5",
    # ---- Performance History ----
    "f_top3_rate5",
    "f_top3_rate_all",
    "f_win_rate5",
    "f_avg_finish5",
    "f_best_finish5",
    # ---- Finishing Speed ----
    "f_up_mean5",
    "f_up_best5",
    # ---- Running Style ----
    "f_run_first5",
    "f_run_last5",
    "f_run_gain5",
    # ---- Condition ----
    "f_rest_days",
    "f_horse_weight",
    "f_weight_change",
    # ---- Horse Attributes ----
    "f_age",
    "f_is_female",
    "f_is_gelding",
    # ---- Experience ----
    "f_career_races",
    "f_surface_exp_ratio",
    "f_surface_exp_count",
    "f_dist_exp_count",
    # ---- Condition-Specific TI (赔率无法反映的适性) ----
    "f_ti_surface_mean",
    "f_ti_surface_max",
    "f_ti_dist_mean",
    # ---- Form Momentum (连胜动量) ----
    "f_top3_last3",
    # ---- Finish Consistency (成绩稳定性) ----
    "f_finish_std5",
    # ---- Class Level ----
    "f_avg_prize",
    "f_max_prize",
    # ---- Jockey ----
    "f_jockey_score",
]


# ============================================================
# 10. Model Training
# ============================================================
def fit_platt_scaler(probs, labels):
    valid = np.isfinite(probs) & np.isfinite(labels)
    p = probs[valid]
    y = labels[valid]
    if len(p) < 30 or len(np.unique(y)) < 2:
        return None
    eps = 1e-7
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    scaler = LogisticRegression(max_iter=2000, C=1e10, solver="lbfgs")
    scaler.fit(logit.reshape(-1, 1), y)
    return scaler


def apply_platt(scaler, probs):
    if scaler is None:
        return probs
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return scaler.predict_proba(logit.reshape(-1, 1))[:, 1]


def build_lgb(variant="A", verbose=-1):
    """Two LGB variants for ensemble diversity."""
    if variant == "A":
        return LGBMClassifier(
            n_estimators=1500, learning_rate=0.02, num_leaves=24,
            max_depth=-1, min_child_samples=25, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.1,
            reg_lambda=1.0, importance_type="gain", random_state=42,
            n_jobs=LGBM_N_JOBS, verbose=verbose,
        )
    else:  # variant B: shallower, more regularized, different seed
        return LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=16,
            max_depth=6, min_child_samples=40, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.6, reg_alpha=0.5,
            reg_lambda=2.0, importance_type="gain", random_state=123,
            n_jobs=LGBM_N_JOBS, verbose=verbose,
        )


def train_and_evaluate(train_df, features):
    """
    时序交叉验证 + 最终模型训练:
    - Expanding window CV (只用过去预测未来, 严格防泄漏)
    - Platt校准 (OOF概率 -> 校准概率)
    - 最终模型在全量数据训练
    """
    d = train_df.sort_values("Date").reset_index(drop=True)

    for col in features:
        if col not in d.columns:
            d[col] = np.nan

    X = d[features]
    y = d["y"].astype(int)

    pos = int(y.sum())
    neg = int((y == 0).sum())
    print(f"[INFO] Training: {len(d)} samples (pos={pos}, neg={neg}, rate={pos/max(pos+neg,1):.3f})")

    if pos == 0 or neg == 0:
        print("[WARN] Single class. Returning constant model.")
        return None, None, float(y.mean())

    # ---- Expanding Window Time-Series CV ----
    n = len(d)
    n_splits = 5
    min_train_frac = 0.40
    test_frac = 0.12

    min_train = int(n * min_train_frac)
    test_size = max(int(n * test_frac), 50)

    oof_probs = np.full(n, np.nan)
    metrics = []

    for fold in range(n_splits):
        train_end = min_train + fold * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_start >= n or test_end <= test_start:
            break

        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

        if len(y_tr.unique()) < 2 or len(y_te.unique()) < 2 or len(X_tr) < 50:
            continue

        # Ensemble: train both variants and average
        # Split a validation set from training data for early stopping (avoid using test fold)
        val_split = int(len(X_tr) * 0.85)
        X_tr_inner, y_tr_inner = X_tr.iloc[:val_split], y_tr.iloc[:val_split]
        X_va_inner, y_va_inner = X_tr.iloc[val_split:], y_tr.iloc[val_split:]
        has_inner_valid = len(y_va_inner.unique()) >= 2 and len(X_va_inner) >= 20

        fold_probs_list = []
        for variant in ("A", "B"):
            fm = build_lgb(variant=variant)
            if has_inner_valid:
                fm.fit(
                    X_tr_inner, y_tr_inner,
                    eval_set=[(X_va_inner, y_va_inner)],
                    eval_metric="logloss",
                    callbacks=[early_stopping(100, verbose=False)],
                )
            else:
                fm.fit(X_tr, y_tr)
            fold_probs_list.append(fm.predict_proba(X_te)[:, 1])
        probs = np.mean(fold_probs_list, axis=0)
        oof_probs[test_start:test_end] = probs

        auc = roc_auc_score(y_te, probs)
        ll = log_loss(y_te, probs)

        # Hit rate by race
        test_data = d.iloc[test_start:test_end].copy()
        test_data["_prob"] = probs
        hits = total_slots = races = hit1 = 0
        for _, grp in test_data.groupby("RaceKey"):
            if len(grp) < 3:
                continue
            races += 1
            top_pred = grp.nlargest(3, "_prob")
            hits += int(top_pred["y"].sum())
            total_slots += min(3, len(top_pred))
            hit1 += int(grp.nlargest(1, "_prob")["y"].sum())

        hr3 = hits / total_slots if total_slots else np.nan
        hr1 = hit1 / races if races else np.nan
        print(
            f"[INFO] CV fold {fold+1}/{n_splits}: "
            f"auc={auc:.4f} ll={ll:.4f} hit@3={hr3:.4f} hit@1={hr1:.4f} "
            f"(train={len(X_tr)}, test={len(X_te)}, races={races})"
        )
        metrics.append((auc, ll, hr3, hr1))

    if metrics:
        avg = [np.nanmean([m[i] for m in metrics]) for i in range(4)]
        print(
            f"[INFO] CV avg: auc={avg[0]:.4f} ll={avg[1]:.4f} "
            f"hit@3={avg[2]:.4f} hit@1={avg[3]:.4f}"
        )

    # ---- Platt Calibration from OOF ----
    valid_oof = np.isfinite(oof_probs)
    platt = None
    if valid_oof.sum() > 50:
        platt = fit_platt_scaler(oof_probs[valid_oof], y.values[valid_oof].astype(float))
        if platt is not None:
            cal = apply_platt(platt, oof_probs[valid_oof])
            print(
                f"[INFO] Platt: raw_mean={oof_probs[valid_oof].mean():.4f} "
                f"cal_mean={cal.mean():.4f} label_mean={y.values[valid_oof].mean():.4f}"
            )

    # ---- Final Ensemble on All Data ----
    # Step 1: Use last 15% as validation to find best iteration count
    print("[INFO] Training final ensemble (2x LGB) on all data...")
    split_idx = int(n * 0.85)
    X_tr_f, y_tr_f = X.iloc[:split_idx], y.iloc[:split_idx]
    X_va_f, y_va_f = X.iloc[split_idx:], y.iloc[split_idx:]
    has_valid = len(y_va_f.unique()) >= 2 and len(X_va_f) >= 20

    final_models = []
    for variant in ("A", "B"):
        if has_valid:
            # First pass: find best iteration via early stopping on validation split
            m_probe = build_lgb(variant=variant)
            m_probe.fit(
                X_tr_f, y_tr_f,
                eval_set=[(X_va_f, y_va_f)],
                eval_metric="logloss",
                callbacks=[early_stopping(100, verbose=False)],
            )
            best = getattr(m_probe, "best_iteration_", m_probe.n_estimators)
            # Second pass: retrain on ALL data with the best iteration count
            m = build_lgb(variant=variant)
            m.set_params(n_estimators=best)
            m.fit(X, y)
        else:
            m = build_lgb(variant=variant)
            m.fit(X, y)
            best = m.n_estimators
        print(f"[INFO] Model {variant}: best_iteration={best}")
        final_models.append(m)

    # Feature importance (from model A)
    imp = pd.Series(final_models[0].feature_importances_, index=features).sort_values(ascending=False)
    print("\n[INFO] Top 20 Feature Importances:")
    for feat, score in imp.head(20).items():
        print(f"  {feat:30s} {score:8.1f}")
    print()

    return final_models, platt, float(y.mean())


def race_level_normalize(probs, top_k=3):
    total = np.nansum(probs)
    if total <= 0 or not np.isfinite(total):
        return probs
    expected = min(float(top_k), float(len(probs)))
    return probs * (expected / total)


# ============================================================
# 11. Main Execution
# ============================================================
print("=" * 60)
print("predictor_v2.py - 高命中率赛马预测器")
print("=" * 60)

# Enrich
print("[INFO] Enriching data...")
kachi_e = enrich(kachiuma)
shusso_e = enrich(shusso)

# Jockey scores
kachi_e = compute_jockey_scores(kachi_e)
jockey_map = build_jockey_score_map(kachi_e)

# Race condition
RACE_SURFACE, RACE_DISTANCE, RACE_TRACK_COND = prompt_race_condition()

# Build training data
print("[INFO] Building training data...")
train_df = build_training_data(kachi_e, RACE_SURFACE, RACE_DISTANCE)
print(f"[INFO] Training samples: {len(train_df)}")

# Check odds coverage in training
odds_coverage = train_df["f_odds"].notna().mean()
print(f"[INFO] Odds coverage in training: {odds_coverage:.1%}")
if odds_coverage < 0.3:
    print("[WARN] Low odds coverage. Market signal features may be weak.")

# Train
model, platt, base_prob = train_and_evaluate(train_df, FEATURES)

# Build prediction profiles
print("[INFO] Building prediction profiles...")
_race_date = shusso_e["Date"].dropna().max() if "Date" in shusso_e.columns else None
pred_profiles = build_prediction_profiles(
    shusso_e, odds_df, RACE_SURFACE, RACE_DISTANCE, jockey_map,
    race_date=_race_date,
)

# Ensure all features exist
for col in FEATURES:
    if col not in pred_profiles.columns:
        pred_profiles[col] = np.nan

# Predict
print("[INFO] Predicting...")
if model is not None:
    # Ensemble: average predictions from all models
    if isinstance(model, list):
        all_probs = [m.predict_proba(pred_profiles[FEATURES])[:, 1] for m in model]
        raw_probs = np.mean(all_probs, axis=0)
    else:
        raw_probs = model.predict_proba(pred_profiles[FEATURES])[:, 1]
    cal_probs = apply_platt(platt, raw_probs)
else:
    raw_probs = np.full(len(pred_profiles), base_prob)
    cal_probs = raw_probs.copy()

# Race-level normalization
cal_probs = race_level_normalize(cal_probs, top_k=3)

pred_profiles["Top3Prob_model"] = cal_probs
pred_profiles["Top3Prob_raw_lr"] = raw_probs
pred_profiles["Top3Prob_raw_lgb"] = raw_probs
pred_profiles["Top3Prob_cal_lr"] = cal_probs
pred_profiles["Top3Prob_cal_lgb"] = cal_probs
pred_profiles["Top3Prob_lr"] = raw_probs
pred_profiles["Top3Prob_lgbm"] = raw_probs
pred_profiles["rank_score"] = cal_probs
pred_profiles["rank_score_raw"] = raw_probs
pred_profiles["rank_score_norm"] = cal_probs

# --- Place odds post-prediction blend ---
def blend_place_odds(pred_df, fuku_df, alpha=0.12):
    """Blend fukusho implied probability into final scores for re-ranking."""
    if fuku_df is None or fuku_df.empty:
        return pred_df, False

    place_map = {}
    for _, row in fuku_df.iterrows():
        hno = parse_int(row.get("horse_no", np.nan))
        mid = parse_float(row.get("odds_mid", np.nan))
        if pd.notna(hno) and hno > 0 and pd.notna(mid) and mid > 0:
            place_map[int(hno)] = 1.0 / mid

    if not place_map:
        return pred_df, False

    df = pred_df.copy()
    score_col = "rank_score" if "rank_score" in df.columns else "Top3Prob_model"

    s = df[score_col].values.astype(float)
    s_min, s_max = float(np.nanmin(s)), float(np.nanmax(s))
    s_norm = (s - s_min) / (s_max - s_min) if s_max - s_min > 1e-9 else np.full_like(s, 0.5)

    place_vals = df["horse_no"].apply(
        lambda h: place_map.get(int(h), 0.0) if pd.notna(h) and h > 0 else 0.0
    ).values.astype(float)
    p_min, p_max = float(np.nanmin(place_vals)), float(np.nanmax(place_vals))
    p_norm = (place_vals - p_min) / (p_max - p_min) if p_max - p_min > 1e-9 else np.full_like(place_vals, 0.0)

    df[score_col] = (1.0 - alpha) * s_norm + alpha * p_norm
    print(f"[INFO] Place odds blend applied (alpha={alpha}, {len(place_map)} horses matched)")
    return df, True


pred_profiles, place_blend_applied = blend_place_odds(pred_profiles, fuku_odds_df, alpha=0.12)

# Sort by score after all rank adjustments
pred_profiles = pred_profiles.sort_values("rank_score", ascending=False).reset_index(drop=True)

rank_vals = pred_profiles["rank_score"].to_numpy(dtype=float)
rank_min = float(np.nanmin(rank_vals)) if len(rank_vals) else 0.0
rank_max = float(np.nanmax(rank_vals)) if len(rank_vals) else 0.0
if len(rank_vals) and np.isfinite(rank_max - rank_min) and (rank_max - rank_min) > 1e-9:
    pred_profiles["rank_score_norm"] = (rank_vals - rank_min) / (rank_max - rank_min)
else:
    pred_profiles["rank_score_norm"] = np.full(len(pred_profiles), 0.5 if len(pred_profiles) else np.nan)

# Confidence score
scores = pred_profiles["rank_score"].values
if len(scores) >= 3:
    gap = float(scores[0]) - float(scores[2])
    n_runners = max(len(scores), 1)
    expected_gap = 3.0 / n_runners
    confidence = min(gap / expected_gap, 1.0) if expected_gap > 0 else 0.0
else:
    confidence = 0.0

pred_profiles["confidence_score"] = round(confidence, 4)
pred_profiles["stability_score"] = round(confidence, 4)
pred_profiles["validity_score"] = round(confidence, 4)
pred_profiles["consistency_score"] = round(confidence, 4)
pred_profiles["rank_ema"] = 0.5
pred_profiles["ev_ema"] = 0.5
pred_profiles["risk_score"] = round(confidence, 4)

# Compatibility columns
if "horse_no" not in pred_profiles.columns:
    pred_profiles["horse_no"] = np.nan
pred_profiles["model_mode"] = "v2_lgb"
pred_profiles["score_is_probability"] = 0 if place_blend_applied else 1
pred_profiles["race_id"] = "current"

# ============================================================
# 12. Output
# ============================================================
print("\n" + "=" * 60)
print("  Top Predictions")
print("=" * 60)
for i, (_, row) in enumerate(pred_profiles.iterrows()):
    hno = row.get("horse_no", "?")
    hno_str = f"{int(hno):>2}" if pd.notna(hno) and np.isfinite(hno) else " ?"
    name = row["HorseName"]
    odds_val = row.get("f_odds", np.nan)
    pop = row.get("f_popularity", np.nan)
    score = row["rank_score"]
    ti = row.get("f_ti_last", np.nan)
    t3r = row.get("f_top3_rate5", np.nan)

    odds_str = f"odds={odds_val:6.1f}" if pd.notna(odds_val) and np.isfinite(odds_val) else "odds=   N/A"
    pop_str = f"pop={int(pop):2}" if pd.notna(pop) and np.isfinite(pop) else "pop=NA"
    ti_str = f"TI={ti:5.1f}" if pd.notna(ti) and np.isfinite(ti) else "TI=  N/A"
    t3r_str = f"T3R={t3r:.0%}" if pd.notna(t3r) and np.isfinite(t3r) else "T3R= N/A"

    marker = " ***" if i < 3 else "    " if i < 5 else ""
    print(f"  {hno_str}  {name:18s}  {odds_str}  {pop_str}  {ti_str}  {t3r_str}  score={score:.4f}{marker}")

print(f"\nConfidence: {confidence:.4f}")
print(f"Runners: {len(pred_profiles)}")

# Save
pred_profiles.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUTPUT_PATH.name}")

elapsed = datetime.now() - START_TIME
print(f"[INFO] Elapsed: {elapsed}")
if not NO_WAIT:
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        pass

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss
from datetime import datetime
from lightgbm import LGBMClassifier, LGBMRanker, early_stopping


def configure_utf8_io():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_io()

PIPELINE_START = datetime.now()


def resolve_lgbm_n_jobs():
    raw = str(os.environ.get("PREDICTOR_LGBM_N_JOBS", "1") or "1").strip()
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return 1


LGBM_N_JOBS = resolve_lgbm_n_jobs()

# ====================================================
# 1. 数据加载
# ====================================================
kachiuma = pd.read_csv("kachiuma.csv")
shusso = pd.read_csv("shutuba.csv")


def normalize_frame_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"\s+", "", str(col or "")).strip() for col in df.columns]
    return df


kachiuma = normalize_frame_columns(kachiuma)
shusso = normalize_frame_columns(shusso)

# race_id ごとに全出走馬の履歴が格納されるため、同一馬・同一日の
# レコードが複数 race_id にまたがって重複する。finish_pos / is_top3
# は race_id 側の成績であり履歴レコード自体の属性ではないため、
# 除外してから重複を排除する。
_before = len(kachiuma)
kachiuma = (
    kachiuma
    .drop(columns=["finish_pos", "is_top3"], errors="ignore")
    .drop_duplicates(subset=["HorseName", "日付"])
    .reset_index(drop=True)
)
if _before != len(kachiuma):
    print(f"[INFO] kachiuma dedup: {_before} -> {len(kachiuma)} rows")

SCOPE_ALIASES = {
    "central_turf": {"central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"},
    "central_dirt": {"central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"},
    "local": {"local", "l", "3"},
}


def normalize_scope_key(raw):
    raw = (raw or "").strip().lower()
    raw = raw.replace(" ", "_").replace("-", "_").replace("/", "_")
    for key, aliases in SCOPE_ALIASES.items():
        if raw in aliases:
            return key
    return ""


def resolve_scope_key():
    resolved = normalize_scope_key(os.environ.get("SCOPE_KEY", ""))
    if resolved:
        return resolved
    try:
        raw = input("Data scope (central_turf/central_dirt/local) [central_dirt]: ").strip().lower()
    except EOFError:
        raw = ""
    resolved = normalize_scope_key(raw)
    return resolved or "central_dirt"


SCOPE_KEY = resolve_scope_key()
PRED_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "pipeline"
    / f"predictor_config_{SCOPE_KEY}.json"
)
DEFAULT_PARAMS = {
    "place_score_base": 0.2,
    "place_score_weight": 0.8,
    "place_score_fill": 0.5,
    "time_trend_window": 3,
    "time_decay_half_life_days": 120.0,
    "tau_window": 300,
    "tau_min": 300.0,
    "tau_max": 1200.0,
    "smooth_p": 1.2,
    "surf_floor": 0.2,
    "dist_floor": 0.3,
    "record_weight_base": 0.7,
    "record_weight_match": 0.3,
    "recent_race_count": 5,
    "top_score_count": 3,
    "top3_scale": 3.0,
    "model_mode": "hybrid",
    "blend_alpha": 0.3,
    "ranker_label_mode": "top5_gain",
    "ranker_blend_alpha": 0.6,
    "ranker_n_estimators": 300,
    "ranker_learning_rate": 0.05,
    "ranker_num_leaves": 31,
    "ranker_min_child_samples": 20,
    "ranker_subsample": 0.8,
    "ranker_colsample_bytree": 0.8,
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def safe_param_float(params, key, default):
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def safe_param_int(params, key, default):
    try:
        return int(float(params.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def normalize_model_mode(raw):
    text = str(raw or "").strip().lower()
    if text in {"rank", "ranker"}:
        return "ranker"
    if text in {"hybrid", "mix", "blend"}:
        return "hybrid"
    return "classifier"


def _sigmoid(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_rank_score_by_race(scores, race_keys):
    s = pd.Series(scores, dtype=float)
    keys = pd.Series(race_keys, index=s.index).fillna("").astype(str)
    keys = keys.replace("", "__single_race__")

    def _normalize(group):
        arr = group.to_numpy(dtype=float)
        std = float(np.nanstd(arr))
        if not np.isfinite(std) or std < 1e-12:
            z = np.zeros_like(arr, dtype=float)
        else:
            mean = float(np.nanmean(arr))
            z = (arr - mean) / std
        return pd.Series(_sigmoid(z), index=group.index, dtype=float)

    out = s.groupby(keys, group_keys=False).apply(_normalize)
    return out.reindex(s.index).to_numpy(dtype=float)


def build_ranker_relevance(rank_series, mode="top5_gain"):
    rank = pd.to_numeric(rank_series, errors="coerce")
    valid = rank.notna() & (rank >= 1)
    rel = pd.Series(np.nan, index=rank.index, dtype=float)
    mode_key = str(mode or "top5_gain").strip().lower()
    if mode_key == "field_inverse":
        if valid.any():
            max_rank = float(rank[valid].max())
            rel.loc[valid] = (max_rank - rank.loc[valid] + 1.0).clip(lower=0.0)
    elif mode_key == "top3_gain":
        rel.loc[valid] = np.where(rank.loc[valid] <= 3, 4.0 - rank.loc[valid], 0.0)
    else:
        # top5_gain: 1st..5th => 5..1, else 0
        mapped = 6.0 - rank.loc[valid]
        rel.loc[valid] = mapped.where(mapped > 0.0, 0.0)
    return rel



def fit_platt_scaler(oof_probs, oof_labels):
    """OOF 予測確率と実際のラベルから Platt scaling 用 sigmoid を学習する。"""
    valid = np.isfinite(oof_probs) & np.isfinite(oof_labels)
    p = np.asarray(oof_probs)[valid]
    y = np.asarray(oof_labels)[valid]
    if len(p) < 30 or len(np.unique(y)) < 2:
        return None
    eps = 1e-7
    p_clipped = np.clip(p, eps, 1.0 - eps)
    X = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    scaler = LogisticRegression(max_iter=2000, C=1e10, solver="lbfgs")
    scaler.fit(X, y)
    cal_probs = scaler.predict_proba(X)[:, 1]
    print(
        f"[INFO] Platt scaler fit: n={len(p)}, "
        f"raw_mean={p.mean():.4f}, cal_mean={cal_probs.mean():.4f}, "
        f"label_mean={y.mean():.4f}"
    )
    return scaler


def apply_platt_scaler(scaler, probs):
    """学習済み Platt scaler で確率を補正する。"""
    if scaler is None:
        return np.asarray(probs, dtype=float)
    arr = np.asarray(probs, dtype=float)
    eps = 1e-7
    p_clipped = np.clip(arr, eps, 1.0 - eps)
    X = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    return scaler.predict_proba(X)[:, 1]


def find_optimal_blend_alpha(cal_lr, cal_lgb, labels, steps=21):
    """校準済み LR/LGB の OOF 確率から最適ブレンド alpha を log_loss で選ぶ。"""
    valid = np.isfinite(cal_lr) & np.isfinite(cal_lgb) & np.isfinite(labels)
    lr = np.asarray(cal_lr)[valid]
    lgb = np.asarray(cal_lgb)[valid]
    y = np.asarray(labels)[valid]
    if len(y) < 30 or len(np.unique(y)) < 2:
        return 0.5
    best_alpha = 0.5
    best_ll = float("inf")
    for i in range(steps):
        a = i / (steps - 1)
        blended = np.clip(a * lgb + (1.0 - a) * lr, 1e-7, 1.0 - 1e-7)
        ll = log_loss(y, blended)
        if ll < best_ll:
            best_ll = ll
            best_alpha = a
    print(f"[INFO] Optimal blend alpha={best_alpha:.2f} (log_loss={best_ll:.4f})")
    return best_alpha


def race_level_normalize(probs, field_size, top_k=3):
    """同一レースの確率合計が期待値 (top_k) になるよう正規化する。"""
    arr = np.asarray(probs, dtype=float)
    total = float(np.nansum(arr))
    if total <= 0.0 or not np.isfinite(total):
        return arr
    expected = min(float(top_k), float(field_size))
    return arr * (expected / total)


def load_state_from_config(config):
    state = config.get("state", {})
    return {
        "rank_ema": clamp(float(state.get("rank_ema", 0.5)), 0.0, 1.0),
        "ev_ema": clamp(float(state.get("ev_ema", 0.5)), 0.0, 1.0),
        "risk_score": clamp(float(state.get("risk_score", 0.5)), 0.0, 1.0),
    }


def compute_confidence_from_pred(pred_df, state):
    rank_ema = state.get("rank_ema", 0.5)
    ev_ema = state.get("ev_ema", 0.5)
    risk_score = state.get("risk_score", 0.5)
    validity = clamp(0.6 * rank_ema + 0.4 * ev_ema, 0.0, 1.0)
    stability = clamp(risk_score, 0.0, 1.0)
    score_col = "rank_score" if "rank_score" in pred_df.columns else "Top3Prob_model"
    probs = pred_df.sort_values(score_col, ascending=False)[score_col].tolist()
    consistency = 0.0
    if probs:
        if len(probs) >= 3:
            gap = float(probs[0]) - float(probs[2])
        elif len(probs) >= 2:
            gap = float(probs[0]) - float(probs[1])
        else:
            gap = float(probs[0])
        # 分母随出马头数变化: 头数越多 gap 越容易大，用 3/N 做期望基准
        n_runners = max(len(probs), 1)
        expected_gap = 3.0 / n_runners
        consistency = clamp(gap / expected_gap, 0.0, 1.0)
    confidence = math.sqrt(stability * validity) * consistency
    return {
        "confidence_score": clamp(confidence, 0.0, 1.0),
        "stability_score": stability,
        "validity_score": validity,
        "consistency_score": consistency,
    }


def load_predictor_config(path=None):
    path = path or PRED_CONFIG_PATH
    if not path.exists():
        fallback = []
        if SCOPE_KEY == "central_turf":
            fallback.extend(
                [
                    Path(__file__).resolve().parent / "pipeline" / "predictor_config_turf_default.json",
                    Path(__file__).resolve().parent / "pipeline" / "predictor_config_turf.json",
                ]
            )
        elif SCOPE_KEY == "central_dirt":
            fallback.extend(
                [
                    Path(__file__).resolve().parent / "pipeline" / "predictor_config_dirt_default.json",
                    Path(__file__).resolve().parent / "pipeline" / "predictor_config_dirt.json",
                ]
            )
        elif SCOPE_KEY == "local":
            fallback.extend(
                [
                    Path(__file__).resolve().parent / "pipeline" / "predictor_config_central_dirt.json",
                    Path(__file__).resolve().parent / "pipeline" / "predictor_config_dirt_default.json",
                    Path(__file__).resolve().parent / "pipeline" / "predictor_config_dirt.json",
                ]
            )
        fallback.append(Path(__file__).resolve().parent / "pipeline" / "predictor_config.json")
        for legacy in fallback:
            if legacy.exists():
                data = json.loads(legacy.read_text(encoding="utf-8"))
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return data
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_predictor_params(config):
    params = dict(DEFAULT_PARAMS)
    params.update(config.get("params", {}))

    strategies = config.get("strategies", {})
    selected = os.environ.get("PREDICTOR_STRATEGY", "").strip()
    if selected not in strategies:
        selected = config.get("active_strategy", "")
    if selected not in strategies and strategies:
        selected = next(iter(strategies))

    overrides = strategies.get(selected, {}).get("overrides", {})
    for key, value in overrides.items():
        params[key] = value
    return params, selected


_config = load_predictor_config()
PARAMS, STRATEGY_USED = build_predictor_params(_config)
if STRATEGY_USED:
    print(f"[INFO] predictor strategy: {STRATEGY_USED}")

# ====================================================
# 2. 特征处理函数
# ====================================================
def parse_surface_distance(s):
    if isinstance(s, str):
        surface = s[0]  # 芝 / ダ
        m = re.search(r"(\d+)", s)
        dist = int(m.group(1)) if m else np.nan
        return surface, dist
    return np.nan, np.nan

def parse_float(x):
    try:
        return float(x)
    except:
        return np.nan

def parse_int(x):
    try:
        if pd.isna(x):
            return np.nan
        m = re.search(r"\d+", str(x))
        return int(m.group(0)) if m else np.nan
    except Exception:
        return np.nan

RUN_POS_RE = re.compile(r"\d+")
PACE_RE = re.compile(r"\d+(?:\.\d+)?")


def parse_run_positions(value):
    if pd.isna(value):
        return np.nan, np.nan
    nums = RUN_POS_RE.findall(str(value))
    if not nums:
        return np.nan, np.nan
    try:
        first = int(nums[0])
        last = int(nums[-1])
        return first, last
    except Exception:
        return np.nan, np.nan


def parse_pace(value):
    if pd.isna(value):
        return np.nan, np.nan, np.nan
    nums = PACE_RE.findall(str(value))
    if len(nums) < 2:
        return np.nan, np.nan, np.nan
    try:
        first = float(nums[0])
        last = float(nums[1])
        return first, last, first - last
    except Exception:
        return np.nan, np.nan, np.nan


def build_race_key_series(df):
    if "race_id" in df.columns and df["race_id"].notna().any():
        return df["race_id"].astype(str)
    parts = []
    for col in ("日付", "開催", "R", "レース名"):
        if col in df.columns:
            parts.append(df[col].astype(str))
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    key = parts[0].fillna("")
    for col in parts[1:]:
        key = key + "|" + col.fillna("")
    return key


def add_running_style_features(df):
    d = df.copy()
    if "通過" in d.columns:
        positions = d["通過"].apply(parse_run_positions)
        d["RunFirst"] = positions.apply(lambda t: t[0])
        d["RunLast"] = positions.apply(lambda t: t[1])
    else:
        d["RunFirst"] = np.nan
        d["RunLast"] = np.nan

    if "FieldSize" in d.columns:
        field_size = pd.to_numeric(d["FieldSize"], errors="coerce")
        d["RunFirstPct"] = d["RunFirst"] / field_size
        d["RunLastPct"] = d["RunLast"] / field_size
        d["RunGainPct"] = d["RunFirstPct"] - d["RunLastPct"]
    else:
        d["RunFirstPct"] = np.nan
        d["RunLastPct"] = np.nan
        d["RunGainPct"] = np.nan

    if "ペース" in d.columns:
        pace_vals = d["ペース"].apply(parse_pace)
        d["PaceFirst"] = pace_vals.apply(lambda t: t[0])
        d["PaceLast"] = pace_vals.apply(lambda t: t[1])
        d["PaceDiff"] = pace_vals.apply(lambda t: t[2])
    else:
        d["PaceFirst"] = np.nan
        d["PaceLast"] = np.nan
        d["PaceDiff"] = np.nan
    return d


def add_race_normalized_features(df):
    d = df.copy()
    d["_race_key"] = build_race_key_series(d)
    cols = ["RunFirstPct", "RunLastPct", "RunGainPct", "Uphill", "PaceDiff"]
    for col in cols:
        if col not in d.columns:
            d[col] = np.nan
    grouped = d.groupby("_race_key")
    for col in cols:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform(lambda s: s.std(ddof=0))
        std = std.replace(0, np.nan)
        d[f"{col}_z"] = (d[col] - mean) / std
    return d.drop(columns=["_race_key"])

SEX_AGE_RE = re.compile(r"([牡牝セ騸])\s*(\d+)")
SEX_CODE_MAP = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}

def parse_sex_age(value):
    if pd.isna(value):
        return "", np.nan
    text = str(value)
    match = SEX_AGE_RE.search(text)
    if not match:
        return "", np.nan
    return match.group(1), float(match.group(2))

def add_sex_age_features(df):
    sex = pd.Series("", index=df.index, dtype=object)
    age = pd.Series(np.nan, index=df.index, dtype=float)

    for col in ("SexAge", "性齢", "性令", "性龄"):
        if col in df.columns:
            parsed = df[col].apply(parse_sex_age)
            parsed_sex = parsed.apply(lambda t: t[0])
            parsed_age = pd.to_numeric(parsed.apply(lambda t: t[1]), errors="coerce")
            sex = sex.where(sex != "", parsed_sex)
            age = age.where(age.notna(), parsed_age)

    if "Sex" in df.columns:
        fallback_sex = df["Sex"].astype(str).str.strip()
        fallback_sex = fallback_sex.where(fallback_sex.isin(SEX_CODE_MAP), "")
        sex = sex.where(sex != "", fallback_sex)
    if "Age" in df.columns:
        fallback_age = pd.to_numeric(df["Age"], errors="coerce")
        age = age.where(age.notna(), fallback_age)

    df["Sex"] = sex.replace("nan", "")
    df["Age"] = age
    sex_known = df["Sex"] != ""
    df["SexMale"] = np.where(sex_known, (df["Sex"] == "牡").astype(int), np.nan)
    df["SexFemale"] = np.where(sex_known, (df["Sex"] == "牝").astype(int), np.nan)
    df["SexGelding"] = np.where(sex_known, df["Sex"].isin(["セ", "騸"]).astype(int), np.nan)
    return df


def normalize_jockey_id(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "none"):
        return ""
    # Unify format: "01150" and "1150" and 1150 -> "1150"
    try:
        return str(int(float(text)))
    except (ValueError, OverflowError):
        return text


def add_jockey_cum_score(df, prior=20.0):
    d = df.copy()
    if "JockeyId" not in d.columns:
        d["JockeyScore"] = np.nan
        return d
    d["JockeyId"] = d["JockeyId"].apply(normalize_jockey_id)
    if "Rank" in d.columns:
        y = (pd.to_numeric(d["Rank"], errors="coerce") <= 3).astype(float)
    elif "is_top3" in d.columns:
        y = pd.to_numeric(d["is_top3"], errors="coerce")
    else:
        d["JockeyScore"] = np.nan
        return d

    d["_y"] = y
    date_col = "日付"
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    else:
        d[date_col] = pd.Timestamp("2000-01-01")
    d[date_col] = d[date_col].fillna(pd.Timestamp("2000-01-01"))

    valid_y = d["_y"].dropna()
    global_rate = float(valid_y.mean()) if len(valid_y) else 0.0

    d = d.sort_values([date_col, "JockeyId"], kind="mergesort")
    scores = pd.Series(index=d.index, dtype=float)
    for jockey_id, g in d.groupby("JockeyId", sort=False):
        if not jockey_id:
            scores.loc[g.index] = np.nan
            continue
        s = 0.0
        c = 0
        for _, g_date in g.groupby(date_col, sort=False):
            denom = c + prior
            score = (s + global_rate * prior) / denom if denom > 0 else global_rate
            scores.loc[g_date.index] = score
            y_vals = g_date["_y"].dropna()
            if len(y_vals):
                s += float(y_vals.sum())
                c += int(y_vals.count())
    d["JockeyScore"] = scores
    d = d.drop(columns=["_y"])
    return d.sort_index()


def build_latest_jockey_score_map(hist_df):
    if "JockeyId" not in hist_df.columns or "JockeyScore" not in hist_df.columns:
        return {}
    d = hist_df.copy()
    d["JockeyId"] = d["JockeyId"].apply(normalize_jockey_id)
    d = d[d["JockeyId"] != ""]
    if d.empty:
        return {}
    date_col = "日付"
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.sort_values([date_col, "JockeyId"], kind="mergesort")
    d = d.dropna(subset=["JockeyScore"])
    if d.empty:
        return {}
    return d.groupby("JockeyId")["JockeyScore"].last().to_dict()


def get_time_decay_tau_days(half_life_days):
    try:
        half_life_days = float(half_life_days)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        return None
    return half_life_days / math.log(2.0)


def compute_time_decay_mean(values, dates, tau_days, include_current=False):
    out = np.full(len(values), np.nan, dtype=float)
    s = 0.0
    w = 0.0
    last_date = None
    use_decay = tau_days is not None and np.isfinite(tau_days) and tau_days > 0

    for i, (x, t) in enumerate(zip(values, dates)):
        if t is not None and not pd.isna(t):
            if last_date is not None and not pd.isna(last_date) and use_decay:
                delta_days = (t - last_date) / np.timedelta64(1, "D")
                if delta_days < 0:
                    delta_days = 0.0
                decay = math.exp(-float(delta_days) / tau_days)
                s *= decay
                w *= decay
            last_date = t

        if include_current:
            s_tmp = s
            w_tmp = w
            if np.isfinite(x):
                s_tmp += float(x)
                w_tmp += 1.0
            out[i] = s_tmp / w_tmp if w_tmp > 0 else np.nan
        else:
            out[i] = s / w if w > 0 else np.nan

        if np.isfinite(x):
            s += float(x)
            w += 1.0

    return out


def add_time_decay_feature(df, value_col, date_col, group_col, out_col, tau_days, include_current=False):
    if value_col not in df.columns or date_col not in df.columns:
        df[out_col] = np.nan
        return df

    def _calc(g):
        values = g[value_col].to_numpy()
        dates = g[date_col].to_numpy()
        return pd.Series(
            compute_time_decay_mean(values, dates, tau_days, include_current=include_current),
            index=g.index,
        )

    df[out_col] = df.groupby(group_col, group_keys=False).apply(_calc)
    return df

TRACK_COND_MAP = {
    "良": 0.0,
    "稍重": 1.0,
    "重": 2.0,
    "不良": 3.0,
}

TRACK_COND_ALIASES = {
    "稍": "稍重",
    "不": "不良",
}

def parse_track_condition(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    if s in TRACK_COND_ALIASES:
        s = TRACK_COND_ALIASES[s]
    if s in TRACK_COND_MAP:
        return TRACK_COND_MAP[s]
    s_lower = s.lower()
    if s_lower in ("good", "firm"):
        return TRACK_COND_MAP["良"]
    if s_lower in ("yielding", "soft", "sloppy", "muddy"):
        return TRACK_COND_MAP["稍重"]
    if s_lower in ("heavy",):
        return TRACK_COND_MAP["重"]
    if s_lower in ("bad", "very_soft"):
        return TRACK_COND_MAP["不良"]
    return np.nan

def compute_track_weight(series, target):
    if series is None or not np.isfinite(target):
        return None
    vals = pd.to_numeric(series, errors="coerce")
    diff = (vals - float(target)).abs()
    weight = 1.0 / (1.0 + diff)
    return weight.fillna(0.7)


def compute_sample_weight(df, track_cond):
    sample_weight = None
    if "sm_mean5" in df.columns:
        weights = df["sm_mean5"].astype(float).fillna(0.0)
        weights = 0.3 + 0.7 * weights
        if float(weights.sum()) > 0:
            sample_weight = weights

    if "TrackCond" in df.columns:
        track_weight = compute_track_weight(df["TrackCond"], track_cond)
        if track_weight is not None:
            track_weight = 0.5 + 0.5 * track_weight
            sample_weight = track_weight if sample_weight is None else sample_weight * track_weight

    return sample_weight


def enrich(df):
    df = df.copy()
    sd = df["距離"].apply(parse_surface_distance)
    df["Surface"] = sd.apply(lambda t: t[0])
    df["Distance"] = sd.apply(lambda t: t[1])
    df["TimeIndex"] = df["ﾀｲﾑ指数"].apply(parse_float)
    df["Uphill"] = df["上り"].apply(parse_float)
    df["IsDirt"] = (df["Surface"] == "ダ").astype(int)
    df["IsTurf"] = (df["Surface"] == "芝").astype(int)

    if "馬場" in df.columns:
        df["TrackCond"] = df["馬場"].apply(parse_track_condition)
    else:
        df["TrackCond"] = np.nan

    # ---- 名次修正 ----
    if "着順" in df.columns:
        df["Rank"] = df["着順"].apply(parse_int)
    else:
        df["Rank"] = np.nan

    if "頭数" in df.columns:
        df["FieldSize"] = df["頭数"].apply(parse_int)
    else:
        df["FieldSize"] = np.nan

    df = add_running_style_features(df)
    df = add_race_normalized_features(df)

    df["PlaceScore"] = np.nan
    valid = (df["Rank"] > 0) & (df["FieldSize"] > 0)
    df.loc[valid, "PlaceScore"] = (
        (df.loc[valid, "FieldSize"] + 1 - df.loc[valid, "Rank"]) /
        df.loc[valid, "FieldSize"]
    )
    df["PlaceScore"] = df["PlaceScore"].clip(0, 1)

    # Avoid leakage: do not adjust by PlaceScore.
    df["TimeIndexEff"] = df["TimeIndex"]
    df = add_sex_age_features(df)
    return df

kachiuma_e = enrich(kachiuma)
shusso_e = enrich(shusso)
kachiuma_e = add_jockey_cum_score(kachiuma_e)
shusso_e = add_jockey_cum_score(shusso_e)
kachiuma_e["JockeyScore_current"] = kachiuma_e["JockeyScore"]
jockey_score_map = build_latest_jockey_score_map(kachiuma_e)
if "JockeyId_current" in shusso_e.columns:
    shusso_e["JockeyId_current"] = shusso_e["JockeyId_current"].apply(normalize_jockey_id)
    shusso_e["JockeyScore_current"] = shusso_e["JockeyId_current"].map(jockey_score_map)
else:
    shusso_e["JockeyScore_current"] = np.nan

# ====================================================
# 3. 比赛趋势：使用 TimeIndexEff
# ====================================================
def add_time_trend(df):
    df = df.copy()
    if "日付" in df.columns:
        df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    else:
        df["日付"] = pd.Timestamp("2000-01-01")

    df = df.sort_values(["HorseName", "日付"])
    trend_window = int(PARAMS["time_trend_window"])
    df["TimeTrend"] = (
        df.groupby("HorseName")["TimeIndexEff"]
          .apply(lambda x: x - x.shift(1).rolling(trend_window, min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )
    return df

kachiuma_e = add_time_trend(kachiuma_e)
shusso_e = add_time_trend(shusso_e)

# ====================================================
# 4. 比赛条件（支持手动输入）
# ====================================================
def get_default_race_surface():
    return "芝" if SCOPE_KEY == "central_turf" else "ダ"


def _env_flag(name):
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _env_surface(default_surface):
    raw = os.environ.get("PREDICTOR_TARGET_SURFACE", "").strip().lower()
    if raw in ("t", "turf", "grass", "shiba", "芝"):
        return "芝"
    if raw in ("d", "dirt", "sand", "ダ"):
        return "ダ"
    return default_surface


def _env_distance(default_distance):
    raw = os.environ.get("PREDICTOR_TARGET_DISTANCE", "").strip()
    if not raw:
        return default_distance
    try:
        return int(raw)
    except Exception:
        return default_distance


def _env_track(default_track):
    raw = os.environ.get("PREDICTOR_TARGET_CONDITION", "").strip()
    if not raw:
        return parse_track_condition(default_track)
    track_code = parse_track_condition(raw)
    if not np.isfinite(track_code):
        print(f"[WARN] Unknown env track condition: {raw}. Fallback to {default_track}.")
        return parse_track_condition(default_track)
    return track_code


def prompt_race_condition(default_surface=None, default_distance=1600, default_track="良"):
    if default_surface is None:
        default_surface = get_default_race_surface()
    if _env_flag("PREDICTOR_NO_PROMPT"):
        return (
            _env_surface(default_surface),
            _env_distance(default_distance),
            _env_track(default_track),
        )
    default_label = "turf" if default_surface == "芝" else "dirt"
    try:
        surf_raw = input(f"Race surface (turf/dirt) [default {default_label}]: ").strip().lower()
    except EOFError:
        surf_raw = ""
    if not surf_raw:
        surf_in = default_surface
    elif surf_raw in ("t", "turf", "grass", "shiba", "芝"):
        surf_in = "芝"
    elif surf_raw in ("d", "dirt", "sand", "ダ"):
        surf_in = "ダ"
    else:
        surf_in = default_surface
    try:
        dist_in = input(f"Distance meters [default {default_distance}]: ").strip()
    except EOFError:
        dist_in = ""
    try:
        dist_val = int(dist_in)
    except Exception:
        dist_val = default_distance
    try:
        track_raw = input(f"Track condition (良/稍重/重/不良) [default {default_track}]: ").strip()
    except EOFError:
        track_raw = ""
    track_in = track_raw or default_track
    track_code = parse_track_condition(track_in)
    if not np.isfinite(track_code):
        print(f"[WARN] Unknown track condition: {track_in}. Fallback to {default_track}.")
        track_code = parse_track_condition(default_track)
    return surf_in, dist_val, track_code

RACE_SURFACE, RACE_DISTANCE, RACE_TRACK_COND = prompt_race_condition()

shusso_e["TrackCond"] = RACE_TRACK_COND

# ====================================================
# 5. 学习距离衰减尺度 tau
# ====================================================
def learn_tau_for_distance(train_df, race_distance, window, tau_min, tau_max):
    """
    根据当前比赛距离，学习此距离区间最佳衰减尺度 tau
    window = 距离窗口（例如 ±300m）
    """

    sub = train_df.copy()

    # 只使用与本次比赛距离接近的历史比赛
    sub = sub[(sub["Distance"] - race_distance).abs() <= window]

    # 如果数据太少，回落到全局
    if len(sub) < 200:
        print("[WARN] Data too small, fallback to global learning.")
        sub = train_df.copy()

    # 学习逻辑回归衰减
    sub = sub.dropna(subset=["DistDiff", "y"]).copy()
    try:
        clf = LogisticRegression(max_iter=2000)
        X = sub[["DistDiff"]].values
        y = sub["y"].values
        clf.fit(X, y)
        b = float(clf.coef_[0][0])
    except Exception:
        b = -1/800.0

    tau = 1.0 / (-b) if b < 0 else 800.0
    tau = float(np.clip(tau, tau_min, tau_max))
    return tau

def _temp_base(df):
    d = df.copy()
    d["DistDiff"] = (d["Distance"] - RACE_DISTANCE).abs()
    d["MatchSurface"] = (d["Surface"] == RACE_SURFACE).astype(int)
    return d

# 历史数据打标签（Top3 为正，其余为负），避免把当前出走马当作训练负例
train = _temp_base(kachiuma_e)
if "Rank" in kachiuma_e.columns:
    train["y"] = np.where(
        kachiuma_e["Rank"].notna(),
        (kachiuma_e["Rank"] <= 3).astype(float),
        np.nan
    )
elif "is_top3" in kachiuma_e.columns:
    train["y"] = kachiuma_e["is_top3"].astype(float)
else:
    train["y"] = np.nan
train = train.dropna(subset=["y", "DistDiff"])
pos_count = int((train["y"] == 1).sum())
neg_count = int((train["y"] == 0).sum())
if pos_count == 0 or neg_count == 0:
    print("[WARN] Only one label class; tau learning may be unstable.")

print(f"[INFO] Train samples: {len(train)} (pos={pos_count}, neg={neg_count})")
tau = learn_tau_for_distance(
    train,
    RACE_DISTANCE,
    window=int(PARAMS["tau_window"]),
    tau_min=float(PARAMS["tau_min"]),
    tau_max=float(PARAMS["tau_max"])
)
print(f"[INFO] learned tau for {RACE_DISTANCE}m = {tau:.1f}")


# ====================================================
# 6. 添加条件平滑特征
# ====================================================
def add_condition_features(df, tau, p, surf_floor, dist_floor):
    df = df.copy()
    df["DistDiff"] = (df["Distance"] - RACE_DISTANCE).abs()
    df["MatchSurface"] = (df["Surface"] == RACE_SURFACE).astype(int)
    df["NegUphill"] = -df["Uphill"]

    base_surface = surf_floor + (1.0 - surf_floor) * df["MatchSurface"]
    dist_weight  = dist_floor + (1.0 - dist_floor) * np.exp(- (df["DistDiff"] / float(tau)) ** p)
    df["SmoothMatch"] = base_surface * dist_weight
    return df

kachiuma_e = add_condition_features(
    kachiuma_e,
    tau=tau,
    p=float(PARAMS["smooth_p"]),
    surf_floor=float(PARAMS["surf_floor"]),
    dist_floor=float(PARAMS["dist_floor"])
)
shusso_e = add_condition_features(
    shusso_e,
    tau=tau,
    p=float(PARAMS["smooth_p"]),
    surf_floor=float(PARAMS["surf_floor"]),
    dist_floor=float(PARAMS["dist_floor"])
)


def build_cup_key_series(df):
    surface = df["Surface"].astype(str) if "Surface" in df.columns else pd.Series("", index=df.index)
    distance = df["Distance"].astype(str) if "Distance" in df.columns else pd.Series("", index=df.index)
    track = df["TrackCond"].astype(str) if "TrackCond" in df.columns else pd.Series("", index=df.index)
    return surface.fillna("") + "|" + distance.fillna("") + "|" + track.fillna("")


def select_target_race_rows(df):
    if "race_id" not in df.columns or "HorseName" not in df.columns:
        return df.head(0)
    d = df.copy()
    d["_row_order"] = np.arange(len(d))
    d = d.sort_values("_row_order")
    target = d.drop_duplicates(subset=["race_id", "HorseName"], keep="first")
    return target.drop(columns=["_row_order"])


def build_cup_profiles(df):
    target = select_target_race_rows(df)
    if target.empty:
        return pd.DataFrame(), {}
    target = target.copy()
    target["CupKey"] = build_cup_key_series(target)

    if "Rank" in target.columns:
        is_top3 = target["Rank"] <= 3
    elif "is_top3" in target.columns:
        is_top3 = target["is_top3"].astype(int) == 1
    else:
        is_top3 = pd.Series(False, index=target.index)

    top3 = target[is_top3]
    if top3.empty:
        return pd.DataFrame(), {}

    top3 = top3.copy()
    for out_col, z_col, raw_col in (
        ("CupRunFirst", "RunFirstPct_z", "RunFirstPct"),
        ("CupRunLast", "RunLastPct_z", "RunLastPct"),
        ("CupRunGain", "RunGainPct_z", "RunGainPct"),
        ("CupUphill", "Uphill_z", "Uphill"),
        ("CupPaceDiff", "PaceDiff_z", "PaceDiff"),
    ):
        if z_col in top3.columns:
            top3[out_col] = top3[z_col]
        elif raw_col in top3.columns:
            top3[out_col] = top3[raw_col]
        else:
            top3[out_col] = np.nan

    profiles = (
        top3.groupby("CupKey")
        .agg(
            cup_run_first_pct=("CupRunFirst", "mean"),
            cup_run_last_pct=("CupRunLast", "mean"),
            cup_run_gain_pct=("CupRunGain", "mean"),
            cup_uphill_mean=("CupUphill", "mean"),
            cup_pace_diff=("CupPaceDiff", "mean"),
            cup_samples=("HorseName", "size"),
        )
        .reset_index()
    )
    defaults = {
        "cup_run_first_pct": float(top3["CupRunFirst"].mean()) if "CupRunFirst" in top3.columns else float("nan"),
        "cup_run_last_pct": float(top3["CupRunLast"].mean()) if "CupRunLast" in top3.columns else float("nan"),
        "cup_run_gain_pct": float(top3["CupRunGain"].mean()) if "CupRunGain" in top3.columns else float("nan"),
        "cup_uphill_mean": float(top3["CupUphill"].mean()) if "CupUphill" in top3.columns else float("nan"),
        "cup_pace_diff": float(top3["CupPaceDiff"].mean()) if "CupPaceDiff" in top3.columns else float("nan"),
        "cup_samples": float(top3["HorseName"].count()) if "HorseName" in top3.columns else float("nan"),
    }
    return profiles, defaults


def attach_cup_profile_features(df, cup_profiles, defaults, cup_key=None):
    d = df.copy()
    if cup_key is None:
        d["CupKey"] = build_cup_key_series(d)
    else:
        d["CupKey"] = cup_key

    if cup_profiles is not None and not cup_profiles.empty:
        d = d.merge(cup_profiles, on="CupKey", how="left")
    else:
        for col in (
            "cup_run_first_pct",
            "cup_run_last_pct",
            "cup_run_gain_pct",
            "cup_uphill_mean",
            "cup_pace_diff",
            "cup_samples",
        ):
            d[col] = np.nan

    # cup_samples の NaN を記録してから fillna する
    # （gap NaN 保護で使うため、先に保存しておく）
    _no_cup_data = d["cup_samples"].isna() | (d["cup_samples"] <= 0) if "cup_samples" in d.columns else pd.Series(False, index=d.index)

    for col, fallback in (defaults or {}).items():
        if col in d.columns:
            d[col] = d[col].fillna(fallback)

    def pick_feature(z_col, raw_col):
        if z_col in d.columns:
            return d[z_col]
        if raw_col in d.columns:
            return d[raw_col]
        return pd.Series(np.nan, index=d.index)

    run_first = pick_feature("run_first_z_mean5", "run_first_mean5")
    run_last = pick_feature("run_last_z_mean5", "run_last_mean5")
    run_gain = pick_feature("run_gain_z_mean5", "run_gain_mean5")
    up_mean = pick_feature("up_z_mean5", "up_mean5")
    pace_diff = pick_feature("pace_diff_z_mean5", "pace_diff_mean5")

    d["cup_run_first_gap"] = (run_first - d["cup_run_first_pct"]).abs()
    d["cup_run_last_gap"] = (run_last - d["cup_run_last_pct"]).abs()
    d["cup_run_gain_gap"] = (run_gain - d["cup_run_gain_pct"]).abs()
    d["cup_uphill_gap"] = (up_mean - d["cup_uphill_mean"]).abs()
    d["cup_pace_gap"] = (pace_diff - d["cup_pace_diff"]).abs()

    # cup_samples が元々 NaN or 0 → gap 特徴は信頼できないので NaN にする
    no_cup = _no_cup_data
    for col in ("cup_run_first_gap", "cup_run_last_gap", "cup_run_gain_gap",
                "cup_uphill_gap", "cup_pace_gap"):
        d.loc[no_cup, col] = np.nan

    return d


cup_profiles, cup_defaults = build_cup_profiles(kachiuma_e)


def print_reliability_bins(df, prob_col="_prob", y_col="y", bins=8):
    if prob_col not in df.columns or y_col not in df.columns:
        return
    tmp = df[[prob_col, y_col]].dropna().copy()
    if tmp.empty:
        return
    edges = np.linspace(0.0, 1.0, bins + 1)
    tmp["bin"] = pd.cut(tmp[prob_col], bins=edges, include_lowest=True)
    grouped = (
        tmp.groupby("bin")
        .agg(
            count=(y_col, "size"),
            prob_mean=(prob_col, "mean"),
            hit_rate=(y_col, "mean"),
        )
        .reset_index()
    )
    print("[INFO] Reliability bins:")
    print(grouped.to_string(index=False))


def build_cup_context(train_df: pd.DataFrame, max_dist: float = None) -> dict:
    d = train_df.copy()
    if "MatchSurface" in d.columns:
        d = d[d["MatchSurface"] == 1]
    if max_dist is not None and "DistDiff" in d.columns:
        d = d[d["DistDiff"] <= max_dist]

    d = d.dropna(subset=["TimeIndexEff"])

    if "Rank" in d.columns:
        is_top3 = d["Rank"] <= 3
    elif "is_top3" in d.columns:
        is_top3 = d["is_top3"].astype(int) == 1
    else:
        is_top3 = pd.Series(False, index=d.index)

    top3 = d[is_top3]
    rest = d[~is_top3]

    ctx = {
        "top3_ti_p50": float(top3["TimeIndexEff"].quantile(0.50)) if len(top3) else float("nan"),
        "top3_ti_p25": float(top3["TimeIndexEff"].quantile(0.25)) if len(top3) else float("nan"),
        "ti_sep": float(top3["TimeIndexEff"].mean() - rest["TimeIndexEff"].mean()) if len(rest) else float("nan"),
    }
    if "FieldSize" in d.columns:
        ctx["fieldsize_med"] = float(d["FieldSize"].median())
    return ctx


def _surface_exp_ratio(h):
    """過去出走のうち同じ surface の割合 (prediction 用)。"""
    if "Surface" not in h.columns or len(h) <= 1:
        return float("nan")
    past = h.iloc[:-1]  # exclude the target race row
    if len(past) == 0:
        return float("nan")
    return float((past["Surface"] == RACE_SURFACE).mean())


def _surface_exp_count(h):
    """過去出走で同じ surface の出走回数 (prediction 用)。"""
    if "Surface" not in h.columns or len(h) <= 1:
        return float("nan")
    past = h.iloc[:-1]
    if len(past) == 0:
        return float("nan")
    return float((past["Surface"] == RACE_SURFACE).sum())


def build_horse_profiles(shusso_df: pd.DataFrame, recent: int = 5) -> pd.DataFrame:
    d = shusso_df.copy()
    d = add_sex_age_features(d)
    date_col = "\u65e5\u4ed8"
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    else:
        d[date_col] = pd.Timestamp("2000-01-01")

    d = d.sort_values(["HorseName", date_col])
    tau_days = get_time_decay_tau_days(PARAMS.get("time_decay_half_life_days"))
    if "PlaceScore" not in d.columns:
        d["PlaceScore"] = np.nan
    d = add_time_decay_feature(
        d,
        value_col="TimeIndexEff",
        date_col=date_col,
        group_col="HorseName",
        out_col="ti_decay",
        tau_days=tau_days,
        include_current=True,
    )
    d = add_time_decay_feature(
        d,
        value_col="PlaceScore",
        date_col=date_col,
        group_col="HorseName",
        out_col="ps_decay",
        tau_days=tau_days,
        include_current=True,
    )

    rows = []
    for horse, h in d.groupby("HorseName"):
        # ??????????????????? TimeIndexEff ??????
        h = h.copy()

        # ?? N ??????? recent ???
        h_recent = h.tail(recent)

        # TimeIndex ??
        ti_series = h_recent["TimeIndexEff"] if "TimeIndexEff" in h_recent.columns else pd.Series(dtype=float)
        trend_series = h.tail(3)["TimeTrend"] if "TimeTrend" in h.columns and len(h) else pd.Series(dtype=float)
        jscore_series = h_recent["JockeyScore"] if "JockeyScore" in h_recent.columns else pd.Series(dtype=float)
        if "JockeyScore_current" in h.columns and h["JockeyScore_current"].notna().any():
            jscore_current = float(h["JockeyScore_current"].dropna().iloc[-1])
        else:
            jscore_current = float("nan")

        # PlaceScore ??
        if "PlaceScore" in h.columns:
            place_all = h["PlaceScore"].dropna()
            place_recent = h_recent["PlaceScore"].dropna()
        else:
            place_all = pd.Series(dtype=float)
            place_recent = pd.Series(dtype=float)

        # Bayesian smoothing prior for PlaceScore (neutral = 0.5, strength = 5 races)
        _ps_prior = 0.5
        _ps_bk = 5

        # 1) AvgPlaceScore: ???????????????????????????????
        _n_all = len(place_all)
        ps_avg_all = float((place_all.sum() + _ps_prior * _ps_bk) / (_n_all + _ps_bk)) if _n_all > 0 else _ps_prior

        # 2) RecentPlaceScore: ?? N ????
        _n_rec = len(place_recent)
        ps_recent_mean = float((place_recent.sum() + _ps_prior * _ps_bk) / (_n_rec + _ps_bk)) if _n_rec > 0 else _ps_prior

        # 3) MaxPlaceScore??? N ???????????????
        ps_max_recent = float(place_recent.max()) if len(place_recent) else float("nan")

        # 4) ExpPlaceScore??? SmoothMatch ?????
        #    ???????????????????????????
        if "SmoothMatch" in h.columns and "PlaceScore" in h.columns:
            w = h["SmoothMatch"].astype(float).clip(lower=0.0)
            ps = h["PlaceScore"].astype(float)
            num = float((ps * w).sum())
            den = float(w.sum())
            # Bayesian smoothing: blend weighted mean toward prior=0.5 with strength _ps_bk
            exp_place = float((num + _ps_prior * _ps_bk) / (den + _ps_bk)) if den > 0 else _ps_prior
        else:
            exp_place = float("nan")

        age_val = float(h["Age"].dropna().iloc[-1]) if "Age" in h.columns and h["Age"].notna().any() else float("nan")
        sex_male = float(h["SexMale"].dropna().iloc[-1]) if "SexMale" in h.columns and h["SexMale"].notna().any() else float("nan")
        sex_female = float(h["SexFemale"].dropna().iloc[-1]) if "SexFemale" in h.columns and h["SexFemale"].notna().any() else float("nan")
        sex_gelding = float(h["SexGelding"].dropna().iloc[-1]) if "SexGelding" in h.columns and h["SexGelding"].notna().any() else float("nan")
        ti_decay = float(h["ti_decay"].dropna().iloc[-1]) if "ti_decay" in h.columns and h["ti_decay"].notna().any() else float("nan")
        ps_decay = float(h["ps_decay"].dropna().iloc[-1]) if "ps_decay" in h.columns and h["ps_decay"].notna().any() else float("nan")

        out = {
            "HorseName": horse,
            # TimeIndex ??
            "ti_last": float(ti_series.iloc[-1]) if len(ti_series) else float("nan"),
            "ti_mean3": float(h.tail(3)["TimeIndexEff"].mean()) if "TimeIndexEff" in h.columns and len(h) else float("nan"),
            "ti_mean5": float(ti_series.mean()) if len(ti_series) else float("nan"),
            "ti_max5": float(ti_series.max()) if len(ti_series) else float("nan"),
            "ti_std5": float(ti_series.std(ddof=0)) if len(ti_series) else float("nan"),
            "ti_decay": ti_decay,
            "sm_mean5": float(h_recent["SmoothMatch"].mean()) if "SmoothMatch" in h_recent.columns and len(h_recent) else float("nan"),
            "dd_mean5": float(h_recent["DistDiff"].mean()) if "DistDiff" in h_recent.columns and len(h_recent) else float("nan"),
            "up_min5": float(h_recent["Uphill"].min()) if "Uphill" in h_recent.columns and len(h_recent) else float("nan"),
            "trend_mean3": float(trend_series.mean()) if len(trend_series) else float("nan"),
            "history_count": int(max(len(h) - 1, 0)),
            "surface_exp_ratio": _surface_exp_ratio(h),
            "surface_exp_count": _surface_exp_count(h),
            "Age": age_val,
            "SexMale": sex_male,
            "SexFemale": sex_female,
            "SexGelding": sex_gelding,
            "jscore_last": float(jscore_series.iloc[-1]) if len(jscore_series) else float("nan"),
            "jscore_mean5": float(jscore_series.mean()) if len(jscore_series) else float("nan"),
            "jscore_max5": float(jscore_series.max()) if len(jscore_series) else float("nan"),
            "jscore_current": jscore_current,
            # PlaceScore ??
            "ps_avg_all": ps_avg_all,
            "ps_recent_mean": ps_recent_mean,
            "ps_max_recent": ps_max_recent,
            "exp_place_recent": exp_place,
            "ps_decay": ps_decay,
            "run_first_mean5": float(h_recent["RunFirstPct"].mean()) if "RunFirstPct" in h_recent.columns and len(h_recent) else float("nan"),
            "run_last_mean5": float(h_recent["RunLastPct"].mean()) if "RunLastPct" in h_recent.columns and len(h_recent) else float("nan"),
            "run_gain_mean5": float(h_recent["RunGainPct"].mean()) if "RunGainPct" in h_recent.columns and len(h_recent) else float("nan"),
            "up_mean5": float(h_recent["Uphill"].mean()) if "Uphill" in h_recent.columns and len(h_recent) else float("nan"),
            "pace_diff_mean5": float(h_recent["PaceDiff"].mean()) if "PaceDiff" in h_recent.columns and len(h_recent) else float("nan"),
            "run_first_z_mean5": float(h_recent["RunFirstPct_z"].mean()) if "RunFirstPct_z" in h_recent.columns and len(h_recent) else float("nan"),
            "run_last_z_mean5": float(h_recent["RunLastPct_z"].mean()) if "RunLastPct_z" in h_recent.columns and len(h_recent) else float("nan"),
            "run_gain_z_mean5": float(h_recent["RunGainPct_z"].mean()) if "RunGainPct_z" in h_recent.columns and len(h_recent) else float("nan"),
            "up_z_mean5": float(h_recent["Uphill_z"].mean()) if "Uphill_z" in h_recent.columns and len(h_recent) else float("nan"),
            "pace_diff_z_mean5": float(h_recent["PaceDiff_z"].mean()) if "PaceDiff_z" in h_recent.columns and len(h_recent) else float("nan"),
        }
        rows.append(out)

    return pd.DataFrame(rows)



def build_profile_training_samples(hist_df: pd.DataFrame, recent: int = 5) -> pd.DataFrame:
    d = hist_df.copy()
    date_col = "\u65e5\u4ed8"
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    else:
        d[date_col] = pd.Timestamp("2000-01-01")

    d = d.sort_values(["HorseName", date_col])
    d = add_sex_age_features(d)
    base_cols = [
        "TimeIndexEff",
        "Uphill",
        "SmoothMatch",
        "DistDiff",
        "TimeTrend",
        "RunFirstPct",
        "RunLastPct",
        "RunGainPct",
        "PaceDiff",
        "RunFirstPct_z",
        "RunLastPct_z",
        "RunGainPct_z",
        "Uphill_z",
        "PaceDiff_z",
    ]
    for col in base_cols:
        if col not in d.columns:
            d[col] = np.nan
    if "JockeyScore" not in d.columns:
        d["JockeyScore"] = np.nan
    if "JockeyScore_current" not in d.columns:
        d["JockeyScore_current"] = np.nan

    if "PlaceScore" not in d.columns:
        d["PlaceScore"] = np.nan
    tau_days = get_time_decay_tau_days(PARAMS.get("time_decay_half_life_days"))
    d = add_time_decay_feature(
        d,
        value_col="TimeIndexEff",
        date_col=date_col,
        group_col="HorseName",
        out_col="ti_decay",
        tau_days=tau_days,
        include_current=False,
    )
    d = add_time_decay_feature(
        d,
        value_col="PlaceScore",
        date_col=date_col,
        group_col="HorseName",
        out_col="ps_decay",
        tau_days=tau_days,
        include_current=False,
    )

    if "TrackCond" not in d.columns:
        d["TrackCond"] = np.nan
    if "race_id" not in d.columns:
        d["race_id"] = np.nan

    g = d.groupby("HorseName", group_keys=False)

    def lag_roll(col, window, agg):
        shifted = g[col].shift(1)
        rolled = shifted.groupby(d["HorseName"]).rolling(window, min_periods=1)
        if agg == "mean":
            out = rolled.mean()
        elif agg == "max":
            out = rolled.max()
        elif agg == "min":
            out = rolled.min()
        elif agg == "std":
            out = rolled.std(ddof=0)
        else:
            out = rolled.aggregate(agg)
        return out.reset_index(level=0, drop=True)

    d["ti_last"] = g["TimeIndexEff"].shift(1)
    d["ti_mean3"] = lag_roll("TimeIndexEff", 3, "mean")
    d["ti_mean5"] = lag_roll("TimeIndexEff", recent, "mean")
    d["ti_max5"] = lag_roll("TimeIndexEff", recent, "max")
    d["ti_std5"] = lag_roll("TimeIndexEff", recent, "std")
    d["sm_mean5"] = lag_roll("SmoothMatch", recent, "mean")
    d["dd_mean5"] = lag_roll("DistDiff", recent, "mean")
    d["up_min5"] = lag_roll("Uphill", recent, "min")
    d["run_first_mean5"] = lag_roll("RunFirstPct", recent, "mean")
    d["run_last_mean5"] = lag_roll("RunLastPct", recent, "mean")
    d["run_gain_mean5"] = lag_roll("RunGainPct", recent, "mean")
    d["up_mean5"] = lag_roll("Uphill", recent, "mean")
    d["pace_diff_mean5"] = lag_roll("PaceDiff", recent, "mean")
    d["run_first_z_mean5"] = lag_roll("RunFirstPct_z", recent, "mean")
    d["run_last_z_mean5"] = lag_roll("RunLastPct_z", recent, "mean")
    d["run_gain_z_mean5"] = lag_roll("RunGainPct_z", recent, "mean")
    d["up_z_mean5"] = lag_roll("Uphill_z", recent, "mean")
    d["pace_diff_z_mean5"] = lag_roll("PaceDiff_z", recent, "mean")
    d["trend_mean3"] = lag_roll("TimeTrend", 3, "mean")   # ← 新增

    # Surface experience: 同じ surface の過去出走数 / 全出走数
    if "Surface" in d.columns:
        d["_is_same_surface"] = (d["Surface"] == RACE_SURFACE).astype(float)
        d["_surface_cum"] = g["_is_same_surface"].shift(1).groupby(d["HorseName"]).cumsum()
        d["_total_cum"] = g.cumcount()  # 0-indexed count of prior rows
        d["surface_exp_ratio"] = d["_surface_cum"] / d["_total_cum"].clip(lower=1)
        d["surface_exp_count"] = d["_surface_cum"].fillna(0)
        d.drop(columns=["_is_same_surface", "_surface_cum", "_total_cum"], inplace=True)
    else:
        d["surface_exp_ratio"] = np.nan
        d["surface_exp_count"] = np.nan

    d["history_count"] = g.cumcount()
    d["jscore_last"] = d["JockeyScore"]
    d["jscore_mean5"] = (
        d.groupby("HorseName")["JockeyScore"]
         .rolling(recent, min_periods=1)
         .mean()
         .reset_index(level=0, drop=True)
    )
    d["jscore_max5"] = (
        d.groupby("HorseName")["JockeyScore"]
         .rolling(recent, min_periods=1)
         .max()
         .reset_index(level=0, drop=True)
    )
    d["jscore_current"] = d["JockeyScore_current"]

    # ---- PlaceScore ???????????? shift+rolling?????????----
    if "PlaceScore" not in d.columns:
        # ??? enrich() ????? PlaceScore???????
        d["PlaceScore"] = np.nan

    # ???? & ???????????????????????????
    recent_win = int(recent)
    long_win = max(recent_win * 2, 6)  # ?? recent=5 ??long_win=10

    ps_prior = 0.5
    ps_bk = 5.0

    # 1) AvgPlaceScore: ???????????????????????????
    d["ps_sum_all"] = lag_roll("PlaceScore", long_win, "sum")
    d["ps_count_all"] = lag_roll("PlaceScore", long_win, "count")
    d["ps_avg_all"] = np.where(
        d["ps_count_all"] > 0,
        (d["ps_sum_all"].fillna(0.0) + ps_prior * ps_bk) / (d["ps_count_all"].fillna(0.0) + ps_bk),
        ps_prior,
    )

    # 2) RecentPlaceScore: ?? N ??????????????
    d["ps_sum_recent"] = lag_roll("PlaceScore", recent_win, "sum")
    d["ps_count_recent"] = lag_roll("PlaceScore", recent_win, "count")
    d["ps_recent_mean"] = np.where(
        d["ps_count_recent"] > 0,
        (d["ps_sum_recent"].fillna(0.0) + ps_prior * ps_bk) / (d["ps_count_recent"].fillna(0.0) + ps_bk),
        ps_prior,
    )

    # 3) MaxPlaceScore??????????? N ????? PlaceScore
    #    ?????????????????????????????? max?
    d["ps_max_recent"] = lag_roll("PlaceScore", recent_win, "max")

    # 4) ExpPlaceScore: ??/???????? PlaceScore?
    #    ?? SmoothMatch ?????? PlaceScore ????????
    if "SmoothMatch" in d.columns:
        d["ps_weighted"] = d["PlaceScore"] * d["SmoothMatch"]
        d["ps_weighted_sum"] = lag_roll("ps_weighted", recent_win, "sum")
        d["smoothmatch_sum"] = lag_roll("SmoothMatch", recent_win, "sum")
        d["exp_place_recent"] = np.where(
            d["smoothmatch_sum"] > 0,
            (d["ps_weighted_sum"].fillna(0.0) + ps_prior * ps_bk) / (d["smoothmatch_sum"].fillna(0.0) + ps_bk),
            ps_prior,
        )
    else:
        d["exp_place_recent"] = np.nan


    if "Rank" in d.columns:
        d["y"] = np.where(
            d["Rank"].notna(),
            (d["Rank"] <= 3).astype(float),
            np.nan
        )
    elif "is_top3" in d.columns:
        d["y"] = d["is_top3"].astype(float)
    else:
        d["y"] = np.nan

    d = d[d["history_count"] >= 1].copy()
    return d


# ====================================================
def build_model_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def build_lgb_classifier():
    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=20,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        importance_type="gain",
        objective="binary",
        random_state=42,
        n_jobs=LGBM_N_JOBS,
    )


def collect_oof_predictions(train_df, features, group_col="race_id", n_splits=5):
    """CV で LR / LGB 両方の OOF 予測を収集し、評価指標も表示する。"""
    empty = {"lr_probs": np.array([]), "lgb_probs": np.array([]), "labels": np.array([])}
    if group_col not in train_df.columns:
        print(f"[INFO] OOF collection skipped: missing {group_col}.")
        return empty
    eval_df = train_df.dropna(subset=[group_col]).copy()
    if eval_df.empty or eval_df[group_col].nunique() < 2:
        print(f"[INFO] OOF collection skipped: not enough {group_col} groups.")
        return empty

    all_lr, all_lgb, all_y = [], [], []
    metrics = []
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    for split_id, (train_idx, test_idx) in enumerate(
        gss.split(eval_df, eval_df["y"], eval_df[group_col]),
        start=1,
    ):
        train_split = eval_df.iloc[train_idx]
        test_split = eval_df.iloc[test_idx]

        pos = int((train_split["y"] == 1).sum())
        neg = int((train_split["y"] == 0).sum())
        if pos == 0 or neg == 0:
            print(f"[WARN] OOF split {split_id}: only one class in train split.")
            continue

        X_train, y_train = train_split[features], train_split["y"]
        X_test, y_test = test_split[features], test_split["y"]
        if len(np.unique(y_test)) < 2:
            print(f"[INFO] OOF split {split_id}: only one class in test split.")
            continue

        _sw_fold = compute_sample_weight(train_split, RACE_TRACK_COND)
        sw = _sw_fold.values if _sw_fold is not None else None

        # LR
        fold_pipe = build_model_pipeline()
        if sw is not None:
            fold_pipe.fit(X_train, y_train, clf__sample_weight=sw)
        else:
            fold_pipe.fit(X_train, y_train)
        lr_probs = fold_pipe.predict_proba(X_test)[:, 1]

        # LGB (with early stopping against the held-out fold)
        fold_lgb = build_lgb_classifier()
        _lgb_fold_kw = {
            "eval_set": [(X_test, y_test)],
            "eval_metric": "logloss",
            "callbacks": [early_stopping(50, verbose=False)],
        }
        if sw is not None:
            _lgb_fold_kw["sample_weight"] = sw
        fold_lgb.fit(X_train, y_train, **_lgb_fold_kw)
        lgb_probs = fold_lgb.predict_proba(X_test)[:, 1]

        all_lr.append(lr_probs)
        all_lgb.append(lgb_probs)
        all_y.append(y_test.values)

        # 評価指標
        auc_lr = roc_auc_score(y_test, lr_probs)
        auc_lgb = roc_auc_score(y_test, lgb_probs)
        ll_lr = log_loss(y_test, lr_probs)
        ll_lgb = log_loss(y_test, lgb_probs)
        print(
            f"[INFO] OOF split {split_id}/{n_splits}: "
            f"LR auc={auc_lr:.4f} ll={ll_lr:.4f} | "
            f"LGB auc={auc_lgb:.4f} ll={ll_lgb:.4f} "
            f"(train={len(train_split)}, test={len(test_split)})"
        )

        test_eval = test_split.copy()
        test_eval["_prob"] = lr_probs
        hits = 0
        total_slots = 0
        race_count = 0
        hit1 = 0
        for _, grp in test_eval.groupby(group_col):
            race_count += 1
            grp_sorted = grp.sort_values("_prob", ascending=False)
            top3_grp = grp_sorted.head(3)
            hits += int(top3_grp["y"].sum())
            total_slots += min(3, len(top3_grp))
            hit1 += int(grp_sorted.head(1)["y"].sum())

        hit_at_3 = hits / total_slots if total_slots else float("nan")
        hit_at_1 = hit1 / race_count if race_count else float("nan")
        print(
            f"[INFO] OOF split {split_id}/{n_splits}: "
            f"hit@3={hit_at_3:.4f} hit@1={hit_at_1:.4f} races={race_count}"
        )
        print_reliability_bins(test_eval, prob_col="_prob", y_col="y", bins=8)
        metrics.append((auc_lr, ll_lr, hit_at_3, hit_at_1))

    if metrics:
        aucs, lls, hit3s, hit1s = zip(*metrics)
        print(
            "[INFO] OOF (avg): "
            f"auc={np.mean(aucs):.4f} logloss={np.mean(lls):.4f} "
            f"hit@3={np.mean(hit3s):.4f} hit@1={np.mean(hit1s):.4f}"
        )
    else:
        print("[WARN] OOF collection: no valid splits.")

    if all_lr:
        return {
            "lr_probs": np.concatenate(all_lr),
            "lgb_probs": np.concatenate(all_lgb),
            "labels": np.concatenate(all_y),
        }
    return empty


# 7. Train Logistic Regression Model
# ====================================================
NUMERIC_FEATURES = [
    # TimeIndex ??
    "ti_last", "ti_mean3", "ti_mean5", "ti_max5", "ti_std5", "ti_decay",
    # ?? & ????
      "sm_mean5", "dd_mean5", "up_min5",
      "run_first_z_mean5", "run_last_z_mean5", "run_gain_z_mean5",
      "up_z_mean5", "pace_diff_z_mean5",
      "trend_mean3",
    "history_count",
    "surface_exp_ratio", "surface_exp_count",
    "Age", "SexMale", "SexFemale", "SexGelding",
    "jscore_last", "jscore_mean5", "jscore_max5", "jscore_current",
    # PlaceScore ??????
    "ps_avg_all",        # ??????????????+??????
    "ps_recent_mean",    # ?? N ??? PlaceScore????
    "ps_max_recent",     # ?? N ????? PlaceScore??????????
    "exp_place_recent",  # ??/??????? PlaceScore?ExpPlaceScore?
      "ps_decay",
      "cup_run_first_gap", "cup_run_last_gap", "cup_run_gain_gap",
      "cup_uphill_gap", "cup_pace_gap", "cup_samples",
      # ??????
      "TargetDistance",
  ]
FEATURES = NUMERIC_FEATURES

cup = build_cup_context(kachiuma_e, max_dist=float(PARAMS["tau_window"]))
train_df = build_profile_training_samples(kachiuma_e, recent=int(PARAMS["recent_race_count"]))
train_df = attach_cup_profile_features(train_df, cup_profiles, cup_defaults)
train_df = train_df.dropna(subset=["y"])


if "race_id" in train_df.columns:
    race_id_coverage = train_df["race_id"].notna().mean()
    print(f"[INFO] race_id coverage (train_df): {race_id_coverage:.3f}")
    if race_id_coverage < 0.8:
        print("[WARN] race_id coverage below 0.8; grouped eval may be unreliable.")
else:
    print("[WARN] race_id missing in train_df; grouped eval will be skipped.")

for k, v in cup.items():
    train_df[k] = v

train_df["TargetDistance"] = RACE_DISTANCE

for col in FEATURES:
    if col not in train_df.columns:
        train_df[col] = np.nan

all_nan_cols = [col for col in NUMERIC_FEATURES if train_df[col].isna().all()]
if all_nan_cols:
    print(
        "[WARN] These features are all-NaN in training and will be filled with 0: "
        f"{all_nan_cols}"
    )
    for col in all_nan_cols:
        train_df[col] = 0.0

pos_count = int((train_df["y"] == 1).sum())
neg_count = int((train_df["y"] == 0).sum())
print(f"[INFO] Model training rows: {len(train_df)} (pos={pos_count}, neg={neg_count})")

model_mode = normalize_model_mode(PARAMS.get("model_mode", "hybrid"))
need_classifier = model_mode in {"classifier", "hybrid"}
need_ranker = model_mode in {"ranker", "hybrid"}
print(f"[INFO] model_mode={model_mode} (classifier={need_classifier}, ranker={need_ranker})")

base_prob = float(train_df["y"].mean()) if len(train_df) else 0.0
if not np.isfinite(base_prob):
    base_prob = 0.0
use_classifier_model = need_classifier and pos_count > 0 and neg_count > 0

platt_lr = None
platt_lgb = None
learned_alpha = clamp(safe_param_float(PARAMS, "blend_alpha", 0.3), 0.0, 1.0)

if need_classifier:
    oof = collect_oof_predictions(train_df, FEATURES, group_col="race_id")
    if oof["labels"].size > 0:
        platt_lr = fit_platt_scaler(oof["lr_probs"], oof["labels"])
        platt_lgb = fit_platt_scaler(oof["lgb_probs"], oof["labels"])
        cal_lr = apply_platt_scaler(platt_lr, oof["lr_probs"])
        cal_lgb = apply_platt_scaler(platt_lgb, oof["lgb_probs"])
        learned_alpha = find_optimal_blend_alpha(cal_lr, cal_lgb, oof["labels"])
else:
    print("[INFO] Grouped classifier eval skipped: model_mode does not use classifier.")


_sw = compute_sample_weight(train_df, RACE_TRACK_COND)
sample_weight = _sw.values if _sw is not None else None

pipe = None
lgb_model = None
ranker_model = None

if use_classifier_model:
    pipe = build_model_pipeline()
    X, y = train_df[FEATURES], train_df["y"]
    if sample_weight is not None:
        pipe.fit(X, y, clf__sample_weight=sample_weight)
    else:
        pipe.fit(X, y)

    print("[INFO] Training LightGBM model for ensemble...")
    lgb_model = build_lgb_classifier()

    # Early stopping: hold out 15% by race group for validation
    _lgb_es_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    _lgb_has_rid = "race_id" in train_df.columns and train_df["race_id"].notna().mean() > 0.5
    _lgb_group_count = 0
    if _lgb_has_rid:
        _lgb_group_count = int(train_df["race_id"].dropna().nunique())
    if _lgb_has_rid and _lgb_group_count >= 2:
        try:
            _lgb_tr_idx, _lgb_va_idx = next(
                _lgb_es_split.split(train_df, train_df["y"], train_df["race_id"])
            )
        except Exception as exc:
            print(f"[WARN] LGB early-stopping split failed: {exc}")
            _lgb_tr_idx, _lgb_va_idx = None, None

        if _lgb_tr_idx is not None and _lgb_va_idx is not None:
            X_lgb_tr, y_lgb_tr = X.iloc[_lgb_tr_idx], y.iloc[_lgb_tr_idx]
            X_lgb_va, y_lgb_va = X.iloc[_lgb_va_idx], y.iloc[_lgb_va_idx]
            sw_lgb = sample_weight[_lgb_tr_idx] if sample_weight is not None else None
        else:
            X_lgb_tr, y_lgb_tr = X, y
            X_lgb_va, y_lgb_va = None, None
            sw_lgb = sample_weight
    else:
        if _lgb_has_rid and _lgb_group_count < 2:
            print("[INFO] LGB early stopping skipped: not enough race_id groups.")
        X_lgb_tr, y_lgb_tr = X, y
        X_lgb_va, y_lgb_va = None, None
        sw_lgb = sample_weight

    _lgb_fit_kw = {}
    if sw_lgb is not None:
        _lgb_fit_kw["sample_weight"] = sw_lgb
    if X_lgb_va is not None:
        _lgb_fit_kw["eval_set"] = [(X_lgb_va, y_lgb_va)]
        _lgb_fit_kw["eval_metric"] = "logloss"
        _lgb_fit_kw["callbacks"] = [early_stopping(50, verbose=False)]

    lgb_model.fit(X_lgb_tr, y_lgb_tr, **_lgb_fit_kw)
    _best = getattr(lgb_model, "best_iteration_", lgb_model.n_estimators)
    print(f"[INFO] LGB final model: best_iteration={_best}")
elif need_classifier:
    print("[WARN] Only one label class; classifier fallback to constant probability.")

if need_ranker:
    if "race_id" not in train_df.columns:
        print("[WARN] Ranker skipped: missing race_id column.")
    else:
        rank_df = train_df.copy()
        rank_df["race_id"] = rank_df["race_id"].fillna("").astype(str).str.strip()
        rank_df = rank_df[rank_df["race_id"] != ""].copy()
        rank_source = rank_df["Rank"] if "Rank" in rank_df.columns else pd.Series(np.nan, index=rank_df.index)
        rank_df["y_rel"] = build_ranker_relevance(
            rank_source,
            mode=PARAMS.get("ranker_label_mode", "top5_gain"),
        )
        rank_df = rank_df.dropna(subset=["y_rel"]).copy()
        if rank_df.empty:
            print("[WARN] Ranker skipped: no valid relevance labels.")
        else:
            race_sizes = rank_df.groupby("race_id").size()
            valid_races = race_sizes[race_sizes >= 2].index
            rank_df = rank_df[rank_df["race_id"].isin(valid_races)].copy()
            race_count = int(rank_df["race_id"].nunique())
            if race_count < 2:
                print("[WARN] Ranker skipped: not enough race groups for train/valid split.")
            else:
                try:
                    gss_rank = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    train_idx, valid_idx = next(
                        gss_rank.split(rank_df, rank_df["y_rel"], rank_df["race_id"])
                    )
                except Exception as exc:
                    print(f"[WARN] Ranker split failed: {exc}")
                    train_idx, valid_idx = None, None

                if train_idx is not None and valid_idx is not None:
                    rank_train = rank_df.iloc[train_idx].copy().sort_values("race_id")
                    rank_valid = rank_df.iloc[valid_idx].copy().sort_values("race_id")

                    group_train = rank_train.groupby("race_id").size().tolist()
                    group_valid = rank_valid.groupby("race_id").size().tolist()
                    if not group_train or not group_valid:
                        print("[WARN] Ranker skipped: empty train/valid groups after split.")
                    else:
                        ranker_model = LGBMRanker(
                            objective="lambdarank",
                            metric="ndcg",
                            n_estimators=max(50, safe_param_int(PARAMS, "ranker_n_estimators", 300)),
                            learning_rate=clamp(
                                safe_param_float(PARAMS, "ranker_learning_rate", 0.05),
                                0.001,
                                0.5,
                            ),
                            num_leaves=max(8, safe_param_int(PARAMS, "ranker_num_leaves", 31)),
                            min_child_samples=max(
                                1,
                                safe_param_int(PARAMS, "ranker_min_child_samples", 20),
                            ),
                            subsample=clamp(
                                safe_param_float(PARAMS, "ranker_subsample", 0.8),
                                0.3,
                                1.0,
                            ),
                            colsample_bytree=clamp(
                                safe_param_float(PARAMS, "ranker_colsample_bytree", 0.8),
                                0.3,
                                1.0,
                            ),
                            random_state=42,
                            n_jobs=LGBM_N_JOBS,
                        )
                        ranker_model.fit(
                            rank_train[FEATURES],
                            rank_train["y_rel"],
                            group=group_train,
                            eval_set=[(rank_valid[FEATURES], rank_valid["y_rel"])],
                            eval_group=[group_valid],
                            eval_at=[1, 3, 5],
                            callbacks=[early_stopping(50, verbose=False)],
                        )
                        print(
                            "[INFO] Ranker trained: "
                            f"train_races={len(group_train)} valid_races={len(group_valid)} "
                            f"train_rows={len(rank_train)} valid_rows={len(rank_valid)}"
                        )
else:
    print("[INFO] Ranker disabled by model_mode.")

# ====================================================
# 8. Predict Runners
# ====================================================
profiles = build_horse_profiles(shusso_e, recent=int(PARAMS["recent_race_count"]))
target_cup_key = f"{RACE_SURFACE}|{RACE_DISTANCE}|{RACE_TRACK_COND}"
profiles = attach_cup_profile_features(profiles, cup_profiles, cup_defaults, cup_key=target_cup_key)
for k, v in cup.items():
    profiles[k] = v
profiles["TargetDistance"] = RACE_DISTANCE

for col in FEATURES:
    if col not in profiles.columns:
        profiles[col] = np.nan
if all_nan_cols:
    for col in all_nan_cols:
        profiles[col] = 0.0

if use_classifier_model and pipe is not None:
    proba_lr = pipe.predict_proba(profiles[FEATURES])[:, 1]
    profiles["Top3Prob_raw_lr"] = proba_lr

    # Platt calibration on LR
    cal_lr = apply_platt_scaler(platt_lr, proba_lr)
    profiles["Top3Prob_cal_lr"] = cal_lr

    if lgb_model is not None:
        raw_lgb = lgb_model.predict_proba(profiles[FEATURES])[:, 1]
        profiles["Top3Prob_raw_lgb"] = raw_lgb

        # Platt calibration on LGB
        cal_lgb = apply_platt_scaler(platt_lgb, raw_lgb)
        profiles["Top3Prob_cal_lgb"] = cal_lgb

        # Blend with learned alpha (from OOF log_loss optimization)
        blended = learned_alpha * cal_lgb + (1.0 - learned_alpha) * cal_lr
        profiles["Top3Prob_model"] = blended
    else:
        profiles["Top3Prob_model"] = cal_lr

    # Race-level normalization: sum of probs should ≈ top_k (3)
    field_size = len(profiles)
    profiles["Top3Prob_model"] = race_level_normalize(
        profiles["Top3Prob_model"].values, field_size, top_k=3
    )
else:
    fallback = np.full(len(profiles), base_prob, dtype=float)
    profiles["Top3Prob_raw_lr"] = fallback
    profiles["Top3Prob_model"] = fallback

profiles["model_mode"] = model_mode
race_keys_pred = build_race_key_series(profiles).fillna("").astype(str).replace("", "__single_race__")

if ranker_model is not None:
    raw_rank_score = ranker_model.predict(profiles[FEATURES])
    profiles["rank_score_raw"] = raw_rank_score
    rank_score_norm = normalize_rank_score_by_race(raw_rank_score, race_keys_pred)
    profiles["rank_score_norm"] = rank_score_norm
else:
    profiles["rank_score_raw"] = np.nan
    profiles["rank_score_norm"] = np.nan
    rank_score_norm = None

if model_mode == "ranker":
    if rank_score_norm is not None:
        profiles["rank_score"] = rank_score_norm
    else:
        print("[WARN] Ranker mode fallback: ranker model unavailable, using Top3Prob_model.")
        profiles["rank_score"] = pd.to_numeric(profiles["Top3Prob_model"], errors="coerce").fillna(base_prob)
    profiles["score_is_probability"] = 0
elif model_mode == "hybrid":
    if rank_score_norm is not None:
        ranker_alpha = clamp(safe_param_float(PARAMS, "ranker_blend_alpha", 0.6), 0.0, 1.0)
        base_score = pd.to_numeric(profiles["Top3Prob_model"], errors="coerce").fillna(base_prob).to_numpy(dtype=float)
        profiles["rank_score"] = ranker_alpha * rank_score_norm + (1.0 - ranker_alpha) * base_score
        profiles["score_is_probability"] = 0
    else:
        print("[WARN] Hybrid mode fallback: ranker model unavailable, using Top3Prob_model.")
        profiles["rank_score"] = pd.to_numeric(profiles["Top3Prob_model"], errors="coerce").fillna(base_prob)
        profiles["score_is_probability"] = 1
else:
    profiles["rank_score"] = pd.to_numeric(profiles["Top3Prob_model"], errors="coerce").fillna(base_prob)
    profiles["score_is_probability"] = 1

# backward-compatible aliases for existing scripts/UI
profiles["Top3Prob_lr"] = profiles["Top3Prob_raw_lr"]
if "Top3Prob_raw_lgb" in profiles.columns:
    profiles["Top3Prob_lgbm"] = profiles["Top3Prob_raw_lgb"]

pred_out = profiles.sort_values("rank_score", ascending=False)

conf_state = load_state_from_config(_config)
conf = compute_confidence_from_pred(pred_out, conf_state)
print(
    "[INFO] confidence: "
    f"{conf['confidence_score']:.4f} "
    f"(stability={conf['stability_score']:.4f}, "
    f"validity={conf['validity_score']:.4f}, "
    f"consistency={conf['consistency_score']:.4f})"
)
pred_out["confidence_score"] = round(conf["confidence_score"], 4)
pred_out["stability_score"] = round(conf["stability_score"], 4)
pred_out["validity_score"] = round(conf["validity_score"], 4)
pred_out["consistency_score"] = round(conf["consistency_score"], 4)
pred_out["rank_ema"] = round(conf_state["rank_ema"], 4)
pred_out["ev_ema"] = round(conf_state["ev_ema"], 4)
pred_out["risk_score"] = round(conf_state["risk_score"], 4)

top5 = pred_out.head(5)
print("\nTop5 predictions:")
print(top5.to_string(index=False))
pred_out.to_csv("predictions.csv", index=False, encoding="utf-8-sig")
print("Saved: predictions.csv")
elapsed = datetime.now() - PIPELINE_START
print(f"[INFO] pipeline elapsed: {elapsed}")
if sys.stdin is not None and sys.stdin.isatty():
    input("\nPress Enter to exit...")

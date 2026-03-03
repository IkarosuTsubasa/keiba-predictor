from typing import Iterable, List

import pandas as pd


_PRED_RACE_COLS = (
    "race_id",
    "run_id",
    "RaceID",
    "race",
)
_PRED_HORSE_COLS = (
    "horse_key",
    "horse_id",
    "horse_no",
    "horse_name",
    "HorseName",
    "name",
    "\u99ac\u756a",  # 馬番
    "\u99ac\u540d",  # 馬名
)
_PRED_SCORE_COLS = (
    "Top3Prob_model",
    "Top3Prob_est",
    "Top3Prob",
    "agg_score",
    "score",
)

_RESULT_RACE_COLS = (
    "race_id",
    "run_id",
    "RaceID",
    "race",
)
_RESULT_HORSE_COLS = (
    "horse_key",
    "horse_id",
    "horse_no",
    "horse_name",
    "HorseName",
    "name",
    "\u99ac\u756a",  # 馬番
    "\u99ac\u540d",  # 馬名
)
_RESULT_RANK_COLS = (
    "rank",
    "result_rank",
    "finish_position",
    "\u7740\u9806",  # 着順
)
_RESULT_TOP3_FLAG_COLS = (
    "is_top3",
    "top3",
    "in_top3",
)

_SINGLE_RACE_ID = "__single_race__"


def _pick_first(columns: Iterable[str], candidates: Iterable[str]) -> str:
    cols = set(columns)
    for name in candidates:
        if name in cols:
            return name
    return ""


def _normalize_horse_key(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
    )


def _parse_rank(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str)
    numeric = text.str.extract(r"(-?\d+)")[0]
    return pd.to_numeric(numeric, errors="coerce")


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) > 0
    text = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "top3",
        "in",
    }
    return text.isin(truthy)


def _prepare_pred(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame(columns=["_race_id", "_horse_key", "_score"])

    race_col = _pick_first(pred_df.columns, _PRED_RACE_COLS)
    horse_col = _pick_first(pred_df.columns, _PRED_HORSE_COLS)
    score_col = _pick_first(pred_df.columns, _PRED_SCORE_COLS)
    if not horse_col or not score_col:
        raise ValueError("pred_df 缺少 horse_key 或 Top3Prob_model（或其别名）字段")

    out = pd.DataFrame()
    if race_col:
        out["_race_id"] = pred_df[race_col].fillna("").astype(str).str.strip()
    else:
        out["_race_id"] = _SINGLE_RACE_ID
    out.loc[out["_race_id"] == "", "_race_id"] = _SINGLE_RACE_ID
    out["_horse_key"] = _normalize_horse_key(pred_df[horse_col])
    out["_score"] = pd.to_numeric(pred_df[score_col], errors="coerce")
    out = out[(out["_horse_key"] != "") & out["_score"].notna()].copy()
    out = out.sort_values(["_race_id", "_score"], ascending=[True, False])
    out = out.drop_duplicates(subset=["_race_id", "_horse_key"], keep="first")
    return out


def _prepare_result(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df is None or result_df.empty:
        return pd.DataFrame(columns=["_race_id", "_horse_key", "_is_top3"])

    race_col = _pick_first(result_df.columns, _RESULT_RACE_COLS)
    horse_col = _pick_first(result_df.columns, _RESULT_HORSE_COLS)
    rank_col = _pick_first(result_df.columns, _RESULT_RANK_COLS)
    top3_col = _pick_first(result_df.columns, _RESULT_TOP3_FLAG_COLS)
    if not horse_col:
        raise ValueError("result_df 缺少 horse_key（或其别名）字段")
    if not rank_col and not top3_col:
        raise ValueError("result_df 缺少 rank（或可判断Top3的字段）")

    out = pd.DataFrame()
    if race_col:
        out["_race_id"] = result_df[race_col].fillna("").astype(str).str.strip()
    else:
        out["_race_id"] = _SINGLE_RACE_ID
    out.loc[out["_race_id"] == "", "_race_id"] = _SINGLE_RACE_ID
    out["_horse_key"] = _normalize_horse_key(result_df[horse_col])

    top3_flags = pd.Series([False] * len(result_df), index=result_df.index)
    if rank_col:
        rank = _parse_rank(result_df[rank_col])
        top3_flags = top3_flags | ((rank >= 1) & (rank <= 3))
    if top3_col:
        top3_flags = top3_flags | _to_bool_series(result_df[top3_col])
    out["_is_top3"] = top3_flags
    out = out[out["_horse_key"] != ""].copy()
    out = (
        out.groupby(["_race_id", "_horse_key"], as_index=False)["_is_top3"]
        .max()
        .copy()
    )
    return out


def _common_races(pred: pd.DataFrame, result: pd.DataFrame) -> List[str]:
    races = sorted(set(pred["_race_id"].tolist()) & set(result["_race_id"].tolist()))
    return races


def compute_hit_at_k(pred_df, result_df, k=5) -> float:
    if int(k) <= 0:
        return 0.0
    pred = _prepare_pred(pred_df)
    result = _prepare_result(result_df)
    races = _common_races(pred, result)
    if not races:
        return 0.0

    pred_topk = (
        pred[pred["_race_id"].isin(races)]
        .sort_values(["_race_id", "_score"], ascending=[True, False])
        .groupby("_race_id", as_index=False)
        .head(int(k))
    )
    pred_map = pred_topk.groupby("_race_id")["_horse_key"].apply(set).to_dict()
    top3 = result[result["_is_top3"]]
    top3_map = top3.groupby("_race_id")["_horse_key"].apply(set).to_dict()

    hits = []
    for race_id in races:
        picks = pred_map.get(race_id, set())
        actual = top3_map.get(race_id, set())
        hits.append(1.0 if picks and actual and len(picks & actual) > 0 else 0.0)
    return float(sum(hits) / len(hits)) if hits else 0.0


def compute_top3_hits_at_k(pred_df, result_df, k=5) -> float:
    if int(k) <= 0:
        return 0.0
    pred = _prepare_pred(pred_df)
    result = _prepare_result(result_df)
    races = _common_races(pred, result)
    if not races:
        return 0.0

    pred_topk = (
        pred[pred["_race_id"].isin(races)]
        .sort_values(["_race_id", "_score"], ascending=[True, False])
        .groupby("_race_id", as_index=False)
        .head(int(k))
    )
    pred_map = pred_topk.groupby("_race_id")["_horse_key"].apply(set).to_dict()
    top3 = result[result["_is_top3"]]
    top3_map = top3.groupby("_race_id")["_horse_key"].apply(set).to_dict()

    counts = []
    for race_id in races:
        picks = pred_map.get(race_id, set())
        actual = top3_map.get(race_id, set())
        counts.append(float(min(3, len(picks & actual))) if picks and actual else 0.0)
    return float(sum(counts) / len(counts)) if counts else 0.0


def compute_mrr_top3(pred_df, result_df, k=10) -> float:
    if int(k) <= 0:
        return 0.0
    pred = _prepare_pred(pred_df)
    result = _prepare_result(result_df)
    races = _common_races(pred, result)
    if not races:
        return 0.0

    top3 = result[result["_is_top3"]]
    top3_map = top3.groupby("_race_id")["_horse_key"].apply(set).to_dict()
    pred_topk = (
        pred[pred["_race_id"].isin(races)]
        .sort_values(["_race_id", "_score"], ascending=[True, False])
        .groupby("_race_id")["_horse_key"]
        .apply(list)
        .to_dict()
    )

    values = []
    limit = int(k)
    for race_id in races:
        order = pred_topk.get(race_id, [])[:limit]
        actual = top3_map.get(race_id, set())
        rr = 0.0
        for idx, horse in enumerate(order, start=1):
            if horse in actual:
                rr = 1.0 / float(idx)
                break
        values.append(rr)
    return float(sum(values) / len(values)) if values else 0.0


def compute_brier_score(pred_df, result_df) -> float:
    pred = _prepare_pred(pred_df)
    result = _prepare_result(result_df)
    races = _common_races(pred, result)
    if not races:
        return 0.0
    left = pred[pred["_race_id"].isin(races)].copy()
    right = result[result["_race_id"].isin(races)][["_race_id", "_horse_key", "_is_top3"]].copy()
    merged = left.merge(right, on=["_race_id", "_horse_key"], how="left")
    if merged.empty:
        return 0.0
    y = merged["_is_top3"].fillna(False).astype(float)
    brier = ((merged["_score"] - y) ** 2).mean()
    if pd.isna(brier):
        return 0.0
    return float(brier)

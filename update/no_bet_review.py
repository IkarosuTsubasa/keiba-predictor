import argparse
import math
import os
from pathlib import Path

import pandas as pd


# 这是一个“反思系统”，不是“自我学习系统”。所有输出只用于人工分析，不用于当场决策。


BASE_DIR = Path(__file__).resolve().parent


def resolve_scope_key():
    raw = os.environ.get("SCOPE_KEY", "").strip().lower()
    raw = raw.replace(" ", "_").replace("-", "_").replace("/", "_")
    if raw in ("central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"):
        return "central_turf"
    if raw in ("central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"):
        return "central_dirt"
    if raw in ("local", "l", "3"):
        return "local"
    return "central_dirt"


def ensure_column(df, name, default=""):
    if name not in df.columns:
        df[name] = default
    return df


def load_log(path):
    if not path.exists():
        print(f"[ERROR] No bet log not found: {path}")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = [
        "scope",
        "race_id",
        "bet_type",
        "horse_pair",
        "model_prob",
        "market_prob_open",
        "ev_ratio_open",
        "no_bet_reason",
        "market_prob_close",
    ]
    for col in expected:
        ensure_column(df, col, "")
    return df


def add_clv_fields(df):
    df["model_prob_num"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["market_prob_close_num"] = pd.to_numeric(df["market_prob_close"], errors="coerce")
    df["ev_ratio_open_num"] = pd.to_numeric(df["ev_ratio_open"], errors="coerce")

    df["clv_ratio"] = pd.NA
    df["clv_log"] = pd.NA
    df["no_bet_correct"] = pd.NA

    valid = (df["model_prob_num"] > 0) & (df["market_prob_close_num"] > 0)
    df.loc[valid, "clv_ratio"] = (
        df.loc[valid, "model_prob_num"] / df.loc[valid, "market_prob_close_num"]
    )
    df.loc[valid, "clv_log"] = df.loc[valid, "clv_ratio"].apply(math.log)
    df.loc[valid, "no_bet_correct"] = df.loc[valid, "clv_ratio"] <= 1.0
    return df, valid


def format_rate_table(df):
    if df.empty:
        return "No rows with market_prob_close."
    grouped = (
        df.groupby(["scope", "bet_type"])["no_bet_correct"]
        .agg(total="size", correct="sum")
        .reset_index()
    )
    grouped["correct_rate"] = (grouped["correct"] / grouped["total"]).round(4)
    return grouped.to_string(index=False)


def add_bins(series, bins):
    return pd.cut(series, bins=bins, right=False, include_lowest=True)


def summarize_missed_edges(df):
    if df.empty:
        return None, None
    ev_bins = [0.0, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0, float("inf")]
    prob_bins = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 1.0]
    df = df.copy()
    df["ev_ratio_bin"] = add_bins(df["ev_ratio_open_num"], ev_bins)
    df["model_prob_bin"] = add_bins(df["model_prob_num"], prob_bins)
    ev_counts = df["ev_ratio_bin"].value_counts().sort_index()
    prob_counts = df["model_prob_bin"].value_counts().sort_index()
    return ev_counts, prob_counts


def main():
    parser = argparse.ArgumentParser(description="No Bet reflection report")
    parser.add_argument("--scope", help="Scope key (central_turf/central_dirt/local)")
    parser.add_argument("--path", help="Path to no_bet_log_<scope>.csv")
    args = parser.parse_args()

    scope = args.scope or resolve_scope_key()
    path = Path(args.path) if args.path else (BASE_DIR / f"no_bet_log_{scope}.csv")

    df = load_log(path)
    if df is None:
        return

    df, valid = add_clv_fields(df)
    df_valid = df.loc[valid]

    print("No Bet correct rate by scope/bet_type:")
    print(format_rate_table(df_valid))

    missing_close = (~valid).sum()
    print(f"Rows missing market_prob_close: {missing_close}")

    missed = df_valid.loc[df_valid["no_bet_correct"] == False]
    ev_counts, prob_counts = summarize_missed_edges(missed)
    if ev_counts is None:
        print("No missed-edge rows (no_bet_correct == false).")
        return

    print("Missed-edge distribution by ev_ratio_open:")
    print(ev_counts.to_string())
    print("Missed-edge distribution by model_prob:")
    print(prob_counts.to_string())


if __name__ == "__main__":
    main()

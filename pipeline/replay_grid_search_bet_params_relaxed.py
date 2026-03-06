import argparse
import itertools
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from bet_engine_v3 import generate_bet_plan_v3
from record_pipeline import (
    load_fuku_odds_map,
    load_odds_map,
    load_quinella_odds_map,
    load_wide_odds_map,
    parse_horse_no,
)
from replay_grid_search_bet_params import (
    _build_no_to_name_map,
    _build_pair_odds_payload,
    _build_place_odds_payload,
    _build_run_rows,
    _calc_metrics,
    _collect_bucket_counts,
    _evaluate_items,
    _load_predictor_config,
)
from surface_scope import normalize_scope_key


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SCOPES = "central_dirt,central_turf,local"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")


def _norm_name(text) -> str:
    return "".join(str(text or "").split())


def _pick_prob_col(df: pd.DataFrame) -> str:
    for col in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if col in df.columns:
            return col
    return ""


def _build_inputs_robust(pred_path: Path, odds_path: Path, fuku_path: Path, wide_path: Path, quinella_path: Path):
    pred = _read_csv(pred_path)
    odds = _read_csv(odds_path)
    if "HorseName" not in pred.columns or "name" not in odds.columns:
        raise ValueError("missing HorseName or name")

    pred = pred.copy()
    odds = odds.copy()
    pred["name_key"] = pred["HorseName"].apply(_norm_name)
    odds["name_key"] = odds["name"].apply(_norm_name)
    odds = odds.drop_duplicates(subset=["name_key"], keep="last")
    merged = pred.merge(odds, on="name_key", how="left")

    prob_col = _pick_prob_col(merged)
    if not prob_col:
        raise ValueError("missing Top3Prob column")

    merged["horse_key"] = merged.get("horse_no").apply(
        lambda x: str(parse_horse_no(x)) if parse_horse_no(x) is not None else ""
    )
    merged["Top3Prob_model"] = pd.to_numeric(merged.get(prob_col), errors="coerce").fillna(0.0)
    if "rank_score" in merged.columns:
        rank_series = pd.to_numeric(merged["rank_score"], errors="coerce")
        merged["rank_score"] = rank_series.fillna(merged["Top3Prob_model"])
    else:
        merged["rank_score"] = merged["Top3Prob_model"]
    merged["odds_num"] = pd.to_numeric(merged.get("odds"), errors="coerce")
    merged = merged[(merged["horse_key"] != "") & merged["Top3Prob_model"].notna()].copy()
    if merged.empty:
        raise ValueError("empty merged rows")

    pred_df = merged[["horse_key", "rank_score", "Top3Prob_model"]].copy()
    pred_df["race_id"] = "__single_race__"

    win_map = {}
    for _, row in merged.iterrows():
        hk = str(row.get("horse_key", "")).strip()
        odd = _safe_float(row.get("odds_num"), 0.0)
        if hk and odd > 0:
            win_map[hk] = odd

    odds_payload = {
        "win": win_map,
        "place": _build_place_odds_payload(fuku_path),
        "wide": _build_pair_odds_payload(wide_path),
        "quinella": _build_pair_odds_payload(quinella_path),
    }
    return pred_df, odds_payload


def _parse_scopes(text: str) -> List[str]:
    out: List[str] = []
    for token in str(text or "").split(","):
        key = normalize_scope_key(token.strip())
        if key and key not in out:
            out.append(key)
    return out or [normalize_scope_key(x) for x in DEFAULT_SCOPES.split(",")]


def _bucket_from_odds(odds_value: float) -> str:
    odd = _safe_float(odds_value, 0.0)
    if odd < 3.0:
        return "low"
    if odd < 10.0:
        return "mid"
    return "high"


def _iter_relaxed_grid() -> List[Dict[str, float]]:
    grids = {
        "min_p_hit_per_ticket": [0.06, 0.08, 0.10],
        "min_p_win_per_ticket": [0.03, 0.04, 0.05],
        "min_edge_per_ticket": [0.00, 0.01, 0.02],
    }
    keys = list(grids.keys())
    values = [grids[k] for k in keys]
    out = []
    for prod in itertools.product(*values):
        out.append(dict(zip(keys, prod)))
    return out


def _iter_coverage_tune_grid() -> List[Dict[str, float]]:
    out = []
    for kelly_scale in (1.0, 3.0):
        for min_p_hit in (0.06, 0.04):
            for fallback_max_odds_place in (10.0, 15.0):
                out.append(
                    {
                        "kelly_scale": float(kelly_scale),
                        "min_p_hit_per_ticket": float(min_p_hit),
                        "fallback_max_odds_place": float(fallback_max_odds_place),
                    }
                )
    return out


def _build_combo_cfg(base_cfg: Dict, combo: Dict) -> Dict:
    cfg = json.loads(json.dumps(base_cfg if isinstance(base_cfg, dict) else {}))
    cfg["min_p_hit_per_ticket"] = float(combo["min_p_hit_per_ticket"])
    cfg["min_p_win_per_ticket"] = float(combo["min_p_win_per_ticket"])
    cfg["min_edge_per_ticket"] = float(combo["min_edge_per_ticket"])
    # Keep legacy gates permissive so this search isolates the three target thresholds.
    cfg["min_p"] = {"win": 0.0, "place": 0.0, "wide": 0.0, "quinella": 0.0}
    cfg["min_ev"] = {"win": -1.0, "place": -1.0, "wide": -1.0, "quinella": -1.0}
    cfg["odds_power"] = 0.70
    cfg["rank_weight_floor"] = 0.70
    cfg["rank_weight_ceil"] = 1.00
    cfg["ticket_prob_floor_scale"] = 3.0
    cfg["max_high_odds_tickets_per_race"] = 1
    cfg["min_low_or_mid_presence"] = False
    return cfg


def _load_cases(scopes: List[str]) -> Tuple[List[Dict], Dict[str, Dict], Dict[str, int]]:
    base_cfgs = {}
    cases: List[Dict] = []
    total_by_scope = {}
    for scope in scopes:
        cfg = _load_predictor_config(scope).get("bet_engine_v3", {})
        base_cfgs[scope] = dict(cfg) if isinstance(cfg, dict) else {}
        runs = _build_run_rows(scope)
        total_by_scope[scope] = len(runs)
        for run in runs:
            try:
                pred_df, odds_payload = _build_inputs_robust(
                    pred_path=run["pred_path"],
                    odds_path=run["odds_path"],
                    fuku_path=run["fuku_path"],
                    wide_path=run["wide_path"],
                    quinella_path=run["quinella_path"],
                )
            except Exception:
                continue
            cases.append(
                {
                    "scope": scope,
                    "run_id": run["run_id"],
                    "pred_df": pred_df,
                    "odds_payload": odds_payload,
                    "actual_top3": run["actual_top3"],
                    "no_to_name": _build_no_to_name_map(run["odds_path"]),
                    "odds_map": load_odds_map(str(run["odds_path"])),
                    "wide_odds_map": load_wide_odds_map(str(run["wide_path"])),
                    "fuku_odds_map": load_fuku_odds_map(str(run["fuku_path"])),
                    "quinella_odds_map": load_quinella_odds_map(str(run["quinella_path"])),
                }
            )
    return cases, base_cfgs, total_by_scope


def _evaluate_combo(cases: List[Dict], cfg_by_scope: Dict[str, Dict], bankroll: int, debug_once: bool = False) -> Dict:
    rows = []
    debug_emitted = False
    reason_keys = (
        "no_candidate_after_min_p",
        "no_candidate_after_min_edge",
        "no_candidate_after_quota",
        "budget_allocation_zero",
        "other",
    )
    reason_counts = {k: 0 for k in reason_keys}
    total_candidates_before = 0
    total_after_min_p = 0
    total_after_min_edge = 0
    total_after_quota = 0
    total_after_budget_alloc = 0
    allocation_zero_race_count = 0
    allocation_fallback_count = 0
    fallback_bucket_counts = {"low": 0, "mid": 0, "high": 0}
    fallback_type_counts = {"win": 0, "place": 0, "wide": 0}
    exposure_cap_trigger_count = 0
    exposure_high_share_sum = 0.0
    exposure_low_mid_share_sum = 0.0
    exposure_share_race_count = 0

    for case in cases:
        cfg = dict(cfg_by_scope[case["scope"]])
        if debug_once and (not debug_emitted):
            cfg["debug_first_bet_race"] = True
        items, _, summary_info = generate_bet_plan_v3(
            pred_df=case["pred_df"],
            odds=case["odds_payload"],
            bankroll_yen=int(bankroll),
            scope_key=case["scope"],
            config=cfg,
        )
        diag_list = summary_info.get("race_diagnostics", []) if isinstance(summary_info, dict) else []
        diag = diag_list[0] if isinstance(diag_list, list) and diag_list else {}
        c_before = int(diag.get("before_min_p", 0))
        c_after_min_p = int(diag.get("after_min_p", 0))
        c_after_min_edge = int(diag.get("after_min_edge", 0))
        c_after_quota = int(diag.get("after_quota", 0))
        c_after_alloc = int(diag.get("after_budget_alloc", 0))
        total_candidates_before += c_before
        total_after_min_p += c_after_min_p
        total_after_min_edge += c_after_min_edge
        total_after_quota += c_after_quota
        total_after_budget_alloc += c_after_alloc
        if c_after_quota > 0 and c_after_alloc <= 0:
            allocation_zero_race_count += 1
        if bool(diag.get("exposure_cap_triggered", False)):
            exposure_cap_trigger_count += 1
        exposure_total_after = int(_safe_float(diag.get("exposure_total_stake_after", 0), 0))
        if exposure_total_after > 0:
            exposure_high_share_sum += float(_safe_float(diag.get("exposure_high_stake_share_after", 0.0), 0.0))
            exposure_low_mid_share_sum += float(_safe_float(diag.get("exposure_low_mid_stake_share_after", 0.0), 0.0))
            exposure_share_race_count += 1
        if bool(diag.get("allocation_fallback_triggered", False)):
            allocation_fallback_count += 1
            fb_bucket = _bucket_from_odds(_safe_float(diag.get("fallback_odds", 0.0), 0.0))
            fallback_bucket_counts[fb_bucket] = fallback_bucket_counts.get(fb_bucket, 0) + 1
            fb_type = str(diag.get("fallback_ticket_type", "")).strip().lower()
            if fb_type in fallback_type_counts:
                fallback_type_counts[fb_type] += 1

        stake, payout, tickets = _evaluate_items(
            items,
            case["no_to_name"],
            case["actual_top3"],
            case["odds_map"],
            case["wide_odds_map"],
            case["fuku_odds_map"],
            case["quinella_odds_map"],
        )
        b = _collect_bucket_counts(items)
        rows.append(
            {
                "run_id": case["run_id"],
                "stake": stake,
                "profit": payout - stake,
                "tickets": tickets,
                "bucket_low": b["bucket_low"],
                "bucket_mid": b["bucket_mid"],
                "bucket_high": b["bucket_high"],
            }
        )
        if int(stake) <= 0:
            reason = str(diag.get("no_bet_reason", "other") or "other").strip()
            if reason not in reason_counts:
                reason = "other"
            reason_counts[reason] += 1
        if debug_once and (not debug_emitted) and int(stake) > 0:
            debug_emitted = True

    m = _calc_metrics(rows, bankroll=int(bankroll))
    bet_runs = int(sum(1 for r in rows if int(r.get("stake", 0)) > 0))
    fb_count = int(allocation_fallback_count)
    if fb_count > 0:
        fb_low_ratio = float(fallback_bucket_counts["low"]) / float(fb_count)
        fb_mid_ratio = float(fallback_bucket_counts["mid"]) / float(fb_count)
        fb_high_ratio = float(fallback_bucket_counts["high"]) / float(fb_count)
        fb_win_ratio = float(fallback_type_counts["win"]) / float(fb_count)
        fb_place_ratio = float(fallback_type_counts["place"]) / float(fb_count)
        fb_wide_ratio = float(fallback_type_counts["wide"]) / float(fb_count)
    else:
        fb_low_ratio = 0.0
        fb_mid_ratio = 0.0
        fb_high_ratio = 0.0
        fb_win_ratio = 0.0
        fb_place_ratio = 0.0
        fb_wide_ratio = 0.0
    fallback_ratio = (float(fb_count) / float(bet_runs)) if bet_runs > 0 else 0.0
    total_races = int(len(rows))
    exposure_cap_trigger_rate = (float(exposure_cap_trigger_count) / float(total_races)) if total_races > 0 else 0.0
    avg_high_stake_share_after = (
        float(exposure_high_share_sum) / float(exposure_share_race_count) if exposure_share_race_count > 0 else 0.0
    )
    avg_low_mid_stake_share_after = (
        float(exposure_low_mid_share_sum) / float(exposure_share_race_count) if exposure_share_race_count > 0 else 0.0
    )
    overall_low = float(m.get("bucket_low_ratio", float("nan")))
    overall_mid = float(m.get("bucket_mid_ratio", float("nan")))
    overall_high = float(m.get("bucket_high_ratio", float("nan")))
    return {
        "runs_used": int(len(rows)),
        "bet_runs": bet_runs,
        "stake_total": int(m.get("stake_total", 0)),
        "stake_roi": float(m.get("roi_stake", float("nan"))),
        "cash_as_1_roi": float(m.get("roi_cash_as_1", float("nan"))),
        "no_bet_ratio": float(m.get("no_bet_rate", float("nan"))),
        "avg_tickets_per_race": float(m.get("avg_tickets", float("nan"))),
        "low_ratio": overall_low,
        "mid_ratio": overall_mid,
        "high_ratio": overall_high,
        "overall_bucket_low": overall_low,
        "overall_bucket_mid": overall_mid,
        "overall_bucket_high": overall_high,
        "total_candidates_before": int(total_candidates_before),
        "total_after_min_p": int(total_after_min_p),
        "total_after_min_edge": int(total_after_min_edge),
        "total_after_quota": int(total_after_quota),
        "total_after_budget_alloc": int(total_after_budget_alloc),
        "no_bet_reason_no_candidate_after_min_p": int(reason_counts["no_candidate_after_min_p"]),
        "no_bet_reason_no_candidate_after_min_edge": int(reason_counts["no_candidate_after_min_edge"]),
        "no_bet_reason_no_candidate_after_quota": int(reason_counts["no_candidate_after_quota"]),
        "no_bet_reason_budget_allocation_zero": int(reason_counts["budget_allocation_zero"]),
        "no_bet_reason_other": int(reason_counts["other"]),
        "allocation_zero_race_count": int(allocation_zero_race_count),
        "allocation_fallback_count": fb_count,
        "fallback_ratio": fallback_ratio,
        "fallback_bucket_low_ratio": fb_low_ratio,
        "fallback_bucket_mid_ratio": fb_mid_ratio,
        "fallback_bucket_high_ratio": fb_high_ratio,
        "fallback_type_win_ratio": fb_win_ratio,
        "fallback_type_place_ratio": fb_place_ratio,
        "fallback_type_wide_ratio": fb_wide_ratio,
        "exposure_cap_trigger_rate": exposure_cap_trigger_rate,
        "avg_high_stake_share_after": avg_high_stake_share_after,
        "avg_low_mid_stake_share_after": avg_low_mid_stake_share_after,
    }


def _run_extreme_sanity(cases: List[Dict], base_cfgs: Dict[str, Dict], scopes: List[str], bankroll: int):
    combo_a = {
        "min_p_hit_per_ticket": 0.0,
        "min_p_win_per_ticket": 0.0,
        "min_edge_per_ticket": -1.0,
    }
    combo_b = {
        "min_p_hit_per_ticket": 0.30,
        "min_p_win_per_ticket": 0.15,
        "min_edge_per_ticket": 0.20,
    }
    cfg_a = {scope: _build_combo_cfg(base_cfgs.get(scope, {}), combo_a) for scope in scopes}
    cfg_b = {scope: _build_combo_cfg(base_cfgs.get(scope, {}), combo_b) for scope in scopes}
    res_a = _evaluate_combo(cases, cfg_a, bankroll=bankroll, debug_once=True)
    res_b = _evaluate_combo(cases, cfg_b, bankroll=bankroll, debug_once=False)
    print("\n[EXTREME SANITY CHECK]")
    print(
        "A(min_p_hit=0.0,min_p_win=0.0,min_edge=-1.0): "
        f"bet_runs={res_a['bet_runs']} stake_total={res_a['stake_total']} "
        f"stake_roi={res_a['stake_roi']:.6f} no_bet_ratio={res_a['no_bet_ratio']:.6f}"
    )
    print(
        "B(min_p_hit=0.30,min_p_win=0.15,min_edge=0.20): "
        f"bet_runs={res_b['bet_runs']} stake_total={res_b['stake_total']} "
        f"stake_roi={res_b['stake_roi']:.6f} no_bet_ratio={res_b['no_bet_ratio']:.6f}"
    )
    bet_diff = abs(int(res_a["bet_runs"]) - int(res_b["bet_runs"]))
    stake_diff = abs(int(res_a["stake_total"]) - int(res_b["stake_total"]))
    if bet_diff < 5 and stake_diff < 2000:
        raise RuntimeError(
            "Extreme sanity check failed: A/B bet_runs and stake_total are not significantly different."
        )


def _sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    w = df.copy()
    w["_cash_sort"] = pd.to_numeric(w["cash_as_1_roi"], errors="coerce").fillna(-1e18)
    w["_stake_sort"] = pd.to_numeric(w["stake_roi"], errors="coerce").fillna(-1e18)
    w["_high_sort"] = pd.to_numeric(w["high_ratio"], errors="coerce").fillna(1e18)
    w["_nobet_sort"] = pd.to_numeric(w["no_bet_ratio"], errors="coerce").fillna(1e18)
    w = w.sort_values(
        by=["_cash_sort", "_stake_sort", "_high_sort", "_nobet_sort"],
        ascending=[False, False, True, True],
    ).drop(columns=["_cash_sort", "_stake_sort", "_high_sort", "_nobet_sort"])
    return w.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Relaxed threshold grid replay for bet_engine_v3.")
    parser.add_argument("--scopes", default=DEFAULT_SCOPES)
    parser.add_argument("--bankroll", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--attribution-only", action="store_true")
    parser.add_argument("--validate-fallback", action="store_true")
    parser.add_argument("--coverage-tune", action="store_true")
    parser.add_argument("--attr-min-p-hit", type=float, default=0.06)
    parser.add_argument("--attr-min-p-win", type=float, default=0.03)
    parser.add_argument("--attr-min-edge", type=float, default=0.00)
    args = parser.parse_args()

    scopes = _parse_scopes(args.scopes)
    bankroll = int(args.bankroll)
    grid = _iter_relaxed_grid()

    cases, base_cfgs, total_by_scope = _load_cases(scopes)
    print(f"scopes={','.join(scopes)} bankroll={bankroll}")
    print(f"total_runs_by_scope={total_by_scope}")
    print(f"usable_runs={len(cases)}")
    if bool(args.coverage_tune):
        coverage_grid = _iter_coverage_tune_grid()
        coverage_rows = []
        for combo_idx, combo in enumerate(coverage_grid, start=1):
            base_combo = {
                "min_p_hit_per_ticket": float(combo["min_p_hit_per_ticket"]),
                "min_p_win_per_ticket": float(args.attr_min_p_win),
                "min_edge_per_ticket": float(args.attr_min_edge),
            }
            cfg_by_scope = {scope: _build_combo_cfg(base_cfgs.get(scope, {}), base_combo) for scope in scopes}
            for scope in scopes:
                cfg_by_scope[scope]["kelly_scale"] = float(combo["kelly_scale"])
                cfg_by_scope[scope]["fallback_max_odds_place"] = float(combo["fallback_max_odds_place"])
            metric = _evaluate_combo(cases, cfg_by_scope, bankroll=bankroll, debug_once=(combo_idx == 1))
            coverage_rows.append(
                {
                    "kelly_scale": float(combo["kelly_scale"]),
                    "min_p_hit_per_ticket": float(combo["min_p_hit_per_ticket"]),
                    "min_p_win_per_ticket": float(args.attr_min_p_win),
                    "min_edge_per_ticket": float(args.attr_min_edge),
                    "fallback_max_odds_place": float(combo["fallback_max_odds_place"]),
                    **metric,
                }
            )

        coverage_df = pd.DataFrame(coverage_rows)
        coverage_df["overall_high_le_010"] = (
            pd.to_numeric(coverage_df["overall_bucket_high"], errors="coerce").fillna(1.0) <= 0.10
        )
        coverage_df["bet_runs_ge_100"] = pd.to_numeric(coverage_df["bet_runs"], errors="coerce").fillna(0) >= 100
        coverage_df["eligible"] = coverage_df["overall_high_le_010"] & coverage_df["bet_runs_ge_100"]
        ranked = coverage_df.sort_values(
            by=["bet_runs", "cash_as_1_roi", "stake_roi"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        eligible_ranked = ranked[ranked["eligible"]].reset_index(drop=True)
        top3_eligible = eligible_ranked.head(3)
        top3_overall = ranked.head(3)

        out_dir = BASE_DIR / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / f"replay_coverage_tune_summary_{ts}.csv"
        coverage_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        cols = [
            "kelly_scale",
            "min_p_hit_per_ticket",
            "fallback_max_odds_place",
            "bet_runs",
            "no_bet_ratio",
            "overall_bucket_low",
            "overall_bucket_mid",
            "overall_bucket_high",
            "stake_roi",
            "cash_as_1_roi",
            "allocation_zero_race_count",
            "allocation_fallback_count",
            "fallback_bucket_low_ratio",
            "fallback_bucket_mid_ratio",
            "fallback_bucket_high_ratio",
            "fallback_type_win_ratio",
            "fallback_type_place_ratio",
            "fallback_type_wide_ratio",
            "exposure_cap_trigger_rate",
            "avg_high_stake_share_after",
            "avg_low_mid_stake_share_after",
            "overall_high_le_010",
            "bet_runs_ge_100",
            "eligible",
        ]
        print("\n[Coverage Tune Top3 Eligible]")
        if top3_eligible.empty:
            print("no eligible combo found (constraint: overall_high<=0.10 and bet_runs>=100)")
        else:
            print(top3_eligible[cols].to_string(index=False))
        print("\n[Coverage Tune Top3 Overall]")
        print(top3_overall[cols].to_string(index=False))
        print(f"saved summary: {summary_path}")
        return
    if bool(args.attribution_only):
        combo = {
            "min_p_hit_per_ticket": float(args.attr_min_p_hit),
            "min_p_win_per_ticket": float(args.attr_min_p_win),
            "min_edge_per_ticket": float(args.attr_min_edge),
        }
        cfg_by_scope = {scope: _build_combo_cfg(base_cfgs.get(scope, {}), combo) for scope in scopes}
        metric = _evaluate_combo(cases, cfg_by_scope, bankroll=bankroll, debug_once=True)
        row = {**combo, **metric}
        out_dir = BASE_DIR / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / f"replay_grid_search_bet_params_relaxed_summary_{ts}.csv"
        pd.DataFrame([row]).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("\n[Attribution Summary]")
        for k in (
            "bet_runs",
            "stake_total",
            "stake_roi",
            "cash_as_1_roi",
            "no_bet_ratio",
            "avg_tickets_per_race",
            "low_ratio",
            "mid_ratio",
            "high_ratio",
            "total_candidates_before",
            "total_after_min_p",
            "total_after_min_edge",
            "total_after_quota",
            "total_after_budget_alloc",
            "no_bet_reason_no_candidate_after_min_p",
            "no_bet_reason_no_candidate_after_min_edge",
            "no_bet_reason_no_candidate_after_quota",
            "no_bet_reason_budget_allocation_zero",
            "no_bet_reason_other",
            "allocation_zero_race_count",
            "allocation_fallback_count",
        ):
            print(f"{k}={row[k]}")
        print(f"saved summary: {summary_path}")
        return
    if bool(args.validate_fallback):
        combo = {
            "min_p_hit_per_ticket": float(args.attr_min_p_hit),
            "min_p_win_per_ticket": float(args.attr_min_p_win),
            "min_edge_per_ticket": float(args.attr_min_edge),
        }
        cfg_by_scope = {scope: _build_combo_cfg(base_cfgs.get(scope, {}), combo) for scope in scopes}
        metric = _evaluate_combo(cases, cfg_by_scope, bankroll=bankroll, debug_once=True)
        row = {**combo, **metric}
        out_dir = BASE_DIR / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / f"replay_validate_fallback_summary_{ts}.csv"
        pd.DataFrame([row]).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("\n[Fallback Validate]")
        print(
            "bet_runs={bet_runs} stake_roi={stake_roi:.6f} cash_as_1_roi={cash_as_1_roi:.6f}".format(
                bet_runs=int(row["bet_runs"]),
                stake_roi=float(row["stake_roi"]),
                cash_as_1_roi=float(row["cash_as_1_roi"]),
            )
        )
        print(
            "overall buckets: low={:.6f} mid={:.6f} high={:.6f}".format(
                float(row["overall_bucket_low"]),
                float(row["overall_bucket_mid"]),
                float(row["overall_bucket_high"]),
            )
        )
        print(
            "fallback buckets: low={:.6f} mid={:.6f} high={:.6f}".format(
                float(row["fallback_bucket_low_ratio"]),
                float(row["fallback_bucket_mid_ratio"]),
                float(row["fallback_bucket_high_ratio"]),
            )
        )
        print(
            "fallback type ratios: win={:.6f} place={:.6f} wide={:.6f}".format(
                float(row["fallback_type_win_ratio"]),
                float(row["fallback_type_place_ratio"]),
                float(row["fallback_type_wide_ratio"]),
            )
        )
        print(f"saved summary: {summary_path}")
        return
    _run_extreme_sanity(cases, base_cfgs, scopes, bankroll=bankroll)

    summary_rows = []
    for combo_idx, combo in enumerate(grid, start=1):
        cfg_by_scope = {scope: _build_combo_cfg(base_cfgs.get(scope, {}), combo) for scope in scopes}
        metric = _evaluate_combo(cases, cfg_by_scope, bankroll=bankroll, debug_once=(combo_idx == 1))
        row = {**combo, **metric}
        row["low_sample_warning"] = bool(int(row["bet_runs"]) < 20)
        row["longshot_bias_warning"] = bool(_safe_float(row["high_ratio"], 0.0) > 0.20)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = _sort_summary(summary_df)
    if int(summary_df["bet_runs"].nunique()) <= 1 and int(summary_df["stake_total"].nunique()) <= 1:
        raise RuntimeError("27-combo grid produced identical bet_runs/stake_total across all combos.")

    out_dir = BASE_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"replay_grid_search_bet_params_relaxed_summary_{ts}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    cols_top = [
        "min_p_hit_per_ticket",
        "min_p_win_per_ticket",
        "min_edge_per_ticket",
        "bet_runs",
        "stake_total",
        "stake_roi",
        "cash_as_1_roi",
        "no_bet_ratio",
        "avg_tickets_per_race",
        "low_ratio",
        "mid_ratio",
        "high_ratio",
        "low_sample_warning",
        "longshot_bias_warning",
    ]
    print("\nTOP 10 参数组合：")
    print(summary_df.head(max(1, int(args.top_k)))[cols_top].to_string(index=False))

    top3 = summary_df.head(3)
    print("\nBEST PARAMS（Top3）:")
    for idx, row in top3.iterrows():
        print(
            f"Top{idx + 1}: "
            f"min_p_hit_per_ticket={row['min_p_hit_per_ticket']:.2f}, "
            f"min_p_win_per_ticket={row['min_p_win_per_ticket']:.2f}, "
            f"min_edge_per_ticket={row['min_edge_per_ticket']:.2f}, "
            f"cash_as_1_roi={row['cash_as_1_roi']:.6f}, "
            f"stake_roi={row['stake_roi']:.6f}, "
            f"high_ratio={row['high_ratio']:.6f}, "
            f"no_bet_ratio={row['no_bet_ratio']:.6f}"
        )

    recommended = summary_df[
        (~summary_df["low_sample_warning"]) & (~summary_df["longshot_bias_warning"])
    ].head(1)
    if recommended.empty:
        recommended = summary_df.head(1)
    rec = recommended.iloc[0].to_dict()

    print("\n===== GRID SEARCH RESULT =====")
    print(f"total combos: {len(grid)}")
    if not top3.empty:
        for i in range(min(3, len(top3))):
            row = top3.iloc[i]
            print(
                f"Top{i + 1} params: "
                f"min_p_hit_per_ticket={row['min_p_hit_per_ticket']:.2f}, "
                f"min_p_win_per_ticket={row['min_p_win_per_ticket']:.2f}, "
                f"min_edge_per_ticket={row['min_edge_per_ticket']:.2f}"
            )
    print(
        "recommended params for next run: "
        f"min_p_hit_per_ticket={float(rec['min_p_hit_per_ticket']):.2f}, "
        f"min_p_win_per_ticket={float(rec['min_p_win_per_ticket']):.2f}, "
        f"min_edge_per_ticket={float(rec['min_edge_per_ticket']):.2f}"
    )
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()

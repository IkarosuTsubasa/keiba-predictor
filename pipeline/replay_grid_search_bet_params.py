import argparse
import itertools
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from bet_engine_v2 import generate_bet_plan_v2
from bet_engine_v3 import generate_bet_plan_v3
from bet_engine_v4 import generate_bet_plan_v4
from bet_engine_v5 import generate_bet_plan_v5
from record_pipeline import (
    estimate_payout_multiplier,
    eval_ticket,
    load_fuku_odds_map,
    load_odds_map,
    load_quinella_odds_map,
    load_wide_odds_map,
    normalize_name,
    parse_horse_no,
)
from surface_scope import get_data_dir, get_predictor_config_path, normalize_scope_key


BASE_DIR = Path(__file__).resolve().parent
RUN_ID_RE = re.compile(r"(\d{8}_\d{6})")
BET_TYPES = ("win", "place", "wide", "quinella")
ENGINE_CHOICES = ("v2", "v3", "v4", "v5")


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _extract_run_id(text: str) -> str:
    m = RUN_ID_RE.search(str(text or ""))
    return m.group(1) if m else ""


def _norm_name(text) -> str:
    return "".join(str(text or "").split())


def _pick_prob_col(df: pd.DataFrame) -> str:
    for col in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if col in df.columns:
            return col
    return ""


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


def _load_predictor_config(scope_key: str) -> Dict:
    path = get_predictor_config_path(BASE_DIR, scope_key)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_run_file_map(scope_dir: Path, prefix: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in scope_dir.rglob(f"{prefix}*.csv"):
        rid = _extract_run_id(p.name) or _extract_run_id(str(p))
        if not rid:
            continue
        prev = out.get(rid)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            out[rid] = p
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")


def _build_no_to_name_map(odds_path: Path) -> Dict[int, str]:
    if not odds_path.exists():
        return {}
    df = _read_csv(odds_path)
    if "horse_no" not in df.columns or "name" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        no = parse_horse_no(row.get("horse_no"))
        if no is None:
            continue
        name = _norm_name(row.get("name"))
        if not name:
            continue
        out[int(no)] = name
    return out


def _canonical_pair(a: int, b: int) -> Tuple[str, str]:
    x, y = int(a), int(b)
    if x > y:
        x, y = y, x
    return str(x), str(y)


def _build_place_odds_payload(path: Path) -> Dict[str, object]:
    out = {}
    if not path.exists():
        return out
    df = _read_csv(path)
    if "horse_no" not in df.columns:
        return out
    for _, row in df.iterrows():
        horse_no = parse_horse_no(row.get("horse_no"))
        if horse_no is None:
            continue
        low = _safe_float(row.get("odds_low"), 0.0)
        high = _safe_float(row.get("odds_high"), 0.0)
        mid = _safe_float(row.get("odds_mid"), 0.0)
        if low > 0 and high > 0:
            out[str(horse_no)] = {"low": min(low, high), "high": max(low, high), "mid": mid}
        elif low > 0 or mid > 0:
            out[str(horse_no)] = mid if mid > 0 else low
    return out


def _build_pair_odds_payload(path: Path) -> Dict[Tuple[str, str], object]:
    out = {}
    if not path.exists():
        return out
    df = _read_csv(path)
    if "horse_no_a" not in df.columns or "horse_no_b" not in df.columns:
        return out
    for _, row in df.iterrows():
        a = parse_horse_no(row.get("horse_no_a"))
        b = parse_horse_no(row.get("horse_no_b"))
        if a is None or b is None or int(a) == int(b):
            continue
        key = _canonical_pair(int(a), int(b))
        low = _safe_float(row.get("odds_low"), 0.0)
        high = _safe_float(row.get("odds_high"), 0.0)
        mid = _safe_float(row.get("odds_mid"), 0.0)
        single = _safe_float(row.get("odds"), 0.0)
        if low > 0 and high > 0:
            out[key] = {"low": min(low, high), "high": max(low, high), "mid": mid}
        elif low > 0 or mid > 0:
            out[key] = mid if mid > 0 else low
        elif single > 0:
            out[key] = single
    return out


def _build_inputs(pred_path: Path, odds_path: Path, fuku_path: Path, wide_path: Path, quinella_path: Path):
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
    merged["horse_key"] = merged.get("horse_no").apply(lambda x: str(parse_horse_no(x)) if parse_horse_no(x) is not None else "")
    merged["Top3Prob_model"] = pd.to_numeric(merged.get(prob_col), errors="coerce").fillna(0.0)
    merged["rank_score"] = pd.to_numeric(merged.get("rank_score"), errors="coerce").fillna(merged["Top3Prob_model"])
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


def _ticket_profit(
    bet_type: str,
    horse_names: List[str],
    horse_nos: List[int],
    amount_yen: int,
    actual_top3: List[str],
    odds_map: Dict,
    wide_odds_map: Dict,
    fuku_odds_map: Dict,
    quinella_odds_map: Dict,
) -> Tuple[int, int]:
    if amount_yen <= 0:
        return 0, 0
    hit = eval_ticket(bet_type, horse_names, actual_top3)
    payout_mult = estimate_payout_multiplier(
        bet_type,
        horse_names,
        odds_map,
        wide_odds_map=wide_odds_map,
        fuku_odds_map=fuku_odds_map,
        quinella_odds_map=quinella_odds_map,
        horse_nos=horse_nos,
    )
    payout = int(round(amount_yen * payout_mult)) if hit else 0
    return int(amount_yen), int(payout)


def _odds_bucket(odds_used: float) -> str:
    odd = _safe_float(odds_used, 0.0)
    if odd < 3.0:
        return "low"
    if odd < 10.0:
        return "mid"
    return "high"


def _collect_bucket_counts(items: List[Dict]) -> Dict[str, int]:
    counts = {"bucket_low": 0, "bucket_mid": 0, "bucket_high": 0}
    for item in items:
        bet_type = str(item.get("bet_type", "")).strip().lower()
        if bet_type not in BET_TYPES:
            continue
        stake = _safe_int(item.get("stake_yen", item.get("amount_yen", 0)), 0)
        if stake <= 0:
            continue
        odds_used = _safe_float(item.get("odds_used", 0.0), 0.0)
        if odds_used <= 0:
            continue
        bucket = _odds_bucket(odds_used)
        counts[f"bucket_{bucket}"] += 1
    return counts


def _evaluate_items(
    items: List[Dict],
    no_to_name: Dict[int, str],
    actual_top3: List[str],
    odds_map: Dict,
    wide_odds_map: Dict,
    fuku_odds_map: Dict,
    quinella_odds_map: Dict,
) -> Tuple[int, int, int]:
    stake_sum = payout_sum = tickets = 0
    for item in items:
        bet_type = str(item.get("bet_type", "")).strip().lower()
        if bet_type not in BET_TYPES:
            continue
        stake = _safe_int(item.get("stake_yen", item.get("amount_yen", 0)), 0)
        if stake <= 0:
            continue
        horses = item.get("horses", ())
        if isinstance(horses, str):
            horses = tuple([x for x in horses.split("-") if x])
        horse_nos = []
        horse_names = []
        for hk in horses:
            no = parse_horse_no(hk)
            if no is None:
                continue
            horse_nos.append(int(no))
            nm = no_to_name.get(int(no), "")
            if nm:
                horse_names.append(nm)
        if not horse_names:
            continue
        st, pay = _ticket_profit(
            bet_type,
            horse_names,
            horse_nos,
            stake,
            actual_top3,
            odds_map,
            wide_odds_map,
            fuku_odds_map,
            quinella_odds_map,
        )
        stake_sum += st
        payout_sum += pay
        tickets += 1
    return int(stake_sum), int(payout_sum), int(tickets)


def _normalize_engine_items(engine: str, raw_items) -> List[Dict]:
    eng = str(engine or "").strip().lower()
    out: List[Dict] = []
    if eng == "v2":
        for item in raw_items.items:
            out.append(
                {
                    "bet_type": item.bet_type,
                    "horses": item.horses,
                    "stake_yen": item.stake_yen,
                    "odds_used": item.odds_used,
                }
            )
        return out

    for item in (raw_items or []):
        if not isinstance(item, dict):
            continue
        bet_type = str(item.get("bet_type", "") or item.get("ticket_type", "")).strip().lower()
        horses = item.get("horses", item.get("horse_ids", ()))
        if isinstance(horses, str):
            horses = tuple([x for x in str(horses).split("-") if x])
        horses = tuple([str(x) for x in (horses or ()) if str(x).strip() != ""])
        out.append(
            {
                "bet_type": bet_type,
                "horses": horses,
                "stake_yen": _safe_int(item.get("stake_yen", item.get("stake", 0)), 0),
                "odds_used": _safe_float(item.get("odds_used", item.get("odds", 0.0)), 0.0),
            }
        )
    return out


def _generate_items_by_engine(engine: str, pred_df: pd.DataFrame, odds_payload: Dict, bankroll: int, scope_key: str, cfg: Dict):
    eng = str(engine or "").strip().lower()
    if eng == "v3":
        items, _, _ = generate_bet_plan_v3(
            pred_df=pred_df,
            odds=odds_payload,
            bankroll_yen=bankroll,
            scope_key=scope_key,
            config=cfg,
        )
        return _normalize_engine_items(eng, items)
    if eng == "v4":
        items, _, _ = generate_bet_plan_v4(
            pred_df=pred_df,
            odds=odds_payload,
            bankroll_yen=bankroll,
            scope_key=scope_key,
            config=cfg,
        )
        return _normalize_engine_items(eng, items)
    if eng == "v5":
        items, _, _ = generate_bet_plan_v5(
            pred_df=pred_df,
            odds=odds_payload,
            bankroll_yen=bankroll,
            scope_key=scope_key,
            config=cfg,
        )
        return _normalize_engine_items(eng, items)

    result = generate_bet_plan_v2(
        pred_df=pred_df,
        odds=odds_payload,
        bankroll_yen=bankroll,
        scope_key=scope_key,
        config=cfg,
    )
    return _normalize_engine_items("v2", result)


def _build_run_rows(scope_key: str) -> List[Dict]:
    scope_dir = get_data_dir(BASE_DIR, scope_key)
    pred_results_path = scope_dir / "predictor_results.csv"
    if not pred_results_path.exists():
        return []
    pred_results = _read_csv(pred_results_path)
    if pred_results.empty or "run_id" not in pred_results.columns:
        return []
    pred_results["run_id"] = pred_results["run_id"].astype(str).str.strip()
    pred_results = pred_results[pred_results["run_id"].str.match(r"^\d{8}_\d{6}$", na=False)]
    pred_results = pred_results.drop_duplicates(subset=["run_id"], keep="last")

    pred_map = _build_run_file_map(scope_dir, "predictions_")
    odds_map = _build_run_file_map(scope_dir, "odds_")
    wide_map = _build_run_file_map(scope_dir, "wide_odds_")
    fuku_map = _build_run_file_map(scope_dir, "fuku_odds_")
    quinella_map = _build_run_file_map(scope_dir, "quinella_odds_")

    old_results_map = {}
    result_path = scope_dir / "results.csv"
    if result_path.exists():
        rdf = _read_csv(result_path)
        if not rdf.empty and "run_id" in rdf.columns:
            rdf["run_id"] = rdf["run_id"].astype(str).str.strip()
            rdf["base_amount"] = pd.to_numeric(rdf.get("base_amount"), errors="coerce").fillna(0).astype(int)
            rdf["profit_yen"] = pd.to_numeric(rdf.get("profit_yen"), errors="coerce").fillna(0).astype(int)
            for run_id, g in rdf.groupby("run_id"):
                old_results_map[run_id] = {
                    "base_amount": int(g["base_amount"].sum()),
                    "profit_yen": int(g["profit_yen"].sum()),
                }

    out = []
    for _, row in pred_results.iterrows():
        run_id = str(row.get("run_id", "")).strip()
        if run_id not in old_results_map:
            continue
        pred_path = Path(str(row.get("predictions_path", "")).strip())
        if not pred_path.exists():
            pred_path = pred_map.get(run_id, Path(""))
        odds_path = odds_map.get(run_id, Path(""))
        if not (pred_path.exists() and odds_path.exists()):
            continue

        actual = [_norm_name(row.get("actual_top1")), _norm_name(row.get("actual_top2")), _norm_name(row.get("actual_top3"))]
        actual = [x for x in actual if x]
        if len(actual) < 3:
            continue
        out.append(
            {
                "run_id": run_id,
                "pred_path": pred_path,
                "odds_path": odds_path,
                "wide_path": wide_map.get(run_id, Path("")),
                "fuku_path": fuku_map.get(run_id, Path("")),
                "quinella_path": quinella_map.get(run_id, Path("")),
                "actual_top3": actual[:3],
                "old_base_amount": int(old_results_map[run_id]["base_amount"]),
                "old_profit": int(old_results_map[run_id]["profit_yen"]),
            }
        )
    out.sort(key=lambda x: x["run_id"])
    return out


def _calc_metrics(rows: List[Dict], bankroll: int) -> Dict:
    if not rows:
        return {
            "runs": 0,
            "stake_total": 0,
            "profit_total": 0,
            "roi_stake": float("nan"),
            "roi_cash_as_1": float("nan"),
            "max_drawdown": 0,
            "no_bet_rate": float("nan"),
            "avg_tickets": float("nan"),
            "bucket_low_ratio": float("nan"),
            "bucket_mid_ratio": float("nan"),
            "bucket_high_ratio": float("nan"),
        }
    stake_total = int(sum(int(r["stake"]) for r in rows))
    profit_total = int(sum(int(r["profit"]) for r in rows))
    roi_stake = (stake_total + profit_total) / stake_total if stake_total > 0 else float("nan")
    bankroll_total = int(len(rows) * bankroll)
    roi_cash = (bankroll_total + profit_total) / bankroll_total if bankroll_total > 0 else float("nan")
    cum = np.cumsum([int(r["profit"]) for r in rows])
    peak = np.maximum.accumulate(cum) if len(cum) else np.array([0])
    dd = cum - peak
    max_dd = int(dd.min()) if len(dd) else 0
    no_bet_rate = float(np.mean([1.0 if int(r["stake"]) <= 0 else 0.0 for r in rows]))
    avg_tickets = float(np.mean([int(r["tickets"]) for r in rows]))
    low_cnt = int(sum(int(r.get("bucket_low", 0)) for r in rows))
    mid_cnt = int(sum(int(r.get("bucket_mid", 0)) for r in rows))
    high_cnt = int(sum(int(r.get("bucket_high", 0)) for r in rows))
    bucket_total = int(low_cnt + mid_cnt + high_cnt)
    if bucket_total > 0:
        low_ratio = float(low_cnt) / float(bucket_total)
        mid_ratio = float(mid_cnt) / float(bucket_total)
        high_ratio = float(high_cnt) / float(bucket_total)
    else:
        low_ratio = float("nan")
        mid_ratio = float("nan")
        high_ratio = float("nan")
    return {
        "runs": int(len(rows)),
        "stake_total": stake_total,
        "profit_total": profit_total,
        "roi_stake": float(roi_stake),
        "roi_cash_as_1": float(roi_cash),
        "max_drawdown": max_dd,
        "no_bet_rate": no_bet_rate,
        "avg_tickets": avg_tickets,
        "bucket_low_ratio": low_ratio,
        "bucket_mid_ratio": mid_ratio,
        "bucket_high_ratio": high_ratio,
    }


def _generate_v2_config(base_cfg: Dict, target_risk_share: float) -> Dict:
    cfg = dict(base_cfg)
    cfg["max_race_share"] = float(target_risk_share)
    return cfg


def _apply_combo_to_v3(base_cfg: Dict, combo: Dict) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["target_risk_share"] = float(combo["target_risk_share"])
    cfg["p_mid_odds_threshold"] = float(combo["p_mid_odds_threshold"])
    cfg["N_rank"] = int(combo["N_rank"])
    cfg["N_value"] = int(combo["N_value"])
    cfg.setdefault("penalty", {})
    cfg["penalty"]["place"] = float(combo["penalty_place"])
    cfg["penalty"]["wide"] = float(combo["penalty_wide"])
    cfg.setdefault("min_ev", {})
    cfg["min_ev"]["win"] = float(combo["min_ev_win"])
    cfg["min_ev"]["place"] = float(combo["min_ev_place"])
    cfg["min_ev"]["wide"] = float(combo["min_ev_wide"])
    cfg["min_ev"]["quinella"] = float(combo["min_ev_quinella"])
    if "min_p_hit_per_ticket" in combo:
        cfg["min_p_hit_per_ticket"] = float(combo["min_p_hit_per_ticket"])
    if "min_p_win_per_ticket" in combo:
        cfg["min_p_win_per_ticket"] = float(combo["min_p_win_per_ticket"])
    if "min_edge_per_ticket" in combo:
        cfg["min_edge_per_ticket"] = float(combo["min_edge_per_ticket"])
    return cfg


def _iter_grid(args) -> Iterable[Dict]:
    grids = {
        "target_risk_share": _parse_grid(args.target_risk_grid, "0.2,0.25"),
        "p_mid_odds_threshold": _parse_grid(args.mid_threshold_grid, "0.15,0.18,0.21"),
        "N_rank": [int(x) for x in _parse_grid(args.n_rank_grid, "10,12")],
        "N_value": [int(x) for x in _parse_grid(args.n_value_grid, "10,12")],
        "penalty_place": _parse_grid(args.penalty_place_grid, "0.02"),
        "penalty_wide": _parse_grid(args.penalty_wide_grid, "0.02,0.04"),
        "min_ev_win": _parse_grid(args.min_ev_win_grid, "0.02,0.03"),
        "min_ev_place": _parse_grid(args.min_ev_place_grid, "0.005,0.01"),
        "min_ev_wide": _parse_grid(args.min_ev_wide_grid, "0.005,0.01"),
        "min_ev_quinella": _parse_grid(args.min_ev_quinella_grid, "0.01,0.02"),
        "min_p_hit_per_ticket": _parse_grid(args.min_p_hit_ticket_grid, "0.12"),
        "min_p_win_per_ticket": _parse_grid(args.min_p_win_ticket_grid, "0.06"),
        "min_edge_per_ticket": _parse_grid(args.min_edge_ticket_grid, "0.03"),
    }
    keys = list(grids.keys())
    vals = [grids[k] for k in keys]
    for prod in itertools.product(*vals):
        yield dict(zip(keys, prod))


def _score(metric: Dict) -> Tuple[float, float, float]:
    # Higher cash/stake ROI better; lower drawdown magnitude better.
    return (
        float(metric.get("roi_cash_as_1", -1e9)),
        float(metric.get("roi_stake", -1e9)),
        -abs(float(metric.get("max_drawdown", 0))),
    )


def run_grid_for_scope(scope_key: str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs = _build_run_rows(scope_key)
    if not runs:
        return pd.DataFrame(), pd.DataFrame()

    cfg = _load_predictor_config(scope_key)
    base_v3 = dict(cfg.get("bet_engine_v3", {})) if isinstance(cfg.get("bet_engine_v3", {}), dict) else {}
    base_v2 = dict(cfg.get("bet_engine_v2", {})) if isinstance(cfg.get("bet_engine_v2", {}), dict) else {}
    bankroll = int(args.bankroll)

    detail_rows = []
    summary_rows = []
    for combo_idx, combo in enumerate(_iter_grid(args), start=1):
        v3_cfg = _apply_combo_to_v3(base_v3, combo)
        v2_cfg = _generate_v2_config(base_v2, target_risk_share=float(combo["target_risk_share"]))
        rows_v3 = []
        rows_v2 = []

        for run in runs:
            try:
                pred_df, odds_payload = _build_inputs(
                    pred_path=run["pred_path"],
                    odds_path=run["odds_path"],
                    fuku_path=run["fuku_path"],
                    wide_path=run["wide_path"],
                    quinella_path=run["quinella_path"],
                )
            except Exception:
                continue
            actual_top3 = run["actual_top3"]
            no_to_name = _build_no_to_name_map(run["odds_path"])
            odds_map = load_odds_map(str(run["odds_path"]))
            wide_odds_map = load_wide_odds_map(str(run["wide_path"]))
            fuku_odds_map = load_fuku_odds_map(str(run["fuku_path"]))
            quinella_odds_map = load_quinella_odds_map(str(run["quinella_path"]))

            v3_items, _, _ = generate_bet_plan_v3(
                pred_df=pred_df,
                odds=odds_payload,
                bankroll_yen=bankroll,
                scope_key=scope_key,
                config=v3_cfg,
            )
            s3, p3, t3 = _evaluate_items(
                v3_items,
                no_to_name,
                actual_top3,
                odds_map,
                wide_odds_map,
                fuku_odds_map,
                quinella_odds_map,
            )
            b3 = _collect_bucket_counts(v3_items)
            rows_v3.append(
                {
                    "run_id": run["run_id"],
                    "stake": s3,
                    "profit": p3 - s3,
                    "tickets": t3,
                    "bucket_low": b3["bucket_low"],
                    "bucket_mid": b3["bucket_mid"],
                    "bucket_high": b3["bucket_high"],
                }
            )

            if args.ab_mode == "fair":
                v2_result = generate_bet_plan_v2(
                    pred_df=pred_df,
                    odds=odds_payload,
                    bankroll_yen=bankroll,
                    scope_key=scope_key,
                    config=v2_cfg,
                )
                v2_items = []
                for item in v2_result.items:
                    v2_items.append(
                        {
                            "bet_type": item.bet_type,
                            "horses": item.horses,
                            "stake_yen": item.stake_yen,
                            "odds_used": item.odds_used,
                        }
                    )
                s2, p2, t2 = _evaluate_items(
                    v2_items,
                    no_to_name,
                    actual_top3,
                    odds_map,
                    wide_odds_map,
                    fuku_odds_map,
                    quinella_odds_map,
                )
                b2 = _collect_bucket_counts(v2_items)
                rows_v2.append(
                    {
                        "run_id": run["run_id"],
                        "stake": s2,
                        "profit": p2 - s2,
                        "tickets": t2,
                        "bucket_low": b2["bucket_low"],
                        "bucket_mid": b2["bucket_mid"],
                        "bucket_high": b2["bucket_high"],
                    }
                )
            else:
                old_base = int(run["old_base_amount"])
                old_profit = int(run["old_profit"])
                if args.bankroll_mode == "fixed":
                    scale = float(bankroll) / float(old_base) if old_base > 0 else 1.0
                    old_profit = int(round(old_profit * scale))
                    old_base = bankroll
                rows_v2.append(
                    {
                        "run_id": run["run_id"],
                        "stake": old_base,
                        "profit": old_profit,
                        "tickets": 0,
                        "bucket_low": 0,
                        "bucket_mid": 0,
                        "bucket_high": 0,
                    }
                )

        if not rows_v3:
            continue
        m_v3 = _calc_metrics(rows_v3, bankroll=bankroll)
        m_v2 = _calc_metrics(rows_v2, bankroll=bankroll)
        delta_cash = float(m_v3["roi_cash_as_1"]) - float(m_v2["roi_cash_as_1"])
        delta_stake = float(m_v3["roi_stake"]) - float(m_v2["roi_stake"])
        summary_rows.append(
            {
                "scope": scope_key,
                "combo_idx": combo_idx,
                **combo,
                "v3_roi_cash_as_1": m_v3["roi_cash_as_1"],
                "v3_roi_stake": m_v3["roi_stake"],
                "v3_max_drawdown": m_v3["max_drawdown"],
                "v3_no_bet_rate": m_v3["no_bet_rate"],
                "v3_avg_tickets": m_v3["avg_tickets"],
                "v3_bucket_low_ratio": m_v3["bucket_low_ratio"],
                "v3_bucket_mid_ratio": m_v3["bucket_mid_ratio"],
                "v3_bucket_high_ratio": m_v3["bucket_high_ratio"],
                "v2_roi_cash_as_1": m_v2["roi_cash_as_1"],
                "v2_roi_stake": m_v2["roi_stake"],
                "v2_max_drawdown": m_v2["max_drawdown"],
                "v2_no_bet_rate": m_v2["no_bet_rate"],
                "v2_avg_tickets": m_v2["avg_tickets"],
                "v2_bucket_low_ratio": m_v2["bucket_low_ratio"],
                "v2_bucket_mid_ratio": m_v2["bucket_mid_ratio"],
                "v2_bucket_high_ratio": m_v2["bucket_high_ratio"],
                "delta_roi_cash_as_1": delta_cash,
                "delta_roi_stake": delta_stake,
                "runs": m_v3["runs"],
            }
        )

        for row in rows_v3:
            detail_rows.append(
                {
                    "scope": scope_key,
                    "combo_idx": combo_idx,
                    "engine": "v3",
                    "run_id": row["run_id"],
                    "stake": row["stake"],
                    "profit": row["profit"],
                    "tickets": row["tickets"],
                    "bucket_low": row.get("bucket_low", 0),
                    "bucket_mid": row.get("bucket_mid", 0),
                    "bucket_high": row.get("bucket_high", 0),
                }
            )
        for row in rows_v2:
            detail_rows.append(
                {
                    "scope": scope_key,
                    "combo_idx": combo_idx,
                    "engine": "baseline",
                    "run_id": row["run_id"],
                    "stake": row["stake"],
                    "profit": row["profit"],
                    "tickets": row["tickets"],
                    "bucket_low": row.get("bucket_low", 0),
                    "bucket_mid": row.get("bucket_mid", 0),
                    "bucket_high": row.get("bucket_high", 0),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def run_engine_compare_for_scope(scope_key: str, args, engine_a: str, engine_b: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs = _build_run_rows(scope_key)
    if not runs:
        return pd.DataFrame(), pd.DataFrame()
    cfg = _load_predictor_config(scope_key)
    cfg_a = dict(cfg.get(f"bet_engine_{engine_a}", {})) if isinstance(cfg.get(f"bet_engine_{engine_a}", {}), dict) else {}
    cfg_b = dict(cfg.get(f"bet_engine_{engine_b}", {})) if isinstance(cfg.get(f"bet_engine_{engine_b}", {}), dict) else {}
    bankroll = int(args.bankroll)

    rows_a = []
    rows_b = []
    detail_rows = []
    for run in runs:
        try:
            pred_df, odds_payload = _build_inputs(
                pred_path=run["pred_path"],
                odds_path=run["odds_path"],
                fuku_path=run["fuku_path"],
                wide_path=run["wide_path"],
                quinella_path=run["quinella_path"],
            )
        except Exception:
            continue

        actual_top3 = run["actual_top3"]
        no_to_name = _build_no_to_name_map(run["odds_path"])
        odds_map = load_odds_map(str(run["odds_path"]))
        wide_odds_map = load_wide_odds_map(str(run["wide_path"]))
        fuku_odds_map = load_fuku_odds_map(str(run["fuku_path"]))
        quinella_odds_map = load_quinella_odds_map(str(run["quinella_path"]))

        items_a = _generate_items_by_engine(engine_a, pred_df, odds_payload, bankroll, scope_key, cfg_a)
        s_a, p_a, t_a = _evaluate_items(
            items_a,
            no_to_name,
            actual_top3,
            odds_map,
            wide_odds_map,
            fuku_odds_map,
            quinella_odds_map,
        )
        b_a = _collect_bucket_counts(items_a)
        rows_a.append(
            {
                "run_id": run["run_id"],
                "stake": s_a,
                "profit": p_a - s_a,
                "tickets": t_a,
                "bucket_low": b_a["bucket_low"],
                "bucket_mid": b_a["bucket_mid"],
                "bucket_high": b_a["bucket_high"],
            }
        )

        items_b = _generate_items_by_engine(engine_b, pred_df, odds_payload, bankroll, scope_key, cfg_b)
        s_b, p_b, t_b = _evaluate_items(
            items_b,
            no_to_name,
            actual_top3,
            odds_map,
            wide_odds_map,
            fuku_odds_map,
            quinella_odds_map,
        )
        b_b = _collect_bucket_counts(items_b)
        rows_b.append(
            {
                "run_id": run["run_id"],
                "stake": s_b,
                "profit": p_b - s_b,
                "tickets": t_b,
                "bucket_low": b_b["bucket_low"],
                "bucket_mid": b_b["bucket_mid"],
                "bucket_high": b_b["bucket_high"],
            }
        )

    if not rows_a or not rows_b:
        return pd.DataFrame(), pd.DataFrame()

    m_a = _calc_metrics(rows_a, bankroll=bankroll)
    m_b = _calc_metrics(rows_b, bankroll=bankroll)
    summary = pd.DataFrame(
        [
            {
                "scope": scope_key,
                "engine_a": engine_a,
                "engine_b": engine_b,
                "runs": m_a["runs"],
                "a_roi_stake": m_a["roi_stake"],
                "a_roi_cash_as_1": m_a["roi_cash_as_1"],
                "a_no_bet_rate": m_a["no_bet_rate"],
                "a_avg_tickets": m_a["avg_tickets"],
                "a_bucket_low_ratio": m_a["bucket_low_ratio"],
                "a_bucket_mid_ratio": m_a["bucket_mid_ratio"],
                "a_bucket_high_ratio": m_a["bucket_high_ratio"],
                "b_roi_stake": m_b["roi_stake"],
                "b_roi_cash_as_1": m_b["roi_cash_as_1"],
                "b_no_bet_rate": m_b["no_bet_rate"],
                "b_avg_tickets": m_b["avg_tickets"],
                "b_bucket_low_ratio": m_b["bucket_low_ratio"],
                "b_bucket_mid_ratio": m_b["bucket_mid_ratio"],
                "b_bucket_high_ratio": m_b["bucket_high_ratio"],
                "delta_roi_stake_b_minus_a": float(m_b["roi_stake"]) - float(m_a["roi_stake"]),
                "delta_roi_cash_as_1_b_minus_a": float(m_b["roi_cash_as_1"]) - float(m_a["roi_cash_as_1"]),
                "delta_bet_runs_b_minus_a": int(sum(1 for r in rows_b if int(r["stake"]) > 0))
                - int(sum(1 for r in rows_a if int(r["stake"]) > 0)),
            }
        ]
    )

    for row in rows_a:
        detail_rows.append({"scope": scope_key, "engine": engine_a, **row})
    for row in rows_b:
        detail_rows.append({"scope": scope_key, "engine": engine_b, **row})
    return summary, pd.DataFrame(detail_rows)


def _render_patch(scope_key: str, combo: Dict) -> str:
    cfg_path = get_predictor_config_path(BASE_DIR, scope_key)
    cfg = _load_predictor_config(scope_key)
    old_txt = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else "{}\n"
    be3 = dict(cfg.get("bet_engine_v3", {})) if isinstance(cfg.get("bet_engine_v3", {}), dict) else {}
    be3["target_risk_share"] = float(combo["target_risk_share"])
    be3["p_mid_odds_threshold"] = float(combo["p_mid_odds_threshold"])
    be3["N_rank"] = int(combo["N_rank"])
    be3["N_value"] = int(combo["N_value"])
    be3.setdefault("penalty", {})
    be3["penalty"]["place"] = float(combo["penalty_place"])
    be3["penalty"]["wide"] = float(combo["penalty_wide"])
    be3.setdefault("min_ev", {})
    be3["min_ev"]["win"] = float(combo["min_ev_win"])
    be3["min_ev"]["place"] = float(combo["min_ev_place"])
    be3["min_ev"]["wide"] = float(combo["min_ev_wide"])
    be3["min_ev"]["quinella"] = float(combo["min_ev_quinella"])
    be3["min_p_hit_per_ticket"] = float(combo["min_p_hit_per_ticket"])
    be3["min_p_win_per_ticket"] = float(combo["min_p_win_per_ticket"])
    be3["min_edge_per_ticket"] = float(combo["min_edge_per_ticket"])
    cfg["bet_engine_v3"] = be3
    new_txt = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"

    import difflib

    diff = difflib.unified_diff(
        old_txt.splitlines(),
        new_txt.splitlines(),
        fromfile=str(cfg_path),
        tofile=str(cfg_path),
        lineterm="",
    )
    return "\n".join(diff) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Replay grid search for bet_engine_v3 with fair A/B.")
    parser.add_argument("--scopes", default="central_dirt,central_turf,local")
    parser.add_argument("--bankroll", type=int, default=2000)
    parser.add_argument("--bankroll-mode", choices=["fixed", "legacy"], default="fixed")
    parser.add_argument("--ab-mode", choices=["fair", "historical"], default="fair")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target-risk-grid", default="0.2,0.25,0.3")
    parser.add_argument("--mid-threshold-grid", default="0.15,0.18,0.21")
    parser.add_argument("--n-rank-grid", default="10,12")
    parser.add_argument("--n-value-grid", default="10,12")
    parser.add_argument("--penalty-place-grid", default="0.02")
    parser.add_argument("--penalty-wide-grid", default="0.02,0.04")
    parser.add_argument("--min-ev-win-grid", default="0.02,0.03")
    parser.add_argument("--min-ev-place-grid", default="0.005,0.01")
    parser.add_argument("--min-ev-wide-grid", default="0.005,0.01")
    parser.add_argument("--min-ev-quinella-grid", default="0.01,0.02")
    parser.add_argument("--min-p-hit-ticket-grid", default="0.12")
    parser.add_argument("--min-p-win-ticket-grid", default="0.06")
    parser.add_argument("--min-edge-ticket-grid", default="0.03")
    parser.add_argument("--engine-a", choices=ENGINE_CHOICES, default="", help="engine compare mode: engine A")
    parser.add_argument("--engine-b", choices=ENGINE_CHOICES, default="", help="engine compare mode: engine B")
    args = parser.parse_args()

    scopes = []
    for token in str(args.scopes or "").split(","):
        key = normalize_scope_key(token.strip())
        if key and key not in scopes:
            scopes.append(key)
    if not scopes:
        scopes = ["central_dirt"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_mode = bool(str(args.engine_a).strip()) and bool(str(args.engine_b).strip())
    out_dir = BASE_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if compare_mode:
        engine_a = str(args.engine_a).strip().lower()
        engine_b = str(args.engine_b).strip().lower()
        all_summary = []
        all_detail = []
        for scope in scopes:
            summary_df, detail_df = run_engine_compare_for_scope(scope, args, engine_a=engine_a, engine_b=engine_b)
            if not summary_df.empty:
                all_summary.append(summary_df)
            if not detail_df.empty:
                all_detail.append(detail_df)

        summary_out = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
        detail_out = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
        summary_path = out_dir / f"replay_engine_compare_summary_{engine_a}_vs_{engine_b}_{ts}.csv"
        detail_path = out_dir / f"replay_engine_compare_detail_{engine_a}_vs_{engine_b}_{ts}.csv"
        summary_out.to_csv(summary_path, index=False, encoding="utf-8-sig")
        detail_out.to_csv(detail_path, index=False, encoding="utf-8-sig")
        if summary_out.empty:
            print("No comparable runs.")
        else:
            print(summary_out.to_string(index=False))
        print(f"\nsaved summary: {summary_path}")
        print(f"saved detail : {detail_path}")
        return

    all_summary = []
    all_detail = []
    for scope in scopes:
        summary_df, detail_df = run_grid_for_scope(scope, args)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(
                by=["delta_roi_cash_as_1", "delta_roi_stake", "v3_max_drawdown"],
                ascending=[False, False, True],
            )
            summary_df["rank"] = np.arange(1, len(summary_df) + 1)
            all_summary.append(summary_df)
        if not detail_df.empty:
            all_detail.append(detail_df)

    summary_path = out_dir / f"replay_grid_search_bet_params_summary_{ts}.csv"
    detail_path = out_dir / f"replay_grid_search_bet_params_detail_{ts}.csv"
    patch_path = out_dir / f"replay_grid_search_bet_params_patch_{ts}.diff"

    summary_out = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    detail_out = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    summary_out.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_out.to_csv(detail_path, index=False, encoding="utf-8-sig")

    patch_chunks = []
    if not summary_out.empty:
        print("Top combinations:")
        for scope in scopes:
            s = summary_out[summary_out["scope"] == scope].head(max(1, int(args.top_k)))
            if s.empty:
                continue
            print(f"\n[{scope}]")
            print(
                s[
                    [
                        "combo_idx",
                        "delta_roi_cash_as_1",
                        "delta_roi_stake",
                        "v3_roi_cash_as_1",
                        "v2_roi_cash_as_1",
                        "v3_max_drawdown",
                        "v3_no_bet_rate",
                        "v3_avg_tickets",
                        "v3_bucket_low_ratio",
                        "v3_bucket_mid_ratio",
                        "v3_bucket_high_ratio",
                    ]
                ].to_string(index=False)
            )
            best = s.iloc[0].to_dict()
            patch_chunks.append(_render_patch(scope, best))
    patch_path.write_text("\n".join(patch_chunks), encoding="utf-8")

    print(f"\nsaved summary: {summary_path}")
    print(f"saved detail : {detail_path}")
    print(f"saved patch  : {patch_path}")


if __name__ == "__main__":
    main()

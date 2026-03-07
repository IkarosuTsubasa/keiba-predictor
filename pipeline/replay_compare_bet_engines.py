import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from bet_engine_v2 import generate_bet_plan_v2
from bet_engine_v3 import generate_bet_plan_v3
from bet_engine_v4 import generate_bet_plan_v4
from bet_engine_v5 import generate_bet_plan_v5
from bet_engine_v6 import generate_bet_plan_v6
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
DEFAULT_SCOPES = ["central_dirt", "central_turf", "local"]
DEFAULT_BUDGET = 0
RUN_ID_RE = re.compile(r"(\d{8}_\d{6})")
ENGINE_CHOICES = ("v2", "v3", "v4", "v5", "v6")


@dataclass
class EvalResult:
    stake: int
    payout: int
    profit: int
    tickets: int
    avg_value_signal_selected: float = float("nan")
    max_value_signal_selected: float = float("nan")
    avg_value_min_eff: float = float("nan")
    dropped_by_value_gate: int = 0
    dropped_by_value_gate_win: int = 0
    best_value_ratio: float = float("nan")
    best_value_ratio_win: float = float("nan")
    value_gate_enabled_by_type: str = ""


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


def _mean_finite(values: List[float]) -> float:
    kept = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not kept:
        return float("nan")
    return float(sum(kept) / float(len(kept)))


def _parse_scopes(text: str) -> List[str]:
    out = []
    for token in str(text or "").split(","):
        key = normalize_scope_key(token.strip())
        if key and key not in out:
            out.append(key)
    return out or list(DEFAULT_SCOPES)


def _extract_run_id(text: str) -> str:
    m = RUN_ID_RE.search(str(text or ""))
    return m.group(1) if m else ""


def _pick_prob_column(df: pd.DataFrame) -> str:
    for col in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if col in df.columns:
            return col
    return ""


def _roi(stake: int, profit: int) -> float:
    if stake <= 0:
        return float("nan")
    return (float(stake) + float(profit)) / float(stake)


def _canonical_pair(a: int, b: int) -> Tuple[str, str]:
    x = int(a)
    y = int(b)
    if x > y:
        x, y = y, x
    return str(x), str(y)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")


def _load_bet_cfg(scope_key: str, engine: str) -> Dict:
    path = get_predictor_config_path(BASE_DIR, scope_key)
    if not path.exists():
        return {}
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        key = f"bet_engine_{engine}"
        node = cfg.get(key, {})
        out = dict(node) if isinstance(node, dict) else {}
        if str(engine) == "v3":
            calibration = cfg.get("calibration", {})
            if isinstance(calibration, dict):
                out["win_temp"] = float(calibration.get("win_temp", out.get("win_temp", 1.0)))
                out["calibration_enabled"] = bool(calibration.get("enabled", True))
        return out
    except Exception:
        return {}


def _load_predictor_rows(scope_dir: Path) -> pd.DataFrame:
    path = scope_dir / "predictor_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read_csv(path)
    if df.empty or "run_id" not in df.columns:
        return pd.DataFrame()
    df["run_id"] = df["run_id"].astype(str).str.strip()
    df = df[df["run_id"].str.match(r"^\d{8}_\d{6}$", na=False)].copy()
    if "predictions_path" not in df.columns:
        df["predictions_path"] = ""
    for col in ("actual_top1", "actual_top2", "actual_top3"):
        if col not in df.columns:
            df[col] = ""
    df = df.drop_duplicates(subset=["run_id"], keep="last").reset_index(drop=True)
    return df


def _load_old_result_map(scope_dir: Path, budget_override: int) -> Dict[str, EvalResult]:
    path = scope_dir / "results.csv"
    if not path.exists():
        return {}
    df = _read_csv(path)
    if df.empty or "run_id" not in df.columns:
        return {}
    df["run_id"] = df["run_id"].astype(str).str.strip()
    df["base_amount"] = pd.to_numeric(df.get("base_amount"), errors="coerce").fillna(0).astype(int)
    df["profit_yen"] = pd.to_numeric(df.get("profit_yen"), errors="coerce").fillna(0).astype(int)
    out = {}
    for run_id, group in df.groupby("run_id"):
        stake = int(group["base_amount"].sum())
        profit = int(group["profit_yen"].sum())
        if int(budget_override) > 0:
            stake = int(budget_override)
        payout = stake + profit
        tickets = 0
        out[run_id] = EvalResult(stake=stake, payout=payout, profit=payout - stake, tickets=tickets)
    return out


def _build_run_file_map(scope_dir: Path, prefix: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in scope_dir.rglob(f"{prefix}*.csv"):
        rid = _extract_run_id(p.name)
        if not rid:
            rid = _extract_run_id(str(p))
        if not rid:
            continue
        prev = out.get(rid)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            out[rid] = p
    return out


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
        name = normalize_name(row.get("name"))
        if not name:
            continue
        out[int(no)] = name
    return out


def _build_place_odds_payload(path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not path.exists():
        return out
    df = _read_csv(path)
    if "horse_no" not in df.columns:
        return out
    for _, row in df.iterrows():
        horse_no = parse_horse_no(row.get("horse_no"))
        if horse_no is None:
            continue
        key = str(horse_no)
        low = _safe_float(row.get("odds_low"), 0.0)
        high = _safe_float(row.get("odds_high"), 0.0)
        mid = _safe_float(row.get("odds_mid"), 0.0)
        if low > 0 and high > 0:
            out[key] = (min(low, high), max(low, high))
        elif low > 0:
            out[key] = low
        elif mid > 0:
            out[key] = mid
    return out


def _build_pair_odds_payload(path: Path) -> Dict[Tuple[str, str], object]:
    out: Dict[Tuple[str, str], object] = {}
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
            out[key] = (min(low, high), max(low, high))
        elif low > 0:
            out[key] = low
        elif mid > 0:
            out[key] = mid
        elif single > 0:
            out[key] = single
    return out


def _build_v2_inputs(pred_path: Path, odds_path: Path, fuku_path: Path, wide_path: Path, quinella_path: Path):
    pred = _read_csv(pred_path)
    odds = _read_csv(odds_path)
    if "HorseName" not in pred.columns or "name" not in odds.columns:
        raise ValueError("missing HorseName or odds name")

    pred = pred.copy()
    odds = odds.copy()
    pred["name_key"] = pred["HorseName"].apply(normalize_name)
    odds["name_key"] = odds["name"].apply(normalize_name)
    odds = odds.drop_duplicates(subset=["name_key"], keep="last")
    merged = pred.merge(odds, on="name_key", how="left")

    prob_col = _pick_prob_column(merged)
    if not prob_col:
        raise ValueError("missing Top3Prob column")

    merged["horse_key"] = merged.get("horse_no").apply(
        lambda x: str(parse_horse_no(x)) if parse_horse_no(x) is not None else ""
    )
    merged["Top3Prob_model"] = pd.to_numeric(merged.get(prob_col), errors="coerce").fillna(0.0)
    merged["rank_score"] = pd.to_numeric(merged.get("rank_score"), errors="coerce")
    merged["rank_score"] = merged["rank_score"].fillna(merged["Top3Prob_model"])
    merged["odds_num"] = pd.to_numeric(merged.get("odds"), errors="coerce")
    merged = merged[(merged["horse_key"] != "") & merged["Top3Prob_model"].notna()].copy()
    if merged.empty:
        raise ValueError("empty merged after horse_key filter")

    pred_df = merged[["horse_key", "rank_score", "Top3Prob_model"]].copy()
    pred_df["race_id"] = "__single_race__"
    win_map = {}
    for _, row in merged.iterrows():
        key = str(row.get("horse_key", "")).strip()
        val = _safe_float(row.get("odds_num"), 0.0)
        if key and val > 0:
            win_map[key] = val
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
    odds_map: Dict[str, float],
    wide_odds_map: Dict[Tuple[int, int], float],
    fuku_odds_map: Dict[int, float],
    quinella_odds_map: Dict[Tuple[int, int], float],
) -> Tuple[int, int]:
    if amount_yen <= 0 or not horse_names:
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


def _normalize_generated_items(engine: str, generated_items) -> List[Dict]:
    out = []
    eng = str(engine or "").strip().lower()
    if eng == "v2":
        for item in generated_items.items:
            out.append(
                {
                    "bet_type": str(item.bet_type),
                    "horses": tuple(item.horses),
                    "stake_yen": int(item.stake_yen),
                }
            )
        return out

    for item in (generated_items or []):
        if not isinstance(item, dict):
            continue
        bet_type = str(item.get("bet_type", "") or item.get("ticket_type", "")).strip().lower()
        horses = item.get("horses", item.get("horse_ids", ()))
        if isinstance(horses, str):
            horses = tuple([x for x in str(horses).split("-") if x])
        horses = tuple([str(x) for x in (horses or ()) if str(x).strip() != ""])
        stake = int(float(item.get("stake_yen", item.get("stake", 0)) or 0))
        if not bet_type or (not horses):
            continue
        out.append({"bet_type": bet_type, "horses": horses, "stake_yen": stake})
    return out


def _evaluate_new_engine(
    engine: str,
    pred_df: pd.DataFrame,
    odds_payload: Dict,
    bet_cfg: Dict,
    budget_yen: int,
    actual_top3: List[str],
    no_to_name: Dict[int, str],
    odds_map: Dict[str, float],
    wide_odds_map: Dict[Tuple[int, int], float],
    fuku_odds_map: Dict[int, float],
    quinella_odds_map: Dict[Tuple[int, int], float],
    scope_key: str,
) -> EvalResult:
    eng = str(engine or "v2").strip().lower()
    summary = {}
    if eng == "v3":
        generated_items, _, summary = generate_bet_plan_v3(
            pred_df=pred_df,
            odds=odds_payload,
            bankroll_yen=int(budget_yen),
            scope_key=scope_key,
            config=bet_cfg,
        )
    elif eng == "v4":
        generated_items, _, summary = generate_bet_plan_v4(
            pred_df=pred_df,
            odds=odds_payload,
            bankroll_yen=int(budget_yen),
            scope_key=scope_key,
            config=bet_cfg,
        )
    elif eng == "v5":
        generated_items, _, summary = generate_bet_plan_v5(
            pred_df=pred_df,
            odds=odds_payload,
            bankroll_yen=int(budget_yen),
            scope_key=scope_key,
            config=bet_cfg,
        )
    elif eng == "v6":
        generated_items, _, summary = generate_bet_plan_v6(
            pred_df=pred_df,
            odds=odds_payload,
            bankroll_yen=int(budget_yen),
            scope_key=scope_key,
            config=bet_cfg,
        )
    else:
        generated_items = generate_bet_plan_v2(
            pred_df=pred_df,
            odds=odds_payload,
            bankroll_yen=int(budget_yen),
            scope_key=scope_key,
            config=bet_cfg,
        )

    normalized_items = _normalize_generated_items(eng, generated_items)
    stake_sum = 0
    payout_sum = 0
    ticket_count = 0
    for item in normalized_items:
        bet_type = str(item.get("bet_type", "")).strip().lower()
        if bet_type not in ("win", "place", "wide", "quinella"):
            continue
        stake = int(item.get("stake_yen", 0))
        if stake <= 0:
            continue
        horse_nos = []
        horse_names = []
        for hk in item.get("horses", ()):
            no = parse_horse_no(hk)
            if no is None:
                continue
            horse_nos.append(int(no))
            name = no_to_name.get(int(no), "")
            if name:
                horse_names.append(name)
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
        ticket_count += 1
    avg_value_signal_selected = float("nan")
    max_value_signal_selected = float("nan")
    avg_value_min_eff = float("nan")
    dropped_by_value_gate = 0
    dropped_by_value_gate_win = 0
    best_value_ratio = float("nan")
    best_value_ratio_win = float("nan")
    value_gate_enabled_by_type = ""
    if eng == "v5" and isinstance(summary, dict):
        race_diags = summary.get("race_diagnostics", summary.get("diagnostics", []))
        avg_vals = []
        max_vals = []
        min_eff_vals = []
        best_ratio_vals = []
        best_ratio_win_vals = []
        gate_by_type_seen = None
        for diag in (race_diags or []):
            if not isinstance(diag, dict):
                continue
            if gate_by_type_seen is None and isinstance(diag.get("value_gate_enabled_by_type"), dict):
                gate_by_type_seen = dict(diag.get("value_gate_enabled_by_type"))
            v_avg = _safe_float(
                diag.get("avg_value_ratio_selected", diag.get("avg_value_signal_selected")),
                float("nan"),
            )
            if math.isfinite(v_avg):
                avg_vals.append(v_avg)
            v_max = _safe_float(
                diag.get("max_value_ratio_selected", diag.get("max_value_signal_selected")),
                float("nan"),
            )
            if math.isfinite(v_max):
                max_vals.append(v_max)
            v_best = _safe_float(diag.get("best_value_ratio"), float("nan"))
            if math.isfinite(v_best):
                best_ratio_vals.append(v_best)
            v_best_win = _safe_float(diag.get("best_value_ratio_win", diag.get("best_value_ratio")), float("nan"))
            if math.isfinite(v_best_win):
                best_ratio_win_vals.append(v_best_win)
            if "avg_value_min_eff" in diag:
                v_min_eff = _safe_float(diag.get("avg_value_min_eff"), float("nan"))
                if math.isfinite(v_min_eff):
                    min_eff_vals.append(v_min_eff)
            else:
                vm = diag.get("value_min_eff_by_type", {})
                if isinstance(vm, dict):
                    parts = []
                    for key in ("win", "place", "pair"):
                        vv = _safe_float(vm.get(key), float("nan"))
                        if math.isfinite(vv):
                            parts.append(vv)
                    if parts:
                        min_eff_vals.append(float(sum(parts) / len(parts)))
            dropped_by_value_gate += _safe_int(diag.get("dropped_by_value_gate_count"), 0)
            dropped_by_value_gate_win += _safe_int(
                diag.get("dropped_by_value_gate_win_count", diag.get("dropped_by_value_gate_count", 0)),
                0,
            )
        avg_value_signal_selected = _mean_finite(avg_vals)
        max_value_signal_selected = _mean_finite(max_vals)
        avg_value_min_eff = _mean_finite(min_eff_vals)
        best_value_ratio = _mean_finite(best_ratio_vals)
        best_value_ratio_win = _mean_finite(best_ratio_win_vals)
        if isinstance(gate_by_type_seen, dict):
            try:
                value_gate_enabled_by_type = json.dumps(gate_by_type_seen, ensure_ascii=False, sort_keys=True)
            except Exception:
                value_gate_enabled_by_type = str(gate_by_type_seen)

    return EvalResult(
        stake_sum,
        payout_sum,
        payout_sum - stake_sum,
        ticket_count,
        avg_value_signal_selected=avg_value_signal_selected,
        max_value_signal_selected=max_value_signal_selected,
        avg_value_min_eff=avg_value_min_eff,
        dropped_by_value_gate=int(dropped_by_value_gate),
        dropped_by_value_gate_win=int(dropped_by_value_gate_win),
        best_value_ratio=best_value_ratio,
        best_value_ratio_win=best_value_ratio_win,
        value_gate_enabled_by_type=value_gate_enabled_by_type,
    )


def compare_scope(scope_key: str, budget_yen: int, engine: str) -> Tuple[pd.DataFrame, Dict]:
    scope_dir = get_data_dir(BASE_DIR, scope_key)
    pred_rows = _load_predictor_rows(scope_dir)
    old_map = _load_old_result_map(scope_dir, budget_yen)
    bet_cfg = _load_bet_cfg(scope_key, engine=engine)

    if pred_rows.empty or not old_map:
        empty = pd.DataFrame()
        return empty, {
            "engine": engine,
            "scope": scope_key,
            "runs": 0,
            "old_stake": 0,
            "old_profit": 0,
            "old_roi": float("nan"),
            "new_stake": 0,
            "new_profit": 0,
            "new_roi": float("nan"),
            "delta_profit": 0,
            "delta_roi": float("nan"),
            "new_better_runs": 0,
            "old_better_runs": 0,
            "same_runs": 0,
            "avg_value_signal_selected_overall": float("nan"),
            "avg_max_value_signal_selected_overall": float("nan"),
            "avg_value_min_eff_overall": float("nan"),
            "total_dropped_by_value_gate": 0,
            "avg_best_value_ratio_overall": float("nan"),
            "p50_best_value_ratio_overall": float("nan"),
            "avg_best_value_ratio_win_overall": float("nan"),
            "p50_best_value_ratio_win_overall": float("nan"),
            "total_dropped_by_value_gate_win": 0,
            "value_gate_enabled_by_type": "",
        }

    pred_map = _build_run_file_map(scope_dir, "predictions_")
    odds_map_by_run = _build_run_file_map(scope_dir, "odds_")
    wide_map_by_run = _build_run_file_map(scope_dir, "wide_odds_")
    fuku_map_by_run = _build_run_file_map(scope_dir, "fuku_odds_")
    quinella_map_by_run = _build_run_file_map(scope_dir, "quinella_odds_")

    rows = []
    for _, row in pred_rows.iterrows():
        run_id = str(row.get("run_id", "")).strip()
        if run_id not in old_map:
            continue

        pred_path = Path(str(row.get("predictions_path", "") or "").strip())
        if not pred_path.exists():
            pred_path = pred_map.get(run_id, Path(""))
        odds_path = odds_map_by_run.get(run_id, Path(""))
        if not (pred_path.exists() and odds_path.exists()):
            continue

        wide_path = wide_map_by_run.get(run_id, Path(""))
        fuku_path = fuku_map_by_run.get(run_id, Path(""))
        quinella_path = quinella_map_by_run.get(run_id, Path(""))
        actual_top3 = [
            normalize_name(row.get("actual_top1", "")),
            normalize_name(row.get("actual_top2", "")),
            normalize_name(row.get("actual_top3", "")),
        ]
        actual_top3 = [x for x in actual_top3 if x]
        if len(actual_top3) < 3:
            continue

        try:
            pred_df, odds_payload = _build_v2_inputs(pred_path, odds_path, fuku_path, wide_path, quinella_path)
        except Exception:
            continue

        no_to_name = _build_no_to_name_map(odds_path)
        odds_map = load_odds_map(str(odds_path))
        wide_odds_map = load_wide_odds_map(str(wide_path))
        fuku_odds_map = load_fuku_odds_map(str(fuku_path))
        quinella_odds_map = load_quinella_odds_map(str(quinella_path))

        old_eval = old_map[run_id]
        bankroll = int(old_eval.stake if budget_yen <= 0 else budget_yen)
        new_eval = _evaluate_new_engine(
            engine=engine,
            pred_df=pred_df,
            odds_payload=odds_payload,
            bet_cfg=bet_cfg,
            budget_yen=bankroll,
            actual_top3=actual_top3,
            no_to_name=no_to_name,
            odds_map=odds_map,
            wide_odds_map=wide_odds_map,
            fuku_odds_map=fuku_odds_map,
            quinella_odds_map=quinella_odds_map,
            scope_key=scope_key,
        )
        rows.append(
            {
                "scope": scope_key,
                "run_id": run_id,
                "old_stake": old_eval.stake,
                "old_payout": old_eval.payout,
                "old_profit": old_eval.profit,
                "old_roi": _roi(old_eval.stake, old_eval.profit),
                "old_tickets": old_eval.tickets,
                "new_stake": new_eval.stake,
                "new_payout": new_eval.payout,
                "new_profit": new_eval.profit,
                "new_roi": _roi(new_eval.stake, new_eval.profit),
                "new_tickets": new_eval.tickets,
                "avg_value_signal_selected": new_eval.avg_value_signal_selected,
                "max_value_signal_selected": new_eval.max_value_signal_selected,
                "avg_value_min_eff": new_eval.avg_value_min_eff,
                "dropped_by_value_gate": int(new_eval.dropped_by_value_gate),
                "dropped_by_value_gate_win": int(new_eval.dropped_by_value_gate_win),
                "best_value_ratio": new_eval.best_value_ratio,
                "best_value_ratio_win": new_eval.best_value_ratio_win,
                "value_gate_enabled_by_type": str(new_eval.value_gate_enabled_by_type or ""),
                "delta_profit": new_eval.profit - old_eval.profit,
                "delta_stake": new_eval.stake - old_eval.stake,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {
            "engine": engine,
            "scope": scope_key,
            "runs": 0,
            "old_stake": 0,
            "old_profit": 0,
            "old_roi": float("nan"),
            "new_stake": 0,
            "new_profit": 0,
            "new_roi": float("nan"),
            "delta_profit": 0,
            "delta_roi": float("nan"),
            "new_better_runs": 0,
            "old_better_runs": 0,
            "same_runs": 0,
            "avg_value_signal_selected_overall": float("nan"),
            "avg_max_value_signal_selected_overall": float("nan"),
            "avg_value_min_eff_overall": float("nan"),
            "total_dropped_by_value_gate": 0,
            "avg_best_value_ratio_overall": float("nan"),
            "p50_best_value_ratio_overall": float("nan"),
            "avg_best_value_ratio_win_overall": float("nan"),
            "p50_best_value_ratio_win_overall": float("nan"),
            "total_dropped_by_value_gate_win": 0,
            "value_gate_enabled_by_type": "",
        }

    old_stake = int(df["old_stake"].sum())
    old_profit = int(df["old_profit"].sum())
    new_stake = int(df["new_stake"].sum())
    new_profit = int(df["new_profit"].sum())
    old_roi = _roi(old_stake, old_profit)
    new_roi = _roi(new_stake, new_profit)
    avg_value_signal_selected_overall = _mean_finite(df.get("avg_value_signal_selected", pd.Series(dtype=float)).tolist())
    avg_max_value_signal_selected_overall = _mean_finite(df.get("max_value_signal_selected", pd.Series(dtype=float)).tolist())
    avg_value_min_eff_overall = _mean_finite(df.get("avg_value_min_eff", pd.Series(dtype=float)).tolist())
    total_dropped_by_value_gate = int(pd.to_numeric(df.get("dropped_by_value_gate", 0), errors="coerce").fillna(0).sum())
    total_dropped_by_value_gate_win = int(
        pd.to_numeric(df.get("dropped_by_value_gate_win", 0), errors="coerce").fillna(0).sum()
    )
    best_series = pd.to_numeric(df.get("best_value_ratio", pd.Series(dtype=float)), errors="coerce")
    avg_best_value_ratio_overall = _mean_finite(best_series.tolist())
    if best_series.notna().any():
        p50_best_value_ratio_overall = float(best_series.quantile(0.5))
    else:
        p50_best_value_ratio_overall = float("nan")
    best_win_series = pd.to_numeric(df.get("best_value_ratio_win", pd.Series(dtype=float)), errors="coerce")
    avg_best_value_ratio_win_overall = _mean_finite(best_win_series.tolist())
    if best_win_series.notna().any():
        p50_best_value_ratio_win_overall = float(best_win_series.quantile(0.5))
    else:
        p50_best_value_ratio_win_overall = float("nan")
    gate_series = df.get("value_gate_enabled_by_type", pd.Series(dtype=object))
    gate_series = gate_series.astype(str)
    gate_series = gate_series[gate_series.str.strip() != ""]
    value_gate_enabled_by_type = str(gate_series.iloc[0]) if len(gate_series) > 0 else ""
    return df, {
        "engine": engine,
        "scope": scope_key,
        "runs": int(len(df)),
        "old_stake": old_stake,
        "old_profit": old_profit,
        "old_roi": old_roi,
        "new_stake": new_stake,
        "new_profit": new_profit,
        "new_roi": new_roi,
        "delta_profit": int(new_profit - old_profit),
        "delta_roi": (new_roi - old_roi) if pd.notna(old_roi) and pd.notna(new_roi) else float("nan"),
        "new_better_runs": int((df["delta_profit"] > 0).sum()),
        "old_better_runs": int((df["delta_profit"] < 0).sum()),
        "same_runs": int((df["delta_profit"] == 0).sum()),
        "avg_value_signal_selected_overall": float(avg_value_signal_selected_overall),
        "avg_max_value_signal_selected_overall": float(avg_max_value_signal_selected_overall),
        "avg_value_min_eff_overall": float(avg_value_min_eff_overall),
        "total_dropped_by_value_gate": int(total_dropped_by_value_gate),
        "avg_best_value_ratio_overall": float(avg_best_value_ratio_overall),
        "p50_best_value_ratio_overall": float(p50_best_value_ratio_overall),
        "avg_best_value_ratio_win_overall": float(avg_best_value_ratio_win_overall),
        "p50_best_value_ratio_win_overall": float(p50_best_value_ratio_win_overall),
        "total_dropped_by_value_gate_win": int(total_dropped_by_value_gate_win),
        "value_gate_enabled_by_type": value_gate_enabled_by_type,
    }


def main():
    parser = argparse.ArgumentParser(description="Replay compare old ticket history vs selected bet engine.")
    parser.add_argument("--scopes", default="central_dirt,central_turf,local", help="Comma-separated scopes")
    parser.add_argument("--engine", choices=ENGINE_CHOICES, default="v2", help="new engine version")
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Override bankroll per run. 0 means use old run base_amount from results.csv",
    )
    args = parser.parse_args()

    scopes = _parse_scopes(args.scopes)
    budget = int(args.budget)
    engine = str(args.engine).strip().lower()
    if budget < 0:
        budget = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_frames = []
    summary_rows = []
    for scope in scopes:
        detail_df, summary = compare_scope(scope, budget, engine=engine)
        summary_rows.append(summary)
        if not detail_df.empty:
            detail_frames.append(detail_df)

    out_root = BASE_DIR / "data"
    out_root.mkdir(parents=True, exist_ok=True)
    detail_path = out_root / f"replay_compare_bet_engines_detail_{ts}.csv"
    summary_path = out_root / f"replay_compare_bet_engines_summary_{ts}.csv"

    detail_out = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    detail_out.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        old_stake = int(summary_df["old_stake"].sum())
        old_profit = int(summary_df["old_profit"].sum())
        new_stake = int(summary_df["new_stake"].sum())
        new_profit = int(summary_df["new_profit"].sum())
        overall = {
            "scope": "overall",
            "runs": int(summary_df["runs"].sum()),
            "old_stake": old_stake,
            "old_profit": old_profit,
            "old_roi": _roi(old_stake, old_profit),
            "new_stake": new_stake,
            "new_profit": new_profit,
            "new_roi": _roi(new_stake, new_profit),
            "delta_profit": int(new_profit - old_profit),
            "delta_roi": _roi(new_stake, new_profit) - _roi(old_stake, old_profit),
            "new_better_runs": int(summary_df["new_better_runs"].sum()),
            "old_better_runs": int(summary_df["old_better_runs"].sum()),
            "same_runs": int(summary_df["same_runs"].sum()),
            "avg_value_signal_selected_overall": _mean_finite(
                pd.to_numeric(summary_df.get("avg_value_signal_selected_overall", pd.Series(dtype=float)), errors="coerce")
                .tolist()
            ),
            "avg_max_value_signal_selected_overall": _mean_finite(
                pd.to_numeric(
                    summary_df.get("avg_max_value_signal_selected_overall", pd.Series(dtype=float)),
                    errors="coerce",
                ).tolist()
            ),
            "avg_value_min_eff_overall": _mean_finite(
                pd.to_numeric(summary_df.get("avg_value_min_eff_overall", pd.Series(dtype=float)), errors="coerce").tolist()
            ),
            "total_dropped_by_value_gate": int(
                pd.to_numeric(summary_df.get("total_dropped_by_value_gate", pd.Series(dtype=float)), errors="coerce")
                .fillna(0)
                .sum()
            ),
            "avg_best_value_ratio_overall": _mean_finite(
                pd.to_numeric(summary_df.get("avg_best_value_ratio_overall", pd.Series(dtype=float)), errors="coerce")
                .tolist()
            ),
            "p50_best_value_ratio_overall": _mean_finite(
                pd.to_numeric(summary_df.get("p50_best_value_ratio_overall", pd.Series(dtype=float)), errors="coerce")
                .tolist()
            ),
            "avg_best_value_ratio_win_overall": _mean_finite(
                pd.to_numeric(summary_df.get("avg_best_value_ratio_win_overall", pd.Series(dtype=float)), errors="coerce")
                .tolist()
            ),
            "p50_best_value_ratio_win_overall": _mean_finite(
                pd.to_numeric(summary_df.get("p50_best_value_ratio_win_overall", pd.Series(dtype=float)), errors="coerce")
                .tolist()
            ),
            "total_dropped_by_value_gate_win": int(
                pd.to_numeric(summary_df.get("total_dropped_by_value_gate_win", pd.Series(dtype=float)), errors="coerce")
                .fillna(0)
                .sum()
            ),
            "value_gate_enabled_by_type": str(
                summary_df.get("value_gate_enabled_by_type", pd.Series(dtype=object))
                .astype(str)
                .replace("nan", "")
                .loc[lambda s: s.str.strip() != ""]
                .head(1)
                .tolist()[0]
            )
            if len(
                summary_df.get("value_gate_enabled_by_type", pd.Series(dtype=object))
                .astype(str)
                .replace("nan", "")
                .loc[lambda s: s.str.strip() != ""]
                .head(1)
            )
            > 0
            else "",
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"engine={engine} budget={budget}")
    if summary_df.empty:
        print("No comparable runs found.")
    else:
        print(summary_df.to_string(index=False))
    print(f"saved summary: {summary_path}")
    print(f"saved detail : {detail_path}")


if __name__ == "__main__":
    main()

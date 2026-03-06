import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from calibration.temperature_scaling import apply_temperature


BASE_DIR = Path(__file__).resolve().parent
CALIB_DIR = BASE_DIR / "data" / "model_calibration"
CONST_DIR = BASE_DIR / "data" / "bet_constants"


@dataclass
class BetCandidateV3:
    bet_type: str
    horses: Tuple[str, ...]
    odds_used: float
    odds_eff: float
    p_hit: float
    edge: float
    kelly_f: float
    stake_yen: int
    why: str


DEFAULT_CONFIG = {
    "enabled": True,
    "p_mix_w": 0.6,
    "rank_temperature": 1.0,
    "win_temp": 1.0,
    "calibration_enabled": True,
    "p_mid_odds_threshold": 0.18,
    "N_rank": 12,
    "N_value": 12,
    "max_candidate_horses": 16,
    "target_risk_share": 0.25,  # Upper cap only; no forced full allocation.
    "kelly_scale": 1.0,
    "odds_shrink_power": 0.5,
    "odds_power": 0.70,
    "max_ticket_share": 0.15,
    "min_yen_unit": 100,
    "max_pair_tickets_per_horse": 3,
    "min_p_hit_per_ticket": 0.12,
    "min_p_win_per_ticket": 0.06,
    "rank_weight_floor": 0.55,
    "rank_weight_ceil": 1.00,
    "min_edge_per_ticket": 0.03,
    "ticket_prob_floor_scale": 1.0,
    "max_high_odds_tickets_per_race": 1,
    "min_low_or_mid_presence": True,
    "fallback_max_odds_place": 10.0,
    "high_bucket_odds_threshold": 10.0,
    "high_exposure_cap_share": 0.15,
    "low_mid_min_share": 0.60,
    "exposure_enforcement_mode": "trim",
    "min_ev": {"win": 0.03, "place": 0.01, "wide": 0.01, "quinella": 0.02},
    "min_p": {"win": 0.03, "place": 0.05, "wide": 0.04, "quinella": 0.04},
    "penalty": {"win": 0.0, "place": 0.02, "wide": 0.02, "quinella": 0.0},
    "K_place": 3.0,
    "C_wide": 1.6,
    "C_quinella": 1.6,
}


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


def _normalize_prob(arr: Sequence[float]) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return x
    x = np.clip(x, 0.0, None)
    s = float(np.sum(x))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(x.shape, 1.0 / float(x.size), dtype=float)
    return x / s


def _softmax(arr: Sequence[float]) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return x
    m = float(np.max(x))
    ex = np.exp(x - m)
    s = float(np.sum(ex))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(x.shape, 1.0 / float(x.size), dtype=float)
    return ex / s


def _is_number_text(text: str) -> bool:
    try:
        float(str(text).strip())
        return True
    except Exception:
        return False


def _sort_pair(a: str, b: str) -> Tuple[str, str]:
    sa, sb = str(a), str(b)
    if _is_number_text(sa) and _is_number_text(sb):
        ia = int(float(sa))
        ib = int(float(sb))
        if ia <= ib:
            return str(ia), str(ib)
        return str(ib), str(ia)
    return (sa, sb) if sa <= sb else (sb, sa)


def _floor_to_unit(value: float, unit: int) -> int:
    if value <= 0 or unit <= 0:
        return 0
    return int(math.floor(float(value) / float(unit)) * unit)


def _priority(edge: float, p_hit: float, odds_used: float) -> float:
    return float(edge) * float(p_hit) * math.log(max(1.000001, float(odds_used)))


def _odds_bucket(odds_used: float) -> str:
    odd = float(odds_used or 0.0)
    if odd <= 0:
        return "unknown"
    if odd < 3.0:
        return "low"
    if odd < 10.0:
        return "mid"
    return "high"


def _calc_odds_eff(odds_used: float, odds_power: float) -> float:
    odd = float(odds_used or 0.0)
    if odd <= 0:
        return 0.0
    return float(odd ** max(0.0, float(odds_power)))


def _rank_score_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, 1.0, dtype=float)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax - vmin < 1e-12:
        return np.full(arr.shape, 1.0, dtype=float)
    out = (arr - vmin) / (vmax - vmin)
    out = np.where(np.isfinite(out), out, 1.0)
    return np.clip(out, 0.0, 1.0)


def _load_external_calibration(scope_key: str) -> float:
    path = CALIB_DIR / f"{scope_key}_temp.json"
    if not path.exists():
        return 1.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return max(1e-6, float(data.get("win_temp", 1.0)))
    except Exception:
        return 1.0


def _load_external_constants(scope_key: str) -> Dict[str, float]:
    path = CONST_DIR / f"{scope_key}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "K_place": _safe_float(data.get("K_place"), 0.0),
            "C_wide": _safe_float(data.get("C_wide"), 0.0),
            "C_quinella": _safe_float(data.get("C_quinella"), 0.0),
        }
    except Exception:
        return {}


def _merge_dict(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            tmp = dict(out[key])
            tmp.update(value)
            out[key] = tmp
        else:
            out[key] = value
    return out


def _normalize_cfg(config: Dict, scope_key: str) -> Dict:
    cfg = _merge_dict(DEFAULT_CONFIG, config or {})
    cfg["p_mix_w"] = min(1.0, max(0.0, _safe_float(cfg.get("p_mix_w"), 0.6)))
    cfg["rank_temperature"] = max(1e-6, _safe_float(cfg.get("rank_temperature"), 1.0))
    cfg["win_temp"] = max(1e-6, _safe_float(cfg.get("win_temp"), 1.0))
    cfg["p_mid_odds_threshold"] = min(1.0, max(0.0, _safe_float(cfg.get("p_mid_odds_threshold"), 0.18)))
    cfg["N_rank"] = max(1, _safe_int(cfg.get("N_rank"), 12))
    cfg["N_value"] = max(1, _safe_int(cfg.get("N_value"), 12))
    cfg["max_candidate_horses"] = max(2, _safe_int(cfg.get("max_candidate_horses"), 16))
    cfg["target_risk_share"] = min(1.0, max(0.0, _safe_float(cfg.get("target_risk_share"), 0.25)))
    cfg["kelly_scale"] = max(0.0, _safe_float(cfg.get("kelly_scale"), 1.0))
    cfg["odds_shrink_power"] = min(2.0, max(0.0, _safe_float(cfg.get("odds_shrink_power"), 0.5)))
    cfg["odds_power"] = min(1.0, max(0.0, _safe_float(cfg.get("odds_power"), 0.70)))
    cfg["max_ticket_share"] = min(1.0, max(0.0, _safe_float(cfg.get("max_ticket_share"), 0.15)))
    cfg["min_yen_unit"] = max(1, _safe_int(cfg.get("min_yen_unit"), 100))
    cfg["max_pair_tickets_per_horse"] = max(1, _safe_int(cfg.get("max_pair_tickets_per_horse"), 3))
    cfg["min_p_hit_per_ticket"] = min(1.0, max(0.0, _safe_float(cfg.get("min_p_hit_per_ticket"), 0.12)))
    cfg["min_p_win_per_ticket"] = min(1.0, max(0.0, _safe_float(cfg.get("min_p_win_per_ticket"), 0.06)))
    cfg["rank_weight_floor"] = min(1.0, max(0.0, _safe_float(cfg.get("rank_weight_floor"), 0.55)))
    cfg["rank_weight_ceil"] = min(1.0, max(0.0, _safe_float(cfg.get("rank_weight_ceil"), 1.00)))
    if cfg["rank_weight_floor"] > cfg["rank_weight_ceil"]:
        cfg["rank_weight_floor"], cfg["rank_weight_ceil"] = cfg["rank_weight_ceil"], cfg["rank_weight_floor"]
    cfg["min_edge_per_ticket"] = _safe_float(cfg.get("min_edge_per_ticket"), 0.03)
    cfg["ticket_prob_floor_scale"] = min(5.0, max(0.1, _safe_float(cfg.get("ticket_prob_floor_scale"), 1.0)))
    cfg["max_high_odds_tickets_per_race"] = max(0, _safe_int(cfg.get("max_high_odds_tickets_per_race"), 1))
    cfg["min_low_or_mid_presence"] = bool(cfg.get("min_low_or_mid_presence", True))
    cfg["fallback_max_odds_place"] = max(1.0, _safe_float(cfg.get("fallback_max_odds_place"), 10.0))
    cfg["high_bucket_odds_threshold"] = max(2.0, _safe_float(cfg.get("high_bucket_odds_threshold"), 10.0))
    cfg["high_exposure_cap_share"] = min(1.0, max(0.0, _safe_float(cfg.get("high_exposure_cap_share"), 0.15)))
    cfg["low_mid_min_share"] = min(1.0, max(0.0, _safe_float(cfg.get("low_mid_min_share"), 0.60)))
    mode = str(cfg.get("exposure_enforcement_mode", "trim")).strip().lower()
    cfg["exposure_enforcement_mode"] = mode if mode in ("trim", "redistribute") else "trim"
    cfg["debug_first_bet_race"] = bool(cfg.get("debug_first_bet_race", False))
    cfg["K_place"] = max(0.0, _safe_float(cfg.get("K_place"), 3.0))
    cfg["C_wide"] = max(0.0, _safe_float(cfg.get("C_wide"), 1.6))
    cfg["C_quinella"] = max(0.0, _safe_float(cfg.get("C_quinella"), 1.6))

    min_ev = dict(DEFAULT_CONFIG["min_ev"])
    min_ev.update(cfg.get("min_ev", {}))
    cfg["min_ev"] = {k: _safe_float(min_ev.get(k), 0.0) for k in ("win", "place", "wide", "quinella")}

    min_p = dict(DEFAULT_CONFIG["min_p"])
    min_p.update(cfg.get("min_p", {}))
    cfg["min_p"] = {
        k: min(1.0, max(0.0, _safe_float(min_p.get(k), 0.0)))
        for k in ("win", "place", "wide", "quinella")
    }

    penalty = dict(DEFAULT_CONFIG["penalty"])
    penalty.update(cfg.get("penalty", {}))
    cfg["penalty"] = {
        k: min(0.99, max(0.0, _safe_float(penalty.get(k), 0.0)))
        for k in ("win", "place", "wide", "quinella")
    }

    if bool(cfg.get("calibration_enabled", True)):
        ext_temp = _load_external_calibration(scope_key)
        cfg["win_temp"] = max(1e-6, _safe_float(ext_temp, cfg["win_temp"]))

    ext_const = _load_external_constants(scope_key)
    for k in ("K_place", "C_wide", "C_quinella"):
        if ext_const.get(k, 0.0) > 0:
            cfg[k] = float(ext_const[k])
    return cfg


def _resolve_low_mid_high(raw) -> Tuple[float, float, float]:
    low = mid = high = 0.0
    if isinstance(raw, dict):
        low = _safe_float(raw.get("low"), 0.0)
        mid = _safe_float(raw.get("mid"), 0.0)
        high = _safe_float(raw.get("high"), 0.0)
        if mid <= 0:
            mid = _safe_float(raw.get("odds_mid"), 0.0)
        if low <= 0:
            low = _safe_float(raw.get("odds_low"), 0.0)
        if high <= 0:
            high = _safe_float(raw.get("odds_high"), 0.0)
        if low <= 0:
            low = _safe_float(raw.get("odds"), 0.0)
        if mid <= 0:
            mid = _safe_float(raw.get("odds"), 0.0)
    elif isinstance(raw, (tuple, list)):
        if len(raw) == 1:
            low = mid = high = _safe_float(raw[0], 0.0)
        elif len(raw) >= 2:
            a, b = _safe_float(raw[0], 0.0), _safe_float(raw[1], 0.0)
            if a > 0 and b > 0:
                low, high = (a, b) if a <= b else (b, a)
                mid = (low + high) / 2.0
            else:
                low = max(a, b, 0.0)
                mid = low
                high = low
    else:
        low = mid = high = _safe_float(raw, 0.0)
    return max(0.0, low), max(0.0, mid), max(0.0, high)


def _resolve_odds_used(raw, p_hit: float, penalty: float, p_mid_threshold: float) -> float:
    low, mid, high = _resolve_low_mid_high(raw)
    base = 0.0
    if low > 0 and (mid > 0 or high > 0):
        if p_hit >= p_mid_threshold and mid > 0:
            base = mid
        else:
            base = low
    elif low > 0:
        base = low
    elif mid > 0:
        base = mid
    elif high > 0:
        base = high
    if base <= 0:
        return 0.0
    return max(0.0, float(base) * (1.0 - float(penalty)))


def _lookup_single_odds(odds_map: Dict, horse: str, p_hit: float, penalty: float, p_mid_threshold: float) -> float:
    if not isinstance(odds_map, dict):
        return 0.0
    raw = odds_map.get(str(horse))
    return _resolve_odds_used(raw, p_hit, penalty, p_mid_threshold)


def _lookup_pair_odds(odds_map: Dict, a: str, b: str, p_hit: float, penalty: float, p_mid_threshold: float) -> float:
    if not isinstance(odds_map, dict):
        return 0.0
    x, y = _sort_pair(a, b)
    raw = odds_map.get((x, y))
    if raw is None:
        raw = odds_map.get((y, x))
    return _resolve_odds_used(raw, p_hit, penalty, p_mid_threshold)


def _build_mapping(items: List[BetCandidateV3]) -> Dict[Tuple[str, ...], int]:
    out: Dict[Tuple[str, ...], int] = {}
    for item in items:
        key = tuple([item.bet_type] + list(item.horses))
        out[key] = out.get(key, 0) + int(item.stake_yen)
    return out


def _to_summary(items: List[BetCandidateV3], bankroll: int, risk_budget: int, no_bet: bool) -> Dict:
    total_stake = int(sum(int(x.stake_yen) for x in items))
    expected_return = float(sum(float(x.stake_yen) * float(x.p_hit) * float(x.odds_eff) for x in items))
    expected_profit = float(sum(float(x.edge) * float(x.stake_yen) for x in items))
    return {
        "bankroll_yen": int(bankroll),
        "risk_budget_yen": int(risk_budget),
        "total_stake_yen": int(total_stake),
        "expected_return_yen": int(round(expected_return)),
        "expected_profit_yen": int(round(expected_profit)),
        "ticket_count": int(len(items)),
        "no_bet": bool(no_bet or len(items) == 0),
        "risk_share_used": (float(total_stake) / float(bankroll)) if bankroll > 0 else 0.0,
    }


def _build_candidate_pool(g: pd.DataFrame, cfg: Dict, odds_win_map: Dict) -> List[str]:
    n_rank = min(len(g), int(cfg["N_rank"]))
    n_value = min(len(g), int(cfg["N_value"]))
    g_rank = g.sort_values("rank_score", ascending=False).head(n_rank)
    rank_keys = g_rank["horse_key"].astype(str).tolist()

    g_val = g.copy()
    val_scores = []
    for _, row in g_val.iterrows():
        hk = str(row.get("horse_key", ""))
        p_win = float(row.get("p_win", 0.0))
        odds_win = _lookup_single_odds(
            odds_map=odds_win_map,
            horse=hk,
            p_hit=p_win,
            penalty=cfg["penalty"]["win"],
            p_mid_threshold=cfg["p_mid_odds_threshold"],
        )
        val_scores.append(p_win * odds_win)
    g_val["value_score"] = val_scores
    g_val = g_val.sort_values("value_score", ascending=False).head(n_value)
    value_keys = g_val["horse_key"].astype(str).tolist()

    merged = []
    seen = set()
    for hk in rank_keys + value_keys:
        if hk in seen:
            continue
        seen.add(hk)
        merged.append(hk)
        if len(merged) >= int(cfg["max_candidate_horses"]):
            break
    return merged


def _calc_p_hit(bet_type: str, p_a: float, p_b: float, cfg: Dict) -> float:
    if bet_type == "win":
        return min(1.0, max(0.0, p_a))
    if bet_type == "place":
        return min(1.0, max(0.0, p_a * float(cfg["K_place"])))
    if bet_type == "wide":
        return min(1.0, max(0.0, p_a * p_b * float(cfg["C_wide"])))
    if bet_type == "quinella":
        return min(1.0, max(0.0, p_a * p_b * float(cfg["C_quinella"])))
    return 0.0


def _calc_gate_prob(bet_type: str, p_a: float, p_b: float) -> float:
    # Use conservative (pre-amplification) probability for ticket-level hard floors.
    if bet_type == "win":
        return min(1.0, max(0.0, p_a))
    if bet_type == "place":
        return min(1.0, max(0.0, p_a))
    if bet_type in ("wide", "quinella"):
        return min(1.0, max(0.0, p_a * p_b))
    return 0.0


def _apply_redundancy_filter(cands: List[BetCandidateV3], cfg: Dict) -> List[BetCandidateV3]:
    if not cands:
        return cands
    pair_limit = int(cfg["max_pair_tickets_per_horse"])
    out = []
    pair_count: Dict[str, int] = {}
    ordered = sorted(cands, key=lambda x: _priority(x.edge, x.p_hit, x.odds_used), reverse=True)
    for item in ordered:
        if item.bet_type in ("wide", "quinella") and len(item.horses) >= 2:
            h1, h2 = str(item.horses[0]), str(item.horses[1])
            if pair_count.get(h1, 0) >= pair_limit or pair_count.get(h2, 0) >= pair_limit:
                continue
            pair_count[h1] = pair_count.get(h1, 0) + 1
            pair_count[h2] = pair_count.get(h2, 0) + 1
        out.append(item)
    return out


def _apply_odds_bucket_quota(cands: List[BetCandidateV3], cfg: Dict) -> List[BetCandidateV3]:
    if not cands:
        return []
    max_high = int(cfg.get("max_high_odds_tickets_per_race", 1))
    need_low_mid = bool(cfg.get("min_low_or_mid_presence", True))

    ordered = sorted(cands, key=lambda x: _priority(x.edge, x.p_hit, x.odds_used), reverse=True)
    selected: List[BetCandidateV3] = []
    high_count = 0

    for item in ordered:
        bucket = _odds_bucket(item.odds_used)
        if bucket == "high":
            if max_high <= 0 or high_count >= max_high:
                continue
            high_count += 1
        selected.append(item)

    if not selected:
        return []

    if need_low_mid:
        has_low_mid = any(_odds_bucket(item.odds_used) in ("low", "mid") for item in selected)
        if not has_low_mid:
            low_mid_candidates = [item for item in ordered if _odds_bucket(item.odds_used) in ("low", "mid")]
            if low_mid_candidates:
                replacement = low_mid_candidates[0]
                selected.append(replacement)
                if len(selected) > 1:
                    high_items = [x for x in selected if _odds_bucket(x.odds_used) == "high"]
                    if high_items and max_high >= 0:
                        high_items_sorted = sorted(high_items, key=lambda x: _priority(x.edge, x.p_hit, x.odds_used))
                        drop = high_items_sorted[0]
                        if drop is not replacement and drop in selected:
                            selected.remove(drop)
    return selected


def _resolve_no_bet_reason(diag: Dict) -> str:
    if int(diag.get("after_min_p", 0)) <= 0:
        return "no_candidate_after_min_p"
    if int(diag.get("after_min_edge", 0)) <= 0:
        return "no_candidate_after_min_edge"
    if int(diag.get("after_quota", 0)) <= 0:
        return "no_candidate_after_quota"
    if int(diag.get("after_quota", 0)) > 0 and int(diag.get("after_budget_alloc", 0)) <= 0:
        return "budget_allocation_zero"
    return "other"


def _bucket_for_exposure(odds_used: float, high_threshold: float) -> str:
    odd = float(odds_used or 0.0)
    if odd < 5.0:
        return "low"
    if odd < float(high_threshold):
        return "mid"
    return "high"


def _stake_by_bucket(items: List[BetCandidateV3], high_threshold: float) -> Dict[str, int]:
    out = {"low": 0, "mid": 0, "high": 0}
    for item in items:
        stake = int(item.stake_yen)
        if stake <= 0:
            continue
        bucket = _bucket_for_exposure(item.odds_used, high_threshold)
        out[bucket] = int(out.get(bucket, 0)) + stake
    return out


def _enforce_exposure_cap(cands: List[BetCandidateV3], cfg: Dict) -> Tuple[List[BetCandidateV3], Dict]:
    diag = {
        "exposure_cap_triggered": False,
        "exposure_cap_before_high_stake": 0,
        "exposure_cap_after_high_stake": 0,
        "exposure_cap_removed_tickets_count": 0,
        "low_mid_min_share_triggered": False,
        "redistribution_amount": 0,
        "exposure_high_stake_share_after": 0.0,
        "exposure_low_mid_stake_share_after": 0.0,
        "exposure_total_stake_after": 0,
    }
    if not cands:
        return [], diag

    high_threshold = float(cfg.get("high_bucket_odds_threshold", 10.0))
    high_cap_share = float(cfg.get("high_exposure_cap_share", 0.15))
    low_mid_min_share = float(cfg.get("low_mid_min_share", 0.60))
    mode = str(cfg.get("exposure_enforcement_mode", "trim")).strip().lower()
    min_unit = int(cfg.get("min_yen_unit", 100))

    total_stake = int(sum(int(x.stake_yen) for x in cands))
    if total_stake <= 0:
        return [], diag
    by_bucket = _stake_by_bucket(cands, high_threshold)
    high_stake_before = int(by_bucket["high"])
    diag["exposure_cap_before_high_stake"] = high_stake_before

    removed_amount = 0
    if high_stake_before > float(total_stake) * high_cap_share:
        diag["exposure_cap_triggered"] = True
        high_items = [x for x in cands if _bucket_for_exposure(x.odds_used, high_threshold) == "high"]
        high_items = sorted(high_items, key=lambda x: _priority(x.edge, x.p_hit, x.odds_used))
        high_cap = float(total_stake) * high_cap_share
        high_stake = float(high_stake_before)
        for item in high_items:
            if high_stake <= high_cap + 1e-12:
                break
            removed_amount += int(item.stake_yen)
            high_stake -= float(item.stake_yen)
            item.stake_yen = 0
            diag["exposure_cap_removed_tickets_count"] = int(diag["exposure_cap_removed_tickets_count"]) + 1
        cands = [x for x in cands if int(x.stake_yen) >= min_unit]

        total_after_trim = int(sum(int(x.stake_yen) for x in cands))
        by_bucket = _stake_by_bucket(cands, high_threshold)
        low_mid_stake = int(by_bucket["low"] + by_bucket["mid"])
        if total_after_trim > 0 and float(low_mid_stake) < float(total_after_trim) * low_mid_min_share:
            diag["low_mid_min_share_triggered"] = True
            if mode == "redistribute" and removed_amount >= min_unit:
                targets = [x for x in cands if _bucket_for_exposure(x.odds_used, high_threshold) in ("low", "mid")]
                targets = sorted(targets, key=lambda x: _priority(x.edge, x.p_hit, x.odds_used), reverse=True)[:2]
                if targets:
                    rem_units = int(removed_amount // min_unit)
                    idx = 0
                    while rem_units > 0:
                        t = targets[idx % len(targets)]
                        t.stake_yen = int(t.stake_yen) + min_unit
                        diag["redistribution_amount"] = int(diag["redistribution_amount"]) + min_unit
                        rem_units -= 1
                        idx += 1
                    cands = [x for x in cands if int(x.stake_yen) >= min_unit]

    total_final = int(sum(int(x.stake_yen) for x in cands))
    by_bucket_final = _stake_by_bucket(cands, high_threshold)
    diag["exposure_cap_after_high_stake"] = int(by_bucket_final["high"])
    diag["exposure_total_stake_after"] = int(total_final)
    if total_final > 0:
        diag["exposure_high_stake_share_after"] = float(by_bucket_final["high"]) / float(total_final)
        diag["exposure_low_mid_stake_share_after"] = float(by_bucket_final["low"] + by_bucket_final["mid"]) / float(total_final)
    return cands, diag


def _allocate_stakes(cands: List[BetCandidateV3], bankroll: int, cfg: Dict) -> List[BetCandidateV3]:
    if not cands or bankroll <= 0:
        return []
    min_unit = int(cfg["min_yen_unit"])
    risk_budget = _floor_to_unit(float(bankroll) * float(cfg["target_risk_share"]), min_unit)
    if risk_budget <= 0:
        return []
    single_cap = _floor_to_unit(float(bankroll) * float(cfg["max_ticket_share"]), min_unit)

    raw_stakes = []
    for item in cands:
        kf = max(0.0, float(item.kelly_f))
        shrink = float(cfg["kelly_scale"]) * kf
        if float(cfg["odds_shrink_power"]) > 0:
            shrink /= max(1.0, float(item.odds_used) ** float(cfg["odds_shrink_power"]))
        stake_raw = float(bankroll) * shrink
        if single_cap > 0:
            stake_raw = min(stake_raw, float(single_cap))
        raw_stakes.append(max(0.0, stake_raw))

    total_raw = float(sum(raw_stakes))
    # target_risk_share is an upper cap only. If total_raw is below budget, do not force fill.
    scale = 1.0
    if total_raw > float(risk_budget) > 0:
        scale = float(risk_budget) / total_raw

    for i, item in enumerate(cands):
        stake = _floor_to_unit(raw_stakes[i] * scale, min_unit)
        item.stake_yen = int(stake)
    cands = [x for x in cands if int(x.stake_yen) >= min_unit]
    if not cands:
        return []

    # Guard against rounding overflow after floor.
    total_stake = int(sum(x.stake_yen for x in cands))
    if total_stake > risk_budget and risk_budget > 0:
        ordered = sorted(cands, key=lambda x: _priority(x.edge, x.p_hit, x.odds_used))
        idx = 0
        while total_stake > risk_budget and ordered:
            item = ordered[idx % len(ordered)]
            if item.stake_yen >= min_unit * 2:
                item.stake_yen -= min_unit
                total_stake -= min_unit
            idx += 1
            if idx > len(ordered) * 50:
                break
        cands = [x for x in cands if x.stake_yen >= min_unit]
    return cands


def generate_bet_plan_v3(
    pred_df,
    odds: Dict,
    bankroll_yen: int,
    scope_key: str,
    config: Dict,
):
    bankroll = max(0, int(bankroll_yen or 0))
    if bankroll <= 0:
        return [], {}, _to_summary([], bankroll=0, risk_budget=0, no_bet=True)
    if pred_df is None or len(pred_df) == 0:
        return [], {}, _to_summary([], bankroll=bankroll, risk_budget=0, no_bet=True)

    cfg = _normalize_cfg(config or {}, str(scope_key or ""))
    work = pred_df.copy()
    if "horse_key" not in work.columns:
        raise ValueError("pred_df missing horse_key")
    if "rank_score" not in work.columns:
        work["rank_score"] = pd.to_numeric(work.get("Top3Prob_model"), errors="coerce")
    if "Top3Prob_model" not in work.columns:
        work["Top3Prob_model"] = pd.to_numeric(work.get("rank_score"), errors="coerce")
    if "race_id" not in work.columns:
        work["race_id"] = "__single_race__"

    work["horse_key"] = work["horse_key"].fillna("").astype(str).str.strip()
    work["rank_score"] = pd.to_numeric(work["rank_score"], errors="coerce")
    work["Top3Prob_model"] = pd.to_numeric(work["Top3Prob_model"], errors="coerce")
    work["race_id"] = work["race_id"].fillna("").astype(str)
    work.loc[work["race_id"] == "", "race_id"] = "__single_race__"
    work = work[(work["horse_key"] != "") & work["rank_score"].notna() & work["Top3Prob_model"].notna()].copy()
    if work.empty:
        return [], {}, _to_summary([], bankroll=bankroll, risk_budget=0, no_bet=True)

    all_items: List[BetCandidateV3] = []
    race_diagnostics: List[Dict] = []
    debug_printed = False
    for _, g in work.groupby("race_id", sort=False):
        g = g.copy()
        race_id = str(g["race_id"].iloc[0]) if "race_id" in g.columns and len(g) > 0 else "__single_race__"
        stage_stats = {
            "race_id": race_id,
            "before_min_p": 0,
            "after_min_p": 0,
            "before_min_edge": 0,
            "after_min_edge": 0,
            "before_quota": 0,
            "after_quota": 0,
            "after_budget_alloc": 0,
            "exposure_cap_triggered": False,
            "exposure_cap_before_high_stake": 0,
            "exposure_cap_after_high_stake": 0,
            "exposure_cap_removed_tickets_count": 0,
            "low_mid_min_share_triggered": False,
            "redistribution_amount": 0,
            "exposure_high_stake_share_after": 0.0,
            "exposure_low_mid_stake_share_after": 0.0,
            "exposure_total_stake_after": 0,
            "allocation_fallback_triggered": False,
            "fallback_reason": "",
            "fallback_ticket_type": "",
            "fallback_p_hit": 0.0,
            "fallback_odds": 0.0,
            "fallback_filtered_by_odds": 0,
            "fallback_no_candidate": False,
            "no_bet_reason": "other",
        }
        scores = g["rank_score"].to_numpy(dtype=float)
        mu = float(np.mean(scores))
        std = float(np.std(scores))
        if (not np.isfinite(std)) or std < 1e-12:
            z = np.zeros_like(scores, dtype=float)
        else:
            z = (scores - mu) / std

        p_base = _softmax(float(cfg["rank_temperature"]) * z)
        p_top3 = _normalize_prob(g["Top3Prob_model"].to_numpy(dtype=float))
        p_mix = float(cfg["p_mix_w"]) * p_base + (1.0 - float(cfg["p_mix_w"])) * p_top3
        p_uncal = _normalize_prob(p_mix)
        if bool(cfg.get("calibration_enabled", True)):
            p_win = apply_temperature(p_uncal, float(cfg["win_temp"]))
        else:
            p_win = p_uncal
        p_win = _normalize_prob(p_win)
        g["p_win"] = p_win
        rank_norm = _rank_score_norm(g["rank_score"].to_numpy(dtype=float))
        rank_floor = float(cfg["rank_weight_floor"])
        rank_ceil = float(cfg["rank_weight_ceil"])
        g["rank_w"] = rank_floor + (rank_ceil - rank_floor) * rank_norm

        horse_pool = _build_candidate_pool(g, cfg, odds.get("win", {}))
        if not horse_pool:
            stage_stats["no_bet_reason"] = _resolve_no_bet_reason(stage_stats)
            race_diagnostics.append(dict(stage_stats))
            continue
        p_map = dict(zip(g["horse_key"].astype(str), g["p_win"].astype(float)))
        rank_w_map = dict(zip(g["horse_key"].astype(str), g["rank_w"].astype(float)))

        race_candidates: List[BetCandidateV3] = []
        for hk in horse_pool:
            p = float(p_map.get(hk, 0.0))
            rank_w = float(rank_w_map.get(hk, 1.0))
            for bet_type in ("win", "place"):
                p_hit = _calc_p_hit(bet_type, p, 0.0, cfg)
                p_gate = _calc_gate_prob(bet_type, p, 0.0)
                odds_used = _lookup_single_odds(
                    odds_map=odds.get(bet_type, {}),
                    horse=hk,
                    p_hit=p_hit,
                    penalty=cfg["penalty"][bet_type],
                    p_mid_threshold=cfg["p_mid_odds_threshold"],
                )
                if odds_used <= 1.0:
                    continue
                odds_eff = _calc_odds_eff(odds_used, cfg["odds_power"])
                if odds_eff <= 1.0:
                    continue
                stage_stats["before_min_p"] += 1
                p_floor_raw = float(cfg["min_p_win_per_ticket"]) if bet_type == "win" else float(cfg["min_p_hit_per_ticket"])
                p_floor = min(1.0, max(0.0, p_floor_raw * float(cfg.get("ticket_prob_floor_scale", 1.0))))
                if p_gate < p_floor:
                    continue
                stage_stats["after_min_p"] += 1
                edge_raw = p_hit * odds_eff - 1.0
                stage_stats["before_min_edge"] += 1
                if edge_raw < float(cfg["min_edge_per_ticket"]):
                    continue
                stage_stats["after_min_edge"] += 1
                edge = edge_raw * rank_w
                if edge < float(cfg["min_ev"][bet_type]):
                    continue
                if p_hit < float(cfg["min_p"][bet_type]):
                    continue
                kelly_f = max(0.0, edge / (odds_eff - 1.0))
                race_candidates.append(
                    BetCandidateV3(
                        bet_type=bet_type,
                        horses=(str(hk),),
                        odds_used=float(odds_used),
                        odds_eff=float(odds_eff),
                        p_hit=float(p_hit),
                        edge=float(edge),
                        kelly_f=float(kelly_f),
                        stake_yen=0,
                        why=f"selected:{bet_type}:rank_w={rank_w:.4f}",
                    )
                )

        for a, b in combinations(horse_pool, 2):
            pa, pb = float(p_map.get(a, 0.0)), float(p_map.get(b, 0.0))
            x, y = _sort_pair(a, b)
            rank_w = min(float(rank_w_map.get(a, 1.0)), float(rank_w_map.get(b, 1.0)))
            for bet_type in ("wide", "quinella"):
                p_hit = _calc_p_hit(bet_type, pa, pb, cfg)
                p_gate = _calc_gate_prob(bet_type, pa, pb)
                odds_used = _lookup_pair_odds(
                    odds_map=odds.get(bet_type, {}),
                    a=x,
                    b=y,
                    p_hit=p_hit,
                    penalty=cfg["penalty"][bet_type],
                    p_mid_threshold=cfg["p_mid_odds_threshold"],
                )
                if odds_used <= 1.0:
                    continue
                odds_eff = _calc_odds_eff(odds_used, cfg["odds_power"])
                if odds_eff <= 1.0:
                    continue
                stage_stats["before_min_p"] += 1
                p_floor = min(
                    1.0,
                    max(
                        0.0,
                        float(cfg["min_p_hit_per_ticket"]) * float(cfg.get("ticket_prob_floor_scale", 1.0)),
                    ),
                )
                if p_gate < p_floor:
                    continue
                stage_stats["after_min_p"] += 1
                edge_raw = p_hit * odds_eff - 1.0
                stage_stats["before_min_edge"] += 1
                if edge_raw < float(cfg["min_edge_per_ticket"]):
                    continue
                stage_stats["after_min_edge"] += 1
                edge = edge_raw * rank_w
                if edge < float(cfg["min_ev"][bet_type]):
                    continue
                if p_hit < float(cfg["min_p"][bet_type]):
                    continue
                kelly_f = max(0.0, edge / (odds_eff - 1.0))
                race_candidates.append(
                    BetCandidateV3(
                        bet_type=bet_type,
                        horses=(x, y),
                        odds_used=float(odds_used),
                        odds_eff=float(odds_eff),
                        p_hit=float(p_hit),
                        edge=float(edge),
                        kelly_f=float(kelly_f),
                        stake_yen=0,
                        why=f"selected:{bet_type}:rank_w={rank_w:.4f}",
                    )
                )

        if not race_candidates:
            stage_stats["no_bet_reason"] = _resolve_no_bet_reason(stage_stats)
            race_diagnostics.append(dict(stage_stats))
            continue
        race_candidates.sort(key=lambda x: _priority(x.edge, x.p_hit, x.odds_used), reverse=True)
        race_candidates = _apply_redundancy_filter(race_candidates, cfg)
        stage_stats["before_quota"] = int(len(race_candidates))
        race_candidates = _apply_odds_bucket_quota(race_candidates, cfg)
        stage_stats["after_quota"] = int(len(race_candidates))
        quota_candidates = list(race_candidates)
        allocated_candidates = _allocate_stakes(race_candidates, bankroll=bankroll, cfg=cfg)
        allocated_candidates, exposure_diag = _enforce_exposure_cap(allocated_candidates, cfg)
        stage_stats["exposure_cap_triggered"] = bool(exposure_diag.get("exposure_cap_triggered", False))
        stage_stats["exposure_cap_before_high_stake"] = int(exposure_diag.get("exposure_cap_before_high_stake", 0))
        stage_stats["exposure_cap_after_high_stake"] = int(exposure_diag.get("exposure_cap_after_high_stake", 0))
        stage_stats["exposure_cap_removed_tickets_count"] = int(
            exposure_diag.get("exposure_cap_removed_tickets_count", 0)
        )
        stage_stats["low_mid_min_share_triggered"] = bool(exposure_diag.get("low_mid_min_share_triggered", False))
        stage_stats["redistribution_amount"] = int(exposure_diag.get("redistribution_amount", 0))
        stage_stats["exposure_high_stake_share_after"] = float(exposure_diag.get("exposure_high_stake_share_after", 0.0))
        stage_stats["exposure_low_mid_stake_share_after"] = float(
            exposure_diag.get("exposure_low_mid_stake_share_after", 0.0)
        )
        stage_stats["exposure_total_stake_after"] = int(exposure_diag.get("exposure_total_stake_after", 0))
        allocated_stake = int(sum(int(x.stake_yen) for x in allocated_candidates))
        race_candidates = allocated_candidates
        if int(stage_stats["after_quota"]) > 0 and allocated_stake <= 0 and quota_candidates:
            min_unit = int(cfg.get("min_yen_unit", 100))
            if min_unit > 0:
                # Fallback only allows place/wide tickets and avoids longshot odds.
                type_filtered = [x for x in quota_candidates if str(x.bet_type) in ("place", "wide")]
                max_place_odds = float(cfg.get("fallback_max_odds_place", 10.0))
                odds_filtered = [
                    x
                    for x in type_filtered
                    if (
                        (str(x.bet_type) == "place" and float(x.odds_used) <= max_place_odds)
                        or (str(x.bet_type) == "wide" and float(x.odds_used) <= 10.0)
                    )
                ]
                stage_stats["fallback_filtered_by_odds"] = int(len(type_filtered) - len(odds_filtered))
                if odds_filtered:
                    ordered = sorted(
                        odds_filtered,
                        key=lambda x: (-(float(x.p_hit) / max(1e-9, float(x.odds_used))), -float(x.p_hit), float(x.odds_used)),
                    )
                    best = ordered[0]
                    best.stake_yen = int(min_unit)
                    race_candidates = [best]
                    stage_stats["allocation_fallback_triggered"] = True
                    stage_stats["fallback_reason"] = "budget_allocation_zero"
                    stage_stats["fallback_ticket_type"] = str(best.bet_type)
                    stage_stats["fallback_p_hit"] = float(best.p_hit)
                    stage_stats["fallback_odds"] = float(best.odds_used)
                else:
                    stage_stats["fallback_no_candidate"] = True
        stage_stats["after_budget_alloc"] = int(len(race_candidates))
        if not race_candidates:
            stage_stats["no_bet_reason"] = _resolve_no_bet_reason(stage_stats)
            race_diagnostics.append(dict(stage_stats))
            continue
        if float(sum(x.edge * x.stake_yen for x in race_candidates)) <= 0:
            stage_stats["no_bet_reason"] = _resolve_no_bet_reason(stage_stats)
            race_diagnostics.append(dict(stage_stats))
            continue
        if bool(cfg.get("debug_first_bet_race", False)) and (not debug_printed):
            eff = {
                "min_p_hit_per_ticket": float(cfg.get("min_p_hit_per_ticket", 0.0)),
                "min_p_win_per_ticket": float(cfg.get("min_p_win_per_ticket", 0.0)),
                "min_edge_per_ticket": float(cfg.get("min_edge_per_ticket", 0.0)),
                "ticket_prob_floor_scale": float(cfg.get("ticket_prob_floor_scale", 1.0)),
                "min_low_or_mid_presence": bool(cfg.get("min_low_or_mid_presence", True)),
                "odds_power": float(cfg.get("odds_power", 0.0)),
            }
            print(f"[DEBUG][bet_engine_v3] race_id={race_id} effective_params={eff}")
            print(f"[DEBUG][bet_engine_v3] race_id={race_id} filter_counts={stage_stats}")
            debug_printed = True
        stage_stats["no_bet_reason"] = ""
        race_diagnostics.append(dict(stage_stats))
        all_items.extend(race_candidates)

    risk_budget = _floor_to_unit(bankroll * float(cfg["target_risk_share"]), int(cfg["min_yen_unit"]))
    summary = _to_summary(all_items, bankroll=bankroll, risk_budget=risk_budget, no_bet=(len(all_items) == 0))
    summary["race_diagnostics"] = race_diagnostics
    mapping = _build_mapping(all_items)
    return [asdict(x) for x in all_items], mapping, summary

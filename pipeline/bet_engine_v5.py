"""
bet_engine_v5: professional portfolio betting engine.

Core formulas (per race):
1) Base probabilities
   p_win_base   = normalize(p_mix_w * softmax(rank_temperature * z(rank_score))
                            + (1 - p_mix_w) * normalize(Top3Prob_model))
   p_place_base = min(
       p_place_cap,
       p_place_blend_w * clip(Top3Prob_model, 0, 1)
       + (1 - p_place_blend_w) * clip(p_win_base * place_scale, 0, 1)
   )

2) Calibration (win/place)
   p_cal = apply_calibration(p_base, calibrator)
   If no calibrator file exists, use identity.

3) Market prior fusion
   q_raw  = 1 / odds_used
   q_norm = normalize q_raw within same race and same ticket_type
   p_final = (1 - lambda_market) * p_cal + lambda_market * q_norm

4) Value + uncertainty penalty
   odds_eff = odds_used ** odds_power
   EV       = p_final * odds_eff - 1
   EV_adj   = EV - ev_margin
   penalty  = clip(1 - uncertainty_u, 0.3, 1.0)
   score    = EV_adj * sqrt(p_final) * penalty

5) Portfolio sizing
   race_budget = bankroll * target_risk_share * budget_conf_factor
   budget_conf_factor = clip(0.5 + 0.8 * (confidence_score - 0.5), 0.2, 1.2)
   race_budget = max(min_race_budget, race_budget)
   Fractional Kelly:
     f = kelly_scale * EV / (odds_eff - 1)
     f = clip(f, 0, f_cap_by_type)
   stake_i = race_budget * normalized(f_i) with soft type-share constraints.

All major control variables are configurable via DEFAULT_CONFIG.
"""

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from calibration.calibration_v5 import apply_calibration, load_calibrators


@dataclass
class BetTicketV5:
    ticket_type: str
    horse_ids: Tuple[str, ...]
    p_base: float
    p_cal: float
    p_final: float
    q_market: float
    odds_used: float
    odds_eff: float
    ev: float
    ev_adj: float
    score: float
    value_signal: float
    value_diff: float
    kelly_f: float
    stake_yen: int
    notes: str = ""


DEFAULT_CONFIG = {
    "enabled": True,
    "p_mix_w": 0.6,
    "rank_temperature": 1.0,
    "place_scale": 3.0,
    "p_place_blend_w": 0.5,
    "p_place_cap": 0.75,
    "wide_scale": 1.6,
    "quinella_scale": 1.6,
    "odds_power": 0.72,
    "base_lambda_market": 0.20,
    "lambda_min": 0.05,
    "lambda_max": 0.45,
    "lambda_gap_weight": 0.25,
    "lambda_conf_weight": 0.20,
    "ev_margin": 0.01,
    "min_ev_per_ticket": -0.02,
    "min_p_by_type": {"win": 0.02, "place": 0.05, "wide": 0.03, "quinella": 0.03},
    "value_gate_enabled": True,
    "value_gate_enabled_by_type": {"win": True, "place": False, "pair": False},
    "takeout_mult_by_type": {"win": 1.10, "place": 1.15, "pair": 1.20},
    "odds_max_by_type": {"win": 200.0, "place": 100.0, "pair": 300.0},
    "q_floor_by_type": {"win": 0.01, "place": 0.02, "pair": 0.005},
    "value_ratio_min_by_type": {"win": 0.20, "place": 0.25, "pair": 0.15},
    "value_gate_gap_boost_k": 1.0,
    "gap_for_boost": 0.05,
    "value_entry_gate_enabled": True,
    "min_best_value_ratio_to_bet": 0.15,
    "target_risk_share": 0.02,
    "min_race_budget": 400,
    "min_yen_unit": 100,
    "kelly_scale": 0.45,
    "f_cap_by_type": {"win": 0.25, "place": 0.25, "pair": 0.20},
    "max_tickets_per_race": 6,
    "high_odds_threshold": 10.0,
    "max_high_odds_tickets_per_race": 2,
    "ensure_diversity": True,
    "win_share_range": [0.10, 0.40],
    "place_share_range": [0.30, 0.70],
    "pair_share_range": [0.20, 0.60],
    "debug_first_bet_race": False,
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


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))


def _normalize_prob(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    s = float(np.sum(arr))
    if s <= 0.0 or (not np.isfinite(s)):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
    return arr / s


def _softmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    m = float(np.max(arr))
    ex = np.exp(arr - m)
    s = float(np.sum(ex))
    if s <= 0.0 or (not np.isfinite(s)):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
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


def _ticket_group(ticket_type: str) -> str:
    t = str(ticket_type).strip().lower()
    if t == "win":
        return "win"
    if t == "place":
        return "place"
    return "pair"


def _merge_cfg(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            node = dict(out[key])
            node.update(value)
            out[key] = node
        else:
            out[key] = value
    return out


def _normalize_cfg(config: Dict) -> Dict:
    cfg = _merge_cfg(DEFAULT_CONFIG, config or {})
    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["p_mix_w"] = min(1.0, max(0.0, _safe_float(cfg.get("p_mix_w"), 0.6)))
    cfg["rank_temperature"] = max(1e-6, _safe_float(cfg.get("rank_temperature"), 1.0))
    cfg["place_scale"] = max(0.0, _safe_float(cfg.get("place_scale"), 3.0))
    cfg["p_place_blend_w"] = min(1.0, max(0.0, _safe_float(cfg.get("p_place_blend_w"), 0.5)))
    cfg["p_place_cap"] = min(1.0, max(0.0, _safe_float(cfg.get("p_place_cap"), 0.75)))
    cfg["wide_scale"] = max(0.0, _safe_float(cfg.get("wide_scale"), 1.6))
    cfg["quinella_scale"] = max(0.0, _safe_float(cfg.get("quinella_scale"), 1.6))
    cfg["odds_power"] = min(1.5, max(0.0, _safe_float(cfg.get("odds_power"), 0.70)))

    cfg["base_lambda_market"] = min(0.9, max(0.0, _safe_float(cfg.get("base_lambda_market"), 0.20)))
    cfg["lambda_min"] = min(0.9, max(0.0, _safe_float(cfg.get("lambda_min"), 0.05)))
    cfg["lambda_max"] = min(0.95, max(cfg["lambda_min"], _safe_float(cfg.get("lambda_max"), 0.50)))
    cfg["lambda_gap_weight"] = max(0.0, _safe_float(cfg.get("lambda_gap_weight"), 0.25))
    cfg["lambda_conf_weight"] = max(0.0, _safe_float(cfg.get("lambda_conf_weight"), 0.20))

    cfg["ev_margin"] = _safe_float(cfg.get("ev_margin"), 0.01)
    cfg["min_ev_per_ticket"] = _safe_float(cfg.get("min_ev_per_ticket"), -0.02)
    min_p = dict(DEFAULT_CONFIG["min_p_by_type"])
    min_p.update(cfg.get("min_p_by_type", {}))
    cfg["min_p_by_type"] = {
        "win": min(1.0, max(0.0, _safe_float(min_p.get("win"), 0.02))),
        "place": min(1.0, max(0.0, _safe_float(min_p.get("place"), 0.05))),
        "wide": min(1.0, max(0.0, _safe_float(min_p.get("wide"), 0.03))),
        "quinella": min(1.0, max(0.0, _safe_float(min_p.get("quinella"), 0.03))),
    }
    cfg["value_gate_enabled"] = bool(cfg.get("value_gate_enabled", True))
    gate_by_type = dict(DEFAULT_CONFIG.get("value_gate_enabled_by_type", {"win": True, "place": False, "pair": False}))
    gate_by_type.update(cfg.get("value_gate_enabled_by_type", {}))
    cfg["value_gate_enabled_by_type"] = {
        "win": bool(gate_by_type.get("win", True)),
        "place": bool(gate_by_type.get("place", False)),
        "pair": bool(gate_by_type.get("pair", False)),
    }
    takeout = dict(DEFAULT_CONFIG["takeout_mult_by_type"])
    takeout.update(cfg.get("takeout_mult_by_type", {}))
    cfg["takeout_mult_by_type"] = {
        "win": max(0.0, _safe_float(takeout.get("win"), 1.10)),
        "place": max(0.0, _safe_float(takeout.get("place"), 1.15)),
        "pair": max(0.0, _safe_float(takeout.get("pair"), 1.20)),
    }
    odds_max = dict(DEFAULT_CONFIG["odds_max_by_type"])
    odds_max.update(cfg.get("odds_max_by_type", {}))
    cfg["odds_max_by_type"] = {
        "win": max(1.01, _safe_float(odds_max.get("win"), 200.0)),
        "place": max(1.01, _safe_float(odds_max.get("place"), 100.0)),
        "pair": max(1.01, _safe_float(odds_max.get("pair"), 300.0)),
    }
    q_floor = dict(DEFAULT_CONFIG["q_floor_by_type"])
    q_floor.update(cfg.get("q_floor_by_type", {}))
    cfg["q_floor_by_type"] = {
        "win": min(0.999, max(0.0, _safe_float(q_floor.get("win"), 0.01))),
        "place": min(0.999, max(0.0, _safe_float(q_floor.get("place"), 0.02))),
        "pair": min(0.999, max(0.0, _safe_float(q_floor.get("pair"), 0.005))),
    }
    ratio_min = dict(DEFAULT_CONFIG["value_ratio_min_by_type"])
    ratio_min.update(cfg.get("value_ratio_min_by_type", {}))
    cfg["value_ratio_min_by_type"] = {
        "win": max(0.0, _safe_float(ratio_min.get("win"), 0.20)),
        "place": max(0.0, _safe_float(ratio_min.get("place"), 0.25)),
        "pair": max(0.0, _safe_float(ratio_min.get("pair"), 0.15)),
    }
    cfg["value_gate_gap_boost_k"] = max(0.0, _safe_float(cfg.get("value_gate_gap_boost_k"), 1.0))
    cfg["gap_for_boost"] = max(0.0, _safe_float(cfg.get("gap_for_boost"), 0.05))
    cfg["value_entry_gate_enabled"] = bool(cfg.get("value_entry_gate_enabled", True))
    cfg["min_best_value_ratio_to_bet"] = max(0.0, _safe_float(cfg.get("min_best_value_ratio_to_bet"), 0.15))

    cfg["target_risk_share"] = min(1.0, max(0.0, _safe_float(cfg.get("target_risk_share"), 0.25)))
    cfg["min_race_budget"] = max(0, _safe_int(cfg.get("min_race_budget"), 400))
    cfg["min_yen_unit"] = max(1, _safe_int(cfg.get("min_yen_unit"), 100))
    cfg["kelly_scale"] = max(0.0, _safe_float(cfg.get("kelly_scale"), 0.50))
    f_cap = dict(DEFAULT_CONFIG["f_cap_by_type"])
    f_cap.update(cfg.get("f_cap_by_type", {}))
    cfg["f_cap_by_type"] = {
        "win": max(0.0, _safe_float(f_cap.get("win"), 0.25)),
        "place": max(0.0, _safe_float(f_cap.get("place"), 0.25)),
        "pair": max(0.0, _safe_float(f_cap.get("pair"), 0.20)),
    }

    cfg["max_tickets_per_race"] = max(0, _safe_int(cfg.get("max_tickets_per_race"), 6))
    cfg["high_odds_threshold"] = max(1.01, _safe_float(cfg.get("high_odds_threshold"), 10.0))
    cfg["max_high_odds_tickets_per_race"] = max(0, _safe_int(cfg.get("max_high_odds_tickets_per_race"), 2))
    cfg["ensure_diversity"] = bool(cfg.get("ensure_diversity", True))

    for key, default in (
        ("win_share_range", [0.10, 0.40]),
        ("place_share_range", [0.30, 0.70]),
        ("pair_share_range", [0.20, 0.60]),
    ):
        raw = cfg.get(key, default)
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            raw = default
        lo = min(1.0, max(0.0, _safe_float(raw[0], default[0])))
        hi = min(1.0, max(lo, _safe_float(raw[1], default[1])))
        cfg[key] = [lo, hi]

    cfg["debug_first_bet_race"] = bool(cfg.get("debug_first_bet_race", True))
    return cfg


def _resolve_odds(raw) -> float:
    values: List[float] = []
    if isinstance(raw, dict):
        for key in ("low", "odds_low", "mid", "odds_mid", "high", "odds_high", "odds", "value"):
            v = _safe_float(raw.get(key), 0.0)
            if v > 0:
                values.append(v)
    elif isinstance(raw, (tuple, list)):
        for x in raw:
            v = _safe_float(x, 0.0)
            if v > 0:
                values.append(v)
    else:
        v = _safe_float(raw, 0.0)
        if v > 0:
            values.append(v)
    if not values:
        return 0.0
    return float(min(values))


def _lookup_single_odds(odds_map: Dict, horse_id: str) -> float:
    if not isinstance(odds_map, dict):
        return 0.0
    return _resolve_odds(odds_map.get(str(horse_id)))


def _lookup_pair_odds(odds_map: Dict, a: str, b: str) -> float:
    if not isinstance(odds_map, dict):
        return 0.0
    x, y = _sort_pair(a, b)
    raw = odds_map.get((x, y))
    if raw is None:
        raw = odds_map.get((y, x))
    return _resolve_odds(raw)


def _calc_confidence(p_win_cal: np.ndarray) -> Dict:
    arr = _normalize_prob(p_win_cal)
    if arr.size == 0:
        return {
            "gap": 0.0,
            "top1_prob": 0.0,
            "stability_score": 0.0,
            "confidence_score": 0.0,
            "uncertainty_u": 1.0,
        }
    ordered = np.sort(arr)[::-1]
    top1 = float(ordered[0]) if len(ordered) >= 1 else 0.0
    top2 = float(ordered[1]) if len(ordered) >= 2 else 0.0
    gap = max(0.0, top1 - top2)
    gap_score = _clip01(gap / 0.15)
    entropy = float(-np.sum(arr * np.log(np.clip(arr, 1e-9, 1.0))))
    entropy_norm = entropy / max(1e-9, math.log(float(max(2, len(arr)))))
    stability = _clip01(1.0 - entropy_norm)
    confidence = _clip01(0.45 * top1 + 0.35 * gap_score + 0.20 * stability)
    uncertainty_u = _clip01(0.5 * (1.0 - stability) + 0.3 * (1.0 - confidence) + 0.2 * (1.0 - gap_score))
    return {
        "gap": float(gap),
        "top1_prob": float(top1),
        "stability_score": float(stability),
        "confidence_score": float(confidence),
        "uncertainty_u": float(uncertainty_u),
    }


def _calc_lambda_market(cfg: Dict, gap: float, confidence_score: float) -> float:
    gap = max(0.0, float(gap))
    confidence_score = _clip01(confidence_score)
    mixedness = _clip01(1.0 - gap / 0.15)  # gap小 => 更混戦 => mixedness高
    lam = (
        float(cfg["base_lambda_market"])
        + float(cfg["lambda_gap_weight"]) * (mixedness - 0.5)
        + float(cfg["lambda_conf_weight"]) * (0.5 - confidence_score)
    )
    return float(min(float(cfg["lambda_max"]), max(float(cfg["lambda_min"]), lam)))


def _to_group(ticket_type: str) -> str:
    t = str(ticket_type).strip().lower()
    if t == "win":
        return "win"
    if t == "place":
        return "place"
    return "pair"


def _candidate_type_counts(items: List[BetTicketV5]) -> Dict[str, int]:
    out = {"win": 0, "place": 0, "wide": 0, "quinella": 0}
    for x in items:
        t = str(x.ticket_type).strip().lower()
        if t in out:
            out[t] += 1
    return out


def _calc_value_ratio_min_eff_by_type(cfg: Dict, gap: float) -> Dict[str, float]:
    gap_for_boost = float(max(0.0, _safe_float(cfg.get("gap_for_boost"), 0.05)))
    boost_k = float(max(0.0, _safe_float(cfg.get("value_gate_gap_boost_k"), 1.0)))
    factor = 1.0
    if gap_for_boost > 0 and float(gap) < gap_for_boost:
        factor = 1.0 + boost_k * (gap_for_boost - float(gap)) / gap_for_boost
    base = cfg.get("value_ratio_min_by_type", {})
    return {
        "win": float(max(0.0, _safe_float(base.get("win"), 0.20)) * factor),
        "place": float(max(0.0, _safe_float(base.get("place"), 0.25)) * factor),
        "pair": float(max(0.0, _safe_float(base.get("pair"), 0.15)) * factor),
    }


def _select_candidates(cands: List[BetTicketV5], cfg: Dict) -> List[BetTicketV5]:
    if not cands or int(cfg["max_tickets_per_race"]) <= 0:
        return []
    ordered = sorted(cands, key=lambda x: float(x.score), reverse=True)
    max_tickets = int(cfg["max_tickets_per_race"])
    high_threshold = float(cfg["high_odds_threshold"])
    max_high = int(cfg["max_high_odds_tickets_per_race"])
    ensure_div = bool(cfg.get("ensure_diversity", True))

    selected: List[BetTicketV5] = []
    selected_keys = set()
    high_count = 0

    def _try_add(cand: BetTicketV5) -> bool:
        nonlocal high_count
        if len(selected) >= max_tickets:
            return False
        key = (cand.ticket_type, tuple(cand.horse_ids))
        if key in selected_keys:
            return False
        is_high = float(cand.odds_used) >= high_threshold
        if is_high and high_count >= max_high:
            return False
        selected.append(cand)
        selected_keys.add(key)
        if is_high:
            high_count += 1
        return True

    if ensure_div:
        best_by_group: Dict[str, BetTicketV5] = {}
        for cand in ordered:
            grp = _to_group(cand.ticket_type)
            if grp not in best_by_group:
                best_by_group[grp] = cand
        for cand in sorted(best_by_group.values(), key=lambda x: float(x.score), reverse=True):
            _try_add(cand)
            if len({_to_group(x.ticket_type) for x in selected}) >= 2:
                break

    for cand in ordered:
        if len(selected) >= max_tickets:
            break
        _try_add(cand)
    return selected


def _bounded_type_shares(group_weights: Dict[str, float], cfg: Dict) -> Dict[str, float]:
    active = [k for k, v in group_weights.items() if float(v) > 0]
    if not active:
        return {}
    if len(active) == 1:
        return {active[0]: 1.0}

    bounds = {
        "win": tuple(cfg["win_share_range"]),
        "place": tuple(cfg["place_share_range"]),
        "pair": tuple(cfg["pair_share_range"]),
    }
    shares = {}
    weight_sum = float(sum(max(0.0, group_weights[g]) for g in active))
    if weight_sum <= 0:
        for g in active:
            shares[g] = 1.0 / float(len(active))
    else:
        for g in active:
            shares[g] = max(0.0, group_weights[g]) / weight_sum

    for g in active:
        lo = float(bounds[g][0])
        shares[g] = max(shares[g], lo)

    total = float(sum(shares[g] for g in active))
    if total > 1.0:
        reducible = {g: max(0.0, shares[g] - float(bounds[g][0])) for g in active}
        red_sum = float(sum(reducible.values()))
        if red_sum > 0:
            need = total - 1.0
            for g in active:
                cut = need * (reducible[g] / red_sum)
                shares[g] = max(float(bounds[g][0]), shares[g] - cut)
        else:
            for g in active:
                shares[g] = shares[g] / total

    for g in active:
        hi = float(bounds[g][1])
        shares[g] = min(shares[g], hi)

    total = float(sum(shares[g] for g in active))
    if total < 1.0:
        rem = 1.0 - total
        headroom = {g: max(0.0, float(bounds[g][1]) - shares[g]) for g in active}
        head_sum = float(sum(headroom.values()))
        if head_sum > 0:
            for g in active:
                shares[g] += rem * (headroom[g] / head_sum)
        else:
            for g in active:
                shares[g] = shares[g] / max(1e-9, total)

    total = float(sum(shares[g] for g in active))
    if total > 0:
        for g in active:
            shares[g] = shares[g] / total
    return shares


def _allocate_stakes(selected: List[BetTicketV5], race_budget: int, cfg: Dict) -> Tuple[List[BetTicketV5], Dict]:
    if not selected or race_budget <= 0:
        return [], {"stake_by_group": {"win": 0, "place": 0, "pair": 0}, "stake_share_by_group": {}}

    min_unit = int(cfg["min_yen_unit"])
    group_weights: Dict[str, float] = {"win": 0.0, "place": 0.0, "pair": 0.0}
    raw_f: List[float] = []
    groups: List[str] = []
    for item in selected:
        grp = _to_group(item.ticket_type)
        cap = float(cfg["f_cap_by_type"]["pair"]) if grp == "pair" else float(cfg["f_cap_by_type"][grp])
        denom = max(1e-9, float(item.odds_eff) - 1.0)
        f = float(cfg["kelly_scale"]) * float(item.ev) / denom
        f = min(cap, max(0.0, f))
        item.kelly_f = float(f)
        raw_f.append(f)
        groups.append(grp)
        group_weights[grp] += max(0.0, f)

    if sum(raw_f) <= 0:
        for item in selected:
            item.stake_yen = 0
        return selected, {"stake_by_group": {"win": 0, "place": 0, "pair": 0}, "stake_share_by_group": {}}

    shares = _bounded_type_shares(group_weights, cfg)
    group_budget = {g: float(race_budget) * float(shares.get(g, 0.0)) for g in ("win", "place", "pair")}

    raw_stakes = [0.0 for _ in selected]
    for grp in ("win", "place", "pair"):
        idxs = [i for i, g in enumerate(groups) if g == grp and raw_f[i] > 0]
        if not idxs:
            continue
        denom = float(sum(raw_f[i] for i in idxs))
        if denom <= 0:
            continue
        for i in idxs:
            raw_stakes[i] = float(group_budget.get(grp, 0.0)) * (raw_f[i] / denom)

    rounded = [int(math.floor(v / float(min_unit)) * min_unit) for v in raw_stakes]
    rounded = [max(0, x) for x in rounded]
    sum_stake = int(sum(rounded))

    protected_idxs = set()
    if bool(cfg.get("ensure_diversity", True)) and race_budget >= min_unit * 2:
        best_by_group = {}
        for i, item in enumerate(selected):
            grp = _to_group(item.ticket_type)
            old = best_by_group.get(grp)
            if old is None or float(selected[i].score) > float(selected[old].score):
                best_by_group[grp] = i
        if len(best_by_group) >= 2:
            target_idxs = sorted(best_by_group.values(), key=lambda i: float(selected[i].score), reverse=True)[:2]
            protected_idxs = set(target_idxs)
            active_groups = {_to_group(selected[i].ticket_type) for i, v in enumerate(rounded) if int(v) > 0}
            if len(active_groups) < 2:
                for i in target_idxs:
                    if rounded[i] < min_unit:
                        rounded[i] = min_unit
                sum_stake = int(sum(rounded))

    if sum_stake <= 0 and race_budget >= min_unit:
        best_idx = max(range(len(selected)), key=lambda i: float(selected[i].score))
        rounded[best_idx] = min_unit
        sum_stake = min_unit

    if sum_stake > race_budget:
        low_to_high = sorted(range(len(selected)), key=lambda i: float(selected[i].score))
        while sum_stake > race_budget:
            changed = False
            for i in low_to_high:
                if sum_stake <= race_budget:
                    break
                if rounded[i] >= min_unit and ((i not in protected_idxs) or (rounded[i] > min_unit)):
                    rounded[i] -= min_unit
                    sum_stake -= min_unit
                    changed = True
            if not changed:
                # If still over budget, relax protected tickets.
                for i in low_to_high:
                    if sum_stake <= race_budget:
                        break
                    if rounded[i] >= min_unit:
                        rounded[i] -= min_unit
                        sum_stake -= min_unit
                        changed = True
                if not changed:
                    break

    for i, item in enumerate(selected):
        item.stake_yen = int(rounded[i])

    stake_by_group = {"win": 0, "place": 0, "pair": 0}
    for item in selected:
        grp = _to_group(item.ticket_type)
        stake_by_group[grp] += int(item.stake_yen)
    total = int(sum(stake_by_group.values()))
    if total > 0:
        share_real = {k: float(v) / float(total) for k, v in stake_by_group.items()}
    else:
        share_real = {k: 0.0 for k in stake_by_group.keys()}
    return selected, {"stake_by_group": stake_by_group, "stake_share_by_group": share_real}


def _build_mapping(items: List[BetTicketV5]) -> Dict[Tuple[str, ...], int]:
    out: Dict[Tuple[str, ...], int] = {}
    for item in items:
        key = tuple([str(item.ticket_type)] + [str(x) for x in item.horse_ids])
        out[key] = out.get(key, 0) + int(item.stake_yen)
    return out


def _build_strategy_text(confidence_score: float, gap: float, lambda_market: float, stake_share_by_group: Dict[str, float]) -> str:
    if confidence_score < 0.45:
        race_state = "混戦想定。"
    elif confidence_score > 0.70 and gap > 0.08:
        race_state = "一本調子寄り。"
    else:
        race_state = "通常レンジ。"

    place_share = float(stake_share_by_group.get("place", 0.0))
    win_share = float(stake_share_by_group.get("win", 0.0))
    pair_share = float(stake_share_by_group.get("pair", 0.0))
    if place_share >= 0.55:
        style = "複勝中心でドローダウンを抑制。"
    elif win_share >= 0.35:
        style = "勝負寄りで単勝比率を引き上げ。"
    elif pair_share >= 0.45:
        style = "連系を厚めにして分散。"
    else:
        style = "票種分散でバランス運用。"

    if lambda_market >= 0.30:
        market_side = "市場寄りの融合（λ高め）。"
    elif lambda_market <= 0.15:
        market_side = "AI寄りの融合（λ低め）。"
    else:
        market_side = "市場とAIを中庸で融合。"

    return "\n".join([race_state, style, market_side])


def _to_output_rows(items: List[BetTicketV5]) -> List[Dict]:
    rows = []
    for x in items:
        rows.append(
            {
                "ticket_type": str(x.ticket_type),
                "horse_ids": list(x.horse_ids),
                "p_base": float(x.p_base),
                "p_cal": float(x.p_cal),
                "p_final": float(x.p_final),
                "q_market": float(x.q_market),
                "odds_used": float(x.odds_used),
                "odds_eff": float(x.odds_eff),
                "EV": float(x.ev),
                "EV_adj": float(x.ev_adj),
                "value_signal": float(x.value_signal),
                "value_ratio": float(x.value_signal),
                "value_diff": float(x.value_diff),
                "score": float(x.score),
                "kelly_f": float(x.kelly_f),
                "stake": int(x.stake_yen),
                # Compatibility aliases for existing pipelines:
                "bet_type": str(x.ticket_type),
                "horses": list(x.horse_ids),
                "p_hit": float(x.p_final),
                "edge": float(x.ev_adj),
                "stake_yen": int(x.stake_yen),
                "notes": str(x.notes or ""),
            }
        )
    return rows


def _to_summary(items: List[BetTicketV5], bankroll: int, race_diags: List[Dict], strategy_text: str, calib_meta: Dict) -> Dict:
    total_stake = int(sum(int(x.stake_yen) for x in items))
    expected_return = float(sum(float(x.stake_yen) * float(x.p_final) * float(x.odds_eff) for x in items))
    expected_profit = float(sum(float(x.stake_yen) * float(x.ev) for x in items))
    return {
        "bankroll_yen": int(bankroll),
        "ticket_count": int(len(items)),
        "total_stake_yen": int(total_stake),
        "expected_return_yen": int(round(expected_return)),
        "expected_profit_yen": int(round(expected_profit)),
        "risk_share_used": (float(total_stake) / float(bankroll)) if bankroll > 0 else 0.0,
        "no_bet": bool(len(items) == 0),
        "strategy_text": str(strategy_text or ""),
        "diagnostics": race_diags,
        "race_diagnostics": race_diags,
        "calibration_meta": calib_meta,
    }


def generate_bet_plan_v5(
    pred_df,
    odds: Dict,
    bankroll_yen: int,
    scope_key: str = "",
    config: Dict = None,
):
    bankroll = max(0, int(bankroll_yen or 0))
    cfg = _normalize_cfg(config or {})
    if (not bool(cfg.get("enabled", True))) or bankroll <= 0:
        return [], {}, _to_summary([], bankroll=bankroll, race_diags=[], strategy_text="", calib_meta={})
    if pred_df is None or len(pred_df) == 0:
        return [], {}, _to_summary([], bankroll=bankroll, race_diags=[], strategy_text="", calib_meta={})

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
    work["race_id"] = work["race_id"].fillna("").astype(str)
    work.loc[work["race_id"] == "", "race_id"] = "__single_race__"
    work["rank_score"] = pd.to_numeric(work["rank_score"], errors="coerce")
    work["Top3Prob_model"] = pd.to_numeric(work["Top3Prob_model"], errors="coerce").fillna(0.0)
    work = work[(work["horse_key"] != "") & work["rank_score"].notna()].copy()
    if work.empty:
        return [], {}, _to_summary([], bankroll=bankroll, race_diags=[], strategy_text="", calib_meta={})

    calib_bundle = load_calibrators(scope_key=str(scope_key or "default"))
    calib_win = calib_bundle.get("models", {}).get("win", {"method": "identity"})
    calib_place = calib_bundle.get("models", {}).get("place", {"method": "identity"})

    all_items: List[BetTicketV5] = []
    race_diags: List[Dict] = []
    strategy_text_all: List[str] = []
    debug_printed = False

    for _, g in work.groupby("race_id", sort=False):
        g = g.copy()
        race_id = str(g["race_id"].iloc[0]) if "race_id" in g.columns and len(g) > 0 else "__single_race__"
        filter_counts = {
            "odds_missing": 0,
            "dropped_by_invalid_odds": 0,
            "low_p_final": 0,
            "low_ev_adj": 0,
            "dropped_by_value_gate": 0,
            "dropped_by_value_gate_win": 0,
            "blocked_by_entry_gate": 0,
            "selected": 0,
            "total_raw": 0,
        }
        dropped_by_invalid_odds_count_by_type = {"win": 0, "place": 0, "wide": 0, "quinella": 0}

        scores = g["rank_score"].to_numpy(dtype=float)
        mu = float(np.mean(scores))
        std = float(np.std(scores))
        if (not np.isfinite(std)) or std < 1e-12:
            z = np.zeros_like(scores, dtype=float)
        else:
            z = (scores - mu) / std

        p_rank = _softmax(float(cfg["rank_temperature"]) * z)
        p_top3_norm = _normalize_prob(g["Top3Prob_model"].to_numpy(dtype=float))
        p_win_base = _normalize_prob(float(cfg["p_mix_w"]) * p_rank + (1.0 - float(cfg["p_mix_w"])) * p_top3_norm)
        p_place_base = []
        for top3_raw, p_win in zip(g["Top3Prob_model"].to_numpy(dtype=float), p_win_base):
            top3_clip = _clip01(top3_raw)
            model_place = _clip01(float(p_win) * float(cfg["place_scale"]))
            p_raw = float(cfg["p_place_blend_w"]) * top3_clip + (1.0 - float(cfg["p_place_blend_w"])) * model_place
            p_place_base.append(min(float(cfg["p_place_cap"]), _clip01(p_raw)))
        p_place_base = np.asarray(p_place_base, dtype=float)

        p_win_cal = np.asarray(apply_calibration(p_win_base, calib_win), dtype=float)
        p_place_cal = np.asarray(apply_calibration(p_place_base, calib_place), dtype=float)
        p_win_cal = _normalize_prob(np.clip(p_win_cal, 0.0, None))
        p_place_cal = np.clip(p_place_cal, 0.0, 1.0)

        conf = _calc_confidence(p_win_cal)
        gap = float(conf["gap"])
        confidence_score = float(conf["confidence_score"])
        stability_score = float(conf["stability_score"])
        uncertainty_u = float(conf["uncertainty_u"])
        lambda_market = _calc_lambda_market(cfg, gap=gap, confidence_score=confidence_score)
        value_ratio_min_eff_by_type = _calc_value_ratio_min_eff_by_type(cfg, gap=gap)
        value_gate_enabled = bool(cfg.get("value_gate_enabled", True))
        value_gate_enabled_by_type = dict(cfg.get("value_gate_enabled_by_type", {"win": True, "place": False, "pair": False}))
        value_entry_gate_enabled = bool(cfg.get("value_entry_gate_enabled", True))
        min_best_value_ratio_to_bet = float(cfg.get("min_best_value_ratio_to_bet", 0.15))
        penalty = float(min(1.0, max(0.3, 1.0 - uncertainty_u)))
        budget_conf_factor = float(min(1.2, max(0.2, 0.5 + 0.8 * (confidence_score - 0.5))))
        race_budget_raw = float(bankroll) * float(cfg["target_risk_share"]) * budget_conf_factor
        race_budget = int(min(float(bankroll), max(float(cfg["min_race_budget"]), race_budget_raw)))

        horse_ids = g["horse_key"].astype(str).tolist()
        p_win_base_map = dict(zip(g["horse_key"].astype(str), p_win_base.astype(float)))
        p_place_base_map = dict(zip(g["horse_key"].astype(str), p_place_base.astype(float)))
        p_win_cal_map = dict(zip(g["horse_key"].astype(str), p_win_cal.astype(float)))
        p_place_cal_map = dict(zip(g["horse_key"].astype(str), p_place_cal.astype(float)))

        raw_candidates: List[BetTicketV5] = []
        for hk in horse_ids:
            win_odds = _lookup_single_odds(odds.get("win", {}), hk)
            if win_odds > 1.0:
                raw_candidates.append(
                    BetTicketV5(
                        ticket_type="win",
                        horse_ids=(str(hk),),
                        p_base=float(p_win_base_map.get(hk, 0.0)),
                        p_cal=float(p_win_cal_map.get(hk, 0.0)),
                        p_final=0.0,
                        q_market=0.0,
                        odds_used=float(win_odds),
                        odds_eff=0.0,
                        ev=0.0,
                        ev_adj=0.0,
                        score=0.0,
                        value_signal=0.0,
                        value_diff=0.0,
                        kelly_f=0.0,
                        stake_yen=0,
                        notes="",
                    )
                )
            else:
                filter_counts["odds_missing"] += 1

            place_odds = _lookup_single_odds(odds.get("place", {}), hk)
            if place_odds > 1.0:
                raw_candidates.append(
                    BetTicketV5(
                        ticket_type="place",
                        horse_ids=(str(hk),),
                        p_base=float(p_place_base_map.get(hk, 0.0)),
                        p_cal=float(p_place_cal_map.get(hk, 0.0)),
                        p_final=0.0,
                        q_market=0.0,
                        odds_used=float(place_odds),
                        odds_eff=0.0,
                        ev=0.0,
                        ev_adj=0.0,
                        score=0.0,
                        value_signal=0.0,
                        value_diff=0.0,
                        kelly_f=0.0,
                        stake_yen=0,
                        notes="",
                    )
                )
            else:
                filter_counts["odds_missing"] += 1

        for a, b in combinations(horse_ids, 2):
            x, y = _sort_pair(a, b)
            p_wide_base = _clip01(float(p_place_base_map.get(x, 0.0)) * float(p_place_base_map.get(y, 0.0)) * float(cfg["wide_scale"]))
            p_wide_cal = _clip01(float(p_place_cal_map.get(x, 0.0)) * float(p_place_cal_map.get(y, 0.0)) * float(cfg["wide_scale"]))
            p_quin_base = _clip01(float(p_win_base_map.get(x, 0.0)) * float(p_win_base_map.get(y, 0.0)) * float(cfg["quinella_scale"]))
            p_quin_cal = _clip01(float(p_win_cal_map.get(x, 0.0)) * float(p_win_cal_map.get(y, 0.0)) * float(cfg["quinella_scale"]))

            wide_odds = _lookup_pair_odds(odds.get("wide", {}), x, y)
            if wide_odds > 1.0:
                raw_candidates.append(
                    BetTicketV5(
                        ticket_type="wide",
                        horse_ids=(x, y),
                        p_base=float(p_wide_base),
                        p_cal=float(p_wide_cal),
                        p_final=0.0,
                        q_market=0.0,
                        odds_used=float(wide_odds),
                        odds_eff=0.0,
                        ev=0.0,
                        ev_adj=0.0,
                        score=0.0,
                        value_signal=0.0,
                        value_diff=0.0,
                        kelly_f=0.0,
                        stake_yen=0,
                        notes="",
                    )
                )
            else:
                filter_counts["odds_missing"] += 1

            quin_odds = _lookup_pair_odds(odds.get("quinella", {}), x, y)
            if quin_odds > 1.0:
                raw_candidates.append(
                    BetTicketV5(
                        ticket_type="quinella",
                        horse_ids=(x, y),
                        p_base=float(p_quin_base),
                        p_cal=float(p_quin_cal),
                        p_final=0.0,
                        q_market=0.0,
                        odds_used=float(quin_odds),
                        odds_eff=0.0,
                        ev=0.0,
                        ev_adj=0.0,
                        score=0.0,
                        value_signal=0.0,
                        value_diff=0.0,
                        kelly_f=0.0,
                        stake_yen=0,
                        notes="",
                    )
                )
            else:
                filter_counts["odds_missing"] += 1

        filter_counts["total_raw"] = int(len(raw_candidates))
        odds_sane_candidates: List[BetTicketV5] = []
        odds_max_by_group = dict(cfg.get("odds_max_by_type", {"win": 200.0, "place": 100.0, "pair": 300.0}))
        for c in raw_candidates:
            t = str(c.ticket_type).strip().lower()
            grp = _to_group(t)
            odds_max = float(odds_max_by_group.get(grp, 300.0))
            odds_used = float(c.odds_used)
            if (not np.isfinite(odds_used)) or (odds_used < 1.01) or (odds_used > odds_max):
                filter_counts["dropped_by_invalid_odds"] += 1
                if t in dropped_by_invalid_odds_count_by_type:
                    dropped_by_invalid_odds_count_by_type[t] += 1
                continue
            odds_sane_candidates.append(c)
        raw_candidates = odds_sane_candidates

        if not raw_candidates:
            race_diags.append(
                {
                    "race_id": race_id,
                    "lambda_market": lambda_market,
                    "confidence_score": confidence_score,
                    "stability_score": stability_score,
                    "gap": gap,
                    "budget_conf_factor": budget_conf_factor,
                    "race_budget": int(race_budget),
                    "candidate_count_by_type": {"win": 0, "place": 0, "wide": 0, "quinella": 0},
                    "kept_ticket_count_by_type": {"win": 0, "place": 0, "wide": 0, "quinella": 0},
                    "filter_counts": dict(filter_counts),
                    "value_gate_enabled": bool(value_gate_enabled),
                    "value_gate_enabled_by_type": dict(value_gate_enabled_by_type),
                    "value_min_eff_by_type": dict(value_ratio_min_eff_by_type),
                    "dropped_by_invalid_odds_count": int(filter_counts.get("dropped_by_invalid_odds", 0)),
                    "dropped_by_invalid_odds_count_by_type": dict(dropped_by_invalid_odds_count_by_type),
                    "dropped_by_value_gate_count": int(filter_counts.get("dropped_by_value_gate", 0)),
                    "dropped_by_value_gate_win_count": int(filter_counts.get("dropped_by_value_gate_win", 0)),
                    "best_value_ratio": float("nan"),
                    "best_value_ratio_win": float("nan"),
                    "selected_types": [],
                    "avg_value_signal_selected": 0.0,
                    "max_value_signal_selected": 0.0,
                    "avg_value_ratio_selected": 0.0,
                    "max_value_ratio_selected": 0.0,
                    "min_q_adj_seen": float("nan"),
                    "p50_q_adj_seen": float("nan"),
                    "max_q_adj_seen": float("nan"),
                    "min_odds_used_seen": float("nan"),
                    "max_odds_used_seen": float("nan"),
                    "avg_value_min_eff": float(
                        sum(value_ratio_min_eff_by_type.values()) / max(1, len(value_ratio_min_eff_by_type))
                    ),
                    "stake_by_group": {"win": 0, "place": 0, "pair": 0},
                    "stake_share_by_group": {"win": 0.0, "place": 0.0, "pair": 0.0},
                }
            )
            continue

        # Build q_market normalized by ticket_type within race.
        q_raw_by_type: Dict[str, List[float]] = {"win": [], "place": [], "wide": [], "quinella": []}
        for c in raw_candidates:
            q_raw_by_type[str(c.ticket_type)].append(1.0 / max(1e-9, float(c.odds_used)))
        q_sum = {k: float(sum(v)) for k, v in q_raw_by_type.items()}
        type_idx = {"win": 0, "place": 0, "wide": 0, "quinella": 0}
        for c in raw_candidates:
            t = str(c.ticket_type)
            idx = type_idx[t]
            type_idx[t] += 1
            q_raw = q_raw_by_type[t][idx]
            c.q_market = float(q_raw / max(1e-9, q_sum[t]))

        # Win market prior for value gate: overround-normalized win implied probs.
        win_q_raw_map: Dict[str, float] = {}
        for c in raw_candidates:
            if str(c.ticket_type).strip().lower() != "win":
                continue
            hk = str(c.horse_ids[0]) if c.horse_ids else ""
            if not hk:
                continue
            win_q_raw_map[hk] = float(1.0 / max(1e-9, float(c.odds_used)))
        win_q_sum = float(sum(win_q_raw_map.values()))
        if win_q_sum > 0:
            win_q_norm_map = {k: float(v / win_q_sum) for k, v in win_q_raw_map.items()}
        elif win_q_raw_map:
            uni = 1.0 / float(len(win_q_raw_map))
            win_q_norm_map = {k: float(uni) for k in win_q_raw_map.keys()}
        else:
            win_q_norm_map = {}

        pre_value_gate_filtered: List[BetTicketV5] = []
        q_adj_seen: List[float] = []
        odds_used_seen: List[float] = []
        for c in raw_candidates:
            odds_used_seen.append(float(c.odds_used))
            if str(c.ticket_type).strip().lower() == "win":
                hk = str(c.horse_ids[0]) if c.horse_ids else ""
                q_win_norm = float(win_q_norm_map.get(hk, 0.0))
                q_adj_seen.append(float(q_win_norm))
                c.value_diff = float(c.p_cal - q_win_norm)
                c.value_signal = float((float(c.p_cal) / max(q_win_norm, 1e-6)) - 1.0)  # value_ratio_win
            else:
                c.value_diff = float("nan")
                c.value_signal = float("nan")
            p_final = (1.0 - lambda_market) * float(c.p_cal) + lambda_market * float(c.q_market)
            p_final = _clip01(p_final)
            c.p_final = float(p_final)
            c.odds_eff = float(max(0.0, float(c.odds_used) ** float(cfg["odds_power"])))
            c.ev = float(c.p_final * c.odds_eff - 1.0)
            c.ev_adj = float(c.ev - float(cfg["ev_margin"]))
            c.score = float(c.ev_adj * math.sqrt(max(0.0, c.p_final)) * penalty)
            if math.isfinite(float(c.value_signal)) and math.isfinite(float(c.value_diff)):
                value_text = f"v_ratio={c.value_signal:.4f};v_diff={c.value_diff:.4f}"
            else:
                value_text = "v_ratio=nan;v_diff=nan"
            c.notes = (
                f"lambda={lambda_market:.3f};conf={confidence_score:.3f};"
                f"stab={stability_score:.3f};gap={gap:.3f};pen={penalty:.3f};"
                f"{value_text}"
            )
            p_floor = float(cfg["min_p_by_type"].get(c.ticket_type, 0.0))
            if c.p_final < p_floor:
                filter_counts["low_p_final"] += 1
                continue
            if c.ev_adj < float(cfg["min_ev_per_ticket"]):
                filter_counts["low_ev_adj"] += 1
                continue
            pre_value_gate_filtered.append(c)

        min_q_adj_seen = float(min(q_adj_seen)) if q_adj_seen else float("nan")
        p50_q_adj_seen = float(np.quantile(np.asarray(q_adj_seen, dtype=float), 0.5)) if q_adj_seen else float("nan")
        max_q_adj_seen = float(max(q_adj_seen)) if q_adj_seen else float("nan")
        min_odds_used_seen = float(min(odds_used_seen)) if odds_used_seen else float("nan")
        max_odds_used_seen = float(max(odds_used_seen)) if odds_used_seen else float("nan")

        win_value_ratio_candidates = [
            float(x.value_signal)
            for x in pre_value_gate_filtered
            if str(x.ticket_type).strip().lower() == "win" and math.isfinite(float(x.value_signal))
        ]
        best_value_ratio_win = (
            float(max(win_value_ratio_candidates)) if win_value_ratio_candidates else float("nan")
        )
        if value_entry_gate_enabled and (
            (not win_value_ratio_candidates) or (best_value_ratio_win < min_best_value_ratio_to_bet)
        ):
            filter_counts["blocked_by_entry_gate"] += 1
            race_diags.append(
                {
                    "race_id": race_id,
                    "lambda_market": lambda_market,
                    "confidence_score": confidence_score,
                    "stability_score": stability_score,
                    "gap": gap,
                    "budget_conf_factor": budget_conf_factor,
                    "race_budget": int(race_budget),
                    "candidate_count_by_type": _candidate_type_counts(raw_candidates),
                    "kept_ticket_count_by_type": {"win": 0, "place": 0, "wide": 0, "quinella": 0},
                    "filter_counts": dict(filter_counts),
                    "value_gate_enabled": bool(value_gate_enabled),
                    "value_gate_enabled_by_type": dict(value_gate_enabled_by_type),
                    "value_min_eff_by_type": dict(value_ratio_min_eff_by_type),
                    "dropped_by_invalid_odds_count": int(filter_counts.get("dropped_by_invalid_odds", 0)),
                    "dropped_by_invalid_odds_count_by_type": dict(dropped_by_invalid_odds_count_by_type),
                    "dropped_by_value_gate_count": int(filter_counts.get("dropped_by_value_gate", 0)),
                    "dropped_by_value_gate_win_count": int(filter_counts.get("dropped_by_value_gate_win", 0)),
                    "best_value_ratio": float(best_value_ratio_win),
                    "best_value_ratio_win": float(best_value_ratio_win),
                    "selected_types": [],
                    "avg_value_signal_selected": 0.0,
                    "max_value_signal_selected": 0.0,
                    "avg_value_ratio_selected": 0.0,
                    "max_value_ratio_selected": 0.0,
                    "min_q_adj_seen": float(min_q_adj_seen),
                    "p50_q_adj_seen": float(p50_q_adj_seen),
                    "max_q_adj_seen": float(max_q_adj_seen),
                    "min_odds_used_seen": float(min_odds_used_seen),
                    "max_odds_used_seen": float(max_odds_used_seen),
                    "avg_value_min_eff": float(
                        sum(value_ratio_min_eff_by_type.values()) / max(1, len(value_ratio_min_eff_by_type))
                    ),
                    "stake_by_group": {"win": 0, "place": 0, "pair": 0},
                    "stake_share_by_group": {"win": 0.0, "place": 0.0, "pair": 0.0},
                }
            )
            continue

        filtered: List[BetTicketV5] = []
        for c in pre_value_gate_filtered:
            grp = _to_group(c.ticket_type)
            gate_on_this_type = bool(value_gate_enabled) and bool(value_gate_enabled_by_type.get(grp, False))
            value_floor = float(value_ratio_min_eff_by_type.get(grp, 0.0))
            if gate_on_this_type and ((not math.isfinite(float(c.value_signal))) or (float(c.value_signal) < value_floor)):
                filter_counts["dropped_by_value_gate"] += 1
                if grp == "win":
                    filter_counts["dropped_by_value_gate_win"] += 1
                continue
            filtered.append(c)

        kept_ticket_count_by_type = _candidate_type_counts(filtered)
        if not filtered:
            race_diags.append(
                {
                    "race_id": race_id,
                    "lambda_market": lambda_market,
                    "confidence_score": confidence_score,
                    "stability_score": stability_score,
                    "gap": gap,
                    "budget_conf_factor": budget_conf_factor,
                    "race_budget": int(race_budget),
                    "candidate_count_by_type": _candidate_type_counts(raw_candidates),
                    "kept_ticket_count_by_type": dict(kept_ticket_count_by_type),
                    "filter_counts": dict(filter_counts),
                    "value_gate_enabled": bool(value_gate_enabled),
                    "value_gate_enabled_by_type": dict(value_gate_enabled_by_type),
                    "value_min_eff_by_type": dict(value_ratio_min_eff_by_type),
                    "dropped_by_invalid_odds_count": int(filter_counts.get("dropped_by_invalid_odds", 0)),
                    "dropped_by_invalid_odds_count_by_type": dict(dropped_by_invalid_odds_count_by_type),
                    "dropped_by_value_gate_count": int(filter_counts.get("dropped_by_value_gate", 0)),
                    "dropped_by_value_gate_win_count": int(filter_counts.get("dropped_by_value_gate_win", 0)),
                    "best_value_ratio": float(best_value_ratio_win),
                    "best_value_ratio_win": float(best_value_ratio_win),
                    "selected_types": [],
                    "avg_value_signal_selected": 0.0,
                    "max_value_signal_selected": 0.0,
                    "avg_value_ratio_selected": 0.0,
                    "max_value_ratio_selected": 0.0,
                    "min_q_adj_seen": float(min_q_adj_seen),
                    "p50_q_adj_seen": float(p50_q_adj_seen),
                    "max_q_adj_seen": float(max_q_adj_seen),
                    "min_odds_used_seen": float(min_odds_used_seen),
                    "max_odds_used_seen": float(max_odds_used_seen),
                    "avg_value_min_eff": float(
                        sum(value_ratio_min_eff_by_type.values()) / max(1, len(value_ratio_min_eff_by_type))
                    ),
                    "stake_by_group": {"win": 0, "place": 0, "pair": 0},
                    "stake_share_by_group": {"win": 0.0, "place": 0.0, "pair": 0.0},
                }
            )
            continue

        selected = _select_candidates(filtered, cfg)
        selected, alloc_diag = _allocate_stakes(selected, race_budget=race_budget, cfg=cfg)
        selected = [x for x in selected if int(x.stake_yen) > 0]
        filter_counts["selected"] = int(len(selected))
        selected_values = [float(x.value_signal) for x in selected if math.isfinite(float(x.value_signal))]
        avg_value_signal_selected = float(sum(selected_values) / len(selected_values)) if selected_values else 0.0
        max_value_signal_selected = float(max(selected_values)) if selected_values else 0.0

        strategy_text = _build_strategy_text(
            confidence_score=confidence_score,
            gap=gap,
            lambda_market=lambda_market,
            stake_share_by_group=alloc_diag.get("stake_share_by_group", {}),
        )
        strategy_text_all.append(strategy_text)

        diag = {
            "race_id": race_id,
            "lambda_market": float(lambda_market),
            "confidence_score": float(confidence_score),
            "stability_score": float(stability_score),
            "gap": float(gap),
            "budget_conf_factor": float(budget_conf_factor),
            "race_budget": int(race_budget),
            "candidate_count_by_type": _candidate_type_counts(raw_candidates),
            "kept_ticket_count_by_type": dict(kept_ticket_count_by_type),
            "filter_counts": dict(filter_counts),
            "value_gate_enabled": bool(value_gate_enabled),
            "value_gate_enabled_by_type": dict(value_gate_enabled_by_type),
            "value_min_eff_by_type": dict(value_ratio_min_eff_by_type),
            "dropped_by_invalid_odds_count": int(filter_counts.get("dropped_by_invalid_odds", 0)),
            "dropped_by_invalid_odds_count_by_type": dict(dropped_by_invalid_odds_count_by_type),
            "dropped_by_value_gate_count": int(filter_counts.get("dropped_by_value_gate", 0)),
            "dropped_by_value_gate_win_count": int(filter_counts.get("dropped_by_value_gate_win", 0)),
            "best_value_ratio": float(best_value_ratio_win),
            "best_value_ratio_win": float(best_value_ratio_win),
            "selected_types": [str(x.ticket_type) for x in selected],
            "avg_value_signal_selected": float(avg_value_signal_selected),
            "max_value_signal_selected": float(max_value_signal_selected),
            "avg_value_ratio_selected": float(avg_value_signal_selected),
            "max_value_ratio_selected": float(max_value_signal_selected),
            "min_q_adj_seen": float(min_q_adj_seen),
            "p50_q_adj_seen": float(p50_q_adj_seen),
            "max_q_adj_seen": float(max_q_adj_seen),
            "min_odds_used_seen": float(min_odds_used_seen),
            "max_odds_used_seen": float(max_odds_used_seen),
            "avg_value_min_eff": float(
                sum(value_ratio_min_eff_by_type.values()) / max(1, len(value_ratio_min_eff_by_type))
            ),
            "stake_by_group": alloc_diag.get("stake_by_group", {}),
            "stake_share_by_group": alloc_diag.get("stake_share_by_group", {}),
            "calibration_params": {
                "win": dict(calib_win),
                "place": dict(calib_place),
            },
            "strategy_text": strategy_text,
        }
        race_diags.append(diag)

        if bool(cfg.get("debug_first_bet_race", False)) and (not debug_printed) and selected:
            print(f"[DEBUG][bet_engine_v5] race_id={race_id} diagnostics={diag}")
            debug_printed = True

        all_items.extend(selected)

    mapping = _build_mapping(all_items)
    strategy_text = strategy_text_all[0] if strategy_text_all else ""
    summary = _to_summary(
        items=all_items,
        bankroll=bankroll,
        race_diags=race_diags,
        strategy_text=strategy_text,
        calib_meta={
            "scope_key": str(scope_key),
            "source": calib_bundle.get("source", ""),
            "models": calib_bundle.get("models", {}),
        },
    )
    return _to_output_rows(all_items), mapping, summary

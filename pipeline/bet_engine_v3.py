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
    "kelly_scale": 0.25,
    "odds_shrink_power": 0.5,
    "max_ticket_share": 0.15,
    "min_yen_unit": 100,
    "max_pair_tickets_per_horse": 3,
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
    cfg["kelly_scale"] = min(1.0, max(0.0, _safe_float(cfg.get("kelly_scale"), 0.25)))
    cfg["odds_shrink_power"] = min(2.0, max(0.0, _safe_float(cfg.get("odds_shrink_power"), 0.5)))
    cfg["max_ticket_share"] = min(1.0, max(0.0, _safe_float(cfg.get("max_ticket_share"), 0.15)))
    cfg["min_yen_unit"] = max(1, _safe_int(cfg.get("min_yen_unit"), 100))
    cfg["max_pair_tickets_per_horse"] = max(1, _safe_int(cfg.get("max_pair_tickets_per_horse"), 3))
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
    expected_return = float(sum(float(x.stake_yen) * float(x.p_hit) * float(x.odds_used) for x in items))
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
    for _, g in work.groupby("race_id", sort=False):
        g = g.copy()
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

        horse_pool = _build_candidate_pool(g, cfg, odds.get("win", {}))
        if not horse_pool:
            continue
        p_map = dict(zip(g["horse_key"].astype(str), g["p_win"].astype(float)))

        race_candidates: List[BetCandidateV3] = []
        for hk in horse_pool:
            p = float(p_map.get(hk, 0.0))
            for bet_type in ("win", "place"):
                p_hit = _calc_p_hit(bet_type, p, 0.0, cfg)
                odds_used = _lookup_single_odds(
                    odds_map=odds.get(bet_type, {}),
                    horse=hk,
                    p_hit=p_hit,
                    penalty=cfg["penalty"][bet_type],
                    p_mid_threshold=cfg["p_mid_odds_threshold"],
                )
                if odds_used <= 1.0:
                    continue
                edge = p_hit * odds_used - 1.0
                if edge < float(cfg["min_ev"][bet_type]):
                    continue
                if p_hit < float(cfg["min_p"][bet_type]):
                    continue
                kelly_f = max(0.0, edge / (odds_used - 1.0))
                race_candidates.append(
                    BetCandidateV3(
                        bet_type=bet_type,
                        horses=(str(hk),),
                        odds_used=float(odds_used),
                        p_hit=float(p_hit),
                        edge=float(edge),
                        kelly_f=float(kelly_f),
                        stake_yen=0,
                        why=f"selected:{bet_type}",
                    )
                )

        for a, b in combinations(horse_pool, 2):
            pa, pb = float(p_map.get(a, 0.0)), float(p_map.get(b, 0.0))
            x, y = _sort_pair(a, b)
            for bet_type in ("wide", "quinella"):
                p_hit = _calc_p_hit(bet_type, pa, pb, cfg)
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
                edge = p_hit * odds_used - 1.0
                if edge < float(cfg["min_ev"][bet_type]):
                    continue
                if p_hit < float(cfg["min_p"][bet_type]):
                    continue
                kelly_f = max(0.0, edge / (odds_used - 1.0))
                race_candidates.append(
                    BetCandidateV3(
                        bet_type=bet_type,
                        horses=(x, y),
                        odds_used=float(odds_used),
                        p_hit=float(p_hit),
                        edge=float(edge),
                        kelly_f=float(kelly_f),
                        stake_yen=0,
                        why=f"selected:{bet_type}",
                    )
                )

        if not race_candidates:
            continue
        race_candidates.sort(key=lambda x: _priority(x.edge, x.p_hit, x.odds_used), reverse=True)
        race_candidates = _apply_redundancy_filter(race_candidates, cfg)
        race_candidates = _allocate_stakes(race_candidates, bankroll=bankroll, cfg=cfg)
        if not race_candidates:
            continue
        if float(sum(x.edge * x.stake_yen for x in race_candidates)) <= 0:
            continue
        all_items.extend(race_candidates)

    risk_budget = _floor_to_unit(bankroll * float(cfg["target_risk_share"]), int(cfg["min_yen_unit"]))
    summary = _to_summary(all_items, bankroll=bankroll, risk_budget=risk_budget, no_bet=(len(all_items) == 0))
    mapping = _build_mapping(all_items)
    return [asdict(x) for x in all_items], mapping, summary


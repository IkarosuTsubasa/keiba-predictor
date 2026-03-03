import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BetCandidate:
    bet_type: str
    horses: Tuple[str, ...]
    odds_used: float
    p_hit: float
    edge: float
    kelly_f: float
    stake_yen: int
    notes: str = ""


@dataclass
class BetPlanResult:
    items: List[BetCandidate]
    mapping: Dict[Tuple[str, ...], int]


DEFAULT_CONFIG = {
    "p_mix_w": 0.6,
    "rank_temperature": 1.0,
    "K_place": 3.0,
    "place_bias": 0.0,
    "place_power": 1.0,
    "C_wide": 1.6,
    "C_quinella": 1.6,
    "odds_penalty": 0.0,
    "win_odds_penalty": None,
    "place_odds_penalty": 0.02,
    "wide_odds_penalty": 0.05,
    "quinella_odds_penalty": None,
    "kelly_scale": 0.25,
    "min_ev_per_ticket": 0.02,
    "min_p_hit_threshold": 0.05,
    "max_ticket_share": 0.20,
    "max_race_share": 0.40,
    "min_yen_unit": 100,
    "max_single_horses": 8,
    "max_tickets_per_race": 0,
}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_cfg(config: dict) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    cfg["p_mix_w"] = min(1.0, max(0.0, _safe_float(cfg.get("p_mix_w"), 0.6)))
    cfg["rank_temperature"] = max(1e-6, _safe_float(cfg.get("rank_temperature"), 1.0))
    cfg["K_place"] = max(0.0, _safe_float(cfg.get("K_place"), 3.0))
    cfg["place_bias"] = min(1.0, max(0.0, _safe_float(cfg.get("place_bias"), 0.0)))
    cfg["place_power"] = max(0.1, _safe_float(cfg.get("place_power"), 1.0))
    cfg["C_wide"] = max(0.0, _safe_float(cfg.get("C_wide"), 1.6))
    cfg["C_quinella"] = max(0.0, _safe_float(cfg.get("C_quinella"), 1.6))
    cfg["odds_penalty"] = min(0.99, max(0.0, _safe_float(cfg.get("odds_penalty"), 0.0)))
    for key in ("win_odds_penalty", "place_odds_penalty", "wide_odds_penalty", "quinella_odds_penalty"):
        raw = cfg.get(key)
        if raw is None:
            cfg[key] = cfg["odds_penalty"]
        else:
            cfg[key] = min(0.99, max(0.0, _safe_float(raw, cfg["odds_penalty"])))
    cfg["kelly_scale"] = min(1.0, max(0.0, _safe_float(cfg.get("kelly_scale"), 0.25)))
    cfg["min_ev_per_ticket"] = _safe_float(cfg.get("min_ev_per_ticket"), 0.02)
    cfg["min_p_hit_threshold"] = min(1.0, max(0.0, _safe_float(cfg.get("min_p_hit_threshold"), 0.05)))
    cfg["max_ticket_share"] = min(1.0, max(0.0, _safe_float(cfg.get("max_ticket_share"), 0.20)))
    cfg["max_race_share"] = min(1.0, max(0.0, _safe_float(cfg.get("max_race_share"), 0.40)))
    cfg["min_yen_unit"] = max(1, int(_safe_float(cfg.get("min_yen_unit"), 100)))
    cfg["max_single_horses"] = max(1, int(_safe_float(cfg.get("max_single_horses"), 8)))
    cfg["max_tickets_per_race"] = max(0, int(_safe_float(cfg.get("max_tickets_per_race"), 0)))
    return cfg


def _floor_to_unit(value: float, unit: int) -> int:
    if value <= 0 or unit <= 0:
        return 0
    return int(math.floor(float(value) / float(unit)) * unit)


def _is_number_text(text: str) -> bool:
    if text is None:
        return False
    t = str(text).strip()
    if not t:
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


def _sort_horse_pair(a: str, b: str) -> Tuple[str, str]:
    sa = str(a)
    sb = str(b)
    if _is_number_text(sa) and _is_number_text(sb):
        ia = int(float(sa))
        ib = int(float(sb))
        if ia <= ib:
            return str(ia), str(ib)
        return str(ib), str(ia)
    if sa <= sb:
        return sa, sb
    return sb, sa


def _softmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    m = float(np.max(arr))
    ex = np.exp(arr - m)
    s = float(np.sum(ex))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
    return ex / s


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    s = float(np.sum(arr))
    if s <= 0 or (not np.isfinite(s)):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
    return arr / s


def _resolve_odds_used(raw, odds_penalty: float) -> float:
    low = 0.0
    if isinstance(raw, dict):
        lo = _safe_float(raw.get("low"), 0.0)
        hi = _safe_float(raw.get("high"), 0.0)
        cands = [x for x in (lo, hi) if x > 0]
        low = min(cands) if cands else 0.0
    elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
        lo = _safe_float(raw[0], 0.0)
        hi = _safe_float(raw[1], 0.0)
        cands = [x for x in (lo, hi) if x > 0]
        low = min(cands) if cands else 0.0
    else:
        low = _safe_float(raw, 0.0)
    if low <= 0:
        return 0.0
    used = low * (1.0 - float(odds_penalty))
    return float(max(0.0, used))


def _lookup_single_odds(odds_map: dict, horse: str, odds_penalty: float) -> float:
    if not isinstance(odds_map, dict):
        return 0.0
    raw = odds_map.get(str(horse))
    return _resolve_odds_used(raw, odds_penalty)


def _lookup_pair_odds(odds_map: dict, a: str, b: str, odds_penalty: float) -> float:
    if not isinstance(odds_map, dict):
        return 0.0
    x, y = _sort_horse_pair(a, b)
    raw = odds_map.get((x, y))
    if raw is None:
        raw = odds_map.get((y, x))
    return _resolve_odds_used(raw, odds_penalty)


def _build_mapping(items: List[BetCandidate]) -> Dict[Tuple[str, ...], int]:
    out: Dict[Tuple[str, ...], int] = {}
    for item in items:
        key = tuple([item.bet_type] + list(item.horses))
        out[key] = out.get(key, 0) + int(item.stake_yen)
    return out


def _append_note(note: str, extra: str) -> str:
    if not note:
        return extra
    return f"{note};{extra}"


def _calc_priority_score(edge: float, p_hit: float, odds_used: float) -> float:
    return float(edge) * float(p_hit) * math.log(max(1.000001, float(odds_used)))


def _calc_place_prob(p_win: float, cfg: Dict) -> float:
    p = max(0.0, min(1.0, float(p_win)))
    score = float(cfg["place_bias"]) + (p ** float(cfg["place_power"])) * float(cfg["K_place"])
    return min(1.0, max(0.0, score))


def _scale_candidates_to_race_cap(candidates: List[BetCandidate], race_cap: int, min_unit: int) -> List[BetCandidate]:
    if race_cap <= 0 or not candidates:
        return candidates
    total_stake = int(sum(x.stake_yen for x in candidates))
    if total_stake <= race_cap:
        return candidates

    scale = float(race_cap) / float(total_stake)
    scaled = []
    for item in candidates:
        new_stake = _floor_to_unit(item.stake_yen * scale, min_unit)
        if new_stake >= min_unit:
            item.stake_yen = int(new_stake)
            item.notes = _append_note(item.notes, "scaled")
            scaled.append(item)
    if not scaled:
        return []

    used = int(sum(x.stake_yen for x in scaled))
    remain = race_cap - used
    if remain >= min_unit:
        # Remaining unit budget goes to higher-priority tickets first.
        ordered = sorted(
            scaled,
            key=lambda x: _calc_priority_score(x.edge, x.p_hit, x.odds_used),
            reverse=True,
        )
        idx = 0
        n = len(ordered)
        while remain >= min_unit and n > 0:
            ordered[idx % n].stake_yen += min_unit
            ordered[idx % n].notes = _append_note(ordered[idx % n].notes, "scaled_fill")
            remain -= min_unit
            idx += 1
    return scaled


def generate_bet_plan_v2(
    pred_df,
    odds: Dict,
    bankroll_yen: int = 50000,
    scope_key: str = "",
    config: Dict = None,
) -> BetPlanResult:
    cfg = _normalize_cfg(config or {})
    bankroll = max(0, int(bankroll_yen or 0))
    if bankroll <= 0:
        return BetPlanResult(items=[], mapping={})

    if pred_df is None or len(pred_df) == 0:
        return BetPlanResult(items=[], mapping={})

    work = pred_df.copy()
    if "horse_key" not in work.columns:
        raise ValueError("pred_df missing column: horse_key")
    if "rank_score" not in work.columns:
        work["rank_score"] = pd.to_numeric(work.get("Top3Prob_model"), errors="coerce")
    if "Top3Prob_model" not in work.columns:
        work["Top3Prob_model"] = pd.to_numeric(work.get("rank_score"), errors="coerce")

    work["horse_key"] = work["horse_key"].fillna("").astype(str).str.strip()
    work["rank_score"] = pd.to_numeric(work["rank_score"], errors="coerce")
    work["Top3Prob_model"] = pd.to_numeric(work["Top3Prob_model"], errors="coerce")
    work = work[(work["horse_key"] != "") & work["rank_score"].notna() & work["Top3Prob_model"].notna()].copy()
    if work.empty:
        return BetPlanResult(items=[], mapping={})

    if "race_id" in work.columns:
        work["race_id"] = work["race_id"].fillna("").astype(str)
        work.loc[work["race_id"] == "", "race_id"] = "__single_race__"
    else:
        work["race_id"] = "__single_race__"

    win_penalty = cfg["win_odds_penalty"]
    place_penalty = cfg["place_odds_penalty"]
    wide_penalty = cfg["wide_odds_penalty"]
    quinella_penalty = cfg["quinella_odds_penalty"]
    all_items: List[BetCandidate] = []

    for race_id, group in work.groupby("race_id", sort=False):
        g = group.copy()
        scores = g["rank_score"].to_numpy(dtype=float)
        mu = float(np.mean(scores))
        std = float(np.std(scores))
        if (not np.isfinite(std)) or std < 1e-12:
            z = np.zeros_like(scores, dtype=float)
        else:
            z = (scores - mu) / std

        p_base = _softmax(cfg["rank_temperature"] * z)
        p_top3_norm = _normalize_positive(g["Top3Prob_model"].to_numpy(dtype=float))
        p_mix = cfg["p_mix_w"] * p_base + (1.0 - cfg["p_mix_w"]) * p_top3_norm
        p_win = _normalize_positive(p_mix)
        g["p_win"] = p_win

        top_n = int(min(cfg["max_single_horses"], len(g)))
        g_top = g.sort_values("p_win", ascending=False).head(top_n).copy()
        horse_list = g_top["horse_key"].astype(str).tolist()
        p_map = dict(zip(g_top["horse_key"].astype(str).tolist(), g_top["p_win"].astype(float).tolist()))

        race_candidates: List[BetCandidate] = []

        for horse in horse_list:
            p = float(p_map.get(horse, 0.0))
            odds_win = _lookup_single_odds(odds.get("win", {}), horse, win_penalty)
            if odds_win > 1.0:
                edge = p * odds_win - 1.0
                if edge >= cfg["min_ev_per_ticket"] and p >= cfg["min_p_hit_threshold"]:
                    kelly_f = max(0.0, edge / (odds_win - 1.0))
                    stake_raw = bankroll * kelly_f * cfg["kelly_scale"]
                    stake = _floor_to_unit(stake_raw, cfg["min_yen_unit"])
                    cap = _floor_to_unit(bankroll * cfg["max_ticket_share"], cfg["min_yen_unit"])
                    if cap > 0:
                        stake = min(stake, cap)
                    if stake >= cfg["min_yen_unit"]:
                        priority = _calc_priority_score(edge, p, odds_win)
                        race_candidates.append(
                            BetCandidate(
                                "win",
                                (horse,),
                                float(odds_win),
                                float(p),
                                float(edge),
                                float(kelly_f),
                                int(stake),
                                f"priority={priority:.6f}",
                            )
                        )

            p_place = _calc_place_prob(p, cfg)
            odds_place = _lookup_single_odds(odds.get("place", {}), horse, place_penalty)
            if odds_place > 1.0:
                edge = p_place * odds_place - 1.0
                if edge >= cfg["min_ev_per_ticket"] and p_place >= cfg["min_p_hit_threshold"]:
                    kelly_f = max(0.0, edge / (odds_place - 1.0))
                    stake_raw = bankroll * kelly_f * cfg["kelly_scale"]
                    stake = _floor_to_unit(stake_raw, cfg["min_yen_unit"])
                    cap = _floor_to_unit(bankroll * cfg["max_ticket_share"], cfg["min_yen_unit"])
                    if cap > 0:
                        stake = min(stake, cap)
                    if stake >= cfg["min_yen_unit"]:
                        priority = _calc_priority_score(edge, p_place, odds_place)
                        race_candidates.append(
                            BetCandidate(
                                "place",
                                (horse,),
                                float(odds_place),
                                float(p_place),
                                float(edge),
                                float(kelly_f),
                                int(stake),
                                f"priority={priority:.6f}",
                            )
                        )

        for a, b in combinations(horse_list, 2):
            pa = float(p_map.get(a, 0.0))
            pb = float(p_map.get(b, 0.0))
            x, y = _sort_horse_pair(a, b)

            p_wide = min(1.0, pa * pb * cfg["C_wide"])
            odds_wide = _lookup_pair_odds(odds.get("wide", {}), x, y, wide_penalty)
            if odds_wide > 1.0:
                edge = p_wide * odds_wide - 1.0
                if edge >= cfg["min_ev_per_ticket"] and p_wide >= cfg["min_p_hit_threshold"]:
                    kelly_f = max(0.0, edge / (odds_wide - 1.0))
                    stake_raw = bankroll * kelly_f * cfg["kelly_scale"]
                    stake = _floor_to_unit(stake_raw, cfg["min_yen_unit"])
                    cap = _floor_to_unit(bankroll * cfg["max_ticket_share"], cfg["min_yen_unit"])
                    if cap > 0:
                        stake = min(stake, cap)
                    if stake >= cfg["min_yen_unit"]:
                        priority = _calc_priority_score(edge, p_wide, odds_wide)
                        race_candidates.append(
                            BetCandidate(
                                "wide",
                                (x, y),
                                float(odds_wide),
                                float(p_wide),
                                float(edge),
                                float(kelly_f),
                                int(stake),
                                f"priority={priority:.6f}",
                            )
                        )

            p_q = min(1.0, pa * pb * cfg["C_quinella"])
            odds_q = _lookup_pair_odds(odds.get("quinella", {}), x, y, quinella_penalty)
            if odds_q > 1.0:
                edge = p_q * odds_q - 1.0
                if edge >= cfg["min_ev_per_ticket"] and p_q >= cfg["min_p_hit_threshold"]:
                    kelly_f = max(0.0, edge / (odds_q - 1.0))
                    stake_raw = bankroll * kelly_f * cfg["kelly_scale"]
                    stake = _floor_to_unit(stake_raw, cfg["min_yen_unit"])
                    cap = _floor_to_unit(bankroll * cfg["max_ticket_share"], cfg["min_yen_unit"])
                    if cap > 0:
                        stake = min(stake, cap)
                    if stake >= cfg["min_yen_unit"]:
                        priority = _calc_priority_score(edge, p_q, odds_q)
                        race_candidates.append(
                            BetCandidate(
                                "quinella",
                                (x, y),
                                float(odds_q),
                                float(p_q),
                                float(edge),
                                float(kelly_f),
                                int(stake),
                                f"priority={priority:.6f}",
                            )
                        )

        if not race_candidates:
            continue

        race_candidates.sort(
            key=lambda x: _calc_priority_score(x.edge, x.p_hit, x.odds_used),
            reverse=True,
        )
        max_tickets = int(cfg.get("max_tickets_per_race", 0))
        if max_tickets > 0 and len(race_candidates) > max_tickets:
            race_candidates = race_candidates[:max_tickets]

        race_cap = _floor_to_unit(bankroll * cfg["max_race_share"], cfg["min_yen_unit"])
        race_candidates = _scale_candidates_to_race_cap(
            candidates=race_candidates,
            race_cap=race_cap,
            min_unit=cfg["min_yen_unit"],
        )
        total_stake = int(sum(x.stake_yen for x in race_candidates))

        if total_stake <= 0:
            continue
        total_edge_weighted = float(sum(item.edge * item.stake_yen for item in race_candidates))
        if total_edge_weighted <= 0:
            continue

        all_items.extend(race_candidates)

    if not all_items:
        return BetPlanResult(items=[], mapping={})
    mapping = _build_mapping(all_items)
    return BetPlanResult(items=all_items, mapping=mapping)

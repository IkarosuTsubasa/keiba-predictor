import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class BetTicketV4:
    ticket_type: str
    horse_ids: Tuple[str, ...]
    p_hit: float
    odds: float
    ev: float
    stake: int
    score: float


DEFAULT_CONFIG = {
    "p_mix_w": 0.6,
    "rank_temperature": 1.0,
    "place_scale": 3.0,
    "wide_scale": 1.6,
    "quinella_scale": 1.6,
    "min_p_hit_win": 0.02,
    "min_p_hit_place": 0.05,
    "min_p_hit_pair": 0.03,
    "min_ev": -0.05,
    "p_place_blend_w": 0.5,
    "p_place_cap": 0.75,
    "ensure_diversity": True,
    "min_yen_unit": 100,
    "race_budget_share": 0.02,
    "max_tickets_per_race": 3,
    "high_odds_threshold": 10.0,
    "max_high_odds_per_race": 1,
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


def _normalize_cfg(config: Dict) -> Dict:
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    cfg["p_mix_w"] = min(1.0, max(0.0, _safe_float(cfg.get("p_mix_w"), 0.6)))
    cfg["rank_temperature"] = max(1e-6, _safe_float(cfg.get("rank_temperature"), 1.0))
    cfg["place_scale"] = max(0.0, _safe_float(cfg.get("place_scale"), 3.0))
    cfg["wide_scale"] = max(0.0, _safe_float(cfg.get("wide_scale"), 1.6))
    cfg["quinella_scale"] = max(0.0, _safe_float(cfg.get("quinella_scale"), 1.6))
    cfg["min_p_hit_win"] = min(1.0, max(0.0, _safe_float(cfg.get("min_p_hit_win"), 0.02)))
    cfg["min_p_hit_place"] = min(1.0, max(0.0, _safe_float(cfg.get("min_p_hit_place"), 0.05)))
    cfg["min_p_hit_pair"] = min(1.0, max(0.0, _safe_float(cfg.get("min_p_hit_pair"), 0.03)))
    cfg["min_ev"] = _safe_float(cfg.get("min_ev"), -0.05)
    cfg["p_place_blend_w"] = min(1.0, max(0.0, _safe_float(cfg.get("p_place_blend_w"), 0.5)))
    cfg["p_place_cap"] = min(1.0, max(0.0, _safe_float(cfg.get("p_place_cap"), 0.75)))
    cfg["ensure_diversity"] = bool(cfg.get("ensure_diversity", True))
    cfg["min_yen_unit"] = max(1, _safe_int(cfg.get("min_yen_unit"), 100))
    cfg["race_budget_share"] = min(1.0, max(0.0, _safe_float(cfg.get("race_budget_share"), 0.02)))
    cfg["max_tickets_per_race"] = max(0, _safe_int(cfg.get("max_tickets_per_race"), 3))
    cfg["high_odds_threshold"] = max(1.01, _safe_float(cfg.get("high_odds_threshold"), 10.0))
    cfg["max_high_odds_per_race"] = max(0, _safe_int(cfg.get("max_high_odds_per_race"), 1))
    return cfg


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


def _build_mapping(items: List[BetTicketV4]) -> Dict[Tuple[str, ...], int]:
    out: Dict[Tuple[str, ...], int] = {}
    for item in items:
        key = tuple([item.ticket_type] + list(item.horse_ids))
        out[key] = out.get(key, 0) + int(item.stake)
    return out


def _to_summary(items: List[BetTicketV4], bankroll: int) -> Dict:
    total_stake = int(sum(int(x.stake) for x in items))
    expected_profit = float(sum(float(x.stake) * float(x.ev) for x in items))
    return {
        "bankroll_yen": int(bankroll),
        "ticket_count": int(len(items)),
        "total_stake": int(total_stake),
        "expected_profit": float(expected_profit),
        "no_bet": bool(len(items) == 0),
        "risk_share_used": (float(total_stake) / float(bankroll)) if bankroll > 0 else 0.0,
    }


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _derive_place_prob(top3_raw: float, p_win: float, place_scale: float, blend_w: float, cap: float) -> float:
    top3_clipped = _clip01(_safe_float(top3_raw, 0.0))
    model_place = _clip01(float(p_win) * float(place_scale))
    p_place_raw = float(blend_w) * top3_clipped + (1.0 - float(blend_w)) * model_place
    return float(min(float(cap), _clip01(p_place_raw)))


def _candidate_score(ev: float, p_hit: float) -> float:
    return float(ev) * math.sqrt(max(0.0, float(p_hit)))


def _ticket_group(ticket_type: str) -> str:
    t = str(ticket_type).strip().lower()
    if t == "win":
        return "win"
    if t == "place":
        return "place"
    if t in ("wide", "quinella"):
        return "pair"
    return "other"


def _select_candidates(ordered: List[BetTicketV4], cfg: Dict) -> List[BetTicketV4]:
    max_tickets = int(cfg["max_tickets_per_race"])
    if max_tickets <= 0 or not ordered:
        return []

    high_threshold = float(cfg["high_odds_threshold"])
    max_high = int(cfg["max_high_odds_per_race"])
    ensure_diversity = bool(cfg.get("ensure_diversity", True))

    selected: List[BetTicketV4] = []
    selected_keys = set()
    high_count = 0

    def _try_add(cand: BetTicketV4) -> bool:
        nonlocal high_count
        if len(selected) >= max_tickets:
            return False
        key = (str(cand.ticket_type), tuple(cand.horse_ids))
        if key in selected_keys:
            return False
        is_high = float(cand.odds) >= high_threshold
        if is_high and high_count >= max_high:
            return False
        selected.append(cand)
        selected_keys.add(key)
        if is_high:
            high_count += 1
        return True

    if ensure_diversity:
        for grp in ("win", "place", "pair"):
            for cand in ordered:
                if _ticket_group(cand.ticket_type) != grp:
                    continue
                if _try_add(cand):
                    break
                # 无论是否添加成功，都继续找下一个同组候选，直到可加或组内耗尽

    for cand in ordered:
        if len(selected) >= max_tickets:
            break
        _try_add(cand)

    return selected


def _allocate_stakes_discrete(selected: List[BetTicketV4], race_budget: float, cfg: Dict) -> int:
    if not selected:
        return 0
    min_unit = max(1, int(cfg.get("min_yen_unit", 100)))
    budget_int = max(0, int(math.floor(float(race_budget))))
    if budget_int <= 0:
        for cand in selected:
            cand.stake = 0
        return 0

    weights = _softmax([x.score for x in selected])
    rounded = []
    for idx, _cand in enumerate(selected):
        raw = float(race_budget) * float(weights[idx])
        stake_round = int(math.floor(raw / float(min_unit)) * min_unit)
        rounded.append(max(0, stake_round))

    sum_stake = int(sum(rounded))
    if sum_stake <= 0 and budget_int >= min_unit:
        best_idx = max(range(len(selected)), key=lambda i: float(selected[i].score))
        rounded[best_idx] = min_unit
        sum_stake = min_unit

    if sum_stake > budget_int:
        low_to_high = sorted(range(len(selected)), key=lambda i: float(selected[i].score))
        guard = 0
        while sum_stake > budget_int and guard < 10000:
            changed = False
            for i in low_to_high:
                if sum_stake <= budget_int:
                    break
                if rounded[i] >= min_unit:
                    rounded[i] -= min_unit
                    sum_stake -= min_unit
                    changed = True
            if not changed:
                break
            guard += 1

    for i, cand in enumerate(selected):
        cand.stake = int(max(0, rounded[i]))
    return int(sum_stake)


def _to_output_rows(items: List[BetTicketV4]) -> List[Dict]:
    rows = []
    for x in items:
        rows.append(
            {
                "ticket_type": str(x.ticket_type),
                "horse_ids": list(x.horse_ids),
                "p_hit": float(x.p_hit),
                "odds": float(x.odds),
                "EV": float(x.ev),
                "stake": int(x.stake),
                "score": float(x.score),
            }
        )
    return rows


def generate_bet_plan_v4(
    pred_df,
    odds: Dict,
    bankroll_yen: int,
    scope_key: str = "",
    config: Dict = None,
):
    _ = scope_key
    bankroll = max(0, int(bankroll_yen or 0))
    if bankroll <= 0:
        return [], {}, _to_summary([], bankroll=0)
    if pred_df is None or len(pred_df) == 0:
        return [], {}, _to_summary([], bankroll=bankroll)

    cfg = _normalize_cfg(config or {})
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
    work["Top3Prob_model"] = pd.to_numeric(work["Top3Prob_model"], errors="coerce")

    work = work[(work["horse_key"] != "") & work["rank_score"].notna()].copy()
    if work.empty:
        return [], {}, _to_summary([], bankroll=bankroll)

    all_items: List[BetTicketV4] = []
    race_diagnostics: List[Dict] = []
    race_budget = float(bankroll) * float(cfg["race_budget_share"])

    for _, g in work.groupby("race_id", sort=False):
        g = g.copy()
        scores = g["rank_score"].to_numpy(dtype=float)
        mu = float(np.mean(scores))
        std = float(np.std(scores))
        if (not np.isfinite(std)) or std < 1e-12:
            z = np.zeros_like(scores, dtype=float)
        else:
            z = (scores - mu) / std

        p_rank = _softmax(float(cfg["rank_temperature"]) * z)
        p_top3_norm = _normalize_prob(g["Top3Prob_model"].fillna(0.0).to_numpy(dtype=float))
        p_mix = float(cfg["p_mix_w"]) * p_rank + (1.0 - float(cfg["p_mix_w"])) * p_top3_norm
        p_win = _normalize_prob(p_mix)

        g["p_win"] = p_win
        g["p_place"] = [
            _derive_place_prob(
                t,
                p,
                cfg["place_scale"],
                cfg["p_place_blend_w"],
                cfg["p_place_cap"],
            )
            for t, p in zip(g["Top3Prob_model"].to_numpy(dtype=float), g["p_win"].to_numpy(dtype=float))
        ]

        horse_ids = g["horse_key"].astype(str).tolist()
        p_win_map = dict(zip(g["horse_key"].astype(str), g["p_win"].astype(float)))
        p_place_map = dict(zip(g["horse_key"].astype(str), g["p_place"].astype(float)))

        race_candidates: List[BetTicketV4] = []

        for hk in horse_ids:
            p_win_hit = min(1.0, max(0.0, float(p_win_map.get(hk, 0.0))))
            p_place_hit = min(1.0, max(0.0, float(p_place_map.get(hk, 0.0))))

            win_odds = _lookup_single_odds(odds.get("win", {}), hk)
            if win_odds > 1.0:
                ev = p_win_hit * win_odds - 1.0
                if p_win_hit > float(cfg["min_p_hit_win"]) and ev > float(cfg["min_ev"]):
                    race_candidates.append(
                        BetTicketV4(
                            ticket_type="win",
                            horse_ids=(str(hk),),
                            p_hit=float(p_win_hit),
                            odds=float(win_odds),
                            ev=float(ev),
                            stake=0,
                            score=_candidate_score(ev, p_win_hit),
                        )
                    )

            place_odds = _lookup_single_odds(odds.get("place", {}), hk)
            if place_odds > 1.0:
                ev = p_place_hit * place_odds - 1.0
                if p_place_hit > float(cfg["min_p_hit_place"]) and ev > float(cfg["min_ev"]):
                    race_candidates.append(
                        BetTicketV4(
                            ticket_type="place",
                            horse_ids=(str(hk),),
                            p_hit=float(p_place_hit),
                            odds=float(place_odds),
                            ev=float(ev),
                            stake=0,
                            score=_candidate_score(ev, p_place_hit),
                        )
                    )

        for a, b in combinations(horse_ids, 2):
            x, y = _sort_pair(a, b)
            pa_win = float(p_win_map.get(x, 0.0))
            pb_win = float(p_win_map.get(y, 0.0))
            pa_place = float(p_place_map.get(x, 0.0))
            pb_place = float(p_place_map.get(y, 0.0))

            p_wide = min(1.0, max(0.0, pa_place * pb_place * float(cfg["wide_scale"])))
            wide_odds = _lookup_pair_odds(odds.get("wide", {}), x, y)
            if wide_odds > 1.0:
                ev = p_wide * wide_odds - 1.0
                if p_wide > float(cfg["min_p_hit_pair"]) and ev > float(cfg["min_ev"]):
                    race_candidates.append(
                        BetTicketV4(
                            ticket_type="wide",
                            horse_ids=(x, y),
                            p_hit=float(p_wide),
                            odds=float(wide_odds),
                            ev=float(ev),
                            stake=0,
                            score=_candidate_score(ev, p_wide),
                        )
                    )

            p_quinella = min(1.0, max(0.0, pa_win * pb_win * float(cfg["quinella_scale"])))
            quinella_odds = _lookup_pair_odds(odds.get("quinella", {}), x, y)
            if quinella_odds > 1.0:
                ev = p_quinella * quinella_odds - 1.0
                if p_quinella > float(cfg["min_p_hit_pair"]) and ev > float(cfg["min_ev"]):
                    race_candidates.append(
                        BetTicketV4(
                            ticket_type="quinella",
                            horse_ids=(x, y),
                            p_hit=float(p_quinella),
                            odds=float(quinella_odds),
                            ev=float(ev),
                            stake=0,
                            score=_candidate_score(ev, p_quinella),
                        )
                    )

        candidate_count_by_type = {"win": 0, "place": 0, "wide": 0, "quinella": 0}
        for cand in race_candidates:
            t = str(cand.ticket_type).strip().lower()
            if t in candidate_count_by_type:
                candidate_count_by_type[t] += 1

        if not race_candidates:
            race_diagnostics.append(
                {
                    "race_id": str(g["race_id"].iloc[0]) if "race_id" in g.columns and len(g) > 0 else "__single_race__",
                    "candidate_count_by_type": candidate_count_by_type,
                    "selected_types": [],
                    "rounded_stake_sum": 0,
                    "race_budget": int(math.floor(float(race_budget))),
                }
            )
            continue

        ordered = sorted(race_candidates, key=lambda x: float(x.score), reverse=True)
        selected = _select_candidates(ordered, cfg)

        if not selected:
            race_diagnostics.append(
                {
                    "race_id": str(g["race_id"].iloc[0]) if "race_id" in g.columns and len(g) > 0 else "__single_race__",
                    "candidate_count_by_type": candidate_count_by_type,
                    "selected_types": [],
                    "rounded_stake_sum": 0,
                    "race_budget": int(math.floor(float(race_budget))),
                }
            )
            continue

        rounded_stake_sum = _allocate_stakes_discrete(selected, race_budget, cfg)
        race_diagnostics.append(
            {
                "race_id": str(g["race_id"].iloc[0]) if "race_id" in g.columns and len(g) > 0 else "__single_race__",
                "candidate_count_by_type": candidate_count_by_type,
                "selected_types": [str(x.ticket_type) for x in selected],
                "rounded_stake_sum": int(rounded_stake_sum),
                "race_budget": int(math.floor(float(race_budget))),
            }
        )

        all_items.extend(selected)

    output_rows = _to_output_rows(all_items)
    mapping = _build_mapping(all_items)
    summary = _to_summary(all_items, bankroll=bankroll)
    summary["race_diagnostics"] = race_diagnostics
    return output_rows, mapping, summary

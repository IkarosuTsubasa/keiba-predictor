"""
bet_engine_v6: scenario-driven betting engine.

v5 の問題点:
  - パラメータ 50+ 個でチューニング困難
  - 全組み合わせ候補を生成 → フィルタ → Kelly という機械的フロー
  - 結果が「複勝1点だけ」か「何でも買う」の両極端になりがち

v6 の設計思想:
  人間の馬券師が考えるように「レースを読んで → シナリオを決めて → 馬券を組む」

  1) レースシナリオ判定
     確率分布とオッズから 5 パターンに分類:
     - honmei:   本命決着 (1頭が抜けている)
     - nitoujiku: 二頭軸 (上位2頭が抜けている)
     - konsen:   混戦 (3頭以上が接戦)
     - ana:      穴狙い (モデルと市場の乖離が大きい馬がいる)
     - miokuri:  見送り (バリューなし)

  2) シナリオごとの買い方テンプレート
     各シナリオに人間的な馬券構成を定義:
     - honmei  → 本命の単勝 + 複勝、相手とのワイド
     - nitoujiku → 2頭の複勝 + 2頭のワイド
     - konsen  → 上位3頭の複勝 (バリューある馬のみ)
     - ana     → 穴馬の単勝 + 複勝
     - miokuri → 買わない

  3) バリューフィルタ
     テンプレートの各候補に対し EV > 0 のものだけ残す

  4) ステーキング
     シンプルな fractional Kelly、最大4-5点
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "enabled": True,
    # scenario thresholds
    "honmei_gap": 0.08,         # top1 - top2 gap to be "honmei"
    "nitoujiku_gap": 0.05,      # top2 - top3 gap to be "nitoujiku"
    "ana_value_ratio": 0.30,    # model/market divergence to trigger "ana"
    "min_ev_to_bet": 0.0,       # minimum EV to include a ticket
    # kelly
    "kelly_fraction": 0.25,     # fractional Kelly multiplier
    "kelly_cap": 0.15,          # max fraction of race budget per ticket
    # budget
    "target_risk_share": 0.05,  # fraction of bankroll per race
    "min_race_budget": 500,
    "max_race_budget": 5000,
    "min_yen_unit": 100,
    # limits
    "max_tickets": 5,
    "odds_max": 150.0,          # ignore odds above this
    "confidence_budget_scale": True,  # scale budget by confidence
}


def _sf(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _normalize(arr):
    a = np.asarray(arr, dtype=float)
    s = a.sum()
    if s > 0 and np.isfinite(s):
        return a / s
    return np.full_like(a, 1.0 / max(1, len(a)))


# ---------------------------------------------------------------------------
# Scenario detection
# ---------------------------------------------------------------------------

def _detect_scenario(probs, odds_win, horse_ids, cfg):
    """
    確率分布とオッズからレースシナリオを判定する。

    Returns: (scenario_name, context_dict)
    """
    n = len(probs)
    if n == 0:
        return "miokuri", {}

    sorted_idx = np.argsort(probs)[::-1]
    p_sorted = probs[sorted_idx]

    gap_12 = float(p_sorted[0] - p_sorted[1]) if n >= 2 else float(p_sorted[0])
    gap_23 = float(p_sorted[1] - p_sorted[2]) if n >= 3 else 0.0

    # Market implied probabilities (overround-normalized)
    q_raw = {}
    for hid in horse_ids:
        o = _sf(odds_win.get(str(hid), 0))
        q_raw[str(hid)] = (1.0 / o) if o > 1.0 else 0.0
    q_sum = sum(q_raw.values())
    q_norm = {k: v / q_sum if q_sum > 0 else 0.0 for k, v in q_raw.items()}

    # Value ratio per horse: (model_prob / market_prob) - 1
    value_ratios = {}
    for i, hid in enumerate(horse_ids):
        q = q_norm.get(str(hid), 0.0)
        if q > 0.01:
            value_ratios[str(hid)] = (probs[i] / q) - 1.0
        else:
            value_ratios[str(hid)] = 0.0

    best_value_hid = max(value_ratios, key=value_ratios.get) if value_ratios else None
    best_value = value_ratios.get(best_value_hid, 0.0) if best_value_hid else 0.0

    ctx = {
        "sorted_idx": sorted_idx,
        "gap_12": gap_12,
        "gap_23": gap_23,
        "q_norm": q_norm,
        "value_ratios": value_ratios,
        "best_value_hid": best_value_hid,
        "best_value": best_value,
    }

    honmei_gap = _sf(cfg.get("honmei_gap", 0.08))
    nitoujiku_gap = _sf(cfg.get("nitoujiku_gap", 0.05))
    ana_ratio = _sf(cfg.get("ana_value_ratio", 0.30))

    # 穴狙い: モデルが市場と大きく乖離している馬がいる
    # (ただし本命が明確な場合は本命シナリオ優先)
    if best_value >= ana_ratio and gap_12 < honmei_gap:
        # 穴馬が上位3頭に入っていなければ ana シナリオ
        top3_hids = set(str(horse_ids[sorted_idx[i]]) for i in range(min(3, n)))
        if best_value_hid not in top3_hids:
            return "ana", ctx

    if gap_12 >= honmei_gap:
        return "honmei", ctx

    if n >= 3 and gap_23 >= nitoujiku_gap:
        return "nitoujiku", ctx

    if n >= 3:
        return "konsen", ctx

    return "miokuri", ctx


# ---------------------------------------------------------------------------
# Ticket building per scenario
# ---------------------------------------------------------------------------

def _make_ticket(ticket_type, horse_ids_tuple, p_hit, odds_used, ev, why):
    return {
        "ticket_type": ticket_type,
        "bet_type": ticket_type,
        "horse_ids": list(horse_ids_tuple),
        "horses": list(horse_ids_tuple),
        "p_hit": p_hit,
        "p_final": p_hit,
        "odds_used": odds_used,
        "ev": ev,
        "EV": ev,
        "ev_adj": ev,
        "EV_adj": ev,
        "edge": ev,
        "score": ev * math.sqrt(max(p_hit, 1e-9)),
        "kelly_f": 0.0,
        "stake_yen": 0,
        "stake": 0,
        "value_signal": 0.0,
        "value_ratio": 0.0,
        "value_diff": 0.0,
        "why": why,
        "notes": why,
    }


def _ev(p, odds):
    return p * odds - 1.0


def _lookup_odds(odds_map, ticket_type, *horse_ids):
    """Unified odds lookup."""
    section = odds_map.get(ticket_type, {})
    if not isinstance(section, dict):
        return 0.0
    if len(horse_ids) == 1:
        raw = section.get(str(horse_ids[0]))
    else:
        a, b = str(horse_ids[0]), str(horse_ids[1])
        pair = (min(a, b), max(a, b)) if a != b else (a, b)
        raw = section.get(pair)
        if raw is None:
            raw = section.get((pair[1], pair[0]))
    if raw is None:
        return 0.0
    if isinstance(raw, dict):
        for k in ("low", "odds_low", "mid", "odds_mid", "odds", "value"):
            v = _sf(raw.get(k, 0))
            if v > 1.0:
                return v
        return 0.0
    return _sf(raw, 0.0)


def _build_honmei_tickets(probs, horse_ids, odds, ctx, cfg):
    """本命決着: 本命の単勝+複勝、相手とのワイド"""
    tickets = []
    si = ctx["sorted_idx"]
    top1 = str(horse_ids[si[0]])
    p1 = float(probs[si[0]])
    min_ev = _sf(cfg.get("min_ev_to_bet", 0.0))
    odds_max = _sf(cfg.get("odds_max", 150.0))

    # 単勝: 本命
    o_win = _lookup_odds(odds, "win", top1)
    if 1.0 < o_win <= odds_max:
        ev = _ev(p1 / 3.0, o_win)  # p_win ~= p_top3 / 3 (rough)
        # Use more accurate estimate: normalize top probs
        p_win_est = p1 / float(np.sum(probs))  # already normalized
        ev = _ev(p_win_est, o_win)
        if ev >= min_ev:
            tickets.append(_make_ticket(
                "win", (top1,), p_win_est, o_win, ev,
                f"honmei: {top1} win (p={p_win_est:.3f}, odds={o_win:.1f})",
            ))

    # 複勝: 本命
    o_place = _lookup_odds(odds, "place", top1)
    if 1.0 < o_place <= odds_max:
        ev = _ev(p1, o_place)
        if ev >= min_ev:
            tickets.append(_make_ticket(
                "place", (top1,), p1, o_place, ev,
                f"honmei: {top1} place (p={p1:.3f}, odds={o_place:.1f})",
            ))

    # ワイド: 本命 x 2番手/3番手
    n = len(probs)
    for rank in range(1, min(3, n)):
        partner = str(horse_ids[si[rank]])
        p_partner = float(probs[si[rank]])
        o_wide = _lookup_odds(odds, "wide", top1, partner)
        if 1.0 < o_wide <= odds_max:
            p_both = p1 * p_partner
            # P(both in top3) ~= p1 * p_partner * scale
            # For a rough estimate: if both are top3 candidates with high prob
            p_wide_est = min(0.8, p1 * p_partner * 3.0)
            ev = _ev(p_wide_est, o_wide)
            if ev >= min_ev:
                tickets.append(_make_ticket(
                    "wide", (min(top1, partner), max(top1, partner)),
                    p_wide_est, o_wide, ev,
                    f"honmei: {top1}-{partner} wide (p={p_wide_est:.3f}, odds={o_wide:.1f})",
                ))

    return tickets


def _build_nitoujiku_tickets(probs, horse_ids, odds, ctx, cfg):
    """二頭軸: 2頭の複勝 + ワイド"""
    tickets = []
    si = ctx["sorted_idx"]
    top1 = str(horse_ids[si[0]])
    top2 = str(horse_ids[si[1]])
    p1 = float(probs[si[0]])
    p2 = float(probs[si[1]])
    min_ev = _sf(cfg.get("min_ev_to_bet", 0.0))
    odds_max = _sf(cfg.get("odds_max", 150.0))

    # 複勝 x 2
    for hid, p in [(top1, p1), (top2, p2)]:
        o_place = _lookup_odds(odds, "place", hid)
        if 1.0 < o_place <= odds_max:
            ev = _ev(p, o_place)
            if ev >= min_ev:
                tickets.append(_make_ticket(
                    "place", (hid,), p, o_place, ev,
                    f"nitoujiku: {hid} place (p={p:.3f}, odds={o_place:.1f})",
                ))

    # ワイド: 1-2
    a, b = (min(top1, top2), max(top1, top2))
    o_wide = _lookup_odds(odds, "wide", a, b)
    if 1.0 < o_wide <= odds_max:
        p_wide_est = min(0.8, p1 * p2 * 3.0)
        ev = _ev(p_wide_est, o_wide)
        if ev >= min_ev:
            tickets.append(_make_ticket(
                "wide", (a, b), p_wide_est, o_wide, ev,
                f"nitoujiku: {a}-{b} wide (p={p_wide_est:.3f}, odds={o_wide:.1f})",
            ))

    # 3番手とのワイドも候補 (top1 x top3, top2 x top3)
    n = len(probs)
    if n >= 3:
        top3 = str(horse_ids[si[2]])
        p3 = float(probs[si[2]])
        for anchor, pa in [(top1, p1), (top2, p2)]:
            a2, b2 = (min(anchor, top3), max(anchor, top3))
            o_w = _lookup_odds(odds, "wide", a2, b2)
            if 1.0 < o_w <= odds_max:
                p_est = min(0.8, pa * p3 * 3.0)
                ev = _ev(p_est, o_w)
                if ev >= min_ev:
                    tickets.append(_make_ticket(
                        "wide", (a2, b2), p_est, o_w, ev,
                        f"nitoujiku: {a2}-{b2} wide (p={p_est:.3f}, odds={o_w:.1f})",
                    ))

    return tickets


def _build_konsen_tickets(probs, horse_ids, odds, ctx, cfg):
    """混戦: 上位3頭の複勝をバリューで厳選"""
    tickets = []
    si = ctx["sorted_idx"]
    n = min(4, len(probs))  # 上位4頭まで見る
    min_ev = _sf(cfg.get("min_ev_to_bet", 0.0))
    odds_max = _sf(cfg.get("odds_max", 150.0))
    vr = ctx.get("value_ratios", {})

    # 複勝: バリューがある馬だけ
    for rank in range(n):
        hid = str(horse_ids[si[rank]])
        p = float(probs[si[rank]])
        o_place = _lookup_odds(odds, "place", hid)
        if 1.0 < o_place <= odds_max:
            ev = _ev(p, o_place)
            v = vr.get(hid, 0.0)
            if ev >= min_ev and v >= 0.0:
                tickets.append(_make_ticket(
                    "place", (hid,), p, o_place, ev,
                    f"konsen: {hid} place (p={p:.3f}, odds={o_place:.1f}, value={v:.2f})",
                ))

    # 上位2頭のワイド (混戦でも最も当たりやすい組み合わせ)
    if n >= 2:
        h1, h2 = str(horse_ids[si[0]]), str(horse_ids[si[1]])
        a, b = (min(h1, h2), max(h1, h2))
        o_wide = _lookup_odds(odds, "wide", a, b)
        if 1.0 < o_wide <= odds_max:
            p1, p2 = float(probs[si[0]]), float(probs[si[1]])
            p_est = min(0.8, p1 * p2 * 3.0)
            ev = _ev(p_est, o_wide)
            if ev >= min_ev:
                tickets.append(_make_ticket(
                    "wide", (a, b), p_est, o_wide, ev,
                    f"konsen: {a}-{b} wide (p={p_est:.3f}, odds={o_wide:.1f})",
                ))

    return tickets


def _build_ana_tickets(probs, horse_ids, odds, ctx, cfg):
    """穴狙い: モデルが過小評価されていると判断した馬"""
    tickets = []
    hid = ctx.get("best_value_hid")
    if not hid:
        return tickets

    idx = None
    for i, h in enumerate(horse_ids):
        if str(h) == str(hid):
            idx = i
            break
    if idx is None:
        return tickets

    p = float(probs[idx])
    min_ev = _sf(cfg.get("min_ev_to_bet", 0.0))
    odds_max = _sf(cfg.get("odds_max", 150.0))
    v = ctx.get("best_value", 0.0)

    # 単勝
    p_win_est = p / float(np.sum(probs))
    o_win = _lookup_odds(odds, "win", hid)
    if 1.0 < o_win <= odds_max:
        ev = _ev(p_win_est, o_win)
        if ev >= min_ev:
            tickets.append(_make_ticket(
                "win", (str(hid),), p_win_est, o_win, ev,
                f"ana: {hid} win (p={p_win_est:.3f}, odds={o_win:.1f}, value={v:.2f})",
            ))

    # 複勝
    o_place = _lookup_odds(odds, "place", hid)
    if 1.0 < o_place <= odds_max:
        ev = _ev(p, o_place)
        if ev >= min_ev:
            tickets.append(_make_ticket(
                "place", (str(hid),), p, o_place, ev,
                f"ana: {hid} place (p={p:.3f}, odds={o_place:.1f}, value={v:.2f})",
            ))

    # 穴馬 x 上位馬のワイド
    si = ctx["sorted_idx"]
    for rank in range(min(2, len(probs))):
        partner = str(horse_ids[si[rank]])
        if partner == str(hid):
            continue
        p_partner = float(probs[si[rank]])
        a, b = (min(str(hid), partner), max(str(hid), partner))
        o_wide = _lookup_odds(odds, "wide", a, b)
        if 1.0 < o_wide <= odds_max:
            p_est = min(0.8, p * p_partner * 3.0)
            ev = _ev(p_est, o_wide)
            if ev >= min_ev:
                tickets.append(_make_ticket(
                    "wide", (a, b), p_est, o_wide, ev,
                    f"ana: {a}-{b} wide (p={p_est:.3f}, odds={o_wide:.1f})",
                ))

    return tickets


SCENARIO_BUILDERS = {
    "honmei": _build_honmei_tickets,
    "nitoujiku": _build_nitoujiku_tickets,
    "konsen": _build_konsen_tickets,
    "ana": _build_ana_tickets,
}


# ---------------------------------------------------------------------------
# Staking (simple Kelly)
# ---------------------------------------------------------------------------

def _allocate_stakes(tickets, race_budget, cfg):
    """Fractional Kelly staking。"""
    if not tickets or race_budget <= 0:
        return tickets

    kelly_frac = _sf(cfg.get("kelly_fraction", 0.25))
    kelly_cap = _sf(cfg.get("kelly_cap", 0.15))
    min_unit = max(1, int(cfg.get("min_yen_unit", 100)))

    # Kelly fraction: f = (p * odds - 1) / (odds - 1)
    raw_f = []
    for t in tickets:
        odds = max(1.01, _sf(t["odds_used"]))
        p = _sf(t["p_hit"])
        edge = p * odds - 1.0
        if edge > 0 and odds > 1.01:
            f = kelly_frac * edge / (odds - 1.0)
            f = min(f, kelly_cap)
        else:
            f = 0.0
        raw_f.append(f)
        t["kelly_f"] = f

    total_f = sum(raw_f)
    if total_f <= 0:
        # fallback: equal split on positive-EV tickets
        pos_ev = [i for i, t in enumerate(tickets) if _sf(t["ev"]) > 0]
        if not pos_ev:
            return tickets
        per_ticket = max(min_unit, int(race_budget / len(pos_ev) / min_unit) * min_unit)
        for i in pos_ev:
            tickets[i]["stake_yen"] = min(per_ticket, race_budget)
            tickets[i]["stake"] = tickets[i]["stake_yen"]
        return tickets

    for i, t in enumerate(tickets):
        raw_stake = race_budget * (raw_f[i] / total_f)
        stake = int(raw_stake / min_unit) * min_unit
        stake = max(0, min(stake, race_budget))
        t["stake_yen"] = stake
        t["stake"] = stake

    # Ensure at least min_unit on best ticket
    if all(t["stake_yen"] == 0 for t in tickets):
        best = max(range(len(tickets)), key=lambda i: _sf(tickets[i]["score"]))
        tickets[best]["stake_yen"] = min_unit
        tickets[best]["stake"] = min_unit

    return tickets


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_bet_plan_v6(
    pred_df,
    odds: Dict,
    bankroll_yen: int,
    scope_key: str = "",
    config: Dict = None,
):
    """
    v6 bet engine entry point.
    Same interface as v5 for drop-in compatibility.

    Returns: (items_list, mapping_dict, summary_dict)
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})

    bankroll = max(0, int(bankroll_yen or 0))
    if not cfg.get("enabled", True) or bankroll <= 0:
        return [], {}, _empty_summary(bankroll)
    if pred_df is None or len(pred_df) == 0:
        return [], {}, _empty_summary(bankroll)

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
    work["race_id"] = work["race_id"].fillna("__single_race__").astype(str)
    work["Top3Prob_model"] = pd.to_numeric(work["Top3Prob_model"], errors="coerce").fillna(0.0)
    work = work[work["horse_key"] != ""].copy()
    if work.empty:
        return [], {}, _empty_summary(bankroll)

    all_items = []
    race_diags = []

    for _, g in work.groupby("race_id", sort=False):
        g = g.copy()
        race_id = str(g["race_id"].iloc[0])
        horse_ids = g["horse_key"].astype(str).tolist()
        probs = g["Top3Prob_model"].to_numpy(dtype=float)
        probs = np.clip(probs, 0.0, 1.0)

        # confidence from predictor (if available)
        conf = _sf(g["confidence_score"].iloc[0], 0.5) if "confidence_score" in g.columns else 0.5

        # Odds for win (for scenario detection)
        odds_win = {}
        win_section = odds.get("win", {})
        if isinstance(win_section, dict):
            for hid in horse_ids:
                o = _lookup_odds(odds, "win", hid)
                if o > 1.0:
                    odds_win[hid] = o

        # 1) Scenario detection
        scenario, ctx = _detect_scenario(probs, odds_win, horse_ids, cfg)

        # 2) Build tickets from scenario template
        builder = SCENARIO_BUILDERS.get(scenario)
        if builder is None:
            tickets = []
        else:
            tickets = builder(probs, horse_ids, odds, ctx, cfg)

        # 3) Sort by score, limit count
        max_tickets = int(cfg.get("max_tickets", 5))
        tickets.sort(key=lambda t: _sf(t["score"]), reverse=True)
        tickets = tickets[:max_tickets]

        # 4) Race budget
        risk_share = _sf(cfg.get("target_risk_share", 0.05))
        min_budget = int(cfg.get("min_race_budget", 500))
        max_budget = int(cfg.get("max_race_budget", 5000))
        budget_raw = bankroll * risk_share
        if cfg.get("confidence_budget_scale", True):
            budget_raw *= _clip(0.5 + conf, 0.3, 1.5)
        race_budget = int(_clip(budget_raw, min_budget, min(max_budget, bankroll)))

        # 5) Staking
        tickets = _allocate_stakes(tickets, race_budget, cfg)
        tickets = [t for t in tickets if t["stake_yen"] > 0]

        # Diagnostics
        diag = {
            "race_id": race_id,
            "scenario": scenario,
            "gap_12": ctx.get("gap_12", 0.0),
            "gap_23": ctx.get("gap_23", 0.0),
            "best_value": ctx.get("best_value", 0.0),
            "best_value_hid": ctx.get("best_value_hid", ""),
            "confidence": conf,
            "race_budget": race_budget,
            "ticket_count": len(tickets),
            "tickets": [
                {"type": t["ticket_type"], "horses": t["horses"],
                 "stake": t["stake_yen"], "ev": round(t["ev"], 4),
                 "why": t["why"]}
                for t in tickets
            ],
        }
        race_diags.append(diag)

        # Add scenario info to each ticket for downstream display
        for t in tickets:
            t["notes"] = f"[{scenario}] {t.get('why', '')}"
            t["strategy_text_ja"] = _scenario_text_ja(scenario)
            t["bet_tendency_ja"] = _tendency_text_ja(scenario, tickets)

        all_items.extend(tickets)

    mapping = _build_mapping(all_items)
    summary = _build_summary(all_items, bankroll, race_diags)
    return all_items, mapping, summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scenario_text_ja(scenario):
    return {
        "honmei": "本命決着想定。単勝+複勝を軸にワイドで手広く。",
        "nitoujiku": "二頭軸想定。上位2頭の複勝とワイドで堅実に。",
        "konsen": "混戦模様。バリューのある馬の複勝に絞る。",
        "ana": "穴狙い。モデルが市場より高く評価する馬に集中。",
        "miokuri": "見送り。バリューが見つからない。",
    }.get(scenario, "")


def _tendency_text_ja(scenario, tickets):
    types = [t["ticket_type"] for t in tickets]
    win_count = types.count("win")
    place_count = types.count("place")
    wide_count = types.count("wide")
    parts = []
    if win_count:
        parts.append(f"単勝{win_count}点")
    if place_count:
        parts.append(f"複勝{place_count}点")
    if wide_count:
        parts.append(f"ワイド{wide_count}点")
    return "、".join(parts) if parts else "見送り"


def _build_mapping(items):
    out = {}
    for item in items:
        key = tuple([str(item.get("ticket_type", ""))] + [str(x) for x in item.get("horse_ids", ())])
        out[key] = out.get(key, 0) + int(item.get("stake_yen", 0))
    return out


def _empty_summary(bankroll):
    return {
        "bankroll_yen": int(bankroll),
        "ticket_count": 0,
        "total_stake_yen": 0,
        "expected_return_yen": 0,
        "expected_profit_yen": 0,
        "risk_share_used": 0.0,
        "no_bet": True,
        "strategy_text": "見送り",
        "diagnostics": [],
        "race_diagnostics": [],
        "calibration_meta": {},
    }


def _build_summary(items, bankroll, race_diags):
    total_stake = sum(int(t.get("stake_yen", 0)) for t in items)
    expected_return = sum(
        float(t.get("stake_yen", 0)) * float(t.get("p_hit", 0)) * float(t.get("odds_used", 0))
        for t in items
    )
    expected_profit = sum(
        float(t.get("stake_yen", 0)) * float(t.get("ev", 0))
        for t in items
    )
    scenarios = list(set(d.get("scenario", "") for d in race_diags))
    strategy = _scenario_text_ja(scenarios[0]) if len(scenarios) == 1 else "複数レース"

    return {
        "bankroll_yen": int(bankroll),
        "ticket_count": len(items),
        "total_stake_yen": int(total_stake),
        "expected_return_yen": int(round(expected_return)),
        "expected_profit_yen": int(round(expected_profit)),
        "risk_share_used": (float(total_stake) / float(bankroll)) if bankroll > 0 else 0.0,
        "no_bet": len(items) == 0,
        "strategy_text": strategy,
        "diagnostics": race_diags,
        "race_diagnostics": race_diags,
        "calibration_meta": {},
    }

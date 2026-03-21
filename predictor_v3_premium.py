#!/usr/bin/env python3
"""
predictor_v3_ultimate.py
========================
High-precision horse racing predictor focused on:
  - historical ability (time index, track-bias adjusted)
  - race-condition fit (venue/surface/distance/going)
  - pace/style suitability
  - market priors (win/place/combination odds)
  - automatic weight optimization from kachiuma backtest

Default target race:
  中山 / 芝 / 1800m / 稍重

Input files (default):
  - kachiuma.csv
  - shutuba.csv
  - odds.csv
  - fuku_odds.csv
  - wide_odds.csv
  - quinella_odds.csv

Output:
  - predictions.csv (pipeline-compatible)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


GOING_MAP = {"良": 0.0, "稍": 1.0, "稍重": 1.0, "重": 2.0, "不良": 3.0}
FEATURE_KEYS = [
    "ability",
    "recent",
    "trend",
    "peak",
    "dist_fit",
    "venue_fit",
    "going_fit",
    "baba_fit",
    "pace_fit",
    "consistency",
    "class_power",
    "freshness",
    "experience",
    "market_win",
    "market_place",
    "market_combo",
]

BASE_WEIGHTS = {
    "ability": 1.35,
    "recent": 0.80,
    "trend": 0.45,
    "peak": 0.30,
    "dist_fit": 1.00,
    "venue_fit": 0.80,
    "going_fit": 0.95,
    "baba_fit": 0.55,
    "pace_fit": 0.70,
    "consistency": 0.95,
    "class_power": 0.40,
    "freshness": 0.35,
    "experience": 0.20,
    "market_win": 1.15,
    "market_place": 0.45,
    "market_combo": 0.20,
}

SCOPE_ALIASES = {
    "central_turf": {"central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"},
    "central_dirt": {"central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"},
    "local": {"local", "l", "3"},
}


@dataclass
class RaceCondition:
    venue: str
    surface: str
    distance: int
    going: str
    expected_baba_index: float

    @property
    def going_level(self) -> float:
        return GOING_MAP.get(self.going, 1.0)


def to_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().replace(",", "").replace('"', "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def to_int(v: object) -> Optional[int]:
    f = to_float(v)
    if f is None:
        return None
    return int(round(f))


def parse_date(v: object) -> Optional[datetime]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def parse_surface_distance(v: object) -> Tuple[str, Optional[int]]:
    if v is None:
        return "", None
    s = str(v).strip()
    m = re.match(r"(芝|ダ|障)(\d+)", s)
    if not m:
        return "", None
    return m.group(1), int(m.group(2))


def parse_venue(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    m = re.search(r"\d*([^\d]+)\d*", s)
    if m:
        return m.group(1).strip()
    return s


def parse_positions(v: object) -> List[int]:
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    out: List[int] = []
    for part in re.split(r"[-\s]+", s):
        p = to_int(part)
        if p is not None:
            out.append(p)
    return out


def mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def stdev(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def exp_weighted_mean(values: Sequence[float], half_life: float = 2.5) -> float:
    if not values:
        return 0.0
    wsum = 0.0
    vsum = 0.0
    for i, v in enumerate(values):
        w = 0.5 ** (i / max(half_life, 1e-6))
        wsum += w
        vsum += w * v
    return vsum / wsum if wsum > 0 else 0.0


def logistic(x: float) -> float:
    x = max(min(x, 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-x))


def softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    t = max(temperature, 1e-6)
    mx = max(scores)
    ex = [math.exp((s - mx) / t) for s in scores]
    z = sum(ex)
    if z <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / z for e in ex]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def configure_utf8_io() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def normalize_scope_key(raw: str) -> str:
    val = (raw or "").strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    for key, aliases in SCOPE_ALIASES.items():
        if val in aliases:
            return key
    return ""


def resolve_scope_key() -> str:
    env_scope = normalize_scope_key(os.environ.get("SCOPE_KEY", ""))
    if env_scope:
        return env_scope
    return "central_dirt"


def map_surface_input(value: str, fallback: str) -> str:
    raw = (value or "").strip().lower()
    if raw in ("2", "d", "dirt", "ダ"):
        return "ダ"
    if raw in ("1", "s", "shiba", "turf", "grass", "芝"):
        return "芝"
    return fallback


def map_distance_input(value: str, fallback: int) -> int:
    raw = (value or "").strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def race_level_normalize(probs: Sequence[float], top_k: int = 3) -> List[float]:
    arr = [float(x) for x in probs]
    total = sum(arr)
    if total <= 0:
        return arr
    expected = min(float(top_k), float(len(arr)))
    scale = expected / total
    return [clamp(p * scale, 0.0, 0.98) for p in arr]


def resolve_race_condition(args: argparse.Namespace) -> RaceCondition:
    scope = resolve_scope_key()
    default_surface = args.race_surface
    if scope == "central_turf":
        default_surface = "芝"
    else:
        default_surface = "ダ"

    surface = default_surface
    distance = int(args.race_distance)
    going = str(args.race_going or "稍重")

    if not args.no_prompt:
        try:
            surface_raw = input("Surface 1=turf 2=dirt [auto]: ").strip()
        except EOFError:
            surface_raw = ""
        surface = map_surface_input(surface_raw, surface)

        try:
            distance_raw = input(f"Distance meters [{distance}]: ").strip()
        except EOFError:
            distance_raw = ""
        distance = map_distance_input(distance_raw, distance)

        try:
            going_raw = input(f"Track condition (良/稍重/重/不良) [{going}]: ").strip()
        except EOFError:
            going_raw = ""
        if going_raw:
            going = going_raw

    return RaceCondition(
        venue=args.race_venue,
        surface=surface,
        distance=distance,
        going=going,
        expected_baba_index=args.expected_baba_index,
    )


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        return [dict(r) for r in rd]


def group_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        out[str(r.get(key, ""))].append(r)
    return out


def _safe_ti_adj(row: Dict[str, str]) -> Optional[float]:
    ti = to_float(row.get("ﾀｲﾑ指数"))
    if ti is None:
        return None
    baba = to_float(row.get("馬場指数"))
    if baba is None:
        baba = 0.0
    return ti - baba


def summarize_history(rows: List[Dict[str, str]], cond: RaceCondition) -> Dict[str, float]:
    parsed: List[Tuple[datetime, Dict[str, str]]] = []
    for r in rows:
        d = parse_date(r.get("日付"))
        if d is not None:
            parsed.append((d, r))
    parsed.sort(key=lambda x: x[0], reverse=True)
    hist = [r for _, r in parsed]

    ti_adj_list: List[float] = []
    ti_adj_dist_fit: List[float] = []
    ti_adj_venue_fit: List[float] = []
    ti_adj_going_fit: List[float] = []
    ti_adj_baba_fit: List[float] = []
    finish_pct: List[float] = []
    top3_list: List[float] = []
    run_first_pct: List[float] = []
    run_gain: List[float] = []
    uphill: List[float] = []
    prize_ln: List[float] = []

    for r in hist:
        ti_adj = _safe_ti_adj(r)
        if ti_adj is None:
            continue

        surf, dist = parse_surface_distance(r.get("距離"))
        venue = parse_venue(r.get("開催"))
        going = GOING_MAP.get(str(r.get("馬場", "")).strip(), 1.0)
        baba = to_float(r.get("馬場指数"))
        if baba is None:
            baba = 0.0
        rank = to_int(r.get("着順"))
        field = to_int(r.get("頭数"))
        up = to_float(r.get("上り"))
        prize = to_float(r.get("賞金"))
        pos = parse_positions(r.get("通過"))

        ti_adj_list.append(ti_adj)

        # distance/surface fit (same surface + distance proximity)
        if surf == cond.surface and dist is not None:
            dist_w = math.exp(-abs(dist - cond.distance) / 260.0)
            ti_adj_dist_fit.append(ti_adj * dist_w)

        # venue fit
        if venue == cond.venue:
            ti_adj_venue_fit.append(ti_adj)

        # going fit
        going_w = 1.0 / (1.0 + abs(going - cond.going_level))
        ti_adj_going_fit.append(ti_adj * going_w)

        # baba fit
        baba_w = max(0.05, 1.0 - abs(baba - cond.expected_baba_index) / 24.0)
        ti_adj_baba_fit.append(ti_adj * baba_w)

        if rank is not None and field and field > 0:
            fp = rank / field
            finish_pct.append(fp)
            top3_list.append(1.0 if rank <= 3 else 0.0)

        if pos and field and field > 0:
            first_pct = pos[0] / field
            last_pct = pos[-1] / field
            run_first_pct.append(first_pct)
            run_gain.append(first_pct - last_pct)  # positive = made up positions

        if up is not None:
            uphill.append(up)
        if prize is not None:
            prize_ln.append(math.log1p(max(prize, 0.0)))

    last8 = ti_adj_list[:8]
    last6 = ti_adj_list[:6]
    recent3 = last6[:3]
    prev3 = last6[3:6]

    ability = exp_weighted_mean(last8, half_life=2.3)
    recent = mean(recent3) if recent3 else ability
    trend = (mean(recent3) - mean(prev3)) if recent3 and prev3 else 0.0
    peak = max(last8) if last8 else ability
    _PRIOR_K = 3  # shrink toward fallback when n < 3
    n_dist = len(ti_adj_dist_fit)
    _dist_prior = ability - 5.0
    dist_fit = (mean(ti_adj_dist_fit) * n_dist + _dist_prior * _PRIOR_K) / (n_dist + _PRIOR_K) if ti_adj_dist_fit else _dist_prior
    n_venue = len(ti_adj_venue_fit)
    _venue_prior = ability - 3.0
    venue_fit = (mean(ti_adj_venue_fit) * n_venue + _venue_prior * _PRIOR_K) / (n_venue + _PRIOR_K) if ti_adj_venue_fit else _venue_prior
    n_going = len(ti_adj_going_fit)
    going_fit = (mean(ti_adj_going_fit) * n_going + ability * _PRIOR_K) / (n_going + _PRIOR_K) if ti_adj_going_fit else ability
    n_baba = len(ti_adj_baba_fit)
    baba_fit = (mean(ti_adj_baba_fit) * n_baba + ability * _PRIOR_K) / (n_baba + _PRIOR_K) if ti_adj_baba_fit else ability

    style = mean(run_first_pct[:5]) if run_first_pct else 0.52
    gain = mean(run_gain[:5]) if run_gain else 0.0
    up_mean = mean(uphill[:5]) if uphill else 35.0

    # Nakayama 1800 turf, slightly heavy: slight advantage to tactical/forward and sustainable pace.
    target_style = 0.48 if cond.distance >= 1800 else 0.45
    style_score = 1.0 - abs(style - target_style) / 0.5
    style_score = clamp(style_score, 0.0, 1.0)
    gain_score = clamp((gain + 0.08) / 0.18, 0.0, 1.0)
    up_score = clamp((36.8 - up_mean) / 3.6, 0.0, 1.0)
    if cond.going_level >= 1.0:
        # softer ground: sustained pace gets a bit more weight than pure late kick
        pace_fit = 100.0 * (0.45 * style_score + 0.40 * gain_score + 0.15 * up_score)
    else:
        pace_fit = 100.0 * (0.30 * style_score + 0.25 * gain_score + 0.45 * up_score)

    _n_t3 = len(top3_list[:8])
    _T3_PRIOR, _T3_K = 0.33, 5
    t3_rate = (sum(top3_list[:8]) + _T3_PRIOR * _T3_K) / (_n_t3 + _T3_K) if top3_list else _T3_PRIOR
    fin_std = stdev(finish_pct[:8]) if finish_pct else 0.25
    consistency = 100.0 * t3_rate - 65.0 * fin_std
    class_power = 10.0 * mean(prize_ln[:6]) if prize_ln else 0.0

    freshness = 0.0
    if len(parsed) >= 1:
        d0 = parsed[0][0]
        d1 = parsed[1][0] if len(parsed) >= 2 else None
        if d1 is not None:
            rest_days = abs((d0 - d1).days)
            freshness = 100.0 * max(0.0, 1.0 - abs(rest_days - 42) / 84.0)
        else:
            freshness = 45.0

    experience = 18.0 * math.log1p(len(hist))

    return {
        "ability": ability,
        "recent": recent,
        "trend": trend,
        "peak": peak,
        "dist_fit": dist_fit,
        "venue_fit": venue_fit,
        "going_fit": going_fit,
        "baba_fit": baba_fit,
        "pace_fit": pace_fit,
        "consistency": consistency,
        "class_power": class_power,
        "freshness": freshness,
        "experience": experience,
        "recent_top3_rate": t3_rate,
        "runs_used": float(len(hist)),
    }


def market_from_row(row: Dict[str, str]) -> Dict[str, float]:
    odds = to_float(row.get("オッズ"))
    pop = to_int(row.get("人気"))
    win_imp = 1.0 / odds if odds and odds > 0 else 0.0
    pop_imp = 1.0 / pop if pop and pop > 0 else 0.0
    market_win = 0.75 * win_imp + 0.25 * pop_imp
    market_place = min(0.80, max(0.0, 1.65 * market_win))
    market_combo = min(0.95, market_place * 0.7)
    return {
        "market_win": market_win * 100.0,
        "market_place": market_place * 100.0,
        "market_combo": market_combo * 100.0,
        "market_odds": odds or 0.0,
        "market_place_odds": 0.0,
        "horse_no": float(to_int(row.get("馬番")) or 0),
    }


def build_market_maps(
    odds_rows: List[Dict[str, str]],
    fuku_rows: List[Dict[str, str]],
    wide_rows: List[Dict[str, str]],
    quinella_rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    by_name: Dict[str, Dict[str, float]] = {}
    horse_no_to_name: Dict[int, str] = {}

    for r in odds_rows:
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        hno = to_int(r.get("horse_no")) or 0
        odds = to_float(r.get("odds")) or 0.0
        imp = 1.0 / odds if odds > 0 else 0.0
        by_name[name] = {
            "market_win": imp * 100.0,
            "market_place": 0.0,
            "market_combo": 0.0,
            "market_odds": odds,
            "market_place_odds": 0.0,
            "horse_no": float(hno),
        }
        if hno > 0:
            horse_no_to_name[hno] = name

    place_imp_by_hno: Dict[int, float] = {}
    for r in fuku_rows:
        hno = to_int(r.get("horse_no"))
        if not hno:
            continue
        mid = to_float(r.get("odds_mid"))
        if mid and mid > 0:
            place_imp_by_hno[hno] = 1.0 / mid
            name = horse_no_to_name.get(hno)
            if name and name in by_name:
                by_name[name]["market_place"] = (1.0 / mid) * 100.0
                by_name[name]["market_place_odds"] = mid

    combo_support: Dict[int, float] = defaultdict(float)
    combo_count: Dict[int, int] = defaultdict(int)

    for src_rows, odd_key in ((wide_rows, "odds_mid"), (quinella_rows, "odds")):
        for r in src_rows:
            a = to_int(r.get("horse_no_a"))
            b = to_int(r.get("horse_no_b"))
            odd = to_float(r.get(odd_key))
            if not a or not b or not odd or odd <= 0:
                continue
            strength = 1.0 / odd
            combo_support[a] += strength
            combo_support[b] += strength
            combo_count[a] += 1
            combo_count[b] += 1

    max_combo = 0.0
    combo_imp: Dict[int, float] = {}
    for hno, s in combo_support.items():
        c = combo_count[hno]
        v = s / c if c > 0 else 0.0
        combo_imp[hno] = v
        if v > max_combo:
            max_combo = v

    if max_combo <= 0:
        max_combo = 1.0

    for hno, v in combo_imp.items():
        name = horse_no_to_name.get(hno)
        if not name or name not in by_name:
            continue
        by_name[name]["market_combo"] = (v / max_combo) * 100.0

        # If place odds missing, infer from win+combo.
        if by_name[name]["market_place"] <= 0:
            inferred_place = min(0.80, (by_name[name]["market_win"] / 100.0) * 1.6 + 0.20 * (v / max_combo))
            by_name[name]["market_place"] = inferred_place * 100.0

    for name, m in by_name.items():
        if m["market_place"] <= 0:
            m["market_place"] = min(90.0, m["market_win"] * 1.5)

    return by_name


def z_norm_features(runners: List[Dict[str, float]]) -> List[Dict[str, float]]:
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for k in FEATURE_KEYS:
        vals = [r.get(k, 0.0) for r in runners]
        means[k] = mean(vals)
        s = stdev(vals)
        stds[k] = s if s > 1e-9 else 1.0
    out: List[Dict[str, float]] = []
    for r in runners:
        z = {k: (r.get(k, 0.0) - means[k]) / stds[k] for k in FEATURE_KEYS}
        out.append(z)
    return out


def score_race(
    runners: List[Dict[str, float]],
    weights: Dict[str, float],
    temperature: float,
) -> Tuple[List[float], List[float], List[float]]:
    z = z_norm_features(runners)
    scores: List[float] = []
    for zf in z:
        s = 0.0
        for k in FEATURE_KEYS:
            s += weights[k] * zf[k]
        scores.append(s)

    win_probs = softmax(scores, temperature=temperature)
    n = len(runners)
    top3_probs = [clamp(p * min(3.0, float(n)), 0.0, 0.98) for p in win_probs]
    return scores, win_probs, top3_probs


def ndcg_at_k(labels_sorted: List[int], k: int = 3) -> float:
    if not labels_sorted:
        return 0.0
    k = min(k, len(labels_sorted))
    dcg = 0.0
    for i in range(k):
        dcg += labels_sorted[i] / math.log2(i + 2)
    ideal = sorted(labels_sorted, reverse=True)
    idcg = 0.0
    for i in range(k):
        idcg += ideal[i] / math.log2(i + 2)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def evaluate_params(
    races: List[Dict[str, object]],
    weights: Dict[str, float],
    temperature: float,
) -> float:
    if not races:
        return -1e9

    brier_all = 0.0
    ndcg_all = 0.0
    top3_precision_all = 0.0
    winner_hit_all = 0.0
    race_count = 0

    for race in races:
        runners = race["runners"]  # type: ignore[index]
        if len(runners) < 5:
            continue
        scores, _winp, top3p = score_race(runners, weights, temperature)

        idx = list(range(len(runners)))
        idx.sort(key=lambda i: scores[i], reverse=True)
        labels = [int(runners[i]["label"]) for i in range(len(runners))]

        brier = mean([(top3p[i] - labels[i]) ** 2 for i in range(len(runners))])
        brier_all += brier

        ranked_labels = [labels[i] for i in idx]
        ndcg_all += ndcg_at_k(ranked_labels, k=3)

        pred_top3 = idx[:3]
        precision = sum(labels[i] for i in pred_top3) / max(1, len(pred_top3))
        top3_precision_all += precision

        winner_hit_all += 1.0 if labels[idx[0]] == 1 else 0.0
        race_count += 1

    if race_count == 0:
        return -1e9

    brier_m = brier_all / race_count
    ndcg_m = ndcg_all / race_count
    top3_p = top3_precision_all / race_count
    winner_h = winner_hit_all / race_count

    l2 = 0.002 * sum(weights[k] ** 2 for k in FEATURE_KEYS)
    objective = (-1.0 * brier_m) + (0.35 * ndcg_m) + (0.25 * top3_p) + (0.12 * winner_h) - l2
    return objective


def optimize_weights(
    races: List[Dict[str, object]],
    iterations: int = 3500,
    seed: int = 20260307,
) -> Tuple[Dict[str, float], float, float]:
    rng = random.Random(seed)
    best_w = dict(BASE_WEIGHTS)
    best_t = 1.05
    best_obj = evaluate_params(races, best_w, best_t)

    for i in range(iterations):
        trial_w = {}
        temp_scale = 0.28 if i < iterations * 0.7 else 0.12
        for k in FEATURE_KEYS:
            base = best_w[k] if i % 2 == 0 else BASE_WEIGHTS[k]
            trial_w[k] = base + rng.gauss(0.0, temp_scale)
        trial_t = clamp(best_t + rng.gauss(0.0, 0.12), 0.72, 1.80)
        obj = evaluate_params(races, trial_w, trial_t)
        if obj > best_obj:
            best_obj = obj
            best_w = trial_w
            best_t = trial_t

    return best_w, best_t, best_obj


def build_training_races(kachiuma_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    by_race = group_by(kachiuma_rows, "race_id")
    races_out: List[Dict[str, object]] = []

    for race_id, rows in by_race.items():
        if not race_id:
            continue

        by_horse = group_by(rows, "HorseName")
        race_runners: List[Dict[str, float]] = []

        for hname, hrows in by_horse.items():
            if not hname.strip():
                continue

            parsed = []
            for r in hrows:
                d = parse_date(r.get("日付"))
                if d is not None:
                    parsed.append((d, r))
            if len(parsed) < 2:
                continue
            parsed.sort(key=lambda x: x[0], reverse=True)

            target_row = parsed[0][1]
            history_rows = [r for _, r in parsed[1:]]

            rank = to_int(target_row.get("着順"))
            if rank is None:
                rank = to_int(target_row.get("finish_pos"))
            if rank is None:
                continue
            label = 1 if rank <= 3 else 0

            surf, dist = parse_surface_distance(target_row.get("距離"))
            venue = parse_venue(target_row.get("開催"))
            going = str(target_row.get("馬場", "")).strip() or "稍重"
            baba = to_float(target_row.get("馬場指数"))
            if baba is None:
                baba = 8.0
            cond = RaceCondition(
                venue=venue or "中山",
                surface=surf or "芝",
                distance=dist or 1800,
                going=going,
                expected_baba_index=baba,
            )

            feat = summarize_history(history_rows, cond)
            market = market_from_row(target_row)
            feat.update(
                {
                    "market_win": market["market_win"],
                    "market_place": market["market_place"],
                    "market_combo": market["market_combo"],
                    "label": float(label),
                }
            )
            race_runners.append(feat)

        if len(race_runners) >= 8:
            races_out.append({"race_id": race_id, "runners": race_runners})

    return races_out


def compose_runner_features(
    horse_name: str,
    rows: List[Dict[str, str]],
    cond: RaceCondition,
    market_map: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    feat = summarize_history(rows, cond)
    market = market_map.get(horse_name, {})
    feat["market_win"] = float(market.get("market_win", 0.0))
    feat["market_place"] = float(market.get("market_place", 0.0))
    feat["market_combo"] = float(market.get("market_combo", 0.0))
    feat["market_odds"] = float(market.get("market_odds", 0.0))
    feat["market_place_odds"] = float(market.get("market_place_odds", 0.0))
    feat["horse_no"] = float(market.get("horse_no", 0.0))
    feat["HorseName"] = horse_name  # type: ignore[assignment]
    return feat


def make_note(row: Dict[str, float]) -> str:
    tags: List[str] = []
    if row["dist_fit"] >= row["ability"] - 1.0:
        tags.append("距離適性")
    if row["venue_fit"] >= row["ability"] - 1.0:
        tags.append("中山適性")
    if row["going_fit"] >= row["ability"] - 0.5:
        tags.append("馬場適性")
    if row["pace_fit"] >= 55.0:
        tags.append("展開噛合")
    if row["trend"] > 1.5:
        tags.append("上昇")
    if row["consistency"] > 48.0:
        tags.append("安定")
    if not tags:
        tags.append("総合型")
    return " / ".join(tags[:3])


def run(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).resolve()
    kachiuma_path = base_dir / args.kachiuma
    shutuba_path = base_dir / args.shutuba
    odds_path = base_dir / args.odds
    fuku_path = base_dir / args.fuku_odds
    wide_path = base_dir / args.wide_odds
    quinella_path = base_dir / args.quinella_odds
    out_path = base_dir / args.output

    print("=" * 68)
    print(" predictor_v3_ultimate.py - condition-aware + market-fusion ranker")
    print("=" * 68)

    kachiuma_rows = load_rows(kachiuma_path)
    shutuba_rows = load_rows(shutuba_path)
    odds_rows = load_rows(odds_path)
    fuku_rows = load_rows(fuku_path)
    wide_rows = load_rows(wide_path)
    quinella_rows = load_rows(quinella_path)

    print(f"[INFO] kachiuma rows: {len(kachiuma_rows)}")
    print(f"[INFO] shutuba rows : {len(shutuba_rows)}")
    print(
        f"[INFO] odds rows(win/place/wide/quinella): "
        f"{len(odds_rows)}/{len(fuku_rows)}/{len(wide_rows)}/{len(quinella_rows)}"
    )

    cond = resolve_race_condition(args)
    print(
        f"[INFO] target condition: {cond.venue} {cond.surface}{cond.distance} "
        f"{cond.going} (expected 馬場指数={cond.expected_baba_index:.1f})"
    )

    train_races = build_training_races(kachiuma_rows)
    print(f"[INFO] training races: {len(train_races)}")
    if not train_races:
        print("[WARN] No train races built; fallback to base weights.")
        best_w = dict(BASE_WEIGHTS)
        best_t = 1.05
        best_obj = -999.0
    else:
        best_w, best_t, best_obj = optimize_weights(
            train_races,
            iterations=max(600, int(args.search_iters)),
            seed=int(args.seed),
        )
    print(f"[INFO] optimized objective: {best_obj:.6f}")
    print(f"[INFO] optimized temperature: {best_t:.3f}")

    market_map = build_market_maps(odds_rows, fuku_rows, wide_rows, quinella_rows)
    by_horse = group_by(shutuba_rows, "HorseName")
    runners: List[Dict[str, float]] = []
    for horse_name, rows in by_horse.items():
        if not horse_name.strip():
            continue
        feat = compose_runner_features(horse_name, rows, cond, market_map)
        runners.append(feat)

    if len(runners) < 2:
        raise RuntimeError("Not enough runners in shutuba.csv to predict.")

    scores, win_probs, top3_probs = score_race(runners, best_w, best_t)
    top3_probs = race_level_normalize(top3_probs, top_k=3)

    for i, r in enumerate(runners):
        r["score"] = scores[i]
        r["win_prob"] = win_probs[i]
        r["top3_prob"] = top3_probs[i]
        market_place_imp = r["market_place"] / 100.0
        r["value_index"] = r["top3_prob"] - market_place_imp
        r["note"] = make_note(r)

    runners.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(runners, start=1):
        r["rank"] = float(i)

    # Race-level confidence
    conf = 0.0
    if len(runners) >= 3:
        conf = clamp((runners[0]["score"] - runners[2]["score"]) / 1.8, 0.0, 1.0)
    score_vals = [float(r["score"]) for r in runners]
    score_min = min(score_vals) if score_vals else 0.0
    score_max = max(score_vals) if score_vals else 0.0
    score_denom = (score_max - score_min) if (score_max - score_min) > 1e-12 else 1.0

    for r in runners:
        r["Top3Prob_model"] = float(r["top3_prob"])
        r["Top3Prob_raw_lr"] = float(r["top3_prob"])
        r["Top3Prob_raw_lgb"] = float(r["top3_prob"])
        r["Top3Prob_cal_lr"] = float(r["top3_prob"])
        r["Top3Prob_cal_lgb"] = float(r["top3_prob"])
        r["Top3Prob_lr"] = float(r["top3_prob"])
        r["Top3Prob_lgbm"] = float(r["top3_prob"])
        r["rank_score"] = float(r["score"])
        r["rank_score_raw"] = float(r["score"])
        r["rank_score_norm"] = (float(r["score"]) - score_min) / score_denom
        r["model_mode"] = "v3_ultimate_hybrid"
        r["score_is_probability"] = 0
        r["race_id"] = "current"
        r["confidence_score"] = round(conf, 4)
        r["stability_score"] = round(conf, 4)
        r["validity_score"] = round(conf, 4)
        r["consistency_score"] = round(conf, 4)
        r["rank_ema"] = 0.5
        r["ev_ema"] = 0.5
        r["risk_score"] = round(1.0 - conf, 4)
        hno = int(r["horse_no"]) if float(r["horse_no"]) > 0 else 0
        r["horse_key"] = str(hno) if hno > 0 else str(r.get("HorseName", "")).strip()

    fieldnames = [
        "rank",
        "horse_no",
        "horse_key",
        "HorseName",
        "score",
        "win_prob",
        "top3_prob",
        "Top3Prob_model",
        "Top3Prob_raw_lr",
        "Top3Prob_raw_lgb",
        "Top3Prob_cal_lr",
        "Top3Prob_cal_lgb",
        "Top3Prob_lr",
        "Top3Prob_lgbm",
        "rank_score",
        "rank_score_raw",
        "rank_score_norm",
        "model_mode",
        "score_is_probability",
        "race_id",
        "confidence_score",
        "stability_score",
        "validity_score",
        "consistency_score",
        "rank_ema",
        "ev_ema",
        "risk_score",
        "value_index",
        "market_odds",
        "market_place_odds",
        "ability",
        "recent",
        "trend",
        "peak",
        "dist_fit",
        "venue_fit",
        "going_fit",
        "baba_fit",
        "pace_fit",
        "consistency",
        "class_power",
        "freshness",
        "experience",
        "recent_top3_rate",
        "runs_used",
        "note",
    ]

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in runners:
            row = {k: r.get(k, "") for k in fieldnames}
            wr.writerow(row)

    print("\nTop5 predictions:")
    print("-" * 68)
    for r in runners[:5]:
        hno = int(r["horse_no"]) if r["horse_no"] > 0 else 0
        hno_str = f"{hno:>2}" if hno > 0 else " ?"
        print(
            f"{int(r['rank']):>2}. {hno_str} {str(r['HorseName']):<18} "
            f"rank_score={r['rank_score']:>6.3f}  Top3Prob_model={r['Top3Prob_model']*100:>5.1f}%"
        )

    print("-" * 68)
    print(f"[INFO] confidence: {conf:.3f}")
    print(f"Saved: {out_path.name}")
    if not args.no_wait:
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ultimate horse race predictor (v3)")
    p.add_argument("--base-dir", default=".", help="Base directory for csv files")
    p.add_argument("--kachiuma", default="kachiuma.csv")
    p.add_argument("--shutuba", default="shutuba.csv")
    p.add_argument("--odds", default="odds.csv")
    p.add_argument("--fuku-odds", default="fuku_odds.csv")
    p.add_argument("--wide-odds", default="wide_odds.csv")
    p.add_argument("--quinella-odds", default="quinella_odds.csv")
    p.add_argument("--output", default="predictions.csv")

    # Race condition defaults reflect your requested race.
    p.add_argument("--race-venue", default="中山")
    p.add_argument("--race-surface", default="芝")
    p.add_argument("--race-distance", type=int, default=1800)
    p.add_argument("--race-going", default="稍重")
    p.add_argument(
        "--expected-baba-index",
        type=float,
        default=8.0,
        help="Expected 馬場指数 for the target race day",
    )

    p.add_argument("--search-iters", type=int, default=3500)
    p.add_argument("--seed", type=int, default=20260307)
    p.add_argument("--no-prompt", action="store_true", help="Disable interactive race condition prompts")
    p.add_argument("--no-wait", action="store_true", help="Disable final Press Enter prompt")
    return p


if __name__ == "__main__":
    configure_utf8_io()
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)

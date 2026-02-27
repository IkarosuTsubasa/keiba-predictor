import csv
import heapq
import itertools
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

from surface_scope import get_data_dir, migrate_legacy_data

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


def resolve_scope_key():
    raw = os.environ.get("SCOPE_KEY", "").strip().lower()
    raw = raw.replace(" ", "_").replace("-", "_").replace("/", "_")
    if raw in ("central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"):
        return "central_turf"
    if raw in ("central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"):
        return "central_dirt"
    if raw in ("local", "l", "3"):
        return "local"
    try:
        raw = input("Data scope (central_turf/central_dirt/local) [central_dirt]: ").strip().lower()
    except EOFError:
        raw = ""
    raw = raw.replace(" ", "_").replace("-", "_").replace("/", "_")
    if raw in ("central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"):
        return "central_turf"
    if raw in ("central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"):
        return "central_dirt"
    if raw in ("local", "l", "3"):
        return "local"
    return "central_dirt"


SCOPE_KEY = resolve_scope_key()
CONFIG_PATH = BASE_DIR / f"config_{SCOPE_KEY}.json"

ODDS_PATH = ROOT_DIR / "odds.csv"
WIDE_ODDS_PATH = ROOT_DIR / "wide_odds.csv"
FUKU_ODDS_PATH = ROOT_DIR / "fuku_odds.csv"
QUINELLA_ODDS_PATH = ROOT_DIR / "quinella_odds.csv"
PRED_PATH = ROOT_DIR / "predictions.csv"
OUT_PATH = BASE_DIR / "bet_plan_update.csv"
NO_BET_LOG_PATH = BASE_DIR / f"no_bet_log_{SCOPE_KEY}.csv"

UNIT_YEN = 100
SIM_RUNS = 3000
SIM_SEED = 42
DEFAULT_BUDGETS = [2000, 5000, 10000, 50000]

# --- Monte Carlo uncertainty helpers (SE / 95% CI) ---
def mc_se(p: float, runs: int) -> float:
    """Standard error of Monte Carlo estimate p with runs samples (binomial approx)."""
    p = float(p or 0.0)
    p = max(0.0, min(1.0, p))
    n = max(1, int(runs or 1))
    return math.sqrt(p * (1.0 - p) / float(n))


def mc_ci95(p: float, runs: int):
    """Return (se, ci_low, ci_high) for 95% CI."""
    se = mc_se(p, runs)
    lo = max(0.0, float(p) - 1.96 * se)
    hi = min(1.0, float(p) + 1.96 * se)
    return se, lo, hi

NO_TRIFECTA_REC_LABEL = "no \u4e09\u8fde\u590d \u63a8\u8350"
NO_BET_LOG_FIELDS = [
    "scope",
    "race_id",
    "budget_yen",
    "bet_type",
    "horse_pair",
    "model_prob",
    "market_prob_open",
    "ev_ratio_open",
    "no_bet_reason",
    "market_prob_close",
]


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        data = (str(text) + "\n").encode(encoding, errors="replace")
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer:
            buffer.write(data)
        else:
            sys.stdout.write(data.decode(encoding, errors="replace"))


def resolve_race_id():
    return os.environ.get("RACE_ID", "").strip()


def resolve_run_paths(race_id):
    if not race_id:
        return {}
    migrate_legacy_data(BASE_DIR, SCOPE_KEY)
    data_dir = get_data_dir(BASE_DIR, SCOPE_KEY)
    runs_path = data_dir / "runs.csv"
    if not runs_path.exists():
        return {}
    with open(runs_path, "r", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if str(row.get("race_id", "")).strip() == str(race_id)
        ]
    if not rows:
        return {}
    run = rows[-1]
    return {
        "predictions_path": run.get("predictions_path", ""),
        "odds_path": run.get("odds_path", ""),
        "wide_odds_path": run.get("wide_odds_path", ""),
        "fuku_odds_path": run.get("fuku_odds_path", ""),
        "quinella_odds_path": run.get("quinella_odds_path", ""),
    }


def load_config(path=None):
    path = path or CONFIG_PATH
    if not path.exists():
        fallback = []
        if SCOPE_KEY == "central_turf":
            fallback.extend(
                [
                    BASE_DIR / "config_turf_default.json",
                    BASE_DIR / "config_turf.json",
                ]
            )
        elif SCOPE_KEY == "central_dirt":
            fallback.extend(
                [
                    BASE_DIR / "config_dirt_default.json",
                    BASE_DIR / "config_dirt.json",
                ]
            )
        elif SCOPE_KEY == "local":
            fallback.extend(
                [
                    BASE_DIR / "config_central_dirt.json",
                    BASE_DIR / "config_dirt_default.json",
                    BASE_DIR / "config_dirt.json",
                ]
            )
        fallback.append(BASE_DIR / "config.json")
        for legacy in fallback:
            if legacy.exists():
                data = json.loads(legacy.read_text(encoding="utf-8"))
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return data
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_strategy_from_env(config):
    strategies = config.get("strategies", {})
    if not strategies:
        return config, ""

    selected = os.environ.get("BET_STRATEGY", "").strip()
    if selected not in strategies:
        selected = config.get("active_strategy", "")
    if selected not in strategies:
        selected = next(iter(strategies))

    overrides = strategies[selected].get("overrides", {})
    merged = dict(config)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged, selected


def prompt_int(label, default_value):
    raw = input(label).strip()
    if not raw:
        return default_value
    try:
        value = int(raw)
        return value if value > 0 else default_value
    except ValueError:
        return default_value


def prompt_style(label, default_value):
    raw = input(label).strip().lower()
    if not raw:
        return default_value
    if raw in ("s", "steady"):
        return "steady"
    if raw in ("a", "aggressive", "risk"):
        return "aggressive"
    return "balanced"


def resolve_budget_list():
    raw = str(os.environ.get("BET_BUDGETS", "")).strip()
    if not raw:
        return list(DEFAULT_BUDGETS)
    budgets = []
    seen = set()
    for token in re.split(r"[,\s]+", raw):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(float(token))
        except (TypeError, ValueError):
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        budgets.append(value)
    return budgets or list(DEFAULT_BUDGETS)


def pause_exit():
    input("Press Enter to exit...")


def append_no_bet_logs(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NO_BET_LOG_FIELDS)
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        for row in rows:
            payload = {field: row.get(field, "") for field in NO_BET_LOG_FIELDS}
            writer.writerow(payload)


def normalize_name(value):
    return "".join(str(value or "").split())


def load_inputs(odds_path, pred_path):
    odds = pd.read_csv(odds_path, encoding="utf-8-sig")
    preds = pd.read_csv(pred_path, encoding="utf-8-sig")
    odds["name_key"] = odds["name"].apply(normalize_name)
    preds["name_key"] = preds["HorseName"].apply(normalize_name)
    dup_mask = odds["name_key"].duplicated(keep=False)
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        key_count = int(odds["name_key"].nunique())
        print(f"[WARN] odds name_key duplicates: {dup_count} rows, {key_count} unique keys; keeping last.")
        odds = odds.drop_duplicates(subset=["name_key"], keep="last")
    merged = preds.merge(odds, on="name_key", how="left")
    if len(merged) != len(preds):
        print(f"[WARN] merge expanded rows: preds={len(preds)} merged={len(merged)} (duplicate name_key in odds?)")
    return odds, preds, merged


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_horse_no(value):
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        digits = re.findall(r"\d+", text)
        if not digits:
            return None
        return int(digits[0])


def normalize_weights(values):
    if not values:
        return []
    total = float(sum(values))
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in values]


def pick_prob_column(columns):
    for key in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if key in columns:
            return key
    return ""


def get_prob_series(df, prob_col):
    if not prob_col:
        return pd.Series([0.0] * len(df))
    return pd.to_numeric(df.get(prob_col), errors="coerce").fillna(0.0)


def compute_strength(merged, blend):
    prob_col = pick_prob_column(merged.columns)
    top3 = get_prob_series(merged, prob_col).astype(float).tolist()
    top3_norm = normalize_weights(top3)

    odds = merged["odds_num"].fillna(0.0).astype(float).tolist()
    win_raw = [(1.0 / v) if v > 0 else 0.0 for v in odds]
    win_norm = normalize_weights(win_raw) if win_raw else top3_norm

    strength = []
    for i in range(len(top3_norm)):
        strength.append((top3_norm[i] ** blend) * (win_norm[i] ** (1.0 - blend)))
    strength = normalize_weights(strength)
    merged = merged.copy()
    merged["strength"] = strength
    return merged


def sample_top3_indices(weights, rng, count=3):
    remaining = list(range(len(weights)))
    w = [max(0.0, float(x)) for x in weights]
    picks = []
    for _ in range(min(count, len(remaining))):
        total = sum(w)
        if total <= 0:
            idx = rng.randrange(len(remaining))
        else:
            r = rng.random() * total
            cum = 0.0
            idx = len(remaining) - 1
            for i, weight in enumerate(w):
                cum += weight
                if r <= cum:
                    idx = i
                    break
        picks.append(remaining.pop(idx))
        w.pop(idx)
    return picks


def sample_order_indices(weights, rng):
    return sample_top3_indices(weights, rng, count=len(weights))


def simulate_prob_maps(weights, runs=SIM_RUNS, seed=SIM_SEED):
    n = len(weights)
    if n == 0 or runs <= 0:
        return {"win": {}, "place": {}, "wide": {}, "quinella": {}, "trifecta": {}}

    win_counts = [0] * n
    place_counts = [0] * n
    wide_counts = {}
    quinella_counts = {}
    trifecta_counts = {}

    rng = random.Random(seed)
    for _ in range(runs):
        order = sample_order_indices(weights, rng)
        if not order:
            continue
        win_counts[order[0]] += 1
        for idx in order[:3]:
            place_counts[idx] += 1

        if len(order) >= 2:
            i, j = order[0], order[1]
            pair = (i, j) if i < j else (j, i)
            quinella_counts[pair] = quinella_counts.get(pair, 0) + 1

        if len(order) >= 3:
            i, j, k = order[0], order[1], order[2]
            pairs = [
                (i, j) if i < j else (j, i),
                (i, k) if i < k else (k, i),
                (j, k) if j < k else (k, j),
            ]
            for pair in pairs:
                wide_counts[pair] = wide_counts.get(pair, 0) + 1
            tri_key = tuple(sorted((i, j, k)))
            trifecta_counts[tri_key] = trifecta_counts.get(tri_key, 0) + 1

    denom = float(runs)
    return {
        "win": {i: c / denom for i, c in enumerate(win_counts)},
        "place": {i: c / denom for i, c in enumerate(place_counts)},
        "wide": {k: v / denom for k, v in wide_counts.items()},
        "quinella": {k: v / denom for k, v in quinella_counts.items()},
        "trifecta": {k: v / denom for k, v in trifecta_counts.items()},
    }


def select_eligible_indices(place_probs, coverage, min_count):
    if not place_probs:
        return []
    items = list(place_probs.items())
    n = len(items)
    min_count = min(min_count, n)
    total = float(sum(prob for _, prob in items))
    if total <= 0:
        return [idx for idx, _ in items]
    items.sort(key=lambda x: x[1], reverse=True)
    eligible = []
    cum = 0.0
    for idx, prob in items:
        eligible.append(idx)
        cum += prob / total
        if cum >= coverage and len(eligible) >= min_count:
            break
    return eligible


def estimate_payout_multiplier(bet_type, horses, factors):
    if bet_type == "win":
        if horses:
            odds_val = safe_float(horses[0].get("odds_num"))
            if odds_val > 0:
                return max(1.0, odds_val)
    if bet_type == "place":
        odds_val = lookup_fuku_odds(horses)
        if odds_val:
            return max(1.0, odds_val)
    if bet_type == "wide":
        odds_val = lookup_wide_odds(horses)
        if odds_val:
            return max(1.0, odds_val)
    if bet_type == "quinella":
        odds_val = lookup_quinella_odds(horses)
        if odds_val:
            return max(1.0, odds_val)
    odds = []
    for horse in horses:
        val = safe_float(horse.get("odds_num"))
        if val > 0:
            odds.append(val)
    base = sum(odds) / len(odds) if odds else 1.0
    factor = factors.get(bet_type, 1.0)
    return max(1.0, base * factor)


def load_wide_odds_map(path):
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return {}
    if "horse_no_a" not in df.columns or "horse_no_b" not in df.columns:
        return {}
    odds_col = "odds_mid" if "odds_mid" in df.columns else None
    if not odds_col:
        return {}
    out = {}
    for _, row in df.iterrows():
        a = parse_horse_no(row.get("horse_no_a", ""))
        b = parse_horse_no(row.get("horse_no_b", ""))
        if a is None or b is None:
            continue
        try:
            val = float(row.get(odds_col, 0))
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        if a > b:
            a, b = b, a
        out[(a, b)] = val
    return out


WIDE_ODDS_MAP = load_wide_odds_map(WIDE_ODDS_PATH)


def load_fuku_odds_map(path):
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return {}
    if "horse_no" not in df.columns:
        return {}
    has_mid = "odds_mid" in df.columns
    has_low = "odds_low" in df.columns
    if not has_mid and not has_low:
        return {}
    out = {}
    for _, row in df.iterrows():
        horse_no = parse_horse_no(row.get("horse_no", ""))
        if horse_no is None:
            continue
        val = safe_float(row.get("odds_mid")) if has_mid else 0.0
        if val <= 0 and has_low:
            val = safe_float(row.get("odds_low"))
        if val <= 0:
            continue
        out[horse_no] = val
    return out


FUKU_ODDS_MAP = load_fuku_odds_map(FUKU_ODDS_PATH)


def load_quinella_odds_map(path):
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return {}
    if "horse_no_a" not in df.columns or "horse_no_b" not in df.columns:
        return {}
    odds_col = "odds" if "odds" in df.columns else "odds_mid"
    out = {}
    for _, row in df.iterrows():
        a = parse_horse_no(row.get("horse_no_a", ""))
        b = parse_horse_no(row.get("horse_no_b", ""))
        if a is None or b is None:
            continue
        try:
            val = float(row.get(odds_col, 0))
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        if a > b:
            a, b = b, a
        out[(a, b)] = val
    return out


QUINELLA_ODDS_MAP = load_quinella_odds_map(QUINELLA_ODDS_PATH)


def lookup_wide_odds(horses):
    if not WIDE_ODDS_MAP or len(horses) < 2:
        return 0.0
    a = parse_horse_no(horses[0].get("horse_no", ""))
    b = parse_horse_no(horses[1].get("horse_no", ""))
    if a is None or b is None or a == b:
        return 0.0
    if a > b:
        a, b = b, a
    return float(WIDE_ODDS_MAP.get((a, b), 0.0))


def lookup_fuku_odds(horses):
    if not FUKU_ODDS_MAP or not horses:
        return 0.0
    horse_no = parse_horse_no(horses[0].get("horse_no", ""))
    if horse_no is None:
        return 0.0
    return float(FUKU_ODDS_MAP.get(horse_no, 0.0))


def lookup_quinella_odds(horses):
    if not QUINELLA_ODDS_MAP or len(horses) < 2:
        return 0.0
    a = parse_horse_no(horses[0].get("horse_no", ""))
    b = parse_horse_no(horses[1].get("horse_no", ""))
    if a is None or b is None or a == b:
        return 0.0
    if a > b:
        a, b = b, a
    return float(QUINELLA_ODDS_MAP.get((a, b), 0.0))


def build_market_prob_map(odds_map):
    if not odds_map:
        return {}
    raw = {}
    for key, odds in odds_map.items():
        val = safe_float(odds)
        if val > 0:
            raw[key] = 1.0 / val
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {key: val / total for key, val in raw.items()}


def lookup_market_prob(horses, market_prob_map):
    if not market_prob_map or len(horses) < 2:
        return 0.0
    a = parse_horse_no(horses[0].get("horse_no", ""))
    b = parse_horse_no(horses[1].get("horse_no", ""))
    if a is None or b is None or a == b:
        return 0.0
    if a > b:
        a, b = b, a
    return float(market_prob_map.get((a, b), 0.0))


def lookup_place_market_prob(horses, market_prob_map):
    if not market_prob_map or not horses:
        return 0.0
    horse_no = parse_horse_no(horses[0].get("horse_no", ""))
    if horse_no is None:
        return 0.0
    return float(market_prob_map.get(horse_no, 0.0))


def split_units(total_units, weights):
    if total_units <= 0:
        return [0] * len(weights)
    if not weights:
        return []
    weights = pd.Series(weights, dtype=float).fillna(0.0)
    if float(weights.sum()) == 0.0:
        weights = pd.Series([1.0] * len(weights))
    raw = weights / weights.sum() * total_units
    base = raw.apply(math.floor).astype(int).tolist()
    remainder = total_units - sum(base)
    frac = (raw - pd.Series(base)).tolist()
    order = sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)
    for i in range(remainder):
        base[order[i % len(base)]] += 1
    return base


def compute_type_quality(keys, prob_map, top_k):
    hits = []
    for key in keys:
        hit = prob_map.get(key, 0.0)
        if hit > 0:
            hits.append(hit)
    if not hits:
        return 0.0
    top_hits = heapq.nlargest(min(top_k, len(hits)), hits)
    return sum(top_hits) / float(len(top_hits))


def build_plan(merged, budget_yen, config, race_id=""):
    if merged.empty:
        return [], None, []
    merged = merged.copy()
    if "odds_num" not in merged.columns:
        if "odds" in merged.columns:
            merged["odds_num"] = pd.to_numeric(merged["odds"], errors="coerce")
        else:
            merged["odds_num"] = 0.0
    prob_col = pick_prob_column(merged.columns)
    merged["Top3Prob_used"] = get_prob_series(merged, prob_col)
    units_total = int(budget_yen // UNIT_YEN)

    merged = merged.sort_values("Top3Prob_used", ascending=False).reset_index(drop=True)
    merged = compute_strength(merged, blend=config["strength_blend"])

    horses_list = merged.to_dict("records")
    runs_used = SIM_RUNS
    prob_maps = simulate_prob_maps(
        merged["strength"].fillna(0.0).astype(float).tolist(),
        runs=runs_used,
        seed=SIM_SEED,
    )
    place_prob_map = prob_maps.get("place", {})
    eligible_indices = select_eligible_indices(
        place_prob_map,
        coverage=config["coverage_target"],
        min_count=config["min_eligible"],
    )
    if not eligible_indices:
        eligible_indices = list(range(len(horses_list)))

    market_prob_maps = {
        "place": build_market_prob_map(FUKU_ODDS_MAP),
        "wide": build_market_prob_map(WIDE_ODDS_MAP),
        "quinella": build_market_prob_map(QUINELLA_ODDS_MAP),
    }

    no_bet_rows = []
    ev_mins = config.get("ev_min", {})
    hit_mins = config.get("hit_min", {})

    def record_no_bet(bet_type, horses, hit_prob, market_prob, ev_ratio, reason):
        labels = []
        for horse in horses:
            horse_no = parse_horse_no(horse.get("horse_no", ""))
            if horse_no is not None:
                labels.append(str(horse_no))
        horse_pair = "-".join([l for l in labels if l])
        no_bet_rows.append(
            {
                "scope": SCOPE_KEY,
                "race_id": race_id,
                "budget_yen": budget_yen,
                "bet_type": bet_type,
                "horse_pair": horse_pair,
                "model_prob": round(float(hit_prob), 6) if hit_prob is not None else "",
                "market_prob_open": round(float(market_prob), 6) if market_prob is not None else "",
                "ev_ratio_open": round(float(ev_ratio), 6) if ev_ratio is not None else "",
                "no_bet_reason": reason,
                "market_prob_close": "",
            }
        )

    trifecta_rec = build_trifecta_recommendation(
        horses_list,
        prob_maps,
        eligible_indices,
        config,
        runs_used,
    )
    if units_total <= 0:
        return [], trifecta_rec, no_bet_rows

    def key_to_horses(key):
        if isinstance(key, int):
            return [horses_list[key]]
        return [horses_list[i] for i in key]

    def candidates_for_type(bet_type):
        if bet_type == "win":
            return list(eligible_indices)
        if bet_type == "place":
            return list(eligible_indices)
        if bet_type in ("wide", "quinella"):
            return list(itertools.combinations(eligible_indices, 2))
        return []

    def calc_cap(bet_type, units_type, style_key):
        if units_type <= 0:
            return 0
        if bet_type in ("win", "place"):
            if units_type < 4:
                base = 2
            elif units_type < 8:
                base = 3
            else:
                base = 4
            style_caps = {"steady": 5, "balanced": 4, "aggressive": 3}
            return min(base, style_caps.get(style_key, 4))
        if bet_type in ("wide", "quinella"):
            if units_type < 4:
                return 2
            if units_type < 8:
                return 4
            if units_type < 12:
                return 6
            return 8
        if units_type < 4:
            return 1
        if units_type < 8:
            return 2
        return 4

    def score_candidates(bet_type, keys, cap, prob_map):
        scored = []
        ev_min = safe_float(ev_mins.get(bet_type, 0.0))
        hit_min = safe_float(hit_mins.get(bet_type, 0.0))
        if bet_type == "place":
            ev_min = max(ev_min, 1.0)
        for key in keys:
            hit_prob = prob_map.get(key, 0.0)
            if hit_prob <= 0:
                continue
            horses = key_to_horses(key)
            payout_mult = None
            ev_ratio = None
            market_prob = None
            if bet_type in ("place", "wide", "quinella"):
                if bet_type == "place":
                    market_prob = lookup_place_market_prob(horses, market_prob_maps.get("place", {}))
                    payout_mult = lookup_fuku_odds(horses)
                elif bet_type == "wide":
                    market_prob = lookup_market_prob(horses, market_prob_maps.get("wide", {}))
                    payout_mult = lookup_wide_odds(horses)
                else:
                    market_prob = lookup_market_prob(horses, market_prob_maps.get("quinella", {}))
                    payout_mult = lookup_quinella_odds(horses)
                if payout_mult <= 0:
                    record_no_bet(bet_type, horses, hit_prob, market_prob, None, "no_odds")
                    continue
                ev_ratio = hit_prob * payout_mult
            else:
                payout_mult = estimate_payout_multiplier(bet_type, horses, payout_factors)
                ev_ratio = hit_prob * payout_mult
            if ev_min > 0.0 and ev_ratio < ev_min:
                record_no_bet(bet_type, horses, hit_prob, market_prob, ev_ratio, "ev_below_threshold")
                continue
            if hit_min > 0.0 and hit_prob < hit_min:
                record_no_bet(bet_type, horses, hit_prob, market_prob, ev_ratio, "hit_below_threshold")
                continue
            score = (hit_prob ** hit_weight) * (payout_mult ** payout_weight)
            scored.append((score, hit_prob, payout_mult, ev_ratio, key, horses))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        max_hit = max(item[1] for item in scored)
        hit_floor = hit_floors.get(bet_type, 0.0)
        filtered = [item for item in scored if item[1] >= max_hit * hit_floor]
        if not filtered:
            filtered = scored
        filtered.sort(key=lambda x: (x[1], x[0]), reverse=True)
        if cap and cap < len(filtered):
            return filtered[:cap]
        return filtered

    hit_weight = config["hit_weight"]
    payout_weight = config["payout_weight"]
    hit_floors = config["hit_floors"]
    payout_factors = {
        "win": 1.0,
        "place": 0.35,
        "wide": 0.25,
        "quinella": 0.6,
    }

    style_key = config["style_default"]
    style_weights = dict(config["style_weights"].get(style_key, config["style_weights"]["balanced"]))
    type_adjust = config.get("type_weight_adjust", {})
    for key in style_weights:
        style_weights[key] = style_weights[key] * float(type_adjust.get(key, 1.0))

    if units_total < 5:
        style_weights = {"win": 0.5, "place": 0.5, "wide": 0.0, "quinella": 0.0}
    elif units_total < 10:
        style_weights["quinella"] = 0.0

    horse_count = len(horses_list)
    if horse_count < 3:
        style_weights["wide"] = 0.0
    if horse_count < 2:
        style_weights["quinella"] = 0.0

    if sum(style_weights.values()) <= 0:
        style_weights = {"win": 0.5, "place": 0.5, "wide": 0.0, "quinella": 0.0}

    order = ["win", "place", "wide", "quinella"]
    type_quality = {}
    for bet_type in order:
        keys = candidates_for_type(bet_type)
        if not keys:
            continue
        prob_map = prob_maps.get(bet_type, {})
        type_quality[bet_type] = compute_type_quality(keys, prob_map, config["type_quality_topk"])

    adjusted_weights = {}
    for bet_type in order:
        base = style_weights.get(bet_type, 0.0)
        quality = type_quality.get(bet_type, 0.0)
        adjusted_weights[bet_type] = base * (quality ** config["type_quality_power"])
    if sum(adjusted_weights.values()) > 0:
        style_weights = adjusted_weights

    units_by_type = split_units(units_total, [style_weights[o] for o in order])

    tickets = []
    scored_by_type = {}
    fallback_units = 0

    for bet_type, units_type in zip(order, units_by_type):
        if units_type <= 0:
            continue
        keys = candidates_for_type(bet_type)
        if not keys:
            fallback_units += units_type
            continue
        cap = calc_cap(bet_type, units_type, style_key)
        prob_map = prob_maps.get(bet_type, {})
        scored = score_candidates(bet_type, keys, cap, prob_map)
        if not scored:
            fallback_units += units_type
            continue
        scored_by_type[bet_type] = (units_type, scored)

    if fallback_units > 0:
        for target in order:
            if target in scored_by_type:
                units_type, scored = scored_by_type[target]
                scored_by_type[target] = (units_type + fallback_units, scored)
                fallback_units = 0
                break
        if fallback_units > 0:
            return tickets, trifecta_rec, no_bet_rows

    for bet_type in order:
        if bet_type not in scored_by_type:
            continue
        units_type, scored = scored_by_type[bet_type]
        hit_weights = [hit ** hit_weight for _, hit, *_ in scored]
        alloc = split_units(units_type, hit_weights)
        for (score, hit_prob, payout_mult, ev_ratio, key, horses), units in zip(scored, alloc):
            if units <= 0:
                continue
            labels = []
            for horse in horses:
                horse_no = parse_horse_no(horse.get("horse_no", ""))
                if horse_no is not None:
                    labels.append(str(horse_no))
            names = [str(h.get("HorseName", "")).strip() for h in horses]
            horse_no = "-".join([l for l in labels if l])
            horse_name = " / ".join([n for n in names if n])
            amount_yen = units * UNIT_YEN
            se, lo, hi = mc_ci95(hit_prob, runs_used)
            if payout_mult is not None:
                expected_return = int(round(amount_yen * hit_prob * payout_mult))
            else:
                expected_return = int(round(amount_yen * hit_prob))
            ev_ratio_est = (hit_prob * payout_mult) if payout_mult is not None else hit_prob
            tickets.append({
                "budget_yen": int(budget_yen),
                "bet_type": bet_type,
                "horse_no": horse_no,
                "horse_name": horse_name,
                "units": units,
                "amount_yen": amount_yen,
                "hit_prob_est": round(hit_prob, 4),
                "hit_prob_se": round(se, 4),
                "hit_prob_ci95_low": round(lo, 4),
                "hit_prob_ci95_high": round(hi, 4),
                "payout_mult": round(float(payout_mult), 4) if payout_mult is not None else "",
                "ev_ratio_est": round(float(ev_ratio_est), 4),
                "expected_return_yen": expected_return,
            })
    # --- PASS gate (optional): skip betting if edge isn't strong enough ---
    pass_gate = config.get("pass_gate", {})
    allow_pass = bool(pass_gate.get("allow_pass", True))
    mode = str(pass_gate.get("mode", "hard")).strip().lower()
    soft_mode = mode == "soft"
    min_portfolio_ev = float(pass_gate.get("min_portfolio_ev", 1.01))
    min_best_ev = float(pass_gate.get("min_best_ev", 1.05))
    min_best_ev_low = float(pass_gate.get("min_best_ev_low", 1.0))
    use_ci_low = bool(pass_gate.get("use_ci_low", True))
    hard_min_portfolio_ev = float(pass_gate.get("hard_min_portfolio_ev", min_portfolio_ev))
    hard_min_best_ev = float(pass_gate.get("hard_min_best_ev", min_best_ev))
    ci_low_min_payout_mult = float(pass_gate.get("ci_low_min_payout_mult", 3.0))
    ci_low_max_hit_prob = float(pass_gate.get("ci_low_max_hit_prob", 0.25))

    if allow_pass and tickets:
        total_bet = sum(int(t.get("amount_yen", 0) or 0) for t in tickets)
        weighted_ev = sum(
            float(t.get("ev_ratio_est") or 0.0) * int(t.get("amount_yen", 0) or 0)
            for t in tickets
        )
        portfolio_ev = (weighted_ev / total_bet) if total_bet > 0 else 0.0

        best_ev = 0.0
        best_ev_low = 0.0
        best_ev_low_set = False
        for t in tickets:
            ev = float(t.get("ev_ratio_est") or 0.0)
            if ev > best_ev:
                best_ev = ev

            if use_ci_low:
                payout_mult = t.get("payout_mult", "")
                if payout_mult != "":
                    payout_mult = float(payout_mult)
                    hit_prob = float(t.get("hit_prob_est") or 0.0)
                    if payout_mult > 0 and (
                        payout_mult >= ci_low_min_payout_mult or hit_prob <= ci_low_max_hit_prob
                    ):
                        ci_low = float(t.get("hit_prob_ci95_low", 0.0) or 0.0)
                        ev_low = ci_low * payout_mult
                        if ev_low > best_ev_low:
                            best_ev_low = ev_low
                        best_ev_low_set = True

        soft_fail = (portfolio_ev < min_portfolio_ev) or (best_ev < min_best_ev)
        if use_ci_low and best_ev_low_set:
            soft_fail = soft_fail or (best_ev_low < min_best_ev_low)

        if soft_mode:
            hard_fail = (portfolio_ev < hard_min_portfolio_ev) and (best_ev < hard_min_best_ev)
        else:
            hard_fail = soft_fail

        soft_reasons = []
        if portfolio_ev < min_portfolio_ev:
            soft_reasons.append(
                f"portfolio_ev {portfolio_ev:.4f} < min_portfolio_ev {min_portfolio_ev:.4f}"
            )
        if best_ev < min_best_ev:
            soft_reasons.append(f"best_ev {best_ev:.4f} < min_best_ev {min_best_ev:.4f}")
        if use_ci_low and best_ev_low_set and best_ev_low < min_best_ev_low:
            soft_reasons.append(
                f"best_ev_low {best_ev_low:.4f} < min_best_ev_low {min_best_ev_low:.4f}"
            )
        soft_reason_text = "; ".join(soft_reasons) if soft_reasons else "soft_gate_failed"

        hard_reasons = []
        hard_portfolio_min = hard_min_portfolio_ev if soft_mode else min_portfolio_ev
        hard_best_min = hard_min_best_ev if soft_mode else min_best_ev
        portfolio_label = "hard_min_portfolio_ev" if soft_mode else "min_portfolio_ev"
        best_label = "hard_min_best_ev" if soft_mode else "min_best_ev"
        if portfolio_ev < hard_portfolio_min:
            hard_reasons.append(
                f"portfolio_ev {portfolio_ev:.4f} < {portfolio_label} {hard_portfolio_min:.4f}"
            )
        if best_ev < hard_best_min:
            hard_reasons.append(
                f"best_ev {best_ev:.4f} < {best_label} {hard_best_min:.4f}"
            )
        if not soft_mode and use_ci_low and best_ev_low_set and best_ev_low < min_best_ev_low:
            hard_reasons.append(
                f"best_ev_low {best_ev_low:.4f} < min_best_ev_low {min_best_ev_low:.4f}"
            )
        hard_reason_text = "; ".join(hard_reasons) if hard_reasons else "hard_gate_failed"

        if soft_mode and soft_fail and not hard_fail:
            for t in tickets:
                t["gate_status"] = "soft_fail"
                t["risk_note"] = "high_risk"
                t["gate_reason"] = soft_reason_text
            if trifecta_rec:
                trifecta_rec["gate_status"] = "soft_fail"
                trifecta_rec["risk_note"] = "high_risk"
                trifecta_rec["gate_reason"] = soft_reason_text
            no_bet_rows.append(
                {
                    "scope": SCOPE_KEY,
                    "race_id": race_id,
                    "budget_yen": budget_yen,
                    "bet_type": "pass",
                    "horse_pair": "",
                    "model_prob": "",
                    "market_prob_open": "",
                    "ev_ratio_open": round(float(portfolio_ev), 6),
                    "no_bet_reason": "pass_gate_soft",
                    "market_prob_close": "",
                }
            )

        if hard_fail:
            for t in tickets:
                t["gate_status"] = "hard_fail"
                t["risk_note"] = "hard_blocked"
                t["gate_reason"] = hard_reason_text
            if trifecta_rec:
                trifecta_rec["gate_status"] = "hard_fail"
                trifecta_rec["risk_note"] = "hard_blocked"
                trifecta_rec["gate_reason"] = hard_reason_text
            no_bet_rows.append(
                {
                    "scope": SCOPE_KEY,
                    "race_id": race_id,
                    "budget_yen": budget_yen,
                    "bet_type": "pass",
                    "horse_pair": "",
                    "model_prob": "",
                    "market_prob_open": "",
                    "ev_ratio_open": round(float(portfolio_ev), 6),
                    "no_bet_reason": "pass_gate",
                    "market_prob_close": "",
                }
            )
    return tickets, trifecta_rec, no_bet_rows


def build_trifecta_recommendation(horses_list, prob_maps, eligible_indices, config, runs_used=SIM_RUNS):
    tri_map = prob_maps.get("trifecta", {})
    min_hit_prob = safe_float(config.get("trifecta_rec_min_hit_prob", 0.12))
    if not tri_map:
        se, lo, hi = mc_ci95(0.0, runs_used)
        return {
            "budget_yen": "",
            "bet_type": "trifecta_rec",
            "horse_no": "",
            "horse_name": NO_TRIFECTA_REC_LABEL,
            "units": 0,
            "amount_yen": 0,
            "hit_prob_est": 0.0,
            "hit_prob_se": round(se, 4),
            "hit_prob_ci95_low": round(lo, 4),
            "hit_prob_ci95_high": round(hi, 4),
            "expected_return_yen": 0,
        }
    eligible_set = set(eligible_indices)
    best_key = None
    best_hit = 0.0
    for key, hit_prob in tri_map.items():
        if not set(key).issubset(eligible_set):
            continue
        if hit_prob > best_hit:
            best_hit = hit_prob
            best_key = key
    if not best_key or best_hit < min_hit_prob:
        se, lo, hi = mc_ci95(best_hit, runs_used)
        return {
            "budget_yen": "",
            "bet_type": "trifecta_rec",
            "horse_no": "",
            "horse_name": NO_TRIFECTA_REC_LABEL,
            "units": 0,
            "amount_yen": 0,
            "hit_prob_est": round(best_hit, 4) if best_hit else 0.0,
            "hit_prob_se": round(se, 4),
            "hit_prob_ci95_low": round(lo, 4),
            "hit_prob_ci95_high": round(hi, 4),
            "expected_return_yen": 0,
        }
    horses = [horses_list[i] for i in best_key]
    labels = []
    for horse in horses:
        horse_no = parse_horse_no(horse.get("horse_no", ""))
        if horse_no is not None:
            labels.append(str(horse_no))
    names = [str(h.get("HorseName", "")).strip() for h in horses]
    horse_no = "-".join([l for l in labels if l])
    horse_name = " / ".join([n for n in names if n])
    se, lo, hi = mc_ci95(best_hit, runs_used)
    return {
        "budget_yen": "",
        "bet_type": "trifecta_rec",
        "horse_no": horse_no,
        "horse_name": horse_name,
        "units": 0,
        "amount_yen": 0,
        "hit_prob_est": round(best_hit, 4),
        "hit_prob_se": round(se, 4),
        "hit_prob_ci95_low": round(lo, 4),
        "hit_prob_ci95_high": round(hi, 4),
        "expected_return_yen": 0,
    }


def main():
    config = load_config()
    config, strategy_used = apply_strategy_from_env(config)
    race_id = resolve_race_id()
    budget_list = resolve_budget_list()

    odds_path = Path(os.environ.get("ODDS_PATH") or ODDS_PATH)
    pred_path = Path(os.environ.get("PRED_PATH") or PRED_PATH)
    wide_odds_path = Path(os.environ.get("WIDE_ODDS_PATH") or WIDE_ODDS_PATH)
    fuku_odds_path = Path(os.environ.get("FUKU_ODDS_PATH") or FUKU_ODDS_PATH)
    quinella_odds_path = Path(os.environ.get("QUINELLA_ODDS_PATH") or QUINELLA_ODDS_PATH)

    odds, preds, merged = load_inputs(odds_path, pred_path)
    if merged["horse_no"].isna().all() and race_id:
        run_paths = resolve_run_paths(race_id)
        if run_paths:
            alt_odds = Path(run_paths.get("odds_path", ""))
            alt_pred = Path(run_paths.get("predictions_path", ""))
            if alt_odds.exists():
                odds_path = alt_odds
            if alt_pred.exists():
                pred_path = alt_pred
            alt_wide = Path(run_paths.get("wide_odds_path", ""))
            if alt_wide.exists():
                wide_odds_path = alt_wide
            alt_fuku = Path(run_paths.get("fuku_odds_path", ""))
            if alt_fuku.exists():
                fuku_odds_path = alt_fuku
            alt_quinella = Path(run_paths.get("quinella_odds_path", ""))
            if alt_quinella.exists():
                quinella_odds_path = alt_quinella
            odds, preds, merged = load_inputs(odds_path, pred_path)
    if merged["horse_no"].isna().all():
        print("[ERROR] odds/predictions mismatch: horse_no missing for all rows.")
        print(f"odds: {odds_path}")
        print(f"preds: {pred_path}")
        return

    global WIDE_ODDS_MAP, FUKU_ODDS_MAP, QUINELLA_ODDS_MAP
    WIDE_ODDS_MAP = load_wide_odds_map(wide_odds_path)
    FUKU_ODDS_MAP = load_fuku_odds_map(fuku_odds_path)
    QUINELLA_ODDS_MAP = load_quinella_odds_map(quinella_odds_path)
    merged["odds_num"] = pd.to_numeric(merged.get("odds"), errors="coerce")
    prob_col = pick_prob_column(merged.columns)
    merged["Top3Prob_used"] = get_prob_series(merged, prob_col)

    missing = merged[merged["horse_no"].isna()]
    if not missing.empty:
        print("[WARN] Missing odds for:")
        for name in missing["HorseName"].tolist():
            print(" -", name)

    out_rows = []
    all_no_bet_rows = []
    for budget_yen in budget_list:
        tickets, trifecta_rec, no_bet_rows = build_plan(
            merged,
            budget_yen=budget_yen,
            config=config,
            race_id=race_id,
        )
        if any(row.get("no_bet_reason") == "pass_gate_soft" for row in no_bet_rows):
            print(f"[WARN] pass_gate soft failed ({budget_yen}); keeping tickets (low edge).")
        if any(row.get("no_bet_reason") == "pass_gate" for row in no_bet_rows):
            print(f"[WARN] pass_gate hard failed ({budget_yen}); tickets blocked (review only).")
        all_no_bet_rows.extend(no_bet_rows)
        out_rows.extend(tickets)
        if trifecta_rec:
            tri = dict(trifecta_rec)
            tri["budget_yen"] = int(budget_yen)
            out_rows.append(tri)

    if not out_rows:
        print("No tickets generated.")
        append_no_bet_logs(NO_BET_LOG_PATH, all_no_bet_rows)
        pause_exit()
        return

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    if strategy_used:
        print(f"Strategy: {strategy_used}")
    print(f"Budgets: {', '.join(str(v) for v in budget_list)}")
    print("Bet plan:")
    for _, row in out_df.iterrows():
        budget_yen = int(float(row.get("budget_yen", 0) or 0))
        line = (
            f"[{budget_yen}] {row['bet_type']}\t{row['horse_no']}\t{row['horse_name']}\t"
            f"{int(row['amount_yen'])} yen\t(exp~{int(row['expected_return_yen'])} yen)"
        )
        safe_print(line)
    for budget_yen in budget_list:
        df_budget = out_df[out_df["budget_yen"].astype(str) == str(budget_yen)]
        total_exp = int(df_budget["expected_return_yen"].sum()) if not df_budget.empty else 0
        print(f"Expected return (est., {budget_yen}): {total_exp} yen")
    print(f"Saved: {OUT_PATH}")
    append_no_bet_logs(NO_BET_LOG_PATH, all_no_bet_rows)
    pause_exit()


if __name__ == "__main__":
    main()

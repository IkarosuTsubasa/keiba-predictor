import csv
import argparse
import heapq
import itertools
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from bet_engine_v2 import generate_bet_plan_v2
from bet_engine_v3 import generate_bet_plan_v3
from bet_engine_v4 import generate_bet_plan_v4
from bet_engine_v5 import generate_bet_plan_v5
from bet_engine_v6 import generate_bet_plan_v6
try:
    from llm.gemini_policy import RacePolicyInput, call_gemini_policy, get_last_call_meta
except Exception:
    RacePolicyInput = None
    call_gemini_policy = None
    get_last_call_meta = None
from gemini_portfolio import build_history_context, extract_ledger_date, reserve_run_tickets, summarize_bankroll
from predictor_catalog import list_predictors
from surface_scope import get_data_dir, get_predictor_config_path, migrate_legacy_data

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
PREDICTOR_PATH_ENV_MAP = {
    "main": "PRED_PATH",
    "v2_opus": "PRED_PATH_V2_OPUS",
    "v3_premium": "PRED_PATH_V3_PREMIUM",
    "v4_gemini": "PRED_PATH_V4_GEMINI",
    "v5_stacking": "PRED_PATH_V5_STACKING",
}
PREDICTOR_PROFILE_HINTS = {
    "main": {
        "strengths_ja": [
            "従来型のバランス重視モデルで、全体の基準線として使いやすい。",
            "過去との比較がしやすく、他モデルとの差分確認に向いている。",
        ],
        "style_ja": "バランス型",
    },
    "v2_opus": {
        "strengths_ja": [
            "機械学習寄りの評価で、人気に寄りすぎない候補を拾いやすい。",
            "波乱寄りの目線を確認したい時の補助線として使いやすい。",
        ],
        "style_ja": "穴寄り探索型",
    },
    "v3_premium": {
        "strengths_ja": [
            "馬場や文脈を厚めに見て、条件差の影響を拾いやすい。",
            "補正を多めに入れた総合評価で、展開差の確認に向いている。",
        ],
        "style_ja": "文脈重視型",
    },
    "v4_gemini": {
        "strengths_ja": [
            "Top3 狙いの順位付けと条件適性の両方を見やすい。",
            "コースや距離、馬場を横断して整合のある予測を作りやすい。",
        ],
        "style_ja": "適性ハイブリッド型",
    },
    "v5_stacking": {
        "strengths_ja": [
            "複数モデルのスタッキングと ranker を組み合わせ、上位評価の整合を取りやすい。",
            "オッズや出走文脈も併用し、人気だけに寄りすぎない総合評価を作りやすい。",
        ],
        "style_ja": "スタッキング統合型",
    },
}

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

BET_ENGINE_V2_DEFAULTS = {
    "p_mix_w": 0.6,
    "rank_temperature": 1.0,
    "K_place": 3.0,
    "place_bias": 0.0,
    "place_power": 1.0,
    "C_wide": 1.6,
    "C_quinella": 1.6,
    "odds_penalty": 0.0,
    "win_odds_penalty": 0.0,
    "place_odds_penalty": 0.02,
    "wide_odds_penalty": 0.05,
    "quinella_odds_penalty": 0.0,
    "kelly_scale": 0.25,
    "min_ev_per_ticket": 0.02,
    "min_p_hit_threshold": 0.05,
    "max_ticket_share": 0.20,
    "max_race_share": 0.40,
    "min_yen_unit": 100,
    "max_single_horses": 8,
    "max_tickets_per_race": 0,
}

BET_ENGINE_V3_DEFAULTS = {
    "enabled": True,
    "p_mix_w": 0.6,
    "rank_temperature": 1.0,
    "p_mid_odds_threshold": 0.18,
    "N_rank": 12,
    "N_value": 12,
    "target_risk_share": 0.25,
    "kelly_scale": 1.0,
    "odds_power": 0.70,
    "max_ticket_share": 0.15,
    "min_yen_unit": 100,
    "min_p_hit_per_ticket": 0.04,
    "min_p_win_per_ticket": 0.03,
    "rank_weight_floor": 0.55,
    "rank_weight_ceil": 1.00,
    "min_edge_per_ticket": 0.00,
    "fallback_max_odds_place": 15.0,
    "high_bucket_odds_threshold": 10.0,
    "high_exposure_cap_share": 0.15,
    "low_mid_min_share": 0.60,
    "exposure_enforcement_mode": "trim",
    "max_high_odds_tickets_per_race": 1,
    "min_low_or_mid_presence": True,
    "min_ev": {"win": 0.03, "place": 0.01, "wide": 0.01, "quinella": 0.02},
    "min_p": {"win": 0.03, "place": 0.05, "wide": 0.04, "quinella": 0.04},
    "penalty": {"win": 0.00, "place": 0.02, "wide": 0.02, "quinella": 0.00},
}

BET_ENGINE_V4_DEFAULTS = {
    "enabled": True,
    "race_budget_share": 0.02,
    "max_tickets_per_race": 3,
    "high_odds_threshold": 10.0,
    "max_high_odds_per_race": 1,
    "ensure_diversity": True,
    "min_yen_unit": 100,
}

BET_ENGINE_V5_DEFAULTS = {
    "enabled": True,
    "target_risk_share": 0.02,
    "min_race_budget": 400,
    "min_yen_unit": 100,
    "base_lambda_market": 0.20,
    "lambda_min": 0.05,
    "lambda_max": 0.45,
    "odds_power": 0.72,
    "ev_margin": 0.01,
    "min_ev_per_ticket": -0.02,
    "kelly_scale": 0.45,
    "f_cap_by_type": {"win": 0.25, "place": 0.25, "pair": 0.20},
    "min_p_by_type": {"win": 0.02, "place": 0.05, "wide": 0.03, "quinella": 0.03},
    "max_tickets_per_race": 6,
    "ensure_diversity": True,
}

CALIBRATION_DEFAULTS = {"win_temp": 1.0, "enabled": True}

BET_ENGINE_V3_PROFILE_OVERRIDES = {
    "publish": {
        "kelly_scale": 1.0,
        "min_p_hit_per_ticket": 0.04,
        "min_p_win_per_ticket": 0.03,
        "min_edge_per_ticket": 0.00,
        "fallback_max_odds_place": 15.0,
    },
    "conservative": {
        "kelly_scale": 1.0,
        "min_p_hit_per_ticket": 0.04,
        "min_p_win_per_ticket": 0.03,
        "min_edge_per_ticket": 0.00,
        "fallback_max_odds_place": 10.0,
    },
}

BET_ENGINE_V5_PROFILE_OVERRIDES = {
    "default": {},
    "conservative": {
        "target_risk_share": 0.06,
        "min_race_budget": 0,
        "min_ev_per_ticket": 0.00,
        "ev_margin": 0.02,
        "min_p_by_type": {"win": 0.03, "place": 0.06, "wide": 0.04, "quinella": 0.04},
        "kelly_scale": 0.20,
        "f_cap_by_type": {"win": 0.10, "place": 0.14, "pair": 0.12},
        "base_lambda_market": 0.18,
        "lambda_min": 0.05,
        "lambda_max": 0.35,
        "odds_power": 0.75,
    },
    "conservative2": {
        "target_risk_share": 0.08,
        "min_race_budget": 200,
        "min_ev_per_ticket": -0.02,
        "ev_margin": 0.01,
        "min_p_by_type": {"win": 0.03, "place": 0.06, "wide": 0.04, "quinella": 0.04},
        "kelly_scale": 0.18,
        "f_cap_by_type": {"win": 0.10, "place": 0.14, "pair": 0.12},
        "base_lambda_market": 0.18,
        "lambda_min": 0.05,
        "lambda_max": 0.40,
        "odds_power": 0.75,
    },
    "conservative3": {
        "target_risk_share": 0.06,
        "min_race_budget": 0,
        "min_ev_per_ticket": 0.01,
        "ev_margin": 0.02,
        "base_lambda_market": 0.30,
        "lambda_min": 0.20,
        "lambda_max": 0.60,
        "odds_power": 0.80,
        "kelly_scale": 0.12,
        "min_p_by_type": {"win": 0.03, "place": 0.06, "wide": 0.04, "quinella": 0.04},
    },
    "conservative4": {
        "target_risk_share": 0.06,
        "min_race_budget": 0,
        "min_ev_per_ticket": 0.01,
        "ev_margin": 0.02,
        "base_lambda_market": 0.30,
        "lambda_min": 0.20,
        "lambda_max": 0.60,
        "odds_power": 0.80,
        "kelly_scale": 0.12,
        "min_p_by_type": {"win": 0.03, "place": 0.06, "wide": 0.04, "quinella": 0.04},
        "value_gate_enabled": True,
        "value_min_by_type": {"win": 0.06, "place": 0.08, "pair": 0.04},
        "value_gate_gap_boost_k": 2.0,
        "gap_for_boost": 0.06,
    },
    "conservative5": {
        "target_risk_share": 0.06,
        "min_race_budget": 0,
        "min_ev_per_ticket": 0.01,
        "ev_margin": 0.02,
        "base_lambda_market": 0.30,
        "lambda_min": 0.20,
        "lambda_max": 0.60,
        "odds_power": 0.80,
        "kelly_scale": 0.12,
        "min_p_by_type": {"win": 0.03, "place": 0.06, "wide": 0.04, "quinella": 0.04},
        "value_gate_enabled": True,
        "value_gate_enabled_by_type": {"win": True, "place": False, "pair": False},
        "takeout_mult_by_type": {"win": 1.10, "place": 1.15, "pair": 1.20},
        "value_ratio_min_by_type": {"win": 0.20, "place": 0.25, "pair": 0.15},
        "value_gate_gap_boost_k": 2.0,
        "gap_for_boost": 0.06,
        "value_entry_gate_enabled": True,
        "min_best_value_ratio_to_bet": 0.15,
    },
}

BET_ENGINE_V3_AUDIT_KEYS = [
    "kelly_scale",
    "min_p_hit_per_ticket",
    "min_p_win_per_ticket",
    "min_edge_per_ticket",
    "fallback_max_odds_place",
    "high_exposure_cap_share",
    "low_mid_min_share",
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
        "predictions_v2_opus_path": run.get("predictions_v2_opus_path", ""),
        "predictions_v3_premium_path": run.get("predictions_v3_premium_path", ""),
        "predictions_v4_gemini_path": run.get("predictions_v4_gemini_path", ""),
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


def load_predictor_config():
    path = get_predictor_config_path(BASE_DIR, SCOPE_KEY)
    if not path.exists():
        fallback = []
        if SCOPE_KEY == "central_turf":
            fallback.extend(
                [
                    BASE_DIR / "predictor_config_turf_default.json",
                    BASE_DIR / "predictor_config_turf.json",
                ]
            )
        elif SCOPE_KEY == "central_dirt":
            fallback.extend(
                [
                    BASE_DIR / "predictor_config_dirt_default.json",
                    BASE_DIR / "predictor_config_dirt.json",
                ]
            )
        elif SCOPE_KEY == "local":
            fallback.extend(
                [
                    BASE_DIR / "predictor_config_central_dirt.json",
                    BASE_DIR / "predictor_config_dirt_default.json",
                    BASE_DIR / "predictor_config_dirt.json",
                ]
            )
        fallback.append(BASE_DIR / "predictor_config.json")
        for legacy in fallback:
            if legacy.exists():
                data = json.loads(legacy.read_text(encoding="utf-8"))
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return ensure_predictor_bet_config_defaults(path, data)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return ensure_predictor_bet_config_defaults(path, data)
    except Exception:
        return ensure_predictor_bet_config_defaults(path, {})


def get_bet_engine_v2_config(predictor_config):
    cfg = dict(BET_ENGINE_V2_DEFAULTS)
    node = predictor_config.get("bet_engine_v2", {}) if isinstance(predictor_config, dict) else {}
    if isinstance(node, dict):
        cfg.update(node)
    return cfg


def get_bet_engine_v3_config(predictor_config):
    cfg = dict(BET_ENGINE_V3_DEFAULTS)
    node = predictor_config.get("bet_engine_v3", {}) if isinstance(predictor_config, dict) else {}
    if isinstance(node, dict):
        cfg.update(node)
    calibration = predictor_config.get("calibration", {}) if isinstance(predictor_config, dict) else {}
    if isinstance(calibration, dict):
        cfg["win_temp"] = float(calibration.get("win_temp", cfg.get("win_temp", 1.0)))
        cfg["calibration_enabled"] = bool(calibration.get("enabled", True))
    return cfg


def get_bet_engine_v4_config(predictor_config):
    cfg = dict(BET_ENGINE_V4_DEFAULTS)
    node = predictor_config.get("bet_engine_v4", {}) if isinstance(predictor_config, dict) else {}
    if isinstance(node, dict):
        cfg.update(node)
    return cfg


def get_bet_engine_v5_config(predictor_config):
    cfg = dict(BET_ENGINE_V5_DEFAULTS)
    node = predictor_config.get("bet_engine_v5", {}) if isinstance(predictor_config, dict) else {}
    if isinstance(node, dict):
        cfg.update(node)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Build bet plan with v2/v3/v4/v5/v6 engines.")
    parser.add_argument(
        "--engine-version",
        choices=["v2", "v3", "v4", "v5", "v6"],
        default=os.environ.get("ENGINE_VERSION", "v6"),
        help="bet engine version",
    )
    parser.add_argument(
        "--bet-profile",
        choices=sorted(BET_ENGINE_V3_PROFILE_OVERRIDES.keys()),
        default=os.environ.get("BET_PROFILE", "publish"),
        help="bet_engine_v3 preset profile (used only when engine-version=v3)",
    )
    parser.add_argument(
        "--v5-profile",
        choices=sorted(BET_ENGINE_V5_PROFILE_OVERRIDES.keys()),
        default=os.environ.get("V5_PROFILE", "default"),
        help="bet_engine_v5 preset profile (used only when engine-version=v5)",
    )
    parser.add_argument(
        "--policy-engine",
        choices=["none", "gemini"],
        default=os.environ.get("POLICY_ENGINE", "none"),
        help="policy engine selector",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        help="Gemini model name for policy layer",
    )
    parser.add_argument(
        "--policy-cache-enable",
        default=os.environ.get("POLICY_CACHE_ENABLE", "true"),
        help="enable/disable policy cache (true/false)",
    )
    parser.add_argument(
        "--policy-budget-reuse",
        default=os.environ.get("POLICY_BUDGET_REUSE", "false"),
        help="reuse gemini policy across budget tiers (true/false)",
    )
    return parser.parse_args()


def resolve_bet_profile(value):
    profile = str(value or "").strip().lower()
    if profile not in BET_ENGINE_V3_PROFILE_OVERRIDES:
        profile = "publish"
    return profile


def resolve_v5_profile(value):
    profile = str(value or "").strip().lower()
    if profile not in BET_ENGINE_V5_PROFILE_OVERRIDES:
        profile = "default"
    return profile


def resolve_engine_version(value):
    v = str(value or "").strip().lower()
    if v not in ("v2", "v3", "v4", "v5"):
        v = "v4"
    return v


def parse_bool_text(value, default=True):
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def apply_bet_profile_to_v3(cfg, profile):
    out = dict(cfg) if isinstance(cfg, dict) else {}
    overrides = BET_ENGINE_V3_PROFILE_OVERRIDES.get(resolve_bet_profile(profile), {})
    out.update(overrides)
    return out


def apply_bet_profile_to_v5(cfg, profile):
    out = dict(cfg) if isinstance(cfg, dict) else {}
    overrides = BET_ENGINE_V5_PROFILE_OVERRIDES.get(resolve_v5_profile(profile), {})
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            node = dict(out[key])
            node.update(value)
            out[key] = node
        else:
            out[key] = value
    return out


def build_bet_engine_v3_audit_summary(cfg):
    out = {}
    for key in BET_ENGINE_V3_AUDIT_KEYS:
        out[key] = cfg.get(key)
    return out


def canonical_horse_key(value):
    horse_no = parse_horse_no(value)
    if horse_no is not None:
        return str(horse_no)
    text = str(value or "").strip()
    return text


def canonical_pair(a, b):
    a_str = str(a)
    b_str = str(b)
    a_no = parse_horse_no(a_str)
    b_no = parse_horse_no(b_str)
    if a_no is not None and b_no is not None:
        if a_no <= b_no:
            return str(a_no), str(b_no)
        return str(b_no), str(a_no)
    if a_str <= b_str:
        return a_str, b_str
    return b_str, a_str


def build_place_odds_payload(path):
    out = {}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return out
    if "horse_no" not in df.columns:
        return out
    for _, row in df.iterrows():
        horse_no = parse_horse_no(row.get("horse_no"))
        if horse_no is None:
            continue
        key = str(horse_no)
        low = safe_float(row.get("odds_low"))
        high = safe_float(row.get("odds_high"))
        mid = safe_float(row.get("odds_mid"))
        if low > 0 and high > 0:
            out[key] = (min(low, high), max(low, high))
        elif low > 0:
            out[key] = low
        elif mid > 0:
            out[key] = mid
    return out


def build_pair_odds_payload(path):
    out = {}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return out
    if "horse_no_a" not in df.columns or "horse_no_b" not in df.columns:
        return out
    for _, row in df.iterrows():
        a = parse_horse_no(row.get("horse_no_a"))
        b = parse_horse_no(row.get("horse_no_b"))
        if a is None or b is None or a == b:
            continue
        key = canonical_pair(a, b)
        low = safe_float(row.get("odds_low"))
        high = safe_float(row.get("odds_high"))
        mid = safe_float(row.get("odds_mid"))
        single = safe_float(row.get("odds"))
        if low > 0 and high > 0:
            out[key] = (min(low, high), max(low, high))
        elif low > 0:
            out[key] = low
        elif mid > 0:
            out[key] = mid
        elif single > 0:
            out[key] = single
    return out


def build_bet_engine_v2_inputs(merged, fuku_odds_path, wide_odds_path, quinella_odds_path):
    d = merged.copy()
    d["horse_key"] = d["horse_no"].apply(canonical_horse_key)
    d["rank_score"] = pd.to_numeric(d.get("rank_score"), errors="coerce")
    d["Top3Prob_model"] = pd.to_numeric(d.get("Top3Prob_model"), errors="coerce")
    if d["rank_score"].isna().all():
        d["rank_score"] = d["Top3Prob_model"]
    d["Top3Prob_model"] = d["Top3Prob_model"].fillna(0.0)
    d["rank_score"] = d["rank_score"].fillna(d["Top3Prob_model"])
    d["race_id"] = "__single_race__"
    pred_df = d[["horse_key", "rank_score", "Top3Prob_model", "HorseName", "horse_no", "race_id"]].copy()

    win_map = {}
    for _, row in d.iterrows():
        key = str(row.get("horse_key", "")).strip()
        if not key:
            continue
        odds_val = safe_float(row.get("odds_num"))
        if odds_val > 0:
            win_map[key] = odds_val

    place_map = build_place_odds_payload(fuku_odds_path)
    wide_map = build_pair_odds_payload(wide_odds_path)
    quinella_map = build_pair_odds_payload(quinella_odds_path)

    odds_payload = {
        "win": win_map,
        "place": place_map,
        "wide": wide_map,
        "quinella": quinella_map,
    }
    return pred_df, odds_payload


def ensure_predictor_bet_config_defaults(cfg_path, predictor_config):
    data = predictor_config if isinstance(predictor_config, dict) else {}
    changed = False

    node_v2 = data.get("bet_engine_v2")
    if not isinstance(node_v2, dict):
        node_v2 = {}
        changed = True
    for key, value in BET_ENGINE_V2_DEFAULTS.items():
        if key not in node_v2:
            node_v2[key] = value
            changed = True
    data["bet_engine_v2"] = node_v2

    node_v3 = data.get("bet_engine_v3")
    if not isinstance(node_v3, dict):
        node_v3 = {}
        changed = True
    for key, value in BET_ENGINE_V3_DEFAULTS.items():
        if key not in node_v3:
            node_v3[key] = value
            changed = True
    data["bet_engine_v3"] = node_v3

    node_v4 = data.get("bet_engine_v4")
    if not isinstance(node_v4, dict):
        node_v4 = {}
        changed = True
    for key, value in BET_ENGINE_V4_DEFAULTS.items():
        if key not in node_v4:
            node_v4[key] = value
            changed = True
    data["bet_engine_v4"] = node_v4

    node_v5 = data.get("bet_engine_v5")
    if not isinstance(node_v5, dict):
        node_v5 = {}
        changed = True
    for key, value in BET_ENGINE_V5_DEFAULTS.items():
        if key not in node_v5:
            node_v5[key] = value
            changed = True
    data["bet_engine_v5"] = node_v5

    calibration = data.get("calibration")
    if not isinstance(calibration, dict):
        calibration = {}
        changed = True
    for key, value in CALIBRATION_DEFAULTS.items():
        if key not in calibration:
            calibration[key] = value
            changed = True
    data["calibration"] = calibration

    if changed:
        try:
            cfg_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
    return data


def build_horse_meta_map(merged):
    meta = {}
    for _, row in merged.iterrows():
        key = canonical_horse_key(row.get("horse_no", ""))
        if not key:
            continue
        horse_no = parse_horse_no(row.get("horse_no", ""))
        horse_no_text = str(horse_no) if horse_no is not None else str(key)
        horse_name = str(row.get("HorseName", "")).strip()
        meta[key] = {"horse_no": horse_no_text, "horse_name": horse_name}
    return meta


def build_rows_from_bet_plan_v2(result, budget_yen, horse_meta):
    out_rows = []
    item_rows = []
    for item in result.items:
        horse_keys = [str(h) for h in item.horses]
        horse_no_labels = []
        horse_names = []
        for hk in horse_keys:
            meta = horse_meta.get(hk, {})
            horse_no_labels.append(str(meta.get("horse_no", hk)))
            name = str(meta.get("horse_name", "")).strip()
            if name:
                horse_names.append(name)

        horse_no = "-".join(horse_no_labels)
        horse_name = " / ".join(horse_names)
        amount_yen = int(item.stake_yen)
        units = int(amount_yen // UNIT_YEN) if UNIT_YEN > 0 else 0
        expected_return = int(round(amount_yen * float(item.p_hit) * float(item.odds_used)))
        ev_ratio_est = float(item.edge) + 1.0
        row = {
            "budget_yen": int(budget_yen),
            "bet_type": item.bet_type,
            "horse_no": horse_no,
            "horse_name": horse_name,
            "units": units,
            "amount_yen": amount_yen,
            "hit_prob_est": round(float(item.p_hit), 4),
            "hit_prob_se": "",
            "hit_prob_ci95_low": "",
            "hit_prob_ci95_high": "",
            "payout_mult": round(float(item.odds_used), 4),
            "ev_ratio_est": round(float(ev_ratio_est), 4),
            "expected_return_yen": expected_return,
            "odds_used": round(float(item.odds_used), 6),
            "p_hit": round(float(item.p_hit), 6),
            "edge": round(float(item.edge), 6),
            "kelly_f": round(float(item.kelly_f), 6),
            "score": round(float(item.edge), 6),
            "stake_yen": amount_yen,
            "notes": str(item.notes or ""),
            "strategy_text_ja": "",
            "bet_tendency_ja": "",
            "policy_engine": "",
            "policy_buy_style": "",
            "policy_bet_decision": "",
            "policy_construction_style": "",
        }
        out_rows.append(row)
        item_rows.append(
            {
                "budget_yen": int(budget_yen),
                "bet_type": item.bet_type,
                "horses": horse_no,
                "odds_used": round(float(item.odds_used), 6),
                "p_hit": round(float(item.p_hit), 6),
                "edge": round(float(item.edge), 6),
                "stake_yen": amount_yen,
                "notes": str(item.notes or ""),
                "score": round(float(item.edge), 6),
                "strategy_text_ja": "",
                "bet_tendency_ja": "",
                "policy_engine": "",
                "policy_buy_style": "",
                "policy_bet_decision": "",
                "policy_construction_style": "",
            }
        )
    return out_rows, item_rows


def build_rows_from_bet_items(items, budget_yen, horse_meta):
    out_rows = []
    item_rows = []
    for item in items:
        bet_type = str(item.get("bet_type", "")).strip()
        horses = item.get("horses", ())
        if isinstance(horses, str):
            horses = tuple([x for x in str(horses).split("-") if x])
        horse_keys = [str(h) for h in horses]
        horse_no_labels = []
        horse_names = []
        for hk in horse_keys:
            meta = horse_meta.get(hk, {})
            horse_no_labels.append(str(meta.get("horse_no", hk)))
            name = str(meta.get("horse_name", "")).strip()
            if name:
                horse_names.append(name)
        horse_no = "-".join(horse_no_labels)
        horse_name = " / ".join(horse_names)
        amount_yen = int(item.get("stake_yen", 0) or 0)
        if amount_yen <= 0:
            continue
        units = int(amount_yen // UNIT_YEN) if UNIT_YEN > 0 else 0
        p_hit = float(item.get("p_hit", 0.0) or 0.0)
        odds_used = float(item.get("odds_used", 0.0) or 0.0)
        edge = float(item.get("edge", 0.0) or 0.0)
        kelly_f = float(item.get("kelly_f", 0.0) or 0.0)
        score = float(item.get("score", edge) or edge)
        expected_return = int(round(amount_yen * p_hit * odds_used))
        ev_ratio_est = edge + 1.0
        why = str(item.get("why", "") or "")
        notes = str(item.get("notes", "") or "")
        strategy_text_ja = str(item.get("strategy_text_ja", "") or "")
        bet_tendency_ja = str(item.get("bet_tendency_ja", "") or "")
        policy_engine = str(item.get("policy_engine", "") or "")
        policy_buy_style = str(item.get("policy_buy_style", "") or "")
        policy_bet_decision = str(item.get("policy_bet_decision", "") or "")
        policy_construction_style = str(item.get("policy_construction_style", "") or "")
        merged_note = why if (why and not notes) else (notes if notes else "")
        if why and notes:
            merged_note = f"{why};{notes}"
        row = {
            "budget_yen": int(budget_yen),
            "bet_type": bet_type,
            "horse_no": horse_no,
            "horse_name": horse_name,
            "units": units,
            "amount_yen": amount_yen,
            "hit_prob_est": round(p_hit, 4),
            "hit_prob_se": "",
            "hit_prob_ci95_low": "",
            "hit_prob_ci95_high": "",
            "payout_mult": round(odds_used, 4),
            "ev_ratio_est": round(ev_ratio_est, 4),
            "expected_return_yen": expected_return,
            "odds_used": round(odds_used, 6),
            "p_hit": round(p_hit, 6),
            "edge": round(edge, 6),
            "kelly_f": round(kelly_f, 6),
            "score": round(score, 6),
            "stake_yen": amount_yen,
            "notes": merged_note,
            "strategy_text_ja": strategy_text_ja,
            "bet_tendency_ja": bet_tendency_ja,
            "policy_engine": policy_engine,
            "policy_buy_style": policy_buy_style,
            "policy_bet_decision": policy_bet_decision,
            "policy_construction_style": policy_construction_style,
        }
        out_rows.append(row)
        item_rows.append(
            {
                "budget_yen": int(budget_yen),
                "bet_type": bet_type,
                "horses": horse_no,
                "odds_used": round(odds_used, 6),
                "p_hit": round(p_hit, 6),
                "edge": round(edge, 6),
                "stake_yen": amount_yen,
                "notes": merged_note,
                "score": round(score, 6),
                "strategy_text_ja": strategy_text_ja,
                "bet_tendency_ja": bet_tendency_ja,
                "policy_engine": policy_engine,
                "policy_buy_style": policy_buy_style,
                "policy_bet_decision": policy_bet_decision,
                "policy_construction_style": policy_construction_style,
            }
        )
    return out_rows, item_rows


def normalize_portfolio_items(items):
    out = []
    for item in (items or []):
        if not isinstance(item, dict):
            continue
        bet_type = str(item.get("bet_type", "") or item.get("ticket_type", "")).strip().lower()
        horses = item.get("horses", item.get("horse_ids", ()))
        if isinstance(horses, str):
            horses = tuple([x for x in str(horses).split("-") if x])
        horses = tuple([str(x) for x in (horses or ()) if str(x).strip() != ""])
        if not bet_type or not horses:
            continue
        stake_yen = int(float(item.get("stake_yen", item.get("stake", 0)) or 0))
        odds_used = float(item.get("odds_used", item.get("odds", 0.0)) or 0.0)
        p_hit = float(item.get("p_hit", item.get("p_final", 0.0)) or 0.0)
        edge = float(item.get("edge", item.get("EV_adj", item.get("EV", 0.0))) or 0.0)
        kelly_f = float(item.get("kelly_f", 0.0) or 0.0)
        score = float(item.get("score", edge) or edge)
        why = str(item.get("why", "") or "")
        notes = str(item.get("notes", "") or "")
        out.append(
            {
                "bet_type": bet_type,
                "horses": horses,
                "stake_yen": stake_yen,
                "odds_used": odds_used,
                "p_hit": p_hit,
                "edge": edge,
                "kelly_f": kelly_f,
                "score": score,
                "why": why,
                "notes": notes,
                "strategy_text_ja": str(item.get("strategy_text_ja", "") or ""),
                "bet_tendency_ja": str(item.get("bet_tendency_ja", "") or ""),
                "policy_engine": str(item.get("policy_engine", "") or ""),
                "policy_buy_style": str(item.get("policy_buy_style", "") or ""),
                "policy_bet_decision": str(item.get("policy_bet_decision", "") or ""),
                "policy_construction_style": str(item.get("policy_construction_style", "") or ""),
            }
        )
    return out


def calc_plan_summary(item_rows):
    total_stake = int(sum(int(r.get("stake_yen", 0) or 0) for r in item_rows))
    ticket_count = int(len(item_rows))
    edge_sum = float(sum(float(r.get("edge", 0.0) or 0.0) for r in item_rows))
    weighted_edge = float(
        sum(float(r.get("edge", 0.0) or 0.0) * int(r.get("stake_yen", 0) or 0) for r in item_rows)
    )
    return total_stake, ticket_count, edge_sum, weighted_edge


def clamp01(value):
    v = safe_float(value)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def normalize_policy_weights(raw):
    d = dict(raw or {})
    win = max(0.0, safe_float(d.get("win", 0.0)))
    place = max(0.0, safe_float(d.get("place", 0.0)))
    pair = max(0.0, safe_float(d.get("pair", 0.0)))
    total = win + place + pair
    if total <= 1e-12:
        return {"win": 0.20, "place": 0.50, "pair": 0.30}
    return {"win": win / total, "place": place / total, "pair": pair / total}


def policy_group_for_type(bet_type):
    t = str(bet_type or "").strip().lower()
    if t == "win":
        return "win"
    if t == "place":
        return "place"
    return "pair"


def stable_ticket_legs(bet_type, horses):
    t = str(bet_type or "").strip().lower()
    legs = [str(x).strip() for x in (horses or ()) if str(x).strip()]
    if t in ("wide", "quinella") and len(legs) >= 2:
        a, b = canonical_pair(legs[0], legs[1])
        return [str(a), str(b)]
    if legs:
        return [str(canonical_horse_key(legs[0]))]
    return []


def build_policy_ticket_id(bet_type, horses):
    t = str(bet_type or "").strip().lower()
    legs = stable_ticket_legs(t, horses)
    return f"{t}:{'-'.join(legs)}"


def build_items_from_v2_result(result):
    out = []
    if result is None:
        return out
    for item in list(getattr(result, "items", []) or []):
        horses = tuple(str(x) for x in (getattr(item, "horses", ()) or ()))
        p_hit = float(getattr(item, "p_hit", 0.0) or 0.0)
        edge = float(getattr(item, "edge", 0.0) or 0.0)
        score = edge * math.sqrt(max(p_hit, 0.0))
        out.append(
            {
                "bet_type": str(getattr(item, "bet_type", "")).strip().lower(),
                "horses": horses,
                "stake_yen": int(getattr(item, "stake_yen", 0) or 0),
                "odds_used": float(getattr(item, "odds_used", 0.0) or 0.0),
                "p_hit": p_hit,
                "edge": edge,
                "kelly_f": float(getattr(item, "kelly_f", 0.0) or 0.0),
                "score": float(score),
                "why": "",
                "notes": str(getattr(item, "notes", "") or ""),
            }
        )
    return out


def policy_candidate_sort_key(item):
    return (
        -float(item.get("score", item.get("edge", 0.0)) or 0.0),
        -float(item.get("edge", 0.0) or 0.0),
        -float(item.get("p_hit", 0.0) or 0.0),
        float(item.get("odds_used", 0.0) or 0.0),
        str(item.get("bet_type", "") or ""),
        "-".join([str(x) for x in (item.get("horses", ()) or ())]),
    )


def _softmax_probs(values, temperature=1.0):
    nums = [float(v or 0.0) for v in list(values or [])]
    if not nums:
        return []
    scale = max(1e-6, float(temperature or 1.0))
    shifted = [v / scale for v in nums]
    mx = max(shifted)
    exps = [math.exp(v - mx) for v in shifted]
    total = float(sum(exps))
    if total <= 1e-12:
        return [1.0 / float(len(nums)) for _ in nums]
    return [float(v / total) for v in exps]


def policy_odds_scalar(value):
    if isinstance(value, (list, tuple)):
        nums = [safe_float(x) for x in list(value or []) if safe_float(x) > 0]
        if nums:
            return float(sum(nums) / float(len(nums)))
        return 0.0
    return safe_float(value)


def build_policy_candidates_from_items(items, per_type_caps=None):
    caps = per_type_caps or {"win": 5, "place": 5, "wide": 10, "quinella": 10}
    bucketed = {"win": [], "place": [], "wide": [], "quinella": []}
    item_by_id = {}
    for item in list(items or []):
        bet_type = str(item.get("bet_type", "")).strip().lower()
        horses = tuple(str(x) for x in (item.get("horses", ()) or ()))
        if bet_type not in bucketed or not horses:
            continue
        cid = build_policy_ticket_id(bet_type, horses)
        legs = stable_ticket_legs(bet_type, horses)
        score = float(item.get("score", item.get("edge", 0.0)) or 0.0)
        candidate = {
            "id": cid,
            "bet_type": bet_type,
            "legs": legs,
            "odds_used": float(item.get("odds_used", 0.0) or 0.0),
            "p_hit": float(item.get("p_hit", 0.0) or 0.0),
            "ev": float(item.get("edge", 0.0) or 0.0),
            "score": score,
        }
        prev = item_by_id.get(cid)
        if prev is None or policy_candidate_sort_key(item) < policy_candidate_sort_key(prev):
            norm = dict(item)
            norm["bet_type"] = bet_type
            norm["horses"] = horses
            norm["score"] = score
            item_by_id[cid] = norm
            if prev is None:
                bucketed[bet_type].append(candidate)
            else:
                for idx, old in enumerate(bucketed[bet_type]):
                    if str(old.get("id", "")) == cid:
                        bucketed[bet_type][idx] = candidate
                        break
    candidates = []
    for bet_type in ("win", "place", "wide", "quinella"):
        rows = sorted(bucketed[bet_type], key=lambda x: policy_candidate_sort_key(item_by_id.get(str(x.get("id", "")), {})))
        cap = int(caps.get(bet_type, len(rows)) or len(rows))
        candidates.extend(rows[:cap])
    candidates.sort(key=lambda x: (-float(x.get("score", 0.0) or 0.0), str(x.get("id", ""))))
    return candidates, item_by_id


def build_policy_candidate_pool(merged, odds_payload, per_type_caps=None):
    if merged is None or merged.empty:
        return [], {}
    d = merged.copy()
    d["horse_key"] = d.get("horse_no").apply(canonical_horse_key)
    d = d[d["horse_key"].fillna("").astype(str).str.strip() != ""].copy()
    if d.empty:
        return [], {}
    if "Top3Prob_model" not in d.columns:
        d["Top3Prob_model"] = pd.to_numeric(d.get("Top3Prob_used"), errors="coerce").fillna(0.0)
    d["Top3Prob_model"] = pd.to_numeric(d["Top3Prob_model"], errors="coerce").fillna(0.0)
    if "rank_score" not in d.columns:
        d["rank_score"] = d["Top3Prob_model"]
    d["rank_score"] = pd.to_numeric(d["rank_score"], errors="coerce").fillna(d["Top3Prob_model"])
    d = d.sort_values(["Top3Prob_model", "rank_score"], ascending=False).reset_index(drop=True)

    rank_vals = d["rank_score"].astype(float).tolist()
    rank_mean = sum(rank_vals) / float(len(rank_vals)) if rank_vals else 0.0
    rank_var = sum((x - rank_mean) ** 2 for x in rank_vals) / float(len(rank_vals)) if rank_vals else 0.0
    rank_std = math.sqrt(rank_var) if rank_var > 1e-12 else 0.0
    z_scores = [((x - rank_mean) / rank_std) if rank_std > 1e-12 else 0.0 for x in rank_vals]
    p_rank = _softmax_probs([0.8 * z for z in z_scores], temperature=1.0)

    top3_raw = [clamp01(x) for x in d["Top3Prob_model"].astype(float).tolist()]
    top3_sum = float(sum(top3_raw))
    if top3_sum <= 1e-12:
        p_top3 = [1.0 / float(len(top3_raw)) for _ in top3_raw]
    else:
        p_top3 = [float(x / top3_sum) for x in top3_raw]
    p_win_mix = [(0.55 * pr) + (0.45 * pt) for pr, pt in zip(p_rank, p_top3)]
    win_mix_sum = float(sum(p_win_mix))
    if win_mix_sum <= 1e-12:
        p_win = [1.0 / float(len(p_win_mix)) for _ in p_win_mix]
    else:
        p_win = [float(x / win_mix_sum) for x in p_win_mix]
    p_place = [clamp01((0.68 * top3) + (0.32 * min(1.0, pw * 2.2))) for top3, pw in zip(top3_raw, p_win)]

    horse_rows = []
    for row, pw, pp in zip(d.to_dict("records"), p_win, p_place):
        horse_key = str(row.get("horse_key", "")).strip()
        if not horse_key:
            continue
        horse_rows.append(
            {
                "horse_key": horse_key,
                "p_win": float(pw),
                "p_place": float(pp),
            }
        )
    if not horse_rows:
        return [], {}

    item_pool = []
    for row in horse_rows:
        horse_key = row["horse_key"]
        win_odds = policy_odds_scalar((odds_payload.get("win", {}) or {}).get(horse_key, 0.0))
        if win_odds > 1.0:
            p_hit = float(row["p_win"])
            edge = float((p_hit * win_odds) - 1.0)
            item_pool.append(
                {
                    "bet_type": "win",
                    "horses": (horse_key,),
                    "stake_yen": 0,
                    "odds_used": win_odds,
                    "p_hit": p_hit,
                    "edge": edge,
                    "kelly_f": 0.0,
                    "score": float(edge * math.sqrt(max(1e-6, p_hit))),
                    "why": "",
                    "notes": "policy_pool=expanded",
                }
            )
        place_odds = policy_odds_scalar((odds_payload.get("place", {}) or {}).get(horse_key, 0.0))
        if place_odds > 1.0:
            p_hit = float(row["p_place"])
            edge = float((p_hit * place_odds) - 1.0)
            item_pool.append(
                {
                    "bet_type": "place",
                    "horses": (horse_key,),
                    "stake_yen": 0,
                    "odds_used": place_odds,
                    "p_hit": p_hit,
                    "edge": edge,
                    "kelly_f": 0.0,
                    "score": float(edge * math.sqrt(max(1e-6, p_hit))),
                    "why": "",
                    "notes": "policy_pool=expanded",
                }
            )

    for i, left in enumerate(horse_rows):
        for right in horse_rows[i + 1 :]:
            a, b = canonical_pair(left["horse_key"], right["horse_key"])
            p_wide = clamp01(float(left["p_place"]) * float(right["p_place"]) * 0.78)
            wide_odds = policy_odds_scalar((odds_payload.get("wide", {}) or {}).get((a, b), 0.0))
            if wide_odds > 1.0:
                edge = float((p_wide * wide_odds) - 1.0)
                item_pool.append(
                    {
                        "bet_type": "wide",
                        "horses": (a, b),
                        "stake_yen": 0,
                        "odds_used": wide_odds,
                        "p_hit": p_wide,
                        "edge": edge,
                        "kelly_f": 0.0,
                        "score": float(edge * math.sqrt(max(1e-6, p_wide))),
                        "why": "",
                        "notes": "policy_pool=expanded",
                    }
                )
            p_quinella = clamp01(float(left["p_win"]) * float(right["p_win"]) * 1.45)
            quinella_odds = policy_odds_scalar((odds_payload.get("quinella", {}) or {}).get((a, b), 0.0))
            if quinella_odds > 1.0:
                edge = float((p_quinella * quinella_odds) - 1.0)
                item_pool.append(
                    {
                        "bet_type": "quinella",
                        "horses": (a, b),
                        "stake_yen": 0,
                        "odds_used": quinella_odds,
                        "p_hit": p_quinella,
                        "edge": edge,
                        "kelly_f": 0.0,
                        "score": float(edge * math.sqrt(max(1e-6, p_quinella))),
                        "why": "",
                        "notes": "policy_pool=expanded",
                    }
                )

    return build_policy_candidates_from_items(item_pool, per_type_caps=per_type_caps)


def build_marks_top5_payload(merged):
    if merged is None or merged.empty:
        return []
    d = merged.copy()
    if "Top3Prob_model" not in d.columns:
        d["Top3Prob_model"] = pd.to_numeric(d.get("Top3Prob_used"), errors="coerce").fillna(0.0)
    d["Top3Prob_model"] = pd.to_numeric(d["Top3Prob_model"], errors="coerce").fillna(0.0)
    if "rank_score" not in d.columns:
        d["rank_score"] = d["Top3Prob_model"]
    d["rank_score"] = pd.to_numeric(d["rank_score"], errors="coerce").fillna(0.0)
    rank_min = float(d["rank_score"].min()) if len(d) > 0 else 0.0
    rank_max = float(d["rank_score"].max()) if len(d) > 0 else 0.0
    denom = rank_max - rank_min
    if denom <= 1e-12:
        d["rank_score_norm"] = 0.5
    else:
        d["rank_score_norm"] = (d["rank_score"] - rank_min) / denom
    d = d.sort_values("Top3Prob_model", ascending=False).reset_index(drop=True)
    rows = []
    for i, row in d.head(5).iterrows():
        horse_no = parse_horse_no(row.get("horse_no", ""))
        rows.append(
            {
                "horse_no": str(horse_no) if horse_no is not None else str(row.get("horse_no", "")).strip(),
                "horse_name": str(row.get("HorseName", "")).strip(),
                "pred_rank": int(i + 1),
                "top3_prob_model": float(row.get("Top3Prob_model", 0.0) or 0.0),
                "rank_score_norm": float(row.get("rank_score_norm", 0.0) or 0.0),
            }
        )
    return rows


def build_predictions_payload(merged, odds_payload, limit=8):
    if merged is None or merged.empty:
        return []
    d = merged.copy()
    d["horse_key"] = d.get("horse_no").apply(canonical_horse_key)
    if "Top3Prob_model" not in d.columns:
        d["Top3Prob_model"] = pd.to_numeric(d.get("Top3Prob_used"), errors="coerce").fillna(0.0)
    d["Top3Prob_model"] = pd.to_numeric(d["Top3Prob_model"], errors="coerce").fillna(0.0)
    if "rank_score" not in d.columns:
        d["rank_score"] = d["Top3Prob_model"]
    d["rank_score"] = pd.to_numeric(d["rank_score"], errors="coerce").fillna(d["Top3Prob_model"])
    rank_min = float(d["rank_score"].min()) if len(d) > 0 else 0.0
    rank_max = float(d["rank_score"].max()) if len(d) > 0 else 0.0
    denom = rank_max - rank_min
    if denom <= 1e-12:
        d["rank_score_norm"] = 0.5
    else:
        d["rank_score_norm"] = (d["rank_score"] - rank_min) / denom
    d = d.sort_values(["Top3Prob_model", "rank_score"], ascending=False).reset_index(drop=True)
    rows = []
    for i, row in d.head(int(max(1, limit))).iterrows():
        horse_key = canonical_horse_key(row.get("horse_no", ""))
        horse_no = parse_horse_no(row.get("horse_no", ""))
        rows.append(
            {
                "horse_no": str(horse_no) if horse_no is not None else str(row.get("horse_no", "")).strip(),
                "horse_name": str(row.get("HorseName", "")).strip(),
                "pred_rank": int(i + 1),
                "top3_prob_model": float(row.get("Top3Prob_model", 0.0) or 0.0),
                "rank_score_norm": float(row.get("rank_score_norm", 0.0) or 0.0),
                "win_odds": policy_odds_scalar((odds_payload.get("win", {}) or {}).get(horse_key, 0.0)),
                "place_odds": policy_odds_scalar((odds_payload.get("place", {}) or {}).get(horse_key, 0.0)),
            }
        )
    return rows


def _sanitize_policy_json_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, str):
        text = str(value).strip()
        return text
    if isinstance(value, (list, tuple)):
        return [_sanitize_policy_json_value(x) for x in list(value)]
    if isinstance(value, dict):
        out = {}
        for key, item in dict(value).items():
            out[str(key)] = _sanitize_policy_json_value(item)
        return out
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _prediction_col_desc(col):
    desc_map = {
        "race_id": "レースID",
        "horse_no": "馬番",
        "HorseName": "馬名",
        "odds_num": "単勝オッズ",
        "Age": "年齢",
        "SexMale": "性別フラグ（牡）",
        "SexFemale": "性別フラグ（牝）",
        "SexGelding": "性別フラグ（騸）",
        "TargetDistance": "対象距離",
        "fieldsize_med": "想定頭数中央値",
        "best_TimeIndexEff": "タイム指数効率（最大）",
        "avg_TimeIndexEff": "タイム指数効率（平均）",
        "dist_close": "距離適性",
        "Top3Prob_lr": "ロジスティック回帰の3着内率",
        "Top3Prob_lgbm": "LightGBMの3着内率",
        "Top3Prob_model": "統合モデルの3着内率",
        "Top3Prob_est": "推定3着内率",
        "Top3Prob": "3着内率",
        "Top3Prob_used": "採用3着内率",
        "jscore_current": "騎手評価スコア",
        "agg_score": "総合スコア",
        "score": "評価スコア",
        "rank_score": "順位付けスコア",
        "rank_score_norm": "順位付けスコア正規化",
        "confidence_score": "予測信頼度",
        "stability_score": "予測安定性",
        "validity_score": "予測妥当性",
        "consistency_score": "予測整合性",
        "rank_ema": "順位EMA",
        "ev_ema": "EV EMA",
        "risk_score": "リスクスコア",
        "horse_key": "馬キー",
        "pred_rank": "予測順位",
        "win_odds": "単勝オッズ代表値",
        "place_odds": "複勝オッズ代表値",
    }
    if col in desc_map:
        return desc_map[col]
    if str(col).startswith("ti_"):
        return "タイム指数系特徴量"
    if str(col).startswith("jscore_"):
        return "騎手・補正スコア系特徴量"
    if str(col).startswith("ps_"):
        return "走法・位置取り特徴量"
    if str(col).startswith("run_"):
        return "直近走パフォーマンス特徴量"
    if str(col).startswith("cup_"):
        return "同条件傾向特徴量"
    if str(col).startswith("top3_ti_"):
        return "上位タイム指数分位特徴量"
    return "モデル内部特徴量"

def build_prediction_field_guide(merged):
    if merged is None or merged.empty:
        cols = []
    else:
        cols = [str(col).strip() for col in list(merged.columns) if str(col).strip()]
    extras = ["pred_rank", "win_odds", "place_odds"]
    ordered = []
    seen = set()
    for col in cols + extras:
        name = str(col).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return {name: _prediction_col_desc(name) for name in ordered}


def build_predictions_full_payload(merged, odds_payload):
    if merged is None or merged.empty:
        return []
    d = merged.copy()
    d["horse_key"] = d.get("horse_no").apply(canonical_horse_key)
    if "Top3Prob_model" not in d.columns:
        d["Top3Prob_model"] = pd.to_numeric(d.get("Top3Prob_used"), errors="coerce").fillna(0.0)
    d["Top3Prob_model"] = pd.to_numeric(d["Top3Prob_model"], errors="coerce").fillna(0.0)
    if "rank_score" not in d.columns:
        d["rank_score"] = d["Top3Prob_model"]
    d["rank_score"] = pd.to_numeric(d["rank_score"], errors="coerce").fillna(d["Top3Prob_model"])
    d = d.sort_values(["Top3Prob_model", "rank_score"], ascending=False).reset_index(drop=True)
    rows = []
    base_records = d.to_dict("records")
    for idx, row in enumerate(base_records):
        horse_key = canonical_horse_key(row.get("horse_no", ""))
        enriched = {}
        for key, value in row.items():
            enriched[str(key)] = _sanitize_policy_json_value(value)
        horse_no = parse_horse_no(row.get("horse_no", ""))
        enriched["horse_no"] = str(horse_no) if horse_no is not None else str(row.get("horse_no", "")).strip()
        enriched["horse_key"] = str(horse_key)
        enriched["pred_rank"] = int(idx + 1)
        enriched["win_odds"] = policy_odds_scalar((odds_payload.get("win", {}) or {}).get(horse_key, 0.0))
        enriched["place_odds"] = policy_odds_scalar((odds_payload.get("place", {}) or {}).get(horse_key, 0.0))
        rows.append(enriched)
    return rows


def _build_single_odds_entry(horse_no, value):
    odds_scalar = policy_odds_scalar(value)
    entry = {
        "horse_no": str(horse_no),
        "odds": float(odds_scalar),
    }
    if isinstance(value, (list, tuple)):
        nums = [safe_float(x) for x in list(value) if safe_float(x) > 0]
        if nums:
            entry["odds_low"] = float(min(nums))
            entry["odds_high"] = float(max(nums))
    return entry


def _build_pair_odds_entry(pair_key, value):
    left, right = pair_key
    odds_scalar = policy_odds_scalar(value)
    entry = {
        "pair": f"{left}-{right}",
        "horse_no_a": str(left),
        "horse_no_b": str(right),
        "odds": float(odds_scalar),
    }
    if isinstance(value, (list, tuple)):
        nums = [safe_float(x) for x in list(value) if safe_float(x) > 0]
        if nums:
            entry["odds_low"] = float(min(nums))
            entry["odds_high"] = float(max(nums))
    return entry


def build_full_odds_payload(odds_payload):
    odds_payload = dict(odds_payload or {})
    out = {"win": [], "place": [], "wide": [], "quinella": []}
    for horse_no, value in sorted((odds_payload.get("win", {}) or {}).items(), key=lambda x: str(x[0])):
        out["win"].append(_build_single_odds_entry(horse_no, value))
    for horse_no, value in sorted((odds_payload.get("place", {}) or {}).items(), key=lambda x: str(x[0])):
        out["place"].append(_build_single_odds_entry(horse_no, value))
    for pair_key, value in sorted((odds_payload.get("wide", {}) or {}).items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        out["wide"].append(_build_pair_odds_entry(pair_key, value))
    for pair_key, value in sorted((odds_payload.get("quinella", {}) or {}).items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        out["quinella"].append(_build_pair_odds_entry(pair_key, value))
    return out


def resolve_policy_predictor_paths(primary_pred_path, race_id="", run_paths=None):
    run_paths = dict(run_paths or {})
    resolved = {}
    for spec in list_predictors():
        env_name = PREDICTOR_PATH_ENV_MAP.get(spec["id"], "")
        raw = str(os.environ.get(env_name, "") or "").strip()
        if (not raw) and spec.get("run_field"):
            raw = str(run_paths.get(spec["run_field"], "") or "").strip()
        if (not raw) and spec["id"] == "main":
            raw = str(primary_pred_path or "")
        if not raw:
            raw = str(ROOT_DIR / spec["latest_filename"])
        path = Path(raw)
        if path.exists():
            resolved[spec["id"]] = path
    return resolved


def _prepare_policy_predictor_frame(merged, odds_payload):
    if merged is None or merged.empty:
        return pd.DataFrame()
    d = merged.copy()
    if "HorseName" not in d.columns:
        if "name" in d.columns:
            d["HorseName"] = d["name"]
        else:
            d["HorseName"] = ""
    if "horse_no" not in d.columns:
        d["horse_no"] = ""
    d["horse_no"] = d["horse_no"].apply(
        lambda value: str(parse_horse_no(value)) if parse_horse_no(value) is not None else str(value or "").strip()
    )
    d["horse_key"] = d["horse_no"].apply(canonical_horse_key)
    prob_col = pick_prob_column(d.columns)
    d["Top3Prob_model"] = get_prob_series(d, prob_col)
    if "rank_score" not in d.columns:
        d["rank_score"] = d["Top3Prob_model"]
    d["rank_score"] = pd.to_numeric(d["rank_score"], errors="coerce").fillna(d["Top3Prob_model"])
    rank_min = float(d["rank_score"].min()) if len(d) > 0 else 0.0
    rank_max = float(d["rank_score"].max()) if len(d) > 0 else 0.0
    denom = rank_max - rank_min
    if denom <= 1e-12:
        d["rank_score_norm"] = 0.5
    else:
        d["rank_score_norm"] = (d["rank_score"] - rank_min) / denom
    d = d.sort_values(["Top3Prob_model", "rank_score"], ascending=False).reset_index(drop=True)
    d["pred_rank"] = [int(i + 1) for i in range(len(d))]
    d["horse_name"] = d.get("HorseName", "").apply(lambda x: str(x or "").strip())
    d["win_odds"] = d["horse_key"].apply(lambda key: policy_odds_scalar((odds_payload.get("win", {}) or {}).get(key, 0.0)))
    d["place_odds"] = d["horse_key"].apply(lambda key: policy_odds_scalar((odds_payload.get("place", {}) or {}).get(key, 0.0)))
    return d


def load_policy_predictor_frames(odds, predictor_paths, odds_payload):
    frames = {}
    odds = odds.copy()
    for predictor_id, path in dict(predictor_paths or {}).items():
        try:
            preds = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[WARN] failed to read {predictor_id} predictions: {path} ({exc})")
            continue
        try:
            _, _, merged = merge_predictions_with_odds(odds, preds)
        except Exception as exc:
            print(f"[WARN] failed to merge {predictor_id} predictions with odds: {exc}")
            continue
        frames[predictor_id] = _prepare_policy_predictor_frame(merged, odds_payload)
    return frames


def build_multi_predictor_policy_payload(predictor_frames):
    predictor_frames = dict(predictor_frames or {})
    profiles = []
    summaries = []
    consensus_bucket = {}
    top1_horses = []
    available_ids = []

    for spec in list_predictors():
        predictor_id = spec["id"]
        frame = predictor_frames.get(predictor_id)
        available = frame is not None and not frame.empty
        profile_hint = PREDICTOR_PROFILE_HINTS.get(predictor_id, {})
        profiles.append(
            {
                "predictor_id": predictor_id,
                "predictor_label": spec["label"],
                "available": bool(available),
                "style_ja": str(profile_hint.get("style_ja", "") or ""),
                "strengths_ja": list(profile_hint.get("strengths_ja", []) or []),
            }
        )
        if not available:
            continue
        available_ids.append(predictor_id)
        top_rows = []
        for _, row in frame.head(8).iterrows():
            horse_no = str(row.get("horse_no", "") or "").strip()
            horse_name = str(row.get("horse_name", "") or "").strip()
            top_rows.append(
                {
                    "horse_no": horse_no,
                    "horse_name": horse_name,
                    "pred_rank": int(row.get("pred_rank", 0) or 0),
                    "top3_prob_model": float(row.get("Top3Prob_model", 0.0) or 0.0),
                    "rank_score_norm": float(row.get("rank_score_norm", 0.0) or 0.0),
                    "win_odds": float(row.get("win_odds", 0.0) or 0.0),
                    "place_odds": float(row.get("place_odds", 0.0) or 0.0),
                }
            )
        top_choice = top_rows[0] if top_rows else {}
        if top_choice.get("horse_no"):
            top1_horses.append(str(top_choice.get("horse_no")))
        summaries.append(
            {
                "predictor_id": predictor_id,
                "predictor_label": spec["label"],
                "style_ja": str(profile_hint.get("style_ja", "") or ""),
                "field_size": int(len(frame)),
                "top_choice_horse_no": str(top_choice.get("horse_no", "") or ""),
                "top_choice_horse_name": str(top_choice.get("horse_name", "") or ""),
                "top_choice_top3_prob_model": float(top_choice.get("top3_prob_model", 0.0) or 0.0),
                "top_horses": top_rows,
            }
        )
        for _, row in frame.iterrows():
            horse_no = str(row.get("horse_no", "") or "").strip()
            if not horse_no:
                continue
            bucket = consensus_bucket.setdefault(
                horse_no,
                {
                    "horse_no": horse_no,
                    "horse_name": str(row.get("horse_name", "") or "").strip(),
                    "top1_votes": 0,
                    "top3_votes": 0,
                    "predictor_count": 0,
                    "pred_rank_sum": 0.0,
                    "top3_prob_sum": 0.0,
                    "rank_score_norm_sum": 0.0,
                    "win_odds": float(row.get("win_odds", 0.0) or 0.0),
                    "place_odds": float(row.get("place_odds", 0.0) or 0.0),
                    "predictors_support": [],
                },
            )
            pred_rank = int(row.get("pred_rank", 0) or 0)
            bucket["predictor_count"] += 1
            bucket["pred_rank_sum"] += float(pred_rank)
            bucket["top3_prob_sum"] += float(row.get("Top3Prob_model", 0.0) or 0.0)
            bucket["rank_score_norm_sum"] += float(row.get("rank_score_norm", 0.0) or 0.0)
            bucket["predictors_support"].append(spec["label"])
            if pred_rank == 1:
                bucket["top1_votes"] += 1
            if 1 <= pred_rank <= 3:
                bucket["top3_votes"] += 1

    consensus = []
    for horse_no, item in consensus_bucket.items():
        count = int(item.get("predictor_count", 0) or 0)
        if count <= 0:
            continue
        consensus.append(
            {
                "horse_no": horse_no,
                "horse_name": str(item.get("horse_name", "") or ""),
                "top1_votes": int(item.get("top1_votes", 0) or 0),
                "top3_votes": int(item.get("top3_votes", 0) or 0),
                "predictor_count": count,
                "avg_pred_rank": round(float(item.get("pred_rank_sum", 0.0) or 0.0) / count, 4),
                "avg_top3_prob_model": round(float(item.get("top3_prob_sum", 0.0) or 0.0) / count, 6),
                "avg_rank_score_norm": round(float(item.get("rank_score_norm_sum", 0.0) or 0.0) / count, 6),
                "win_odds": float(item.get("win_odds", 0.0) or 0.0),
                "place_odds": float(item.get("place_odds", 0.0) or 0.0),
                "predictors_support": list(item.get("predictors_support", []) or []),
            }
        )
    consensus = sorted(
        consensus,
        key=lambda x: (
            -int(x.get("top1_votes", 0) or 0),
            -int(x.get("top3_votes", 0) or 0),
            float(x.get("avg_pred_rank", 999.0) or 999.0),
            -float(x.get("avg_top3_prob_model", 0.0) or 0.0),
            str(x.get("horse_no", "")),
        ),
    )
    unique_top1 = sorted({horse_no for horse_no in top1_horses if str(horse_no).strip()}, key=lambda x: int(x) if str(x).isdigit() else str(x))
    return {
        "profiles": profiles,
        "summaries": summaries,
        "consensus": consensus[:12],
        "meta": {
            "available_predictor_ids": available_ids,
            "available_predictor_count": int(len(available_ids)),
            "unique_top1_horses": unique_top1,
            "unique_top1_count": int(len(unique_top1)),
            "consensus_top_horse_no": str(consensus[0].get("horse_no", "") or "") if consensus else "",
        },
    }


def build_pair_odds_top_payload(candidates, limit_per_type=5):
    bucketed = {"wide": [], "quinella": []}
    for candidate in list(candidates or []):
        bet_type = str(candidate.get("bet_type", "") or "").strip().lower()
        if bet_type not in bucketed:
            continue
        legs = [str(x) for x in list(candidate.get("legs", []) or []) if str(x).strip()]
        if len(legs) < 2:
            continue
        bucketed[bet_type].append(
            {
                "bet_type": bet_type,
                "pair": f"{legs[0]}-{legs[1]}",
                "odds": float(candidate.get("odds_used", 0.0) or 0.0),
                "score": float(candidate.get("score", 0.0) or 0.0),
            }
        )
    rows = []
    for bet_type in ("wide", "quinella"):
        ordered = sorted(
            bucketed[bet_type],
            key=lambda x: (-float(x.get("score", 0.0) or 0.0), float(x.get("odds", 0.0) or 0.0), str(x.get("pair", ""))),
        )
        for item in ordered[: int(max(1, limit_per_type))]:
            rows.append(
                {
                    "bet_type": bet_type,
                    "pair": str(item.get("pair", "")),
                    "odds": float(item.get("odds", 0.0) or 0.0),
                }
            )
    return rows


def extract_ai_payload(merged, summary_info):
    top_vals = []
    if merged is not None and not merged.empty:
        vals = pd.to_numeric(merged.get("Top3Prob_model"), errors="coerce").fillna(0.0).astype(float).tolist()
        top_vals = sorted(vals, reverse=True)
    gap = (top_vals[0] - top_vals[1]) if len(top_vals) >= 2 else 0.0
    confidence_score = 0.5 + max(0.0, min(0.4, gap * 3.0))
    stability_score = confidence_score
    risk_score = 0.5
    if merged is not None and not merged.empty:
        if "confidence_score" in merged.columns:
            confidence_score = clamp01(float(pd.to_numeric(merged["confidence_score"], errors="coerce").fillna(confidence_score).iloc[0]))
        if "risk_score" in merged.columns:
            risk_score = clamp01(float(pd.to_numeric(merged["risk_score"], errors="coerce").fillna(risk_score).iloc[0]))
        if "rank_ema" in merged.columns:
            stability_score = clamp01(float(pd.to_numeric(merged["rank_ema"], errors="coerce").fillna(stability_score).iloc[0]))
    if isinstance(summary_info, dict):
        race_diags = summary_info.get("diagnostics", []) or summary_info.get("race_diagnostics", [])
        if isinstance(race_diags, list) and race_diags:
            first = race_diags[0]
            try:
                gap = float(first.get("gap"))
            except (TypeError, ValueError):
                gap = float(gap)
            confidence_score = clamp01(first.get("confidence_score", confidence_score))
            stability_score = clamp01(first.get("stability_score", stability_score))
    return {
        "gap": float(gap),
        "confidence_score": float(confidence_score),
        "stability_score": float(stability_score),
        "risk_score": float(risk_score),
    }


def resolve_policy_constraints(
    engine_version,
    budget_yen,
    summary_info,
    candidates_count,
    v2_cfg,
    v3_cfg,
    v4_cfg,
    v5_cfg,
    bankroll_yen=0,
    race_budget_yen=0,
):
    high_odds_threshold = 10.0
    max_tickets_per_race = 6
    if engine_version == "v2":
        max_tickets_per_race = int(v2_cfg.get("max_tickets_per_race", 0) or 0)
        high_odds_threshold = safe_float(v2_cfg.get("high_odds_threshold", 10.0))
    elif engine_version == "v3":
        max_tickets_per_race = int(v3_cfg.get("max_tickets_per_race", 6) or 6)
        high_odds_threshold = safe_float(v3_cfg.get("high_bucket_odds_threshold", 10.0))
    elif engine_version == "v4":
        max_tickets_per_race = int(v4_cfg.get("max_tickets_per_race", 3) or 3)
        high_odds_threshold = safe_float(v4_cfg.get("high_odds_threshold", 10.0))
    elif engine_version == "v5":
        max_tickets_per_race = int(v5_cfg.get("max_tickets_per_race", 6) or 6)
        high_odds_threshold = safe_float(v5_cfg.get("high_odds_threshold", 10.0))
    if max_tickets_per_race <= 0:
        max_tickets_per_race = int(max(1, candidates_count))
    bankroll_value = max(0, int(bankroll_yen or 0))
    race_budget_value = max(0, int(race_budget_yen or bankroll_value or 0))
    return {
        "bankroll_yen": bankroll_value,
        "race_budget_yen": race_budget_value,
        "max_tickets_per_race": int(max_tickets_per_race),
        "high_odds_threshold": float(high_odds_threshold if high_odds_threshold > 0 else 10.0),
        "allowed_types": ["win", "place", "wide", "quinella"],
    }


def build_policy_input_payload(
    race_id,
    scope_key,
    merged,
    summary_info,
    candidates,
    constraints,
    odds_payload,
    multi_predictor=None,
    portfolio_history=None,
):
    ai_payload = extract_ai_payload(merged, summary_info)
    marks_top5 = build_marks_top5_payload(merged)
    predictions = build_predictions_payload(merged, odds_payload)
    predictions_full = build_predictions_full_payload(merged, odds_payload)
    pair_odds_top = build_pair_odds_top_payload(candidates)
    odds_full = build_full_odds_payload(odds_payload)
    prediction_field_guide = build_prediction_field_guide(merged)
    payload = {
        "race_id": str(race_id or "__single_race__"),
        "scope_key": str(scope_key or ""),
        "field_size": int(len(merged) if merged is not None else 0),
        "ai": ai_payload,
        "marks_top5": marks_top5,
        "predictions": predictions,
        "predictions_full": predictions_full,
        "pair_odds_top": pair_odds_top,
        "odds_full": odds_full,
        "prediction_field_guide": prediction_field_guide,
        "multi_predictor": dict(multi_predictor or {}),
        "portfolio_history": dict(portfolio_history or {}),
        "candidates": list(candidates or []),
        "constraints": dict(constraints or {}),
    }
    return payload


def blueprint_group_weights(policy_output, selected_items):
    buy_style = str(policy_output.get("buy_style", "") or "").strip().lower()
    strategy_mode = str(policy_output.get("strategy_mode", "") or "").strip().lower()
    participation_level = str(policy_output.get("participation_level", "") or "").strip().lower()
    construction_style = str(policy_output.get("construction_style", "") or "").strip().lower()
    if not construction_style:
        if strategy_mode in ("pair_focus", "spread"):
            construction_style = "pair_spread"
        elif strategy_mode in ("place_only", "conservative_single", "small_probe"):
            construction_style = "conservative_single"
        else:
            construction_style = "single_axis"
    risk_tilt = str(policy_output.get("risk_tilt", "") or "").strip().lower()
    enabled_types = {str(x).strip().lower() for x in list(policy_output.get("enabled_bet_types", []) or []) if str(x).strip()}
    present_groups = {policy_group_for_type(item.get("bet_type", "")) for item in list(selected_items or [])}
    if buy_style == "win_focus":
        weights = {"win": 0.42, "place": 0.28, "pair": 0.30}
    elif buy_style == "pair_focus":
        weights = {"win": 0.06, "place": 0.30, "pair": 0.64}
    elif buy_style == "place_only":
        weights = {"win": 0.0, "place": 1.0, "pair": 0.0}
    elif buy_style == "conservative":
        weights = {"win": 0.03, "place": 0.82, "pair": 0.15}
    elif buy_style == "place_focus":
        weights = {"win": 0.06, "place": 0.72, "pair": 0.22}
    else:
        weights = {"win": 0.20, "place": 0.46, "pair": 0.34}
    if construction_style == "pair_spread":
        weights["pair"] += 0.18
        weights["place"] -= 0.08
        weights["win"] -= 0.10
    elif construction_style == "conservative_single":
        weights["place"] += 0.18
        weights["win"] -= 0.08
        weights["pair"] -= 0.10
    elif construction_style == "value_hunt":
        weights["pair"] += 0.06
        weights["place"] += 0.04
        weights["win"] -= 0.10
    if risk_tilt == "low":
        weights["place"] += 0.08
        weights["win"] -= 0.04
        weights["pair"] -= 0.04
    elif risk_tilt == "high":
        weights["pair"] += 0.08
        weights["win"] += 0.04
        weights["place"] -= 0.12
    if participation_level == "small_bet":
        weights["place"] += 0.10
        weights["win"] -= 0.04
        weights["pair"] -= 0.06
    for group_name in ("win", "place", "pair"):
        if group_name not in present_groups:
            weights[group_name] = 0.0
    if enabled_types:
        if "win" not in enabled_types:
            weights["win"] = 0.0
        if "place" not in enabled_types:
            weights["place"] = 0.0
        if ("wide" not in enabled_types) and ("quinella" not in enabled_types):
            weights["pair"] = 0.0
    return normalize_policy_weights(weights)


def build_tickets_from_policy_blueprint(policy, candidates, bankroll, cfg):
    if not isinstance(policy, dict):
        return []
    if str(policy.get("bet_decision", "") or "").strip().lower() == "no_bet":
        return []
    if str(policy.get("buy_style", "") or "").strip().lower() == "no_bet":
        return []
    constraints = dict((cfg or {}).get("constraints", {}) or {})
    ticket_item_by_id = dict((cfg or {}).get("ticket_item_by_id", {}) or {})
    selected_candidates = list(candidates or [])
    if not selected_candidates or not ticket_item_by_id:
        return []

    explicit_ticket_plan = list(policy.get("ticket_plan", []) or [])
    if explicit_ticket_plan:
        out = []
        for ticket in explicit_ticket_plan:
            ticket_id = str(ticket.get("id", "") or "").strip()
            stake_yen = int(ticket.get("stake_yen", 0) or 0)
            if (not ticket_id) or stake_yen <= 0:
                continue
            item = dict(ticket_item_by_id.get(ticket_id, {}) or {})
            if not item:
                continue
            rec = dict(item)
            rec["stake_yen"] = stake_yen
            out.append(rec)
        if out:
            return out

    enabled_types = {
        str(x).strip().lower() for x in list(policy.get("enabled_bet_types", []) or []) if str(x).strip()
    }
    focus_points = list(policy.get("focus_points", []) or [])
    key_horses = {str(x).strip() for x in list(policy.get("key_horses", []) or []) if str(x).strip()}
    secondary_horses = {str(x).strip() for x in list(policy.get("secondary_horses", []) or []) if str(x).strip()}
    longshot_horses = {str(x).strip() for x in list(policy.get("longshot_horses", []) or []) if str(x).strip()}
    if not key_horses:
        for point in focus_points:
            if str(point.get("type", "")).strip().lower() == "horse":
                value = str(point.get("value", "")).strip()
                if value:
                    key_horses.add(value)
                    break
    if not enabled_types:
        for point in focus_points:
            if str(point.get("type", "")).strip().lower() == "bet_type":
                value = str(point.get("value", "")).strip().lower()
                if value:
                    enabled_types.add(value)
    preferred_ids = {str(x).strip() for x in list(policy.get("pick_ids", []) or []) if str(x).strip()}
    max_tickets = int(policy.get("max_ticket_count", 0) or 0)
    hard_cap = int(constraints.get("max_tickets_per_race", 0) or 0)
    if hard_cap > 0:
        max_tickets = min(max_tickets if max_tickets > 0 else hard_cap, hard_cap)
    if max_tickets <= 0:
        max_tickets = max(1, min(len(selected_candidates), 2))
    strategy_mode = str(policy.get("strategy_mode", "") or "").strip().lower()
    participation_level = str(policy.get("participation_level", "") or "").strip().lower()
    construction_style = str(policy.get("construction_style", "") or "").strip().lower()
    if not construction_style:
        if strategy_mode in ("pair_focus", "spread"):
            construction_style = "pair_spread"
        elif strategy_mode in ("place_only", "conservative_single", "small_probe"):
            construction_style = "conservative_single"
        else:
            construction_style = "single_axis"
    risk_tilt = str(policy.get("risk_tilt", "") or "").strip().lower()
    high_odds_threshold = float(constraints.get("high_odds_threshold", 10.0) or 10.0)

    def style_score(candidate):
        bet_type = str(candidate.get("bet_type", "") or "").strip().lower()
        legs = [str(x) for x in list(candidate.get("legs", []) or []) if str(x).strip()]
        base = float(candidate.get("score", candidate.get("ev", 0.0)) or 0.0)
        score = base
        if preferred_ids and str(candidate.get("id", "")) in preferred_ids:
            score += 5.0
        if enabled_types and bet_type not in enabled_types:
            score -= 1000.0
        key_hits = len(key_horses.intersection(legs))
        secondary_hits = len(secondary_horses.intersection(legs))
        longshot_hits = len(longshot_horses.intersection(legs))
        score += 1.8 * float(key_hits)
        score += 0.8 * float(secondary_hits)
        if construction_style == "single_axis":
            if bet_type == "place" and key_hits:
                score += 2.4
            elif bet_type == "win" and key_hits:
                score += 2.1
            elif bet_type in ("wide", "quinella") and key_hits:
                score += 1.8 + (0.6 * float(secondary_hits))
            elif bet_type in ("wide", "quinella"):
                score -= 1.2
        elif construction_style == "pair_spread":
            if bet_type in ("wide", "quinella"):
                score += 2.3 + (0.5 * float(key_hits + secondary_hits))
            else:
                score -= 0.8
        elif construction_style == "value_hunt":
            if longshot_hits:
                score += 2.0
            if float(candidate.get("ev", 0.0) or 0.0) > 0.0:
                score += 0.9
            if bet_type == "place" and key_hits:
                score += 1.3
        elif construction_style == "conservative_single":
            if bet_type == "place":
                score += 2.4 + (0.4 * float(key_hits))
            elif bet_type == "win" and key_hits:
                score += 0.7
            elif bet_type in ("wide", "quinella") and key_hits and secondary_hits:
                score += 0.6
            else:
                score -= 1.4
        if risk_tilt == "low":
            if bet_type == "place":
                score += 0.5
            if float(candidate.get("odds_used", 0.0) or 0.0) >= high_odds_threshold:
                score -= 0.7
        elif risk_tilt == "high":
            if bet_type in ("wide", "quinella"):
                score += 0.4
        if participation_level == "small_bet" and bet_type in ("wide", "quinella"):
            score -= 0.4
        return score

    ordered = sorted(
        selected_candidates,
        key=lambda x: (
            -style_score(x),
            -float(x.get("p_hit", 0.0) or 0.0),
            -float(x.get("ev", 0.0) or 0.0),
            str(x.get("id", "")),
        ),
    )
    picked = []
    high_count = 0
    longshot_count = 0
    seen_ids = set()
    for candidate in ordered:
        cid = str(candidate.get("id", "") or "").strip()
        if (not cid) or (cid in seen_ids):
            continue
        item = dict(ticket_item_by_id.get(cid, {}) or {})
        if not item:
            continue
        odds_used = float(item.get("odds_used", 0.0) or 0.0)
        item_horses = {str(x) for x in list(item.get("horses", ()) or ()) if str(x).strip()}
        is_high = odds_used >= high_odds_threshold
        is_longshot = bool(item_horses.intersection(longshot_horses)) or is_high
        if is_high and high_count >= 1:
            continue
        if is_longshot and longshot_horses and longshot_count >= 1:
            continue
        seen_ids.add(cid)
        picked.append(item)
        if is_high:
            high_count += 1
        if is_longshot and longshot_horses:
            longshot_count += 1
        if len(picked) >= max_tickets:
            break
    if not picked:
        return []

    units_total = int(max(0, int(bankroll) // int(UNIT_YEN)))
    if units_total <= 0:
        return []
    weights = blueprint_group_weights(policy, picked)
    group_items = {"win": [], "place": [], "pair": []}
    for item in picked:
        group_items[policy_group_for_type(item.get("bet_type", ""))].append(item)
    units_by_group = split_units(units_total, [weights["win"], weights["place"], weights["pair"]])
    alloc_map = {"win": int(units_by_group[0]), "place": int(units_by_group[1]), "pair": int(units_by_group[2])}
    out = []
    for group_name in ("win", "place", "pair"):
        rows = list(group_items[group_name] or [])
        if not rows:
            continue
        units_g = int(alloc_map.get(group_name, 0) or 0)
        if units_g <= 0:
            continue
        row_weights = []
        for row in rows:
            score = float(row.get("score", row.get("edge", 0.0)) or 0.0)
            if score <= 0:
                score = max(1e-6, float(row.get("p_hit", 0.0) or 0.0))
            row_weights.append(max(1e-6, score))
        alloc = split_units(units_g, row_weights)
        for row, units in zip(rows, alloc):
            if int(units) <= 0:
                continue
            rec = dict(row)
            rec["stake_yen"] = int(units) * int(UNIT_YEN)
            out.append(rec)
    return out


def attach_policy_text(items, policy_output):
    if not items:
        return []
    strategy_text_ja = str(policy_output.get("strategy_text_ja", "") or "")
    bet_tendency_ja = str(policy_output.get("bet_tendency_ja", "") or "")
    buy_style = str(policy_output.get("buy_style", "") or "")
    bet_decision = str(policy_output.get("bet_decision", "") or "")
    construction_style = str(policy_output.get("strategy_mode", "") or policy_output.get("construction_style", "") or "")
    reason_codes = ",".join([str(x) for x in list(policy_output.get("reason_codes", []) or []) if str(x).strip()])
    out = []
    for item in items:
        rec = dict(item)
        notes = str(rec.get("notes", "") or "")
        policy_note = f"policy={buy_style}"
        if bet_decision:
            policy_note = f"{policy_note};decision={bet_decision}"
        if construction_style:
            policy_note = f"{policy_note};construction={construction_style}"
        if reason_codes:
            policy_note = f"{policy_note};reasons={reason_codes}"
        rec["notes"] = f"{notes};{policy_note}" if notes else policy_note
        rec["strategy_text_ja"] = strategy_text_ja
        rec["bet_tendency_ja"] = bet_tendency_ja
        rec["policy_engine"] = "gemini"
        rec["policy_buy_style"] = buy_style
        rec["policy_bet_decision"] = bet_decision
        rec["policy_construction_style"] = construction_style
        out.append(rec)
    return out


def build_gemini_policy_artifact_path(scope_data_dir, race_id, run_id):
    race_text = str(race_id or "").strip()
    run_text = str(run_id or "").strip()
    if race_text:
        race_dir = scope_data_dir / race_text
        race_dir.mkdir(parents=True, exist_ok=True)
        return race_dir / f"gemini_policy_{run_text}_{race_text}.json"
    return scope_data_dir / f"gemini_policy_{run_text}.json"


def save_gemini_policy_artifact(path, payload):
    if not path or not payload:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        print(f"[WARN] failed to save gemini_policy artifact: {exc}")
        return False


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
    if not sys.stdin or (hasattr(sys.stdin, "isatty") and not sys.stdin.isatty()):
        return
    try:
        input("Press Enter to exit...")
    except EOFError:
        return


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


def merge_predictions_with_odds(odds, preds):
    odds = odds.copy()
    preds = preds.copy()
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


def load_inputs(odds_path, pred_path):
    odds = pd.read_csv(odds_path, encoding="utf-8-sig")
    preds = pd.read_csv(pred_path, encoding="utf-8-sig")
    return merge_predictions_with_odds(odds, preds)


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
    args = parse_args()
    engine_version = resolve_engine_version(args.engine_version)
    bet_profile = resolve_bet_profile(args.bet_profile)
    v5_profile = resolve_v5_profile(getattr(args, "v5_profile", "default"))
    policy_engine = str(getattr(args, "policy_engine", "none") or "none").strip().lower()
    if policy_engine not in ("none", "gemini"):
        policy_engine = "none"
    gemini_model = str(
        getattr(args, "gemini_model", "gemini-3.1-flash-lite-preview") or "gemini-3.1-flash-lite-preview"
    ).strip()
    policy_cache_enable = parse_bool_text(getattr(args, "policy_cache_enable", "true"), default=True)
    policy_budget_reuse = parse_bool_text(getattr(args, "policy_budget_reuse", "false"), default=False)
    config = load_config()
    config, strategy_used = apply_strategy_from_env(config)
    predictor_config = load_predictor_config()
    bet_engine_v2_cfg = get_bet_engine_v2_config(predictor_config)
    bet_engine_v3_cfg = apply_bet_profile_to_v3(get_bet_engine_v3_config(predictor_config), bet_profile)
    bet_engine_v4_cfg = get_bet_engine_v4_config(predictor_config)
    bet_engine_v5_cfg = apply_bet_profile_to_v5(get_bet_engine_v5_config(predictor_config), v5_profile)
    race_id = resolve_race_id()
    budget_list = resolve_budget_list()

    odds_path = Path(os.environ.get("ODDS_PATH") or ODDS_PATH)
    pred_path = Path(os.environ.get("PRED_PATH") or PRED_PATH)
    wide_odds_path = Path(os.environ.get("WIDE_ODDS_PATH") or WIDE_ODDS_PATH)
    fuku_odds_path = Path(os.environ.get("FUKU_ODDS_PATH") or FUKU_ODDS_PATH)
    quinella_odds_path = Path(os.environ.get("QUINELLA_ODDS_PATH") or QUINELLA_ODDS_PATH)
    run_paths = {}
    predictor_paths = resolve_policy_predictor_paths(pred_path, race_id=race_id, run_paths=run_paths)

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
    predictor_paths = resolve_policy_predictor_paths(pred_path, race_id=race_id, run_paths=run_paths)
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
    if "Top3Prob_model" not in merged.columns:
        merged["Top3Prob_model"] = merged["Top3Prob_used"]
    merged["Top3Prob_model"] = pd.to_numeric(merged["Top3Prob_model"], errors="coerce").fillna(0.0)
    if "rank_score" not in merged.columns:
        merged["rank_score"] = merged["Top3Prob_model"]
    merged["rank_score"] = pd.to_numeric(merged["rank_score"], errors="coerce").fillna(merged["Top3Prob_model"])

    missing = merged[merged["horse_no"].isna()]
    if not missing.empty:
        print("[WARN] Missing odds for:")
        for name in missing["HorseName"].tolist():
            print(" -", name)

    pred_df, odds_payload = build_bet_engine_v2_inputs(
        merged,
        fuku_odds_path=fuku_odds_path,
        wide_odds_path=wide_odds_path,
        quinella_odds_path=quinella_odds_path,
    )
    predictor_frames = load_policy_predictor_frames(odds, predictor_paths, odds_payload)
    multi_predictor_payload = build_multi_predictor_policy_payload(predictor_frames)
    horse_meta = build_horse_meta_map(merged)

    run_id = str(os.environ.get("RUN_ID", "")).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    scope_data_dir = get_data_dir(BASE_DIR, SCOPE_KEY)
    scope_data_dir.mkdir(parents=True, exist_ok=True)
    items_path = scope_data_dir / f"bet_plan_items_{run_id}.csv"
    cfg_dump_path = scope_data_dir / f"bet_engine_{engine_version}_cfg_{run_id}.json"
    if engine_version == "v3":
        v3_summary = build_bet_engine_v3_audit_summary(bet_engine_v3_cfg)
        try:
            payload = {
                "scope": SCOPE_KEY,
                "run_id": run_id,
                "engine_version": engine_version,
                "bet_profile": bet_profile,
                "params": v3_summary,
            }
            cfg_dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] failed to save bet_engine_v3 cfg summary: {exc}")
        print(
            "[bet_engine_v3] profile={profile} ".format(profile=bet_profile)
            + " ".join([f"{k}={v3_summary.get(k)}" for k in BET_ENGINE_V3_AUDIT_KEYS])
        )
    elif engine_version in ("v4", "v5", "v6"):
        try:
            if engine_version == "v6":
                _v6_cfg = data.get("bet_engine_v6", {}) if isinstance(data, dict) else {}
                payload = {
                    "scope": SCOPE_KEY,
                    "run_id": run_id,
                    "engine_version": "v6",
                    "params": _v6_cfg,
                }
            else:
                payload = {
                    "scope": SCOPE_KEY,
                    "run_id": run_id,
                    "engine_version": engine_version,
                    "v5_profile": (v5_profile if engine_version == "v5" else ""),
                    "params": (bet_engine_v4_cfg if engine_version == "v4" else bet_engine_v5_cfg),
                }
            cfg_dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] failed to save bet_engine_{engine_version} cfg summary: {exc}")

    out_rows = []
    item_rows_all = []
    all_no_bet_rows = []
    gemini_policy_records = []
    gemini_policy_path = None
    shared_policy_output_payload = None
    shared_policy_meta = {}
    shared_policy_source_budget = None
    gemini_ledger_date = extract_ledger_date(run_id)
    policy_is_budget_agnostic = bool(policy_engine == "gemini")
    plan_columns = [
        "budget_yen",
        "bet_type",
        "horse_no",
        "horse_name",
        "units",
        "amount_yen",
        "hit_prob_est",
        "hit_prob_se",
        "hit_prob_ci95_low",
        "hit_prob_ci95_high",
        "payout_mult",
        "ev_ratio_est",
        "expected_return_yen",
        "odds_used",
        "p_hit",
        "edge",
        "kelly_f",
        "score",
        "stake_yen",
        "notes",
        "strategy_text_ja",
        "bet_tendency_ja",
        "policy_engine",
        "policy_buy_style",
        "policy_bet_decision",
        "policy_construction_style",
    ]
    item_columns = [
        "run_id",
        "scope",
        "race_id",
        "budget_yen",
        "bet_type",
        "horses",
        "odds_used",
        "p_hit",
        "edge",
        "score",
        "stake_yen",
        "notes",
        "strategy_text_ja",
        "bet_tendency_ja",
        "policy_engine",
        "policy_buy_style",
        "policy_bet_decision",
        "policy_construction_style",
    ]

    v5_diag_printed = False
    for budget_yen in budget_list:
        engine_used = engine_version
        summary_info = {}
        budget_rows, budget_item_rows = [], []
        normalized_items = []
        policy_applied = False
        policy_output_payload = None

        try:
            if engine_version == "v3":
                items_raw, _, summary_info = generate_bet_plan_v3(
                    pred_df=pred_df,
                    odds=odds_payload,
                    bankroll_yen=budget_yen,
                    scope_key=SCOPE_KEY,
                    config=bet_engine_v3_cfg,
                )
                normalized_items = normalize_portfolio_items(items_raw)
            elif engine_version == "v4":
                items_raw, _, summary_info = generate_bet_plan_v4(
                    pred_df=pred_df,
                    odds=odds_payload,
                    bankroll_yen=budget_yen,
                    scope_key=SCOPE_KEY,
                    config=bet_engine_v4_cfg,
                )
                normalized_items = normalize_portfolio_items(items_raw)
            elif engine_version == "v5":
                items_raw, _, summary_info = generate_bet_plan_v5(
                    pred_df=pred_df,
                    odds=odds_payload,
                    bankroll_yen=budget_yen,
                    scope_key=SCOPE_KEY,
                    config=bet_engine_v5_cfg,
                )
                normalized_items = normalize_portfolio_items(items_raw)
                if (not v5_diag_printed) and isinstance(summary_info, dict):
                    race_diags = summary_info.get("diagnostics", [])
                    if race_diags:
                        print(f"[v5][diag][first_race] {race_diags[0]}")
                        v5_diag_printed = True
                    strategy_text = str(summary_info.get("strategy_text", "") or "").strip()
                    if strategy_text:
                        print(f"[v5][strategy]\n{strategy_text}")
            elif engine_version == "v6":
                v6_cfg = data.get("bet_engine_v6", {}) if isinstance(data, dict) else {}
                items_raw, _, summary_info = generate_bet_plan_v6(
                    pred_df=pred_df,
                    odds=odds_payload,
                    bankroll_yen=budget_yen,
                    scope_key=SCOPE_KEY,
                    config=v6_cfg,
                )
                normalized_items = normalize_portfolio_items(items_raw)
                if isinstance(summary_info, dict):
                    race_diags = summary_info.get("diagnostics", [])
                    for rd in race_diags:
                        scenario = rd.get("scenario", "")
                        print(f"[v6][scenario] {scenario}: {rd.get('ticket_count', 0)} tickets, budget={rd.get('race_budget', 0)}")
                    strategy_text = str(summary_info.get("strategy_text", "") or "").strip()
                    if strategy_text:
                        print(f"[v6][strategy] {strategy_text}")
            else:
                result = generate_bet_plan_v2(
                    pred_df=pred_df,
                    odds=odds_payload,
                    bankroll_yen=budget_yen,
                    scope_key=SCOPE_KEY,
                    config=bet_engine_v2_cfg,
                )
                normalized_items = build_items_from_v2_result(result)
        except Exception as exc:
            print(f"[WARN] bet_engine_{engine_version} failed for budget={budget_yen}, fallback to v2: {exc}")
            engine_used = "v2_fallback"
            result = generate_bet_plan_v2(
                pred_df=pred_df,
                odds=odds_payload,
                bankroll_yen=budget_yen,
                scope_key=SCOPE_KEY,
                config=bet_engine_v2_cfg,
            )
            normalized_items = build_items_from_v2_result(result)

        if policy_engine == "gemini":
            if (RacePolicyInput is None) or (call_gemini_policy is None):
                print("[WARN] policy_engine=gemini but module unavailable; keep original plan.")
            else:
                policy_candidates, ticket_item_by_id = build_policy_candidate_pool(
                    merged=merged,
                    odds_payload=odds_payload,
                    per_type_caps={"win": 5, "place": 5, "wide": 10, "quinella": 10},
                )
                if not policy_candidates:
                    policy_candidates, ticket_item_by_id = build_policy_candidates_from_items(normalized_items)
                bankroll_summary_before = summarize_bankroll(BASE_DIR, gemini_ledger_date)
                portfolio_history = build_history_context(
                    BASE_DIR,
                    gemini_ledger_date,
                    lookback_days=14,
                    recent_ticket_limit=10,
                )
                constraints = resolve_policy_constraints(
                    engine_version=engine_version,
                    budget_yen=budget_yen,
                    summary_info=summary_info,
                    candidates_count=len(policy_candidates),
                    v2_cfg=bet_engine_v2_cfg,
                    v3_cfg=bet_engine_v3_cfg,
                    v4_cfg=bet_engine_v4_cfg,
                    v5_cfg=bet_engine_v5_cfg,
                    bankroll_yen=int(bankroll_summary_before.get("available_bankroll_yen", 0) or 0),
                    race_budget_yen=int(bankroll_summary_before.get("available_bankroll_yen", 0) or 0),
                )
                policy_payload = build_policy_input_payload(
                    race_id=race_id,
                    scope_key=SCOPE_KEY,
                    merged=merged,
                    summary_info=summary_info,
                    candidates=policy_candidates,
                    constraints=constraints,
                    odds_payload=odds_payload,
                    multi_predictor=multi_predictor_payload,
                    portfolio_history=portfolio_history,
                )
                try:
                    reused_shared_policy = False
                    use_shared_policy = bool(policy_is_budget_agnostic or policy_budget_reuse)
                    if (not use_shared_policy) or (shared_policy_output_payload is None):
                        policy_input_obj = RacePolicyInput(**policy_payload)
                        policy_output_obj = call_gemini_policy(
                            input=policy_input_obj,
                            model=gemini_model,
                            timeout_s=20,
                            cache_enable=policy_cache_enable,
                        )
                        if hasattr(policy_output_obj, "model_dump"):
                            policy_output_payload = policy_output_obj.model_dump()
                        else:
                            policy_output_payload = policy_output_obj.dict()
                        policy_meta = get_last_call_meta() if callable(get_last_call_meta) else {}
                        if use_shared_policy:
                            shared_policy_output_payload = dict(policy_output_payload)
                            shared_policy_meta = dict(policy_meta)
                            shared_policy_source_budget = 0
                    else:
                        policy_output_payload = dict(shared_policy_output_payload)
                        policy_meta = dict(shared_policy_meta)
                        reused_shared_policy = True
                    rebalanced = build_tickets_from_policy_blueprint(
                        policy=policy_output_payload,
                        candidates=policy_candidates,
                        bankroll=budget_yen,
                        cfg={
                            "constraints": constraints,
                            "ticket_item_by_id": ticket_item_by_id,
                        },
                    )
                    rebalanced = attach_policy_text(rebalanced, policy_output_payload)
                    normalized_items = rebalanced
                    policy_applied = True
                    engine_used = f"{engine_used}+gemini"
                    if reused_shared_policy:
                        policy_meta = {
                            **dict(shared_policy_meta),
                            "requested_budget_yen": int(constraints.get("bankroll_yen", 0) or 0),
                            "requested_race_budget_yen": int(constraints.get("race_budget_yen", 0) or 0),
                            "reused": True,
                            "source_budget_yen": int(constraints.get("bankroll_yen", 0) or 0),
                        }
                    else:
                        policy_meta = {
                            **dict(policy_meta),
                            "requested_budget_yen": int(constraints.get("bankroll_yen", 0) or 0),
                            "requested_race_budget_yen": int(constraints.get("race_budget_yen", 0) or 0),
                            "reused": False,
                            "source_budget_yen": int(constraints.get("bankroll_yen", 0) or 0),
                        }
                    strategy_text_ja = str(policy_output_payload.get("strategy_text_ja", "") or "").strip()
                    bet_tendency_ja = str(policy_output_payload.get("bet_tendency_ja", "") or "").strip()
                    reason_codes = [str(x) for x in list(policy_output_payload.get("reason_codes", []) or []) if str(x).strip()]
                    warnings_list = [str(x) for x in list(policy_output_payload.get("warnings", []) or []) if str(x).strip()]
                    if reused_shared_policy:
                        pass
                    else:
                        print(
                            "[policy][gemini] scope=race decision={decision} buy_style={buy_style} max_ticket_count={picked} "
                            "enabled={enabled} construction={construction} cache_hit={cache_hit} fallback_reason={fallback_reason}".format(
                                decision=str(policy_output_payload.get("bet_decision", "")),
                                buy_style=str(policy_output_payload.get("buy_style", "")),
                                picked=int(policy_output_payload.get("max_ticket_count", 0) or 0),
                                enabled=str(policy_output_payload.get("enabled_bet_types", [])),
                                construction=str(policy_output_payload.get("construction_style", "")),
                                cache_hit=int(bool(policy_meta.get("cache_hit", False))),
                                fallback_reason=str(policy_meta.get("fallback_reason", "") or ""),
                            )
                        )
                        if strategy_text_ja:
                            print(f"[gemini][strategy]\n{strategy_text_ja}")
                        if bet_tendency_ja:
                            print(f"[gemini][tendency] {bet_tendency_ja}")
                        if reason_codes:
                            print(f"[gemini][reasons] {', '.join(reason_codes)}")
                        if warnings_list:
                            print(f"[gemini][warnings] {', '.join(warnings_list)}")
                    if not reused_shared_policy:
                        rendered_policy_rows, _ = build_rows_from_bet_items(
                            items=normalized_items,
                            budget_yen=0,
                            horse_meta=horse_meta,
                        )
                        reserve_run_tickets(
                            BASE_DIR,
                            run_id=run_id,
                            scope_key=SCOPE_KEY,
                            race_id=race_id,
                            ledger_date=gemini_ledger_date,
                            tickets=rendered_policy_rows,
                        )
                        bankroll_summary_after = summarize_bankroll(BASE_DIR, gemini_ledger_date)
                        gemini_policy_records.append(
                            {
                                "budget_yen": 0,
                                "shared_policy": True,
                                "output": dict(policy_output_payload),
                                "meta": dict(policy_meta),
                                "portfolio": {
                                    "ledger_date": gemini_ledger_date,
                                    "before": dict(bankroll_summary_before),
                                    "after": dict(bankroll_summary_after),
                                },
                                "tickets": list(rendered_policy_rows),
                            }
                        )
                except Exception as exc:
                    print(f"[WARN] gemini policy failed for budget={budget_yen}, keep original plan: {exc}")

        budget_rows, budget_item_rows = build_rows_from_bet_items(
            items=normalized_items,
            budget_yen=budget_yen,
            horse_meta=horse_meta,
        )

        out_rows.extend(budget_rows)
        for row in budget_item_rows:
            row["run_id"] = run_id
            row["scope"] = SCOPE_KEY
            row["race_id"] = race_id
            item_rows_all.append(row)
        total_stake, ticket_count, edge_sum, weighted_edge = calc_plan_summary(budget_item_rows)
        expected_return = int(round(sum(float(r.get("p_hit", 0.0) or 0.0) * float(r.get("odds_used", 0.0) or 0.0) * int(r.get("stake_yen", 0) or 0) for r in budget_item_rows)))
        expected_profit = int(round(sum(float(r.get("edge", 0.0) or 0.0) * int(r.get("stake_yen", 0) or 0) for r in budget_item_rows)))
        no_bet = bool(ticket_count == 0)
        if isinstance(summary_info, dict) and (not policy_applied):
            if "expected_return_yen" in summary_info:
                expected_return = int(summary_info.get("expected_return_yen", expected_return) or expected_return)
            if "expected_profit_yen" in summary_info:
                expected_profit = int(summary_info.get("expected_profit_yen", expected_profit) or expected_profit)
            no_bet = bool(summary_info.get("no_bet", no_bet))
        print(
            f"[summary][{budget_yen}][{engine_used}] stake={total_stake} expected_return={expected_return} "
            f"expected_profit={expected_profit} ticket_count={ticket_count} no_bet={int(no_bet)} "
            f"estimated_total_edge_sum={weighted_edge:.2f} edge_sum={edge_sum:.4f}"
        )

    out_df = pd.DataFrame(out_rows, columns=plan_columns)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    items_df = pd.DataFrame(item_rows_all, columns=item_columns)
    items_df.to_csv(items_path, index=False, encoding="utf-8-sig")
    if policy_engine == "gemini" and gemini_policy_records:
        gemini_policy_path = build_gemini_policy_artifact_path(scope_data_dir, race_id, run_id)
        artifact_payload = {
            "scope": SCOPE_KEY,
            "race_id": str(race_id or ""),
            "run_id": str(run_id or ""),
            "policy_engine": "gemini",
            "gemini_model": gemini_model,
            "policy_budget_reuse": bool(policy_budget_reuse),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "budgets": gemini_policy_records,
        }
        if save_gemini_policy_artifact(gemini_policy_path, artifact_payload):
            print(f"Saved Gemini policy: {gemini_policy_path}")
    total_stake_all, ticket_count_all, edge_sum_all, weighted_edge_all = calc_plan_summary(item_rows_all)

    if strategy_used:
        print(f"Strategy: {strategy_used}")
    print(f"Engine version: {engine_version}")
    print(f"Policy engine: {policy_engine}")
    if policy_engine == "gemini":
        print(f"Gemini model: {gemini_model}")
        print(f"Policy cache: {int(bool(policy_cache_enable))}")
        print(f"Policy budget reuse: {int(bool(policy_budget_reuse))}")
    if engine_version == "v3":
        print(f"Bet profile: {bet_profile}")
    print(f"Budgets: {', '.join(str(v) for v in budget_list)}")
    if out_df.empty:
        print("No tickets generated.")
    else:
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
    print(
        f"[summary][all] total_stake={total_stake_all} ticket_count={ticket_count_all} "
        f"estimated_total_edge_sum={weighted_edge_all:.2f} edge_sum={edge_sum_all:.4f}"
    )
    expected_return_all = int(round(sum(float(r.get("p_hit", 0.0) or 0.0) * float(r.get("odds_used", 0.0) or 0.0) * int(r.get("stake_yen", 0) or 0) for r in item_rows_all)))
    expected_profit_all = int(round(sum(float(r.get("edge", 0.0) or 0.0) * int(r.get("stake_yen", 0) or 0) for r in item_rows_all)))
    print(
        f"[summary][all_ext] stake={total_stake_all} expected_return={expected_return_all} "
        f"expected_profit={expected_profit_all} ticket_count={ticket_count_all} no_bet={int(ticket_count_all==0)}"
    )
    print(f"Saved: {OUT_PATH}")
    print(f"Saved items: {items_path}")
    if engine_version in ("v3", "v4", "v5", "v6"):
        print(f"Saved bet_engine_{engine_version} cfg: {cfg_dump_path}")
    append_no_bet_logs(NO_BET_LOG_PATH, all_no_bet_rows)
    pause_exit()


if __name__ == "__main__":
    main()

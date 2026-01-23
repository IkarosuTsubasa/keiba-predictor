import csv
import json
import sys
from pathlib import Path

import os

from surface_scope import (
    get_config_path,
    get_data_dir,
    get_predictor_config_path,
    get_scope_key,
    migrate_legacy_data,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = None
CONFIG_PATH = None
PRED_CONFIG_PATH = None


def init_scope():
    scope_key = get_scope_key()
    migrate_legacy_data(BASE_DIR, scope_key)
    os.environ["SCOPE_KEY"] = scope_key
    global DATA_DIR, CONFIG_PATH, PRED_CONFIG_PATH
    DATA_DIR = get_data_dir(BASE_DIR, scope_key)
    CONFIG_PATH = get_config_path(BASE_DIR, scope_key)
    PRED_CONFIG_PATH = get_predictor_config_path(BASE_DIR, scope_key)
    return scope_key

DEFAULT_BET_CONFIG = {
    "version": 2,
    "style_default": "balanced",
    "active_strategy": "balanced",
    "hit_weight": 1.3,
    "payout_weight": 0.25,
    "type_quality_power": 1.4,
    "type_quality_topk": 3,
    "strength_blend": 0.7,
    "coverage_target": 0.65,
    "min_eligible": 5,
    "trifecta_rec_min_hit_prob": 0.06,
    "hit_floors": {
        "win": 0.55,
        "place": 0.6,
        "wide": 0.45,
        "quinella": 0.35
    },
    "style_weights": {
        "steady": {
            "win": 0.2,
            "place": 0.5,
            "wide": 0.2,
            "quinella": 0.1
        },
        "balanced": {
            "win": 0.2,
            "place": 0.45,
            "wide": 0.2,
            "quinella": 0.15
        },
        "aggressive": {
            "win": 0.25,
            "place": 0.35,
            "wide": 0.2,
            "quinella": 0.2
        }
    },
    "type_weight_adjust": {
        "win": 1.0,
        "place": 1.0,
        "wide": 1.0,
        "quinella": 1.0
    },
    "selector": {
        "mode": "epsilon_greedy",
        "epsilon": 0.2,
        "min_samples": 3
    },
    "strategies": {
        "balanced": {
            "label": "balanced",
            "overrides": {}
        },
        "steady": {
            "label": "steady",
            "overrides": {
                "style_default": "steady",
                "hit_weight": 1.45,
                "payout_weight": 0.18,
                "coverage_target": 0.7,
                "min_eligible": 6
            }
        },
        "value": {
            "label": "value",
            "overrides": {
                "style_default": "balanced",
                "hit_weight": 1.1,
                "payout_weight": 0.35,
                "coverage_target": 0.6
            }
        },
        "aggressive": {
            "label": "aggressive",
            "overrides": {
                "style_default": "aggressive",
                "hit_weight": 1.0,
                "payout_weight": 0.45,
                "coverage_target": 0.55,
                "min_eligible": 4
            }
        }
    }
}

DEFAULT_PRED_CONFIG = {
    "version": 1,
    "active_strategy": "balanced",
    "selector": {
        "mode": "epsilon_greedy",
        "epsilon": 0.2,
        "min_samples": 3
    },
    "optimizer": {
        "target_score": 2.0
    },
    "params": {
        "place_score_base": 0.2,
        "place_score_weight": 0.8,
        "place_score_fill": 0.5,
        "time_trend_window": 3,
        "tau_window": 300,
        "tau_min": 300.0,
        "tau_max": 1200.0,
        "smooth_p": 1.2,
        "surf_floor": 0.2,
        "dist_floor": 0.3,
        "record_weight_base": 0.7,
        "record_weight_match": 0.3,
        "recent_race_count": 5,
        "top_score_count": 3,
        "top3_scale": 3.0
    },
    "strategies": {
        "balanced": {
            "label": "balanced",
            "overrides": {}
        },
        "steady": {
            "label": "steady",
            "overrides": {
                "record_weight_base": 0.75,
                "record_weight_match": 0.25,
                "smooth_p": 1.1,
                "top_score_count": 2,
                "top3_scale": 2.8
            }
        },
        "trend": {
            "label": "trend",
            "overrides": {
                "time_trend_window": 5,
                "record_weight_match": 0.35,
                "recent_race_count": 6,
                "top_score_count": 3
            }
        },
        "distance": {
            "label": "distance",
            "overrides": {
                "tau_window": 400,
                "dist_floor": 0.2,
                "smooth_p": 1.4,
                "top3_scale": 3.2
            }
        }
    }
}

CSV_SPECS = {
    "runs.csv": [
        "run_id",
        "timestamp",
        "race_url",
        "race_id",
        "history_url",
        "trigger_race",
        "scope",
        "surface",
        "distance",
        "budget_yen",
        "style",
        "strategy",
        "strategy_reason",
        "predictor_strategy",
        "predictor_reason",
        "config_version",
        "predictions_path",
        "odds_path",
        "wide_odds_path",
        "fuku_odds_path",
        "quinella_odds_path",
        "trifecta_odds_path",
        "plan_path",
        "tickets",
        "amount_yen",
    ],
    "results.csv": [
        "run_id",
        "strategy",
        "profit_yen",
        "base_amount",
        "roi",
        "note",
    ],
    "predictor_results.csv": [
        "run_id",
        "strategy",
        "predictions_path",
        "pred_top1",
        "pred_top2",
        "pred_top3",
        "actual_top1",
        "actual_top2",
        "actual_top3",
        "top3_hit_count",
        "top5_hit_count",
        "top1_hit",
        "top1_in_top3",
        "top3_exact",
        "score",
        "rank_score",
        "ev_score",
        "hit_rate",
        "score_total",
        "confidence_score",
        "stability_score",
        "validity_score",
        "consistency_score",
        "rank_ema",
        "ev_ema",
        "risk_score",
    ],
    "config_history.csv": [
        "timestamp",
        "action",
        "reason",
        "avg_roi",
        "roi_std",
        "step_mult",
        "window",
        "roi_samples",
        "tune_group",
        "adjusted_types",
        "hit_weight",
        "payout_weight",
        "coverage_target",
        "hit_floors_json",
        "type_adjust_json",
        "version",
    ],
    "predictor_config_history.csv": [
        "timestamp",
        "avg_score",
        "avg_quality",
        "action",
        "reason",
        "ema_score",
        "ema_quality",
        "rank_ema",
        "ev_ema",
        "risk_score",
        "score_std",
        "max_drawdown",
        "record_weight_base",
        "record_weight_match",
        "recent_race_count",
        "top_score_count",
        "smooth_p",
        "top3_scale",
    ],
    "offline_eval.csv": [
        "timestamp",
        "window",
        "runs",
        "roi_samples",
        "roi_avg",
        "roi_std",
        "roi_min",
        "roi_max",
        "profit_avg",
        "profit_total",
        "win_count",
        "loss_count",
        "flat_count",
        "pred_samples",
        "pred_hit_avg",
        "pred_top1_rate",
        "pred_top1_in_top3_rate",
        "pred_exact_rate",
    ],
    "race_results.csv": [
        "run_id",
        "timestamp",
        "actual_top1",
        "actual_top2",
        "actual_top3",
    ],
    "bet_ticket_results.csv": [
        "run_id",
        "bet_type",
        "horse_no",
        "horse_name",
        "amount_yen",
        "hit",
        "est_payout_yen",
    ],
    "bet_type_stats.csv": [
        "run_id",
        "timestamp",
        "bet_type",
        "bets",
        "hits",
        "hit_rate",
        "amount_yen",
        "est_payout_yen",
        "est_profit_yen",
    ],
}


def ensure_csv(path, fieldnames, reset=False):
    created = False
    if reset or not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        created = True
    return created


def write_json_if_missing(path, data):
    if path.exists():
        return False
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def main():
    reset = "--reset" in sys.argv
    init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    created_files = []
    if write_json_if_missing(CONFIG_PATH, DEFAULT_BET_CONFIG):
        created_files.append(str(CONFIG_PATH))
    if write_json_if_missing(PRED_CONFIG_PATH, DEFAULT_PRED_CONFIG):
        created_files.append(str(PRED_CONFIG_PATH))

    for name, fields in CSV_SPECS.items():
        path = DATA_DIR / name
        if ensure_csv(path, fields, reset=reset):
            created_files.append(str(path))

    if created_files:
        print("Initialized files:")
        for item in created_files:
            print(f"- {item}")
    else:
        print("Nothing to initialize.")

    if reset:
        print("Logs were reset.")

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()

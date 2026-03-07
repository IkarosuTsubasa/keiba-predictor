import pandas as pd

from bet_engine_v3 import generate_bet_plan_v3
from bet_engine_v4 import generate_bet_plan_v4
from bet_engine_v5 import generate_bet_plan_v5
from bet_engine_v6 import generate_bet_plan_v6


def _case_pred_df(race_id: str, rows):
    return pd.DataFrame(
        [
            {
                "race_id": race_id,
                "horse_key": str(horse_key),
                "rank_score": float(rank_score),
                "Top3Prob_model": float(top3_prob),
                "confidence_score": 0.55,
            }
            for horse_key, rank_score, top3_prob in rows
        ]
    )


def _cases():
    return [
        {
            "name": "balanced",
            "pred_df": _case_pred_df(
                "CMP_BALANCED",
                [
                    ("1", 2.2, 0.34),
                    ("2", 1.9, 0.29),
                    ("3", 1.6, 0.24),
                    ("4", 1.2, 0.18),
                    ("5", 0.9, 0.15),
                    ("6", 0.6, 0.12),
                    ("7", 0.3, 0.10),
                    ("8", 0.1, 0.08),
                ],
            ),
            "odds": {
                "win": {"1": 4.8, "2": 6.2, "3": 7.0, "4": 9.5, "5": 12.0, "6": 15.0, "7": 18.0, "8": 22.0},
                "place": {
                    "1": (3.2, 3.8),
                    "2": (3.0, 3.6),
                    "3": (2.8, 3.4),
                    "4": (2.5, 3.1),
                    "5": (2.3, 2.9),
                    "6": (2.1, 2.8),
                    "7": (2.0, 2.6),
                    "8": (1.9, 2.5),
                },
                "wide": {
                    ("1", "2"): (4.8, 6.2),
                    ("1", "3"): (5.4, 7.0),
                    ("2", "3"): (5.8, 7.8),
                    ("1", "4"): (6.9, 9.0),
                    ("2", "4"): (7.5, 10.0),
                    ("3", "4"): (8.6, 11.6),
                },
                "quinella": {
                    ("1", "2"): 11.8,
                    ("1", "3"): 14.2,
                    ("2", "3"): 17.0,
                    ("1", "4"): 19.0,
                    ("2", "4"): 22.6,
                    ("3", "4"): 27.0,
                },
            },
        },
        {
            "name": "honmei",
            "pred_df": _case_pred_df(
                "CMP_HONMEI",
                [
                    ("1", 2.6, 0.40),
                    ("2", 1.7, 0.28),
                    ("3", 1.1, 0.17),
                    ("4", 0.4, 0.09),
                    ("5", -0.1, 0.06),
                ],
            ),
            "odds": {
                "win": {"1": 3.1, "2": 5.8, "3": 8.0, "4": 15.0, "5": 25.0},
                "place": {"1": 1.6, "2": 2.2, "3": 3.0, "4": 4.8, "5": 6.8},
                "wide": {
                    ("1", "2"): 3.5,
                    ("1", "3"): 4.9,
                    ("2", "3"): 6.4,
                },
                "quinella": {
                    ("1", "2"): 9.8,
                    ("1", "3"): 13.5,
                    ("2", "3"): 18.0,
                },
            },
        },
        {
            "name": "ana",
            "pred_df": _case_pred_df(
                "CMP_ANA",
                [
                    ("1", 1.8, 0.27),
                    ("2", 1.6, 0.24),
                    ("3", 1.3, 0.20),
                    ("4", 0.6, 0.13),
                    ("5", 0.9, 0.16),
                    ("6", 0.2, 0.10),
                ],
            ),
            "odds": {
                "win": {"1": 3.8, "2": 4.5, "3": 5.2, "4": 8.0, "5": 15.0, "6": 18.0},
                "place": {"1": 1.8, "2": 2.0, "3": 2.2, "4": 2.9, "5": 4.6, "6": 5.4},
                "wide": {
                    ("1", "2"): 3.9,
                    ("1", "5"): 6.8,
                    ("2", "5"): 6.2,
                    ("3", "5"): 7.4,
                },
                "quinella": {
                    ("1", "2"): 10.5,
                    ("1", "5"): 20.0,
                    ("2", "5"): 18.5,
                    ("3", "5"): 22.0,
                },
            },
        },
    ]


def _stake(item):
    return int(item.get("stake_yen", item.get("stake", 0)) or 0)


def _ticket_type(item):
    return str(item.get("bet_type", item.get("ticket_type", "")))


def _horse_text(item):
    horses = item.get("horses", item.get("horse_ids", []))
    return "-".join(str(x) for x in horses)


def _ev(item):
    return float(item.get("edge", item.get("ev", item.get("EV", 0.0))) or 0.0)


def _diag_text(summary):
    diags = summary.get("race_diagnostics") or summary.get("diagnostics") or []
    if not diags:
        return ""
    diag = diags[0]
    if "scenario" in diag:
        return f"scenario={diag.get('scenario')} race_budget={diag.get('race_budget')}"
    if "lambda_market" in diag:
        return (
            f"lambda_market={float(diag.get('lambda_market', 0.0)):.3f} "
            f"confidence={float(diag.get('confidence_score', 0.0)):.3f} "
            f"race_budget={diag.get('race_budget')}"
        )
    if "candidate_count_by_type" in diag:
        return f"race_budget={diag.get('race_budget')} selected_types={diag.get('selected_types')}"
    if "after_budget_alloc" in diag:
        return f"after_budget_alloc={diag.get('after_budget_alloc')} no_bet_reason={diag.get('no_bet_reason')}"
    return str(diag)


def _run_versions(pred_df, odds, bankroll):
    cfg3 = {
        "min_p_hit_per_ticket": 0.01,
        "min_p_win_per_ticket": 0.01,
        "min_edge_per_ticket": -1.0,
        "min_ev": {"win": -1.0, "place": -1.0, "wide": -1.0, "quinella": -1.0},
        "target_risk_share": 0.20,
        "kelly_scale": 0.50,
    }
    cfg4 = {
        "min_ev": -1.0,
        "race_budget_share": 0.02,
        "max_tickets_per_race": 5,
        "ensure_diversity": True,
    }
    cfg5 = {
        "min_ev_per_ticket": -1.0,
        "min_p_by_type": {"win": 0.01, "place": 0.01, "wide": 0.01, "quinella": 0.01},
        "target_risk_share": 0.20,
        "min_race_budget": 400,
        "value_gate_enabled": False,
        "value_entry_gate_enabled": False,
    }
    cfg6 = {
        "min_ev_to_bet": -1.0,
        "target_risk_share": 0.05,
        "min_race_budget": 500,
        "max_tickets": 5,
    }
    return [
        ("v3", generate_bet_plan_v3(pred_df, odds, bankroll, "compare", cfg3)),
        ("v4", generate_bet_plan_v4(pred_df, odds, bankroll, "compare", cfg4)),
        ("v5", generate_bet_plan_v5(pred_df, odds, bankroll, "compare", cfg5)),
        ("v6", generate_bet_plan_v6(pred_df, odds, bankroll, "compare", cfg6)),
    ]


def main():
    bankroll = 50000
    for case in _cases():
        print(f"=== CASE {case['name']} ===")
        for version, result in _run_versions(case["pred_df"], case["odds"], bankroll):
            items, _, summary = result
            total_stake = int(summary.get("total_stake_yen", summary.get("total_stake", 0)) or 0)
            types = sorted({_ticket_type(item) for item in items if _ticket_type(item)})
            print(
                f"{version}: ticket_count={len(items)} total_stake={total_stake} "
                f"types={types} no_bet={summary.get('no_bet', False)}"
            )
            diag_text = _diag_text(summary)
            if diag_text:
                print(f"  diag: {diag_text}")
            for item in items[:3]:
                print(
                    f"  {_ticket_type(item)}:{_horse_text(item)} stake={_stake(item)} "
                    f"odds={float(item.get('odds_used', item.get('odds', 0.0)) or 0.0):.2f} "
                    f"edge={_ev(item):.3f}"
                )
            if not items:
                print("  no tickets")
        print("")


if __name__ == "__main__":
    main()

import pandas as pd

from bet_engine_v5 import generate_bet_plan_v5


def main():
    pred_df = pd.DataFrame(
        [
            {"race_id": "R_V5", "horse_key": "1", "rank_score": 2.2, "Top3Prob_model": 0.34},
            {"race_id": "R_V5", "horse_key": "2", "rank_score": 1.9, "Top3Prob_model": 0.29},
            {"race_id": "R_V5", "horse_key": "3", "rank_score": 1.6, "Top3Prob_model": 0.24},
            {"race_id": "R_V5", "horse_key": "4", "rank_score": 1.2, "Top3Prob_model": 0.18},
            {"race_id": "R_V5", "horse_key": "5", "rank_score": 0.9, "Top3Prob_model": 0.15},
            {"race_id": "R_V5", "horse_key": "6", "rank_score": 0.6, "Top3Prob_model": 0.12},
            {"race_id": "R_V5", "horse_key": "7", "rank_score": 0.3, "Top3Prob_model": 0.10},
            {"race_id": "R_V5", "horse_key": "8", "rank_score": 0.1, "Top3Prob_model": 0.08},
            {"race_id": "R_V5", "horse_key": "9", "rank_score": -0.2, "Top3Prob_model": 0.06},
            {"race_id": "R_V5", "horse_key": "10", "rank_score": -0.5, "Top3Prob_model": 0.05},
        ]
    )

    odds = {
        "win": {
            "1": 4.8,
            "2": 6.2,
            "3": 7.0,
            "4": 9.5,
            "5": 12.0,
            "6": 15.0,
            "7": 18.0,
            "8": 22.0,
            "9": 27.0,
            "10": 34.0,
        },
        "place": {
            "1": (3.2, 3.8),
            "2": (3.0, 3.6),
            "3": (2.8, 3.4),
            "4": (2.5, 3.1),
            "5": (2.3, 2.9),
            "6": (2.1, 2.8),
            "7": (2.0, 2.6),
            "8": (1.9, 2.5),
            "9": (1.8, 2.4),
            "10": (1.7, 2.3),
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
    }

    cfg = {
        "debug_first_bet_race": True,
        "ensure_diversity": True,
        "min_yen_unit": 100,
        "target_risk_share": 0.25,
        "min_race_budget": 400,
        "max_tickets_per_race": 6,
        "ev_margin": 0.0,
        "min_ev_per_ticket": -0.30,
        "min_p_by_type": {"win": 0.01, "place": 0.01, "wide": 0.01, "quinella": 0.01},
    }
    bankroll = 50000

    items, _, summary = generate_bet_plan_v5(
        pred_df=pred_df,
        odds=odds,
        bankroll_yen=bankroll,
        scope_key="smoke_v5",
        config=cfg,
    )

    types = sorted(set(str(x.get("ticket_type", "")).strip().lower() for x in items))
    stakes = [int(x.get("stake", 0) or 0) for x in items]
    total_stake = int(sum(stakes))
    all_100x = all((s % 100) == 0 for s in stakes)
    diag = summary.get("diagnostics", [{}])
    race_budget = int(diag[0].get("race_budget", 0)) if diag else 0

    print(f"ticket_count={len(items)}")
    print(f"types={types}")
    print(f"total_stake={total_stake} race_budget={race_budget}")
    print(f"all_stake_100x={all_100x}")
    print(f"strategy_text=\n{summary.get('strategy_text', '')}")
    if diag:
        print(f"first_diagnostics={diag[0]}")

    assert len(types) >= 2, "expected at least 2 ticket types"
    assert "place" in types, "expected place + another type"
    assert all_100x, "stake must be 100-yen unit"
    assert total_stake <= race_budget, "total stake exceeds race budget"


if __name__ == "__main__":
    main()

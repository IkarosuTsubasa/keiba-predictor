import pandas as pd

from bet_engine_v2 import generate_bet_plan_v2


def main():
    pred_df = pd.DataFrame(
        [
            {"race_id": "R1", "horse_key": "1", "rank_score": 1.25, "Top3Prob_model": 0.24},
            {"race_id": "R1", "horse_key": "2", "rank_score": 0.92, "Top3Prob_model": 0.18},
            {"race_id": "R1", "horse_key": "3", "rank_score": 0.63, "Top3Prob_model": 0.15},
            {"race_id": "R1", "horse_key": "4", "rank_score": 0.12, "Top3Prob_model": 0.10},
            {"race_id": "R1", "horse_key": "5", "rank_score": -0.35, "Top3Prob_model": 0.08},
            {"race_id": "R1", "horse_key": "6", "rank_score": -0.57, "Top3Prob_model": 0.06},
        ]
    )

    odds = {
        "win": {"1": 4.8, "2": 6.2, "3": 8.0, "4": 12.5, "5": 18.0, "6": 22.0},
        "place": {"1": (1.6, 2.3), "2": (2.1, 3.0), "3": (2.8, 3.8), "4": (3.5, 5.0)},
        "wide": {("1", "2"): (2.5, 3.5), ("1", "3"): (3.8, 5.6), ("2", "3"): (5.0, 7.2)},
        "quinella": {("1", "2"): 8.8, ("1", "3"): 12.5, ("2", "3"): 15.0},
    }

    result = generate_bet_plan_v2(
        pred_df=pred_df,
        odds=odds,
        bankroll_yen=50000,
        scope_key="central_dirt",
        config={},
    )

    total_stake = sum(item.stake_yen for item in result.items)
    print(f"tickets={len(result.items)} total_stake={total_stake}")
    for item in result.items[:5]:
        horses = "-".join(item.horses)
        print(
            f"{item.bet_type}:{horses} stake={item.stake_yen} "
            f"p_hit={item.p_hit:.4f} odds={item.odds_used:.2f} edge={item.edge:.4f}"
        )


if __name__ == "__main__":
    main()

import pandas as pd

from bet_engine_v3 import generate_bet_plan_v3


def _bucket(odds_used: float) -> str:
    odd = float(odds_used or 0.0)
    if odd < 3.0:
        return "low"
    if odd < 10.0:
        return "mid"
    return "high"


def main():
    pred_df = pd.DataFrame(
        [
            {"race_id": "R1", "horse_key": "1", "rank_score": 2.0, "Top3Prob_model": 0.20},
            {"race_id": "R1", "horse_key": "2", "rank_score": 1.0, "Top3Prob_model": 0.14},
            {"race_id": "R1", "horse_key": "3", "rank_score": 0.8, "Top3Prob_model": 0.11},
            {"race_id": "R1", "horse_key": "4", "rank_score": 0.5, "Top3Prob_model": 0.08},
            {"race_id": "R1", "horse_key": "5", "rank_score": 1.8, "Top3Prob_model": 0.40},
            {"race_id": "R1", "horse_key": "6", "rank_score": -0.2, "Top3Prob_model": 0.07},
        ]
    )

    odds = {
        "win": {"1": 2.6, "2": 4.8, "3": 7.5, "4": 11.0, "5": 20.0, "6": 30.0},
        "place": {
            "1": (1.4, 1.8),
            "2": (2.0, 2.6),
            "3": (2.8, 3.8),
            "4": (3.8, 5.6),
            "5": (5.0, 7.4),
            "6": (8.0, 13.0),
        },
        "wide": {
            ("1", "5"): (8.0, 12.0),
            ("2", "5"): (10.0, 16.0),
            ("3", "5"): (14.0, 22.0),
            ("5", "6"): (20.0, 35.0),
        },
        "quinella": {
            ("1", "5"): 22.0,
            ("2", "5"): 31.0,
            ("3", "5"): 45.0,
            ("5", "6"): 70.0,
        },
    }

    cfg = {
        "min_p_hit_per_ticket": 0.05,
        "min_p_win_per_ticket": 0.02,
        "min_edge_per_ticket": 0.0,
        "min_ev": {"win": 0.0, "place": 0.0, "wide": 0.0, "quinella": 0.0},
        "kelly_scale": 0.5,
        "target_risk_share": 0.3,
        "max_high_odds_tickets_per_race": 1,
    }
    items, _, summary = generate_bet_plan_v3(
        pred_df=pred_df,
        odds=odds,
        bankroll_yen=50000,
        scope_key="smoke",
        config=cfg,
    )

    bucket_counts = {"low": 0, "mid": 0, "high": 0}
    for item in items:
        if int(item.get("stake_yen", 0)) <= 0:
            continue
        b = _bucket(float(item.get("odds_used", 0.0)))
        bucket_counts[b] += 1

    print(
        f"ticket_count={summary.get('ticket_count', 0)} total_stake={summary.get('total_stake_yen', 0)} "
        f"no_bet={summary.get('no_bet', False)}"
    )
    print(
        f"odds_bucket low={bucket_counts['low']} mid={bucket_counts['mid']} high={bucket_counts['high']}"
    )
    if items:
        assert bucket_counts["high"] <= 1, "high odds ticket quota violated"
        assert (bucket_counts["low"] + bucket_counts["mid"]) >= 1, "low/mid presence violated"
    for item in items[:8]:
        horses = "-".join(item.get("horses", []))
        print(
            f"{item.get('bet_type')}:{horses} stake={item.get('stake_yen')} "
            f"odds={float(item.get('odds_used', 0.0)):.2f} odds_eff={float(item.get('odds_eff', 0.0)):.2f} "
            f"p_hit={float(item.get('p_hit', 0.0)):.4f} edge={float(item.get('edge', 0.0)):.4f}"
        )


if __name__ == "__main__":
    main()

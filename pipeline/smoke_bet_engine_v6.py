import pandas as pd

from bet_engine_v6 import generate_bet_plan_v6


def _rows_from_probs(race_id: str, probs):
    rows = []
    for idx, prob in enumerate(probs, start=1):
        rows.append(
            {
                "race_id": race_id,
                "horse_key": str(idx),
                "rank_score": float(prob),
                "Top3Prob_model": float(prob),
                "confidence_score": 0.60,
            }
        )
    return pd.DataFrame(rows)


def _scenario_cases():
    return [
        {
            "name": "honmei",
            "pred_df": _rows_from_probs("R_HONMEI", [0.40, 0.28, 0.18, 0.09, 0.05]),
            "odds": {
                "win": {"1": 3.2, "2": 5.5, "3": 7.8, "4": 15.0, "5": 28.0},
                "place": {"1": 1.6, "2": 2.2, "3": 3.0, "4": 5.8, "5": 8.5},
                "wide": {
                    ("1", "2"): 3.4,
                    ("1", "3"): 4.8,
                    ("2", "3"): 6.5,
                },
            },
            "expect_scenario": "honmei",
        },
        {
            "name": "nitoujiku",
            "pred_df": _rows_from_probs("R_NITOUJIKU", [0.32, 0.28, 0.18, 0.13, 0.09]),
            "odds": {
                "win": {"1": 4.6, "2": 5.1, "3": 7.2, "4": 10.0, "5": 14.0},
                "place": {"1": 2.0, "2": 2.1, "3": 2.8, "4": 3.5, "5": 4.0},
                "wide": {
                    ("1", "2"): 4.2,
                    ("1", "3"): 5.8,
                    ("2", "3"): 5.4,
                },
            },
            "expect_scenario": "nitoujiku",
        },
        {
            "name": "konsen",
            "pred_df": _rows_from_probs("R_KONSEN", [0.28, 0.25, 0.22, 0.15, 0.10]),
            "odds": {
                "win": {"1": 4.0, "2": 4.4, "3": 4.8, "4": 6.5, "5": 8.0},
                "place": {"1": 4.0, "2": 4.2, "3": 4.4, "4": 3.8, "5": 3.4},
                "wide": {
                    ("1", "2"): 5.2,
                    ("1", "3"): 5.6,
                    ("2", "3"): 5.4,
                },
            },
            "expect_scenario": "konsen",
        },
        {
            "name": "ana",
            "pred_df": _rows_from_probs("R_ANA", [0.27, 0.24, 0.20, 0.13, 0.16]),
            "odds": {
                "win": {"1": 3.8, "2": 4.5, "3": 5.2, "4": 8.0, "5": 15.0},
                "place": {"1": 1.8, "2": 2.0, "3": 2.2, "4": 3.0, "5": 4.6},
                "wide": {
                    ("1", "5"): 6.8,
                    ("2", "5"): 6.2,
                    ("1", "2"): 3.9,
                },
            },
            "expect_scenario": "ana",
        },
        {
            "name": "miokuri",
            "pred_df": _rows_from_probs("R_MIOKURI", [0.53, 0.47]),
            "odds": {
                "win": {"1": 1.8, "2": 2.0},
                "place": {"1": 1.2, "2": 1.3},
                "wide": {("1", "2"): 1.5},
            },
            "expect_scenario": "miokuri",
        },
    ]


def _ticket_label(item):
    ticket_type = str(item.get("ticket_type", item.get("bet_type", "")))
    horses = "-".join(str(x) for x in item.get("horses", item.get("horse_ids", [])))
    stake = int(item.get("stake_yen", item.get("stake", 0)) or 0)
    ev = float(item.get("ev", item.get("EV", 0.0)) or 0.0)
    return f"{ticket_type}:{horses} stake={stake} ev={ev:.3f}"


def main():
    cfg = {
        "min_ev_to_bet": -0.20,
        "target_risk_share": 0.05,
        "min_race_budget": 500,
        "max_race_budget": 2500,
        "max_tickets": 5,
        "confidence_budget_scale": True,
    }
    bankroll = 50000

    for case in _scenario_cases():
        items, _, summary = generate_bet_plan_v6(
            pred_df=case["pred_df"],
            odds=case["odds"],
            bankroll_yen=bankroll,
            scope_key="smoke_v6",
            config=cfg,
        )
        diag = (summary.get("race_diagnostics") or [{}])[0]
        scenario = str(diag.get("scenario", ""))
        total_stake = int(summary.get("total_stake_yen", 0))
        labels = [_ticket_label(item) for item in items[:5]]

        print(f"[{case['name']}] scenario={scenario} ticket_count={len(items)} total_stake={total_stake}")
        for label in labels:
            print(f"  {label}")

        assert scenario == case["expect_scenario"], f"unexpected scenario for {case['name']}: {scenario}"
        if scenario == "miokuri":
            assert len(items) == 0, "miokuri should not create tickets"
        else:
            assert len(items) >= 1, f"{scenario} should create at least one ticket"
            assert total_stake > 0, f"{scenario} should allocate positive stake"


if __name__ == "__main__":
    main()

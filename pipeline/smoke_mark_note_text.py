import web_app


def main():
    ability_rows = [
        {
            "mark": "◎",
            "horse_no": "1",
            "horse_name": "Alpha",
            "pred_rank": "1",
            "bet_types": "place,wide",
            "recommended_bet_types": "place,wide",
            "race_type": "混戦",
            "confidence": "B",
            "gap_1_2": 0.0812,
            "risk_share": 0.18,
        },
        {
            "mark": "○",
            "horse_no": "2",
            "horse_name": "Beta",
            "pred_rank": "2",
            "bet_types": "win",
            "recommended_bet_types": "win",
            "race_type": "混戦",
            "confidence": "B",
            "gap_1_2": 0.0812,
        },
    ]
    pred_csv = (
        "HorseName,Top3Prob_model,Top3Prob_lgbm,Top3Prob_lr,confidence_score,risk_score\n"
        "Alpha,0.42,0.40,0.38,0.81,0.73\n"
        "Beta,0.36,0.33,0.31,0.81,0.73\n"
        "Delta,0.18,0.16,0.15,0.81,0.73\n"
    )
    gemini_policy_payload = {
        "budgets": [
            {
                "shared_policy": True,
                "output": {
                    "key_horses": ["4"],
                    "secondary_horses": ["9", "5", "10"],
                    "longshot_horses": ["9", "5", "10"],
                    "strategy_text_ja": "AI strategy text",
                    "bet_tendency_ja": "wide focus",
                },
            }
        ]
    }

    text = web_app.build_mark_note_text(
        ability_rows,
        predictions_filename="",
        predictions_csv_text=pred_csv,
        gemini_policy_payload=gemini_policy_payload,
    )

    assert "【AIレース評価】" in text
    assert "AI評価：混戦" in text
    assert "AI信頼度：B" in text
    assert "予測データ総合評価：S" in text
    assert "【能力印（◎○▲△☆）】" in text
    assert "◎ 1番 Alpha" in text
    assert "○ 2番 Beta" in text
    assert "【買い目】" in text
    assert "4-9,5,10" in text
    assert "★ 9,5,10" in text
    assert "【買い目戦略】" in text
    assert "AI strategy text" in text
    assert "【買い目傾向】" in text
    assert "wide focus" in text
    assert "【CSVデータ】" in text
    assert "Top3Prob_model" in text
    assert "Alpha,0.42,0.40,0.38" in text

    print("OK: mark note smoke passed")


if __name__ == "__main__":
    main()

import web_app


def main():
    ability_rows = [
        {
            "mark": "◎",
            "horse_no": "1",
            "horse_name": "テストホースA",
            "pred_rank": "1",
            "bet_types": "place,wide",
            "recommended_bet_types": "place,wide",
            "bet_ev_norm": 0.92,
            "risk_signal": "注意",
            "race_type": "混戦",
            "confidence": "B",
            "gap_1_2": 0.0812,
            "risk_share": 0.18,
        },
        {
            "mark": "○",
            "horse_no": "2",
            "horse_name": "テストホースB",
            "pred_rank": "2",
            "bet_types": "win",
            "recommended_bet_types": "win",
            "bet_ev_norm": 0.88,
            "risk_signal": "見送り級",
            "race_type": "混戦",
            "confidence": "B",
            "gap_1_2": 0.0812,
        },
    ]
    value_rows = [
        {
            "value_mark": "★",
            "horse_no": "5",
            "horse_name": "テストホースD",
            "pred_rank": "9",
            "value_score": 0.38,
            "reason_tags": "期待値上位 / オッズ妙味",
            "recommended_bet_types": "win,wide",
        }
    ]
    pred_csv = (
        "HorseName,Top3Prob_model,Top3Prob_lgbm,Top3Prob_lr,confidence_score,risk_score\n"
        "テストホースA,0.42,0.40,0.38,0.81,0.73\n"
        "テストホースB,0.36,0.33,0.31,0.81,0.73\n"
        "テストホースD,0.18,0.16,0.15,0.81,0.73\n"
    )
    gemini_policy_payload = {
        "budgets": [
            {
                "shared_policy": True,
                "output": {
                    "key_horses": ["4"],
                    "secondary_horses": ["9", "5", "10"],
                    "longshot_horses": ["9", "5", "10"],
                    "strategy_text_ja": "AI評価においてスパークリシャール(4)の複勝圏内確率が一定の評価を得ており、オッズ妙味を考慮して少額で参加します。混戦模様のため、無理な勝負は避け、4を軸にした複勝およびワイドでの保守的な構成とします。信頼度が低いため、あくまでリスクを抑えた参加に留めます。",
                    "bet_tendency_ja": "買い目傾向：4の複勝および4を軸としたワイド少額",
                }
            }
        ]
    }
    text = web_app.build_mark_note_text(
        ability_rows,
        value_rows,
        predictions_filename="",
        predictions_csv_text=pred_csv,
        gemini_policy_payload=gemini_policy_payload,
    )

    assert "【AIレース評価】" in text
    assert "【能力印（◎○▲△☆）】" in text
    assert "【買い目】" in text
    assert "AI評価：混戦" in text
    assert "AI信頼度：B（gap=0.081）" in text
    assert "予測データ総合評価：" in text
    assert "※AI信頼度=印5頭のgap" in text
    assert "◎ 1番 テストホースA" in text
    assert "○ 2番 テストホースB" in text
    assert "4-9,5,10" in text
    assert "★ 9,5,10" in text
    assert "【買い目戦略】" in text
    assert "4を軸にした複勝およびワイドでの保守的な構成" in text
    assert "【買い目傾向】" in text
    assert "買い目傾向：4の複勝および4を軸としたワイド少額" in text
    assert "【予測データ】" in text
    assert "Top3Prob_model：統合モデルの3着内確率" in text
    assert "Top3Prob_lgbm：LightGBMモデル" in text
    assert "Top3Prob_lr：ロジスティック回帰モデル" in text
    assert "【CSVデータ】" in text
    assert "※本記事はAIによる予測データを公開するものであり、" in text

    print("OK: mark note readability smoke passed")


if __name__ == "__main__":
    main()

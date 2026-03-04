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
    text = web_app.build_mark_note_text(
        ability_rows,
        value_rows,
        predictions_filename="",
        predictions_csv_text=pred_csv,
    )

    assert "【AI競馬予想】AI能力印（◎○▲△☆）＋EV狙い馬（★）" in text
    assert "【AIレース評価】" in text
    assert "【AI能力印（◎○▲△☆）】" in text
    assert "【EV狙い馬（★）】" in text
    assert "AI評価：混戦" in text
    assert "AI信頼度：B（gap=0.081）" in text
    assert "予測データ総合評価：" in text
    assert "※AI信頼度=印5頭のgap" in text
    assert "◎ 1番 テストホースA【注意】" in text
    assert "○ 2番 テストホースB【見送り級】" in text
    assert "★ 5番 テストホースD（期待値上位 / オッズ妙味）" in text
    assert "推奨：単勝・ワイド" in text
    assert "【AI戦略】" in text
    assert "【買い目傾向】" in text
    assert "※AIは期待値(EV)ベースで券種を選択しています" in text
    assert "【予測データ説明】" in text
    assert "Top3Prob_model：統合モデルの3着内確率" in text
    assert "Top3Prob_lgbm：LightGBMモデル" in text
    assert "Top3Prob_lr：ロジスティック回帰モデル" in text
    assert "【CSVデータ】" in text
    assert "※本記事はAIによる予測データを公開するものであり、" in text

    idx_eval = text.find("【AIレース評価】")
    idx_ability = text.find("【AI能力印（◎○▲△☆）】")
    idx_value = text.find("【EV狙い馬（★）】")
    idx_strategy = text.find("【AI戦略】")
    idx_tendency = text.find("【買い目傾向】")
    idx_desc = text.find("【予測データ説明】")
    idx_csv = text.find("【CSVデータ】")
    idx_disclaimer = text.find("※本記事はAIによる予測データを公開するものであり、")
    assert -1 not in (idx_eval, idx_ability, idx_value, idx_strategy, idx_tendency, idx_desc, idx_csv, idx_disclaimer)
    assert idx_eval < idx_ability < idx_value < idx_strategy < idx_tendency < idx_desc < idx_csv < idx_disclaimer

    print("OK: mark note readability smoke passed")


if __name__ == "__main__":
    main()

import web_app


def main():
    rows = [
        {
            "mark": "◎",
            "horse_no": "1",
            "horse_name": "テストホースA",
            "pred_rank": "1",
            "bet_types": "place,wide",
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
            "bet_ev_norm": 0.88,
            "risk_signal": "見送り級",
            "race_type": "混戦",
            "confidence": "B",
            "gap_1_2": 0.0812,
        },
        {
            "mark": "▲",
            "horse_no": "3",
            "horse_name": "テストホースC",
            "pred_rank": "3",
            "bet_types": "place",
            "bet_ev_norm": 0.40,
            "risk_signal": "通常",
            "race_type": "混戦",
            "confidence": "B",
            "gap_1_2": 0.0812,
        },
    ]

    pred_csv = (
        "HorseName,Top3Prob_model,Top3Prob_lgbm,Top3Prob_lr,confidence_score,risk_score\n"
        "テストホースA,0.42,0.40,0.38,0.81,0.73\n"
        "テストホースB,0.36,0.33,0.31,0.81,0.73\n"
        "テストホースC,0.28,0.26,0.24,0.81,0.73\n"
    )
    text = web_app.build_mark_note_text(rows, predictions_filename="", predictions_csv_text=pred_csv)

    assert "【レース評価】" in text
    assert "AI評価：混戦" in text
    assert "AI信頼度：B（gap=0.081）" in text
    assert "予測データ総合評価：" in text
    assert "※AI信頼度=印5頭のgap" in text
    assert "/ 総合評価=0.65*confidence_score+0.35*risk_score" not in text
    assert "【印】" in text
    assert "テストホースA【注意】" in text
    assert "テストホースB【見送り級】" in text
    assert "■ AI戦略" in text
    assert "上位馬の能力差が小さく、展開次第で結果が変わる可能性。" in text
    assert "期待値（EV）計算の結果、" in text
    assert "今回は複勝を中心とした構成となりました。" in text
    assert "資金配分はリスクを抑えた保守的な設定です。" in text
    assert "【買い目傾向】" in text
    assert "買い目傾向：複勝・ワイド中心" in text
    assert "※AIは期待値(EV)ベースで券種を選択しています" in text
    assert "【予測データ説明】" in text
    assert "Top3Prob_model：統合モデルの3着内確率" in text
    assert "Top3Prob_lgbm：LightGBMモデル" in text
    assert "Top3Prob_lr：ロジスティック回帰モデル" in text
    assert "【CSVデータ】" in text
    assert "※本記事はAIによる予測データを公開するものであり、" in text
    assert "投資判断は自己責任でお願いします。" in text

    idx_ai = text.find("AI評価：")
    idx_conf = text.find("AI信頼度：")
    idx_grade = text.find("予測データ総合評価：")
    idx_mark = text.find("【印】")
    idx_strategy = text.find("■ AI戦略")
    idx_tendency = text.find("【買い目傾向】")
    idx_disclaimer = text.find("※本記事はAIによる予測データを公開するものであり、")
    idx_desc = text.find("【予測データ】")
    idx_csv = text.find("【CSVデータ】")
    assert -1 not in (idx_ai, idx_conf, idx_grade, idx_mark, idx_strategy, idx_tendency, idx_disclaimer, idx_desc, idx_csv)
    assert idx_ai < idx_conf < idx_grade < idx_mark < idx_strategy < idx_tendency < idx_disclaimer < idx_desc < idx_csv

    print("OK: mark note readability smoke passed")


if __name__ == "__main__":
    main()

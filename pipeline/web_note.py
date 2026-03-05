import csv
import io


def _escape_md_cell(value):
    return str(value or "").strip().replace("|", "\\|")


def _predict_col_desc(col):
    desc_map = {
        "HorseName": "馬名",
        "Age": "年齢",
        "SexMale": "性別フラグ（牡）",
        "SexFemale": "性別フラグ（牝）",
        "SexGelding": "性別フラグ（騸）",
        "TargetDistance": "対象距離（m）",
        "fieldsize_med": "想定頭数の中央値",
        "best_TimeIndexEff": "タイム指数効率（最良）",
        "avg_TimeIndexEff": "タイム指数効率（平均）",
        "dist_close": "距離適性の近さ",
        "Top3Prob_lr": "ロジスティック回帰モデル",
        "Top3Prob_lgbm": "LightGBMモデル",
        "Top3Prob_model": "統合モデルの3着内確率",
        "Top3Prob_est": "推定3着内確率",
        "Top3Prob": "3着内確率",
        "jscore_current": "騎手評価スコア（当該レース時点）",
        "agg_score": "総合評価スコア",
        "score": "評価スコア",
        "confidence_score": "予測信頼度スコア",
        "stability_score": "予測安定性スコア",
        "validity_score": "予測妥当性スコア",
        "consistency_score": "予測整合性スコア",
        "rank_ema": "順位実績のEMA指標",
        "ev_ema": "期待値のEMA指標",
        "risk_score": "安定度スコア（高いほど安定）",
    }
    if col in desc_map:
        return desc_map[col]
    if col.startswith("ti_"):
        return "タイム指数系特徴量"
    if col.startswith("jscore_"):
        return "騎手/補正スコア系特徴量"
    if col.startswith("ps_"):
        return "走法・ポジション系特徴量"
    if col.startswith("run_"):
        return "直近走パフォーマンス特徴量"
    if col.startswith("cup_"):
        return "同条件（クラス/距離/馬場）傾向特徴量"
    if col.startswith("top3_ti_"):
        return "上位3着タイム指数の分位特徴量"
    return "モデル特徴量（内部定義）"


def _build_predictions_column_guide(cols):
    if not cols:
        return []
    out = []
    for col in cols:
        name = str(col or "").strip()
        if not name or name == "HorseName":
            continue
        out.append(f"{name}: {_predict_col_desc(name)}")
    return out


def _to_float_or_zero(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_text_or_dash(value):
    text = str(value).strip()
    return text if text else "-"


def _extract_v3_param_summary(payload):
    if not isinstance(payload, dict):
        return {}
    params = payload.get("params")
    if isinstance(params, dict):
        return dict(params)
    return dict(payload)


def _select_prediction_score_key(fieldnames):
    for key in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if key in fieldnames:
            return key
    return ""


def _build_curated_predictions_csv(predictions_csv_text, top_n=None):
    text = str(predictions_csv_text or "").replace("\ufeff", "").strip()
    if not text:
        return "", [], 0, {}
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    except Exception:
        return "", [], 0, {}
    if not rows or not fieldnames:
        return "", [], 0, {}

    score_key = _select_prediction_score_key(fieldnames)
    if score_key:
        rows = sorted(rows, key=lambda r: _to_float_or_zero(r.get(score_key)), reverse=True)
    if top_n is not None:
        top_n = max(1, int(top_n))
        rows = rows[:top_n]

    preferred_cols = [
        "HorseName",
        "Top3Prob_model",
        "Top3Prob_lgbm",
        "Top3Prob_lr",
        "jscore_current",
    ]
    cols = [c for c in preferred_cols if c in fieldnames]
    if not cols:
        cols = fieldnames[:5]

    race_metric_keys = ["confidence_score", "risk_score", "rank_ema", "ev_ema"]
    race_metrics = {}
    for key in race_metric_keys:
        val = ""
        for row in rows:
            text = str(row.get(key, "")).strip()
            if text:
                val = text
                break
        if val:
            race_metrics[key] = val

    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=cols, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in cols})
    return sio.getvalue().strip(), cols, len(rows), race_metrics


def build_ai_strategy_text(race_type, bet_types_counter, risk_share):
    race = str(race_type or "").strip()
    if race == "一本命":
        race_text = "本命馬と対抗馬の能力差が大きく、比較的堅めのレース構造。"
    elif race == "混戦":
        race_text = "上位馬の能力差が小さく、展開次第で結果が変わる可能性。"
    else:
        race_text = "上位馬と中位馬の差が一定程度あり、標準的なレース構造。"

    counter = dict(bet_types_counter or {})
    place_n = int(counter.get("place", 0) or 0)
    wide_n = int(counter.get("wide", 0) or 0)
    win_n = int(counter.get("win", 0) or 0)
    top_type = max(
        [("place", place_n), ("wide", wide_n), ("win", win_n)],
        key=lambda x: (x[1], {"place": 3, "wide": 2, "win": 1}.get(x[0], 0)),
    )[0]
    if top_type == "place" and place_n > 0:
        bet_text = "今回は複勝を中心とした構成となりました。"
    elif top_type == "wide" and wide_n > 0:
        bet_text = "今回はワイドを軸にした分散型の構成となりました。"
    elif top_type == "win" and win_n > 0:
        bet_text = "今回は単勝を軸にした集中型の構成となりました。"
    else:
        bet_text = "今回は複数券種による分散構成となりました。"

    rs = _to_float_or_zero(risk_share)
    if rs < 0.2:
        risk_text = "資金配分はリスクを抑えた保守的な設定です。"
    elif rs <= 0.4:
        risk_text = "資金配分は標準的なリスク設定です。"
    else:
        risk_text = "資金配分はやや積極的なリスク設定です。"

    return f"■ AI戦略\n\n{race_text}\n\n期待値（EV）計算の結果、\n{bet_text}\n\n{risk_text}"


def _normalize_bet_type_tokens(text):
    return [t.strip().lower() for t in str(text or "").split(",") if t.strip()]


def _bet_types_to_label(text, sep="・"):
    bet_type_labels = {
        "win": "単勝",
        "place": "複勝",
        "wide": "ワイド",
        "quinella": "馬連",
        "trifecta": "三連複",
        "superfecta": "三連単",
        "-": "なし",
        "": "なし",
    }
    tokens = _normalize_bet_type_tokens(text)
    if not tokens:
        tokens = ["-"]
    return sep.join(bet_type_labels.get(t, t) for t in tokens)


def build_mark_note_text(
    ability_rows,
    value_rows=None,
    predictions_filename="",
    predictions_csv_text="",
    bet_engine_v3_summary=None,
):
    # Backward compatibility:
    # old: build_mark_note_text(rows, predictions_filename, predictions_csv_text)
    if isinstance(value_rows, str) and not predictions_csv_text:
        predictions_csv_text = str(predictions_filename or "")
        predictions_filename = str(value_rows or "")
        value_rows = []
    rows = list(ability_rows or [])
    value_rows = list(value_rows or [])

    csv_text, selected_cols, _, race_metrics = _build_curated_predictions_csv(predictions_csv_text, top_n=None)
    conf_text = str(race_metrics.get("confidence_score", "")).strip()
    risk_text = str(race_metrics.get("risk_score", "")).strip()
    grade = "-"
    if conf_text and risk_text:
        conf = _to_float_or_zero(conf_text)
        risk = _to_float_or_zero(risk_text)
        overall = 0.65 * conf + 0.35 * risk
        if overall >= 0.90:
            grade = "SSS"
        elif overall >= 0.84:
            grade = "SS"
        elif overall >= 0.78:
            grade = "S"
        elif overall >= 0.72:
            grade = "A"
        elif overall >= 0.64:
            grade = "B"
        else:
            grade = "C"

    lines = ["【AI競馬予想】AI能力印（◎○▲△☆）＋EV狙い馬（★）", ""]
    tendency_line = "見送り（買い目なし）"
    strategy_text = build_ai_strategy_text("通常", {}, 0.25).replace("■ AI戦略\n\n", "")
    race_type = "通常"
    confidence = "C"
    gap_val = 0.0
    if rows:
        race_type = str(rows[0].get("race_type", "")).strip() or "通常"
        confidence = str(rows[0].get("confidence", "")).strip() or "C"
        gap_val = _to_float_or_zero(rows[0].get("gap_1_2", 0.0))
        tendency_counter = {}
        risk_share = _to_float_or_zero(
            rows[0].get("risk_share", rows[0].get("target_risk_share", rows[0].get("risk_share_used", 0.25)))
        )
        strategy_text = build_ai_strategy_text(race_type, tendency_counter, risk_share).replace("■ AI戦略\n\n", "")
    else:
        tendency_counter = {}

    lines.extend(
        [
            "【AIレース評価】",
            f"AI評価：{race_type}",
            f"AI信頼度：{confidence}（gap={gap_val:.3f}）",
            f"予測データ総合評価：{grade}",
            "※AI信頼度=印5頭のgap",
            "",
            "【AI能力印（◎○▲△☆）】",
        ]
    )

    if rows:
        ev_max = max((_to_float_or_zero(r.get("bet_ev_norm", r.get("value_score", 0.0))) for r in rows), default=0.0)
        for row in rows:
            mark = _escape_md_cell(row.get("mark", ""))
            horse_no = _escape_md_cell(row.get("horse_no", "")) or "-"
            horse_name = _escape_md_cell(row.get("horse_name", ""))
            pred_rank = _escape_md_cell(row.get("pred_rank", "")) or "-"
            bet_types_raw = _escape_md_cell(row.get("recommended_bet_types", row.get("bet_types", ""))) or "-"
            for item in _normalize_bet_type_tokens(bet_types_raw):
                tendency_counter[item] = int(tendency_counter.get(item, 0)) + 1
            bet_types = _bet_types_to_label(bet_types_raw, sep="・")
            tags = [x.strip() for x in str(row.get("reason_tags", "")).split("/") if x.strip()]
            try:
                rank_val = int(pred_rank)
            except (TypeError, ValueError):
                rank_val = 999
            if rank_val <= 2 and "総合上位" not in tags:
                tags.append("総合上位")
            ev_norm = _to_float_or_zero(row.get("bet_ev_norm", row.get("value_score", 0.0)))
            if ev_max > 0 and abs(ev_norm - ev_max) <= 1e-9 and "期待値上位" not in tags:
                tags.append("期待値上位")
            if any(t in ("wide", "quinella") for t in _normalize_bet_type_tokens(bet_types_raw)):
                if "相手候補" not in tags:
                    tags.append("相手候補")
            reason_text = f"（{' / '.join(tags)}）" if tags else ""
            risk_signal = _escape_md_cell(row.get("risk_signal", ""))
            risk_label = ""
            if risk_signal == "注意":
                risk_label = "【注意】"
            elif risk_signal == "見送り級":
                risk_label = "【見送り級】"
            lines.append(f"{mark} {horse_no}番 {horse_name}{risk_label}{reason_text}")
            lines.append(f"  予想順位: {pred_rank} / 推奨券種: {bet_types}")
    else:
        lines.append("該当なし")

    valid_tendency = {k: v for k, v in tendency_counter.items() if k and k not in ("-", "なし")}
    if valid_tendency:
        top_items = sorted(valid_tendency.items(), key=lambda x: (-x[1], x[0]))[:2]
        tendency_line = "・".join(_bet_types_to_label(k, sep="") for k, _ in top_items) + "中心"

    lines.extend(["", "【EV狙い馬（★）】"])
    if value_rows:
        for row in value_rows:
            vmark = _escape_md_cell(row.get("value_mark", "★")) or "★"
            horse_no = _escape_md_cell(row.get("horse_no", "")) or "-"
            horse_name = _escape_md_cell(row.get("horse_name", ""))
            reasons = " / ".join([x.strip() for x in str(row.get("reason_tags", "")).split("/") if x.strip()])
            reason_text = f"（{reasons}）" if reasons else ""
            rec = _bet_types_to_label(row.get("recommended_bet_types", "win"), sep="・")
            lines.append(f"{vmark} {horse_no}番 {horse_name}{reason_text}")
            lines.append(f"  推奨：{rec}")
    else:
        lines.append("該当なし")

    lines.extend(
        [
            "",
            "【AI戦略】",
            strategy_text,
            "",
            "【買い目傾向】",
            tendency_line,
            "",
            "※AIは期待値(EV)ベースで券種を選択しています",
        ]
    )

    v3 = _extract_v3_param_summary(bet_engine_v3_summary)
    if v3:
        lines.extend(
            [
                "",
                "【AIベット設定（v3）】",
                f"- kelly_scale={_to_text_or_dash(v3.get('kelly_scale'))}",
                f"- min_p_hit={_to_text_or_dash(v3.get('min_p_hit_per_ticket'))}",
                f"- min_p_win={_to_text_or_dash(v3.get('min_p_win_per_ticket'))}",
                f"- min_edge={_to_text_or_dash(v3.get('min_edge_per_ticket'))}",
                f"- fallback_max_odds_place={_to_text_or_dash(v3.get('fallback_max_odds_place'))}",
                f"- high_exposure_cap_share={_to_text_or_dash(v3.get('high_exposure_cap_share'))}",
                f"- low_mid_min_share={_to_text_or_dash(v3.get('low_mid_min_share'))}",
            ]
        )

    if csv_text:
        lines.append("")
        lines.append("【予測データ説明】")
        lines.append("")
        lines.append("Top3Prob_model：統合モデルの3着内確率")
        lines.append("")
        lines.append("Top3Prob_lgbm：LightGBMモデル")
        lines.append("")
        lines.append("Top3Prob_lr：ロジスティック回帰モデル")
        lines.append("")
        if "jscore_current" in selected_cols:
            lines.append("jscore_current：騎手評価スコア")
            lines.append("")
        metric_labels = [
            ("confidence_score", "予測信頼度スコア"),
            ("risk_score", "安定度スコア"),
            ("rank_ema", "順位実績EMA"),
            ("ev_ema", "期待値EMA"),
        ]
        for key, label in metric_labels:
            val = race_metrics.get(key, "-")
            lines.append(f"{label}：{val}")
            lines.append("")
        lines.append("【CSVデータ】")
        lines.append("```csv")
        lines.append(csv_text.replace("```", "'''"))
        lines.append("```")

    lines.extend(["", "※本記事はAIによる予測データを公開するものであり、", "　投資判断は自己責任でお願いします。"])
    return "\n".join(lines).strip()

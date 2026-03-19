def _pick_first_value(row, keys, default=""):
    for key in keys:
        if key in row:
            value = row.get(key)
            text = str(value or "").strip()
            if text:
                return value
    return default


def _prediction_prob_value(row, to_float):
    for key in (
        "Top3Prob_model",
        "top3_prob_model",
        "Top3Prob_est",
        "top3_prob_est",
        "Top3Prob",
        "top3_prob",
        "agg_score",
        "score",
    ):
        if key in row:
            return max(0.0, to_float(row.get(key)))
    return 0.0


def _prediction_score_value(row, to_float):
    for key in (
        "rank_score_norm",
        "rank_score",
        "RankScore",
        "agg_score",
        "Top3Prob_model",
        "top3_prob_model",
        "Top3Prob_est",
        "top3_prob_est",
        "Top3Prob",
        "top3_prob",
        "score",
    ):
        if key in row:
            value = to_float(row.get(key))
            if value:
                return value
    return 0.0


def _linear_rank_norm(index, total):
    if total <= 1:
        return 1.0
    return round(max(0.05, 1.0 - (float(index) / float(total - 1))), 6)


def _normalize_score_list(items):
    if not items:
        return items
    values = [float(item.get("_raw_rank_score", 0.0) or 0.0) for item in items]
    max_v = max(values)
    min_v = min(values)
    if max_v > min_v:
        span = max_v - min_v
        for item in items:
            raw = float(item.get("_raw_rank_score", 0.0) or 0.0)
            item["rank_score_norm"] = round((raw - min_v) / span, 6)
        return items
    for idx, item in enumerate(items):
        item["rank_score_norm"] = _linear_rank_norm(idx, len(items))
    return items


def build_policy_prediction_rows(
    pred_rows,
    name_to_no_map,
    win_odds_map,
    place_odds_map,
    *,
    normalize_horse_no_text,
    normalize_name,
    parse_horse_no,
    to_float,
    to_int_or_none,
):
    items = []
    has_explicit_rank = False
    for idx, row in enumerate(list(pred_rows or []), start=1):
        horse_name = str(_pick_first_value(row, ("HorseName", "horse_name", "name"), "") or "").strip()
        if not horse_name:
            continue
        horse_no_raw = _pick_first_value(row, ("horse_no", "HorseNo", "umaban", "鬩ｬ逡ｪ"), "")
        horse_no = normalize_horse_no_text(horse_no_raw)
        if not horse_no:
            mapped = name_to_no_map.get(normalize_name(horse_name))
            if mapped is not None:
                horse_no = normalize_horse_no_text(mapped)
        top3_prob = _prediction_prob_value(row, to_float)
        explicit_rank = to_int_or_none(_pick_first_value(row, ("pred_rank", "PredRank", "rank", "Rank"), ""))
        if explicit_rank is not None and explicit_rank > 0:
            has_explicit_rank = True
        items.append(
            {
                "horse_no": horse_no,
                "horse_name": horse_name,
                "pred_rank": explicit_rank if explicit_rank is not None and explicit_rank > 0 else int(idx),
                "top3_prob_model": top3_prob,
                "confidence_score": max(0.0, to_float(row.get("confidence_score"))),
                "stability_score": max(0.0, to_float(row.get("stability_score"))),
                "risk_score": max(0.0, to_float(row.get("risk_score"))),
                "_raw_rank_score": _prediction_score_value(row, to_float),
                "_input_order": int(idx),
                "source_row": dict(row),
            }
        )
    if not items:
        return []
    if has_explicit_rank:
        items.sort(
            key=lambda item: (
                int(item.get("pred_rank", 9999) or 9999),
                -float(item.get("top3_prob_model", 0.0) or 0.0),
                str(item.get("horse_no", "")),
                str(item.get("horse_name", "")),
            )
        )
    else:
        items.sort(
            key=lambda item: (
                -float(item.get("top3_prob_model", 0.0) or 0.0),
                -float(item.get("_raw_rank_score", 0.0) or 0.0),
                str(item.get("horse_no", "")),
                str(item.get("horse_name", "")),
                int(item.get("_input_order", 9999) or 9999),
            )
        )
    for idx, item in enumerate(items, start=1):
        item["pred_rank"] = idx
    _normalize_score_list(items)
    top3_sum = sum(max(float(item.get("top3_prob_model", 0.0) or 0.0), 0.000001) for item in items)
    for item in items:
        horse_name = str(item.get("horse_name", "") or "")
        horse_no_int = parse_horse_no(item.get("horse_no"))
        item["win_odds"] = round(float(win_odds_map.get(normalize_name(horse_name), 0.0) or 0.0), 6)
        item["place_odds"] = round(float(place_odds_map.get(horse_no_int, 0.0) or 0.0), 6) if horse_no_int else 0.0
        item["win_prob_est"] = round(
            min(
                float(item.get("top3_prob_model", 0.0) or 0.0),
                float(item.get("top3_prob_model", 0.0) or 0.0) / max(top3_sum, 1e-6),
            ),
            6,
        )
    return items


def build_policy_candidates(
    predictions,
    wide_odds_map,
    quinella_odds_map,
    exacta_odds_map,
    trio_odds_map,
    trifecta_odds_map,
    allowed_types,
):
    candidates = []
    candidate_lookup = {}
    horse_map = {str(item.get("horse_no", "") or ""): item for item in list(predictions or []) if str(item.get("horse_no", "") or "").strip()}
    bet_type_order = {"win": 0, "place": 1, "wide": 2, "quinella": 3, "exacta": 4, "trio": 5, "trifecta": 6}
    for item in predictions:
        horse_no = str(item.get("horse_no", "") or "").strip()
        if not horse_no:
            continue
        if "win" in allowed_types:
            win_odds = float(item.get("win_odds", 0.0) or 0.0)
            if win_odds > 0:
                candidate = {"id": f"win:{horse_no}", "bet_type": "win", "legs": [horse_no], "odds_used": round(win_odds, 6), "p_hit": 0.0, "ev": 0.0, "score": 0.0}
                candidates.append(candidate)
                candidate_lookup[candidate["id"]] = candidate
        if "place" in allowed_types:
            place_odds = float(item.get("place_odds", 0.0) or 0.0)
            if place_odds > 0:
                candidate = {"id": f"place:{horse_no}", "bet_type": "place", "legs": [horse_no], "odds_used": round(place_odds, 6), "p_hit": 0.0, "ev": 0.0, "score": 0.0}
                candidates.append(candidate)
                candidate_lookup[candidate["id"]] = candidate
    if "wide" in allowed_types:
        for pair, odds in sorted(wide_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {"id": f"wide:{pair[0]}-{pair[1]}", "bet_type": "wide", "legs": [str(pair[0]), str(pair[1])], "odds_used": round(float(odds), 6), "p_hit": 0.0, "ev": 0.0, "score": 0.0}
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    if "quinella" in allowed_types:
        for pair, odds in sorted(quinella_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {"id": f"quinella:{pair[0]}-{pair[1]}", "bet_type": "quinella", "legs": [str(pair[0]), str(pair[1])], "odds_used": round(float(odds), 6), "p_hit": 0.0, "ev": 0.0, "score": 0.0}
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    if "exacta" in allowed_types:
        for pair, odds in sorted(exacta_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {"id": f"exacta:{pair[0]}-{pair[1]}", "bet_type": "exacta", "legs": [str(pair[0]), str(pair[1])], "odds_used": round(float(odds), 6), "p_hit": 0.0, "ev": 0.0, "score": 0.0}
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    if "trio" in allowed_types:
        for trio_key, odds in sorted(trio_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {"id": f"trio:{trio_key[0]}-{trio_key[1]}-{trio_key[2]}", "bet_type": "trio", "legs": [str(trio_key[0]), str(trio_key[1]), str(trio_key[2])], "odds_used": round(float(odds), 6), "p_hit": 0.0, "ev": 0.0, "score": 0.0}
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    if "trifecta" in allowed_types:
        for trio_key, odds in sorted(trifecta_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {"id": f"trifecta:{trio_key[0]}-{trio_key[1]}-{trio_key[2]}", "bet_type": "trifecta", "legs": [str(trio_key[0]), str(trio_key[1]), str(trio_key[2])], "odds_used": round(float(odds), 6), "p_hit": 0.0, "ev": 0.0, "score": 0.0}
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    candidates.sort(
        key=lambda item: (
            bet_type_order.get(str(item.get("bet_type", "") or ""), 99),
            float(item.get("odds_used", 0.0) or 0.0),
            str(item.get("id", "")),
        )
    )
    return candidates, candidate_lookup, horse_map


def build_pair_odds_top(candidate_lookup):
    rows = []
    for candidate in list(candidate_lookup.values()):
        if str(candidate.get("bet_type", "") or "") not in ("wide", "quinella", "exacta", "trio", "trifecta"):
            continue
        legs = list(candidate.get("legs", []) or [])
        if len(legs) < 2:
            continue
        rows.append(
            {
                "bet_type": str(candidate.get("bet_type", "") or ""),
                "pair": "-".join(str(x) for x in legs),
                "odds": round(float(candidate.get("odds_used", 0.0) or 0.0), 6),
                "score": float(candidate.get("score", 0.0) or 0.0),
            }
        )
    rows.sort(key=lambda item: (float(item.get("odds", 0.0) or 0.0), str(item.get("bet_type", "") or ""), str(item.get("pair", "") or "")))
    return [{"bet_type": row["bet_type"], "pair": row["pair"], "odds": row["odds"]} for row in rows[:10]]


def build_odds_full(win_rows, place_rows, wide_rows, quinella_rows, exacta_rows=None, trio_rows=None, trifecta_rows=None, *, to_float):
    def _pair_rows(rows):
        out = []
        for row in list(rows or []):
            a = str(row.get("horse_no_a", "") or "").strip()
            b = str(row.get("horse_no_b", "") or "").strip()
            if not a or not b:
                continue
            odds = to_float(row.get("odds_mid", row.get("odds", 0)) or 0)
            out.append({"pair": f"{a}-{b}", "horse_no_a": a, "horse_no_b": b, "odds": round(odds, 6)})
        return out

    def _triple_rows(rows):
        out = []
        for row in list(rows or []):
            a = str(row.get("horse_no_a", "") or "").strip()
            b = str(row.get("horse_no_b", "") or "").strip()
            c = str(row.get("horse_no_c", "") or "").strip()
            if not a or not b or not c:
                continue
            odds = to_float(row.get("odds", 0) or 0)
            out.append({"triple": f"{a}-{b}-{c}", "horse_no_a": a, "horse_no_b": b, "horse_no_c": c, "odds": round(odds, 6)})
        return out

    def _single_rows(rows, odds_key):
        out = []
        for row in list(rows or []):
            horse_no = str(row.get("horse_no", "") or "").strip()
            if not horse_no:
                continue
            out.append({"horse_no": horse_no, "name": str(row.get("name", "") or "").strip(), "odds": round(to_float(row.get(odds_key, 0) or 0), 6)})
        return out

    return {
        "win": _single_rows(win_rows, "odds"),
        "place": _single_rows(place_rows, "odds_low" if place_rows and "odds_low" in place_rows[0] else "odds_mid"),
        "wide": _pair_rows(wide_rows),
        "quinella": _pair_rows(quinella_rows),
        "exacta": _pair_rows(exacta_rows),
        "trio": _triple_rows(trio_rows),
        "trifecta": _triple_rows(trifecta_rows),
    }


def build_prediction_field_guide():
    return {
        "horse_no": "马番",
        "HorseName": "马名",
        "pred_rank": "预测排名",
        "Top3Prob_model": "模型前三概率",
        "rank_score_norm": "标准化排名分",
        "confidence_score": "信心分",
        "stability_score": "稳定分",
        "risk_score": "风险分",
        "win_odds": "单胜赔率",
        "place_odds": "复胜赔率",
    }


def build_policy_input_payload(
    scope_key,
    run_id,
    run_row,
    pred_path,
    odds_path,
    fuku_odds_path,
    wide_odds_path,
    quinella_odds_path,
    exacta_odds_path,
    trio_odds_path,
    trifecta_odds_path,
    policy_engine,
    *,
    load_csv_rows_flexible,
    load_name_to_no,
    load_win_odds_map,
    load_place_odds_map,
    load_pair_odds_map,
    load_exacta_odds_map,
    load_triple_odds_map,
    build_policy_prediction_rows_fn,
    extract_ledger_date,
    summarize_bankroll,
    base_dir,
    build_multi_predictor_context,
    build_history_context,
    to_float,
):
    pred_rows = load_csv_rows_flexible(pred_path)
    if not pred_rows:
        return None, "Prediction file is missing or empty."
    name_to_no_map = load_name_to_no(odds_path)
    win_odds_map = load_win_odds_map(odds_path)
    place_odds_map = load_place_odds_map(fuku_odds_path)
    wide_odds_map = load_pair_odds_map(wide_odds_path)
    quinella_odds_map = load_pair_odds_map(quinella_odds_path)
    exacta_odds_map = load_exacta_odds_map(exacta_odds_path)
    trio_odds_map = load_triple_odds_map(trio_odds_path, ordered=False)
    trifecta_odds_map = load_triple_odds_map(trifecta_odds_path, ordered=True)
    predictions = build_policy_prediction_rows_fn(pred_rows, name_to_no_map, win_odds_map, place_odds_map)
    if not predictions:
        return None, "No valid prediction rows could be built for policy input."
    allowed_types = []
    if any(float(item.get("win_odds", 0.0) or 0.0) > 0 for item in predictions):
        allowed_types.append("win")
    if any(float(item.get("place_odds", 0.0) or 0.0) > 0 for item in predictions):
        allowed_types.append("place")
    if wide_odds_map:
        allowed_types.append("wide")
    if quinella_odds_map:
        allowed_types.append("quinella")
    if exacta_odds_map:
        allowed_types.append("exacta")
    if trio_odds_map:
        allowed_types.append("trio")
    if trifecta_odds_map:
        allowed_types.append("trifecta")
    if not allowed_types:
        return None, "No usable odds were found for LLM buy."
    candidates, candidate_lookup, horse_map = build_policy_candidates(
        predictions,
        wide_odds_map,
        quinella_odds_map,
        exacta_odds_map,
        trio_odds_map,
        trifecta_odds_map,
        allowed_types,
    )
    if not candidates:
        return None, "Candidate generation failed because odds data is incomplete."
    ledger_date = extract_ledger_date(run_id, (run_row or {}).get("timestamp", ""))
    bankroll = summarize_bankroll(base_dir, ledger_date, policy_engine=policy_engine)
    bankroll_yen = max(0, int(bankroll.get("available_bankroll_yen", 0) or 0))
    multi_predictor = build_multi_predictor_context(scope_key, run_id, run_row, name_to_no_map, win_odds_map, place_odds_map)
    payload = {
        "race_id": str((run_row or {}).get("race_id", "") or ""),
        "scope_key": str(scope_key or ""),
        "field_size": len(predictions),
        "ai": {
            "gap": round(
                max(
                    0.0,
                    float(predictions[0].get("top3_prob_model", 0.0) or 0.0)
                    - float(predictions[1].get("top3_prob_model", 0.0) or 0.0 if len(predictions) > 1 else 0.0),
                ),
                6,
            ),
            "confidence_score": round(float(predictions[0].get("confidence_score", 0.5) or 0.5), 6),
            "stability_score": round(float(predictions[0].get("stability_score", 0.5) or 0.5), 6),
            "risk_score": round(float(predictions[0].get("risk_score", 0.5) or 0.5), 6),
        },
        "marks_top5": [
            {
                "horse_no": str(item.get("horse_no", "") or ""),
                "horse_name": str(item.get("horse_name", "") or ""),
                "pred_rank": int(item.get("pred_rank", 0) or 0),
                "top3_prob_model": round(float(item.get("top3_prob_model", 0.0) or 0.0), 6),
                "rank_score_norm": round(float(item.get("rank_score_norm", 0.0) or 0.0), 6),
            }
            for item in predictions[:5]
        ],
        "predictions": [
            {
                "horse_no": str(item.get("horse_no", "") or ""),
                "horse_name": str(item.get("horse_name", "") or ""),
                "pred_rank": int(item.get("pred_rank", 0) or 0),
                "top3_prob_model": round(float(item.get("top3_prob_model", 0.0) or 0.0), 6),
                "rank_score_norm": round(float(item.get("rank_score_norm", 0.0) or 0.0), 6),
                "win_odds": round(float(item.get("win_odds", 0.0) or 0.0), 6),
                "place_odds": round(float(item.get("place_odds", 0.0) or 0.0), 6),
            }
            for item in predictions[:10]
        ],
        "predictions_full": [
            {
                **dict(item.get("source_row", {}) or {}),
                "horse_no": str(item.get("horse_no", "") or ""),
                "HorseName": str(item.get("horse_name", "") or ""),
                "pred_rank": int(item.get("pred_rank", 0) or 0),
                "Top3Prob_model": round(float(item.get("top3_prob_model", 0.0) or 0.0), 6),
                "rank_score_norm": round(float(item.get("rank_score_norm", 0.0) or 0.0), 6),
                "win_odds": round(float(item.get("win_odds", 0.0) or 0.0), 6),
                "place_odds": round(float(item.get("place_odds", 0.0) or 0.0), 6),
            }
            for item in predictions
        ],
        "pair_odds_top": build_pair_odds_top(candidate_lookup),
        "odds_full": build_odds_full(
            load_csv_rows_flexible(odds_path),
            load_csv_rows_flexible(fuku_odds_path),
            load_csv_rows_flexible(wide_odds_path),
            load_csv_rows_flexible(quinella_odds_path),
            load_csv_rows_flexible(exacta_odds_path),
            load_csv_rows_flexible(trio_odds_path),
            load_csv_rows_flexible(trifecta_odds_path),
            to_float=to_float,
        ),
        "prediction_field_guide": build_prediction_field_guide(),
        "multi_predictor": multi_predictor,
        "portfolio_history": build_history_context(base_dir, ledger_date, lookback_days=14, recent_ticket_limit=8, policy_engine=policy_engine),
        "candidates": candidates,
        "constraints": {
            "bankroll_yen": bankroll_yen,
            "race_budget_yen": bankroll_yen,
            "max_tickets_per_race": min(8, max(1, len(allowed_types) * 2)),
            "high_odds_threshold": 12.0,
            "allowed_types": allowed_types,
        },
    }
    return {
        "input": payload,
        "predictions": predictions,
        "candidate_lookup": candidate_lookup,
        "horse_map": horse_map,
        "summary_before": bankroll,
        "ledger_date": ledger_date,
    }, ""

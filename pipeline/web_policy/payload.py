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


def _clamp(value, low=0.0, high=1.0):
    return max(float(low), min(float(high), float(value)))


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_consensus_map(consensus_rows):
    out = {}
    for row in list(consensus_rows or []):
        horse_no = str(row.get("horse_no", "") or "").strip()
        if horse_no:
            out[horse_no] = dict(row)
    return out


def _build_horse_signal_map(predictions, consensus_rows):
    consensus_map = _build_consensus_map(consensus_rows)
    out = {}
    field_size = max(1, len(list(predictions or [])))
    for item in list(predictions or []):
        horse_no = str(item.get("horse_no", "") or "").strip()
        if not horse_no:
            continue
        consensus = dict(consensus_map.get(horse_no, {}) or {})
        predictor_count = max(1, int(consensus.get("predictor_count", 0) or 0))
        top3_prob = _clamp(item.get("top3_prob_model", 0.0) or 0.0)
        win_prob = _clamp(item.get("win_prob_est", min(top3_prob, top3_prob * 0.58)) or 0.0)
        rank_score = _clamp(item.get("rank_score_norm", 0.0) or 0.0)
        top1_vote_ratio = _clamp((consensus.get("top1_votes", 0) or 0) / float(predictor_count))
        top3_vote_ratio = _clamp((consensus.get("top3_votes", 0) or 0) / float(predictor_count))
        rank_std = _safe_float(consensus.get("rank_std", 0.0), 0.0)
        disagreement = _clamp(rank_std / max(1.0, field_size / 3.0))
        agreement = _clamp(1.0 - disagreement)
        support_score = _clamp(0.45 * top3_prob + 0.20 * win_prob + 0.20 * rank_score + 0.15 * top3_vote_ratio)
        out[horse_no] = {
            "horse_no": horse_no,
            "horse_name": str(item.get("horse_name", "") or ""),
            "top3_prob": top3_prob,
            "win_prob": win_prob,
            "place_prob": _clamp(top3_prob * (0.96 + 0.08 * agreement), 0.0, 0.95),
            "rank_score": rank_score,
            "top1_vote_ratio": top1_vote_ratio,
            "top3_vote_ratio": top3_vote_ratio,
            "agreement": agreement,
            "disagreement": disagreement,
            "support_score": support_score,
            "pred_rank": int(item.get("pred_rank", 0) or 0),
        }
    return out


def _score_single_candidate(item, signal_map, bet_type):
    horse_no = str(item.get("horse_no", "") or "").strip()
    signal = dict(signal_map.get(horse_no, {}) or {})
    if not signal:
        return 0.0, 0.0, 0.0
    odds_used = _safe_float(item.get("win_odds" if bet_type == "win" else "place_odds", 0.0), 0.0)
    if odds_used <= 0:
        return 0.0, 0.0, 0.0
    implied_prob = _clamp(1.0 / odds_used) if odds_used > 0 else 0.0
    if bet_type == "win":
        raw_p_hit = signal.get("win_prob", 0.0) * (0.90 + 0.15 * signal.get("agreement", 0.0))
        p_hit = _clamp(min(raw_p_hit, implied_prob * 3.0 + 0.03), 0.0, 0.45)
    else:
        raw_p_hit = signal.get("place_prob", 0.0) * (0.94 + 0.08 * signal.get("agreement", 0.0))
        p_hit = _clamp(min(raw_p_hit, implied_prob * 1.8 + 0.08), 0.0, 0.92)
    ev = round(p_hit * odds_used - 1.0, 6)
    ev_for_score = max(-1.0, min(1.5, ev))
    score = round(ev_for_score * 0.45 + p_hit * 0.30 + signal.get("support_score", 0.0) * 0.25, 6)
    return round(p_hit, 6), ev, score


def _score_combo_candidate(legs, signal_map, bet_type, odds_used):
    signals = [dict(signal_map.get(str(leg), {}) or {}) for leg in list(legs or [])]
    if not signals or any(not signal for signal in signals) or float(odds_used or 0.0) <= 0:
        return 0.0, 0.0, 0.0
    agreement = sum(float(signal.get("agreement", 0.0) or 0.0) for signal in signals) / float(len(signals))
    support = sum(float(signal.get("support_score", 0.0) or 0.0) for signal in signals) / float(len(signals))
    place_probs = [float(signal.get("place_prob", 0.0) or 0.0) for signal in signals]
    win_probs = [float(signal.get("win_prob", 0.0) or 0.0) for signal in signals]
    if bet_type == "wide":
        p_hit = 0.72 * place_probs[0] * place_probs[1] * (0.85 + 0.15 * agreement)
    elif bet_type == "quinella":
        p_hit = 0.42 * place_probs[0] * place_probs[1] * (0.82 + 0.18 * agreement)
    elif bet_type == "exacta":
        p_hit = 0.34 * win_probs[0] * place_probs[1] * (0.82 + 0.18 * agreement)
    elif bet_type == "trio":
        p_hit = 0.26 * place_probs[0] * place_probs[1] * place_probs[2] * (0.82 + 0.18 * agreement)
    else:
        p_hit = 0.0
    implied_prob = _clamp(1.0 / float(odds_used)) if float(odds_used) > 0 else 0.0
    type_cap = {
        "wide": 0.55,
        "quinella": 0.35,
        "exacta": 0.22,
        "trio": 0.16,
    }.get(bet_type, 0.30)
    p_hit = _clamp(min(p_hit, implied_prob * 3.5 + 0.02), 0.0, type_cap)
    ev = round(p_hit * float(odds_used) - 1.0, 6)
    ev_for_score = max(-1.0, min(1.8, ev))
    score = round(ev_for_score * 0.50 + p_hit * 0.20 + support * 0.30, 6)
    return round(p_hit, 6), ev, score


def _trim_candidates(candidates):
    quotas = {
        "win": 6,
        "place": 6,
        "wide": 16,
        "quinella": 12,
        "exacta": 14,
        "trio": 10,
    }
    grouped = {}
    for candidate in list(candidates or []):
        bet_type = str(candidate.get("bet_type", "") or "").strip().lower()
        grouped.setdefault(bet_type, []).append(candidate)
    selected = []
    for bet_type, items in grouped.items():
        items.sort(
            key=lambda item: (
                -float(item.get("score", 0.0) or 0.0),
                -float(item.get("ev", 0.0) or 0.0),
                -float(item.get("p_hit", 0.0) or 0.0),
                float(item.get("odds_used", 0.0) or 0.0),
                str(item.get("id", "")),
            )
        )
        selected.extend(items[: quotas.get(bet_type, 8)])
    selected.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -float(item.get("ev", 0.0) or 0.0),
            -float(item.get("p_hit", 0.0) or 0.0),
            float(item.get("odds_used", 0.0) or 0.0),
            str(item.get("id", "")),
        )
    )
    return selected[:80]


def build_policy_candidates(
    predictions,
    wide_odds_map,
    quinella_odds_map,
    exacta_odds_map,
    trio_odds_map,
    allowed_types,
    consensus_rows=None,
):
    candidates = []
    candidate_lookup = {}
    horse_map = {str(item.get("horse_no", "") or ""): item for item in list(predictions or []) if str(item.get("horse_no", "") or "").strip()}
    signal_map = _build_horse_signal_map(predictions, consensus_rows)
    for item in predictions:
        horse_no = str(item.get("horse_no", "") or "").strip()
        if not horse_no:
            continue
        if "win" in allowed_types:
            win_odds = float(item.get("win_odds", 0.0) or 0.0)
            if win_odds > 0:
                p_hit, ev, score = _score_single_candidate(item, signal_map, "win")
                candidate = {"id": f"win:{horse_no}", "bet_type": "win", "legs": [horse_no], "odds_used": round(win_odds, 6), "p_hit": p_hit, "ev": ev, "score": score}
                candidates.append(candidate)
        if "place" in allowed_types:
            place_odds = float(item.get("place_odds", 0.0) or 0.0)
            if place_odds > 0:
                p_hit, ev, score = _score_single_candidate(item, signal_map, "place")
                candidate = {"id": f"place:{horse_no}", "bet_type": "place", "legs": [horse_no], "odds_used": round(place_odds, 6), "p_hit": p_hit, "ev": ev, "score": score}
                candidates.append(candidate)
    if "wide" in allowed_types:
        for pair, odds in sorted(wide_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            legs = [str(pair[0]), str(pair[1])]
            p_hit, ev, score = _score_combo_candidate(legs, signal_map, "wide", odds)
            candidate = {"id": f"wide:{pair[0]}-{pair[1]}", "bet_type": "wide", "legs": legs, "odds_used": round(float(odds), 6), "p_hit": p_hit, "ev": ev, "score": score}
            candidates.append(candidate)
    if "quinella" in allowed_types:
        for pair, odds in sorted(quinella_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            legs = [str(pair[0]), str(pair[1])]
            p_hit, ev, score = _score_combo_candidate(legs, signal_map, "quinella", odds)
            candidate = {"id": f"quinella:{pair[0]}-{pair[1]}", "bet_type": "quinella", "legs": legs, "odds_used": round(float(odds), 6), "p_hit": p_hit, "ev": ev, "score": score}
            candidates.append(candidate)
    if "exacta" in allowed_types:
        for pair, odds in sorted(exacta_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            legs = [str(pair[0]), str(pair[1])]
            p_hit, ev, score = _score_combo_candidate(legs, signal_map, "exacta", odds)
            candidate = {"id": f"exacta:{pair[0]}-{pair[1]}", "bet_type": "exacta", "legs": legs, "odds_used": round(float(odds), 6), "p_hit": p_hit, "ev": ev, "score": score}
            candidates.append(candidate)
    if "trio" in allowed_types:
        for trio_key, odds in sorted(trio_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            legs = [str(trio_key[0]), str(trio_key[1]), str(trio_key[2])]
            p_hit, ev, score = _score_combo_candidate(legs, signal_map, "trio", odds)
            candidate = {"id": f"trio:{trio_key[0]}-{trio_key[1]}-{trio_key[2]}", "bet_type": "trio", "legs": legs, "odds_used": round(float(odds), 6), "p_hit": p_hit, "ev": ev, "score": score}
            candidates.append(candidate)
    candidates = _trim_candidates(candidates)
    candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -float(item.get("ev", 0.0) or 0.0),
            -float(item.get("p_hit", 0.0) or 0.0),
            str(item.get("bet_type", "") or ""),
            float(item.get("odds_used", 0.0) or 0.0),
            str(item.get("id", "")),
        )
    )
    for candidate in candidates:
        candidate_lookup[candidate["id"]] = candidate
    return candidates, candidate_lookup, horse_map


def build_pair_odds_top(candidate_lookup):
    rows = []
    for candidate in list(candidate_lookup.values()):
        if str(candidate.get("bet_type", "") or "") not in ("wide", "quinella", "exacta", "trio"):
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


def build_odds_full(win_rows, place_rows, wide_rows, quinella_rows, exacta_rows=None, trio_rows=None, *, to_float):
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


def _clone_prediction_row(item):
    return {
        "horse_no": str(item.get("horse_no", "") or ""),
        "horse_name": str(item.get("horse_name", "") or ""),
        "pred_rank": int(item.get("pred_rank", 0) or 0),
        "top3_prob_model": round(float(item.get("top3_prob_model", 0.0) or 0.0), 6),
        "rank_score_norm": round(float(item.get("rank_score_norm", 0.0) or 0.0), 6),
        "win_odds": round(float(item.get("win_odds", 0.0) or 0.0), 6),
        "place_odds": round(float(item.get("place_odds", 0.0) or 0.0), 6),
        "confidence_score": round(float(item.get("confidence_score", 0.0) or 0.0), 6),
        "stability_score": round(float(item.get("stability_score", 0.0) or 0.0), 6),
        "risk_score": round(float(item.get("risk_score", 0.0) or 0.0), 6),
        "source_row": dict(item.get("source_row", {}) or {}),
    }


def _choose_primary_predictions(multi_predictor, fallback_predictions):
    multi_predictor = dict(multi_predictor or {})
    meta = dict(multi_predictor.get("meta", {}) or {})
    ranking_by_id = {}
    for block in list(multi_predictor.get("predictor_rankings", []) or []):
        predictor_id = str(block.get("predictor_id", "") or "").strip()
        ranking = [_clone_prediction_row(item) for item in list(block.get("ranking", []) or [])]
        if predictor_id and ranking:
            ranking_by_id[predictor_id] = ranking

    preferred_ids = []
    primary_predictor_id = str(meta.get("primary_predictor_id", "") or "").strip()
    if primary_predictor_id:
        preferred_ids.append(primary_predictor_id)
    for predictor_id in ("v5_stacking", "v4_gemini", "v3_premium", "v2_opus", "main"):
        if predictor_id not in preferred_ids:
            preferred_ids.append(predictor_id)

    for predictor_id in preferred_ids:
        ranking = ranking_by_id.get(predictor_id)
        if ranking:
            return ranking, predictor_id
    return [_clone_prediction_row(item) for item in list(fallback_predictions or [])], "main"


def _load_selected_predictor_predictions(selected_predictor_id, run_row, fallback_path, load_csv_rows_flexible, build_policy_prediction_rows_fn, name_to_no_map, win_odds_map, place_odds_map):
    field_map = {
        "main": "predictions_path",
        "v2_opus": "predictions_v2_opus_path",
        "v3_premium": "predictions_v3_premium_path",
        "v4_gemini": "predictions_v4_gemini_path",
        "v5_stacking": "predictions_v5_stacking_path",
    }
    selected_field = field_map.get(str(selected_predictor_id or "").strip(), "predictions_path")
    selected_path = str((run_row or {}).get(selected_field, "") or "").strip()
    if not selected_path:
        selected_path = str(fallback_path or "")
    if not selected_path:
        return []
    pred_rows = load_csv_rows_flexible(selected_path)
    if not pred_rows:
        return []
    return build_policy_prediction_rows_fn(pred_rows, name_to_no_map, win_odds_map, place_odds_map)


def _build_race_context(run_row, scope_key, field_size):
    row = dict(run_row or {})
    distance = str(row.get("distance", "") or "").strip()
    try:
        distance_value = int(float(distance)) if distance else 0
    except (TypeError, ValueError):
        distance_value = 0
    return {
        "race_id": str(row.get("race_id", "") or ""),
        "scope_key": str(scope_key or ""),
        "location": str(row.get("location", "") or ""),
        "surface": str(row.get("surface", "") or ""),
        "distance": distance_value,
        "track_condition": str(row.get("track_condition", "") or ""),
        "race_date": str(row.get("race_date", "") or ""),
        "field_size": int(field_size or 0),
    }


def _build_multi_model_ai(predictions, multi_predictor):
    consensus = list((multi_predictor or {}).get("consensus", []) or [])
    if not predictions:
        return {
            "compatibility_primary_predictor_id": str((multi_predictor or {}).get("meta", {}).get("compatibility_primary_predictor_id", "") or ""),
            "consensus_gap": 0.0,
            "top1_vote_margin": 0.0,
            "mean_rank_std": 0.0,
            "max_rank_std": 0.0,
            "mean_top3_prob_range": 0.0,
            "disagreement_score": 0.0,
        }
    top_cons = consensus[:2]
    consensus_gap = 0.0
    top1_vote_margin = 0.0
    if top_cons:
        first_prob = _safe_float(top_cons[0].get("avg_top3_prob_model", 0.0), 0.0)
        second_prob = _safe_float(top_cons[1].get("avg_top3_prob_model", 0.0), 0.0) if len(top_cons) > 1 else 0.0
        consensus_gap = round(max(0.0, first_prob - second_prob), 6)
        first_votes = _safe_float(top_cons[0].get("top1_votes", 0.0), 0.0)
        second_votes = _safe_float(top_cons[1].get("top1_votes", 0.0), 0.0) if len(top_cons) > 1 else 0.0
        predictor_count = max(1.0, _safe_float(top_cons[0].get("predictor_count", 1.0), 1.0))
        top1_vote_margin = round(max(0.0, (first_votes - second_votes) / predictor_count), 6)
    rank_stds = [_safe_float(row.get("rank_std", 0.0), 0.0) for row in consensus]
    prob_ranges = [_safe_float(row.get("top3_prob_range", 0.0), 0.0) for row in consensus]
    mean_rank_std = sum(rank_stds) / float(len(rank_stds)) if rank_stds else 0.0
    mean_prob_range = sum(prob_ranges) / float(len(prob_ranges)) if prob_ranges else 0.0
    disagreement_score = _clamp((mean_rank_std / max(1.0, len(predictions) / 3.0)) * 0.6 + min(1.0, mean_prob_range / 0.35) * 0.4)
    return {
        "compatibility_primary_predictor_id": str((multi_predictor or {}).get("meta", {}).get("compatibility_primary_predictor_id", "") or ""),
        "consensus_gap": round(consensus_gap, 6),
        "top1_vote_margin": round(top1_vote_margin, 6),
        "mean_rank_std": round(mean_rank_std, 6),
        "max_rank_std": round(max(rank_stds), 6) if rank_stds else 0.0,
        "mean_top3_prob_range": round(mean_prob_range, 6),
        "disagreement_score": round(disagreement_score, 6),
    }


def _first_non_empty(item, keys):
    source = dict(item.get("source_row", {}) or {})
    for key in list(keys or []):
        value = source.get(key)
        text = str(value or "").strip()
        if text:
            return value
    return ""


def _build_horse_facts(predictions, consensus_rows):
    consensus_map = _build_consensus_map(consensus_rows)
    facts = []
    for item in list(predictions or []):
        horse_no = str(item.get("horse_no", "") or "").strip()
        if not horse_no:
            continue
        consensus = dict(consensus_map.get(horse_no, {}) or {})
        odds = _safe_float(item.get("win_odds", 0.0), 0.0)
        implied_prob = _safe_float(_first_non_empty(item, ("implied_prob_win", "win_prob_est")), 0.0)
        if implied_prob <= 0 and odds > 0:
            implied_prob = 1.0 / odds
        facts.append(
            {
                "horse_no": horse_no,
                "horse_name": str(item.get("horse_name", "") or ""),
                "pred_rank": int(item.get("pred_rank", 0) or 0),
                "consensus_avg_rank": round(_safe_float(consensus.get("avg_pred_rank", 0.0), 0.0), 4),
                "consensus_rank_std": round(_safe_float(consensus.get("rank_std", 0.0), 0.0), 4),
                "consensus_top1_votes": int(consensus.get("top1_votes", 0) or 0),
                "consensus_top3_votes": int(consensus.get("top3_votes", 0) or 0),
                "last_ti": round(_safe_float(_first_non_empty(item, ("ti_last", "adj_ti_last", "f_ti_last")), 0.0), 6),
                "recent_ti_mean": round(_safe_float(_first_non_empty(item, ("ti_mean3", "ti_mean5", "adj_ti_mean5", "f_ti_mean5")), 0.0), 6),
                "career_runs": int(_safe_float(_first_non_empty(item, ("history_count", "f_career_races", "dist_exp_count")), 0.0)),
                "surface_exp_ratio": round(_safe_float(_first_non_empty(item, ("surface_exp_ratio", "f_surface_exp_ratio")), 0.0), 6),
                "jockey_score": round(_safe_float(_first_non_empty(item, ("jockey_score", "jscore_last", "jockey_win_course")), 0.0), 6),
                "odds": round(odds, 6),
                "implied_prob": round(_clamp(implied_prob), 6),
                "rest_days": int(_safe_float(_first_non_empty(item, ("rest_days",)), 0.0)),
            }
        )
    return facts


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
    name_to_no_map = load_name_to_no(odds_path)
    win_odds_map = load_win_odds_map(odds_path)
    place_odds_map = load_place_odds_map(fuku_odds_path)
    wide_odds_map = load_pair_odds_map(wide_odds_path)
    quinella_odds_map = load_pair_odds_map(quinella_odds_path)
    exacta_odds_map = load_exacta_odds_map(exacta_odds_path)
    trio_odds_map = load_triple_odds_map(trio_odds_path, ordered=False)
    multi_predictor = build_multi_predictor_context(scope_key, run_id, run_row, name_to_no_map, win_odds_map, place_odds_map)
    pred_rows = load_csv_rows_flexible(pred_path)
    fallback_predictions = build_policy_prediction_rows_fn(pred_rows, name_to_no_map, win_odds_map, place_odds_map) if pred_rows else []
    predictions, selected_predictor_id = _choose_primary_predictions(multi_predictor, fallback_predictions)
    selected_predictions = _load_selected_predictor_predictions(
        selected_predictor_id,
        run_row,
        pred_path,
        load_csv_rows_flexible,
        build_policy_prediction_rows_fn,
        name_to_no_map,
        win_odds_map,
        place_odds_map,
    )
    if selected_predictions:
        predictions = selected_predictions
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
    if not allowed_types:
        return None, "No usable odds were found for LLM buy."
    candidates, candidate_lookup, horse_map = build_policy_candidates(
        predictions,
        wide_odds_map,
        quinella_odds_map,
        exacta_odds_map,
        trio_odds_map,
        allowed_types,
        consensus_rows=multi_predictor.get("consensus", []),
    )
    if not candidates:
        return None, "Candidate generation failed because odds data is incomplete."
    ledger_date = extract_ledger_date(run_id, (run_row or {}).get("timestamp", ""))
    bankroll = summarize_bankroll(base_dir, ledger_date, policy_engine=policy_engine)
    bankroll_yen = max(0, int(bankroll.get("available_bankroll_yen", 0) or 0))
    multi_predictor_meta = dict(multi_predictor.get("meta", {}) or {})
    multi_predictor_meta["primary_predictor_id"] = selected_predictor_id
    multi_predictor_meta["compatibility_primary_predictor_id"] = selected_predictor_id
    multi_predictor_meta["predictor_rankings_nonempty"] = all(
        len(list(block.get("ranking", []) or [])) > 0 for block in list(multi_predictor.get("predictor_rankings", []) or [])
    )
    multi_predictor_meta["empty_predictor_ids"] = [
        str(block.get("predictor_id", "") or "")
        for block in list(multi_predictor.get("predictor_rankings", []) or [])
        if len(list(block.get("ranking", []) or [])) <= 0
    ]
    multi_predictor["meta"] = multi_predictor_meta
    race_context = _build_race_context(run_row, scope_key, len(predictions))
    multi_model_ai = _build_multi_model_ai(predictions, multi_predictor)
    horse_facts = _build_horse_facts(predictions, multi_predictor.get("consensus", []))
    payload = {
        "race_id": str((run_row or {}).get("race_id", "") or ""),
        "scope_key": str(scope_key or ""),
        "field_size": len(predictions),
        "race_context": race_context,
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
        "multi_model_ai": multi_model_ai,
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
                "horse_no": str(item.get("horse_no", "") or ""),
                "HorseName": str(item.get("horse_name", "") or ""),
                "pred_rank": int(item.get("pred_rank", 0) or 0),
                "Top3Prob_model": round(float(item.get("top3_prob_model", 0.0) or 0.0), 6),
                "rank_score_norm": round(float(item.get("rank_score_norm", 0.0) or 0.0), 6),
                "confidence_score": round(float(item.get("confidence_score", 0.0) or 0.0), 6),
                "stability_score": round(float(item.get("stability_score", 0.0) or 0.0), 6),
                "risk_score": round(float(item.get("risk_score", 0.0) or 0.0), 6),
                "win_prob_est": round(float(item.get("win_prob_est", 0.0) or 0.0), 6),
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
            to_float=to_float,
        ),
        "prediction_field_guide": build_prediction_field_guide(),
        "multi_predictor": multi_predictor,
        "horse_facts": horse_facts,
        "portfolio_history": build_history_context(base_dir, ledger_date, lookback_days=14, recent_ticket_limit=8, policy_engine=policy_engine),
        "candidates": candidates,
        "candidates_meta": {
            "shortlist_count": len(candidates),
            "allowed_types": list(allowed_types),
            "selection_policy": "scored_shortlist_v1",
            "counts_by_type": {
                bet_type: sum(1 for item in candidates if str(item.get("bet_type", "") or "") == bet_type)
                for bet_type in list(allowed_types)
            },
        },
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

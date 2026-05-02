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
        mapped = name_to_no_map.get(normalize_name(horse_name))
        mapped_horse_no = normalize_horse_no_text(mapped) if mapped is not None else ""
        horse_no = mapped_horse_no or normalize_horse_no_text(horse_no_raw)
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
        anchor_strength = _clamp(0.6 * top1_vote_ratio + 0.4 * top3_vote_ratio)
        support_score = _clamp(
            0.30 * top3_prob
            + 0.10 * win_prob
            + 0.12 * rank_score
            + 0.26 * top1_vote_ratio
            + 0.18 * top3_vote_ratio
            - 0.05 * disagreement
        )
        out[horse_no] = {
            "horse_no": horse_no,
            "horse_name": str(item.get("horse_name", "") or ""),
            "top3_prob": top3_prob,
            "win_prob": win_prob,
            "place_prob": _clamp(top3_prob * (0.95 + 0.08 * agreement) * (0.95 + 0.07 * anchor_strength), 0.0, 0.95),
            "rank_score": rank_score,
            "top1_vote_ratio": top1_vote_ratio,
            "top3_vote_ratio": top3_vote_ratio,
            "anchor_strength": anchor_strength,
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
    anchor_strength = float(signal.get("anchor_strength", 0.0) or 0.0)
    if bet_type == "win":
        raw_p_hit = signal.get("win_prob", 0.0) * (0.88 + 0.16 * signal.get("agreement", 0.0)) * (0.88 + 0.18 * anchor_strength)
        p_hit = _clamp(min(raw_p_hit, implied_prob * 3.0 + 0.03), 0.0, 0.45)
    else:
        raw_p_hit = signal.get("place_prob", 0.0) * (0.93 + 0.09 * signal.get("agreement", 0.0)) * (0.90 + 0.12 * anchor_strength)
        p_hit = _clamp(min(raw_p_hit, implied_prob * 1.8 + 0.08), 0.0, 0.92)
    ev = round(p_hit * odds_used - 1.0, 6)
    ev_for_score = max(-1.0, min(1.5, ev))
    quant_support_score = round(float(signal.get("support_score", 0.0) or 0.0), 6)
    quant_anchor_strength = round(anchor_strength, 6)
    quant_agreement = round(float(signal.get("agreement", 0.0) or 0.0), 6)
    type_bias = {"place": 0.05, "win": 0.0}.get(str(bet_type or "").strip().lower(), 0.0)
    score = round(
        p_hit * 0.52
        + quant_support_score * 0.24
        + quant_anchor_strength * 0.14
        + quant_agreement * 0.08
        + ev_for_score * 0.02
        + type_bias,
        6,
    )
    return round(p_hit, 6), ev, score, quant_support_score, quant_anchor_strength, quant_agreement


def _score_combo_candidate(legs, signal_map, bet_type, odds_used):
    signals = [dict(signal_map.get(str(leg), {}) or {}) for leg in list(legs or [])]
    if not signals or any(not signal for signal in signals) or float(odds_used or 0.0) <= 0:
        return 0.0, 0.0, 0.0
    agreement = sum(float(signal.get("agreement", 0.0) or 0.0) for signal in signals) / float(len(signals))
    support = sum(float(signal.get("support_score", 0.0) or 0.0) for signal in signals) / float(len(signals))
    anchor_strength = sum(float(signal.get("anchor_strength", 0.0) or 0.0) for signal in signals) / float(len(signals))
    place_probs = [float(signal.get("place_prob", 0.0) or 0.0) for signal in signals]
    win_probs = [float(signal.get("win_prob", 0.0) or 0.0) for signal in signals]
    if bet_type == "wide":
        p_hit = 0.72 * place_probs[0] * place_probs[1] * (0.84 + 0.16 * agreement) * (0.90 + 0.10 * anchor_strength)
    elif bet_type == "quinella":
        p_hit = 0.42 * place_probs[0] * place_probs[1] * (0.82 + 0.18 * agreement) * (0.90 + 0.10 * anchor_strength)
    elif bet_type == "exacta":
        p_hit = 0.34 * win_probs[0] * place_probs[1] * (0.82 + 0.18 * agreement) * (0.89 + 0.10 * anchor_strength)
    elif bet_type == "trio":
        p_hit = 0.26 * place_probs[0] * place_probs[1] * place_probs[2] * (0.82 + 0.18 * agreement) * (0.89 + 0.10 * anchor_strength)
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
    quant_support_score = round(float(support or 0.0), 6)
    quant_anchor_strength = round(float(anchor_strength or 0.0), 6)
    quant_agreement = round(float(agreement or 0.0), 6)
    type_penalty = {
        "wide": 1.00,
        "quinella": 0.86,
        "exacta": 0.72,
        "trio": 0.58,
    }.get(bet_type, 1.0)
    type_bias = {
        "wide": 0.03,
        "quinella": 0.0,
        "exacta": -0.04,
        "trio": -0.08,
    }.get(bet_type, 0.0)
    low_hit_floor = {
        "wide": 0.20,
        "quinella": 0.14,
        "exacta": 0.10,
        "trio": 0.08,
    }.get(bet_type, 0.10)
    low_hit_penalty = max(0.0, low_hit_floor - float(p_hit or 0.0)) * 1.15
    score = round(
        (
            p_hit * 0.50
            + quant_support_score * 0.24
            + quant_anchor_strength * 0.14
            + quant_agreement * 0.08
            + ev_for_score * 0.04
            - low_hit_penalty
        )
        * type_penalty
        + type_bias,
        6,
    )
    return round(p_hit, 6), ev, score, quant_support_score, quant_anchor_strength, quant_agreement


def _trim_candidates(candidates):
    quotas = {
        "win": 18,
        "place": 18,
        "wide": 100,
        "quinella": 50,
        "exacta": 25,
        "trio": 6,
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
                -float(item.get("p_hit", 0.0) or 0.0),
                -float(item.get("quant_support_score", 0.0) or 0.0),
                -float(item.get("ev", 0.0) or 0.0),
                float(item.get("odds_used", 0.0) or 0.0),
                str(item.get("id", "")),
            )
        )
        selected.extend(items[: quotas.get(bet_type, 8)])
    selected.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -float(item.get("p_hit", 0.0) or 0.0),
            -float(item.get("quant_support_score", 0.0) or 0.0),
            -float(item.get("ev", 0.0) or 0.0),
            float(item.get("odds_used", 0.0) or 0.0),
            str(item.get("id", "")),
        )
    )
    return selected[:100]


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
                p_hit, ev, score, quant_support_score, quant_anchor_strength, quant_agreement = _score_single_candidate(item, signal_map, "win")
                candidate = {"id": f"win:{horse_no}", "bet_type": "win", "legs": [horse_no], "odds_used": round(win_odds, 6), "p_hit": p_hit, "ev": ev, "score": score, "quant_support_score": quant_support_score, "quant_anchor_strength": quant_anchor_strength, "quant_agreement": quant_agreement}
                candidates.append(candidate)
        if "place" in allowed_types:
            place_odds = float(item.get("place_odds", 0.0) or 0.0)
            if place_odds > 0:
                p_hit, ev, score, quant_support_score, quant_anchor_strength, quant_agreement = _score_single_candidate(item, signal_map, "place")
                candidate = {"id": f"place:{horse_no}", "bet_type": "place", "legs": [horse_no], "odds_used": round(place_odds, 6), "p_hit": p_hit, "ev": ev, "score": score, "quant_support_score": quant_support_score, "quant_anchor_strength": quant_anchor_strength, "quant_agreement": quant_agreement}
                candidates.append(candidate)
    if "wide" in allowed_types:
        for pair, odds in sorted(wide_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            legs = [str(pair[0]), str(pair[1])]
            p_hit, ev, score, quant_support_score, quant_anchor_strength, quant_agreement = _score_combo_candidate(legs, signal_map, "wide", odds)
            candidate = {"id": f"wide:{pair[0]}-{pair[1]}", "bet_type": "wide", "legs": legs, "odds_used": round(float(odds), 6), "p_hit": p_hit, "ev": ev, "score": score, "quant_support_score": quant_support_score, "quant_anchor_strength": quant_anchor_strength, "quant_agreement": quant_agreement}
            candidates.append(candidate)
    if "quinella" in allowed_types:
        for pair, odds in sorted(quinella_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            legs = [str(pair[0]), str(pair[1])]
            p_hit, ev, score, quant_support_score, quant_anchor_strength, quant_agreement = _score_combo_candidate(legs, signal_map, "quinella", odds)
            candidate = {"id": f"quinella:{pair[0]}-{pair[1]}", "bet_type": "quinella", "legs": legs, "odds_used": round(float(odds), 6), "p_hit": p_hit, "ev": ev, "score": score, "quant_support_score": quant_support_score, "quant_anchor_strength": quant_anchor_strength, "quant_agreement": quant_agreement}
            candidates.append(candidate)
    if "exacta" in allowed_types:
        for pair, odds in sorted(exacta_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            legs = [str(pair[0]), str(pair[1])]
            p_hit, ev, score, quant_support_score, quant_anchor_strength, quant_agreement = _score_combo_candidate(legs, signal_map, "exacta", odds)
            candidate = {"id": f"exacta:{pair[0]}-{pair[1]}", "bet_type": "exacta", "legs": legs, "odds_used": round(float(odds), 6), "p_hit": p_hit, "ev": ev, "score": score, "quant_support_score": quant_support_score, "quant_anchor_strength": quant_anchor_strength, "quant_agreement": quant_agreement}
            candidates.append(candidate)
    if "trio" in allowed_types:
        for trio_key, odds in sorted(trio_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            legs = [str(trio_key[0]), str(trio_key[1]), str(trio_key[2])]
            p_hit, ev, score, quant_support_score, quant_anchor_strength, quant_agreement = _score_combo_candidate(legs, signal_map, "trio", odds)
            candidate = {"id": f"trio:{trio_key[0]}-{trio_key[1]}-{trio_key[2]}", "bet_type": "trio", "legs": legs, "odds_used": round(float(odds), 6), "p_hit": p_hit, "ev": ev, "score": score, "quant_support_score": quant_support_score, "quant_anchor_strength": quant_anchor_strength, "quant_agreement": quant_agreement}
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


def _build_equalized_reference_predictions(multi_predictor, fallback_predictions):
    multi_predictor = dict(multi_predictor or {})
    ranking_blocks = list(multi_predictor.get("predictor_rankings", []) or [])
    aggregates = {}
    order = []

    for block in ranking_blocks:
        ranking = list(block.get("ranking", []) or [])
        if not ranking:
            continue
        for idx, item in enumerate(ranking, start=1):
            row = _clone_prediction_row(item)
            horse_no = str(row.get("horse_no", "") or "").strip()
            if not horse_no:
                continue
            if horse_no not in aggregates:
                aggregates[horse_no] = {
                    "horse_no": horse_no,
                    "horse_name": str(row.get("horse_name", "") or ""),
                    "top3_prob_sum": 0.0,
                    "rank_score_sum": 0.0,
                    "confidence_sum": 0.0,
                    "stability_sum": 0.0,
                    "risk_sum": 0.0,
                    "win_prob_sum": 0.0,
                    "rank_sum": 0.0,
                    "count": 0,
                    "win_odds": round(float(row.get("win_odds", 0.0) or 0.0), 6),
                    "place_odds": round(float(row.get("place_odds", 0.0) or 0.0), 6),
                    "source_row": dict(row.get("source_row", {}) or {}),
                }
                order.append(horse_no)
            agg = aggregates[horse_no]
            agg["top3_prob_sum"] += float(row.get("top3_prob_model", 0.0) or 0.0)
            agg["rank_score_sum"] += float(row.get("rank_score_norm", 0.0) or 0.0)
            agg["confidence_sum"] += float(row.get("confidence_score", 0.0) or 0.0)
            agg["stability_sum"] += float(row.get("stability_score", 0.0) or 0.0)
            agg["risk_sum"] += float(row.get("risk_score", 0.0) or 0.0)
            agg["win_prob_sum"] += float(row.get("win_prob_est", 0.0) or 0.0)
            agg["rank_sum"] += float(idx)
            agg["count"] += 1
            if not agg["horse_name"]:
                agg["horse_name"] = str(row.get("horse_name", "") or "")
            if float(agg.get("win_odds", 0.0) or 0.0) <= 0.0 and float(row.get("win_odds", 0.0) or 0.0) > 0.0:
                agg["win_odds"] = round(float(row.get("win_odds", 0.0) or 0.0), 6)
            if float(agg.get("place_odds", 0.0) or 0.0) <= 0.0 and float(row.get("place_odds", 0.0) or 0.0) > 0.0:
                agg["place_odds"] = round(float(row.get("place_odds", 0.0) or 0.0), 6)
            if not agg["source_row"] and row.get("source_row"):
                agg["source_row"] = dict(row.get("source_row", {}) or {})

    if not aggregates:
        return [_clone_prediction_row(item) for item in list(fallback_predictions or [])]

    rows = []
    for horse_no in order:
        agg = dict(aggregates.get(horse_no, {}) or {})
        count = max(1, int(agg.get("count", 0) or 0))
        rows.append(
            {
                "horse_no": horse_no,
                "horse_name": str(agg.get("horse_name", "") or ""),
                "pred_rank": 0,
                "top3_prob_model": round(float(agg.get("top3_prob_sum", 0.0) or 0.0) / float(count), 6),
                "rank_score_norm": round(float(agg.get("rank_score_sum", 0.0) or 0.0) / float(count), 6),
                "win_odds": round(float(agg.get("win_odds", 0.0) or 0.0), 6),
                "place_odds": round(float(agg.get("place_odds", 0.0) or 0.0), 6),
                "confidence_score": round(float(agg.get("confidence_sum", 0.0) or 0.0) / float(count), 6),
                "stability_score": round(float(agg.get("stability_sum", 0.0) or 0.0) / float(count), 6),
                "risk_score": round(float(agg.get("risk_sum", 0.0) or 0.0) / float(count), 6),
                "win_prob_est": round(float(agg.get("win_prob_sum", 0.0) or 0.0) / float(count), 6),
                "source_row": dict(agg.get("source_row", {}) or {}),
                "_avg_rank": round(float(agg.get("rank_sum", 0.0) or 0.0) / float(count), 6),
            }
        )

    rows.sort(
        key=lambda row: (
            float(row.get("_avg_rank", 999.0) or 999.0),
            -float(row.get("top3_prob_model", 0.0) or 0.0),
            -float(row.get("rank_score_norm", 0.0) or 0.0),
            str(row.get("horse_no", "") or ""),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["pred_rank"] = idx
        row.pop("_avg_rank", None)
    return rows


def _build_consensus_primary_predictions(multi_predictor, reference_predictions):
    multi_predictor = dict(multi_predictor or {})
    consensus_rows = list(multi_predictor.get("consensus", []) or [])
    if not consensus_rows:
        return [_clone_prediction_row(item) for item in list(reference_predictions or [])]

    reference_items = [_clone_prediction_row(item) for item in list(reference_predictions or [])]
    reference_map = {}
    for idx, item in enumerate(reference_items, start=1):
        horse_no = str(item.get("horse_no", "") or "").strip()
        if horse_no and horse_no not in reference_map:
            row = dict(item)
            row["_reference_rank"] = idx
            reference_map[horse_no] = row

    consensus_map = _build_consensus_map(consensus_rows)
    merged = []
    seen = set()
    predictor_count_cap = max(
        1.0,
        max(float(row.get("predictor_count", 0) or 0) for row in consensus_rows) if consensus_rows else 1.0,
    )

    for item in reference_items:
        horse_no = str(item.get("horse_no", "") or "").strip()
        if not horse_no or horse_no in seen:
            continue
        seen.add(horse_no)
        consensus = dict(consensus_map.get(horse_no, {}) or {})
        reference_rank = float(reference_map.get(horse_no, {}).get("_reference_rank", len(reference_items) + 1))
        consensus_rank = float(consensus.get("avg_pred_rank", reference_rank) or reference_rank)
        top1_ratio = float(consensus.get("top1_votes", 0) or 0) / predictor_count_cap
        top3_ratio = float(consensus.get("top3_votes", 0) or 0) / predictor_count_cap
        blended_rank = 0.55 * reference_rank + 0.45 * consensus_rank - 0.28 * top1_ratio - 0.12 * top3_ratio
        merged.append(
            {
                "horse_no": horse_no,
                "horse_name": str(consensus.get("horse_name", "") or item.get("horse_name", "") or ""),
                "pred_rank": int(item.get("pred_rank", 0) or 0),
                "top3_prob_model": round(0.65 * float(item.get("top3_prob_model", 0.0) or 0.0) + 0.35 * float(consensus.get("avg_top3_prob_model", item.get("top3_prob_model", 0.0)) or 0.0), 6),
                "rank_score_norm": round(0.65 * float(item.get("rank_score_norm", 0.0) or 0.0) + 0.35 * float(consensus.get("avg_rank_score_norm", item.get("rank_score_norm", 0.0)) or 0.0), 6),
                "win_odds": round(float(consensus.get("win_odds", item.get("win_odds", 0.0)) or 0.0), 6),
                "place_odds": round(float(consensus.get("place_odds", item.get("place_odds", 0.0)) or 0.0), 6),
                "confidence_score": round(float(item.get("confidence_score", 0.5) or 0.5), 6),
                "stability_score": round(float(item.get("stability_score", 0.5) or 0.5), 6),
                "risk_score": round(float(item.get("risk_score", 0.5) or 0.5), 6),
                "win_prob_est": round(float(item.get("win_prob_est", 0.0) or 0.0), 6),
                "source_row": dict(item.get("source_row", {}) or {}),
                "_blended_rank": blended_rank,
            }
        )

    merged.sort(
        key=lambda row: (
            float(row.get("_blended_rank", 999.0) or 999.0),
            -float(row.get("top3_prob_model", 0.0) or 0.0),
            -float(row.get("rank_score_norm", 0.0) or 0.0),
            str(row.get("horse_no", "") or ""),
        )
    )
    for idx, row in enumerate(merged, start=1):
        row["pred_rank"] = idx
        row.pop("_blended_rank", None)
    return merged


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
    for predictor_id in ("v6_kiwami", "v5_stacking", "v4_gemini", "v3_premium", "v2_opus", "main"):
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
        "v6_kiwami": "predictions_v6_kiwami_path",
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


def _ledger_date_to_job_date_text(ledger_date=""):
    digits = "".join(ch for ch in str(ledger_date or "").strip() if ch.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _job_date_text(job):
    race_date = str((job or {}).get("race_date", "") or "").strip()
    if race_date:
        return race_date[:10]
    scheduled_off_time = str((job or {}).get("scheduled_off_time", "") or "").strip()
    if len(scheduled_off_time) >= 10:
        return scheduled_off_time[:10]
    return ""


def _daily_budget_plan_context(base_dir, ledger_date, run_row, load_race_jobs):
    target_date = _ledger_date_to_job_date_text(ledger_date)
    if not target_date:
        return {}
    jobs = [dict(item or {}) for item in list(load_race_jobs(base_dir) or []) if isinstance(item, dict)]
    if not jobs:
        return {
            "budget_scope_note": "このサイトでは終日の回収よりも、このレースで当てに行ける形を優先します。資金上限だけ守ってください。",
        }

    active_status_exclusions = {"settled", "failed", "deleted"}
    planned_jobs = []
    active_jobs = []
    for job in jobs:
        if _job_date_text(job) != target_date:
            continue
        status = str(job.get("status", "") or "").strip().lower()
        if status in {"failed", "deleted"}:
            continue
        planned_jobs.append(job)
        if status not in active_status_exclusions:
            active_jobs.append(job)

    if not planned_jobs:
        return {
            "budget_scope_note": "このサイトでは終日の回収よりも、このレースで当てに行ける形を優先します。資金上限だけ守ってください。",
        }

    def _job_sort_key(job):
        return (
            str(job.get("scheduled_off_time", "") or ""),
            str(job.get("race_date", "") or ""),
            str(job.get("race_id", "") or ""),
            str(job.get("created_at", "") or ""),
        )

    planned_jobs.sort(key=_job_sort_key)
    active_jobs.sort(key=_job_sort_key)

    current_race_id = str((run_row or {}).get("race_id", "") or "").strip()
    current_run_id = str((run_row or {}).get("run_id", "") or "").strip()
    current_index = -1
    for idx, job in enumerate(active_jobs):
        job_race_id = str(job.get("race_id", "") or "").strip()
        job_run_id = str(job.get("run_id", "") or "").strip()
        if current_race_id and job_race_id == current_race_id:
            current_index = idx
            break
        if current_run_id and job_run_id and job_run_id == current_run_id:
            current_index = idx
            break

    planned_count = len(planned_jobs)
    active_count = len(active_jobs)
    remaining_count = active_count if current_index < 0 else max(1, active_count - current_index)
    sequence_for_day = 0
    for idx, job in enumerate(planned_jobs, start=1):
        if current_race_id and str(job.get("race_id", "") or "").strip() == current_race_id:
            sequence_for_day = idx
            break
        if current_run_id and str(job.get("run_id", "") or "").strip() == current_run_id:
            sequence_for_day = idx
            break

    return {
        "planned_races_for_day": planned_count,
        "remaining_races_for_day": remaining_count,
        "active_races_for_day": active_count,
        "race_sequence_for_day": sequence_for_day,
        "budget_scope_note": "このサイトでは終日の回収よりも、このレースで当てに行ける形を優先します。資金上限だけ守ってください。",
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
    load_race_jobs,
    base_dir,
    build_multi_predictor_context,
    build_history_context,
    to_float,
):
    core_bet_types = ("win", "place", "wide", "quinella", "exacta", "trio")
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
    equalized_predictions = _build_equalized_reference_predictions(multi_predictor, fallback_predictions)
    reference_predictions = equalized_predictions or [_clone_prediction_row(item) for item in list(fallback_predictions or [])]
    predictions = [_clone_prediction_row(item) for item in list(reference_predictions or [])]
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
    allowed_types = [bet_type for bet_type in allowed_types if bet_type in core_bet_types]
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
    multi_predictor_meta.pop("primary_predictor_id", None)
    multi_predictor_meta.pop("compatibility_primary_predictor_id", None)
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
    race_context.update(_daily_budget_plan_context(base_dir, ledger_date, run_row, load_race_jobs))
    multi_model_ai = _build_multi_model_ai(predictions, multi_predictor)
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

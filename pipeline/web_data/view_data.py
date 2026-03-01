import re
from pathlib import Path


def pick_score_key(rows):
    if not rows:
        return ""
    sample = rows[0]
    for key in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if key in sample:
            return key
    return ""


def _normalize_name(value):
    return "".join(str(value or "").split())


def _resolve_predictions_path(get_data_dir, base_dir, scope_key, run_id, run_row=None):
    path = ""
    if run_row:
        path = run_row.get("predictions_path", "")
    if not path:
        path = str(get_data_dir(base_dir, scope_key) / f"predictions_{run_id}.csv")
    return Path(path)


def _resolve_plan_path(get_data_dir, base_dir, scope_key, run_id, run_row=None):
    path = ""
    if run_row:
        path = run_row.get("plan_path", "")
    if not path:
        path = str(get_data_dir(base_dir, scope_key) / f"bet_plan_{run_id}.csv")
    return Path(path)


def _resolve_odds_path(get_data_dir, base_dir, scope_key, run_id, run_row=None):
    path = ""
    if run_row:
        path = run_row.get("odds_path", "")
    if not path:
        path = str(get_data_dir(base_dir, scope_key) / f"odds_{run_id}.csv")
    return Path(path)


def _build_name_to_horse_no_map(rows):
    out = {}
    for row in rows:
        name = row.get("name") or row.get("HorseName") or row.get("horse_name")
        horse_no = row.get("horse_no") or row.get("horse") or row.get("\u99ac\u756a")
        norm = _normalize_name(name)
        no_text = str(horse_no or "").strip()
        if not norm or not no_text:
            continue
        if norm not in out:
            out[norm] = no_text
    return out


def load_top5_table(get_data_dir, base_dir, load_csv_rows, to_float, scope_key, run_id, run_row=None):
    rows = load_csv_rows(_resolve_predictions_path(get_data_dir, base_dir, scope_key, run_id, run_row))
    if not rows:
        return [], []
    score_key = pick_score_key(rows)
    if score_key:
        rows_sorted = sorted(rows, key=lambda r: to_float(r.get(score_key)), reverse=True)
    else:
        rows_sorted = rows
    top_rows = rows_sorted[:5]
    preferred_cols = [
        "HorseName",
        "Top3Prob_model",
        "Top3Prob_lgbm",
        "Top3Prob_lr",
        "agg_score",
        "best_TimeIndexEff",
        "avg_TimeIndexEff",
        "dist_close",
    ]
    if score_key and score_key not in preferred_cols:
        preferred_cols.append(score_key)
    global_cols = {"confidence_score", "rank_ema", "ev_ema", "risk_score"}
    columns = [col for col in preferred_cols if col in rows[0]]
    if not columns:
        columns = [col for col in rows[0].keys() if col not in global_cols][:6]
    else:
        columns = columns[:6]
    return top_rows, columns


def _parse_plan_horse_names(value):
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*/\s*|\uFF0F", text)
    out = []
    seen = set()
    for part in parts:
        name = part.strip()
        if not name:
            continue
        norm = _normalize_name(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(name)
    return out


def _parse_plan_horse_nos(value):
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*-\s*|\uFF0D|\u30FC|\u2015|\u2212|\uFF5E|~|/|\uFF0F", text)
    out = []
    for part in parts:
        token = str(part or "").strip()
        if re.fullmatch(r"\d+", token):
            out.append(token)
    return out


def _support_strength_label(amount_value, max_amount):
    if max_amount <= 0 or amount_value <= 0:
        return "-"
    ratio = float(amount_value) / float(max_amount)
    if ratio >= 0.67:
        return "\u9ad8"
    if ratio >= 0.34:
        return "\u4e2d"
    return "\u4f4e"


def _risk_signal_label(gate_status, has_bet_types, pred_rank, support_strength):
    status = str(gate_status or "").strip().lower()
    if status == "hard_fail":
        return "\u898b\u9001\u308a\u7d1a"
    if status == "soft_fail":
        return "\u6ce8\u610f"
    if not has_bet_types:
        return "\u6ce8\u610f"
    try:
        rank_val = int(pred_rank)
    except (TypeError, ValueError):
        rank_val = 999
    if rank_val >= 6 and support_strength in ("-", "\u4f4e"):
        return "\u6ce8\u610f"
    return "\u901a\u5e38"


def load_mark_recommendation_table(get_data_dir, base_dir, load_csv_rows, to_float, scope_key, run_id, run_row=None):
    pred_rows = load_csv_rows(_resolve_predictions_path(get_data_dir, base_dir, scope_key, run_id, run_row))
    if not pred_rows:
        return [], []
    odds_rows = load_csv_rows(_resolve_odds_path(get_data_dir, base_dir, scope_key, run_id, run_row))
    odds_name_to_no = _build_name_to_horse_no_map(odds_rows)

    score_key = pick_score_key(pred_rows)
    if score_key:
        pred_sorted = sorted(pred_rows, key=lambda r: to_float(r.get(score_key)), reverse=True)
    else:
        pred_sorted = list(pred_rows)

    pred_map = {}
    pred_order = []
    for idx, row in enumerate(pred_sorted):
        name = row.get("HorseName") or row.get("name") or row.get("horse_name")
        norm = _normalize_name(name)
        if not norm or norm in pred_map:
            continue
        score_val = to_float(row.get(score_key)) if score_key else float(max(len(pred_sorted) - idx, 1))
        pred_map[norm] = {
            "horse_name": str(name).strip(),
            "pred_rank": idx + 1,
            "pred_prob": score_val,
            "horse_no": str(row.get("horse_no", "")).strip(),
        }
        pred_order.append(norm)

    plan_rows = load_csv_rows(_resolve_plan_path(get_data_dir, base_dir, scope_key, run_id, run_row))
    gate_status = ""
    bet_map = {}
    for row in plan_rows:
        if not gate_status:
            status_text = str(row.get("gate_status", "")).strip().lower()
            if status_text:
                gate_status = status_text
        bet_type = str(row.get("bet_type", "")).strip().lower()
        if bet_type in ("", "trifecta_rec", "pass"):
            continue
        amount = to_float(row.get("amount_yen"))
        expected = to_float(row.get("expected_return_yen"))
        hit_prob = to_float(row.get("hit_prob_est"))
        ev_ratio = to_float(row.get("ev_ratio_est"))
        if amount <= 0 and expected <= 0 and hit_prob <= 0 and ev_ratio <= 0:
            continue
        horse_names = _parse_plan_horse_names(row.get("horse_name", ""))
        if not horse_names:
            continue
        share = 1.0 / float(len(horse_names))
        for horse_name in horse_names:
            norm = _normalize_name(horse_name)
            item = bet_map.setdefault(
                norm,
                {
                    "horse_name": horse_name,
                    "horse_no": "",
                    "bet_types": set(),
                    "ticket_count": 0,
                    "amount_max": 0.0,
                    "expected_max": 0.0,
                    "hit_prob_max": 0.0,
                    "ev_ratio_max": 0.0,
                },
            )
            item["bet_types"].add(bet_type)
            item["ticket_count"] += 1
            item["amount_max"] = max(item["amount_max"], amount * share)
            item["expected_max"] = max(item["expected_max"], expected * share)
            item["hit_prob_max"] = max(item["hit_prob_max"], hit_prob)
            item["ev_ratio_max"] = max(item["ev_ratio_max"], ev_ratio)
        horse_nos = _parse_plan_horse_nos(row.get("horse_no", ""))
        if len(horse_nos) == len(horse_names):
            for idx, horse_name in enumerate(horse_names):
                norm = _normalize_name(horse_name)
                if norm in bet_map and not bet_map[norm].get("horse_no"):
                    bet_map[norm]["horse_no"] = horse_nos[idx]
        elif len(horse_names) == 1 and horse_nos:
            norm = _normalize_name(horse_names[0])
            if norm in bet_map and not bet_map[norm].get("horse_no"):
                bet_map[norm]["horse_no"] = horse_nos[0]

    candidate_keys = []
    for key in pred_order[:8]:
        if key not in candidate_keys:
            candidate_keys.append(key)
    for key in bet_map.keys():
        if key not in candidate_keys:
            candidate_keys.append(key)
    if not candidate_keys:
        return [], []

    max_pred_prob = max((pred_map.get(k, {}).get("pred_prob", 0.0) for k in candidate_keys), default=0.0)
    max_amount = max((bet_map.get(k, {}).get("amount_max", 0.0) for k in candidate_keys), default=0.0)
    max_hit = max((bet_map.get(k, {}).get("hit_prob_max", 0.0) for k in candidate_keys), default=0.0)
    max_ev = max((bet_map.get(k, {}).get("ev_ratio_max", 0.0) for k in candidate_keys), default=0.0)
    max_ticket = max((bet_map.get(k, {}).get("ticket_count", 0) for k in candidate_keys), default=0)
    pred_total = max(1, len(pred_map))

    scored = []
    for key in candidate_keys:
        pred_item = pred_map.get(key, {})
        bet_item = bet_map.get(key, {})

        pred_rank = pred_item.get("pred_rank")
        pred_prob = float(pred_item.get("pred_prob", 0.0) or 0.0)
        pred_prob_norm = (pred_prob / max_pred_prob) if max_pred_prob > 0 else 0.0
        rank_norm = 0.0
        if pred_rank:
            rank_norm = (pred_total - int(pred_rank) + 1) / float(pred_total)
            if rank_norm < 0:
                rank_norm = 0.0
        pred_score = (0.75 * pred_prob_norm + 0.25 * rank_norm) if pred_rank else 0.0

        amount_max = float(bet_item.get("amount_max", 0.0) or 0.0)
        amount_norm = (amount_max / max_amount) if max_amount > 0 else 0.0
        hit_norm = (float(bet_item.get("hit_prob_max", 0.0) or 0.0) / max_hit) if max_hit > 0 else 0.0
        ev_norm = (float(bet_item.get("ev_ratio_max", 0.0) or 0.0) / max_ev) if max_ev > 0 else 0.0
        ticket_norm = (float(bet_item.get("ticket_count", 0) or 0) / float(max_ticket)) if max_ticket > 0 else 0.0
        bet_score = (0.35 * amount_norm + 0.30 * hit_norm + 0.20 * ev_norm + 0.15 * ticket_norm) if bet_item else 0.0

        if pred_rank and bet_item:
            combined = 0.70 * pred_score + 0.30 * bet_score
        elif pred_rank:
            combined = pred_score
        else:
            combined = 0.60 * bet_score

        horse_name = pred_item.get("horse_name") or bet_item.get("horse_name") or key
        horse_no = str(pred_item.get("horse_no", "")).strip() or str(bet_item.get("horse_no", "")).strip()
        if not horse_no:
            horse_no = odds_name_to_no.get(key, "")
        bet_types = ",".join(sorted(bet_item.get("bet_types", set())))
        support_strength = _support_strength_label(amount_max, max_amount)
        risk_signal = _risk_signal_label(gate_status, bool(bet_types), pred_rank, support_strength)

        if pred_rank and bet_types:
            reason = "model+bet"
        elif pred_rank and int(pred_rank) <= 3:
            reason = "model_top"
        elif bet_types:
            reason = "bet_support"
        else:
            reason = "model_candidate"

        scored.append(
            {
                "horse_name": horse_name,
                "horse_no": horse_no,
                "pred_rank": pred_rank if pred_rank else "",
                "top3_prob": f"{pred_prob:.4f}" if pred_rank else "",
                "pred_score": pred_score,
                "bet_score": bet_score,
                "combined_score": combined,
                "bet_types": bet_types,
                "support_strength": support_strength,
                "risk_signal": risk_signal,
                "reason": reason,
                "_pred_rank_sort": int(pred_rank) if pred_rank else 999,
            }
        )

    scored = sorted(
        scored,
        key=lambda r: (-r["combined_score"], r["_pred_rank_sort"], -r["bet_score"], str(r["horse_name"])),
    )
    selected = scored[:5]
    marks = ["\u25CE", "\u25CB", "\u25B2", "\u25B3", "\u2606"]
    out_rows = []
    for idx, row in enumerate(selected):
        out_rows.append(
            {
                "mark": marks[idx],
                "horse_no": row["horse_no"],
                "horse_name": row["horse_name"],
                "pred_rank": row["pred_rank"],
                "bet_types": row["bet_types"] or "-",
            }
        )
    columns = [
        "mark",
        "horse_no",
        "horse_name",
        "pred_rank",
        "bet_types",
    ]
    return out_rows, columns


def load_mc_uncertainty_summary(get_data_dir, base_dir, load_csv_rows, to_float, scope_key, run_id, run_row=None):
    rows = load_csv_rows(_resolve_plan_path(get_data_dir, base_dir, scope_key, run_id, run_row))
    if not rows:
        return []
    filtered = [
        row
        for row in rows
        if str(row.get("bet_type", "")).strip().lower() != "trifecta_rec"
    ]
    filtered = [
        row
        for row in filtered
        if row.get("hit_prob_se") or row.get("hit_prob_ci95_low") or row.get("hit_prob_ci95_high")
    ]
    if not filtered:
        return []
    target = max(filtered, key=lambda r: to_float(r.get("hit_prob_est")))
    budget = str(target.get("budget_yen", "")).strip()
    bet_type = str(target.get("bet_type", "")).strip()
    horse_no = str(target.get("horse_no", "")).strip()
    label_parts = [f"[{budget}]" if budget else "", bet_type, horse_no]
    label = " ".join(part for part in label_parts if part).strip()
    return [
        {"metric": f"MC SE ({label})", "value": target.get("hit_prob_se", "")},
        {"metric": f"MC CI95 Low ({label})", "value": target.get("hit_prob_ci95_low", "")},
        {"metric": f"MC CI95 High ({label})", "value": target.get("hit_prob_ci95_high", "")},
    ]


def load_prediction_summary(
    get_data_dir,
    base_dir,
    load_csv_rows,
    load_mc_uncertainty_summary_func,
    scope_key,
    run_id,
    run_row=None,
):
    rows = load_csv_rows(_resolve_predictions_path(get_data_dir, base_dir, scope_key, run_id, run_row))
    if not rows:
        return []
    row = rows[0]
    summary_keys = ["confidence_score", "rank_ema", "ev_ema", "risk_score"]
    summary = []
    for key in summary_keys:
        if key in row:
            summary.append({"metric": key, "value": row.get(key, "")})
    summary.extend(load_mc_uncertainty_summary_func(scope_key, run_id, run_row))
    return summary


def load_bet_plan_table(get_data_dir, base_dir, load_csv_rows, scope_key, run_id, run_row=None):
    rows = load_csv_rows(_resolve_plan_path(get_data_dir, base_dir, scope_key, run_id, run_row))
    if not rows:
        return [], []
    columns = [
        "budget_yen",
        "bet_type",
        "horse_no",
        "horse_name",
        "amount_yen",
        "expected_return_yen",
        "hit_prob_est",
        "hit_prob_se",
        "hit_prob_ci95_low",
        "hit_prob_ci95_high",
        "units",
        "gate_status",
        "risk_note",
        "gate_reason",
    ]
    columns = [col for col in columns if col in rows[0]] or list(rows[0].keys())
    return rows, columns

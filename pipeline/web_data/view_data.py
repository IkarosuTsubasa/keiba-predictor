from pathlib import Path


def pick_score_key(rows):
    if not rows:
        return ""
    sample = rows[0]
    for key in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if key in sample:
            return key
    return ""


def load_top5_table(get_data_dir, base_dir, load_csv_rows, to_float, scope_key, run_id, run_row=None):
    path = ""
    if run_row:
        path = run_row.get("predictions_path", "")
    if not path:
        path = str(get_data_dir(base_dir, scope_key) / f"predictions_{run_id}.csv")
    path = Path(path)
    rows = load_csv_rows(path)
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


def load_mc_uncertainty_summary(get_data_dir, base_dir, load_csv_rows, to_float, scope_key, run_id, run_row=None):
    path = ""
    if run_row:
        path = run_row.get("plan_path", "")
    if not path:
        path = str(get_data_dir(base_dir, scope_key) / f"bet_plan_{run_id}.csv")
    rows = load_csv_rows(Path(path))
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
    path = ""
    if run_row:
        path = run_row.get("predictions_path", "")
    if not path:
        path = str(get_data_dir(base_dir, scope_key) / f"predictions_{run_id}.csv")
    path = Path(path)
    rows = load_csv_rows(path)
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
    path = ""
    if run_row:
        path = run_row.get("plan_path", "")
    if not path:
        path = str(get_data_dir(base_dir, scope_key) / f"bet_plan_{run_id}.csv")
    path = Path(path)
    rows = load_csv_rows(path)
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

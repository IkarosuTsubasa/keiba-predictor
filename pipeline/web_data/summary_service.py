from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path


def extract_date_prefix(value):
    text = str(value or "")
    if len(text) < 8 or not text[:8].isdigit():
        return None
    raw = text[:8]
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def extract_run_date(run_id):
    return extract_date_prefix(run_id)


def extract_year_prefix(value):
    text = str(value or "").strip()
    if len(text) < 4 or not text[:4].isdigit():
        return None
    try:
        return int(text[:4])
    except ValueError:
        return None


def extract_iso_date_prefix(value):
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    raw = text[:10]
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def build_run_record_date_map(load_csv_rows, path):
    rows = load_csv_rows(path)
    out = {}
    for row in rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        date_obj = extract_iso_date_prefix(row.get("timestamp", ""))
        if date_obj:
            out[run_id] = date_obj
    return out


def resolve_record_date(run_id, run_record_date_map=None, timestamp=""):
    if run_record_date_map:
        date_obj = run_record_date_map.get(run_id)
        if date_obj:
            return date_obj
    date_obj = extract_iso_date_prefix(timestamp)
    if date_obj:
        return date_obj
    return extract_run_date(run_id)


def build_run_race_map(load_runs, infer_run_id_from_row, scope_key):
    runs = load_runs(scope_key)
    out = {}
    for row in runs:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            run_id = infer_run_id_from_row(row)
        if not run_id:
            continue
        out[run_id] = row.get("race_id", "")
    return out


def load_profit_summary(get_data_dir, base_dir, load_csv_rows, scope_key):
    path = get_data_dir(base_dir, scope_key) / "results.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    latest = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id:
            latest[run_id] = row
    rows = list(latest.values())
    total_profit = 0
    total_base = 0
    sample_count = 0
    for row in rows:
        try:
            profit = int(float(row.get("profit_yen", 0)))
        except (TypeError, ValueError):
            profit = 0
        try:
            base = int(float(row.get("base_amount", 0)))
        except (TypeError, ValueError):
            base = 0
        total_profit += profit
        total_base += base
        sample_count += 1
    roi = ""
    if total_base > 0:
        roi = round((total_base + total_profit) / total_base, 4)
    return [
        {"metric": "runs", "value": sample_count},
        {"metric": "total_stake_yen", "value": total_base},
        {"metric": "total_profit_yen", "value": total_profit},
        {"metric": "overall_roi", "value": roi},
    ]


def load_daily_profit_summary(
    get_data_dir,
    base_dir,
    load_csv_rows,
    build_run_race_map_func,
    min_race_year,
    scope_key,
    days=30,
):
    path = get_data_dir(base_dir, scope_key) / "results.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    latest = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id:
            latest[run_id] = row
    rows = list(latest.values())
    run_race_map = build_run_race_map_func(scope_key)
    run_record_date_map = build_run_record_date_map(
        load_csv_rows,
        get_data_dir(base_dir, scope_key) / "race_results.csv",
    )
    cutoff = None
    if days is not None:
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 30
        if days > 0:
            cutoff = datetime.now().date() - timedelta(days=days - 1)
    daily = {}
    for row in rows:
        run_id = row.get("run_id", "")
        race_id = run_race_map.get(run_id, "")
        race_year = extract_year_prefix(race_id)
        if race_year is not None and race_year < min_race_year:
            continue
        date_obj = resolve_record_date(run_id, run_record_date_map=run_record_date_map)
        if not date_obj:
            continue
        if cutoff and date_obj < cutoff:
            continue
        try:
            profit = int(float(row.get("profit_yen", 0)))
        except (TypeError, ValueError):
            profit = 0
        try:
            base = int(float(row.get("base_amount", 0)))
        except (TypeError, ValueError):
            base = 0
        item = daily.setdefault(date_obj, {"runs": 0, "profit": 0, "base": 0})
        item["runs"] += 1
        item["profit"] += profit
        item["base"] += base
    if not daily:
        return []
    items = sorted(daily.items(), key=lambda pair: pair[0], reverse=True)
    out = []
    for date_obj, item in items:
        base = item["base"]
        roi = round((base + item["profit"]) / base, 4) if base > 0 else ""
        out.append(
            {
                "date": date_obj.strftime("%Y-%m-%d"),
                "runs": item["runs"],
                "profit_yen": item["profit"],
                "base_amount": base,
                "roi": roi,
            }
        )
    return out


def load_daily_profit_summary_all_scopes(load_daily_profit_summary_func, to_int_or_none, days=30):
    daily = {}
    for scope_key in ("central_dirt", "central_turf", "local"):
        rows = load_daily_profit_summary_func(scope_key, days=days)
        for row in rows:
            date_key = str(row.get("date", "")).strip()
            if not date_key:
                continue
            item = daily.setdefault(
                date_key,
                {"date": date_key, "runs": 0, "profit_yen": 0, "base_amount": 0},
            )
            item["runs"] += to_int_or_none(row.get("runs")) or 0
            item["profit_yen"] += to_int_or_none(row.get("profit_yen")) or 0
            item["base_amount"] += to_int_or_none(row.get("base_amount")) or 0
    if not daily:
        return []
    out = []
    for date_key in sorted(daily.keys(), reverse=True):
        item = daily[date_key]
        base = item["base_amount"]
        profit = item["profit_yen"]
        roi = round((base + profit) / base, 4) if base > 0 else ""
        out.append(
            {
                "date": date_key,
                "runs": item["runs"],
                "profit_yen": profit,
                "base_amount": base,
                "roi": roi,
            }
        )
    return out


def normalize_name(value):
    return "".join(str(value or "").split())


def pick_score_key(rows):
    if not rows:
        return ""
    sample = rows[0]
    for key in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if key in sample:
            return key
    return ""


def load_top5_names(load_csv_rows, to_float, path):
    rows = load_csv_rows(path)
    if not rows:
        return []
    score_key = pick_score_key(rows)
    if score_key:
        rows = sorted(rows, key=lambda r: to_float(r.get(score_key)), reverse=True)
    names = []
    seen = set()
    for row in rows:
        name = row.get("HorseName") or row.get("name")
        if not name:
            continue
        norm = normalize_name(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        names.append(name)
        if len(names) >= 5:
            break
    return names


def load_odds_name_to_no(load_csv_rows, path):
    rows = load_csv_rows(path)
    if not rows:
        return {}
    out = {}
    for row in rows:
        name = row.get("name") or row.get("HorseName") or row.get("horse_name")
        horse_no = row.get("horse_no") or row.get("horse")
        if not name or horse_no is None:
            continue
        try:
            horse_no = int(float(horse_no))
        except (TypeError, ValueError):
            continue
        out[normalize_name(name)] = horse_no
    return out


def load_wide_odds_map(load_csv_rows, path):
    rows = load_csv_rows(path)
    if not rows:
        return {}
    odds_key = "odds_mid" if "odds_mid" in rows[0] else "odds"
    out = {}
    for row in rows:
        a = row.get("horse_no_a") or row.get("horse_a")
        b = row.get("horse_no_b") or row.get("horse_b")
        if a is None or b is None:
            continue
        try:
            a_i = int(float(a))
            b_i = int(float(b))
        except (TypeError, ValueError):
            continue
        try:
            odds = float(row.get(odds_key, 0))
        except (TypeError, ValueError):
            odds = 0.0
        if odds <= 0:
            continue
        if a_i > b_i:
            a_i, b_i = b_i, a_i
        out[(a_i, b_i)] = odds
    return out


def load_race_results(get_data_dir, base_dir, load_csv_rows, scope_key):
    path = get_data_dir(base_dir, scope_key) / "race_results.csv"
    rows = load_csv_rows(path)
    if not rows:
        return {}
    return {row.get("run_id", ""): row for row in rows if row.get("run_id")}


def compute_wide_box_profit(
    scope_key,
    run_row,
    race_row,
    budget_yen,
    resolve_pred_path,
    resolve_odds_path,
    resolve_wide_odds_path,
    load_top5_names_func,
    load_odds_name_to_no_func,
    load_wide_odds_map_func,
):
    run_id = run_row.get("run_id", "")
    if not run_id:
        return None
    pred_path = resolve_pred_path(scope_key, run_id, run_row)
    if not pred_path.exists():
        return None
    top5_names = load_top5_names_func(pred_path)
    if len(top5_names) < 5:
        return None
    odds_path = resolve_odds_path(scope_key, run_id, run_row)
    if not odds_path or not odds_path.exists():
        return None
    name_to_no = load_odds_name_to_no_func(odds_path)
    if not name_to_no:
        return None
    wide_path = resolve_wide_odds_path(scope_key, run_id, run_row)
    if not wide_path or not wide_path.exists():
        return None
    wide_map = load_wide_odds_map_func(wide_path)
    if not wide_map:
        return None

    actual_names = [race_row.get("actual_top1"), race_row.get("actual_top2"), race_row.get("actual_top3")]
    actual_norm = [normalize_name(n) for n in actual_names if n]
    if len(actual_norm) < 3:
        return None
    try:
        actual_nos = {name_to_no[n] for n in actual_norm}
    except KeyError:
        return None
    if len(actual_nos) < 2:
        return None

    pred_norm = [normalize_name(n) for n in top5_names]
    try:
        pred_nos = [name_to_no[n] for n in pred_norm]
    except KeyError:
        return None
    combos = list(combinations(pred_nos, 2))
    if not combos:
        return None
    per_ticket = int(budget_yen // len(combos))
    if per_ticket <= 0:
        return None
    total_amount = per_ticket * len(combos)
    total_payout = 0.0
    for a, b in combos:
        if a not in actual_nos or b not in actual_nos:
            continue
        key = (a, b) if a <= b else (b, a)
        odds = wide_map.get(key, 0.0)
        if odds:
            total_payout += per_ticket * odds
    profit = int(round(total_payout - total_amount))
    return {"amount": total_amount, "profit": profit}


def _aggregate_daily_profit(items):
    out = []
    for date_obj, item in sorted(items.items(), key=lambda pair: pair[0], reverse=True):
        base = item["base"]
        roi = round((base + item["profit"]) / base, 4) if base > 0 else ""
        out.append(
            {
                "date": date_obj.strftime("%Y-%m-%d"),
                "runs": item["runs"],
                "profit_yen": item["profit"],
                "base_amount": base,
                "roi": roi,
            }
        )
    return out


def load_wide_box_daily_profit_summary(
    get_data_dir,
    base_dir,
    load_csv_rows,
    load_runs,
    build_run_race_map_func,
    load_race_results_func,
    compute_wide_box_profit_func,
    min_race_year,
    scope_key,
    days=30,
    budget_yen=1000,
):
    rows = load_csv_rows(get_data_dir(base_dir, scope_key) / "wide_box_results.csv")
    run_race_map = build_run_race_map_func(scope_key)
    cutoff = None
    if days is not None:
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 30
        if days > 0:
            cutoff = datetime.now().date() - timedelta(days=days - 1)

    if rows:
        daily = {}
        for row in rows:
            run_id = row.get("run_id", "")
            race_id = run_race_map.get(run_id, "")
            race_year = extract_year_prefix(race_id)
            if race_year is not None and race_year < min_race_year:
                continue
            date_obj = resolve_record_date(run_id, timestamp=row.get("timestamp", ""))
            if not date_obj:
                continue
            if cutoff and date_obj < cutoff:
                continue
            try:
                profit = int(float(row.get("profit_yen", 0)))
            except (TypeError, ValueError):
                profit = 0
            try:
                base = int(float(row.get("amount_yen", 0)))
            except (TypeError, ValueError):
                base = 0
            item = daily.setdefault(date_obj, {"runs": 0, "profit": 0, "base": 0})
            item["runs"] += 1
            item["profit"] += profit
            item["base"] += base
        return _aggregate_daily_profit(daily) if daily else []

    runs = load_runs(scope_key)
    if not runs:
        return []
    results = load_race_results_func(scope_key)
    daily = {}
    for run in runs:
        run_id = run.get("run_id", "")
        race_id = run.get("race_id", "")
        race_year = extract_year_prefix(race_id)
        if race_year is not None and race_year < min_race_year:
            continue
        race_row = results.get(run_id)
        if not race_row:
            continue
        date_obj = resolve_record_date(run_id, timestamp=race_row.get("timestamp", ""))
        if not date_obj:
            continue
        if cutoff and date_obj < cutoff:
            continue
        info = compute_wide_box_profit_func(scope_key, run, race_row, budget_yen=budget_yen)
        if not info:
            continue
        item = daily.setdefault(date_obj, {"runs": 0, "profit": 0, "base": 0})
        item["runs"] += 1
        item["profit"] += info["profit"]
        item["base"] += info["amount"]
    return _aggregate_daily_profit(daily) if daily else []


def load_bet_type_summary(get_data_dir, base_dir, load_csv_rows, scope_key):
    path = get_data_dir(base_dir, scope_key) / "bet_type_stats.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    stats = {}
    for row in rows:
        bet_type = str(row.get("bet_type", "")).strip() or "unknown"
        bets = int(float(row.get("bets", 0) or 0))
        hits = int(float(row.get("hits", 0) or 0))
        amount = int(float(row.get("amount_yen", 0) or 0))
        est_profit = int(float(row.get("est_profit_yen", 0) or 0))
        item = stats.setdefault(bet_type, {"bets": 0, "hits": 0, "amount": 0, "est_profit": 0})
        item["bets"] += bets
        item["hits"] += hits
        item["amount"] += amount
        item["est_profit"] += est_profit
    out = []
    for bet_type, item in sorted(stats.items()):
        hit_rate = round(item["hits"] / item["bets"], 4) if item["bets"] else ""
        out.append(
            {
                "bet_type": bet_type,
                "bets": item["bets"],
                "hits": item["hits"],
                "hit_rate": hit_rate,
                "amount_yen": item["amount"],
                "est_profit_yen": item["est_profit"],
            }
        )
    return out


def load_bet_type_profit_summary(
    get_data_dir,
    base_dir,
    load_csv_rows,
    build_run_race_map_func,
    min_race_year,
    scope_key,
):
    path = get_data_dir(base_dir, scope_key) / "bet_type_stats.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    run_race_map = build_run_race_map_func(scope_key)
    labels = {"win": "win", "place": "place", "wide": "wide"}
    totals = {key: {"amount": 0, "profit": 0} for key in labels}
    for row in rows:
        bet_type = str(row.get("bet_type", "")).strip().lower()
        if bet_type not in totals:
            continue
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        race_id = run_race_map.get(run_id, "")
        race_year = extract_year_prefix(race_id)
        if race_year is None:
            race_year = extract_year_prefix(run_id)
        if race_year is None:
            race_year = extract_year_prefix(row.get("timestamp", ""))
        if race_year is None or race_year < min_race_year:
            continue
        try:
            amount = int(float(row.get("amount_yen", 0) or 0))
        except (TypeError, ValueError):
            amount = 0
        try:
            profit = int(float(row.get("est_profit_yen", 0) or 0))
        except (TypeError, ValueError):
            profit = 0
        totals[bet_type]["amount"] += amount
        totals[bet_type]["profit"] += profit
    out = []
    for bet_type in ("win", "place", "wide"):
        item = totals.get(bet_type)
        if not item:
            continue
        amount = item["amount"]
        profit = item["profit"]
        roi = round((amount + profit) / amount, 4) if amount > 0 else ""
        out.append(
            {
                "bet_type": labels[bet_type],
                "amount_yen": amount,
                "est_profit_yen": profit,
                "roi": roi,
            }
        )
    return out


def compute_top5_hit_count(
    get_data_dir,
    base_dir,
    to_int_or_none,
    load_top5_names_func,
    scope_key,
    row,
):
    hit_count = to_int_or_none(row.get("top5_hit_count"))
    if hit_count is not None:
        return hit_count
    run_id = str(row.get("run_id", "") or "")
    pred_path = row.get("predictions_path", "")
    if not pred_path and run_id:
        pred_path = str(get_data_dir(base_dir, scope_key) / f"predictions_{run_id}.csv")
    if not pred_path:
        return None
    pred_path = Path(pred_path)
    if not pred_path.exists():
        return None
    top5_names = load_top5_names_func(pred_path)
    if not top5_names:
        return None
    actual_names = [row.get("actual_top1"), row.get("actual_top2"), row.get("actual_top3")]
    actual_norm = [normalize_name(n) for n in actual_names if n]
    if len(actual_norm) < 3:
        return None
    pred_norm = [normalize_name(n) for n in top5_names if n]
    return len(set(pred_norm) & set(actual_norm))


def load_predictor_summary(get_data_dir, base_dir, load_csv_rows, compute_top5_hit_count_func, scope_key):
    path = get_data_dir(base_dir, scope_key) / "predictor_results.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    total = len(rows)
    top1_hit = sum(int(float(r.get("top1_hit", 0) or 0)) for r in rows)
    top1_in_top3 = sum(int(float(r.get("top1_in_top3", 0) or 0)) for r in rows)
    top3_exact = sum(int(float(r.get("top3_exact", 0) or 0)) for r in rows)
    top3_hit = sum(int(float(r.get("top3_hit_count", 0) or 0)) for r in rows)
    top5_hit = 0
    top5_total = 0
    for row in rows:
        hit_count = compute_top5_hit_count_func(scope_key, row)
        if hit_count is None:
            continue
        top5_hit += hit_count
        top5_total += 1
    top1_rate = round(top1_hit / total, 4) if total else ""
    top1_in_top3_rate = round(top1_in_top3 / total, 4) if total else ""
    top3_exact_rate = round(top3_exact / total, 4) if total else ""
    top3_hit_rate = round(top3_hit / (3 * total), 4) if total else ""
    top5_hit_rate = round(top5_hit / (3 * top5_total), 4) if top5_total else ""
    summary = [
        {"metric": "samples", "value": total},
        {"metric": "top3_hit_rate", "value": top3_hit_rate},
        {"metric": "top1_hit_rate", "value": top1_rate},
        {"metric": "top1_in_top3_rate", "value": top1_in_top3_rate},
        {"metric": "top3_exact_rate", "value": top3_exact_rate},
    ]
    if top5_hit_rate != "":
        summary.insert(2, {"metric": "top5_to_top3_hit_rate", "value": top5_hit_rate})
    return summary


def load_run_result_summary(get_data_dir, base_dir, load_csv_rows, scope_key, run_id):
    path = get_data_dir(base_dir, scope_key) / "results.csv"
    rows = load_csv_rows(path)
    row = None
    for item in rows:
        if item.get("run_id") == run_id:
            row = item
    if not row:
        return []
    return [
        {"metric": "run_profit_yen", "value": row.get("profit_yen", "")},
        {"metric": "run_stake_yen", "value": row.get("base_amount", "")},
        {"metric": "run_roi", "value": row.get("roi", "")},
    ]


def load_run_bet_type_summary(get_data_dir, base_dir, load_csv_rows, scope_key, run_id):
    path = get_data_dir(base_dir, scope_key) / "bet_type_stats.csv"
    rows = [r for r in load_csv_rows(path) if r.get("run_id") == run_id]
    if not rows:
        return []
    out = []
    for row in rows:
        out.append(
            {
                "bet_type": row.get("bet_type", ""),
                "bets": row.get("bets", ""),
                "hits": row.get("hits", ""),
                "hit_rate": row.get("hit_rate", ""),
                "amount_yen": row.get("amount_yen", ""),
                "est_profit_yen": row.get("est_profit_yen", ""),
            }
        )
    return out


def load_run_bet_ticket_summary(get_data_dir, base_dir, load_csv_rows, scope_key, run_id):
    path = get_data_dir(base_dir, scope_key) / "bet_ticket_results.csv"
    rows = [r for r in load_csv_rows(path) if r.get("run_id") == run_id]
    if not rows:
        return []
    out = []
    for row in rows:
        try:
            amount = int(float(row.get("amount_yen", 0) or 0))
        except ValueError:
            amount = 0
        try:
            est_payout = int(float(row.get("est_payout_yen", 0) or 0))
        except ValueError:
            est_payout = 0
        out.append(
            {
                "bet_type": row.get("bet_type", ""),
                "horse_no": row.get("horse_no", ""),
                "horse_name": row.get("horse_name", ""),
                "amount_yen": amount,
                "hit": row.get("hit", ""),
                "est_payout_yen": est_payout,
                "profit_yen": est_payout - amount,
            }
        )
    return out


def load_run_predictor_summary(
    get_data_dir,
    base_dir,
    load_csv_rows,
    to_float,
    compute_top5_hit_count_func,
    scope_key,
    run_id,
):
    path = get_data_dir(base_dir, scope_key) / "predictor_results.csv"
    rows = load_csv_rows(path)
    row = next((r for r in rows if r.get("run_id") == run_id), None)
    if not row:
        return []
    top3_hit = to_float(row.get("top3_hit_count")) / 3.0 if row.get("top3_hit_count") is not None else ""
    top5_hit_count = compute_top5_hit_count_func(scope_key, row)
    top5_hit = round(top5_hit_count / 3.0, 4) if top5_hit_count is not None else ""
    summary = [
        {"metric": "run_top3_hit_rate", "value": round(top3_hit, 4) if top3_hit != "" else ""},
        {"metric": "run_top1_hit", "value": row.get("top1_hit", "")},
        {"metric": "top1_in_top3", "value": row.get("top1_in_top3", "")},
        {"metric": "top3_exact", "value": row.get("top3_exact", "")},
    ]
    if top5_hit != "":
        summary.insert(1, {"metric": "run_top5_to_top3_hit_rate", "value": top5_hit})
    return summary

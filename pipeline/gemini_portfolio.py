import csv
from datetime import datetime, timedelta
from pathlib import Path
import re


LEGACY_DAILY_BANKROLL_YEN = 10000
DAILY_BANKROLL_YEN = 50000
DAILY_BANKROLL_SWITCH_DATE = "20260317"
DEFAULT_POLICY_ENGINE = "gemini"
LEDGER_HEADERS = [
    "ledger_date",
    "policy_engine",
    "run_id",
    "scope_key",
    "race_id",
    "status",
    "ticket_id",
    "bet_type",
    "horse_nos",
    "horse_names",
    "stake_yen",
    "odds_used",
    "p_hit",
    "edge",
    "hit",
    "payout_yen",
    "profit_yen",
    "reserved_at",
    "settled_at",
]


def _shared_dir(base_dir):
    path = Path(base_dir) / "data" / "_shared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_policy_engine(policy_engine="gemini"):
    text = str(policy_engine or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return text or DEFAULT_POLICY_ENGINE


def ledger_path(base_dir, policy_engine="gemini"):
    engine = normalize_policy_engine(policy_engine)
    if engine == "gemini":
        legacy = _shared_dir(base_dir) / "gemini_ticket_ledger.csv"
        return legacy
    return _shared_dir(base_dir) / f"{engine}_ticket_ledger.csv"


def ensure_csv_header(path, fieldnames):
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        existing = reader.fieldnames or []
        rows = list(reader)
    if existing == fieldnames:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_rows(path):
    text = str(path or "").strip()
    if not text:
        return []
    path = Path(text)
    if (not path.exists()) or path.is_dir():
        return []
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    return []


def write_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def normalize_name(value):
    return "".join(str(value or "").split())


def parse_horse_no(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        pass
    digits = re.findall(r"\d+", text)
    if len(digits) == 1:
        return int(digits[0])
    return None


def extract_ledger_date(run_id="", timestamp=""):
    text = str(run_id or "").strip()
    if len(text) >= 8 and text[:8].isdigit():
        return text[:8]
    stamp = str(timestamp or "").strip()
    if len(stamp) >= 10:
        return stamp[:10].replace("-", "")
    return datetime.now().strftime("%Y%m%d")


def resolve_daily_bankroll_yen(ledger_date=""):
    date_key = str(ledger_date or "").strip() or datetime.now().strftime("%Y%m%d")
    digits = "".join(ch for ch in date_key if ch.isdigit())
    if len(digits) >= 8:
        date_key = digits[:8]
    else:
        date_key = datetime.now().strftime("%Y%m%d")
    if date_key >= DAILY_BANKROLL_SWITCH_DATE:
        return DAILY_BANKROLL_YEN
    return LEGACY_DAILY_BANKROLL_YEN


def summarize_bankroll(base_dir, ledger_date, policy_engine="gemini"):
    date_key = str(ledger_date or "").strip() or datetime.now().strftime("%Y%m%d")
    engine = normalize_policy_engine(policy_engine)
    rows = [row for row in load_rows(ledger_path(base_dir, engine)) if str(row.get("ledger_date", "")).strip() == date_key]
    open_stake = 0
    realized_profit = 0
    topup_yen = 0
    pending_runs = set()
    settled_runs = set()
    for row in rows:
        run_id = str(row.get("run_id", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        stake = to_int(row.get("stake_yen"))
        profit = to_int(row.get("profit_yen"))
        if status == "pending":
            open_stake += stake
            if run_id:
                pending_runs.add(run_id)
        elif status == "settled":
            realized_profit += profit
            if run_id:
                settled_runs.add(run_id)
        elif status == "topup":
            topup_yen += profit
    start_bankroll_yen = resolve_daily_bankroll_yen(date_key)
    available = start_bankroll_yen + topup_yen + realized_profit - open_stake
    return {
        "ledger_date": date_key,
        "start_bankroll_yen": start_bankroll_yen,
        "topup_yen": topup_yen,
        "open_stake_yen": open_stake,
        "realized_profit_yen": realized_profit,
        "available_bankroll_yen": available,
        "pending_runs": len(pending_runs),
        "settled_runs": len(settled_runs),
        "pending_tickets": sum(1 for row in rows if str(row.get("status", "")).strip().lower() == "pending"),
        "settled_tickets": sum(1 for row in rows if str(row.get("status", "")).strip().lower() == "settled"),
        "policy_engine": engine,
    }


def add_bankroll_topup(base_dir, ledger_date, amount_yen, policy_engine="gemini"):
    engine = normalize_policy_engine(policy_engine)
    amount = max(0, to_int(amount_yen))
    date_key = str(ledger_date or "").strip() or datetime.now().strftime("%Y%m%d")
    path = ledger_path(base_dir, engine)
    rows = load_rows(path)
    ts = datetime.now().isoformat(timespec="seconds")
    rows.append(
        {
            "ledger_date": date_key,
            "policy_engine": engine,
            "run_id": "",
            "scope_key": "",
            "race_id": "",
            "status": "topup",
            "ticket_id": f"topup_{engine}_{ts.replace(':', '').replace('-', '').replace('T', '_')}",
            "bet_type": "topup",
            "horse_nos": "",
            "horse_names": "",
            "stake_yen": 0,
            "odds_used": "",
            "p_hit": "",
            "edge": "",
            "hit": "",
            "payout_yen": "",
            "profit_yen": amount,
            "reserved_at": ts,
            "settled_at": ts,
        }
    )
    write_rows(path, LEDGER_HEADERS, rows)
    summary = summarize_bankroll(base_dir, date_key, policy_engine=engine)
    summary["topup_amount_yen"] = amount
    return summary


def load_run_tickets(base_dir, run_id, policy_engine="gemini"):
    engine = normalize_policy_engine(policy_engine)
    rows = [
        row
        for row in load_rows(ledger_path(base_dir, engine))
        if str(row.get("run_id", "")).strip() == str(run_id or "").strip()
    ]
    if not rows:
        return []
    return rows


def reserve_run_tickets(base_dir, run_id, scope_key, race_id, ledger_date, tickets, policy_engine="gemini"):
    engine = normalize_policy_engine(policy_engine)
    path = ledger_path(base_dir, engine)
    existing = load_rows(path)
    keep = []
    run_text = str(run_id or "").strip()
    for row in existing:
        row_run_id = str(row.get("run_id", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        if row_run_id == run_text and status != "settled":
            continue
        keep.append(row)
    reserved_at = datetime.now().isoformat(timespec="seconds")
    for ticket in list(tickets or []):
        keep.append(
            {
                "ledger_date": str(ledger_date or ""),
                "policy_engine": engine,
                "run_id": run_text,
                "scope_key": str(scope_key or ""),
                "race_id": str(race_id or ""),
                "status": "pending",
                "ticket_id": str(ticket.get("ticket_id", "") or ""),
                "bet_type": str(ticket.get("bet_type", "") or ""),
                "horse_nos": str(ticket.get("horse_no", "") or ticket.get("horse_nos", "") or ""),
                "horse_names": str(ticket.get("horse_name", "") or ticket.get("horse_names", "") or ""),
                "stake_yen": to_int(ticket.get("amount_yen", ticket.get("stake_yen", 0))),
                "odds_used": str(ticket.get("odds_used", "") or ""),
                "p_hit": str(ticket.get("p_hit", "") or ""),
                "edge": str(ticket.get("edge", "") or ""),
                "hit": "",
                "payout_yen": "",
                "profit_yen": "",
                "reserved_at": reserved_at,
                "settled_at": "",
            }
        )
    write_rows(path, LEDGER_HEADERS, keep)


def load_name_to_no(path):
    rows = load_rows(path)
    if not rows:
        return {}
    out = {}
    for row in rows:
        name = row.get("name") or row.get("HorseName") or row.get("horse_name")
        horse_no = parse_horse_no(row.get("horse_no") or row.get("HorseNo") or row.get("馬番"))
        if not name or horse_no is None:
            continue
        out[normalize_name(name)] = horse_no
    return out


def load_win_odds_map(path):
    rows = load_rows(path)
    if not rows:
        return {}
    out = {}
    for row in rows:
        name = row.get("name") or row.get("HorseName") or row.get("horse_name")
        if not name:
            continue
        try:
            odds = float(row.get("odds", 0) or 0)
        except (TypeError, ValueError):
            continue
        if odds > 0:
            out[normalize_name(name)] = odds
    return out


def load_place_odds_map(path):
    rows = load_rows(path)
    if not rows:
        return {}
    odds_col = "odds_low" if rows and "odds_low" in rows[0] else "odds_mid"
    out = {}
    for row in rows:
        horse_no = parse_horse_no(row.get("horse_no"))
        if horse_no is None:
            continue
        try:
            odds = float(row.get(odds_col, 0) or 0)
        except (TypeError, ValueError):
            continue
        if odds > 0:
            out[horse_no] = odds
    return out


def load_pair_odds_map(path):
    rows = load_rows(path)
    if not rows:
        return {}
    odds_col = "odds_mid" if "odds_mid" in rows[0] else "odds"
    out = {}
    for row in rows:
        a = parse_horse_no(row.get("horse_no_a"))
        b = parse_horse_no(row.get("horse_no_b"))
        if a is None or b is None:
            continue
        if a > b:
            a, b = b, a
        try:
            odds = float(row.get(odds_col, 0) or 0)
        except (TypeError, ValueError):
            continue
        if odds > 0:
            out[(a, b)] = odds
    return out


def load_exacta_odds_map(path):
    rows = load_rows(path)
    if not rows:
        return {}
    out = {}
    for row in rows:
        a = parse_horse_no(row.get("horse_no_a"))
        b = parse_horse_no(row.get("horse_no_b"))
        if a is None or b is None:
            continue
        try:
            odds = float(row.get("odds", 0) or 0)
        except (TypeError, ValueError):
            continue
        if odds > 0:
            out[(a, b)] = odds
    return out


def load_triple_odds_map(path, ordered=False):
    rows = load_rows(path)
    if not rows:
        return {}
    out = {}
    for row in rows:
        a = parse_horse_no(row.get("horse_no_a"))
        b = parse_horse_no(row.get("horse_no_b"))
        c = parse_horse_no(row.get("horse_no_c"))
        if a is None or b is None or c is None:
            continue
        key = (a, b, c) if ordered else tuple(sorted((a, b, c)))
        try:
            odds = float(row.get("odds", 0) or 0)
        except (TypeError, ValueError):
            continue
        if odds > 0:
            out[key] = odds
    return out


def eval_ticket_hit(bet_type, horse_names, actual_order):
    names = [normalize_name(x) for x in list(horse_names or []) if normalize_name(x)]
    actual = [normalize_name(x) for x in list(actual_order or []) if normalize_name(x)]
    if not names or len(actual) < 3:
        return 0
    top3 = actual[:3]
    top2 = actual[:2]
    bet = str(bet_type or "").strip().lower()
    if bet == "win":
        return 1 if names[0] == actual[0] else 0
    if bet == "place":
        return 1 if names[0] in top3 else 0
    if bet == "wide":
        return 1 if len(names) >= 2 and names[0] in top3 and names[1] in top3 else 0
    if bet == "quinella":
        return 1 if len(names) >= 2 and names[0] in top2 and names[1] in top2 else 0
    if bet == "exacta":
        return 1 if len(names) >= 2 and names[0] == actual[0] and names[1] == actual[1] else 0
    if bet == "trio":
        return 1 if len(names) >= 3 and set(names[:3]) == set(top3) else 0
    return 0


def eval_ticket_hit_by_nos(bet_type, horse_nos, actual_top3_nos):
    nos = [parse_horse_no(x) for x in list(horse_nos or [])]
    nos = [x for x in nos if x is not None]
    actual = [parse_horse_no(x) for x in list(actual_top3_nos or [])]
    actual = [x for x in actual if x is not None]
    if not nos or len(actual) < 3:
        return 0
    top3 = actual[:3]
    top2 = actual[:2]
    bet = str(bet_type or "").strip().lower()
    if bet == "win":
        return 1 if nos[0] == actual[0] else 0
    if bet == "place":
        return 1 if nos[0] in top3 else 0
    if bet == "wide":
        return 1 if len(nos) >= 2 and nos[0] in top3 and nos[1] in top3 else 0
    if bet == "quinella":
        return 1 if len(nos) >= 2 and nos[0] in top2 and nos[1] in top2 else 0
    if bet == "exacta":
        return 1 if len(nos) >= 2 and nos[0] == actual[0] and nos[1] == actual[1] else 0
    if bet == "trio":
        return 1 if len(nos) >= 3 and set(nos[:3]) == set(top3) else 0
    return 0


def build_official_payout_lookup(official_result_payload):
    bet_type_map = {
        "単勝": "win",
        "複勝": "place",
        "ワイド": "wide",
        "馬連": "quinella",
        "馬単": "exacta",
        "3連複": "trio",
    }
    payouts = {}
    for jp_bet_type, entries in dict((official_result_payload or {}).get("payouts", {}) or {}).items():
        bet_type = bet_type_map.get(str(jp_bet_type or "").strip(), "")
        if not bet_type:
            continue
        bucket = payouts.setdefault(bet_type, {})
        for entry in list(entries or []):
            payout_yen = to_int((entry or {}).get("payout_yen"))
            horse_numbers = [parse_horse_no(x) for x in list((entry or {}).get("horse_numbers", []) or [])]
            horse_numbers = [x for x in horse_numbers if x is not None]
            if payout_yen <= 0 or not horse_numbers:
                continue
            if bet_type in ("wide", "quinella", "trio"):
                key = tuple(sorted(horse_numbers))
            else:
                key = tuple(horse_numbers)
            bucket[key] = payout_yen
    return payouts


def _resolve_official_ticket_payout(stake_yen, bet_type, nos, official_payouts):
    if not official_payouts:
        return 0
    bet = str(bet_type or "").strip().lower()
    if bet not in dict(official_payouts or {}):
        return 0
    values = [parse_horse_no(x) for x in list(nos or [])]
    values = [x for x in values if x is not None]
    if not values:
        return 0
    if bet in ("wide", "quinella", "trio"):
        key = tuple(sorted(values))
    else:
        key = tuple(values)
    payout_per_100 = to_int(dict(official_payouts or {}).get(bet, {}).get(key))
    if payout_per_100 <= 0:
        return 0
    return int(round((to_int(stake_yen) / 100.0) * payout_per_100))


def settle_run_tickets(base_dir, run_row, actual_top3_names, policy_engine="gemini", official_result_payload=None):
    run_id = str((run_row or {}).get("run_id", "")).strip()
    if not run_id:
        return None
    engine = normalize_policy_engine(policy_engine)
    path = ledger_path(base_dir, engine)
    rows = load_rows(path)
    pending = [row for row in rows if str(row.get("run_id", "")).strip() == run_id and str(row.get("status", "")).strip().lower() == "pending"]
    if not pending:
        return None
    odds_path = (run_row or {}).get("odds_path", "")
    fuku_path = (run_row or {}).get("fuku_odds_path", "")
    wide_path = (run_row or {}).get("wide_odds_path", "")
    quinella_path = (run_row or {}).get("quinella_odds_path", "")
    exacta_path = (run_row or {}).get("exacta_odds_path", "")
    trio_path = (run_row or {}).get("trio_odds_path", "")
    name_to_no = load_name_to_no(odds_path)
    win_odds_map = load_win_odds_map(odds_path)
    place_odds_map = load_place_odds_map(fuku_path)
    wide_odds_map = load_pair_odds_map(wide_path)
    quinella_odds_map = load_pair_odds_map(quinella_path)
    exacta_odds_map = load_exacta_odds_map(exacta_path)
    trio_odds_map = load_triple_odds_map(trio_path, ordered=False)
    official_payouts = build_official_payout_lookup(official_result_payload)
    official_top3_nos = [parse_horse_no(item.get("horse_no")) for item in list((official_result_payload or {}).get("top3", []) or [])[:3]]
    official_top3_nos = [x for x in official_top3_nos if x is not None]
    settled_at = datetime.now().isoformat(timespec="seconds")
    total_stake = 0
    total_payout = 0
    total_profit = 0
    settled_count = 0
    updated = []
    for row in rows:
        row_run_id = str(row.get("run_id", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        if row_run_id != run_id or status != "pending":
            updated.append(row)
            continue
        bet_type = str(row.get("bet_type", "")).strip().lower()
        names = [x.strip() for x in str(row.get("horse_names", "")).split("/") if x.strip()]
        nos = [parse_horse_no(x) for x in str(row.get("horse_nos", "")).split("-") if str(x).strip()]
        nos = [x for x in nos if x is not None]
        hit = eval_ticket_hit_by_nos(bet_type, nos, official_top3_nos) if official_top3_nos else eval_ticket_hit(bet_type, names, actual_top3_names)
        stake = to_int(row.get("stake_yen"))
        payout = 0
        if hit:
            payout = _resolve_official_ticket_payout(stake, bet_type, nos, official_payouts)
            if payout <= 0 and bet_type == "win" and names:
                payout = int(round(stake * float(win_odds_map.get(normalize_name(names[0]), 0) or 0)))
            elif payout <= 0 and bet_type == "place" and nos:
                payout = int(round(stake * float(place_odds_map.get(nos[0], 0) or 0)))
            elif payout <= 0 and bet_type == "wide" and len(nos) >= 2:
                a, b = sorted(nos[:2])
                payout = int(round(stake * float(wide_odds_map.get((a, b), 0) or 0)))
            elif payout <= 0 and bet_type == "quinella" and len(nos) >= 2:
                a, b = sorted(nos[:2])
                payout = int(round(stake * float(quinella_odds_map.get((a, b), 0) or 0)))
            elif payout <= 0 and bet_type == "exacta" and len(nos) >= 2:
                payout = int(round(stake * float(exacta_odds_map.get((nos[0], nos[1]), 0) or 0)))
            elif payout <= 0 and bet_type == "trio" and len(nos) >= 3:
                key = tuple(sorted(nos[:3]))
                payout = int(round(stake * float(trio_odds_map.get(key, 0) or 0)))
        profit = int(payout - stake)
        settled_count += 1
        total_stake += stake
        total_payout += payout
        total_profit += profit
        settled_row = dict(row)
        settled_row["status"] = "settled"
        settled_row["hit"] = int(hit)
        settled_row["payout_yen"] = payout
        settled_row["profit_yen"] = profit
        settled_row["settled_at"] = settled_at
        updated.append(settled_row)
    write_rows(path, LEDGER_HEADERS, updated)
    summary = summarize_bankroll(
        base_dir,
        extract_ledger_date(run_id, (run_row or {}).get("timestamp", "")),
        policy_engine=engine,
    )
    summary.update(
        {
            "run_id": run_id,
            "policy_engine": engine,
            "settled_ticket_count": settled_count,
            "run_stake_yen": total_stake,
            "run_payout_yen": total_payout,
            "run_profit_yen": total_profit,
        }
    )
    return summary


def load_daily_profit_rows(base_dir, days=30, policy_engine="gemini"):
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 30
    cutoff = datetime.now().date() - timedelta(days=max(0, days - 1))
    daily = {}
    engine = normalize_policy_engine(policy_engine)
    for row in load_rows(ledger_path(base_dir, engine)):
        if str(row.get("status", "")).strip().lower() != "settled":
            continue
        date_key = str(row.get("ledger_date", "")).strip()
        if len(date_key) != 8:
            continue
        try:
            date_obj = datetime.strptime(date_key, "%Y%m%d").date()
        except ValueError:
            continue
        if date_obj < cutoff:
            continue
        item = daily.setdefault(
            date_key,
            {"date": date_obj.strftime("%Y-%m-%d"), "runs": set(), "profit_yen": 0, "base_amount": 0},
        )
        item["runs"].add(str(row.get("run_id", "")).strip())
        item["profit_yen"] += to_int(row.get("profit_yen"))
        item["base_amount"] += to_int(row.get("stake_yen"))
    out = []
    for date_key in sorted(daily.keys(), reverse=True):
        item = daily[date_key]
        base = int(item["base_amount"])
        profit = int(item["profit_yen"])
        roi = round((base + profit) / base, 4) if base > 0 else ""
        out.append(
            {
                "date": item["date"],
                "runs": len(item["runs"]),
                "profit_yen": profit,
                "base_amount": base,
                "roi": roi,
                "policy_engine": engine,
            }
        )
    return out


def _row_datetime_key(row):
    for field in ("settled_at", "reserved_at"):
        text = str(row.get(field, "")).strip()
        if not text:
            continue
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            continue
    date_key = str(row.get("ledger_date", "")).strip()
    if len(date_key) == 8:
        try:
            return datetime.strptime(date_key, "%Y%m%d")
        except ValueError:
            pass
    return datetime.min


def build_history_context(base_dir, ledger_date, lookback_days=14, recent_ticket_limit=8, policy_engine="gemini"):
    try:
        lookback_days = int(lookback_days)
    except (TypeError, ValueError):
        lookback_days = 14
    try:
        recent_ticket_limit = int(recent_ticket_limit)
    except (TypeError, ValueError):
        recent_ticket_limit = 8
    lookback_days = max(1, lookback_days)
    recent_ticket_limit = max(1, recent_ticket_limit)
    today_key = str(ledger_date or "").strip() or datetime.now().strftime("%Y%m%d")
    cutoff = datetime.now().date() - timedelta(days=max(0, lookback_days - 1))
    engine = normalize_policy_engine(policy_engine)
    rows = load_rows(ledger_path(base_dir, engine))
    settled_rows = []
    active_rows = []
    bet_type_stats = {}
    recent_tickets = []
    for row in rows:
        date_key = str(row.get("ledger_date", "")).strip()
        if len(date_key) != 8:
            continue
        try:
            date_obj = datetime.strptime(date_key, "%Y%m%d").date()
        except ValueError:
            continue
        if date_obj < cutoff:
            continue
        status = str(row.get("status", "")).strip().lower()
        if status == "topup":
            continue
        stake = to_int(row.get("stake_yen"))
        payout = to_int(row.get("payout_yen"))
        profit = to_int(row.get("profit_yen"))
        hit = to_int(row.get("hit"))
        bet_type = str(row.get("bet_type", "")).strip().lower() or "unknown"
        stat = bet_type_stats.setdefault(
            bet_type,
            {
                "bet_type": bet_type,
                "tickets": 0,
                "stake_yen": 0,
                "payout_yen": 0,
                "profit_yen": 0,
                "hits": 0,
            },
        )
        stat["tickets"] += 1
        stat["stake_yen"] += stake
        if status == "settled":
            stat["payout_yen"] += payout
            stat["profit_yen"] += profit
            stat["hits"] += 1 if hit else 0
            settled_rows.append(row)
        active_rows.append(row)
        recent_tickets.append(
            {
                "ledger_date": date_key,
                "race_id": str(row.get("race_id", "")).strip(),
                "run_id": str(row.get("run_id", "")).strip(),
                "status": status,
                "bet_type": bet_type,
                "horse_nos": str(row.get("horse_nos", "")).strip(),
                "horse_names": str(row.get("horse_names", "")).strip(),
                "stake_yen": stake,
                "payout_yen": payout if status == "settled" else "",
                "profit_yen": profit if status == "settled" else "",
                "hit": hit if status == "settled" else "",
                "reserved_at": str(row.get("reserved_at", "")).strip(),
                "settled_at": str(row.get("settled_at", "")).strip(),
            }
        )
    settled_stake = sum(to_int(row.get("stake_yen")) for row in settled_rows)
    settled_payout = sum(to_int(row.get("payout_yen")) for row in settled_rows)
    settled_profit = sum(to_int(row.get("profit_yen")) for row in settled_rows)
    hit_count = sum(1 for row in settled_rows if to_int(row.get("hit")) > 0)
    recent_tickets = sorted(recent_tickets, key=_row_datetime_key, reverse=True)[:recent_ticket_limit]
    bet_type_rows = []
    for key in sorted(bet_type_stats.keys()):
        stat = dict(bet_type_stats[key])
        stake_yen = int(stat.get("stake_yen", 0) or 0)
        payout_yen = int(stat.get("payout_yen", 0) or 0)
        stat["roi"] = round(payout_yen / stake_yen, 4) if stake_yen > 0 else ""
        bet_type_rows.append(stat)
    return {
        "today": summarize_bankroll(base_dir, today_key, policy_engine=engine),
        "recent_days": load_daily_profit_rows(base_dir, days=lookback_days, policy_engine=engine),
        "lookback_summary": {
            "days": lookback_days,
            "settled_runs": len({str(row.get("run_id", "")).strip() for row in settled_rows if str(row.get("run_id", "")).strip()}),
            "settled_tickets": len(settled_rows),
            "lookback_tickets": len(active_rows),
            "hit_tickets": hit_count,
            "hit_rate": round(hit_count / len(settled_rows), 4) if settled_rows else "",
            "stake_yen": settled_stake,
            "payout_yen": settled_payout,
            "profit_yen": settled_profit,
            "roi": round(settled_payout / settled_stake, 4) if settled_stake > 0 else "",
        },
        "bet_type_breakdown": bet_type_rows,
        "recent_tickets": recent_tickets,
        "policy_engine": engine,
    }

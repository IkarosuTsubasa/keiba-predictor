import re
from datetime import datetime, timedelta

from surface_scope import normalize_scope_key
from web_helpers import normalize_race_id, to_int_or_none


LLM_BATTLE_ORDER = ("openai", "gemini", "deepseek", "grok")
LLM_BATTLE_LABELS = {
    "openai": "ChatGPT",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
}
LLM_NOTE_LABELS = dict(LLM_BATTLE_LABELS)
LLM_BATTLE_SHORT_LABELS = {
    "openai": "chatgpt",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "grok": "grok",
}
LLM_REPORT_SCOPE_KEYS = ("central_dirt", "central_turf", "local")
BET_TYPE_TEXT_MAP = {
    "win": "単勝",
    "place": "複勝",
    "wide": "ワイド",
    "quinella": "馬連",
    "exacta": "馬単",
    "trio": "3連複",
}
MARK_SYMBOL_ORDER = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}


def safe_text(value):
    return str(value or "").strip()


def scope_display_name(scope_key):
    mapping = {
        "central_dirt": "中央ダート",
        "central_turf": "中央芝",
        "local": "地方",
    }
    return mapping.get(str(scope_key or "").strip(), str(scope_key or "").strip() or "-")


def policy_primary_budget(payload):
    budgets = list((payload or {}).get("budgets", []) or [])
    for item in budgets:
        if isinstance(item, dict):
            return item
    return {}


def policy_primary_output(payload):
    item = policy_primary_budget(payload)
    return dict(item.get("output", {}) or {})


def policy_marks_map(payload):
    marks = {}
    for row in list(policy_primary_output(payload).get("marks", []) or []):
        symbol = safe_text(row.get("symbol"))
        horse_no = safe_text(row.get("horse_no"))
        if symbol and horse_no and horse_no not in marks:
            marks[horse_no] = symbol
    return marks


def format_bet_type_text(value):
    text = safe_text(value).lower()
    return BET_TYPE_TEXT_MAP.get(text, text or "-")


def format_ticket_target_text(bet_type, target):
    bet_type_text = safe_text(bet_type).lower()
    text = safe_text(target)
    if not text:
        return "-"
    parts = [part.strip() for part in text.split("-") if part.strip()]
    if bet_type_text == "exacta" and len(parts) == 2:
        return f"{parts[0]}→{parts[1]}"
    return "-".join(parts) if parts else text


def _format_amount_text(value):
    amount = to_int_or_none(value)
    return f"¥{amount:,}" if amount is not None else "-"


def format_ticket_plan_text(ticket_rows, output):
    rows = list(ticket_rows or [])
    grouped = {}
    for row in rows:
        bet_type_key = safe_text(row.get("bet_type")).lower()
        bet_type = format_bet_type_text(bet_type_key)
        horse_no = format_ticket_target_text(bet_type_key, row.get("horse_no"))
        amount = row.get("amount_yen")
        if amount in ("", None):
            amount = row.get("stake_yen")
        amount_text = _format_amount_text(amount)
        grouped.setdefault(bet_type, []).append(f"{horse_no} {amount_text}")
    if not grouped:
        plan_rows = list((output or {}).get("ticket_plan", []) or [])
        for row in plan_rows:
            ticket_id = safe_text(row.get("id"))
            if ticket_id:
                bet_type, _, target = ticket_id.partition(":")
            else:
                bet_type = safe_text(row.get("bet_type")).lower()
                legs = [safe_text(x) for x in list(row.get("legs", []) or []) if safe_text(x)]
                if not bet_type or not legs:
                    continue
                target = "-".join(legs)
            bet_type_label = format_bet_type_text(bet_type)
            target_text = format_ticket_target_text(bet_type, target or "-")
            amount_text = _format_amount_text(row.get("stake_yen"))
            grouped.setdefault(bet_type_label, []).append(f"{target_text} {amount_text}")
    if not grouped:
        return "買い目なし"

    ordered_types = [
        BET_TYPE_TEXT_MAP[key]
        for key in ("exacta", "quinella", "wide", "trio", "win", "place")
        if BET_TYPE_TEXT_MAP.get(key) in grouped
    ]
    extra_types = [bet_type for bet_type in grouped if bet_type not in ordered_types]
    lines = []
    for bet_type in ordered_types + extra_types:
        lines.append(bet_type)
        lines.extend(grouped.get(bet_type, []))
        lines.append("")
    return "\n".join(lines).strip()


def format_marks_text(marks_map):
    if not marks_map:
        return "印なし"
    ordered = []
    for horse_no, symbol in marks_map.items():
        ordered.append((MARK_SYMBOL_ORDER.get(symbol, 99), horse_no, symbol))
    ordered.sort(key=lambda item: (item[0], to_int_or_none(item[1]) or 999, item[1]))
    return " ".join(f"{symbol}{horse_no}" for _, horse_no, symbol in ordered)


def run_date_key(run_row):
    race_date = safe_text((run_row or {}).get("race_date"))
    if race_date:
        return race_date
    timestamp = safe_text((run_row or {}).get("timestamp"))
    return timestamp[:10] if len(timestamp) >= 10 else ""


def parse_run_date(date_text):
    text = safe_text(date_text)
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def format_jp_date_value(date_value):
    if not date_value:
        return ""
    return f"{date_value.year}年{date_value.month}月{date_value.day}日"


def has_llm_policy_assets(run_row):
    row = dict(run_row or {})
    return bool(
        safe_text(row.get("gemini_policy_path"))
        or safe_text(row.get("deepseek_policy_path"))
        or safe_text(row.get("openai_policy_path"))
        or safe_text(row.get("grok_policy_path"))
    )


def report_scope_key_for_row(run_row, fallback_scope=""):
    row = dict(run_row or {})
    return normalize_scope_key(row.get("_report_scope_key") or row.get("scope_key") or fallback_scope) or ""


def payload_run_id(payload, fallback_run_id=""):
    return safe_text((payload or {}).get("run_id")) or safe_text(fallback_run_id)


def race_no_text(race_id):
    digits = normalize_race_id(race_id)
    if len(digits) >= 2:
        try:
            return f"{int(digits[-2:])}R"
        except (TypeError, ValueError):
            return ""
    return ""


def format_race_label(run_row):
    row = dict(run_row or {})
    venue = safe_text(row.get("location")) or safe_text(row.get("trigger_race"))
    race_no = race_no_text(row.get("race_id"))
    if venue and race_no:
        return f"{venue}{race_no}"
    if venue:
        return venue
    if race_no:
        return race_no
    return safe_text(row.get("race_id")) or safe_text(row.get("run_id")) or "-"


def format_jp_date_text(run_row):
    date_text = safe_text((run_row or {}).get("race_date"))
    if not date_text:
        timestamp = safe_text((run_row or {}).get("timestamp"))
        if len(timestamp) >= 10:
            date_text = timestamp[:10]
    if not date_text:
        return ""
    matched = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_text)
    if not matched:
        return date_text
    return f"{int(matched.group(1))}年{int(matched.group(2))}月{int(matched.group(3))}日"


def build_battle_title(run_row):
    race_label = format_race_label(run_row)
    date_label = format_jp_date_text(run_row)
    race_name = safe_text((run_row or {}).get("trigger_race"))
    parts = ["4つのAI競馬対決"]
    if date_label or race_label:
        detail = " ".join(part for part in [date_label, race_label] if part)
        if race_name:
            detail = f"{detail}「{race_name}」" if detail else f"「{race_name}」"
        if detail:
            parts.append(detail)
    return " ".join(parts)


def jst_today_text():
    return (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")


def normalize_report_date_text(date_text=""):
    text = safe_text(date_text)
    if not text:
        return jst_today_text()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    parsed = parse_run_date(text)
    if parsed:
        return parsed.strftime("%Y-%m-%d")
    return jst_today_text()


def llm_today_scope_keys(scope_key=""):
    scope_norm = normalize_scope_key(scope_key)
    if scope_norm:
        return [scope_norm]
    return list(LLM_REPORT_SCOPE_KEYS)


def format_yen_text(value):
    try:
        amount = int(value or 0)
    except (TypeError, ValueError):
        amount = 0
    sign = "-" if amount < 0 else ""
    return f"{sign}¥{abs(amount):,}"


def format_percent_text(value):
    if value in ("", None):
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def llm_today_scope_label_ja(scope_key):
    mapping = {
        "central_dirt": "中央ダート",
        "central_turf": "中央芝",
        "local": "地方",
    }
    return mapping.get(str(scope_key or "").strip(), str(scope_key or "").strip() or "-")


def public_scope_label_ja(scope_key):
    mapping = {
        "central_dirt": "中央ダート",
        "central_turf": "中央芝",
        "local": "地方競馬",
    }
    return mapping.get(str(scope_key or "").strip(), str(scope_key or "").strip() or "-")


import html
import json


def load_policy_payload(
    scope_key,
    run_id,
    run_row=None,
    *,
    resolve_run_asset_path,
    load_json_file,
):
    candidates = [
        ("policy_path", "policy"),
        ("gemini_policy_path", "gemini_policy"),
        ("deepseek_policy_path", "deepseek_policy"),
        ("openai_policy_path", "openai_policy"),
        ("grok_policy_path", "grok_policy"),
    ]
    for field_name, prefix in candidates:
        path = resolve_run_asset_path(
            scope_key,
            run_id,
            run_row,
            field_name,
            prefix,
            ext=".json",
        )
        payload = load_json_file(path)
        if payload:
            return payload
    return {}


def load_policy_payloads(
    scope_key,
    run_id,
    run_row=None,
    *,
    resolve_run_asset_path,
    load_json_file,
    normalize_policy_engine,
):
    payloads = []
    seen = set()
    candidates = [
        ("gemini", "gemini_policy_path", "gemini_policy"),
        ("deepseek", "deepseek_policy_path", "deepseek_policy"),
        ("openai", "openai_policy_path", "openai_policy"),
        ("grok", "grok_policy_path", "grok_policy"),
        ("", "policy_path", "policy"),
    ]
    for default_engine, field_name, prefix in candidates:
        path = resolve_run_asset_path(
            scope_key,
            run_id,
            run_row,
            field_name,
            prefix,
            ext=".json",
        )
        payload = load_json_file(path)
        if not payload:
            continue
        item = dict(payload)
        policy_engine = normalize_policy_engine(item.get("policy_engine", "") or default_engine or "gemini")
        item["policy_engine"] = policy_engine
        key = (
            policy_engine,
            str(item.get("run_id", "") or run_id or ""),
            str(item.get("saved_at", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        payloads.append(item)
    return payloads


def _policy_chip_row(label, values, tone=""):
    chips = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        tone_class = f" policy-chip--{tone}" if tone else ""
        chips.append(f'<span class="policy-chip{tone_class}">{html.escape(text)}</span>')
    if not chips:
        chips.append('<span class="policy-chip policy-chip--empty">none</span>')
    return (
        '<div class="policy-horse-group">'
        f'<div class="policy-label">{html.escape(label)}</div>'
        f'<div class="policy-chip-row">{"".join(chips)}</div>'
        "</div>"
    )


def _policy_detail_row(label, value, code=False):
    text = str(value or "").strip()
    if not text:
        return ""
    body = f"<code>{html.escape(text)}</code>" if code else f"<span>{html.escape(text)}</span>"
    return f'<div class="policy-detail-row"><strong>{html.escape(label)}</strong>{body}</div>'


def _policy_mark_html(marks):
    items = []
    for mark in list(marks or []):
        symbol = str(mark.get("symbol", "") or "").strip()
        horse_no = str(mark.get("horse_no", "") or "").strip()
        if not symbol or not horse_no:
            continue
        items.append(
            '<div class="policy-mark-item">'
            f'<span class="policy-mark-symbol">{html.escape(symbol)}</span>'
            f'<span class="policy-mark-horse">{html.escape(horse_no)}</span>'
            "</div>"
        )
    if not items:
        return ""
    return (
        '<section class="policy-marks">'
        '<div class="policy-label">Marks</div>'
        f'<div class="policy-mark-row">{"".join(items)}</div>'
        "</section>"
    )


def _policy_ticket_html(rows):
    if not rows:
        return ""
    body = []
    for row in rows:
        status = str(row.get("status", "") or "").strip() or "pending"
        body.append(
            "<tr>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(str(row.get('bet_type', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('horse_no', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('horse_name', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('amount_yen', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('odds_used', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('profit_yen', '') or ''))}</td>"
            "</tr>"
        )
    return (
        '<section class="policy-ticket-block">'
        '<div class="policy-label">Ticket Plan</div>'
        '<div class="table-wrap table-wrap--medium">'
        '<table class="data-table data-table--medium">'
        "<thead><tr><th>status</th><th>bet_type</th><th>horse_no</th><th>horse_name</th><th>amount_yen</th><th>odds_used</th><th>profit_yen</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def build_policy_html(payload, *, to_float):
    if not isinstance(payload, dict) or not payload:
        return ""
    budgets = list(payload.get("budgets", []) or [])
    model = str(payload.get("policy_model", "") or payload.get("gemini_model", "") or "")
    policy_engine = str(payload.get("policy_engine", "") or "")
    engine_label_map = {
        "gemini": "Gemini",
        "deepseek": "DeepSeek",
        "openai": "ChatGPT",
        "grok": "xAI Grok",
    }
    panel_title = engine_label_map.get(policy_engine, policy_engine or "LLM")
    header_tags = []
    if model:
        header_tags.append(f'<span class="policy-meta-tag">{html.escape(model)}</span>')
    if policy_engine:
        header_tags.append(f'<span class="policy-meta-tag">{html.escape(policy_engine)}</span>')
    budget_sections = []
    for item in budgets:
        if not isinstance(item, dict):
            continue
        is_shared = bool(item.get("shared_policy", False))
        budget = int(to_float(item.get("budget_yen")))
        output = dict(item.get("output", {}) or {})
        meta = dict(item.get("meta", {}) or {})
        header = "[shared]" if is_shared else f"[{budget}]"
        decision = str(output.get("bet_decision", "") or "")
        participation = str(output.get("participation_level", "") or "")
        portfolio = dict(item.get("portfolio", {}) or {})
        portfolio_before = dict(portfolio.get("before", {}) or {})
        portfolio_after = dict(portfolio.get("after", {}) or {})
        summary_tags = "".join(
            f'<span class="policy-meta-tag">{html.escape(tag)}</span>'
            for tag in [header, decision, participation]
            if str(tag or "").strip()
        )
        bankroll_cards = ""
        if portfolio_before or portfolio_after:
            bankroll_cards = (
                '<div class="policy-bankroll-grid">'
                '<article class="policy-text-card">'
                '<div class="policy-label">Available Before</div>'
                f"<p>{html.escape(str(portfolio_before.get('available_bankroll_yen', '')))} JPY</p>"
                "</article>"
                '<article class="policy-text-card">'
                '<div class="policy-label">Available After</div>'
                f"<p>{html.escape(str(portfolio_after.get('available_bankroll_yen', '')))} JPY</p>"
                "</article>"
                '<article class="policy-text-card">'
                '<div class="policy-label">Realized P/L Today</div>'
                f"<p>{html.escape(str(portfolio_after.get('realized_profit_yen', portfolio_before.get('realized_profit_yen', ''))))} JPY</p>"
                "</article>"
                '<article class="policy-text-card">'
                '<div class="policy-label">Pending Tickets Before</div>'
                f"<p>{html.escape(str(portfolio_before.get('pending_tickets', '')))}</p>"
                "</article>"
                '<article class="policy-text-card">'
                '<div class="policy-label">Pending Tickets After</div>'
                f"<p>{html.escape(str(portfolio_after.get('pending_tickets', '')))}</p>"
                "</article>"
                "</div>"
            )
        horse_groups = "".join(
            [
                _policy_chip_row("Key Horses", list(output.get("key_horses", []) or []), "key"),
                _policy_chip_row("Secondary", list(output.get("secondary_horses", []) or []), "secondary"),
                _policy_chip_row("Longshot", list(output.get("longshot_horses", []) or []), "longshot"),
            ]
        )
        mark_block = _policy_mark_html(list(output.get("marks", []) or []))
        ticket_block = _policy_ticket_html(list(item.get("tickets", []) or []))
        text_cards = ""
        fallback_reason = str(meta.get("fallback_reason", "") or "").strip()
        error_detail = str(meta.get("error_detail", "") or "").strip()
        if fallback_reason or error_detail:
            error_lines = [fallback_reason] if fallback_reason else []
            if error_detail:
                error_lines.append(error_detail)
            for warn in list(output.get("warnings", []) or []):
                text = str(warn or "").strip()
                if text and text not in error_lines:
                    error_lines.append(text)
            error_html = (
                '<article class="policy-text-card policy-text-card--primary">'
                '<div class="policy-label">Error</div>'
                f"<p>{html.escape(' | '.join(error_lines))}</p>"
                "</article>"
            )
            text_cards = f'<div class="policy-text-grid">{error_html}</div>'
        detail_rows = []
        detail_rows.append(
            _policy_detail_row(
                "enabled_bet_types",
                json.dumps(output.get("enabled_bet_types", []), ensure_ascii=False),
                code=True,
            )
        )
        detail_rows.append(
            _policy_detail_row("reason_codes", json.dumps(output.get("reason_codes", []), ensure_ascii=False), code=True)
        )
        detail_rows.append(_policy_detail_row("comment", output.get("comment", "")))
        detail_rows.append(_policy_detail_row("bet_decision", output.get("bet_decision", "")))
        detail_rows.append(_policy_detail_row("participation_level", output.get("participation_level", "")))
        detail_rows.append(_policy_detail_row("ticket_summary", json.dumps(output.get("ticket_summary", {}), ensure_ascii=False), code=True))
        detail_rows.append(_policy_detail_row("notes", json.dumps(meta, ensure_ascii=False), code=True))
        budget_sections.append(
            '<section class="policy-panel">'
            f'<div class="policy-panel-head"><h3>{html.escape(panel_title)}</h3><div class="policy-meta-row">{"".join(header_tags)}{summary_tags}</div></div>'
            f"{bankroll_cards}"
            f"{text_cards}"
            f"{mark_block}"
            f'<section class="policy-horse-grid">{horse_groups}</section>'
            f"{ticket_block}"
            f'<section class="policy-detail-grid">{"".join([row for row in detail_rows if row])}</section>'
            "</section>"
        )
    return "".join(budget_sections)
def build_policy_workspace_html(payloads, *, to_float):
    blocks = []
    for payload in list(payloads or []):
        block = build_policy_html(payload, to_float=to_float)
        if block:
            blocks.append(block)
    return "".join(blocks)

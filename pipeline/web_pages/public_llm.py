from __future__ import annotations

import html
import re
from datetime import datetime, timedelta

from public_share_copy import PUBLIC_SHARE_TICKETS_LABEL
from surface_scope import normalize_scope_key
from web_report.helpers import (
    LLM_BATTLE_LABELS,
    LLM_BATTLE_ORDER,
    LLM_REPORT_SCOPE_KEYS,
    format_race_label,
    format_marks_text,
    format_percent_text,
    payload_run_id,
    policy_marks_map,
    policy_primary_budget,
    policy_primary_output,
    public_scope_label_ja,
    race_no_text,
    report_scope_key_for_row,
    run_date_key,
    safe_text,
)


BET_TYPE_TEXT_MAP = {
    "win": "単勝",
    "place": "複勝",
    "wide": "ワイド",
    "quinella": "馬連",
    "exacta": "馬単",
    "trio": "三連複",
    "trifecta": "三連単",
}

MARK_ORDER = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}


def public_status_meta_ja(ticket_summary, actual_names):
    status = safe_text((ticket_summary or {}).get("status", "")).lower()
    actual_ready = any(safe_text(name) for name in list(actual_names or []))
    if status == "settled":
        return "結果確定", "settled"
    if status == "pending":
        return "精算待ち", "pending"
    if actual_ready:
        return "結果確定", "result"
    return "確定待ち", "planned"


def public_yen_text(value):
    try:
        amount = int(value or 0)
    except (TypeError, ValueError):
        amount = 0
    sign = "-" if amount < 0 else ""
    return f"{sign}¥{abs(amount):,}"


def public_ticket_plan_text(ticket_rows):
    rows = list(ticket_rows or [])
    if not rows:
        return "買い目なし"
    lines = []
    for row in rows:
        bet_type_key = safe_text(row.get("bet_type")).lower()
        bet_type = BET_TYPE_TEXT_MAP.get(bet_type_key, safe_text(row.get("bet_type")) or "-")
        horse_no = safe_text(row.get("horse_no")) or "-"
        amount = public_yen_text(row.get("amount_yen") or row.get("stake_yen") or 0)
        lines.append(f"{bet_type} {horse_no} {amount}")
    return "\n".join(lines)


def _share_hashtag_race_label(run_row):
    venue = safe_text((run_row or {}).get("location")) or safe_text((run_row or {}).get("trigger_race"))
    race_no = race_no_text((run_row or {}).get("race_id")) or ""
    venue = re.sub(r"\s+", "", venue)
    if venue and not venue.endswith("競馬"):
        venue = f"{venue}競馬"
    if venue and race_no:
        return f"#{venue} {race_no}"
    if venue:
        return f"#{venue}"
    if race_no:
        return race_no
    return "#競馬AI"


def _share_ticket_lines(ticket_rows, *, to_int_or_none):
    lines = []
    for row in list(ticket_rows or []):
        bet_type_key = safe_text(row.get("bet_type")).lower()
        bet_type = BET_TYPE_TEXT_MAP.get(bet_type_key, safe_text(row.get("bet_type")) or "-")
        horse_no = safe_text(row.get("horse_no")) or "-"
        amount = to_int_or_none(row.get("amount_yen"))
        if amount is None:
            amount = to_int_or_none(row.get("stake_yen"))
        amount_text = f"¥{int(amount)}" if amount is not None else "-"
        lines.append(f"{bet_type} {horse_no} {amount_text}")
    return lines


def _share_marks_text(marks_map, *, to_int_or_none):
    if not marks_map:
        return "印なし"
    ordered = []
    for horse_no, symbol in dict(marks_map or {}).items():
        ordered.append((MARK_ORDER.get(safe_text(symbol), 99), to_int_or_none(horse_no) or 999, safe_text(horse_no), safe_text(symbol)))
    ordered.sort(key=lambda item: (item[0], item[1], item[2]))
    parts = [f"{symbol}{horse_no}" for _, _, horse_no, symbol in ordered if horse_no and symbol]
    return " ".join(parts) if parts else "印なし"


def build_public_share_text(
    run_row,
    engine,
    marks_map,
    ticket_rows,
    *,
    max_chars,
    share_detail_label,
    share_url,
    share_hashtag,
    to_int_or_none,
):
    del engine
    header = _share_hashtag_race_label(run_row)
    marks_text = _share_marks_text(marks_map, to_int_or_none=to_int_or_none)
    tail_lines = [share_detail_label, share_url, share_hashtag]
    base_lines = [header, "", marks_text, "", PUBLIC_SHARE_TICKETS_LABEL]
    ticket_lines = _share_ticket_lines(ticket_rows, to_int_or_none=to_int_or_none)
    lines = list(base_lines)
    for ticket_line in ticket_lines:
        candidate = "\n".join(lines + [ticket_line, "", *tail_lines])
        if len(candidate) > int(max_chars):
            break
        lines.append(ticket_line)
    if len(lines) == len(base_lines):
        placeholder = "買い目なし"
        candidate = "\n".join(base_lines + [placeholder, "", *tail_lines])
        if len(candidate) <= int(max_chars):
            lines.append(placeholder)
    text = "\n".join(lines + ["", *tail_lines])
    if len(text) <= int(max_chars):
        return text
    fallback_lines = [header, "", marks_text, "", *tail_lines]
    text = "\n".join(fallback_lines)
    if len(text) <= int(max_chars):
        return text
    tail_len = len("\n".join(["", *tail_lines]))
    compact_marks = marks_text[: max(0, int(max_chars) - len(header) - tail_len - 4)]
    compact_lines = [header, "", compact_marks or "印なし", "", *tail_lines]
    return "\n".join(compact_lines)


def public_result_triplet_text(actual_names):
    names = [safe_text(name) for name in list(actual_names or [])[:3] if safe_text(name)]
    if not names:
        return "結果未確定"
    return " / ".join(f"{idx + 1}着 {name}" for idx, name in enumerate(names))


def public_result_triplet_text_with_nos(actual_names, actual_horse_nos):
    names = [safe_text(name) for name in list(actual_names or [])[:3]]
    horse_nos = [safe_text(horse_no) for horse_no in list(actual_horse_nos or [])[:3]]
    entries = []
    max_len = max(len(names), len(horse_nos))
    for idx in range(max_len):
        name = names[idx] if idx < len(names) else ""
        horse_no = horse_nos[idx] if idx < len(horse_nos) else ""
        if not name and not horse_no:
            continue
        parts = [f"{idx + 1}着"]
        if horse_no:
            parts.append(horse_no)
        if name:
            parts.append(name)
        entries.append(" ".join(parts))
    if not entries:
        return public_result_triplet_text(actual_names)
    return " / ".join(entries)


def public_date_label(date_text, *, parse_run_date):
    parsed = parse_run_date(date_text)
    if not parsed:
        return str(date_text or "").strip()
    return f"{parsed.year}年{parsed.month}月{parsed.day}日"


def build_public_board_payload(
    date_text="",
    scope_key="",
    *,
    normalize_report_date_text,
    llm_today_scope_keys,
    resolve_llm_today_target_date,
    load_actual_result_map,
    load_combined_llm_report_runs,
    find_job_meta_for_run,
    load_policy_payloads,
    normalize_policy_engine,
    actual_result_snapshot,
    load_policy_run_ticket_rows,
    summarize_ticket_rows,
    format_triplet_text,
    marks_result_triplet,
    format_confidence_text,
    format_distance_label,
    public_all_time_roi_summary,
    public_trend_series,
    parse_run_date,
    share_detail_label,
    share_url,
    share_hashtag,
    share_max_chars,
    to_int_or_none,
):
    requested_date = normalize_report_date_text(date_text)
    scope_norm = normalize_scope_key(scope_key)
    scope_keys = llm_today_scope_keys(scope_norm)
    target_date, fallback_notice, combined_rows = resolve_llm_today_target_date(requested_date, scope_norm)
    actual_result_maps = {report_scope_key: load_actual_result_map(report_scope_key) for report_scope_key in scope_keys}

    runs = []
    for row in combined_rows:
        report_scope_key = report_scope_key_for_row(row, scope_norm)
        if report_scope_key not in scope_keys:
            continue
        if run_date_key(row) != target_date:
            continue
        runs.append(row)
    runs.sort(
        key=lambda row: (
            safe_text(row.get("race_date")),
            safe_text(row.get("location")),
            safe_text(row.get("race_id")),
            safe_text(row.get("timestamp")),
            safe_text(row.get("run_id")),
        )
    )

    summary_by_engine = {
        engine: {
            "engine": engine,
            "label": LLM_BATTLE_LABELS.get(engine, engine),
            "races": 0,
            "settled_races": 0,
            "pending_races": 0,
            "hit_races": 0,
            "ticket_count": 0,
            "stake_yen": 0,
            "payout_yen": 0,
            "profit_yen": 0,
        }
        for engine in LLM_BATTLE_ORDER
    }

    race_items = []
    for run_row in runs:
        run_id = safe_text(run_row.get("run_id"))
        report_scope_key = report_scope_key_for_row(run_row, scope_norm)
        job_meta = find_job_meta_for_run(report_scope_key, run_id, run_row)
        payload_map = {}
        for payload in load_policy_payloads(report_scope_key, run_id, run_row):
            engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
            if engine:
                payload_map[engine] = payload
        if not payload_map:
            continue

        actual_snapshot = actual_result_snapshot(report_scope_key, run_id, run_row, actual_result_maps.get(report_scope_key, {}))
        actual_names = list(actual_snapshot.get("actual_names", []) or [])
        actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])
        actual_text = public_result_triplet_text_with_nos(actual_names, actual_horse_nos)

        cards = []
        for engine in LLM_BATTLE_ORDER:
            payload = payload_map.get(engine)
            if not payload:
                continue
            output = policy_primary_output(payload)
            payload_input = dict((payload or {}).get("input", {}) or {})
            ai_summary = dict(payload_input.get("ai", {}) or {})
            marks_map = policy_marks_map(payload)
            ticket_run_id = payload_run_id(payload, run_id)
            ticket_rows = load_policy_run_ticket_rows(ticket_run_id, policy_engine=engine) or list(
                policy_primary_budget(payload).get("tickets", []) or []
            )
            ticket_summary = summarize_ticket_rows(ticket_rows)
            status_label, status_tone = public_status_meta_ja(ticket_summary, actual_names)
            result_triplet = format_triplet_text(marks_result_triplet(marks_map, actual_horse_nos))
            result_triplet_text = result_triplet if any(safe_text(x) for x in actual_horse_nos) else "結果未確定"
            decision_text = safe_text(output.get("bet_decision")) or "-"
            confidence_value = ai_summary.get("confidence_score", "")
            confidence_text = format_confidence_text(confidence_value)

            stats = summary_by_engine[engine]
            stats["races"] += 1
            stats["ticket_count"] += int(ticket_summary.get("ticket_count", 0) or 0)
            stats["stake_yen"] += int(ticket_summary.get("stake_yen", 0) or 0)
            stats["payout_yen"] += int(ticket_summary.get("payout_yen", 0) or 0)
            stats["profit_yen"] += int(ticket_summary.get("profit_yen", 0) or 0)
            if ticket_summary.get("status") == "settled":
                stats["settled_races"] += 1
            elif ticket_summary.get("status") == "pending":
                stats["pending_races"] += 1
            if int(ticket_summary.get("hit_count", 0) or 0) > 0:
                stats["hit_races"] += 1

            cards.append(
                {
                    "engine": engine,
                    "label": LLM_BATTLE_LABELS.get(engine, engine),
                    "decision_text": decision_text,
                    "status_label": status_label,
                    "status_tone": status_tone,
                    "ticket_plan_text": public_ticket_plan_text(ticket_rows),
                    "marks_text": format_marks_text(marks_map),
                    "result_triplet_text": result_triplet_text,
                    "ticket_count": int(ticket_summary.get("ticket_count", 0) or 0),
                    "stake_yen": int(ticket_summary.get("stake_yen", 0) or 0),
                    "payout_yen": int(ticket_summary.get("payout_yen", 0) or 0),
                    "profit_yen": int(ticket_summary.get("profit_yen", 0) or 0),
                    "hit_count": int(ticket_summary.get("hit_count", 0) or 0),
                    "roi_text": format_percent_text(ticket_summary.get("roi", "")),
                    "confidence_text": confidence_text,
                    "confidence_value": confidence_value,
                    "share_text": build_public_share_text(
                        run_row,
                        engine,
                        marks_map,
                        ticket_rows,
                        max_chars=share_max_chars,
                        share_detail_label=share_detail_label,
                        share_url=share_url,
                        share_hashtag=share_hashtag,
                        to_int_or_none=to_int_or_none,
                    ),
                }
            )

        if not cards:
            continue

        race_items.append(
            {
                "scope_key": report_scope_key,
                "scope_label": public_scope_label_ja(report_scope_key),
                "race_title": format_race_label(run_row),
                "date_label": public_date_label(safe_text(run_row.get("race_date")) or target_date, parse_run_date=parse_run_date),
                "actual_text": actual_text,
                "location": safe_text(job_meta.get("location")) or safe_text(run_row.get("location")),
                "scheduled_off_time": safe_text(job_meta.get("scheduled_off_time")),
                "distance_label": format_distance_label(job_meta.get("target_distance")),
                "track_condition": safe_text(job_meta.get("target_track_condition")) or "良",
                "run_id": run_id or "-",
                "cards": cards,
            }
        )

    summary_cards = []
    total_stake = 0
    total_payout = 0
    total_profit = 0
    total_settled = 0
    total_pending = 0
    visible_engine_count = 0
    for engine in LLM_BATTLE_ORDER:
        stats = summary_by_engine[engine]
        if int(stats.get("races", 0) or 0) <= 0:
            continue
        visible_engine_count += 1
        total_stake += int(stats.get("stake_yen", 0) or 0)
        total_payout += int(stats.get("payout_yen", 0) or 0)
        total_profit += int(stats.get("profit_yen", 0) or 0)
        total_settled += int(stats.get("settled_races", 0) or 0)
        total_pending += int(stats.get("pending_races", 0) or 0)
        roi_value = ""
        if int(stats.get("stake_yen", 0) or 0) > 0:
            roi_value = round(float(stats["payout_yen"]) / float(stats["stake_yen"]), 4)
        summary_cards.append(
            {
                "engine": engine,
                "label": stats["label"],
                "races": int(stats["races"]),
                "settled_races": int(stats["settled_races"]),
                "pending_races": int(stats["pending_races"]),
                "hit_races": int(stats["hit_races"]),
                "ticket_count": int(stats["ticket_count"]),
                "stake_yen": int(stats["stake_yen"]),
                "payout_yen": int(stats["payout_yen"]),
                "profit_yen": int(stats["profit_yen"]),
                "roi_text": format_percent_text(roi_value),
            }
        )

    total_roi_value = ""
    if total_stake > 0:
        total_roi_value = round(float(total_payout) / float(total_stake), 4)
    generated_at_label = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M JST")
    ranked_summary_cards = sorted(
        list(summary_cards),
        key=lambda item: (
            1 if safe_text(item.get("roi_text")) in ("", "-") else 0,
            -(float(str(item.get("roi_text", "0")).replace("%", "") or 0.0) if safe_text(item.get("roi_text")) not in ("", "-") else -9999.0),
            -int(item.get("profit_yen", 0) or 0),
        ),
    )
    hero_races = sorted(
        list(race_items),
        key=lambda item: (
            -(int(re.search(r"(\d+)R", safe_text(item.get("race_title"))).group(1)) if re.search(r"(\d+)R", safe_text(item.get("race_title"))) else 0),
            safe_text(item.get("race_title")),
        ),
    )
    lead_race = dict(hero_races[0]) if hero_races else {}

    return {
        "requested_date": requested_date,
        "target_date": target_date,
        "target_date_label": public_date_label(target_date, parse_run_date=parse_run_date),
        "generated_at_label": generated_at_label,
        "scope_key": scope_norm or "",
        "scope_options": [{"value": "", "label": "全場"}]
        + [{"value": key, "label": public_scope_label_ja(key)} for key in LLM_REPORT_SCOPE_KEYS],
        "fallback_notice": fallback_notice,
        "all_time_roi": public_all_time_roi_summary(),
        "trend": public_trend_series(days=10),
        "totals": {
            "race_count": len(race_items),
            "engine_count": visible_engine_count,
            "stake_yen": total_stake,
            "payout_yen": total_payout,
            "profit_yen": total_profit,
            "settled_count": total_settled,
            "pending_count": total_pending,
            "roi_text": format_percent_text(total_roi_value),
        },
        "summary_cards": summary_cards,
        "hero": {
            "lead_race": lead_race,
            "leader": ranked_summary_cards[0] if ranked_summary_cards else {},
        },
        "races": race_items,
    }


def build_public_llm_page(
    *,
    payload,
    prefix_public_html_routes,
):
    payload = dict(payload or {})
    totals = dict(payload.get("totals", {}) or {})
    summary_cards = list(payload.get("summary_cards", []) or [])
    races = list(payload.get("races", []) or [])
    target_date_label = safe_text(payload.get("target_date_label")) or "-"
    fallback_notice = safe_text(payload.get("fallback_notice"))

    summary_html = []
    for item in summary_cards:
        summary_html.append(
            f"""
            <article class="card">
              <h3>{html.escape(safe_text(item.get('label')) or '-')}</h3>
              <p>利益 {html.escape(public_yen_text(item.get('profit_yen', 0)))}</p>
              <p>回収率 {html.escape(safe_text(item.get('roi_text')) or '-')}</p>
              <p>的中 {int(item.get('hit_races', 0) or 0)} / {int(item.get('races', 0) or 0)}</p>
            </article>
            """
        )

    race_html = []
    for race in races:
        cards_html = []
        for card in list(race.get("cards", []) or []):
            cards_html.append(
                f"""
                <article class="card">
                  <h4>{html.escape(safe_text(card.get('label')) or '-')}</h4>
                  <p>判断 {html.escape(safe_text(card.get('decision_text')) or '-')}</p>
                  <p>印 {html.escape(safe_text(card.get('marks_text')) or '-')}</p>
                  <p style="white-space: pre-wrap;">{html.escape(safe_text(card.get('ticket_plan_text')) or '-')}</p>
                </article>
                """
            )
        race_html.append(
            f"""
            <section class="race">
              <div class="race-head">
                <div>
                  <div class="eyebrow">{html.escape(safe_text(race.get('scope_label')) or '-')}</div>
                  <h2>{html.escape(safe_text(race.get('race_title')) or '-')}</h2>
                  <p>{html.escape(safe_text(race.get('date_label')) or '-')}</p>
                </div>
                <div class="actual">{html.escape(safe_text(race.get('actual_text')) or '結果未確定')}</div>
              </div>
              <div class="grid">{''.join(cards_html)}</div>
            </section>
            """
        )

    notice_html = f'<section class="notice">{html.escape(fallback_notice)}</section>' if fallback_notice else ""
    html_text = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Public Board</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, sans-serif;
      background: #f6f1e8;
      color: #1f1b16;
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px 16px 40px;
      display: grid;
      gap: 20px;
    }}
    .hero, .notice, .race, .card {{
      background: #fffdf8;
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 18px;
    }}
    .hero, .notice, .race {{
      padding: 20px;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #0f5f4d;
      font-weight: 700;
    }}
    .hero h1, .race h2, .card h3, .card h4 {{
      margin: 0;
    }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .summary {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    }}
    .card {{
      padding: 16px;
    }}
    .race-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }}
    .actual {{
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(15,95,77,0.08);
      color: #0f5f4d;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="eyebrow">Legacy Public LLM Page</div>
      <h1>{html.escape(target_date_label)}</h1>
      <p>レース数 {int(totals.get('race_count', 0) or 0)} / 利益 {html.escape(public_yen_text(totals.get('profit_yen', 0)))}</p>
      <p>投資 {html.escape(public_yen_text(totals.get('stake_yen', 0)))} / 払戻 {html.escape(public_yen_text(totals.get('payout_yen', 0)))} / 回収率 {html.escape(safe_text(totals.get('roi_text')) or '-')}</p>
    </section>
    {notice_html}
    <section class="summary">{''.join(summary_html)}</section>
    {''.join(race_html)}
  </main>
</body>
</html>"""
    return prefix_public_html_routes(html_text)

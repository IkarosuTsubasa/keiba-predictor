from __future__ import annotations

import html
from datetime import timedelta

from web_report.helpers import (
    LLM_BATTLE_LABELS,
    LLM_BATTLE_ORDER,
    LLM_BATTLE_SHORT_LABELS,
    LLM_NOTE_LABELS,
    LLM_REPORT_SCOPE_KEYS,
    build_battle_title,
    format_jp_date_text,
    format_jp_date_value,
    format_marks_text,
    format_percent_text,
    format_race_label,
    format_ticket_plan_text,
    payload_run_id,
    policy_marks_map,
    policy_primary_budget,
    policy_primary_output,
    report_scope_key_for_row,
    run_date_key,
    safe_text,
)


def public_all_time_roi_summary(base_dir, *, ledger_path, load_rows):
    cards = []
    total_stake = 0
    total_payout = 0
    total_profit = 0
    for engine in LLM_BATTLE_ORDER:
        rows = load_rows(ledger_path(base_dir, policy_engine=engine))
        stake_yen = 0
        payout_yen = 0
        profit_yen = 0
        for row in rows:
            if safe_text(row.get("status")).lower() != "settled":
                continue
            stake_yen += int(row.get("stake_yen", 0) or 0)
            payout_yen += int(row.get("payout_yen", 0) or 0)
            profit_yen += int(row.get("profit_yen", 0) or 0)
        if stake_yen <= 0:
            continue
        total_stake += stake_yen
        total_payout += payout_yen
        total_profit += profit_yen
        cards.append(
            {
                "engine": engine,
                "label": LLM_BATTLE_LABELS.get(engine, engine),
                "stake_yen": stake_yen,
                "payout_yen": payout_yen,
                "profit_yen": profit_yen,
                "roi_text": format_percent_text(round(float(payout_yen) / float(stake_yen), 4)),
            }
        )
    total_roi_text = ""
    if total_stake > 0:
        total_roi_text = format_percent_text(round(float(total_payout) / float(total_stake), 4))
    return {
        "cards": cards,
        "totals": {
            "stake_yen": total_stake,
            "payout_yen": total_payout,
            "profit_yen": total_profit,
            "roi_text": total_roi_text,
        },
    }


def _policy_primary_choice(output):
    marks = list((output or {}).get("marks", []) or [])
    symbol_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}
    best = None
    for row in marks:
        symbol = safe_text(row.get("symbol"))
        horse_no = safe_text(row.get("horse_no"))
        if not symbol or not horse_no:
            continue
        score = symbol_order.get(symbol, 99)
        if best is None or score < best[0]:
            best = (score, horse_no)
    if best is not None:
        return best[1]
    for horse_no in list((output or {}).get("key_horses", []) or []):
        text = safe_text(horse_no)
        if text:
            return text
    return ""


def _llm_agreement_summary(outputs):
    top_choices = []
    for output in list(outputs or []):
        horse_no = _policy_primary_choice(output)
        if horse_no:
            top_choices.append(horse_no)
    if not top_choices:
        return {"label": "中立", "score": 0.5}
    counts = {}
    for horse_no in top_choices:
        counts[horse_no] = counts.get(horse_no, 0) + 1
    max_count = max(counts.values()) if counts else 0
    unique_count = len(counts)
    if max_count >= 4:
        return {"label": "高一致", "score": 1.0}
    if max_count == 3:
        return {"label": "やや一致", "score": 0.8}
    if max_count == 2 and unique_count <= 2:
        return {"label": "中立", "score": 0.5}
    if max_count == 2:
        return {"label": "やや分散", "score": 0.3}
    return {"label": "分散", "score": 0.0}


def _difficulty_summary(outputs, agreement_score):
    outputs = list(outputs or [])
    if not outputs:
        return "★★★☆☆"
    score = 2.0
    mixed_count = 0
    low_conf_count = 0
    low_stability_count = 0
    one_shot_count = 0
    strong_favorite_count = 0
    place_focus_count = 0
    risk_total = 0.0
    for output in outputs:
        reason_codes = {
            str(item or "").strip() for item in list(output.get("reason_codes", []) or []) if str(item or "").strip()
        }
        if "MIXED_FIELD" in reason_codes:
            mixed_count += 1
        if "LOW_CONFIDENCE" in reason_codes:
            low_conf_count += 1
        if "LOW_STABILITY" in reason_codes:
            low_stability_count += 1
        if "HIGH_ODDS_ONE_SHOT" in reason_codes:
            one_shot_count += 1
        if "STRONG_FAVORITE" in reason_codes:
            strong_favorite_count += 1
        if "PLACE_FOCUS" in reason_codes:
            place_focus_count += 1
        risk_tilt = safe_text(output.get("risk_tilt")).lower()
        if risk_tilt == "high":
            risk_total += 0.8
        elif risk_tilt == "medium":
            risk_total += 0.4
        elif risk_tilt == "low":
            risk_total += 0.1
    total = float(len(outputs))
    score += (mixed_count / total) * 2.0
    score += (low_conf_count / total) * 1.0
    score += (low_stability_count / total) * 0.8
    score += (one_shot_count / total) * 0.5
    score -= (strong_favorite_count / total) * 1.1
    score -= (place_focus_count / total) * 0.3
    score += risk_total / total
    score += (1.0 - float(agreement_score or 0.0)) * 1.6
    star_count = max(1, min(5, int(round(score))))
    return "★" * star_count + "☆" * (5 - star_count)


def build_llm_battle_bundle(
    scope_key,
    run_id,
    run_row,
    payloads,
    actual_result_map,
    *,
    normalize_policy_engine,
    actual_result_snapshot,
    load_policy_run_ticket_rows,
    summarize_ticket_rows,
    marks_result_triplet,
    format_triplet_text,
):
    payload_map = {}
    for payload in list(payloads or []):
        engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
        if engine:
            payload_map[engine] = payload
    if not payload_map:
        return {"html": "", "note_text": ""}

    actual_snapshot = actual_result_snapshot(scope_key, run_id, run_row, actual_result_map)
    actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])
    race_label = format_race_label(run_row)
    battle_title = build_battle_title(run_row)
    outputs = []
    cards = []
    premium_lines = ["第1回 いかいもAI競馬対決杯", ""]
    note_lines = [battle_title, "", f"レース：{race_label}"]

    for engine in LLM_BATTLE_ORDER:
        payload = payload_map.get(engine)
        if not payload:
            continue
        output = policy_primary_output(payload)
        outputs.append(output)
        marks_map = policy_marks_map(payload)
        ticket_rows = load_policy_run_ticket_rows(run_id, policy_engine=engine) or list(
            policy_primary_budget(payload).get("tickets", []) or []
        )
        ticket_summary = summarize_ticket_rows(ticket_rows)
        result_triplet = marks_result_triplet(marks_map, actual_horse_nos)
        marks_text = format_marks_text(marks_map)
        ticket_plan_text = format_ticket_plan_text(ticket_rows, output)
        tendency_text = safe_text(output.get("bet_tendency_ja")) or "情報なし"
        strategy_text = safe_text(output.get("strategy_text_ja")) or "情報なし"
        meta_tags = [
            f"成績 {ticket_summary['status']}",
            f"投資 {ticket_summary['stake_yen']}円" if ticket_summary["stake_yen"] else "未購入",
        ]
        if ticket_summary["roi"] != "":
            meta_tags.append(f"ROI {ticket_summary['roi']:.4f}")
        cards.append(
            {
                "engine": engine,
                "label": LLM_BATTLE_LABELS.get(engine, engine),
                "marks_text": marks_text,
                "ticket_plan_text": ticket_plan_text,
                "tendency_text": tendency_text,
                "strategy_text": strategy_text,
                "result_triplet_text": format_triplet_text(result_triplet),
                "meta_tags": meta_tags,
            }
        )
        premium_lines.extend(
            [
                f"{LLM_NOTE_LABELS.get(engine, LLM_BATTLE_LABELS.get(engine, engine))}",
                strategy_text,
                "",
            ]
        )

    agreement = _llm_agreement_summary(outputs)
    difficulty = _difficulty_summary(outputs, agreement.get("score", 0.5))
    note_lines.extend(
        [
            f"難易度：{difficulty}",
            f"一致度：{agreement.get('label', '中立')}",
            "",
        ]
    )

    card_html = []
    premium_html = []
    for card in cards:
        tag_html = "".join(
            f'<span class="battle-meta-chip">{html.escape(tag)}</span>' for tag in list(card.get("meta_tags", []) or [])
        )
        card_html.append(
            f"""
            <article class="battle-card">
              <div class="battle-card-head">
                <div>
                  <div class="eyebrow">LLM Battle</div>
                  <h3>{html.escape(card['label'])}</h3>
                </div>
                <div class="battle-meta-row">{tag_html}</div>
              </div>
              <div class="battle-mark-band">
                <span class="battle-band-label">印</span>
                <strong>{html.escape(card['marks_text'])}</strong>
              </div>
              <div class="battle-public-grid">
                <section class="battle-copy-card">
                  <div class="policy-label">買い目</div>
                  <p>{html.escape(card['ticket_plan_text'])}</p>
                </section>
                <section class="battle-copy-card">
                  <div class="policy-label">買い目傾向</div>
                  <p>{html.escape(card['tendency_text'])}</p>
                </section>
              </div>
              <div class="battle-result-row">
                <span class="policy-label">実着順との対応</span>
                <code>{html.escape(card['result_triplet_text'])}</code>
              </div>
            </article>
            """
        )
        note_lines.extend(
            [
                f"{LLM_NOTE_LABELS.get(card['engine'], card['label'])}",
                f"印：{card['marks_text']}",
                "",
                "買い目",
                "",
                card["ticket_plan_text"],
                "",
                f"買い目傾向：{card['tendency_text']}",
                "",
            ]
        )
        premium_html.append(
            f"""
            <section class="battle-copy-card">
              <div class="policy-label">{html.escape(card['label'])} Strategy</div>
              <p>{html.escape(card['strategy_text'])}</p>
            </section>
            """
        )
    summary_html = (
        '<section class="panel battle-hero-panel">'
        '<div class="panel-title-row">'
        f'<div><div class="eyebrow">Note Layout</div><h2>{html.escape(battle_title)}</h2></div>'
        f'<span class="section-chip">{html.escape(race_label)}</span>'
        "</div>"
        '<div class="battle-overview-grid">'
        f'<section class="battle-overview-card"><span class="battle-band-label">レース</span><strong>{html.escape(race_label)}</strong></section>'
        f'<section class="battle-overview-card"><span class="battle-band-label">難易度</span><strong>{html.escape(difficulty)}</strong></section>'
        f'<section class="battle-overview-card"><span class="battle-band-label">一致度</span><strong>{html.escape(str(agreement.get("label", "中立")))}</strong></section>'
        "</div>"
        '<p class="helper-text">難易度は各AIの信頼度・荒れ度・一致度から集計した目安です。一致度は 4 LLM の本命がどれだけ揃っているかを見ています。</p>'
        f'<div class="battle-grid">{"".join(card_html)}</div>'
        '<section class="panel battle-hero-panel" style="margin-top:14px;">'
        '<div class="panel-title-row">'
        '<div><div class="eyebrow">Paid Block</div><h2>AI陣営の考え方</h2></div>'
        '<span class="section-chip">note footer</span>'
        "</div>"
        '<p class="helper-text">このブロックは note 記事の有料部分に掲載する想定で、4 LLM の Strategy をまとめています。</p>'
        f'<div class="battle-public-grid">{"".join(premium_html)}</div>'
        "</section>"
        "</section>"
    )
    note_text = "\n".join(line for line in note_lines if line is not None).strip()
    premium_text = "\n".join(line for line in premium_lines if line is not None).strip()
    if premium_text:
        note_text = f"{note_text}\n\n{premium_text}".strip()
    return {"html": summary_html, "note_text": note_text}


def build_llm_daily_report_bundle(
    scope_key,
    current_run_row,
    _actual_result_map_unused,
    *,
    load_actual_result_map,
    load_combined_llm_report_runs,
    load_policy_payloads,
    normalize_policy_engine,
    actual_result_snapshot,
    load_policy_run_ticket_rows,
    summarize_ticket_rows,
    marks_result_triplet,
    format_triplet_text,
    ratio_text,
    percent_text_from_ratio,
    build_table_html,
):
    target_date = run_date_key(current_run_row)
    if not target_date:
        return {"html": "", "text": ""}

    actual_result_maps = {report_scope_key: load_actual_result_map(report_scope_key) for report_scope_key in LLM_REPORT_SCOPE_KEYS}
    runs = [row for row in load_combined_llm_report_runs() if run_date_key(row) == target_date]
    if not runs:
        return {"html": "", "text": ""}

    runs.sort(key=lambda row: (safe_text(row.get("race_id")), safe_text(row.get("timestamp")), safe_text(row.get("run_id"))))
    metric_rows = {
        engine: {
            "model": LLM_BATTLE_LABELS.get(engine, engine),
            "runs": 0,
            "race_hit_count": 0,
            "ticket_hit_count": 0,
            "ticket_count": 0,
            "honmei_hit_count": 0,
            "mark_hit_count": 0,
            "mark_total_count": 0,
            "stake_yen": 0,
            "payout_yen": 0,
            "profit_yen": 0,
        }
        for engine in LLM_BATTLE_ORDER
    }
    race_rows = []
    report_lines = [f"{target_date} 第1回 いかいもAI競馬対決杯 日報", ""]

    for run_row in runs:
        run_id = safe_text(run_row.get("run_id"))
        report_scope_key = report_scope_key_for_row(run_row, scope_key)
        payload_map = {}
        for payload in load_policy_payloads(report_scope_key, run_id, run_row):
            engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
            if engine:
                payload_map[engine] = payload
        if not payload_map:
            continue
        race_run_id = run_id
        if not race_run_id:
            for payload in payload_map.values():
                race_run_id = payload_run_id(payload, race_run_id)
                if race_run_id:
                    break
        if not race_run_id:
            continue
        actual_snapshot = actual_result_snapshot(report_scope_key, race_run_id, run_row, actual_result_maps.get(report_scope_key, {}))
        actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])
        race_label = format_race_label(run_row)
        race_row = {"race": race_label}
        report_lines.append(race_label)
        for engine in LLM_BATTLE_ORDER:
            payload = payload_map.get(engine)
            triplet_text = "- - -"
            report_line_text = triplet_text
            if payload:
                output = policy_primary_output(payload)
                marks_map = policy_marks_map(payload)
                triplet = marks_result_triplet(marks_map, actual_horse_nos)
                triplet_text = format_triplet_text(triplet)
                bet_decision = safe_text(output.get("bet_decision")).lower()
                ticket_run_id = payload_run_id(payload, race_run_id)
                ticket_rows = load_policy_run_ticket_rows(ticket_run_id, policy_engine=engine)
                ticket_summary = summarize_ticket_rows(ticket_rows)
                if bet_decision == "no_bet" or int(ticket_summary.get("ticket_count", 0) or 0) <= 0:
                    report_line_text = "見送り"
                else:
                    recovery_rate_text = percent_text_from_ratio(ticket_summary.get("roi", ""))
                    report_line_text = triplet_text if recovery_rate_text == "-" else f"{triplet_text} 回収率{recovery_rate_text}"
                stats = metric_rows[engine]
                stats["runs"] += 1
                stats["race_hit_count"] += 1 if ticket_summary["hit_count"] > 0 else 0
                stats["ticket_hit_count"] += int(ticket_summary["hit_count"])
                stats["ticket_count"] += int(ticket_summary["ticket_count"])
                stats["stake_yen"] += int(ticket_summary["stake_yen"])
                stats["payout_yen"] += int(ticket_summary["payout_yen"])
                stats["profit_yen"] += int(ticket_summary["profit_yen"])
                if actual_horse_nos:
                    if triplet and triplet[0] == "◎":
                        stats["honmei_hit_count"] += 1
                    actual_set = {horse_no for horse_no in actual_horse_nos if horse_no}
                    stats["mark_hit_count"] += len(set(marks_map.keys()) & actual_set)
                    stats["mark_total_count"] += len(set(marks_map.keys()))
            race_row[LLM_BATTLE_SHORT_LABELS.get(engine, engine)] = triplet_text
            report_lines.append(f"{LLM_BATTLE_SHORT_LABELS.get(engine, engine)} {report_line_text}")
        race_rows.append(race_row)
        report_lines.append("")

    summary_rows = []
    for engine in LLM_BATTLE_ORDER:
        stats = metric_rows[engine]
        if stats["runs"] <= 0:
            continue
        stake_yen = int(stats["stake_yen"])
        payout_yen = int(stats["payout_yen"])
        recovery_rate_value = round(payout_yen / stake_yen, 4) if stake_yen > 0 else ""
        stats["recovery_rate_value"] = recovery_rate_value
        stats["honmei_hit_rate"] = ratio_text(stats["honmei_hit_count"], stats["runs"])
        stats["mark_hit_rate"] = ratio_text(stats["mark_hit_count"], stats["mark_total_count"])
        stats["race_hit_rate"] = ratio_text(stats["race_hit_count"], stats["runs"])
        stats["ticket_hit_rate"] = ratio_text(stats["ticket_hit_count"], stats["ticket_count"])
        summary_rows.append(
            {
                "model": stats["model"],
                "レース数": stats["runs"],
                "的中": f"{stats['runs']}レース中 {stats['race_hit_count']}レース的中",
                "的中率": stats["race_hit_rate"],
                "本命的中率": stats["honmei_hit_rate"],
                "印的中率": stats["mark_hit_rate"],
                "honmei_hit_rate_value": round(float(stats["honmei_hit_count"]) / float(stats["runs"]), 6) if stats["runs"] else "",
                "mark_hit_rate_value": round(float(stats["mark_hit_count"]) / float(stats["mark_total_count"]), 6)
                if stats["mark_total_count"]
                else "",
                "hit_rate_value": round(float(stats["race_hit_count"]) / float(stats["runs"]), 6) if stats["runs"] else "",
                "投資額": stake_yen,
                "払戻額": payout_yen,
                "収支": int(stats["profit_yen"]),
                "回収率": percent_text_from_ratio(recovery_rate_value),
                "recovery_rate_value": recovery_rate_value,
            }
        )

    if not race_rows and not summary_rows:
        return {"html": "", "text": ""}

    report_lines.append("【日次集計】")
    report_lines.append("")
    for row in summary_rows:
        report_lines.append(str(row.get("model", "-") or "-"))
        report_lines.append(
            "{hit_summary} 的中率{hit_rate} 投資{stake_yen}円 払戻{payout_yen}円 収支{profit_yen}円 回収率{recovery_rate} 本命的中率{honmei_hit_rate} 印的中率{mark_hit_rate}".format(
                hit_summary=row.get("的中", "-"),
                hit_rate=row.get("的中率", "-"),
                stake_yen=row.get("投資額", 0),
                payout_yen=row.get("払戻額", 0),
                profit_yen=row.get("収支", 0),
                recovery_rate=row.get("回収率", "-"),
                honmei_hit_rate=row.get("本命的中率", "-"),
                mark_hit_rate=row.get("印的中率", "-"),
            )
        )
        report_lines.append("")

    def _leader_rows(metric_key, title):
        ordered = sorted(
            summary_rows,
            key=lambda row: (
                float(row.get(metric_key)) if row.get(metric_key, "") not in ("", None) else -1.0,
                float(safe_text(row.get("収支")) or 0.0),
            ),
            reverse=True,
        )
        out = []
        for idx, row in enumerate(ordered[:4], start=1):
            value = row.get(metric_key, "")
            if metric_key in ("recovery_rate_value", "honmei_hit_rate_value", "mark_hit_rate_value", "hit_rate_value"):
                value_text = percent_text_from_ratio(value)
            else:
                value_text = safe_text(value) or "-"
            out.append({"順位": idx, "model": row.get("model", ""), "value": value_text})
        report_lines.append("")
        report_lines.append(f"【{title}】")
        for item in out:
            report_lines.append(f"{item['順位']}. {item['model']} {item['value']}")
        return out

    leaderboard_hit = _leader_rows("hit_rate_value", "的中率ランキング")
    leaderboard_honmei = _leader_rows("honmei_hit_rate_value", "本命的中率ランキング")
    leaderboard_recovery = _leader_rows("recovery_rate_value", "回収率ランキング")
    leaderboard_mark = _leader_rows("mark_hit_rate_value", "印的中率ランキング")

    race_table_html = build_table_html(
        race_rows,
        ["race", "chatgpt", "gemini", "deepseek", "grok"],
        "各レース対戦表",
    )
    summary_table_html = build_table_html(
        summary_rows,
        ["model", "レース数", "的中", "的中率", "本命的中率", "印的中率", "投資額", "払戻額", "収支", "回収率"],
        "日次集計",
    )
    leaderboard_html = "".join(
        [
            build_table_html(leaderboard_hit, ["順位", "model", "value"], "的中率ランキング"),
            build_table_html(leaderboard_honmei, ["順位", "model", "value"], "本命的中率ランキング"),
            build_table_html(leaderboard_recovery, ["順位", "model", "value"], "回収率ランキング"),
            build_table_html(leaderboard_mark, ["順位", "model", "value"], "印的中率ランキング"),
        ]
    )
    header_html = (
        '<section class="panel battle-hero-panel">'
        '<div class="panel-title-row">'
        '<div><div class="eyebrow">Daily Report</div><h2>LLM 日報</h2></div>'
        f'<span class="section-chip">{html.escape(target_date)}</span>'
        "</div>"
        '<p class="helper-text">同日・同条件の中央/地方レースを横断して、各モデルの的中と収支を見比べるための集計です。</p>'
        "</section>"
    )
    return {
        "html": f"{header_html}{race_table_html}{summary_table_html}{leaderboard_html}",
        "text": "\n".join(report_lines).strip(),
    }


def build_llm_weekly_report_bundle(
    scope_key,
    current_run_row,
    _actual_result_map_unused,
    *,
    load_actual_result_map,
    load_combined_llm_report_runs,
    load_policy_payloads,
    normalize_policy_engine,
    actual_result_snapshot,
    load_policy_run_ticket_rows,
    summarize_ticket_rows,
    marks_result_triplet,
    format_triplet_text,
    percent_text_from_ratio,
    parse_run_date,
    build_table_html,
):
    target_date = parse_run_date(run_date_key(current_run_row))
    if not target_date:
        return {"html": "", "text": ""}

    week_start = target_date - timedelta(days=target_date.weekday())
    week_end = week_start + timedelta(days=6)

    actual_result_maps = {report_scope_key: load_actual_result_map(report_scope_key) for report_scope_key in LLM_REPORT_SCOPE_KEYS}
    runs = []
    for row in load_combined_llm_report_runs():
        run_date = parse_run_date(run_date_key(row))
        if run_date is None:
            continue
        if week_start <= run_date <= week_end:
            runs.append((run_date, row))
    if not runs:
        return {"html": "", "text": ""}

    runs.sort(
        key=lambda item: (
            item[0].isoformat(),
            safe_text(item[1].get("race_id")),
            safe_text(item[1].get("timestamp")),
            safe_text(item[1].get("run_id")),
        )
    )
    metric_rows = {
        engine: {
            "model": LLM_BATTLE_LABELS.get(engine, engine),
            "runs": 0,
            "race_hit_count": 0,
            "stake_yen": 0,
            "payout_yen": 0,
            "profit_yen": 0,
        }
        for engine in LLM_BATTLE_ORDER
    }
    best_entry = None
    week_label = f"{format_jp_date_value(week_start)}〜{format_jp_date_value(week_end)}"
    report_lines = [f"【LLM 週報】{week_label} 中央・地方横断", ""]

    for run_date, run_row in runs:
        run_id = safe_text(run_row.get("run_id"))
        report_scope_key = report_scope_key_for_row(run_row, scope_key)
        payload_map = {}
        for payload in load_policy_payloads(report_scope_key, run_id, run_row):
            engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
            if engine:
                payload_map[engine] = payload
        if not payload_map:
            continue
        race_run_id = run_id
        if not race_run_id:
            for payload in payload_map.values():
                race_run_id = payload_run_id(payload, race_run_id)
                if race_run_id:
                    break
        if not race_run_id:
            continue
        actual_snapshot = actual_result_snapshot(report_scope_key, race_run_id, run_row, actual_result_maps.get(report_scope_key, {}))
        actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])
        race_label = format_race_label(run_row)
        race_date_text = format_jp_date_text(run_row) or format_jp_date_value(run_date)
        for engine in LLM_BATTLE_ORDER:
            payload = payload_map.get(engine)
            if not payload:
                continue
            marks_map = policy_marks_map(payload)
            triplet = marks_result_triplet(marks_map, actual_horse_nos)
            triplet_text = format_triplet_text(triplet)
            ticket_run_id = payload_run_id(payload, race_run_id)
            ticket_rows = load_policy_run_ticket_rows(ticket_run_id, policy_engine=engine)
            ticket_summary = summarize_ticket_rows(ticket_rows)
            stats = metric_rows[engine]
            stats["runs"] += 1
            stats["race_hit_count"] += 1 if ticket_summary["hit_count"] > 0 else 0
            stats["stake_yen"] += int(ticket_summary["stake_yen"])
            stats["payout_yen"] += int(ticket_summary["payout_yen"])
            stats["profit_yen"] += int(ticket_summary["profit_yen"])
            if ticket_summary["stake_yen"] > 0:
                recovery_rate_value = round(ticket_summary["payout_yen"] / ticket_summary["stake_yen"], 4)
                candidate = {
                    "model": LLM_BATTLE_LABELS.get(engine, engine),
                    "race": race_label,
                    "date": race_date_text,
                    "的中印": triplet_text,
                    "投資額": int(ticket_summary["stake_yen"]),
                    "払戻額": int(ticket_summary["payout_yen"]),
                    "収支": int(ticket_summary["profit_yen"]),
                    "回収率": percent_text_from_ratio(recovery_rate_value),
                    "recovery_rate_value": recovery_rate_value,
                    "status_rank": 1 if ticket_summary.get("status") == "settled" else 0,
                }
                if best_entry is None or (
                    candidate["recovery_rate_value"],
                    candidate["払戻額"],
                    candidate["status_rank"],
                ) > (
                    best_entry["recovery_rate_value"],
                    best_entry["払戻額"],
                    best_entry["status_rank"],
                ):
                    best_entry = candidate

    summary_rows = []
    for engine in LLM_BATTLE_ORDER:
        stats = metric_rows[engine]
        if stats["runs"] <= 0:
            continue
        stake_yen = int(stats["stake_yen"])
        payout_yen = int(stats["payout_yen"])
        profit_yen = int(stats["profit_yen"])
        recovery_rate_value = round(payout_yen / stake_yen, 4) if stake_yen > 0 else ""
        hit_rate_value = round(float(stats["race_hit_count"]) / float(stats["runs"]), 6) if stats["runs"] else ""
        summary_rows.append(
            {
                "model": stats["model"],
                "レース数": stats["runs"],
                "的中": f"{stats['runs']}レース中 {stats['race_hit_count']}レース的中",
                "的中率": percent_text_from_ratio(hit_rate_value),
                "投資額": stake_yen,
                "払戻額": payout_yen,
                "収支": profit_yen,
                "回収率": percent_text_from_ratio(recovery_rate_value),
                "stake_value": stake_yen,
                "payout_value": payout_yen,
                "profit_value": profit_yen,
                "hit_rate_value": hit_rate_value,
                "recovery_rate_value": recovery_rate_value,
            }
        )

    if not summary_rows:
        return {"html": "", "text": ""}

    report_lines.append("【週間集計】")
    for row in summary_rows:
        report_lines.append(
            "{model} {hit_summary} 的中率{hit_rate} 投資{stake_yen}円 払戻{payout_yen}円 収支{profit_yen}円 回収率{recovery_rate}".format(
                model=row.get("model", "-"),
                hit_summary=row.get("的中", "-"),
                hit_rate=row.get("的中率", "-"),
                stake_yen=row.get("投資額", 0),
                payout_yen=row.get("払戻額", 0),
                profit_yen=row.get("収支", 0),
                recovery_rate=row.get("回収率", "-"),
            )
        )

    def _leader_rows(metric_key, title, value_kind):
        ordered = sorted(
            summary_rows,
            key=lambda row: (
                float(row.get(metric_key)) if row.get(metric_key, "") not in ("", None) else -1.0,
                float(safe_text(row.get("収支")) or 0.0),
            ),
            reverse=True,
        )
        out = []
        for idx, row in enumerate(ordered[:4], start=1):
            value = row.get(metric_key, "")
            if value_kind == "percent":
                value_text = percent_text_from_ratio(value)
            else:
                value_text = f"{int(value)}円" if value not in ("", None) else "-"
            out.append({"順位": idx, "model": row.get("model", ""), "value": value_text})
        report_lines.append("")
        report_lines.append(f"【{title}】")
        for item in out:
            report_lines.append(f"{item['順位']}. {item['model']} {item['value']}")
        return out

    leaderboard_stake = _leader_rows("stake_value", "投資額ランキング", "yen")
    leaderboard_payout = _leader_rows("payout_value", "払戻額ランキング", "yen")
    leaderboard_profit = _leader_rows("profit_value", "収支ランキング", "yen")
    leaderboard_recovery = _leader_rows("recovery_rate_value", "回収率ランキング", "percent")
    leaderboard_hit = _leader_rows("hit_rate_value", "的中率ランキング", "percent")

    if best_entry:
        report_lines.extend(
            [
                "",
                "【今週のベスト】",
                "{model} {date} {race} 的中印 {triplet} 投資{stake_yen}円 払戻{payout_yen}円 収支{profit_yen}円 回収率{recovery_rate}".format(
                    model=best_entry.get("model", "-"),
                    date=best_entry.get("date", "-"),
                    race=best_entry.get("race", "-"),
                    triplet=best_entry.get("的中印", "- - -"),
                    stake_yen=best_entry.get("投資額", 0),
                    payout_yen=best_entry.get("払戻額", 0),
                    profit_yen=best_entry.get("収支", 0),
                    recovery_rate=best_entry.get("回収率", "-"),
                ),
            ]
        )

    summary_table_html = build_table_html(
        summary_rows,
        ["model", "レース数", "的中", "的中率", "投資額", "払戻額", "収支", "回収率"],
        "週間サマリー",
    )
    leaderboard_html = "".join(
        [
            build_table_html(leaderboard_stake, ["順位", "model", "value"], "投資額ランキング"),
            build_table_html(leaderboard_payout, ["順位", "model", "value"], "払戻額ランキング"),
            build_table_html(leaderboard_profit, ["順位", "model", "value"], "収支ランキング"),
            build_table_html(leaderboard_recovery, ["順位", "model", "value"], "回収率ランキング"),
            build_table_html(leaderboard_hit, ["順位", "model", "value"], "的中率ランキング"),
        ]
    )
    best_html = ""
    if best_entry:
        best_table_rows = [
            {
                "model": best_entry.get("model", "-"),
                "date": best_entry.get("date", "-"),
                "race": best_entry.get("race", "-"),
                "的中印": best_entry.get("的中印", "- - -"),
                "投資額": best_entry.get("投資額", 0),
                "払戻額": best_entry.get("払戻額", 0),
                "収支": best_entry.get("収支", 0),
                "回収率": best_entry.get("回収率", "-"),
            }
        ]
        best_html = (
            '<section class="panel battle-hero-panel">'
            '<div class="panel-title-row">'
            '<div><div class="eyebrow">Best Of Week</div><h2>今週のベスト</h2></div>'
            f'<span class="section-chip">{html.escape(best_entry.get("model", "-"))}</span>'
            "</div>"
            '<p class="helper-text">今週の中で回収率が最も高かったレースです。</p>'
            f"{build_table_html(best_table_rows, ['model', 'date', 'race', '的中印', '投資額', '払戻額', '収支', '回収率'], '今週のベスト')}"
            "</section>"
        )
    header_html = (
        '<section class="panel battle-hero-panel">'
        '<div class="panel-title-row">'
        '<div><div class="eyebrow">Weekly Report</div><h2>LLM 週報</h2></div>'
        f'<span class="section-chip">{html.escape(week_label)}</span>'
        "</div>"
        '<p class="helper-text">同週の中央・地方を横断して、投資・払戻・回収率・的中率をまとめています。</p>'
        "</section>"
    )
    return {
        "html": f"{header_html}{summary_table_html}{leaderboard_html}{best_html}",
        "text": "\n".join(report_lines).strip(),
    }

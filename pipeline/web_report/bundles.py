import html

from web_report.helpers import (
    LLM_BATTLE_LABELS,
    LLM_BATTLE_ORDER,
    build_battle_title,
    format_marks_text,
    format_percent_text,
    format_race_label,
    format_ticket_plan_text,
    policy_marks_map,
    policy_primary_budget,
    policy_primary_output,
    safe_text,
)

MARK_SYMBOL_ORDER = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}


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
    best = None
    for row in marks:
        symbol = safe_text(row.get("symbol"))
        horse_no = safe_text(row.get("horse_no"))
        if not symbol or not horse_no:
            continue
        score = MARK_SYMBOL_ORDER.get(symbol, 99)
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
        return {"label": "中", "score": 0.5}
    counts = {}
    for horse_no in top_choices:
        counts[horse_no] = counts.get(horse_no, 0) + 1
    max_count = max(counts.values()) if counts else 0
    unique_count = len(counts)
    if max_count >= 4:
        return {"label": "かなり高い", "score": 1.0}
    if max_count == 3:
        return {"label": "高い", "score": 0.8}
    if max_count == 2 and unique_count <= 2:
        return {"label": "中", "score": 0.5}
    if max_count == 2:
        return {"label": "やや低い", "score": 0.3}
    return {"label": "低い", "score": 0.0}


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
    note_lines = [battle_title, "", race_label]

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
        comment_text = str(output.get("comment", "") or "").strip()
        meta_tags = [f"状態 {ticket_summary['status']}"]
        if ticket_summary["stake_yen"]:
            meta_tags.append(f"投資 ¥{ticket_summary['stake_yen']}")
        else:
            meta_tags.append("投資なし")
        if ticket_summary["roi"] != "":
            meta_tags.append(f"ROI {ticket_summary['roi']:.4f}")
        cards.append(
            {
                "engine": engine,
                "label": LLM_BATTLE_LABELS.get(engine, engine),
                "marks_text": marks_text,
                "ticket_plan_text": ticket_plan_text,
                "comment_text": comment_text,
                "result_triplet_text": format_triplet_text(result_triplet),
                "meta_tags": meta_tags,
            }
        )

    agreement = _llm_agreement_summary(outputs)
    difficulty = _difficulty_summary(outputs, agreement.get("score", 0.5))
    note_lines.extend([f"難易度: {difficulty}", f"一致度: {agreement.get('label', '中')}", ""])

    card_html = []
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
                  <div class="policy-label">コメント</div>
                  <p>{html.escape(card['comment_text'] or '-')}</p>
                </section>
              </div>
              <div class="battle-result-row">
                <span class="policy-label">結果一致</span>
                <code>{html.escape(card['result_triplet_text'])}</code>
              </div>
            </article>
            """
        )
        note_lines.extend(
            [
                card["label"],
                f"印: {card['marks_text']}",
                "",
                "買い目",
                "",
                card["ticket_plan_text"],
                "",
                "コメント",
                "",
                card["comment_text"] or "-",
                "",
            ]
        )

    summary_html = (
        '<section class="panel battle-hero-panel">'
        '<div class="panel-title-row">'
        f'<div><div class="eyebrow">Battle</div><h2>{html.escape(battle_title)}</h2></div>'
        f'<span class="section-chip">{html.escape(race_label)}</span>'
        "</div>"
        '<div class="battle-overview-grid">'
        f'<section class="battle-overview-card"><span class="battle-band-label">対象</span><strong>{html.escape(race_label)}</strong></section>'
        f'<section class="battle-overview-card"><span class="battle-band-label">難易度</span><strong>{html.escape(difficulty)}</strong></section>'
        f'<section class="battle-overview-card"><span class="battle-band-label">一致度</span><strong>{html.escape(str(agreement.get("label", "中")))}</strong></section>'
        "</div>"
        '<p class="helper-text">各AIの印、買い目、結果一致を比較できます。</p>'
        f'<div class="battle-grid">{"".join(card_html)}</div>'
        "</section>"
    )
    note_text = "\n".join(line for line in note_lines if line is not None).strip()
    return {"html": summary_html, "note_text": note_text}

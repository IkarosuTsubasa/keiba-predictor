def llm_today_scope_label_ja(scope_key):
    mapping = {
        "central_dirt": "中央ダート",
        "central_turf": "中央芝",
        "local": "地方",
    }
    return mapping.get(str(scope_key or "").strip(), str(scope_key or "").strip() or "-")


def llm_today_status_meta_ja(ticket_summary, actual_names, *, safe_text):
    status = safe_text((ticket_summary or {}).get("status", "")).lower()
    actual_ready = any(safe_text(name) for name in list(actual_names or []))
    if status == "settled":
        return "確定", "settled"
    if status == "pending":
        return "進行中", "pending"
    if actual_ready:
        return "結果あり", "result"
    return "公開中", "planned"


def format_yen_text_ja(value):
    try:
        amount = int(value or 0)
    except (TypeError, ValueError):
        amount = 0
    sign = "-" if amount < 0 else ""
    return f"{sign}¥{abs(amount):,}"


def format_ticket_plan_text_ja(ticket_rows, *, safe_text):
    bet_type_map = {
        "win": "単勝",
        "place": "複勝",
        "wide": "ワイド",
        "quinella": "馬連",
        "exacta": "馬単",
        "trio": "三連複",
        "trifecta": "三連単",
    }
    rows = list(ticket_rows or [])
    if not rows:
        return "買い目なし"
    lines = []
    for row in rows:
        bet_type = bet_type_map.get(safe_text(row.get("bet_type")).lower(), safe_text(row.get("bet_type")) or "-")
        horse_no = safe_text(row.get("horse_no")) or "-"
        amount = format_yen_text_ja(row.get("amount_yen") or row.get("stake_yen") or 0)
        horse_name = safe_text(row.get("horse_name"))
        line = f"{bet_type} {horse_no} {amount}"
        if horse_name:
            line += f"  {horse_name}"
        lines.append(line)
    return "\n".join(lines)


def format_result_triplet_text_ja(actual_names, *, safe_text):
    names = [safe_text(name) for name in list(actual_names or [])[:3] if safe_text(name)]
    if not names:
        return "結果未取得"
    return " / ".join(f"{idx + 1}着 {name}" for idx, name in enumerate(names))

def resolve_llm_today_target_date(target_date="", scope_key="", *, ctx):
    normalize_scope_key = ctx["normalize_scope_key"]
    _llm_today_scope_keys = ctx["_llm_today_scope_keys"]
    _load_combined_llm_report_runs = ctx["_load_combined_llm_report_runs"]
    _report_scope_key_for_row = ctx["_report_scope_key_for_row"]
    _run_date_key = ctx["_run_date_key"]
    scope_norm = normalize_scope_key(scope_key)
    scope_keys = _llm_today_scope_keys(scope_norm)
    scoped_rows = []
    available_dates = []
    for row in _load_combined_llm_report_runs():
        report_scope_key = _report_scope_key_for_row(row, scope_norm)
        if report_scope_key not in scope_keys:
            continue
        scoped_rows.append(row)
        date_key = _run_date_key(row)
        if date_key:
            available_dates.append(date_key)
    available_dates = sorted(set(available_dates), reverse=True)
    if target_date and target_date in available_dates:
        return target_date, "", scoped_rows
    if available_dates:
        fallback_date = available_dates[0]
        if target_date and target_date != fallback_date:
            return (
                fallback_date,
                f"{target_date} のデータがないため、直近の {fallback_date} を表示しています。",
                scoped_rows,
            )
        return fallback_date, "", scoped_rows
    return target_date, "", scoped_rows

def build_llm_today_page(date_text="", scope_key="", *, ctx):
    _normalize_report_date_text = ctx["_normalize_report_date_text"]
    normalize_scope_key = ctx["normalize_scope_key"]
    _llm_today_scope_keys = ctx["_llm_today_scope_keys"]
    _load_actual_result_map = ctx["_load_actual_result_map"]
    _load_combined_llm_report_runs = ctx["_load_combined_llm_report_runs"]
    _report_scope_key_for_row = ctx["_report_scope_key_for_row"]
    _run_date_key = ctx["_run_date_key"]
    _safe_text = ctx["_safe_text"]
    LLM_BATTLE_LABELS = ctx["LLM_BATTLE_LABELS"]
    LLM_BATTLE_ORDER = ctx["LLM_BATTLE_ORDER"]
    load_policy_payloads = ctx["load_policy_payloads"]
    normalize_policy_engine = ctx["normalize_policy_engine"]
    _actual_result_snapshot = ctx["_actual_result_snapshot"]
    _policy_primary_output = ctx["_policy_primary_output"]
    _policy_marks_map = ctx["_policy_marks_map"]
    _payload_run_id = ctx["_payload_run_id"]
    load_policy_run_ticket_rows = ctx["load_policy_run_ticket_rows"]
    _policy_primary_budget = ctx["_policy_primary_budget"]
    _summarize_ticket_rows = ctx["_summarize_ticket_rows"]
    _llm_today_status_meta = ctx["_llm_today_status_meta"]
    _format_ticket_plan_text = ctx["_format_ticket_plan_text"]
    _format_marks_text = ctx["_format_marks_text"]
    _format_triplet_text = ctx["_format_triplet_text"]
    _marks_result_triplet = ctx["_marks_result_triplet"]
    _build_public_share_text = ctx["_build_public_share_text"]
    _format_race_label = ctx["_format_race_label"]
    _scope_display_name = ctx["_scope_display_name"]
    _format_jp_date_text = ctx["_format_jp_date_text"]
    _format_yen_text = ctx["_format_yen_text"]
    _format_percent_text = ctx["_format_percent_text"]
    target_date = _normalize_report_date_text(date_text)
    scope_norm = normalize_scope_key(scope_key)
    scope_keys = _llm_today_scope_keys(scope_norm)
    actual_result_maps = {
        report_scope_key: _load_actual_result_map(report_scope_key)
        for report_scope_key in scope_keys
    }
    runs = []
    for row in _load_combined_llm_report_runs():
        report_scope_key = _report_scope_key_for_row(row, scope_norm)
        if report_scope_key not in scope_keys:
            continue
        if _run_date_key(row) != target_date:
            continue
        runs.append(row)
    runs.sort(
        key=lambda row: (
            _safe_text(row.get("race_date")),
            _safe_text(row.get("location")),
            _safe_text(row.get("race_id")),
            _safe_text(row.get("timestamp")),
            _safe_text(row.get("run_id")),
        )
    )

    summary_by_engine = {
        engine: {
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
    race_sections = []

    for run_row in runs:
        run_id = _safe_text(run_row.get("run_id"))
        report_scope_key = _report_scope_key_for_row(run_row, scope_norm)
        payload_map = {}
        for payload in load_policy_payloads(report_scope_key, run_id, run_row):
            engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
            if engine:
                payload_map[engine] = payload
        if not payload_map:
            continue

        actual_snapshot = _actual_result_snapshot(
            report_scope_key,
            run_id,
            run_row,
            actual_result_maps.get(report_scope_key, {}),
        )
        actual_names = list(actual_snapshot.get("actual_names", []) or [])
        actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])
        actual_text = " / ".join(
            f"{idx + 1}着 {name}"
            for idx, name in enumerate(actual_names[:3])
            if _safe_text(name)
        ) or "待录入"

        engine_cards = []
        share_options = []
        for engine in LLM_BATTLE_ORDER:
            payload = payload_map.get(engine)
            if not payload:
                continue
            output = _policy_primary_output(payload)
            payload_input = dict((payload or {}).get("input", {}) or {})
            ai_summary = dict(payload_input.get("ai", {}) or {})
            marks_map = _policy_marks_map(payload)
            ticket_run_id = _payload_run_id(payload, run_id)
            ticket_rows = load_policy_run_ticket_rows(ticket_run_id, policy_engine=engine) or list(
                _policy_primary_budget(payload).get("tickets", []) or []
            )
            ticket_summary = _summarize_ticket_rows(ticket_rows)
            status_label, status_tone = _llm_today_status_meta(ticket_summary, actual_names)
            ticket_text = _format_ticket_plan_text(ticket_rows, output).replace("\n", "<br>")
            marks_text = _format_marks_text(marks_map)
            result_triplet = _format_triplet_text(_marks_result_triplet(marks_map, actual_horse_nos))
            strategy_text = _safe_text(output.get("strategy_text_ja")) or _safe_text(output.get("strategy_mode")) or "未生成"
            tendency_text = _safe_text(output.get("bet_tendency_ja")) or _safe_text(output.get("buy_style")) or "未生成"
            decision_text = _safe_text(output.get("bet_decision")) or "-"
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

            share_text = _build_public_share_text(run_row, engine, marks_map, ticket_rows)
            if share_text:
                share_options.append(share_text)

            engine_cards.append(
                f"""
                <article class="llm-today-card llm-today-card--{html.escape(status_tone)}">
                  <div class="llm-today-card-head">
                    <div>
                      <div class="llm-today-engine">{html.escape(LLM_BATTLE_LABELS.get(engine, engine))}</div>
                      <div class="llm-today-decision">决策：{html.escape(decision_text)}</div>
                    </div>
                    <span class="llm-today-badge llm-today-badge--{html.escape(status_tone)}">{html.escape(status_label)}</span>
                  </div>
                  <div class="llm-today-grid">
                    <section>
                      <h4>购买马券</h4>
                      <p>{ticket_text}</p>
                    </section>
                    <section>
                      <h4>评价</h4>
                      <p>{html.escape(strategy_text)}</p>
                      <p class="llm-today-subtext">{html.escape(tendency_text)}</p>
                    </section>
                    <section>
                      <h4>马印</h4>
                      <p>{html.escape(marks_text)}</p>
                    </section>
                    <section>
                      <h4>结果映射</h4>
                      <p>{html.escape(result_triplet if any(_safe_text(x) for x in actual_horse_nos) else "待录入")}</p>
                    </section>
                  </div>
                  <div class="llm-today-metrics">
                    <span>票数 {int(ticket_summary.get("ticket_count", 0) or 0)}</span>
                    <span>投入 {_format_yen_text(ticket_summary.get("stake_yen", 0))}</span>
                    <span>回收 {_format_yen_text(ticket_summary.get("payout_yen", 0))}</span>
                    <span>收支 {_format_yen_text(ticket_summary.get("profit_yen", 0))}</span>
                    <span>命中 {int(ticket_summary.get("hit_count", 0) or 0)}</span>
                    <span>回収率 {_format_percent_text(ticket_summary.get("roi", ""))}</span>
                  </div>
                </article>
                """
            )

        if not engine_cards:
            continue
        race_title = _format_race_label(run_row)
        scope_label = _scope_display_name(report_scope_key)
        race_sections.append(
            f"""
            <section class="llm-race-section">
              <div class="llm-race-head">
                <div>
                  <div class="llm-race-eyebrow">{html.escape(scope_label)}</div>
                  <h2>{html.escape(race_title)}</h2>
                </div>
                <div class="llm-race-meta">
                  <span>{html.escape(_format_jp_date_text(run_row) or target_date)}</span>
                  <span>Run {html.escape(run_id or "-")}</span>
                </div>
              </div>
              <div class="llm-race-result">实际结果：{html.escape(actual_text)}</div>
              <div class="llm-today-card-grid">
                {"".join(engine_cards)}
              </div>
            </section>
            """
        )

    summary_cards = []
    for engine in LLM_BATTLE_ORDER:
        stats = summary_by_engine[engine]
        if int(stats.get("races", 0) or 0) <= 0:
            continue
        roi = ""
        if int(stats.get("stake_yen", 0) or 0) > 0:
            roi = round(float(stats["payout_yen"]) / float(stats["stake_yen"]), 4)
        summary_cards.append(
            f"""
            <article class="llm-summary-card">
              <div class="llm-summary-head">
                <strong>{html.escape(stats['label'])}</strong>
                <span>{int(stats['races'])} 场</span>
              </div>
              <div class="llm-summary-metrics">
                <span>已结算 {int(stats['settled_races'])}</span>
                <span>待结算 {int(stats['pending_races'])}</span>
                <span>命中场次 {int(stats['hit_races'])}</span>
                <span>总票数 {int(stats['ticket_count'])}</span>
                <span>投入 {_format_yen_text(stats['stake_yen'])}</span>
                <span>回收 {_format_yen_text(stats['payout_yen'])}</span>
                <span>收支 {_format_yen_text(stats['profit_yen'])}</span>
                <span>回収率 {_format_percent_text(roi)}</span>
              </div>
            </article>
            """
        )

    scope_options = ['<option value="">全部范围</option>']
    for key in LLM_REPORT_SCOPE_KEYS:
        selected_attr = " selected" if scope_norm == key else ""
        scope_options.append(
            f'<option value="{html.escape(key)}"{selected_attr}>{html.escape(_scope_display_name(key))}</option>'
        )

    empty_state = ""
    if not race_sections:
        empty_state = """
        <section class="llm-empty">
          <h2>今天还没有可展示的 LLM 票据</h2>
          <p>先运行当日的 LLM buy。录入赛果后，这个页面会自动显示已结算结果与收支。</p>
        </section>
        <form method="post" action="/topup_all_llm_budget" class="stack-form">
          <input type="hidden" name="scope_key" value="{html.escape(scope_key)}">
          <input type="hidden" name="run_id" value="{html.escape(current_run_id)}">
          <input type="hidden" name="token" value="{html.escape(admin_token)}">
          <button type="submit" class="secondary-button">所有LLM追加当日预算</button>
        </form>
        """

    summary_html = "".join(summary_cards) if summary_cards else '<p class="llm-empty-inline">今天还没有模型汇总。</p>'
    fallback_notice_html = ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM 当日看板</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --paper: rgba(255, 251, 246, 0.92);
      --paper-strong: #fffaf3;
      --ink: #182018;
      --muted: #5d6a60;
      --line: rgba(24, 32, 24, 0.1);
      --accent: #135d48;
      --accent-soft: rgba(19, 93, 72, 0.12);
      --settled: #1c6b43;
      --settled-soft: rgba(28, 107, 67, 0.14);
      --pending: #a56a16;
      --pending-soft: rgba(165, 106, 22, 0.14);
      --planned: #5a6678;
      --planned-soft: rgba(90, 102, 120, 0.14);
      --shadow: 0 18px 45px rgba(34, 38, 30, 0.08);
      --title-font: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --body-font: "Aptos", "Segoe UI Variable Text", "Yu Gothic UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--body-font);
      background:
        radial-gradient(circle at top left, rgba(227, 214, 189, 0.65), transparent 28%),
        radial-gradient(circle at top right, rgba(195, 221, 210, 0.7), transparent 32%),
        linear-gradient(180deg, #fbf7f1 0%, var(--bg) 100%);
    }}
    .llm-page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 20px 40px;
      display: grid;
      gap: 22px;
    }}
    .llm-hero, .llm-section, .llm-race-section, .llm-empty {{
      border: 1px solid rgba(255, 255, 255, 0.65);
      background: var(--paper);
      border-radius: 26px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .llm-hero {{
      padding: 28px;
      display: grid;
      gap: 18px;
    }}
    .llm-eyebrow {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .llm-hero h1, .llm-race-head h2, .llm-section h2, .llm-empty h2 {{
      margin: 0;
      font-family: var(--title-font);
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    .llm-hero h1 {{ font-size: clamp(34px, 5vw, 52px); line-height: 0.95; }}
    .llm-hero p {{
      margin: 0;
      max-width: 70ch;
      line-height: 1.65;
      color: var(--muted);
    }}
    .llm-filter {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
    }}
    .llm-filter label {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }}
    .llm-filter input, .llm-filter select {{
      min-width: 180px;
      min-height: 42px;
      padding: 0 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--paper-strong);
      color: var(--ink);
      font: inherit;
    }}
    .llm-filter button, .llm-hero a {{
      min-height: 42px;
      padding: 0 16px;
      border-radius: 999px;
      border: 0;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }}
    .llm-section {{
      padding: 22px;
      display: grid;
      gap: 14px;
    }}
    .llm-notice {{
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(19, 93, 72, 0.14);
      background: rgba(19, 93, 72, 0.08);
      color: var(--ink);
      font-size: 14px;
      line-height: 1.6;
    }}
    .llm-summary-grid, .llm-today-card-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    .llm-summary-card, .llm-today-card {{
      background: rgba(255, 255, 255, 0.76);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      display: grid;
      gap: 14px;
    }}
    .llm-summary-head, .llm-today-card-head, .llm-race-head {{
      display: flex;
      gap: 12px;
      justify-content: space-between;
      align-items: start;
    }}
    .llm-summary-metrics, .llm-today-metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .llm-summary-metrics span, .llm-today-metrics span, .llm-race-meta span {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(24, 32, 24, 0.06);
      color: var(--muted);
      font-size: 12px;
    }}
    .llm-race-section {{
      padding: 22px;
      display: grid;
      gap: 14px;
    }}
    .llm-race-eyebrow {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}
    .llm-race-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: end;
    }}
    .llm-race-result {{
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(19, 93, 72, 0.08);
      color: var(--ink);
      font-weight: 600;
    }}
    .llm-today-card--settled {{ border-color: rgba(28, 107, 67, 0.22); background: linear-gradient(180deg, rgba(28, 107, 67, 0.08), rgba(255,255,255,0.82)); }}
    .llm-today-card--pending {{ border-color: rgba(165, 106, 22, 0.22); background: linear-gradient(180deg, rgba(165, 106, 22, 0.08), rgba(255,255,255,0.82)); }}
    .llm-today-card--planned, .llm-today-card--result {{ border-color: rgba(90, 102, 120, 0.18); }}
    .llm-today-engine {{ font-size: 22px; font-family: var(--title-font); }}
    .llm-today-decision {{ margin-top: 4px; color: var(--muted); font-size: 13px; }}
    .llm-today-badge {{
      padding: 7px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .llm-today-badge--settled {{ background: var(--settled-soft); color: var(--settled); }}
    .llm-today-badge--pending {{ background: var(--pending-soft); color: var(--pending); }}
    .llm-today-badge--planned, .llm-today-badge--result {{ background: var(--planned-soft); color: var(--planned); }}
    .llm-today-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .llm-today-grid section {{
      padding: 14px;
      border-radius: 16px;
      background: rgba(24, 32, 24, 0.04);
      min-height: 128px;
    }}
    .llm-today-grid h4 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .llm-today-grid p {{
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.55;
    }}
    .llm-today-subtext {{
      margin-top: 8px !important;
      color: var(--muted);
      font-size: 13px;
    }}
    .llm-empty {{
      padding: 28px;
      text-align: center;
    }}
    .llm-empty p, .llm-empty-inline {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    @media (max-width: 760px) {{
      .llm-page {{ padding: 18px 14px 28px; }}
      .llm-hero, .llm-section, .llm-race-section {{ padding: 18px; }}
      .llm-filter {{ flex-direction: column; align-items: stretch; }}
      .llm-filter label, .llm-filter input, .llm-filter select, .llm-filter button {{ width: 100%; }}
      .llm-race-head, .llm-today-card-head {{ flex-direction: column; }}
      .llm-race-meta {{ justify-content: start; }}
    }}
  </style>
</head>
<body>
  <main class="llm-page">
    <section class="llm-hero">
      <div class="llm-eyebrow">LLM Daily Board</div>
      <h1>当日 LLM 购票看板</h1>
      <p>先看今天每个模型到底买了什么、怎么评价这场比赛；录入赛果后，同一页直接显示结算结果、回收和收支。</p>
      <form class="llm-filter" method="get" action="/llm_today">
        <label>日期
          <input type="date" name="date" value="{html.escape(target_date)}">
        </label>
        <label>范围
          <select name="scope_key">
            {"".join(scope_options)}
          </select>
        </label>
        <button type="submit">刷新看板</button>
        <a href="/">回主控制台</a>
      </form>
    </section>
    {fallback_notice_html}
    <section class="llm-section">
      <div class="llm-eyebrow">Daily Summary</div>
      <h2>模型总览</h2>
      <div class="llm-summary-grid">{summary_html}</div>
    </section>
    {empty_state}
    {"".join(race_sections)}
  </main>
  <script>
    (() => {{
      const openShareIntent = async (text) => {{
        const url = `https://x.com/intent/post?text=${{encodeURIComponent(text)}}`;
        const isMobileShare =
          /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "") ||
          (window.matchMedia && window.matchMedia("(max-width: 760px)").matches) ||
          ("ontouchstart" in window);
        if (isMobileShare && navigator.share) {{
          try {{
            await navigator.share({{ text }});
            return;
          }} catch (error) {{
            if (error && error.name === "AbortError") {{
              return;
            }}
          }}
        }}
        if (isMobileShare) {{
          window.location.href = url;
          return;
        }}
        const width = 720;
        const height = 640;
        const left = Math.max(0, Math.round((window.screen.width - width) / 2));
        const top = Math.max(0, Math.round((window.screen.height - height) / 2));
        const popup = window.open(
          url,
          "ikaimo-share",
          `popup=yes,width=${{width}},height=${{height}},left=${{left}},top=${{top}},resizable=yes,scrollbars=yes`
        );
        if (popup && !popup.closed) {{
          try {{
            popup.focus();
          }} catch (_error) {{
          }}
          return;
        }}
        window.location.href = url;
      }};
      document.addEventListener("click", async (event) => {{
        const button = event.target.closest(".front-share-button");
        if (!button) return;
        const raw = button.getAttribute("data-share-options") || "[]";
        let options = [];
        try {{
          options = JSON.parse(raw);
        }} catch (_error) {{
          options = [];
        }}
        if (!Array.isArray(options) || options.length === 0) return;
        const text = String(options[Math.floor(Math.random() * options.length)] || "").trim();
        if (!text) return;
        await openShareIntent(text);
      }});
    }})();
  </script>
</body>
</html>"""

def build_llm_today_page_clean(date_text="", scope_key="", *, ctx):
    _normalize_report_date_text = ctx["_normalize_report_date_text"]
    normalize_scope_key = ctx["normalize_scope_key"]
    _llm_today_scope_keys = ctx["_llm_today_scope_keys"]
    _resolve_llm_today_target_date = ctx["_resolve_llm_today_target_date"]
    _load_actual_result_map = ctx["_load_actual_result_map"]
    _report_scope_key_for_row = ctx["_report_scope_key_for_row"]
    _run_date_key = ctx["_run_date_key"]
    _safe_text = ctx["_safe_text"]
    LLM_BATTLE_LABELS = ctx["LLM_BATTLE_LABELS"]
    LLM_BATTLE_ORDER = ctx["LLM_BATTLE_ORDER"]
    LLM_REPORT_SCOPE_KEYS = ctx["LLM_REPORT_SCOPE_KEYS"]
    load_policy_payloads = ctx["load_policy_payloads"]
    normalize_policy_engine = ctx["normalize_policy_engine"]
    _actual_result_snapshot = ctx["_actual_result_snapshot"]
    _policy_primary_output = ctx["_policy_primary_output"]
    _policy_marks_map = ctx["_policy_marks_map"]
    _payload_run_id = ctx["_payload_run_id"]
    load_policy_run_ticket_rows = ctx["load_policy_run_ticket_rows"]
    _policy_primary_budget = ctx["_policy_primary_budget"]
    _summarize_ticket_rows = ctx["_summarize_ticket_rows"]
    _llm_today_status_meta_ja = ctx["_llm_today_status_meta_ja"]
    _format_ticket_plan_text_ja = ctx["_format_ticket_plan_text_ja"]
    _format_marks_text = ctx["_format_marks_text"]
    _format_triplet_text = ctx["_format_triplet_text"]
    _marks_result_triplet = ctx["_marks_result_triplet"]
    _format_result_triplet_text_ja = ctx["_format_result_triplet_text_ja"]
    _format_confidence_text = ctx["_format_confidence_text"]
    _format_race_label = ctx["_format_race_label"]
    _llm_today_scope_label_ja = ctx["_llm_today_scope_label_ja"]
    _format_jp_date_text = ctx["_format_jp_date_text"]
    _format_yen_text_ja = ctx["_format_yen_text_ja"]
    _format_percent_text = ctx["_format_percent_text"]
    requested_date = _normalize_report_date_text(date_text)
    scope_norm = normalize_scope_key(scope_key)
    scope_keys = _llm_today_scope_keys(scope_norm)
    target_date, fallback_notice, combined_rows = _resolve_llm_today_target_date(requested_date, scope_norm)
    actual_result_maps = {
        report_scope_key: _load_actual_result_map(report_scope_key)
        for report_scope_key in scope_keys
    }
    runs = []
    for row in combined_rows:
        report_scope_key = _report_scope_key_for_row(row, scope_norm)
        if report_scope_key not in scope_keys:
            continue
        if _run_date_key(row) != target_date:
            continue
        runs.append(row)
    runs.sort(
        key=lambda row: (
            _safe_text(row.get("race_date")),
            _safe_text(row.get("location")),
            _safe_text(row.get("race_id")),
            _safe_text(row.get("timestamp")),
            _safe_text(row.get("run_id")),
        )
    )

    summary_by_engine = {
        engine: {
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
    race_sections = []

    for run_row in runs:
        run_id = _safe_text(run_row.get("run_id"))
        report_scope_key = _report_scope_key_for_row(run_row, scope_norm)
        payload_map = {}
        for payload in load_policy_payloads(report_scope_key, run_id, run_row):
            engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
            if engine:
                payload_map[engine] = payload
        if not payload_map:
            continue

        actual_snapshot = _actual_result_snapshot(
            report_scope_key,
            run_id,
            run_row,
            actual_result_maps.get(report_scope_key, {}),
        )
        actual_names = list(actual_snapshot.get("actual_names", []) or [])
        actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])
        actual_text = _format_result_triplet_text_ja(actual_names)

        engine_cards = []
        for engine in LLM_BATTLE_ORDER:
            payload = payload_map.get(engine)
            if not payload:
                continue
            output = _policy_primary_output(payload)
            payload_input = dict((payload or {}).get("input", {}) or {})
            ai_summary = dict(payload_input.get("ai", {}) or {})
            marks_map = _policy_marks_map(payload)
            ticket_run_id = _payload_run_id(payload, run_id)
            ticket_rows = load_policy_run_ticket_rows(ticket_run_id, policy_engine=engine) or list(
                _policy_primary_budget(payload).get("tickets", []) or []
            )
            ticket_summary = _summarize_ticket_rows(ticket_rows)
            status_label, status_tone = _llm_today_status_meta_ja(ticket_summary, actual_names)
            ticket_text = html.escape(_format_ticket_plan_text_ja(ticket_rows)).replace("\n", "<br>")
            marks_text = html.escape(_format_marks_text(marks_map))
            result_triplet = _format_triplet_text(_marks_result_triplet(marks_map, actual_horse_nos))
            result_triplet_text = html.escape(result_triplet if any(_safe_text(x) for x in actual_horse_nos) else "結果未登録")
            strategy_text = _safe_text(output.get("strategy_text_ja")) or _safe_text(output.get("strategy_mode")) or "記載なし"
            tendency_text = _safe_text(output.get("bet_tendency_ja")) or _safe_text(output.get("buy_style")) or "記載なし"
            decision_text = _safe_text(output.get("bet_decision")) or "-"
            confidence_value = ai_summary.get("confidence_score", "")
            confidence_text = _format_confidence_text(confidence_value)
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

            engine_cards.append(
                f"""
                <article class="llm-today-card llm-today-card--{html.escape(status_tone)}">
                  <div class="llm-today-card-head">
                    <div>
                      <div class="llm-today-engine">{html.escape(LLM_BATTLE_LABELS.get(engine, engine))}</div>
                      <div class="llm-today-decision">判定: {html.escape(decision_text)}</div>
                    </div>
                    <span class="llm-today-badge llm-today-badge--{html.escape(status_tone)}">{html.escape(status_label)}</span>
                  </div>
                  <div class="llm-today-grid">
                    <section>
                      <h4>買い目</h4>
                      <p>{ticket_text}</p>
                    </section>
                    <section>
                      <h4>戦略</h4>
                      <p>{html.escape(strategy_text)}</p>
                      <p class="llm-today-subtext">{html.escape(tendency_text)}</p>
                    </section>
                    <section>
                      <h4>印</h4>
                      <p>{marks_text}</p>
                    </section>
                    <section>
                      <h4>印の結果</h4>
                      <p>{result_triplet_text}</p>
                    </section>
                  </div>
                  <div class="llm-today-metrics">
                    <span>点数 {_safe_text(ticket_summary.get("ticket_count")) or "0"}</span>
                    <span>投資 {_format_yen_text_ja(ticket_summary.get("stake_yen", 0))}</span>
                    <span>払戻 {_format_yen_text_ja(ticket_summary.get("payout_yen", 0))}</span>
                    <span>収支 {_format_yen_text_ja(ticket_summary.get("profit_yen", 0))}</span>
                    <span>的中 {int(ticket_summary.get("hit_count", 0) or 0)}</span>
                    <span>回収率 {_format_percent_text(ticket_summary.get("roi", ""))}</span>
                  </div>
                </article>
                """
            )

        if not engine_cards:
            continue
        race_title = _format_race_label(run_row)
        scope_label = _llm_today_scope_label_ja(report_scope_key)
        race_sections.append(
            f"""
            <section class="llm-race-section">
              <div class="llm-race-head">
                <div>
                  <div class="llm-race-eyebrow">{html.escape(scope_label)}</div>
                  <h2>{html.escape(race_title)}</h2>
                </div>
                <div class="llm-race-meta">
                  <span>{html.escape(_format_jp_date_text(run_row) or target_date)}</span>
                  <span>Run {html.escape(run_id or "-")}</span>
                </div>
              </div>
              <div class="llm-race-result">実際の結果: {html.escape(actual_text)}</div>
              <div class="llm-today-card-grid">
                {"".join(engine_cards)}
              </div>
            </section>
            """
        )

    summary_cards = []
    for engine in LLM_BATTLE_ORDER:
        stats = summary_by_engine[engine]
        if int(stats.get("races", 0) or 0) <= 0:
            continue
        roi = ""
        if int(stats.get("stake_yen", 0) or 0) > 0:
            roi = round(float(stats["payout_yen"]) / float(stats["stake_yen"]), 4)
        summary_cards.append(
            f"""
            <article class="llm-summary-card">
              <div class="llm-summary-head">
                <strong>{html.escape(stats['label'])}</strong>
                <span>{int(stats['races'])}レース</span>
              </div>
              <div class="llm-summary-metrics">
                <span>確定 {int(stats['settled_races'])}</span>
                <span>未確定 {int(stats['pending_races'])}</span>
                <span>的中レース {int(stats['hit_races'])}</span>
                <span>買い目数 {int(stats['ticket_count'])}</span>
                <span>投資 {_format_yen_text_ja(stats['stake_yen'])}</span>
                <span>払戻 {_format_yen_text_ja(stats['payout_yen'])}</span>
                <span>収支 {_format_yen_text_ja(stats['profit_yen'])}</span>
                <span>回収率 {_format_percent_text(roi)}</span>
              </div>
            </article>
            """
        )

    scope_options = ['<option value="">すべての範囲</option>']
    for key in LLM_REPORT_SCOPE_KEYS:
        selected_attr = " selected" if scope_norm == key else ""
        scope_options.append(
            f'<option value="{html.escape(key)}"{selected_attr}>{html.escape(_llm_today_scope_label_ja(key))}</option>'
        )

    empty_state = ""
    if not race_sections:
        empty_state = """
        <section class="llm-empty">
          <h2>この日の公開データはまだありません</h2>
          <p>予測処理が完了すると、各 LLM の買い目、戦略、印、結果がここに表示されます。</p>
        </section>
        """

    summary_html = "".join(summary_cards) if summary_cards else '<p class="llm-empty-inline">この日の集計データはまだありません。</p>'
    fallback_notice_html = ""
    if fallback_notice:
        fallback_notice_html = f'<section class="llm-notice">{html.escape(fallback_notice)}</section>'
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM予想ボード</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --paper: rgba(255, 251, 246, 0.92);
      --paper-strong: #fffaf3;
      --ink: #182018;
      --muted: #5d6a60;
      --line: rgba(24, 32, 24, 0.1);
      --accent: #135d48;
      --accent-soft: rgba(19, 93, 72, 0.12);
      --settled: #1c6b43;
      --settled-soft: rgba(28, 107, 67, 0.14);
      --pending: #a56a16;
      --pending-soft: rgba(165, 106, 22, 0.14);
      --planned: #5a6678;
      --planned-soft: rgba(90, 102, 120, 0.14);
      --shadow: 0 18px 45px rgba(34, 38, 30, 0.08);
      --title-font: "Hiragino Mincho ProN", "Yu Mincho", "Iowan Old Style", Georgia, serif;
      --body-font: "Yu Gothic UI", "Hiragino Sans", "Aptos", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--body-font);
      background:
        radial-gradient(circle at top left, rgba(227, 214, 189, 0.65), transparent 28%),
        radial-gradient(circle at top right, rgba(195, 221, 210, 0.7), transparent 32%),
        linear-gradient(180deg, #fbf7f1 0%, var(--bg) 100%);
    }}
    .llm-page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 20px 40px;
      display: grid;
      gap: 22px;
    }}
    .llm-hero, .llm-section, .llm-race-section, .llm-empty {{
      border: 1px solid rgba(255, 255, 255, 0.65);
      background: var(--paper);
      border-radius: 26px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .llm-hero {{
      padding: 28px;
      display: grid;
      gap: 18px;
    }}
    .llm-eyebrow {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .llm-hero h1, .llm-race-head h2, .llm-section h2, .llm-empty h2 {{
      margin: 0;
      font-family: var(--title-font);
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    .llm-hero h1 {{ font-size: clamp(34px, 5vw, 52px); line-height: 0.95; }}
    .llm-hero p {{
      margin: 0;
      max-width: 70ch;
      line-height: 1.65;
      color: var(--muted);
    }}
    .llm-filter {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
    }}
    .llm-filter label {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }}
    .llm-filter input, .llm-filter select {{
      min-width: 180px;
      min-height: 42px;
      padding: 0 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--paper-strong);
      color: var(--ink);
      font: inherit;
    }}
    .llm-filter button {{
      min-height: 42px;
      padding: 0 16px;
      border-radius: 999px;
      border: 0;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .llm-section {{
      padding: 22px;
      display: grid;
      gap: 14px;
    }}
    .llm-summary-grid, .llm-today-card-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    .llm-summary-card, .llm-today-card {{
      background: rgba(255, 255, 255, 0.76);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      display: grid;
      gap: 14px;
    }}
    .llm-summary-head, .llm-today-card-head, .llm-race-head {{
      display: flex;
      gap: 12px;
      justify-content: space-between;
      align-items: start;
    }}
    .llm-summary-metrics, .llm-today-metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .llm-summary-metrics span, .llm-today-metrics span, .llm-race-meta span {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(24, 32, 24, 0.06);
      color: var(--muted);
      font-size: 12px;
    }}
    .llm-race-section {{
      padding: 22px;
      display: grid;
      gap: 14px;
    }}
    .llm-race-eyebrow {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}
    .llm-race-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: end;
    }}
    .llm-race-result {{
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(19, 93, 72, 0.08);
      color: var(--ink);
      font-weight: 600;
    }}
    .llm-today-card--settled {{ border-color: rgba(28, 107, 67, 0.22); background: linear-gradient(180deg, rgba(28, 107, 67, 0.08), rgba(255,255,255,0.82)); }}
    .llm-today-card--pending {{ border-color: rgba(165, 106, 22, 0.22); background: linear-gradient(180deg, rgba(165, 106, 22, 0.08), rgba(255,255,255,0.82)); }}
    .llm-today-card--planned, .llm-today-card--result {{ border-color: rgba(90, 102, 120, 0.18); }}
    .llm-today-engine {{ font-size: 22px; font-family: var(--title-font); }}
    .llm-today-decision {{ margin-top: 4px; color: var(--muted); font-size: 13px; }}
    .llm-today-badge {{
      padding: 7px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .llm-today-badge--settled {{ background: var(--settled-soft); color: var(--settled); }}
    .llm-today-badge--pending {{ background: var(--pending-soft); color: var(--pending); }}
    .llm-today-badge--planned, .llm-today-badge--result {{ background: var(--planned-soft); color: var(--planned); }}
    .llm-today-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .llm-today-grid section {{
      padding: 14px;
      border-radius: 16px;
      background: rgba(24, 32, 24, 0.04);
      min-height: 128px;
    }}
    .llm-today-grid h4 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .llm-today-grid p {{
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.55;
    }}
    .llm-today-subtext {{
      margin-top: 8px !important;
      color: var(--muted);
      font-size: 13px;
    }}
    .llm-empty {{
      padding: 28px;
      text-align: center;
    }}
    .llm-empty p, .llm-empty-inline {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    @media (max-width: 760px) {{
      .llm-page {{ padding: 18px 14px 28px; }}
      .llm-hero, .llm-section, .llm-race-section {{ padding: 18px; }}
      .llm-filter {{ flex-direction: column; align-items: stretch; }}
      .llm-filter label, .llm-filter input, .llm-filter select, .llm-filter button {{ width: 100%; }}
      .llm-race-head, .llm-today-card-head {{ flex-direction: column; }}
      .llm-race-meta {{ justify-content: start; }}
    }}
  </style>
</head>
<body>
  <main class="llm-page">
    <section class="llm-hero">
      <div class="llm-eyebrow">LLM Daily Board</div>
      <h1>本日の LLM 予想ボード</h1>
      <p>当日公開されている各 LLM の買い目、戦略、印、回収状況をまとめて確認できます。レース終了後は結果と収支も反映されます。</p>
      <form class="llm-filter" method="get" action="/llm_today">
        <label>日付
          <input type="date" name="date" value="{html.escape(target_date)}">
        </label>
        <label>範囲
          <select name="scope_key">
            {"".join(scope_options)}
          </select>
        </label>
        <button type="submit">表示を更新</button>
      </form>
    </section>
    {fallback_notice_html}
    <section class="llm-section">
      <div class="llm-eyebrow">Daily Summary</div>
      <h2>日次サマリー</h2>
      <div class="llm-summary-grid">{summary_html}</div>
    </section>
    {empty_state}
    {"".join(race_sections)}
  </main>
</body>
</html>"""

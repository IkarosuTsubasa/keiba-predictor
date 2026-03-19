import html
import re
from urllib.parse import quote_plus

from surface_scope import normalize_scope_key


def build_console_gate_page(*, admin_token="", error_text="", prefix_public_html_routes):
    error_block = ""
    if error_text:
        error_block = f'<section class="job-flash job-flash--error">{html.escape(error_text)}</section>'
    return prefix_public_html_routes(f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>謗ｧ蛻ｶ蜿ｰ鬪瑚ｯ・/title>
  <style>
    :root {{
      --bg: #f4efe8;
      --paper: rgba(255, 250, 244, 0.94);
      --ink: #17201a;
      --muted: #617064;
      --accent: #145846;
      --danger: #ad4d3b;
      --line: rgba(23,31,26,0.1);
      --shadow: 0 18px 50px rgba(28,33,29,0.09);
      --title-font: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --body-font: "Aptos", "Segoe UI Variable Text", "Yu Gothic UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--body-font);
      color: var(--ink);
      background:
        radial-gradient(circle at 0% 0%, rgba(226, 209, 180, 0.55), transparent 28%),
        radial-gradient(circle at 100% 0%, rgba(198, 221, 210, 0.65), transparent 30%),
        linear-gradient(180deg, #faf6f0 0%, var(--bg) 100%);
    }}
    .gate {{
      max-width: 760px;
      margin: 64px auto;
      padding: 0 18px;
      display: grid;
      gap: 18px;
    }}
    .panel, .job-flash {{
      padding: 24px;
      border-radius: 24px;
      background: var(--paper);
      border: 1px solid rgba(255,255,255,0.75);
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    h1 {{
      margin: 8px 0 10px;
      font-family: var(--title-font);
      font-size: clamp(34px, 5vw, 50px);
      line-height: 0.96;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
    }}
    form {{
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }}
    input {{
      min-height: 44px;
      padding: 0 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      font: inherit;
    }}
    button, a {{
      min-height: 42px;
      padding: 0 14px;
      border-radius: 999px;
      border: 0;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 700;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
    }}
    .job-flash--error {{
      color: var(--danger);
      border-color: rgba(173,77,59,0.24);
      background: rgba(255, 245, 242, 0.95);
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
  </style>
</head>
<body>
  <main class="gate">
    {error_block}
    <section class="panel">
      <div class="eyebrow">Protected</div>
      <h1>謗ｧ蛻ｶ蜿ｰ鬪瑚ｯ・/h1>
      <p>霑咎㈹譏ｯ菴逧・錘蜿ｰ縲りｾ灘・豁｣遑ｮ逧・`ADMIN_TOKEN` 蜷取燕閭ｽ霑帛・ `/console`縲・/p>
      <form method="get" action="/console">
        <input type="password" name="token" placeholder="ADMIN_TOKEN" value="{html.escape(admin_token)}">
        <div class="actions">
          <button type="submit">霑帛・謗ｧ蛻ｶ蜿ｰ</button>
          <a href="/llm_today">霑泌屓蜑榊床</a>
        </div>
      </form>
    </section>
  </main>
</body>
</html>""")


def render_console_page(
    *,
    message_text="",
    error_text="",
    admin_token="",
    show_settled=False,
    render_page,
    build_import_archive_panel,
    build_admin_filter_panel,
    build_admin_workspace_html_clean,
    admin_token_valid,
):
    return render_page(
        "",
        admin_token=admin_token,
        admin_workspace_html=(
            build_import_archive_panel(admin_token=admin_token)
            + build_admin_filter_panel(admin_token=admin_token, show_settled=show_settled)
            + build_admin_workspace_html_clean(
                message_text=message_text,
                error_text=error_text,
                admin_token=admin_token,
                authorized=admin_token_valid(admin_token),
                show_settled=show_settled,
            )
        ),
    )


def resolve_console_run_state(
    *,
    scope_key="",
    selected_run_id="",
    summary_run_id="",
    output_text="",
    resolve_run,
    parse_run_id,
    normalize_race_id,
):
    scope_norm = normalize_scope_key(scope_key)
    run_id = ""
    run_row = None
    if scope_norm:
        if selected_run_id:
            run_row = resolve_run(selected_run_id, scope_norm)
            if run_row:
                run_id = run_row.get("run_id", selected_run_id)
            else:
                run_id = selected_run_id
        else:
            run_id = parse_run_id(output_text)
            if run_id:
                run_row = resolve_run(run_id, scope_norm)
            else:
                run_row = resolve_run("", scope_norm)
                if run_row:
                    run_id = run_row.get("run_id", "")
    view_selected_run_id = selected_run_id or run_id or summary_run_id
    current_race_id = ""
    if run_row:
        current_race_id = normalize_race_id(run_row.get("race_id", ""))
    if not current_race_id:
        race_candidate = re.sub(r"\D", "", selected_run_id or summary_run_id or "")
        if re.fullmatch(r"\d{12}", race_candidate):
            current_race_id = race_candidate
    return {
        "scope_key": scope_norm or scope_key or "central_dirt",
        "run_id": str(run_id or "").strip(),
        "run_row": dict(run_row or {}),
        "current_race_id": str(current_race_id or "").strip(),
        "selected_run_id": str(view_selected_run_id or "").strip(),
    }


def load_note_workspace(
    *,
    scope_key="",
    run_id="",
    run_row=None,
    load_bet_engine_v3_cfg_summary,
    load_actual_result_map,
    load_policy_payloads,
    normalize_policy_engine,
    load_policy_bankroll_summary,
    load_policy_run_ticket_rows,
    build_llm_battle_bundle,
    build_llm_daily_report_bundle,
    build_llm_weekly_report_bundle,
    resolve_predictor_paths,
    load_ability_marks_table,
    load_mark_recommendation_table,
    load_text_file,
    build_mark_note_text,
):
    mark_note_text = ""
    llm_note_text = ""
    daily_report_text = ""
    weekly_report_text = ""
    if not run_id:
        return {
            "mark_note_text": "",
            "llm_note_text": "",
            "daily_report_text": "",
            "weekly_report_text": "",
        }

    scope_norm = normalize_scope_key(scope_key)
    run_row = dict(run_row or {})
    bet_engine_v3_summary = load_bet_engine_v3_cfg_summary(scope_norm, run_id)
    policy_payloads = []
    actual_result_map = load_actual_result_map(scope_norm)
    for payload in load_policy_payloads(scope_norm, run_id, run_row):
        payload = dict(payload)
        budget_items = [dict(item) for item in list(payload.get("budgets", []) or [])]
        live_policy_engine = normalize_policy_engine(payload.get("policy_engine", "gemini"))
        live_summary = load_policy_bankroll_summary(
            run_id,
            (run_row or {}).get("timestamp", ""),
            policy_engine=live_policy_engine,
        )
        live_tickets = load_policy_run_ticket_rows(run_id, policy_engine=live_policy_engine)
        if budget_items:
            first = dict(budget_items[0])
            portfolio = dict(first.get("portfolio", {}) or {})
            portfolio.setdefault("before", dict(live_summary))
            portfolio["after"] = dict(live_summary)
            first["portfolio"] = portfolio
            if live_tickets:
                first["tickets"] = live_tickets
            budget_items[0] = first
        payload["budgets"] = budget_items
        policy_payloads.append(payload)

    primary_policy_payload = dict(policy_payloads[0]) if policy_payloads else {}
    battle_bundle = build_llm_battle_bundle(
        scope_norm,
        run_id,
        run_row,
        policy_payloads,
        actual_result_map,
    )
    llm_note_text = str(battle_bundle.get("note_text", "") or "").strip()
    daily_report_text = str(build_llm_daily_report_bundle(scope_norm, run_row, actual_result_map).get("text", "") or "").strip()
    weekly_report_text = str(build_llm_weekly_report_bundle(scope_norm, run_row, actual_result_map).get("text", "") or "").strip()

    predictor_note_texts = []
    for spec, pred_path in resolve_predictor_paths(scope_norm, run_id, run_row):
        if not pred_path or not pred_path.exists():
            continue
        predictor_run_row = dict(run_row or {})
        predictor_run_row["predictions_path"] = str(pred_path)
        ability_rows, _ability_cols = load_ability_marks_table(scope_norm, run_id, predictor_run_row)
        if not ability_rows:
            ability_rows, _mark_cols = load_mark_recommendation_table(scope_norm, run_id, predictor_run_row)
        pred_csv_text = load_text_file(pred_path)
        note_text = build_mark_note_text(
            ability_rows if ability_rows else [],
            pred_path.name if pred_path else "",
            pred_csv_text,
            bet_engine_v3_summary=bet_engine_v3_summary,
            gemini_policy_payload=primary_policy_payload,
        ).strip()
        if note_text:
            predictor_note_texts.append(f"[{spec['label']}]\n{note_text}")
    if predictor_note_texts:
        mark_note_text = "\n\n".join(predictor_note_texts)

    return {
        "mark_note_text": mark_note_text,
        "llm_note_text": llm_note_text,
        "daily_report_text": daily_report_text,
        "weekly_report_text": weekly_report_text,
    }


def build_note_workspace_page(
    *,
    scope_key="",
    run_id="",
    admin_token="",
    resolve_console_run_state,
    load_note_workspace,
    prefix_public_html_routes,
):
    state = resolve_console_run_state(scope_key=scope_key, selected_run_id=run_id)
    scope_norm = normalize_scope_key(state["scope_key"]) or "central_dirt"
    note_bundle = load_note_workspace(scope_key=scope_norm, run_id=state["run_id"], run_row=state["run_row"])
    encoded_token = quote_plus(str(admin_token or "").strip()) if admin_token else ""
    back_href = f"/console?token={encoded_token}" if encoded_token else "/console"
    encoded_scope = quote_plus(scope_norm)
    run_value = state["selected_run_id"] or state["current_race_id"]
    note_href = f"/console/note?scope_key={encoded_scope}"
    if run_value:
        note_href += f"&run_id={quote_plus(run_value)}"
    if encoded_token:
        note_href += f"&token={encoded_token}"

    def _note_block(title, text_value, dom_id, tone=""):
        text_value = str(text_value or "").strip()
        tone_class = f" note-card--{tone}" if tone else ""
        if not text_value:
            return f"""
            <section class="note-card{tone_class}">
              <div class="note-card-head">
                <div>
                  <div class="note-eyebrow">Note</div>
                  <h2>{html.escape(title)}</h2>
                </div>
                <span class="note-chip">empty</span>
              </div>
              <p class="note-empty">蠖灘燕豐｡譛牙庄螟榊宛逧・・螳ｹ縲・/p>
            </section>
            """
        return f"""
        <section class="note-card{tone_class}">
          <div class="note-card-head">
            <div>
              <div class="note-eyebrow">Note</div>
              <h2>{html.escape(title)}</h2>
            </div>
            <span class="note-chip">ready</span>
          </div>
          <div class="note-actions">
            <button type="button" class="note-copy-button" data-copy-target="{html.escape(dom_id)}" data-copy-status="{html.escape(dom_id)}-status">螟榊宛</button>
            <span id="{html.escape(dom_id)}-status" class="note-copy-status"></span>
          </div>
          <details class="note-preview" open>
            <summary>鬚・ｧ・/summary>
            <pre>{html.escape(text_value)}</pre>
          </details>
          <textarea id="{html.escape(dom_id)}" class="note-source" readonly>{html.escape(text_value)}</textarea>
        </section>
        """

    blocks_html = "".join(
        [
            _note_block("蜊募惻 LLM Note", note_bundle["llm_note_text"], "note-llm", tone="accent"),
            _note_block("Predictor Note", note_bundle["mark_note_text"], "note-predictor"),
            _note_block("Daily Report", note_bundle["daily_report_text"], "note-daily"),
            _note_block("Weekly Report", note_bundle["weekly_report_text"], "note-weekly"),
        ]
    )

    run_label = state["run_id"] or "???"
    race_label = state["current_race_id"] or "-"
    return prefix_public_html_routes(f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Note Workspace</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --paper: rgba(255, 251, 246, 0.95);
      --ink: #17201a;
      --muted: #607063;
      --accent: #145846;
      --line: rgba(23,31,26,0.1);
      --shadow: 0 18px 50px rgba(28,33,29,0.08);
      --title-font: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --body-font: "Aptos", "Segoe UI Variable Text", "Yu Gothic UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--body-font);
      background:
        radial-gradient(circle at 0% 0%, rgba(226, 209, 180, 0.55), transparent 28%),
        radial-gradient(circle at 100% 0%, rgba(198, 221, 210, 0.6), transparent 30%),
        linear-gradient(180deg, #faf6f0 0%, var(--bg) 100%);
    }}
    .note-page {{ max-width: 1320px; margin: 0 auto; padding: 28px 20px 40px; display: grid; gap: 20px; }}
    .note-hero, .note-tools, .note-card {{
      background: var(--paper);
      border: 1px solid rgba(255,255,255,0.75);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 22px;
    }}
    .note-eyebrow {{ font-size: 12px; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent); }}
    .note-hero h1, .note-card h2 {{ margin: 6px 0 0; font-family: var(--title-font); }}
    .note-meta, .note-actions, .note-links {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    .note-pill {{
      padding: 7px 12px;
      border-radius: 999px;
      background: rgba(23,31,26,0.06);
      color: var(--muted);
      font-size: 12px;
    }}
    .note-links a, .note-links button {{
      min-height: 40px;
      padding: 0 14px;
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: #fff;
      text-decoration: none;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .note-tools form {{ display: grid; gap: 12px; }}
    .note-form-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .note-tools label {{ display: grid; gap: 6px; color: var(--muted); font-size: 13px; }}
    .note-tools input, .note-tools select {{
      width: 100%;
      min-height: 42px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.95);
      font: inherit;
      color: var(--ink);
    }}
    .note-grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .note-card-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
    .note-chip {{ padding: 7px 12px; border-radius: 999px; background: rgba(20,88,70,0.1); color: var(--accent); font-size: 12px; font-weight: 700; }}
    .note-copy-button {{ min-height: 38px; padding: 0 14px; border: 0; border-radius: 999px; background: var(--accent); color: #fff; font: inherit; font-weight: 700; cursor: pointer; }}
    .note-copy-status {{ color: var(--muted); font-size: 12px; }}
    .note-preview summary {{ cursor: pointer; font-weight: 700; }}
    .note-preview pre {{
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 16px;
      background: rgba(23,31,26,0.05);
      overflow: auto;
      white-space: pre-wrap;
      line-height: 1.55;
    }}
    .note-source {{ position: absolute; left: -9999px; width: 1px; height: 1px; opacity: 0; }}
    .note-empty {{ margin: 10px 0 0; color: var(--muted); }}
    @media (max-width: 760px) {{
      .note-page {{ padding: 18px 14px 30px; }}
      .note-hero, .note-tools, .note-card {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <main class="note-page">
    <section class="note-hero">
      <div class="note-eyebrow">Console</div>
      <h1>Note Workspace</h1>
      <div class="note-meta" style="margin-top:12px;">
        <span class="note-pill">Scope: {html.escape(scope_norm)}</span>
        <span class="note-pill">Run: {html.escape(run_label)}</span>
        <span class="note-pill">Race: {html.escape(race_label)}</span>
      </div>
      <div class="note-links" style="margin-top:14px;">
        <a href="{html.escape(back_href)}">?????</a>
      </div>
    </section>
    <section class="note-tools">
      <form method="get" action="/console/note">
        <input type="hidden" name="token" value="{html.escape(admin_token)}">
        <div class="note-form-grid">
          <label>闌・峩
            <select name="scope_key">
              <option value="central_dirt"{' selected' if scope_norm == 'central_dirt' else ''}>central_dirt</option>
              <option value="central_turf"{' selected' if scope_norm == 'central_turf' else ''}>central_turf</option>
              <option value="local"{' selected' if scope_norm == 'local' else ''}>local</option>
            </select>
          </label>
          <label>Run ID / Race ID
            <input type="text" name="run_id" value="{html.escape(run_value)}" placeholder="202501010101 謌・20250101_123456">
          </label>
        </div>
        <div class="note-links">
          <button type="submit">謇灘ｼ霑吝惻 Note</button>
          <a href="{html.escape(note_href)}">蛻ｷ譁ｰ蠖灘燕鬘ｵ髱｢</a>
        </div>
      </form>
    </section>
    <section class="note-grid">
      {blocks_html}
    </section>
  </main>
  <script>
    const copyButtons = document.querySelectorAll(".note-copy-button");
    copyButtons.forEach((button) => {{
      button.addEventListener("click", async () => {{
        const targetId = button.getAttribute("data-copy-target");
        const statusId = button.getAttribute("data-copy-status");
        const target = document.getElementById(targetId);
        const status = document.getElementById(statusId);
        if (!target) return;
        try {{
          await navigator.clipboard.writeText(target.value || "");
          if (status) status.textContent = "蟾ｲ螟榊宛";
        }} catch (error) {{
          target.focus();
          target.select();
          document.execCommand("copy");
          if (status) status.textContent = "蟾ｲ螟榊宛";
        }}
      }});
    }});
  </script>
</body>
</html>""")

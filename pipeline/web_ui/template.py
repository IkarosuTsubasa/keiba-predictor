import html


def page_template(
    output_text="",
    error_text="",
    run_options="",
    view_run_options="",
    view_selected_run_id="",
    top5_text="",
    bet_plan_text="",
    top5_table_html="",
    bet_plan_table_html="",
    summary_table_html="",
    run_summary_block="",
    stats_block="",
    default_scope="central_dirt",
):
    output_block = ""
    if output_text:
        output_block = f"""
        <section class="panel">
            <h2>Output</h2>
            <pre>{html.escape(output_text)}</pre>
        </section>
        """
    if error_text:
        output_block = f"""
        <section class="panel error">
            <h2>Error</h2>
            <pre>{html.escape(error_text)}</pre>
        </section>
        """ + output_block
    run_button_attr = ""

    top5_block = ""
    if top5_table_html:
        top5_block = top5_table_html
    elif top5_text:
        top5_block = f"""
        <section class="panel">
            <h2>Top5 Predictions</h2>
            <pre>{html.escape(top5_text)}</pre>
        </section>
        """

    bet_plan_block = ""
    if bet_plan_table_html:
        bet_plan_block = bet_plan_table_html
    elif bet_plan_text:
        bet_plan_block = f"""
        <section class="panel">
            <h2>Bet Plan</h2>
            <pre>{html.escape(bet_plan_text)}</pre>
        </section>
        """

    summary_block = ""
    if bet_plan_block and summary_table_html:
        bet_plan_block = f"{bet_plan_block}{summary_table_html}"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Keiba Local Console</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --ink: #1f1f1c;
      --accent: #2e6a4f;
      --muted: #6c665f;
      --border: #e6d8c8;
      --shadow: 0 14px 30px rgba(22, 24, 20, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Palatino Linotype", "Book Antiqua", Palatino, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 12% -10%, #f6e5d2 0%, transparent 60%),
        radial-gradient(800px 500px at 90% -5%, #e2f1e8 0%, transparent 55%),
        linear-gradient(180deg, #f8f4ef 0%, #efe6db 100%);
      min-height: 100vh;
    }}
    header {{ padding: 30px 24px 8px; text-align: center; }}
    h1 {{ margin: 0; font-size: 32px; letter-spacing: 0.2px; }}
    .subtitle {{ color: var(--muted); margin-top: 8px; font-size: 13px; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 12px 24px 40px; display: grid; gap: 14px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 14px;
      animation: fadeIn .25s ease;
    }}
    .panel h2 {{ margin: 0 0 10px; font-size: 18px; }}
    label {{ display: block; font-size: 12px; color: var(--muted); margin: 8px 0 6px; }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 14px;
      background: #fffdf9;
      color: var(--ink);
    }}
    textarea {{ min-height: 70px; resize: vertical; }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .radio-group {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .radio-option {{ display: inline-flex; align-items: center; gap: 6px; padding: 0; border: none; background: transparent; cursor: pointer; }}
    .radio-option .radio-text {{
      display: flex; align-items: center; justify-content: center;
      min-width: 96px; padding: 10px 12px; border-radius: 12px;
      border: 1px solid var(--border); background: #f7efe6;
      font-size: 12px; line-height: 1.2; text-align: center;
      transition: transform .2s ease, border .2s ease, background .2s ease, box-shadow .2s ease;
    }}
    .radio-option input {{ position: absolute; opacity: 0; pointer-events: none; }}
    .radio-option input:checked + .radio-text {{
      background: #e8f1ea;
      border-color: rgba(46,106,79,.6);
      color: #1f4b39;
      box-shadow: 0 6px 14px rgba(46,106,79,.18);
      transform: translateY(-1px);
    }}
    .table-wrap {{ overflow-x: auto; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 680px; }}
    .data-table th, .data-table td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
    .data-table th {{ background: #f3e9de; position: sticky; top: 0; }}
    .budget-tab-list {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 12px; }}
    .budget-tab {{
      border: 1px solid var(--border);
      background: #f7efe6;
      color: #5a534c;
      border-radius: 999px;
      padding: 5px 11px;
      font-size: 12px;
      cursor: pointer;
    }}
    .budget-tab.is-active {{
      background: #e8f1ea;
      border-color: rgba(46,106,79,.6);
      color: #1f4b39;
      box-shadow: 0 4px 10px rgba(46,106,79,.16);
    }}
    .budget-group {{ border: 1px dashed var(--border); border-radius: 12px; padding: 10px; margin-bottom: 10px; background: #fffdf9; }}
    .budget-group:last-child {{ margin-bottom: 0; }}
    .budget-group-panel {{ display: none; }}
    .budget-group-panel.is-active {{ display: block; }}
    .budget-group-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 8px; }}
    .budget-group-head h3 {{ margin: 0; font-size: 15px; color: #1f4b39; }}
    .budget-group-meta {{ font-size: 12px; color: var(--muted); }}
    pre {{
      white-space: pre-wrap; background: #f4efe9; padding: 12px;
      border-radius: 10px; border: 1px dashed var(--border);
      margin: 0; max-height: 320px; overflow: auto;
    }}
    .error {{ border-color: #d98b6b; background: #fff6f0; }}
    button {{
      background: linear-gradient(120deg, var(--accent), #1f4b39);
      border: none; color: white; padding: 11px 16px; font-size: 14px;
      border-radius: 12px; cursor: pointer;
    }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @media (max-width: 720px) {{ header {{ padding: 24px 18px 6px; }} .wrap {{ padding: 10px 18px 36px; }} h1 {{ font-size: 26px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Keiba Local Console</h1>
    <div class="subtitle">Run pipeline and record results from a single web UI.</div>
  </header>
  <main class="wrap">
    <section class="panel">
      <h2>Run Pipeline</h2>
      <form action="/run_pipeline" method="post">
        <label>Race ID</label>
        <input name="race_id" inputmode="numeric" pattern="[0-9]*" placeholder="e.g. 202501010101">
        <label>History URL</label>
        <input name="history_url" placeholder="https://db.netkeiba.com/...">
        <label>Data Scope</label>
        <div class="radio-group" id="scope-radio">
          <label class="radio-option"><input type="radio" name="scope_key" value="central_dirt"><span class="radio-text">Central Dirt</span></label>
          <label class="radio-option"><input type="radio" name="scope_key" value="central_turf"><span class="radio-text">Central Turf</span></label>
          <label class="radio-option"><input type="radio" name="scope_key" value="local"><span class="radio-text">Local</span></label>
        </div>
        <div class="grid">
          <div>
            <label>Distance (m)</label>
            <input name="distance" placeholder="1600">
          </div>
          <div>
            <label>Track Condition</label>
            <div class="radio-group" id="track-cond">
              <label class="radio-option"><input type="radio" name="track_cond" value="good" checked><span class="radio-text">Good</span></label>
              <label class="radio-option"><input type="radio" name="track_cond" value="slightly_heavy"><span class="radio-text">Slightly Heavy</span></label>
              <label class="radio-option"><input type="radio" name="track_cond" value="heavy"><span class="radio-text">Heavy</span></label>
              <label class="radio-option"><input type="radio" name="track_cond" value="bad"><span class="radio-text">Bad</span></label>
            </div>
          </div>
        </div>
        <div class="subtitle">Bet Plan: fixed auto budgets 2000 / 5000 / 10000 / 50000.</div>
        <button type="submit" {run_button_attr}>Run</button>
      </form>
    </section>

    <section class="panel">
      <h2>Single Run Actions</h2>
      <form id="single-action-form" action="/view_run" method="post">
        <label>Run ID / Race ID</label>
        <input id="action_id_input" inputmode="text" pattern="[0-9_]*" placeholder="e.g. 202501010101 or 20250101_123456">
        <input type="hidden" id="action_run_id" name="run_id">
        <input type="hidden" id="action_race_id" name="race_id">
        <label>Action Type</label>
        <div class="radio-group" id="action-type">
          <label class="radio-option"><input type="radio" name="action_type" value="view" checked><span class="radio-text">View</span></label>
          <label class="radio-option"><input type="radio" name="action_type" value="update"><span class="radio-text">Update Bet Plan</span></label>
          <label class="radio-option"><input type="radio" name="action_type" value="record"><span class="radio-text">Record Result</span></label>
        </div>
        <div class="subtitle">Data scope is inferred from Run ID / Race ID automatically.</div>
        <div class="subtitle" id="action-update-fields" style="display:none;">Update Bet Plan uses fixed auto budgets: 2000 / 5000 / 10000 / 50000.</div>
        <div class="grid" id="action-record-fields" style="display:none;">
          <div><label>Actual 1st</label><input name="top1"></div>
          <div><label>Actual 2nd</label><input name="top2"></div>
          <div><label>Actual 3rd</label><input name="top3"></div>
        </div>
        <button type="submit" id="action-submit">Run</button>
      </form>
    </section>

    {top5_block}
    {summary_block}
    {bet_plan_block}
    {run_summary_block}
    {stats_block}
    {output_block}
  </main>
  <script>
    const actionForm = document.getElementById("single-action-form");
    const actionInput = document.getElementById("action_id_input");
    const actionRunId = document.getElementById("action_run_id");
    const actionRaceId = document.getElementById("action_race_id");
    const actionTypeRadios = document.querySelectorAll('input[name="action_type"]');
    const updateFields = document.getElementById("action-update-fields");
    const recordFields = document.getElementById("action-record-fields");
    const actionSubmit = document.getElementById("action-submit");

    function syncActionIds() {{
      const value = actionInput ? actionInput.value.trim() : "";
      if (actionRunId) actionRunId.value = value;
      if (actionRaceId) actionRaceId.value = value;
    }}

    function getActionType() {{
      const checked = document.querySelector('input[name="action_type"]:checked');
      return checked ? checked.value : "view";
    }}

    function refreshActionUI() {{
      const actionType = getActionType();
      if (updateFields) updateFields.style.display = actionType === "update" ? "grid" : "none";
      if (recordFields) recordFields.style.display = actionType === "record" ? "grid" : "none";
      if (actionForm) {{
        if (actionType === "update") {{
          actionForm.action = "/update_bet_plan";
          if (actionSubmit) actionSubmit.textContent = "Update";
        }} else if (actionType === "record") {{
          actionForm.action = "/record_pipeline";
          if (actionSubmit) actionSubmit.textContent = "Record";
        }} else {{
          actionForm.action = "/view_run";
          if (actionSubmit) actionSubmit.textContent = "View";
        }}
      }}
    }}

    if (actionInput) {{
      actionInput.addEventListener("input", syncActionIds);
      syncActionIds();
    }}
    if (actionTypeRadios.length) {{
      actionTypeRadios.forEach((radio) => radio.addEventListener("change", refreshActionUI));
    }}
    refreshActionUI();

    const budgetTabs = document.querySelectorAll(".budget-tab");
    if (budgetTabs.length) {{
      budgetTabs.forEach((tab) => {{
        tab.addEventListener("click", () => {{
          const key = tab.getAttribute("data-budget-tab");
          document.querySelectorAll(".budget-tab").forEach((item) => {{
            item.classList.toggle("is-active", item === tab);
          }});
          document.querySelectorAll(".budget-group-panel").forEach((panel) => {{
            panel.classList.toggle("is-active", panel.getAttribute("data-budget-panel") === key);
          }});
        }});
      }});
    }}
  </script>
</body>
</html>
"""

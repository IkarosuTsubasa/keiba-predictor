import html


def _scope_label(scope_key):
    mapping = {
        "central_dirt": "Central Dirt",
        "central_turf": "Central Turf",
        "local": "Local",
    }
    return mapping.get(str(scope_key or "").strip(), "Unspecified")


def _metric_card(label, value):
    if not value:
        return ""
    return (
        '<div class="hero-metric">'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong>"
        "</div>"
    )


def _section_link(section_id, label):
    return f'<a class="jump-link" href="#{html.escape(section_id)}">{html.escape(label)}</a>'


def _cluster(section_id, eyebrow, title, note, content, layout_class=""):
    if not content:
        return ""
    layout_attr = f" {layout_class.strip()}" if layout_class else ""
    return f"""
    <section class="content-cluster" id="{html.escape(section_id)}">
      <div class="cluster-head">
        <div>
          <div class="eyebrow">{html.escape(eyebrow)}</div>
          <h2>{html.escape(title)}</h2>
        </div>
        <p>{html.escape(note)}</p>
      </div>
      <div class="cluster-grid{layout_attr}">
        {content}
      </div>
    </section>
    """


def _fold_cluster(section_id, eyebrow, title, note, content, open_by_default=False):
    if not content:
        return ""
    open_attr = " open" if open_by_default else ""
    return f"""
    <section class="content-cluster" id="{html.escape(section_id)}">
      <details class="fold-panel"{open_attr}>
        <summary>
          <div class="fold-copy">
            <div class="eyebrow">{html.escape(eyebrow)}</div>
            <h2>{html.escape(title)}</h2>
          </div>
          <span class="fold-note">{html.escape(note)}</span>
        </summary>
        <div class="cluster-grid cluster-grid--stack">
          {content}
        </div>
      </details>
    </section>
    """


def page_template(
    output_text="",
    error_text="",
    run_options="",
    view_run_options="",
    view_selected_run_id="",
    top5_text="",
    top5_table_html="",
    mark_table_html="",
    mark_note_text="",
    gemini_policy_html="",
    summary_table_html="",
    stats_block="",
    default_scope="central_dirt",
):
    scope_value = str(default_scope or "central_dirt").strip() or "central_dirt"
    scope_label = _scope_label(scope_value)
    current_run = str(view_selected_run_id or "").strip()
    recent_options = view_run_options or run_options or ""

    output_panel = ""
    if output_text:
        output_panel = f"""
        <section class="panel panel-console">
          <div class="panel-title-row">
            <h3>Console Output</h3>
            <span class="panel-tag">stdout</span>
          </div>
          <pre>{html.escape(output_text)}</pre>
        </section>
        """

    error_panel = ""
    if error_text:
        error_panel = f"""
        <section class="panel panel-error">
          <div class="panel-title-row">
            <h3>Error</h3>
            <span class="panel-tag panel-tag-danger">needs attention</span>
          </div>
          <pre>{html.escape(error_text)}</pre>
        </section>
        """

    top5_block = ""
    if top5_table_html:
        top5_block = top5_table_html
    elif top5_text:
        top5_block = f"""
        <section class="panel">
          <h3>Top5 Predictions</h3>
          <pre>{html.escape(top5_text)}</pre>
        </section>
        """

    note_copy_panel = ""
    if mark_note_text:
        note_copy_panel = f"""
        <section class="panel panel-note-copy">
          <div class="section-title">
            <div>
              <div class="eyebrow">Utility</div>
              <h2>Note Copy</h2>
            </div>
            <span class="section-chip">clipboard</span>
          </div>
          <div class="copy-row">
            <button type="button" class="secondary-button" id="copy-note-marks">Copy Note</button>
            <span id="copy-note-status" class="copy-status"></span>
          </div>
          <textarea id="note-marks-text" class="hidden-copy-source" readonly>{html.escape(mark_note_text)}</textarea>
          <details class="mini-fold">
            <summary>Preview note text</summary>
            <pre>{html.escape(mark_note_text)}</pre>
          </details>
        </section>
        """

    mark_block = mark_table_html or ""

    analysis_cluster = _cluster(
        "analysis-zone",
        "Model Analysis",
        "Prediction Workspace",
        "Keep prediction tables, marks, and model status together.",
        f"{top5_block}{mark_block}{summary_table_html or ''}",
        "cluster-grid--double",
    )
    policy_cluster = _cluster(
        "policy-zone",
        "Gemini",
        "Policy Workspace",
        "Gemini policy output stays available without the old bet plan tables.",
        gemini_policy_html or "",
        "cluster-grid--stack",
    )
    stats_cluster = _fold_cluster(
        "stats-zone",
        "Performance",
        "Predictor Stats",
        "Historical predictor summary is secondary by default.",
        stats_block or "",
        open_by_default=False,
    )
    console_cluster = _fold_cluster(
        "console-zone",
        "Diagnostics",
        "Run Log",
        "Execution traces and validation messages live here.",
        f"{error_panel}{output_panel}",
        open_by_default=bool(error_text or output_text),
    )

    jump_links = []
    if analysis_cluster:
        jump_links.append(_section_link("analysis-zone", "Analysis"))
    if policy_cluster:
        jump_links.append(_section_link("policy-zone", "Policy"))
    if stats_cluster:
        jump_links.append(_section_link("stats-zone", "Stats"))
    if console_cluster:
        jump_links.append(_section_link("console-zone", "Log"))

    hero_metrics = "".join(
        [
            _metric_card("Scope", scope_label),
            _metric_card("Active Run", current_run or "Not Selected"),
            _metric_card("Panels", str(len(jump_links)) if jump_links else "Controls Only"),
            _metric_card("State", "Error" if error_text else ("Live Output" if output_text else "Ready")),
        ]
    )

    recent_runs_panel = ""
    if recent_options:
        recent_runs_panel = f"""
        <section class="panel panel-compact">
          <div class="section-title">
            <div>
              <div class="eyebrow">Quick Access</div>
              <h2>Recent Runs</h2>
            </div>
            <span class="section-chip" id="recent-run-status">scope: {html.escape(scope_label)}</span>
          </div>
          <form action="/view_run" method="post" class="stack-form">
            <input type="hidden" name="scope_key" id="recent_scope_key" value="{html.escape(scope_value)}">
            <label>Latest Run Snapshot</label>
            <select name="run_id" id="recent_run_select">
              {recent_options}
            </select>
            <button type="submit">Open Selected Run</button>
          </form>
        </section>
        """

    empty_state = ""
    if not any([analysis_cluster, policy_cluster, stats_cluster, console_cluster]):
        empty_state = """
        <section class="empty-state panel">
          <div class="eyebrow">Workspace</div>
          <h2>Prediction First</h2>
          <p>
            Start with a new pipeline run on the left, or open a recent run to inspect predictions,
            marks, Gemini policy output, and model status.
          </p>
        </section>
        """

    central_dirt_checked = " checked" if scope_value == "central_dirt" else ""
    central_turf_checked = " checked" if scope_value == "central_turf" else ""
    local_checked = " checked" if scope_value == "local" else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Keiba Local Console</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --bg-strong: #e5ddd0;
      --surface: rgba(255, 250, 243, 0.88);
      --ink: #1f2321;
      --ink-soft: #4a514d;
      --muted: #6a716b;
      --accent: #1e5a46;
      --accent-strong: #123b30;
      --danger: #b8573f;
      --danger-soft: rgba(184, 87, 63, 0.12);
      --shadow: 0 22px 60px rgba(25, 31, 28, 0.10);
      --hero-shadow: 0 28px 80px rgba(23, 30, 28, 0.14);
      --title-font: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --body-font: "Aptos", "Segoe UI Variable Text", "Trebuchet MS", "Yu Gothic UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: var(--body-font);
      color: var(--ink);
      min-height: 100vh;
      background:
        radial-gradient(1200px 500px at -5% -10%, rgba(226, 208, 181, 0.75) 0%, transparent 60%),
        radial-gradient(960px 620px at 100% 0%, rgba(206, 227, 217, 0.72) 0%, transparent 58%),
        linear-gradient(180deg, #f7f3ec 0%, var(--bg) 40%, var(--bg-strong) 100%);
    }}
    a {{ color: inherit; }}
    .page-shell {{ max-width: 1520px; margin: 0 auto; padding: 28px 22px 44px; }}
    .hero {{
      position: relative;
      overflow: hidden;
      border-radius: 32px;
      padding: 30px;
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(300px, 1fr);
      gap: 24px;
      background:
        linear-gradient(135deg, rgba(255, 248, 238, 0.94), rgba(240, 247, 242, 0.92)),
        linear-gradient(180deg, rgba(255, 255, 255, 0.45), rgba(255, 255, 255, 0.0));
      border: 1px solid rgba(255, 255, 255, 0.55);
      box-shadow: var(--hero-shadow);
      backdrop-filter: blur(16px);
    }}
    .hero-copy {{ position: relative; z-index: 1; display: grid; gap: 14px; align-content: start; }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .hero h1,
    .content-cluster h2,
    .panel h2,
    .panel h3,
    .empty-state h2 {{
      font-family: var(--title-font);
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    .hero h1 {{ margin: 0; font-size: clamp(34px, 5vw, 54px); line-height: 0.96; }}
    .hero p {{ margin: 0; max-width: 62ch; color: var(--ink-soft); font-size: 15px; line-height: 1.65; }}
    .hero-subline {{ display: flex; flex-wrap: wrap; gap: 10px; padding-top: 4px; }}
    .hero-pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(255, 255, 255, 0.78);
      color: var(--ink-soft);
      font-size: 12px;
    }}
    .hero-metrics {{ position: relative; z-index: 1; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .hero-metric {{
      padding: 16px 18px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid rgba(255, 255, 255, 0.82);
      min-height: 98px;
      display: grid;
      gap: 10px;
      align-content: space-between;
    }}
    .metric-label {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .hero-metric strong {{ font-size: 20px; line-height: 1.18; word-break: break-word; }}
    .jump-links {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 4px 0; }}
    .jump-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 0 14px;
      border-radius: 999px;
      border: 1px solid rgba(61, 68, 62, 0.11);
      background: rgba(255, 250, 243, 0.72);
      color: var(--ink-soft);
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
    }}
    .app-shell {{ display: grid; grid-template-columns: 360px minmax(0, 1fr); gap: 22px; margin-top: 22px; align-items: start; }}
    .control-rail {{ position: sticky; top: 18px; display: grid; gap: 16px; }}
    .content-stage {{ display: grid; gap: 18px; }}
    .panel, .fold-panel {{
      background: var(--surface);
      border: 1px solid rgba(255, 255, 255, 0.58);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .panel {{ border-radius: 22px; padding: 18px; }}
    .panel-hero {{
      background:
        linear-gradient(155deg, rgba(255, 255, 255, 0.82), rgba(246, 239, 227, 0.9)),
        linear-gradient(180deg, rgba(30, 90, 70, 0.06), rgba(30, 90, 70, 0.00));
    }}
    .panel-compact {{ padding: 16px; }}
    .panel-console pre, .panel-error pre {{ max-height: 420px; }}
    .panel-error {{
      border-color: rgba(184, 87, 63, 0.28);
      background: linear-gradient(160deg, rgba(255, 245, 241, 0.95), rgba(255, 250, 243, 0.92));
    }}
    .panel-note-copy {{
      background: linear-gradient(160deg, rgba(247, 244, 255, 0.74), rgba(255, 250, 243, 0.88));
    }}
    .policy-panel {{ display: grid; gap: 16px; }}
    .policy-meta-row {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
    .policy-meta-tag {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      background: rgba(30, 90, 70, 0.10);
      border: 1px solid rgba(30, 90, 70, 0.12);
      color: var(--accent-strong);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}
    .policy-block {{
      display: grid;
      gap: 14px;
      padding: 16px;
      border-radius: 18px;
      background: linear-gradient(160deg, rgba(255, 255, 255, 0.70), rgba(241, 247, 243, 0.62));
      border: 1px solid rgba(61, 68, 62, 0.08);
    }}
    .policy-summary {{ display: grid; gap: 8px; }}
    .policy-summary-tags {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .policy-summary-line {{ color: var(--muted); font-size: 12px; }}
    .policy-horse-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .policy-horse-group {{
      display: grid;
      gap: 10px;
      padding: 14px;
      border-radius: 16px;
      background: rgba(255, 252, 246, 0.84);
      border: 1px solid rgba(61, 68, 62, 0.08);
    }}
    .policy-label {{
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .policy-chip-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .policy-chip {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 38px;
      min-height: 38px;
      padding: 0 12px;
      border-radius: 999px;
      background: rgba(61, 68, 62, 0.08);
      color: var(--ink);
      font-size: 16px;
      font-weight: 800;
    }}
    .policy-chip--key {{
      background: linear-gradient(135deg, #1e5a46, #123b30);
      color: #fff;
      box-shadow: 0 10px 24px rgba(18, 59, 48, 0.18);
    }}
    .policy-chip--secondary {{
      background: rgba(30, 90, 70, 0.12);
      color: var(--accent-strong);
    }}
    .policy-chip--longshot {{
      background: rgba(184, 87, 63, 0.12);
      color: var(--danger);
    }}
    .policy-chip--empty {{
      background: rgba(61, 68, 62, 0.06);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      min-width: 64px;
    }}
    .policy-text-grid {{ display: grid; gap: 12px; grid-template-columns: minmax(0, 1.4fr) minmax(240px, 0.9fr); }}
    .policy-text-card {{
      display: grid;
      gap: 8px;
      padding: 16px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(61, 68, 62, 0.08);
    }}
    .policy-text-card--primary {{
      background: linear-gradient(165deg, rgba(246, 252, 249, 0.96), rgba(255, 250, 243, 0.92));
      border-color: rgba(30, 90, 70, 0.16);
    }}
    .policy-text-card p {{ margin: 0; line-height: 1.7; color: var(--ink-soft); }}
    .policy-fold {{
      border-radius: 16px;
      border: 1px solid rgba(61, 68, 62, 0.10);
      background: rgba(255, 255, 255, 0.56);
      overflow: hidden;
    }}
    .policy-fold summary {{
      list-style: none;
      cursor: pointer;
      padding: 13px 14px;
      font-size: 13px;
      font-weight: 700;
      color: var(--ink-soft);
    }}
    .policy-fold summary::-webkit-details-marker {{ display: none; }}
    .policy-detail-grid {{ display: grid; gap: 10px; padding: 0 14px 14px; }}
    .policy-detail-row {{
      display: grid;
      gap: 6px;
      padding-top: 10px;
      border-top: 1px dashed rgba(61, 68, 62, 0.12);
    }}
    .policy-detail-row:first-child {{ border-top: none; padding-top: 0; }}
    .policy-detail-row strong {{
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .policy-detail-row span, .policy-detail-row code {{
      color: var(--ink-soft);
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .section-title, .cluster-head, .panel-title-row {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
    }}
    .section-title h2, .cluster-head h2, .panel h2, .panel h3 {{ margin: 0; font-size: 28px; line-height: 1.02; }}
    .panel h2, .panel h3 {{ font-size: 21px; margin-bottom: 10px; }}
    .cluster-head p, .fold-note {{ margin: 0; max-width: 44ch; color: var(--muted); font-size: 13px; line-height: 1.6; }}
    .section-chip, .panel-tag {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      background: rgba(30, 90, 70, 0.10);
      border: 1px solid rgba(30, 90, 70, 0.14);
      white-space: nowrap;
    }}
    .panel-tag-danger {{ color: var(--danger); background: var(--danger-soft); border-color: rgba(184, 87, 63, 0.20); }}
    .stack-form {{ display: grid; gap: 12px; }}
    .field-grid, .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    label {{
      display: block;
      margin: 0 0 6px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      color: var(--ink-soft);
      text-transform: uppercase;
    }}
    input, select, textarea {{
      width: 100%;
      min-height: 46px;
      padding: 12px 13px;
      border-radius: 14px;
      border: 1px solid rgba(61, 68, 62, 0.10);
      background: rgba(255, 255, 255, 0.84);
      color: var(--ink);
      font: inherit;
    }}
    textarea {{ min-height: 76px; resize: vertical; }}
    .helper-text {{ margin: 0; color: var(--muted); font-size: 12px; line-height: 1.6; }}
    .radio-group {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .radio-option {{ position: relative; display: inline-flex; align-items: center; cursor: pointer; }}
    .radio-option input {{ position: absolute; opacity: 0; pointer-events: none; }}
    .radio-text {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 108px;
      min-height: 44px;
      padding: 0 14px;
      border-radius: 999px;
      border: 1px solid rgba(61, 68, 62, 0.11);
      background: rgba(252, 247, 240, 0.84);
      color: var(--ink-soft);
      font-size: 13px;
      font-weight: 700;
      text-align: center;
    }}
    .radio-option input:checked + .radio-text {{
      background: linear-gradient(140deg, rgba(232, 243, 237, 0.94), rgba(246, 239, 227, 0.90));
      border-color: rgba(30, 90, 70, 0.28);
      color: var(--accent-strong);
    }}
    button {{
      min-height: 46px;
      border: none;
      border-radius: 14px;
      padding: 0 16px;
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .secondary-button {{ background: rgba(255, 255, 255, 0.9); color: var(--accent-strong); border: 1px solid rgba(30, 90, 70, 0.16); }}
    .content-cluster {{ display: grid; gap: 12px; }}
    .cluster-grid {{ display: grid; gap: 16px; grid-template-columns: minmax(0, 1fr); }}
    .cluster-grid--double {{ grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }}
    .cluster-grid--stack {{ grid-template-columns: minmax(0, 1fr); padding-top: 14px; }}
    .table-wrap {{ overflow-x: auto; border-radius: 14px; border: 1px solid rgba(61, 68, 62, 0.08); }}
    .table-wrap--fit {{ display: inline-block; max-width: 100%; }}
    .table-wrap--narrow {{ width: min(100%, 780px); }}
    .table-wrap--medium {{ width: min(100%, 980px); }}
    .data-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 680px;
      font-size: 13px;
      background: rgba(255, 255, 255, 0.75);
    }}
    .data-table--compact {{
      min-width: 0;
      width: auto;
    }}
    .data-table--narrow {{
      min-width: 460px;
    }}
    .data-table--medium {{
      min-width: 560px;
    }}
    .data-table th, .data-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(61, 68, 62, 0.08);
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: linear-gradient(180deg, #f8f1e6, #f1e6d7);
      color: var(--ink-soft);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .copy-row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
    .copy-status {{ min-height: 18px; font-size: 12px; color: var(--muted); }}
    .mini-fold, .fold-panel {{ border-radius: 22px; }}
    .mini-fold {{ margin-top: 8px; border: 1px solid rgba(61, 68, 62, 0.10); background: rgba(255, 255, 255, 0.45); overflow: hidden; }}
    .mini-fold summary, .fold-panel summary {{ list-style: none; cursor: pointer; }}
    .mini-fold summary::-webkit-details-marker, .fold-panel summary::-webkit-details-marker {{ display: none; }}
    .mini-fold summary {{ padding: 12px 14px; font-size: 13px; font-weight: 700; color: var(--ink-soft); }}
    .mini-fold pre {{ margin: 0 14px 14px; }}
    .fold-panel {{ border-radius: 22px; }}
    .fold-panel summary {{ list-style: none; cursor: pointer; }}
    .fold-panel summary::-webkit-details-marker {{ display: none; }}
    .fold-panel {{ overflow: hidden; }}
    .fold-panel summary {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 18px 20px; }}
    .fold-panel summary::after {{
      content: "+";
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: rgba(30, 90, 70, 0.10);
      color: var(--accent-strong);
      font-size: 20px;
      flex: none;
    }}
    .fold-panel[open] summary::after {{ content: "−"; }}
    .fold-copy {{ display: grid; gap: 6px; }}
    pre {{
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      border: 1px dashed rgba(61, 68, 62, 0.14);
      background: rgba(244, 238, 230, 0.74);
      overflow: auto;
      white-space: pre-wrap;
      line-height: 1.55;
      max-height: 340px;
    }}
    .hidden-copy-source {{ position: absolute; left: -9999px; top: -9999px; width: 1px; height: 1px; opacity: 0; pointer-events: none; }}
    .empty-state {{ padding: 26px; background: linear-gradient(155deg, rgba(255, 255, 255, 0.86), rgba(239, 246, 242, 0.78)); }}
    .empty-state p {{ margin: 0; max-width: 56ch; color: var(--ink-soft); line-height: 1.7; }}
    @media (max-width: 1180px) {{
      .hero {{ grid-template-columns: minmax(0, 1fr); }}
      .app-shell {{ grid-template-columns: minmax(0, 1fr); }}
      .control-rail {{ position: static; }}
    }}
    @media (max-width: 760px) {{
      .page-shell {{ padding: 16px 14px 30px; }}
      .hero {{ padding: 22px 18px; border-radius: 24px; }}
      .hero-metrics {{ grid-template-columns: minmax(0, 1fr); }}
      .section-title, .cluster-head, .panel-title-row, .fold-panel summary {{ flex-direction: column; align-items: flex-start; }}
      .cluster-grid--double {{ grid-template-columns: minmax(0, 1fr); }}
      .panel, .fold-panel summary {{ padding-left: 16px; padding-right: 16px; }}
      .table-wrap--fit, .table-wrap--narrow, .table-wrap--medium {{ width: 100%; max-width: 100%; }}
      .data-table--compact {{ width: 100%; }}
      .policy-meta-row {{ justify-content: flex-start; }}
      .policy-text-grid {{ grid-template-columns: minmax(0, 1fr); }}
    }}
  </style>
</head>
<body>
  <div class="page-shell">
    <header class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Keiba Workstation</div>
        <h1>Keiba Local Console</h1>
        <p>One focused workspace for launching runs, reviewing predictions, inspecting marks, and checking Gemini policy output without carrying the old betting workflow.</p>
        <div class="hero-subline">
          <span class="hero-pill" id="scope-pill">Current scope: {html.escape(scope_label)}</span>
          <span class="hero-pill">{html.escape("Recent runs ready" if recent_options else "Open a scope to load run history")}</span>
          <span class="hero-pill">{html.escape("Run selected" if current_run else "No active run yet")}</span>
        </div>
      </div>
      <div class="hero-metrics">{hero_metrics}</div>
    </header>
    <nav class="jump-links" aria-label="Page sections">{''.join(jump_links)}</nav>
    <div class="app-shell">
      <aside class="control-rail">
        <section class="panel panel-hero">
          <div class="section-title">
            <div>
              <div class="eyebrow">Pipeline</div>
              <h2>Run Pipeline</h2>
            </div>
            <span class="section-chip">new run</span>
          </div>
          <form action="/run_pipeline" method="post" class="stack-form">
            <div>
              <label>Race ID</label>
              <input name="race_id" inputmode="numeric" pattern="[0-9]*" placeholder="e.g. 202501010101">
            </div>
            <div>
              <label>History URL</label>
              <input name="history_url" placeholder="https://db.netkeiba.com/...">
            </div>
            <div>
              <label>Data Scope</label>
              <div class="radio-group" id="scope-radio">
                <label class="radio-option"><input type="radio" name="scope_key" value="central_dirt"{central_dirt_checked}><span class="radio-text">Central Dirt</span></label>
                <label class="radio-option"><input type="radio" name="scope_key" value="central_turf"{central_turf_checked}><span class="radio-text">Central Turf</span></label>
                <label class="radio-option"><input type="radio" name="scope_key" value="local"{local_checked}><span class="radio-text">Local</span></label>
              </div>
            </div>
            <div class="field-grid">
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
            <p class="helper-text">Pipeline runs now focus on prediction artifacts and Gemini policy output.</p>
            <button type="submit">Run Pipeline</button>
          </form>
        </section>
        <section class="panel">
          <div class="section-title">
            <div>
              <div class="eyebrow">Inspection</div>
              <h2>Open Run</h2>
            </div>
            <span class="section-chip">view</span>
          </div>
          <form id="single-action-form" action="/view_run" method="post" class="stack-form">
            <div>
              <label>Run ID / Race ID</label>
              <input id="action_id_input" inputmode="text" pattern="[0-9_]*" placeholder="e.g. 202501010101 or 20250101_123456">
              <input type="hidden" id="action_run_id" name="run_id">
            </div>
            <p class="helper-text">Scope is inferred automatically from the entered run or race identifier.</p>
            <button type="submit" id="action-submit">View Run</button>
          </form>
        </section>
        <section class="panel">
          <div class="section-title">
            <div>
              <div class="eyebrow">Predictor</div>
              <h2>Record Result</h2>
            </div>
            <span class="section-chip">stats</span>
          </div>
          <form action="/record_predictor" method="post" class="stack-form">
            <input type="hidden" name="scope_key" id="record_scope_key" value="{html.escape(scope_value)}">
            <div>
              <label>Run ID</label>
              <input name="run_id" id="record_run_id" inputmode="text" pattern="[0-9_]*" placeholder="e.g. 20250101_123456">
            </div>
            <div class="field-grid">
              <div>
                <label>Top1</label>
                <input name="top1" inputmode="numeric" pattern="[0-9]*" placeholder="1">
              </div>
              <div>
                <label>Top2</label>
                <input name="top2" inputmode="numeric" pattern="[0-9]*" placeholder="2">
              </div>
              <div>
                <label>Top3</label>
                <input name="top3" inputmode="numeric" pattern="[0-9]*" placeholder="3">
              </div>
            </div>
            <p class="helper-text">Record the actual top-3 finish to refresh predictor accuracy stats.</p>
            <button type="submit">Record Predictor</button>
          </form>
        </section>
        {note_copy_panel}
        {recent_runs_panel}
      </aside>
      <main class="content-stage">
        {analysis_cluster}
        {policy_cluster}
        {stats_cluster}
        {console_cluster}
        {empty_state}
      </main>
    </div>
  </div>
  <script>
    const defaultScope = "{html.escape(scope_value)}";
    const scopeLabels = {{ central_dirt: "Central Dirt", central_turf: "Central Turf", local: "Local" }};
    const actionForm = document.getElementById("single-action-form");
    const actionInput = document.getElementById("action_id_input");
    const actionRunId = document.getElementById("action_run_id");
    const actionSubmit = document.getElementById("action-submit");
    const recordScopeInput = document.getElementById("record_scope_key");
    const recordRunInput = document.getElementById("record_run_id");
    const scopeRadios = document.querySelectorAll('#scope-radio input[name="scope_key"]');
    const recentSelect = document.getElementById("recent_run_select");
    const recentScopeInput = document.getElementById("recent_scope_key");
    const recentRunStatus = document.getElementById("recent-run-status");
    const scopePill = document.getElementById("scope-pill");

    function syncActionIds() {{
      const value = actionInput ? actionInput.value.trim() : "";
      if (actionRunId) actionRunId.value = value;
      if (recordRunInput && !recordRunInput.value.trim()) recordRunInput.value = value;
    }}

    function getSelectedScope() {{
      const checked = document.querySelector('#scope-radio input[name="scope_key"]:checked');
      return checked ? checked.value : defaultScope;
    }}

    function updateScopeLabels(scopeKey, detailText) {{
      const label = scopeLabels[scopeKey] || "Unspecified";
      if (scopePill) scopePill.textContent = `Current scope: ${{label}}`;
      if (recentRunStatus) recentRunStatus.textContent = detailText || `scope: ${{label}}`;
      if (recentScopeInput) recentScopeInput.value = scopeKey;
      if (recordScopeInput) recordScopeInput.value = scopeKey;
    }}

    async function refreshRecentRuns(scopeKey) {{
      if (!recentSelect || !scopeKey) {{
        updateScopeLabels(scopeKey || defaultScope);
        return;
      }}
      recentSelect.disabled = true;
      updateScopeLabels(scopeKey, "loading...");
      try {{
        const resp = await fetch(`/api/runs?scope_key=${{encodeURIComponent(scopeKey)}}&limit=30`);
        if (!resp.ok) throw new Error(`HTTP ${{resp.status}}`);
        const items = await resp.json();
        recentSelect.innerHTML = "";
        if (Array.isArray(items) && items.length) {{
          items.forEach((item, index) => {{
            const opt = document.createElement("option");
            opt.value = item.run_id || "";
            opt.textContent = item.label || item.run_id || "";
            if (index === 0) opt.selected = true;
            recentSelect.appendChild(opt);
          }});
          updateScopeLabels(scopeKey, `${{items.length}} recent runs`);
          if (actionInput && !actionInput.value.trim()) {{
            actionInput.value = recentSelect.value;
            syncActionIds();
          }}
        }} else {{
          const opt = document.createElement("option");
          opt.value = "";
          opt.textContent = "No runs";
          opt.disabled = true;
          opt.selected = true;
          recentSelect.appendChild(opt);
          updateScopeLabels(scopeKey, "no runs");
        }}
      }} catch (error) {{
        updateScopeLabels(scopeKey, "history unavailable");
      }} finally {{
        recentSelect.disabled = false;
      }}
    }}

    if (actionInput) {{
      actionInput.addEventListener("input", syncActionIds);
      syncActionIds();
    }}

    if (actionForm && actionSubmit) {{
      actionForm.action = "/view_run";
      actionSubmit.textContent = "View Run";
    }}

    if (recentSelect) {{
      recentSelect.addEventListener("change", () => {{
        if (actionInput && recentSelect.value) {{
          actionInput.value = recentSelect.value;
          syncActionIds();
        }}
      }});
    }}

    if (scopeRadios.length) {{
      scopeRadios.forEach((radio) => {{
        radio.addEventListener("change", () => {{
          refreshRecentRuns(getSelectedScope());
        }});
      }});
      updateScopeLabels(getSelectedScope());
    }}

    const copyNoteBtn = document.getElementById("copy-note-marks");
    const copyNoteStatus = document.getElementById("copy-note-status");
    const copySource = document.getElementById("note-marks-text");
    if (copyNoteBtn && copySource) {{
      copyNoteBtn.addEventListener("click", async () => {{
        const text = copySource.value || "";
        if (!text) {{
          if (copyNoteStatus) copyNoteStatus.textContent = "No note text available.";
          return;
        }}
        try {{
          if (navigator.clipboard && window.isSecureContext) {{
            await navigator.clipboard.writeText(text);
          }} else {{
            copySource.focus();
            copySource.select();
            document.execCommand("copy");
          }}
          if (copyNoteStatus) copyNoteStatus.textContent = "Copied.";
          setTimeout(() => {{
            if (copyNoteStatus) copyNoteStatus.textContent = "";
          }}, 1500);
        }} catch (e) {{
          if (copyNoteStatus) copyNoteStatus.textContent = "Copy failed. Please copy manually.";
        }}
      }});
    }}

  </script>
</body>
</html>
"""

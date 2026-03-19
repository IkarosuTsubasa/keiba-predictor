import html
import re

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
  <title>控制台入口</title>
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
      <h1>控制台入口</h1>
      <p>请输入管理员 Token 进入 `/console`。</p>
      <form method="get" action="/console">
        <input type="password" name="token" placeholder="ADMIN_TOKEN" value="{html.escape(admin_token)}">
        <div class="actions">
          <button type="submit">进入控制台</button>
          <a href="/llm_today">返回公开页</a>
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

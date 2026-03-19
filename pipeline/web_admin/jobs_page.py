import html
from datetime import datetime, timedelta


def race_job_status_label_clean(status):
    mapping = {
        "uploaded": "已上传",
        "scheduled": "已计划",
        "queued_process": "待处理",
        "processing": "处理中",
        "waiting_v5": "等待远程预测",
        "queued_policy": "等待 LLM",
        "processing_policy": "LLM 处理中",
        "ready": "待结算",
        "queued_settle": "待结算",
        "settling": "结算中",
        "settled": "已结算",
        "failed": "失败",
    }
    text = str(status or "").strip().lower()
    return mapping.get(text, text or "-")


def race_job_settle_form_clean(row, admin_token=""):
    current_run_id = str((row or {}).get("current_run_id", "") or "").strip()
    if not current_run_id:
        return ""
    status_text = str((row or {}).get("status", "") or "").strip().lower()
    if status_text not in ("ready", "queued_settle", "settling", "settled", "failed"):
        return ""
    if status_text == "settled":
        return ""
    job_id = str((row or {}).get("job_id", "") or "").strip()
    top1 = str((row or {}).get("actual_top1", "") or "").strip()
    top2 = str((row or {}).get("actual_top2", "") or "").strip()
    top3 = str((row or {}).get("actual_top3", "") or "").strip()
    return f"""
    <section class="job-settle-panel">
      <div class="job-settle-head">
        <strong>Step 3: 录入赛果并结算</strong>
        <span>请填写 1-3 着马名后执行结算。</span>
      </div>
      <div class="job-settle-actions">
        <form method="post" action="/console/tasks/settle_now" class="job-settle-form">
          <input type="hidden" name="job_id" value="{html.escape(job_id)}">
          <input type="hidden" name="token" value="{html.escape(admin_token)}">
          <input type="text" name="actual_top1" value="{html.escape(top1)}" placeholder="1着马名">
          <input type="text" name="actual_top2" value="{html.escape(top2)}" placeholder="2着马名">
          <input type="text" name="actual_top3" value="{html.escape(top3)}" placeholder="3着马名">
          <button type="submit">执行结算</button>
        </form>
      </div>
    </section>
    """


def race_job_edit_form_clean(row, admin_token="", default_job_race_date_text=None):
    row = dict(row or {})
    job_id = str(row.get("job_id", "") or "").strip()
    race_id = str(row.get("race_id", "") or "").strip()
    location = str(row.get("location", "") or "").strip()
    default_date = default_job_race_date_text() if callable(default_job_race_date_text) else ""
    race_date = str(row.get("race_date", "") or "").strip() or default_date
    scheduled_off_time = str(row.get("scheduled_off_time", "") or "").strip()
    lead_minutes = str(row.get("lead_minutes", 30) or 30).strip() or "30"
    target_distance = str(row.get("target_distance", "") or "").strip()
    target_track_condition = str(row.get("target_track_condition", "") or "").strip()
    notes = str(row.get("notes", "") or "").strip()
    return f"""
    <details class="panel-subtle" style="margin-top:12px;">
      <summary style="cursor:pointer;font-weight:700;">编辑任务信息</summary>
      <form method="post" action="/console/tasks/edit" class="stack-form" style="margin-top:12px;">
        <input type="hidden" name="token" value="{html.escape(admin_token)}">
        <input type="hidden" name="job_id" value="{html.escape(job_id)}">
        <div class="field-grid">
          <div>
            <label>Race ID</label>
            <input type="text" name="race_id" value="{html.escape(race_id)}">
          </div>
          <div>
            <label>场地</label>
            <input type="text" name="location" value="{html.escape(location)}">
          </div>
          <div>
            <label>比赛日期</label>
            <input type="date" name="race_date" value="{html.escape(race_date)}">
          </div>
          <div>
            <label>开赛时间</label>
            <input type="datetime-local" name="scheduled_off_time" value="{html.escape(scheduled_off_time[:16])}">
          </div>
          <div>
            <label>提前分钟</label>
            <input type="number" name="lead_minutes" min="0" value="{html.escape(lead_minutes)}">
          </div>
          <div>
            <label>距离</label>
            <input type="number" name="target_distance" min="100" step="100" value="{html.escape(target_distance)}">
          </div>
          <div>
            <label>马场状态</label>
            <select name="target_track_condition">
              <option value="良"{' selected' if target_track_condition == '良' else ''}>良</option>
              <option value="稍重"{' selected' if target_track_condition == '稍重' else ''}>稍重</option>
              <option value="重"{' selected' if target_track_condition == '重' else ''}>重</option>
              <option value="不良"{' selected' if target_track_condition == '不良' else ''}>不良</option>
            </select>
          </div>
        </div>
        <div>
          <label>备注</label>
          <textarea name="notes">{html.escape(notes)}</textarea>
        </div>
        <button type="submit" class="secondary-button">保存修改</button>
      </form>
    </details>
    """


def admin_job_card_html_clean(
    row,
    *,
    admin_token="",
    scope_display_name,
    race_job_action_buttons_v2,
    race_job_status_label_clean,
    race_job_edit_form_clean,
    race_job_settle_form_clean,
    default_job_race_date_text,
):
    row = dict(row or {})
    job_id = str(row.get("job_id", "") or "").strip()
    status = str(row.get("status", "") or "").strip()
    current_run_id = str(row.get("current_run_id", "") or "").strip()
    scope_key = str(row.get("scope_key", "") or "").strip()
    race_date = str(row.get("race_date", "") or "").strip()
    location = str(row.get("location", "") or "").strip()
    race_id = str(row.get("race_id", "") or "").strip()
    notes = str(row.get("notes", "") or "").strip()
    actual_top1 = str(row.get("actual_top1", "") or "").strip()
    actual_top2 = str(row.get("actual_top2", "") or "").strip()
    actual_top3 = str(row.get("actual_top3", "") or "").strip()
    artifacts = list(row.get("artifacts", []) or [])
    artifact_map = {
        str(item.get("artifact_type", "")).strip().lower(): dict(item)
        for item in artifacts
        if isinstance(item, dict)
    }
    kachiuma_name = str((artifact_map.get("kachiuma") or {}).get("original_name", "") or "未上传")
    shutuba_name = str((artifact_map.get("shutuba") or {}).get("original_name", "") or "未上传")
    timing_items = []
    for label, key in (
        ("开赛", "scheduled_off_time"),
        ("开始处理", "process_after_time"),
        ("进入队列", "queued_process_at"),
        ("准备完成", "ready_at"),
        ("结算完成", "settled_at"),
    ):
        value = str(row.get(key, "") or "").strip()
        if value:
            timing_items.append(f"<span>{html.escape(label)} {html.escape(value)}</span>")
    open_links = ""
    if current_run_id:
        open_links = f"""
        <form method="post" action="/view_run" class="stack-form">
          <input type="hidden" name="scope_key" value="{html.escape(scope_key)}">
          <input type="hidden" name="run_id" value="{html.escape(current_run_id)}">
          <input type="hidden" name="token" value="{html.escape(admin_token)}">
          <button type="submit" class="secondary-button">查看 Run</button>
        </form>
        <form method="get" action="/llm_today" class="stack-form">
          <input type="hidden" name="scope_key" value="{html.escape(scope_key)}">
          <input type="hidden" name="date" value="{html.escape(race_date)}">
          <button type="submit" class="secondary-button">打开公开页</button>
        </form>
        """
    result_text = " / ".join(x for x in [actual_top1, actual_top2, actual_top3] if x) or "未录入"
    title_text = (location + " " + race_id).strip() or job_id or "未命名任务"
    return f"""
    <section class="panel panel--tight">
      <div class="section-title">
        <div>
          <div class="eyebrow">Task</div>
          <h2>{html.escape(title_text)}</h2>
        </div>
        <span class="section-chip">{html.escape(race_job_status_label_clean(status))}</span>
      </div>
      <div class="copy-row">
        <span class="hero-pill">范围 {html.escape(scope_display_name(scope_key))}</span>
        <span class="hero-pill">日期 {html.escape(race_date or '-')}</span>
        <span class="hero-pill">条件 {html.escape(str(row.get('target_surface', '') or '-'))} / {html.escape(str(row.get('target_distance', '') or '-'))}m / {html.escape(str(row.get('target_track_condition', '') or '-'))}</span>
        <span class="hero-pill">Run {html.escape(current_run_id or '-')}</span>
        <span class="hero-pill">赛果 {html.escape(result_text)}</span>
      </div>
      <div class="copy-row">
        <span class="hero-pill">kachiuma: {html.escape(kachiuma_name)}</span>
        <span class="hero-pill">shutuba: {html.escape(shutuba_name)}</span>
        {''.join(timing_items)}
      </div>
      <p class="helper-text">{html.escape(notes or '无备注')}</p>
      <div class="copy-row">
        {race_job_action_buttons_v2(job_id, status, admin_token=admin_token)}
        {open_links}
      </div>
      {race_job_edit_form_clean(row, admin_token=admin_token, default_job_race_date_text=default_job_race_date_text)}
      {race_job_settle_form_clean(row, admin_token=admin_token)}
    </section>
    """


def build_admin_filter_panel(admin_token="", show_settled=False):
    toggle_value = "0" if show_settled else "1"
    toggle_label = "隐藏已结算任务" if show_settled else "显示已结算任务"
    return f"""
    <section class="panel panel--tight">
      <div class="section-title">
        <div>
          <div class="eyebrow">Filter</div>
          <h2>任务筛选</h2>
        </div>
        <span class="section-chip">view</span>
      </div>
      <div class="copy-row">
        <span class="hero-pill">可切换是否展示已结算任务</span>
        <span class="hero-pill">结算后仍可查看 Step 3 结果</span>
      </div>
      <div class="copy-row">
        <a href="/console?token={html.escape(admin_token)}&show_settled={toggle_value}" class="secondary-button">{toggle_label}</a>
        <form method="post" action="/console/tasks/topup_today_all_llm" class="stack-form">
          <input type="hidden" name="token" value="{html.escape(admin_token)}">
          <button type="submit" class="secondary-button">追加今日 LLM 预算</button>
        </form>
      </div>
    </section>
    """


def build_admin_workspace_html_clean(
    *,
    message_text="",
    error_text="",
    admin_token="",
    authorized=True,
    show_settled=False,
    load_race_jobs,
    admin_job_card_html_clean,
    default_job_race_date_text,
):
    if not authorized:
        return f"""
        <section class="content-cluster" id="admin-zone">
          <section class="panel panel-error">
            <div class="section-title">
              <div>
                <div class="eyebrow">Admin</div>
                <h2>任务后台</h2>
              </div>
              <span class="section-chip">locked</span>
            </div>
            <p class="helper-text">{html.escape(error_text or "管理员口令无效。")}</p>
          </section>
        </section>
        """

    jobs = load_race_jobs()
    summary = {"total": len(jobs), "scheduled": 0, "processing": 0, "ready": 0, "settled": 0}
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        if status == "scheduled":
            summary["scheduled"] += 1
        elif status in ("queued_process", "processing", "queued_settle", "settling", "waiting_v5", "queued_policy", "processing_policy"):
            summary["processing"] += 1
        elif status == "ready":
            summary["ready"] += 1
        elif status == "settled":
            summary["settled"] += 1

    visible_jobs = [job for job in jobs if show_settled or str(job.get("status", "") or "").strip().lower() != "settled"]
    cards_html = "".join(admin_job_card_html_clean(job, admin_token=admin_token) for job in visible_jobs)
    if not cards_html:
        cards_html = """
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Tasks</div>
              <h2>暂无任务</h2>
            </div>
            <span class="section-chip">empty</span>
          </div>
          <p class="helper-text">先上传一场比赛的 `kachiuma.csv` 和 `shutuba.csv`，再加入处理队列。</p>
        </section>
        """

    message_panel = ""
    if message_text:
        message_panel = f"""
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Result</div>
              <h2>执行结果</h2>
            </div>
            <span class="section-chip">ok</span>
          </div>
          <p class="helper-text">{html.escape(message_text)}</p>
        </section>
        """
    error_panel = ""
    if error_text:
        error_panel = f"""
        <section class="panel panel-error">
          <div class="section-title">
            <div>
              <div class="eyebrow">Error</div>
              <h2>错误信息</h2>
            </div>
            <span class="section-chip">error</span>
          </div>
          <pre>{html.escape(error_text)}</pre>
        </section>
        """

    default_dt = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%dT15:00")
    default_date = default_job_race_date_text()
    return f"""
    <section class="content-cluster" id="admin-zone">
      <div class="cluster-head">
        <div>
          <div class="eyebrow">Admin Workspace</div>
          <h2>任务后台</h2>
        </div>
        <p>这里可以手动上传输入文件、创建任务、推进赔率与预测流程，并在比赛结束后录入赛果结算。</p>
      </div>
      <div class="cluster-grid cluster-grid--stack">
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Overview</div>
              <h2>任务概览</h2>
            </div>
            <span class="section-chip">manual</span>
          </div>
          <div class="copy-row">
            <span class="hero-pill">总任务 {summary['total']}</span>
            <span class="hero-pill">已计划 {summary['scheduled']}</span>
            <span class="hero-pill">处理中 {summary['processing']}</span>
            <span class="hero-pill">待结算 {summary['ready']}</span>
            <span class="hero-pill">已结算 {summary['settled']}</span>
          </div>
          <div class="copy-row">
            <form method="post" action="/console/tasks/scan_due" class="stack-form">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit" class="secondary-button">扫描到点任务</button>
            </form>
            <form method="post" action="/console/tasks/run_due_now" class="stack-form">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit" class="secondary-button">立即执行到点任务</button>
            </form>
            <a href="/llm_today" class="secondary-button">打开公开页</a>
          </div>
          <p class="helper-text">上传后的任务会先进入计划状态，到达处理时间后再进入预测流程。</p>
        </section>
        {message_panel}
        {error_panel}
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Step 1</div>
              <h2>上传输入文件</h2>
            </div>
            <span class="section-chip">upload</span>
          </div>
          <form class="stack-form" method="post" action="/console/tasks/create" enctype="multipart/form-data">
            <input type="hidden" name="token" value="{html.escape(admin_token)}">
            <div class="field-grid">
              <div>
                <label>范围</label>
                <select name="scope_key">
                  <option value="central_dirt">中央 Dirt</option>
                  <option value="central_turf">中央 Turf</option>
                  <option value="local">地方</option>
                </select>
              </div>
              <div>
                <label>Race ID</label>
                <input type="text" name="race_id" placeholder="202606010109">
              </div>
              <div>
                <label>场地</label>
                <input type="text" name="location" placeholder="中山">
              </div>
              <div>
                <label>比赛日期</label>
                <input type="date" name="race_date" value="{html.escape(default_date)}">
              </div>
              <div>
                <label>开赛时间</label>
                <input type="datetime-local" name="scheduled_off_time" value="{html.escape(default_dt)}">
              </div>
              <div>
                <label>提前分钟</label>
                <input type="number" name="lead_minutes" min="0" value="30">
              </div>
              <div>
                <label>距离</label>
                <input type="number" name="target_distance" min="100" step="100" value="1600" placeholder="1600">
              </div>
              <div>
                <label>马场状态</label>
                <select name="target_track_condition">
                  <option value="良">良</option>
                  <option value="稍重">稍重</option>
                  <option value="重">重</option>
                  <option value="不良">不良</option>
                </select>
              </div>
              <div>
                <label>kachiuma.csv</label>
                <input type="file" name="kachiuma_file" accept=".csv">
              </div>
              <div>
                <label>shutuba.csv</label>
                <input type="file" name="shutuba_file" accept=".csv">
              </div>
            </div>
            <div>
              <label>备注</label>
              <textarea name="notes" placeholder="可以记录这场比赛的说明或处理备注。"></textarea>
            </div>
            <button type="submit">创建任务</button>
          </form>
        </section>
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Step 2 / Step 3</div>
              <h2>预测与结算任务</h2>
            </div>
            <span class="section-chip">jobs</span>
          </div>
          <div class="cluster-grid cluster-grid--stack">
            {cards_html}
          </div>
        </section>
      </div>
    </section>
    """


def build_import_archive_panel(admin_token=""):
    return f"""
    <section class="content-cluster" id="import-zone">
      <div class="cluster-head">
        <div>
          <div class="eyebrow">Import</div>
          <h2>导入历史数据</h2>
        </div>
        <p>支持上传包含 `pipeline/data/...` 或 `data/...` 结构的 ZIP 包，直接导入到 Render 磁盘。</p>
      </div>
      <div class="cluster-grid cluster-grid--stack">
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Archive</div>
              <h2>导入历史数据 ZIP</h2>
            </div>
            <span class="section-chip">disk</span>
          </div>
          <form class="stack-form" method="post" action="/console/tasks/import_archive" enctype="multipart/form-data">
            <input type="hidden" name="token" value="{html.escape(admin_token)}">
            <div>
              <label>历史数据 ZIP</label>
              <input type="file" name="archive_file" accept=".zip,application/zip">
            </div>
            <label class="radio-option">
              <input type="checkbox" name="overwrite" value="1">
              <span class="radio-text">覆盖同名文件</span>
            </label>
            <p class="helper-text">允许导入 `central_dirt`、`central_turf`、`local`、`_shared` 这些目录的数据包。</p>
            <button type="submit">导入到 Render 磁盘</button>
          </form>
        </section>
      </div>
    </section>
    """


def build_race_jobs_page(*, message_text="", error_text="", admin_token="", authorized=True, render_console_page):
    return render_console_page(
        message_text=message_text,
        error_text=error_text,
        admin_token=admin_token,
    )

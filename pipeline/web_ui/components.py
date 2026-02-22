import html


def build_table_html(rows, columns, title):
    if not rows or not columns:
        return ""
    head_cells = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            cells.append(f"<td>{html.escape(str(val))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
        <section class="panel">
            <h2>{html.escape(title)}</h2>
            <div class="table-wrap">
                <table class="data-table">
                    <thead><tr>{head_cells}</tr></thead>
                    <tbody>
                        {''.join(body_rows)}
                    </tbody>
                </table>
            </div>
        </section>
        """


def build_metric_table(rows, title):
    if not rows:
        return ""
    sample = rows[0]
    if "metric" in sample and "value" in sample:
        return build_table_html(rows, ["metric", "value"], title)
    return build_table_html(rows, ["metric", "value"], title)


def build_daily_profit_chart_html(rows, title):
    if not rows:
        return ""
    data = list(reversed(rows[:30]))
    if len(data) < 2:
        return ""
    profits = [int(float(r.get("profit_yen", 0) or 0)) for r in data]
    max_abs = max(abs(v) for v in profits) if profits else 1
    if max_abs <= 0:
        max_abs = 1

    w = 860
    h = 250
    pad_l = 44
    pad_r = 16
    pad_t = 20
    pad_b = 34
    inner_w = w - pad_l - pad_r
    inner_h = h - pad_t - pad_b
    zero_y = pad_t + inner_h * 0.5
    bar_w = max(4, int(inner_w / max(1, len(data)) * 0.58))

    def x_at(i):
        if len(data) == 1:
            return pad_l + inner_w / 2
        return pad_l + (inner_w * i / (len(data) - 1))

    def y_at(v):
        return zero_y - (float(v) / max_abs) * (inner_h * 0.46)

    polyline_pts = []
    bars = []
    labels = []
    for i, row in enumerate(data):
        x = x_at(i)
        v = int(float(row.get("profit_yen", 0) or 0))
        y = y_at(v)
        polyline_pts.append(f"{x:.1f},{y:.1f}")
        top = min(zero_y, y)
        height = max(1.0, abs(zero_y - y))
        color = "#2e7d5b" if v >= 0 else "#c85f45"
        bars.append(
            f'<rect x="{x - bar_w/2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{height:.1f}" '
            f'fill="{color}" opacity="0.34"></rect>'
        )
        if i == 0 or i == len(data) - 1 or i % 5 == 0:
            date_str = html.escape(str(row.get("date", ""))[5:])
            labels.append(
                f'<text x="{x:.1f}" y="{h - 10}" text-anchor="middle" font-size="10" fill="#6c665f">{date_str}</text>'
            )

    polyline = " ".join(polyline_pts)
    bars_html = "".join(bars)
    labels_html = "".join(labels)
    title_html = html.escape(title)
    y_top_val = f"{max_abs}"
    y_bottom_val = f"-{max_abs}"

    return f"""
        <section class="panel">
            <h2>{title_html}</h2>
            <div class="table-wrap">
                <svg viewBox="0 0 {w} {h}" width="100%" height="auto" role="img" aria-label="{title_html}">
                    <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h-pad_b}" stroke="#d9ccbc" stroke-width="1"></line>
                    <line x1="{pad_l}" y1="{zero_y:.1f}" x2="{w-pad_r}" y2="{zero_y:.1f}" stroke="#d9ccbc" stroke-width="1"></line>
                    <line x1="{pad_l}" y1="{h-pad_b}" x2="{w-pad_r}" y2="{h-pad_b}" stroke="#d9ccbc" stroke-width="1"></line>
                    <text x="{pad_l-8}" y="{pad_t+4}" text-anchor="end" font-size="10" fill="#6c665f">{y_top_val}</text>
                    <text x="{pad_l-8}" y="{h-pad_b+4}" text-anchor="end" font-size="10" fill="#6c665f">{y_bottom_val}</text>
                    {bars_html}
                    <polyline points="{polyline}" fill="none" stroke="#1f4b39" stroke-width="2.2"></polyline>
                    {labels_html}
                </svg>
            </div>
        </section>
        """


def detect_gate_status(rows):
    for row in rows:
        status = str(row.get("gate_status", "")).strip().lower()
        if status:
            reason = str(row.get("gate_reason", "")).strip()
            return status, reason
    return "", ""


def build_gate_notice_html(status, reason):
    reason_html = f"<div>{html.escape(reason)}</div>" if reason else ""
    if status == "soft_fail":
        return (
            '<div class="alert"><strong>Pass Gate Soft</strong>'
            "High risk: soft gate failed; still showing tickets."
            f"{reason_html}</div>"
        )
    if status == "hard_fail":
        return (
            '<div class="alert"><strong>Pass Gate Hard</strong>'
            "Hard gate blocked tickets."
            f"{reason_html}</div>"
        )
    return ""


def build_gate_notice_text(status, reason):
    reason_text = f" | {reason}" if reason else ""
    if status == "soft_fail":
        return f"[WARN] SOFT_GATE: high risk (showing tickets){reason_text}"
    if status == "hard_fail":
        return f"[WARN] HARD_GATE: blocked{reason_text}"
    return ""


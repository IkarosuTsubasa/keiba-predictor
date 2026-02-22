import csv
import html
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from surface_scope import get_data_dir, migrate_legacy_data, normalize_scope_key
from web_data import odds_service, run_resolver, run_store, summary_service, view_data
from web_ui.components import (
    build_daily_profit_chart_html as ui_build_daily_profit_chart_html,
    build_gate_notice_html as ui_build_gate_notice_html,
    build_gate_notice_text as ui_build_gate_notice_text,
    build_metric_table as ui_build_metric_table,
    build_table_html as ui_build_table_html,
    detect_gate_status as ui_detect_gate_status,
)
from web_ui.template import page_template as ui_page_template
from web_ui.stats_block import (
    build_run_summary_block as ui_build_run_summary_block,
    build_stats_block as ui_build_stats_block,
)


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
RUN_PIPELINE = BASE_DIR / "run_pipeline.py"
BET_PLAN_UPDATE = BASE_DIR / "bet_plan_update.py"
ODDS_EXTRACT = ROOT_DIR / "odds_extract.py"
RECORD_PIPELINE = BASE_DIR / "record_pipeline.py"
RECORD_PREDICTOR = BASE_DIR / "record_predictor_result.py"
OPTIMIZE_PARAMS = BASE_DIR / "optimize_params.py"
OPTIMIZE_PREDICTOR = BASE_DIR / "optimize_predictor_params.py"
OFFLINE_EVAL = BASE_DIR / "offline_eval.py"
INIT_UPDATE = BASE_DIR / "init_update.py"
DEFAULT_RUN_LIMIT = 200
MAX_RUN_LIMIT = 500
MIN_RACE_YEAR = 2026

app = FastAPI()


def load_runs(scope_key):
    migrate_legacy_data(BASE_DIR, scope_key)
    runs_path = get_data_dir(BASE_DIR, scope_key) / "runs.csv"
    if not runs_path.exists():
        return []
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            with open(runs_path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
                rows, _ = normalize_csv_rows(rows, fieldnames)
                return rows
        except UnicodeDecodeError:
            continue
    return []


def normalize_csv_fieldnames(fieldnames):
    return run_store.normalize_csv_fieldnames(fieldnames)


def normalize_csv_rows(rows, fieldnames):
    return run_store.normalize_csv_rows(rows, fieldnames)


def load_runs_with_header(scope_key):
    return run_store.load_runs_with_header(migrate_legacy_data, get_data_dir, BASE_DIR, scope_key)


def get_recent_runs(scope_key, limit=DEFAULT_RUN_LIMIT, query=""):
    runs = load_runs(scope_key)
    if not runs:
        return []
    query = (query or "").strip().lower()
    if query:
        runs = [
            row
            for row in runs
            if query in str(row.get("run_id", "")).lower()
            or query in str(row.get("race_id", "")).lower()
            or query in str(row.get("timestamp", "")).lower()
        ]
    if limit is not None:
        runs = runs[-limit:]
    runs.reverse()
    out = []
    for row in runs:
        run_id = row.get("run_id", "")
        race_id = row.get("race_id", "")
        timestamp = row.get("timestamp", "")
        label_parts = [run_id]
        if race_id:
            label_parts.append(f"race_id={race_id}")
        if timestamp:
            label_parts.append(timestamp)
        label = " | ".join([part for part in label_parts if part])
        out.append({"run_id": run_id, "label": label})
    return out


def build_run_options(scope_key, selected_run_id=""):
    options = []
    scope_key = normalize_scope_key(scope_key)
    selected_run_id = str(selected_run_id or "")
    if not scope_key:
        return '<option value="" disabled>Select data scope</option>'
    for item in get_recent_runs(scope_key, limit=DEFAULT_RUN_LIMIT):
        value = html.escape(item["run_id"])
        label = html.escape(item["label"])
        selected_attr = ' selected' if selected_run_id and item["run_id"] == selected_run_id else ""
        options.append(f'<option value="{value}"{selected_attr}>{label}</option>')
    if not options:
        options.append('<option value="" disabled>No runs</option>')
    return "\n".join(options)

def resolve_run(run_id, scope_key):
    return run_resolver.resolve_run(load_runs, run_id, scope_key)


def resolve_latest_run_by_race_id(race_id, scope_key):
    return run_resolver.resolve_latest_run_by_race_id(load_runs, normalize_race_id, race_id, scope_key)


def infer_run_id_from_path(path):
    return run_resolver.infer_run_id_from_path(path)


def infer_run_id_from_row(run_row):
    return run_resolver.infer_run_id_from_row(run_row)


def update_run_row_fields(scope_key, run_row, updates):
    return run_store.update_run_row_fields(
        get_data_dir,
        BASE_DIR,
        load_runs_with_header,
        normalize_race_id,
        scope_key,
        run_row,
        updates,
    )


def resolve_plan_path(scope_key, run_id, run_row):
    return run_resolver.resolve_plan_path(get_data_dir, BASE_DIR, scope_key, run_id, run_row)


def update_run_plan_path(scope_key, run_id, plan_path):
    return run_store.update_run_plan_path(
        get_data_dir,
        BASE_DIR,
        load_runs_with_header,
        scope_key,
        run_id,
        plan_path,
    )


def refresh_odds_for_run(
    run_row,
    scope_key,
    odds_path,
    wide_odds_path=None,
    fuku_odds_path=None,
    quinella_odds_path=None,
    trifecta_odds_path=None,
):
    race_url = str(run_row.get("race_url") or "").strip()
    race_id = normalize_race_id(run_row.get("race_id", ""))
    if not race_url and race_id:
        if scope_key in ("central_turf", "central_dirt"):
            base = "https://race.netkeiba.com/race/shutuba.htmlheavyrace_id="
        else:
            base = "https://nar.netkeiba.com/race/shutuba.htmlheavyrace_id="
        race_url = f"{base}{race_id}"
    if not race_url:
        return False, "Race URL missing for odds update.", []
    if not ODDS_EXTRACT.exists():
        return False, "odds_extract.py not found.", []
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PIPELINE_HEADLESS", "0")
    result = subprocess.run(
        [sys.executable, str(ODDS_EXTRACT)],
        input=f"{race_url}\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=str(ROOT_DIR),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, f"odds_extract failed: {detail}", []
    if "Saved: odds.csv" not in (result.stdout or ""):
        return False, "odds_extract produced no new odds.", []
    tmp_path = ROOT_DIR / "odds.csv"
    if not tmp_path.exists():
        return False, "odds.csv not generated.", []
    warnings = []
    try:
        Path(odds_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path, odds_path)
    except Exception as exc:
        return False, f"Failed to update odds file: {exc}", []
    wide_tmp = ROOT_DIR / "wide_odds.csv"
    if wide_odds_path and wide_tmp.exists():
        try:
            Path(wide_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wide_tmp, wide_odds_path)
        except Exception as exc:
            return False, f"Failed to update wide odds file: {exc}", []
    elif wide_odds_path:
        warnings.append("wide_odds.csv not generated.")
    fuku_tmp = ROOT_DIR / "fuku_odds.csv"
    if fuku_odds_path and fuku_tmp.exists():
        try:
            Path(fuku_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fuku_tmp, fuku_odds_path)
        except Exception as exc:
            return False, f"Failed to update place odds file: {exc}", []
    elif fuku_odds_path:
        warnings.append("fuku_odds.csv not generated.")
    quinella_tmp = ROOT_DIR / "quinella_odds.csv"
    if quinella_odds_path and quinella_tmp.exists():
        try:
            Path(quinella_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(quinella_tmp, quinella_odds_path)
        except Exception as exc:
            return False, f"Failed to update quinella odds file: {exc}", []
    elif quinella_odds_path:
        warnings.append("quinella_odds.csv not generated.")
    trifecta_tmp = ROOT_DIR / "trifecta_odds.csv"
    if trifecta_odds_path and trifecta_tmp.exists():
        try:
            Path(trifecta_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trifecta_tmp, trifecta_odds_path)
        except Exception as exc:
            return False, f"Failed to update trifecta odds file: {exc}", []
    elif trifecta_odds_path:
        warnings.append("trifecta_odds.csv not generated.")
    return True, "", warnings


def run_script(script_path, inputs=None, args=None, extra_blanks=0, extra_env=None):
    payload = ""
    if inputs is not None:
        payload = "\n".join([str(v) for v in inputs] + [""] * int(extra_blanks)) + "\n"
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)
    result = subprocess.run(
        cmd,
        input=payload,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=str(BASE_DIR),
        env=env,
    )
    output = result.stdout or ""
    if result.stderr:
        output = f"{output}\n[stderr]\n{result.stderr}"
    return result.returncode, output.strip()


def extract_section(output_text, start_label, end_label=None):
    if not output_text:
        return ""
    start_idx = output_text.find(start_label)
    if start_idx < 0:
        return ""
    start_idx += len(start_label)
    section = output_text[start_idx:]
    if section.startswith("\n"):
        section = section[1:]
    if end_label:
        end_idx = section.find(end_label)
        if end_idx >= 0:
            section = section[:end_idx]
    return section.strip()


def extract_top5(output_text):
    return extract_section(output_text, "Top5 predictions:", "Saved: predictions.csv")


def extract_bet_plan(output_text):
    section = extract_section(output_text, "Bet plan:")
    if not section:
        return ""
    end_idx = section.find("Saved:")
    if end_idx >= 0:
        section = section[:end_idx]
    return section.strip()

def parse_run_id(output_text):
    if not output_text:
        return ""
    match = re.search(r"Logged run: (\d{8}_\d{6})", output_text)
    return match.group(1) if match else ""

def normalize_race_id(value):
    raw = str(value or "").strip()
    if not raw:
        return ""
    match = re.search(r"race_id=(\d+)", raw)
    if match:
        return match.group(1)
    return re.sub(r"\D", "", raw)


def is_run_id(value):
    return bool(re.fullmatch(r"\d{8}_\d{6}", str(value or "").strip()))


def find_run_in_scope(scope_key, id_text):
    return run_resolver.find_run_in_scope(load_runs, normalize_race_id, scope_key, id_text)


def infer_scope_and_run(id_text):
    return run_resolver.infer_scope_and_run(load_runs, normalize_race_id, id_text)


def load_csv_rows(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_int_or_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def format_path_mtime(path, label):
    if not path:
        return f"{label}: (missing path)"
    path = Path(path)
    if not path.exists():
        return f"{label}: {path} (missing)"
    ts = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return f"{label}: {path} (mtime={ts})"


def parse_odds_value(value):
    return odds_service.parse_odds_value(value)


def load_odds_snapshot(path):
    return odds_service.load_odds_snapshot(normalize_name, path)


def odds_changed(prev_item, curr_item):
    return odds_service.odds_changed(prev_item, curr_item)


def odds_item_label(item):
    return odds_service.odds_item_label(item)


def odds_sort_key(item):
    return odds_service.odds_sort_key(item)


def format_odds_diff(prev_snapshot, curr_snapshot, limit=50):
    return odds_service.format_odds_diff(prev_snapshot, curr_snapshot, limit=limit)


def build_table_html(rows, columns, title):
    return ui_build_table_html(rows, columns, title)


def build_metric_table(rows, title):
    return ui_build_metric_table(rows, title)


def load_top5_table(scope_key, run_id, run_row=None):
    return view_data.load_top5_table(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        to_float,
        scope_key,
        run_id,
        run_row,
    )


def load_prediction_summary(scope_key, run_id, run_row=None):
    return view_data.load_prediction_summary(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        load_mc_uncertainty_summary,
        scope_key,
        run_id,
        run_row,
    )


def load_mc_uncertainty_summary(scope_key, run_id, run_row=None):
    return view_data.load_mc_uncertainty_summary(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        to_float,
        scope_key,
        run_id,
        run_row,
    )


def load_profit_summary(scope_key):
    return summary_service.load_profit_summary(get_data_dir, BASE_DIR, load_csv_rows, scope_key)


def build_run_race_map(scope_key):
    return summary_service.build_run_race_map(load_runs, infer_run_id_from_row, scope_key)


def load_daily_profit_summary(scope_key, days=30):
    return summary_service.load_daily_profit_summary(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        build_run_race_map,
        MIN_RACE_YEAR,
        scope_key,
        days=days,
    )


def load_daily_profit_summary_all_scopes(days=30):
    return summary_service.load_daily_profit_summary_all_scopes(load_daily_profit_summary, to_int_or_none, days=days)


def build_daily_profit_chart_html(rows, title):
    return ui_build_daily_profit_chart_html(rows, title)


def normalize_name(value):
    return summary_service.normalize_name(value)


def pick_score_key(rows):
    return view_data.pick_score_key(rows)


def load_top5_names(path):
    return summary_service.load_top5_names(load_csv_rows, to_float, path)


def load_odds_name_to_no(path):
    return summary_service.load_odds_name_to_no(load_csv_rows, path)


def load_wide_odds_map(path):
    return summary_service.load_wide_odds_map(load_csv_rows, path)


def resolve_pred_path(scope_key, run_id, run_row):
    return run_resolver.resolve_pred_path(get_data_dir, BASE_DIR, scope_key, run_id, run_row)


def resolve_odds_path(scope_key, run_id, run_row):
    return run_resolver.resolve_odds_path(get_data_dir, BASE_DIR, scope_key, run_id, run_row)


def resolve_wide_odds_path(scope_key, run_id, run_row):
    return run_resolver.resolve_wide_odds_path(get_data_dir, BASE_DIR, scope_key, run_id, run_row)


def resolve_run_asset_path(scope_key, run_id, run_row, field_name, prefix):
    return run_resolver.resolve_run_asset_path(
        get_data_dir,
        BASE_DIR,
        scope_key,
        run_id,
        run_row,
        field_name,
        prefix,
    )


def load_race_results(scope_key):
    return summary_service.load_race_results(get_data_dir, BASE_DIR, load_csv_rows, scope_key)


def compute_wide_box_profit(scope_key, run_row, race_row, budget_yen=1000):
    return summary_service.compute_wide_box_profit(
        scope_key,
        run_row,
        race_row,
        budget_yen,
        resolve_pred_path,
        resolve_odds_path,
        resolve_wide_odds_path,
        load_top5_names,
        load_odds_name_to_no,
        load_wide_odds_map,
    )


def load_wide_box_daily_profit_summary(scope_key, days=30, budget_yen=1000):
    return summary_service.load_wide_box_daily_profit_summary(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        load_runs,
        build_run_race_map,
        load_race_results,
        compute_wide_box_profit,
        MIN_RACE_YEAR,
        scope_key,
        days=days,
        budget_yen=budget_yen,
    )


def load_bet_type_summary(scope_key):
    return summary_service.load_bet_type_summary(get_data_dir, BASE_DIR, load_csv_rows, scope_key)


def load_bet_type_profit_summary(scope_key):
    return summary_service.load_bet_type_profit_summary(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        build_run_race_map,
        MIN_RACE_YEAR,
        scope_key,
    )


def compute_top5_hit_count(scope_key, row):
    return summary_service.compute_top5_hit_count(
        get_data_dir,
        BASE_DIR,
        to_int_or_none,
        load_top5_names,
        scope_key,
        row,
    )


def load_predictor_summary(scope_key):
    return summary_service.load_predictor_summary(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        compute_top5_hit_count,
        scope_key,
    )


def load_run_result_summary(scope_key, run_id):
    return summary_service.load_run_result_summary(get_data_dir, BASE_DIR, load_csv_rows, scope_key, run_id)


def load_run_bet_type_summary(scope_key, run_id):
    return summary_service.load_run_bet_type_summary(get_data_dir, BASE_DIR, load_csv_rows, scope_key, run_id)


def load_run_bet_ticket_summary(scope_key, run_id):
    return summary_service.load_run_bet_ticket_summary(get_data_dir, BASE_DIR, load_csv_rows, scope_key, run_id)


def load_run_predictor_summary(scope_key, run_id):
    return summary_service.load_run_predictor_summary(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        to_float,
        compute_top5_hit_count,
        scope_key,
        run_id,
    )


def load_bet_plan_table(scope_key, run_id, run_row=None):
    return view_data.load_bet_plan_table(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        scope_key,
        run_id,
        run_row,
    )


def detect_gate_status(rows):
    return ui_detect_gate_status(rows)


def build_gate_notice_html(status, reason):
    return ui_build_gate_notice_html(status, reason)


def build_gate_notice_text(status, reason):
    return ui_build_gate_notice_text(status, reason)


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
    return ui_page_template(
        output_text=output_text,
        error_text=error_text,
        run_options=run_options,
        view_run_options=view_run_options,
        view_selected_run_id=view_selected_run_id,
        top5_text=top5_text,
        bet_plan_text=bet_plan_text,
        top5_table_html=top5_table_html,
        bet_plan_table_html=bet_plan_table_html,
        summary_table_html=summary_table_html,
        run_summary_block=run_summary_block,
        stats_block=stats_block,
        default_scope=default_scope,
    )


def render_page(
    scope_key="central_dirt",
    output_text="",
    error_text="",
    top5_text="",
    bet_plan_text="",
    summary_run_id="",
    selected_run_id="",
):
    scope_norm = normalize_scope_key(scope_key)
    default_scope = scope_norm or "central_dirt"
    run_options = build_run_options(scope_norm or scope_key)
    selected_run_id = str(selected_run_id or "").strip()
    stats_block = ui_build_stats_block(
        scope_norm,
        load_profit_summary=load_profit_summary,
        load_daily_profit_summary=load_daily_profit_summary,
        load_daily_profit_summary_all_scopes=load_daily_profit_summary_all_scopes,
        load_wide_box_daily_profit_summary=load_wide_box_daily_profit_summary,
        load_bet_type_profit_summary=load_bet_type_profit_summary,
        load_bet_type_summary=load_bet_type_summary,
        load_predictor_summary=load_predictor_summary,
        build_metric_table=build_metric_table,
        build_table_html=build_table_html,
        build_daily_profit_chart_html=build_daily_profit_chart_html,
    )
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
    view_run_options = build_run_options(scope_norm or scope_key, view_selected_run_id)
    top5_table_html = ""
    bet_plan_table_html = ""
    summary_table_html = ""
    bet_rows = []
    if run_id:
        top_rows, top_cols = load_top5_table(scope_norm or scope_key, run_id, run_row)
        top5_table_html = build_table_html(top_rows, top_cols, "Top5 Predictions")
        summary_rows = load_prediction_summary(scope_norm or scope_key, run_id, run_row)
        if summary_rows:
            summary_table_html = build_table_html(summary_rows, ["metric", "value"], "Model Status")
        bet_rows, bet_cols = load_bet_plan_table(scope_norm or scope_key, run_id, run_row)
        bet_plan_table_html = build_table_html(bet_rows, bet_cols, "Bet Plan")
    gate_status, gate_reason = detect_gate_status(bet_rows)
    gate_notice_html = build_gate_notice_html(gate_status, gate_reason)
    gate_notice_text = build_gate_notice_text(gate_status, gate_reason)
    if gate_notice_html and bet_plan_table_html:
        bet_plan_table_html = f"{gate_notice_html}{bet_plan_table_html}"
    if gate_notice_text and bet_plan_text:
        bet_plan_text = f"{gate_notice_text}\n{bet_plan_text}"
    run_summary_block = ""
    summary_id = summary_run_id or (run_row.get("run_id") if run_row else "")
    run_summary_block = ui_build_run_summary_block(
        scope_norm,
        summary_id,
        load_run_result_summary=load_run_result_summary,
        load_run_bet_ticket_summary=load_run_bet_ticket_summary,
        load_run_predictor_summary=load_run_predictor_summary,
        load_run_bet_type_summary=load_run_bet_type_summary,
        build_metric_table=build_metric_table,
        build_table_html=build_table_html,
    )
    return page_template(
        output_text=output_text,
        error_text=error_text,
        run_options=run_options,
        view_run_options=view_run_options,
        view_selected_run_id=view_selected_run_id,
        top5_text=top5_text if not top5_table_html else "",
        bet_plan_text=bet_plan_text if not bet_plan_table_html else "",
        top5_table_html=top5_table_html,
        bet_plan_table_html=bet_plan_table_html,
        summary_table_html=summary_table_html,
        run_summary_block=run_summary_block,
        stats_block=stats_block,
        default_scope=default_scope,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return render_page("")


@app.post("/view_run", response_class=HTMLResponse)
def view_run(
    run_id: str = Form(""),
    scope_key: str = Form(""),
):
    run_id = run_id.strip()
    scope_key = normalize_scope_key(scope_key)
    run_row = None
    if not scope_key:
        scope_key, run_row = infer_scope_and_run(run_id)
    if not scope_key:
        return render_page("", error_text="Enter Run ID or Race ID to view history.")
    if run_row is None:
        run_row = resolve_run(run_id, scope_key)
    if run_row is None:
        race_id = normalize_race_id(run_id)
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_key)
    if run_row is None:
        return render_page(
            scope_key,
            error_text="Run ID / Race ID not found.",
            selected_run_id=run_id,
        )
    resolved_run_id = run_row.get("run_id", run_id)
    return render_page(
        scope_key,
        selected_run_id=resolved_run_id,
        summary_run_id=resolved_run_id,
    )


@app.get("/api/runs")
def api_runs(scope_key: str = "central_dirt", limit: int = DEFAULT_RUN_LIMIT, q: str = ""):
    scope_key = normalize_scope_key(scope_key) or "central_dirt"
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = DEFAULT_RUN_LIMIT
    limit = max(1, min(MAX_RUN_LIMIT, limit))
    return get_recent_runs(scope_key, limit=limit, query=q)


@app.post("/run_pipeline", response_class=HTMLResponse)
def run_pipeline(
    race_id: str = Form(""),
    race_url: str = Form(""),
    history_url: str = Form(""),
    scope_key: str = Form(""),
    surface: str = Form(""),
    distance: str = Form(""),
    track_cond: str = Form(""),
    budget: str = Form(""),
    style: str = Form(""),
):
    scope_key = normalize_scope_key(scope_key)
    if not scope_key:
        return render_page("", error_text="Please select a data scope.")
    race_id = normalize_race_id(race_id or race_url)
    history_url = history_url.strip()
    if not race_id or not history_url:
        return render_page(
            scope_key,
            error_text="Race ID and History URL are required.",
        )
    if scope_key == "local":
        race_url = f"https://nar.netkeiba.com/race/shutuba.htmlheavyrace_id={race_id}"
    else:
        race_url = f"https://race.netkeiba.com/race/shutuba.htmlheavyrace_id={race_id}"
    if not surface.strip():
        if scope_key == "central_turf":
            surface = "1"
        elif scope_key == "central_dirt":
            surface = "2"
        else:
            surface = "1"
    track_cond_norm = str(track_cond or "").strip().lower()
    track_cond_map = {
        "good": "good",
        "slightly_heavy": "slightly_heavy",
        "heavy": "heavy",
        "bad": "bad",
    }
    if track_cond_norm in track_cond_map:
        track_cond = track_cond_map[track_cond_norm]
    inputs = [
        race_url,
        history_url,
        surface,
        distance,
        track_cond,
        budget,
        style,
    ]
    extra_env = {"SCOPE_KEY": scope_key}
    code, output = run_script(
        RUN_PIPELINE,
        inputs=inputs,
        extra_env=extra_env,
    )
    label = f"Exit code: {code}"
    top5_text = extract_top5(output)
    bet_plan_text = extract_bet_plan(output)
    return render_page(
        scope_key,
        output_text=f"{label}\n{output}",
        top5_text=top5_text,
        bet_plan_text=bet_plan_text,
        summary_run_id=parse_run_id(output),
    )


@app.post("/update_bet_plan", response_class=HTMLResponse)
def update_bet_plan(
    race_id: str = Form(""),
    scope_key: str = Form(""),
    budget: str = Form(""),
    style: str = Form(""),
):
    raw_id = (race_id or "").strip()
    scope_key = normalize_scope_key(scope_key)
    run_row = None
    if not scope_key:
        scope_key, run_row = infer_scope_and_run(raw_id)
    if not scope_key:
        return render_page("", error_text="Enter Run ID or Race ID to update.")
    race_id = "" if is_run_id(raw_id) else normalize_race_id(raw_id)
    if run_row is None:
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_key)
        elif raw_id:
            run_row = resolve_run(raw_id, scope_key)
    if run_row is None:
        return render_page(scope_key, error_text="Run ID / Race ID not found.")
    if not race_id:
        race_id = normalize_race_id(run_row.get("race_id", ""))
    if not race_id:
        return render_page(scope_key, error_text="Race ID missing; cannot update.")
    run_id = str(run_row.get("run_id", "")).strip()
    if not run_id:
        run_id = infer_run_id_from_row(run_row)
        if run_id:
            update_run_row_fields(scope_key, run_row, {"run_id": run_id})
            run_row["run_id"] = run_id
    if not run_id:
        return render_page(scope_key, error_text="Missing run_id for this race.")
    pred_path = resolve_pred_path(scope_key, run_id, run_row)
    if not pred_path.exists():
        return render_page(scope_key, error_text=f"Predictions file not found: {pred_path}")
    odds_path = resolve_odds_path(scope_key, run_id, run_row)
    if not odds_path:
        return render_page(scope_key, error_text="Odds path not found for this run.")

    wide_path = resolve_wide_odds_path(scope_key, run_id, run_row)
    fuku_path = resolve_run_asset_path(scope_key, run_id, run_row, "fuku_odds_path", "fuku_odds")
    quinella_path = resolve_run_asset_path(scope_key, run_id, run_row, "quinella_odds_path", "quinella_odds")
    trifecta_path = resolve_run_asset_path(scope_key, run_id, run_row, "trifecta_odds_path", "trifecta_odds")
    prev_odds_snapshot = load_odds_snapshot(odds_path)
    warnings = []
    updated, msg, odds_warnings = refresh_odds_for_run(
        run_row,
        scope_key,
        odds_path,
        wide_odds_path=wide_path,
        fuku_odds_path=fuku_path,
        quinella_odds_path=quinella_path,
        trifecta_odds_path=trifecta_path,
    )
    warnings.extend(odds_warnings)
    if not updated:
        return render_page(scope_key, error_text=msg)

    if not odds_path.exists():
        return render_page(scope_key, error_text=f"Odds file not found: {odds_path}")
    if wide_path and not wide_path.exists():
        warnings.append(f"wide odds not found: {wide_path}")
    if fuku_path and not fuku_path.exists():
        warnings.append(f"fuku odds not found: {fuku_path}")
    if quinella_path and not quinella_path.exists():
        warnings.append(f"quinella odds not found: {quinella_path}")

    budget_val = to_int_or_none(budget)
    if budget_val is None:
        budget_val = to_int_or_none(run_row.get("budget_yen"))
    budget_input = str(budget_val) if budget_val and budget_val > 0 else ""
    style_input = style.strip().lower()
    if not style_input:
        style_input = str(run_row.get("style", "")).strip().lower()
    if style_input not in ("steady", "balanced", "aggressive"):
        style_input = ""

    extra_env = {
        "SCOPE_KEY": scope_key,
        "RACE_ID": race_id,
        "ODDS_PATH": str(odds_path),
        "PRED_PATH": str(pred_path),
    }
    if wide_path:
        extra_env["WIDE_ODDS_PATH"] = str(wide_path)
    if fuku_path:
        extra_env["FUKU_ODDS_PATH"] = str(fuku_path)
    if quinella_path:
        extra_env["QUINELLA_ODDS_PATH"] = str(quinella_path)

    code, output = run_script(
        BET_PLAN_UPDATE,
        inputs=[budget_input, style_input],
        extra_blanks=1,
        extra_env=extra_env,
    )

    plan_src = BASE_DIR / "bet_plan_update.csv"
    if code == 0 and plan_src.exists():
        plan_dest = resolve_plan_path(scope_key, run_id, run_row)
        plan_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_src, plan_dest)
        if not update_run_plan_path(scope_key, run_id, plan_dest):
            warnings.append("runs.csv not updated; plan_path may be stale.")
    else:
        if not plan_src.exists():
            warnings.append("bet_plan_update.csv not found after update.")

    curr_odds_snapshot = load_odds_snapshot(odds_path)
    diff_lines = format_odds_diff(prev_odds_snapshot, curr_odds_snapshot)
    label = f"Exit code: {code}"
    output_lines = [label]
    if warnings:
        output_lines.extend([f"[WARN] {item}" for item in warnings])
    output_lines.append(format_path_mtime(odds_path, "odds_path"))
    output_lines.append(format_path_mtime(wide_path, "wide_odds_path"))
    output_lines.append(format_path_mtime(fuku_path, "fuku_odds_path"))
    output_lines.append(format_path_mtime(quinella_path, "quinella_odds_path"))
    output_lines.append(format_path_mtime(trifecta_path, "trifecta_odds_path"))
    output_lines.extend(diff_lines)
    if output:
        output_lines.append(output)
    output_text = "\n".join(output_lines).strip()
    bet_plan_text = extract_bet_plan(output_text)
    return render_page(
        scope_key,
        output_text=output_text,
        bet_plan_text=bet_plan_text,
        summary_run_id=run_id,
        selected_run_id=run_id,
    )


@app.post("/record_pipeline", response_class=HTMLResponse)
def record_pipeline(
    run_id: str = Form(""),
    scope_key: str = Form(""),
    profit: str = Form(""),
    note: str = Form(""),
    top1: str = Form(""),
    top2: str = Form(""),
    top3: str = Form(""),
):
    if not top1 or not top2 or not top3:
        return render_page(
            scope_key,
            error_text="Actual 1st/2nd/3rd are required.",
        )
    run_id = run_id.strip()
    scope_key = normalize_scope_key(scope_key)
    run_row = None
    if not scope_key:
        scope_key, run_row = infer_scope_and_run(run_id)
    if not scope_key:
        return render_page("", error_text="Enter Run ID or Race ID to record results.")
    if run_row is None:
        run_row = resolve_run(run_id, scope_key)
    if run_row is None:
        race_id = normalize_race_id(run_id)
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_key)
    if run_row is None:
        return render_page(
            scope_key,
            error_text="Run ID / Race ID not found.",
        )
    odds_path = run_row.get("odds_path", "")
    resolved_run_id = run_row.get("run_id", run_id)
    if not odds_path:
        odds_path = str(get_data_dir(BASE_DIR, scope_key) / f"odds_{resolved_run_id}.csv")
    inputs = [resolved_run_id, profit, note, top1, top2, top3]
    code, output = run_script(
        RECORD_PIPELINE,
        inputs=inputs,
        extra_blanks=4,
        extra_env={"SCOPE_KEY": scope_key},
    )
    label = f"Exit code: {code}"
    return render_page(
        scope_key,
        output_text=f"{label}\n{output}",
        summary_run_id=resolved_run_id,
        selected_run_id=resolved_run_id,
    )

@app.post("/record_predictor", response_class=HTMLResponse)
def record_predictor(
    run_id: str = Form(""),
    top1: str = Form(""),
    top2: str = Form(""),
    top3: str = Form(""),
):
    if not top1 or not top2 or not top3:
        return page_template(
            error_text="Top1/Top2/Top3 are required.",
        )
    inputs = [run_id, top1, top2, top3]
    code, output = run_script(RECORD_PREDICTOR, inputs=inputs, extra_blanks=2)
    label = f"Exit code: {code}"
    return page_template(
        output_text=f"{label}\n{output}",
    )


@app.post("/optimize_params", response_class=HTMLResponse)
def optimize_params():
    code, output = run_script(OPTIMIZE_PARAMS, inputs=[""], extra_blanks=1)
    label = f"Exit code: {code}"
    return page_template(
        output_text=f"{label}\n{output}",
    )


@app.post("/optimize_predictor", response_class=HTMLResponse)
def optimize_predictor():
    code, output = run_script(OPTIMIZE_PREDICTOR, inputs=[""], extra_blanks=1)
    label = f"Exit code: {code}"
    return page_template(
        output_text=f"{label}\n{output}",
    )


@app.post("/offline_eval", response_class=HTMLResponse)
def offline_eval(window: str = Form("")):
    inputs = [window, ""]
    code, output = run_script(OFFLINE_EVAL, inputs=inputs, extra_blanks=1)
    label = f"Exit code: {code}"
    return page_template(
        output_text=f"{label}\n{output}",
    )


@app.post("/init_update", response_class=HTMLResponse)
def init_update():
    code, output = run_script(INIT_UPDATE, inputs=[""], extra_blanks=1)
    label = f"Exit code: {code}"
    return page_template(
        output_text=f"{label}\n{output}",
    )


@app.post("/init_update_reset", response_class=HTMLResponse)
def init_update_reset():
    code, output = run_script(INIT_UPDATE, inputs=[""], args=["--reset"], extra_blanks=1)
    label = f"Exit code: {code}"
    return page_template(
        output_text=f"{label}\n{output}",
    )

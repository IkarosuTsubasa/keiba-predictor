import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
import base64
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

from predictor_catalog import canonical_predictor_id, list_predictors, predictor_label, resolve_run_prediction_path, snapshot_prediction_path
from gemini_portfolio import (
    add_bankroll_topup,
    build_history_context,
    extract_ledger_date,
    ledger_path,
    load_exacta_odds_map,
    load_daily_profit_rows,
    load_name_to_no,
    load_pair_odds_map,
    load_place_odds_map,
    load_rows,
    load_run_tickets,
    load_triple_odds_map,
    load_win_odds_map,
    reserve_run_tickets,
    resolve_daily_bankroll_yen,
    summarize_bankroll,
)
from llm.policy_runtime import (
    DEFAULT_GEMINI_MODEL,
    call_policy,
    get_last_call_meta,
    normalize_policy_engine,
    resolve_policy_model,
)
from llm_state import reset_llm_state as reset_llm_state_files
from local_env import load_local_env
from race_job_store import (
    apply_job_action as apply_race_job_action,
    compute_initial_status as compute_race_job_initial_status,
    create_job as create_race_job,
    delete_job as delete_race_job,
    derive_job_display_state as derive_race_job_display_state,
    hydrate_job_step_states as hydrate_race_job_step_states,
    initialize_job_step_fields as initialize_race_job_step_fields,
    load_jobs as load_race_jobs,
    save_artifact as save_race_job_artifact,
    scan_due_jobs as scan_due_race_jobs,
    set_job_step_state as set_race_job_step_state,
    update_job as update_race_job,
)
from surface_scope import get_data_dir, migrate_legacy_data, normalize_scope_key
from v5_remote_tasks import find_latest_task_for_job, get_task as get_v5_remote_task, update_task as update_v5_remote_task
from web_auth import (
    admin_token_enabled as _admin_token_enabled,
    admin_token_expected as _admin_token_expected,
    admin_token_valid as _admin_token_valid,
    verify_callback_hmac as _verify_callback_hmac,
)
from web_admin import console as web_admin_console
from web_admin import jobs_page as web_admin_jobs
from web_admin import operations as web_admin_ops
from web_admin import remote_predictors as web_remote_predictors
from web_admin import task_routes as web_admin_tasks
from web_data import odds_service, run_resolver, run_store, summary_service, view_data
from web_helpers import (
    extract_section,
    extract_top5,
    format_path_mtime,
    get_env_timeout,
    is_run_id,
    load_csv_rows,
    load_csv_rows_flexible,
    load_json_file,
    load_text_file,
    normalize_race_id,
    parse_horse_no,
    parse_run_id,
    run_script,
    to_float,
    to_int_or_none,
)
from web_note import build_mark_note_text
from web_public import (
    CONSOLE_BASE_PATH,
    PUBLIC_API_BASE_PATH,
    PUBLIC_BASE_PATH,
    PUBLIC_SHARE_DETAIL_LABEL,
    PUBLIC_SHARE_HASHTAG,
    PUBLIC_SHARE_MAX_CHARS,
    PUBLIC_SHARE_URL,
    build_public_index_response,
    mount_public_assets,
    prefix_public_html_routes as _prefix_public_html_routes,
    register_public_static_routes,
)
from web_pages import public_llm as web_public_llm
from web_pages import workspace as web_workspace
from web_policy import engine as web_policy_engine
from web_policy import html as web_policy_html
from web_policy import payload as web_policy_payload
from web_policy import runtime as web_policy_runtime
from web_report import bundles as report_bundles
from web_report import data as report_data
from web_report.helpers import (
    BET_TYPE_TEXT_MAP as REPORT_BET_TYPE_TEXT_MAP,
    LLM_BATTLE_LABELS as REPORT_LLM_BATTLE_LABELS,
    LLM_BATTLE_ORDER as REPORT_LLM_BATTLE_ORDER,
    LLM_BATTLE_SHORT_LABELS as REPORT_LLM_BATTLE_SHORT_LABELS,
    LLM_NOTE_LABELS as REPORT_LLM_NOTE_LABELS,
    LLM_REPORT_SCOPE_KEYS as REPORT_LLM_REPORT_SCOPE_KEYS,
    build_battle_title as report_build_battle_title,
    format_jp_date_text as report_format_jp_date_text,
    format_jp_date_value as report_format_jp_date_value,
    format_marks_text as report_format_marks_text,
    format_percent_text as report_format_percent_text,
    format_race_label as report_format_race_label,
    format_ticket_plan_text as report_format_ticket_plan_text,
    format_yen_text as report_format_yen_text,
    has_llm_policy_assets as report_has_llm_policy_assets,
    jst_today_text as report_jst_today_text,
    llm_today_scope_keys as report_llm_today_scope_keys,
    llm_today_scope_label_ja as report_llm_today_scope_label_ja,
    normalize_report_date_text as report_normalize_report_date_text,
    parse_run_date as report_parse_run_date,
    payload_run_id as report_payload_run_id,
    policy_marks_map as report_policy_marks_map,
    policy_primary_budget as report_policy_primary_budget,
    policy_primary_output as report_policy_primary_output,
    public_scope_label_ja as report_public_scope_label_ja,
    race_no_text as report_race_no_text,
    report_scope_key_for_row as report_scope_key_for_row,
    run_date_key as report_run_date_key,
    safe_text as report_safe_text,
    scope_display_name as report_scope_display_name,
)
from web_ui.components import (
    build_daily_profit_chart_html as ui_build_daily_profit_chart_html,
    build_metric_table as ui_build_metric_table,
    build_table_html as ui_build_table_html,
)
from web_ui.template import page_template as ui_page_template
from web_ui.stats_block import build_llm_compare_block as ui_build_llm_compare_block
from web_ui.stats_block import build_stats_block as ui_build_stats_block


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ADS_TXT_PATH = BASE_DIR / "ads.txt"
RUN_PIPELINE = BASE_DIR / "run_pipeline.py"
ODDS_EXTRACT = ROOT_DIR / "odds_extract.py"
RECORD_PREDICTOR = BASE_DIR / "record_predictor_result.py"
OPTIMIZE_PARAMS = BASE_DIR / "optimize_params.py"
OPTIMIZE_PREDICTOR = BASE_DIR / "optimize_predictor_params.py"
OFFLINE_EVAL = BASE_DIR / "offline_eval.py"
INIT_UPDATE = BASE_DIR / "init_update.py"
DEFAULT_RUN_LIMIT = 200
MAX_RUN_LIMIT = 500
app = FastAPI()
load_local_env(BASE_DIR, override=False)
mount_public_assets(app)
register_public_static_routes(app)


def _pick_next_process_job_id():
    return web_admin_tasks.pick_next_process_job_id(
        load_race_jobs=lambda: load_race_jobs(BASE_DIR),
    )


def _pick_next_settle_job_id():
    return web_admin_tasks.pick_next_settle_job_id(
        load_race_jobs=lambda: load_race_jobs(BASE_DIR),
    )


def run_due_jobs_once():
    return web_admin_tasks.run_due_jobs_once(
        base_dir=BASE_DIR,
        scan_due_race_jobs=scan_due_race_jobs,
        load_race_jobs=load_race_jobs,
    )


def build_console_gate_page(admin_token="", error_text=""):
    return web_admin_console.build_console_gate_page(
        admin_token=admin_token,
        error_text=error_text,
        prefix_public_html_routes=_prefix_public_html_routes,
    )


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


def remote_v5_enabled():
    return web_remote_predictors.remote_v5_enabled()


def remote_predictor_auto_continue_enabled():
    return web_remote_predictors.remote_predictor_auto_continue_enabled()


def _append_job_process_log_entry(row, step, code, output):
    return web_remote_predictors.append_job_process_log_entry(row, step, code, output)


def _remote_v5_bundle_zip_bytes(task):
    return web_remote_predictors.remote_v5_bundle_zip_bytes(task)


def _promote_job_after_remote_v5(job_id, run_id, task_id, log_output):
    return web_remote_predictors.promote_job_after_remote_v5(
        base_dir=BASE_DIR,
        job_id=job_id,
        run_id=run_id,
        task_id=task_id,
        log_output=log_output,
        update_race_job=update_race_job,
        initialize_race_job_step_fields=initialize_race_job_step_fields,
        set_race_job_step_state=set_race_job_step_state,
    )


def _auto_continue_remote_policy(job_id):
    return web_remote_predictors.auto_continue_remote_policy(base_dir=BASE_DIR, job_id=job_id)


def strict_llm_odds_gate_enabled():
    return web_policy_runtime.strict_llm_odds_gate_enabled()


def expected_odds_output_names(scope_key):
    return web_policy_runtime.expected_odds_output_names(scope_key)


def capture_output_mtimes(root_dir, names):
    return web_policy_runtime.capture_output_mtimes(root_dir, names)


def is_fresh_output(path, previous_mtime, started_at):
    return web_policy_runtime.is_fresh_output(path, previous_mtime, started_at)


def copy_fresh_odds_output(tmp_path, dest_path, before_mtimes, started_at, warnings):
    return web_policy_runtime.copy_fresh_odds_output(tmp_path, dest_path, before_mtimes, started_at, warnings)


def refresh_odds_for_run(
    run_row,
    scope_key,
    odds_path,
    wide_odds_path=None,
    fuku_odds_path=None,
    quinella_odds_path=None,
    exacta_odds_path=None,
    trio_odds_path=None,
    trifecta_odds_path=None,
):
    return web_policy_runtime.refresh_odds_for_run(
        run_row,
        scope_key,
        odds_path,
        wide_odds_path=wide_odds_path,
        fuku_odds_path=fuku_odds_path,
        quinella_odds_path=quinella_odds_path,
        exacta_odds_path=exacta_odds_path,
        trio_odds_path=trio_odds_path,
        trifecta_odds_path=trifecta_odds_path,
        get_env_timeout=get_env_timeout,
        normalize_race_id=normalize_race_id,
        odds_extract_path=ODDS_EXTRACT,
        root_dir=ROOT_DIR,
    )


def find_run_in_scope(scope_key, id_text):
    return run_resolver.find_run_in_scope(load_runs, normalize_race_id, scope_key, id_text)


def infer_scope_and_run(id_text):
    return run_resolver.infer_scope_and_run(load_runs, normalize_race_id, id_text)


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


def normalize_name(value):
    return summary_service.normalize_name(value)


def pick_score_key(rows):
    return view_data.pick_score_key(rows)


def load_top5_names(path):
    return summary_service.load_top5_names(load_csv_rows, to_float, path)


def resolve_pred_path(scope_key, run_id, run_row):
    return run_resolver.resolve_pred_path(get_data_dir, BASE_DIR, scope_key, run_id, run_row)


def resolve_odds_path(scope_key, run_id, run_row):
    return run_resolver.resolve_odds_path(get_data_dir, BASE_DIR, scope_key, run_id, run_row)


def resolve_run_asset_path(scope_key, run_id, run_row, field_name, prefix, ext=".csv"):
    return run_resolver.resolve_run_asset_path(
        get_data_dir,
        BASE_DIR,
        scope_key,
        run_id,
        run_row,
        field_name,
        prefix,
        ext,
    )


def resolve_predictor_paths(scope_key, run_id, run_row):
    data_dir = get_data_dir(BASE_DIR, scope_key)
    race_id = str(run_row.get("race_id", "") or "") if run_row else ""
    last_run_id = str(run_row.get("run_id", "") or "") if run_row else ""
    out = []
    for spec in list_predictors():
        path = resolve_run_prediction_path(
            run_row,
            data_dir,
            ROOT_DIR,
            run_id=run_id,
            race_id=race_id,
            predictor_id=spec["id"],
            last_run_id=last_run_id,
        )
        out.append((spec, Path(path) if path else None))
    return out


def build_predictor_env(scope_norm, resolved_run_id, run_row):
    predictor_env = {}
    for spec, pred_path in resolve_predictor_paths(scope_norm, resolved_run_id, run_row):
        if not pred_path or not pred_path.exists():
            continue
        env_name = {
            "main": "PRED_PATH",
            "v2_opus": "PRED_PATH_V2_OPUS",
            "v3_premium": "PRED_PATH_V3_PREMIUM",
            "v4_gemini": "PRED_PATH_V4_GEMINI",
            "v5_stacking": "PRED_PATH_V5_STACKING",
        }.get(spec["id"])
        if env_name:
            predictor_env[env_name] = str(pred_path)
    odds_env_map = {
        "odds_path": ("ODDS_PATH", "odds"),
        "fuku_odds_path": ("FUKU_ODDS_PATH", "fuku_odds"),
        "wide_odds_path": ("WIDE_ODDS_PATH", "wide_odds"),
        "quinella_odds_path": ("QUINELLA_ODDS_PATH", "quinella_odds"),
        "exacta_odds_path": ("EXACTA_ODDS_PATH", "exacta_odds"),
        "trio_odds_path": ("TRIO_ODDS_PATH", "trio_odds"),
        "trifecta_odds_path": ("TRIFECTA_ODDS_PATH", "trifecta_odds"),
    }
    for field_name, (env_name, prefix) in odds_env_map.items():
        path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, field_name, prefix)
        if path and path.exists():
            predictor_env[env_name] = str(path)
    return predictor_env


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


def load_policy_bankroll_summary(run_id="", timestamp="", policy_engine="gemini"):
    return web_policy_runtime.load_policy_bankroll_summary(
        base_dir=BASE_DIR,
        run_id=run_id,
        timestamp=timestamp,
        policy_engine=policy_engine,
        extract_ledger_date=extract_ledger_date,
        summarize_bankroll=summarize_bankroll,
    )


def load_gemini_bankroll_summary(run_id="", timestamp=""):
    return load_policy_bankroll_summary(run_id, timestamp, policy_engine="gemini")


def load_policy_daily_profit_summary(days=30, policy_engine="gemini"):
    return web_policy_runtime.load_policy_daily_profit_summary(
        base_dir=BASE_DIR,
        days=days,
        policy_engine=policy_engine,
        load_daily_profit_rows=load_daily_profit_rows,
    )


def load_gemini_daily_profit_summary(days=30):
    return load_policy_daily_profit_summary(days=days, policy_engine="gemini")


def load_policy_run_ticket_rows(run_id, policy_engine="gemini"):
    return web_policy_runtime.load_policy_run_ticket_rows(
        base_dir=BASE_DIR,
        run_id=run_id,
        policy_engine=policy_engine,
        load_run_tickets=load_run_tickets,
    )


def load_gemini_run_ticket_rows(run_id):
    return load_policy_run_ticket_rows(run_id, policy_engine="gemini")


def build_llm_buy_output(summary_before, refresh_ok, refresh_message, refresh_warnings, script_output, policy_engine=""):
    return web_policy_runtime.build_llm_buy_output(
        summary_before,
        refresh_ok,
        refresh_message,
        refresh_warnings,
        script_output,
        policy_engine,
    )


def build_gemini_buy_output(summary_before, refresh_ok, refresh_message, refresh_warnings, script_output):
    return build_llm_buy_output(summary_before, refresh_ok, refresh_message, refresh_warnings, script_output, "gemini")



def normalize_horse_no_text(value):
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = parse_horse_no(text)
    if parsed is not None:
        return str(parsed)
    return text


def _pick_first_value(row, keys, default=""):
    for key in keys:
        if key in row:
            value = row.get(key)
            text = str(value or "").strip()
            if text:
                return value
    return default


def _prediction_prob_value(row):
    for key in (
        "Top3Prob_model",
        "top3_prob_model",
        "Top3Prob_est",
        "top3_prob_est",
        "Top3Prob",
        "top3_prob",
        "agg_score",
        "score",
    ):
        if key in row:
            return max(0.0, to_float(row.get(key)))
    return 0.0


def _prediction_rank_value(row, fallback_rank):
    explicit = to_int_or_none(_pick_first_value(row, ("pred_rank", "PredRank", "rank", "Rank"), ""))
    return explicit if explicit is not None and explicit > 0 else int(fallback_rank)


def _prediction_score_value(row):
    for key in (
        "rank_score_norm",
        "rank_score",
        "RankScore",
        "agg_score",
        "Top3Prob_model",
        "top3_prob_model",
        "Top3Prob_est",
        "top3_prob_est",
        "Top3Prob",
        "top3_prob",
        "score",
    ):
        if key in row:
            value = to_float(row.get(key))
            if value:
                return value
    return 0.0


def _linear_rank_norm(index, total):
    if total <= 1:
        return 1.0
    return round(max(0.05, 1.0 - (float(index) / float(total - 1))), 6)


def _normalize_score_list(items):
    if not items:
        return items
    values = [float(item.get("_raw_rank_score", 0.0) or 0.0) for item in items]
    max_v = max(values)
    min_v = min(values)
    if max_v > min_v:
        span = max_v - min_v
        for item in items:
            raw = float(item.get("_raw_rank_score", 0.0) or 0.0)
            item["rank_score_norm"] = round((raw - min_v) / span, 6)
        return items
    for idx, item in enumerate(items):
        item["rank_score_norm"] = _linear_rank_norm(idx, len(items))
    return items


def build_policy_prediction_rows(pred_rows, name_to_no_map, win_odds_map, place_odds_map):
    return web_policy_payload.build_policy_prediction_rows(
        pred_rows,
        name_to_no_map,
        win_odds_map,
        place_odds_map,
        normalize_horse_no_text=normalize_horse_no_text,
        normalize_name=normalize_name,
        parse_horse_no=parse_horse_no,
        to_float=to_float,
        to_int_or_none=to_int_or_none,
    )


def _build_predictor_history_summary(scope_key, items, predictor_ids):
    rows_by_predictor = {}
    for row in items:
        predictor_id = canonical_predictor_id(row.get("predictor_id"))
        if predictor_ids and predictor_id not in predictor_ids:
            continue
        rows_by_predictor.setdefault(predictor_id, []).append(row)
    summary = []
    for predictor_id in predictor_ids or sorted(rows_by_predictor.keys()):
        predictor_rows = rows_by_predictor.get(predictor_id, [])
        total = len(predictor_rows)
        top1_hit = sum(int(float(r.get("top1_hit", 0) or 0)) for r in predictor_rows)
        top1_in_top3 = sum(int(float(r.get("top1_in_top3", 0) or 0)) for r in predictor_rows)
        top3_exact = sum(int(float(r.get("top3_exact", 0) or 0)) for r in predictor_rows)
        top3_hit = sum(int(float(r.get("top3_hit_count", 0) or 0)) for r in predictor_rows)
        top5_hit = 0
        top5_total = 0
        for row in predictor_rows:
            hit_count = compute_top5_hit_count(scope_key, row)
            if hit_count is None:
                continue
            top5_hit += hit_count
            top5_total += 1
        summary.append(
            {
                "predictor_id": predictor_id,
                "predictor_label": predictor_label(predictor_id),
                "samples": total,
                "top1_hit_rate": round(top1_hit / total, 4) if total else "",
                "top1_in_top3_rate": round(top1_in_top3 / total, 4) if total else "",
                "top3_hit_rate": round(top3_hit / (3 * total), 4) if total else "",
                "top3_exact_rate": round(top3_exact / total, 4) if total else "",
                "top5_to_top3_hit_rate": round(top5_hit / (3 * top5_total), 4) if top5_total else "",
            }
        )
    return summary


def build_predictor_performance_context(scope_key, run_id, run_row, predictor_ids):
    predictor_ids = [str(item or "").strip() for item in list(predictor_ids or []) if str(item or "").strip()]
    path = get_data_dir(BASE_DIR, scope_key) / "predictor_results.csv"
    predictor_rows = load_csv_rows(path)
    scope_label_map = {
        "central_turf": "中央草地",
        "central_dirt": "中央泥地",
        "local": "地方",
    }
    filtered_rows = []
    for row in predictor_rows:
        row_run_id = str(row.get("run_id", "") or "").strip()
        if row_run_id and row_run_id == str(run_id or "").strip():
            continue
        filtered_rows.append(row)
    if not predictor_rows:
        return {
            "current_context": {
                "scope_key": str(scope_key or ""),
                "scope_label_ja": scope_label_map.get(str(scope_key or ""), str(scope_key or "")),
            },
            "current_scope_history": _build_predictor_history_summary(scope_key, [], predictor_ids),
        }
    return {
        "current_context": {
            "scope_key": str(scope_key or ""),
            "scope_label_ja": scope_label_map.get(str(scope_key or ""), str(scope_key or "")),
        },
        "current_scope_history": _build_predictor_history_summary(scope_key, filtered_rows, predictor_ids),
    }


def build_multi_predictor_context(scope_key, run_id, run_row, name_to_no_map, win_odds_map, place_odds_map):
    profiles = []
    summaries = []
    consensus = {}
    available_ids = []
    top1_horses = []
    for spec, pred_path in resolve_predictor_paths(scope_key, run_id, run_row):
        if not pred_path or not pred_path.exists():
            continue
        pred_rows = load_csv_rows_flexible(pred_path)
        ranking = build_policy_prediction_rows(pred_rows, name_to_no_map, win_odds_map, place_odds_map)
        if not ranking:
            continue
        available_ids.append(spec["id"])
        profiles.append(
            {
                "predictor_id": spec["id"],
                "predictor_label": spec["label"],
                "available": True,
                "style_ja": "",
                "strengths_ja": [],
            }
        )
        top_slice = ranking[:5]
        top_choice = top_slice[0]
        top1_horses.append(normalize_horse_no_text(top_choice.get("horse_no", "")))
        summaries.append(
            {
                "predictor_id": spec["id"],
                "predictor_label": spec["label"],
                "top_choice_horse_no": normalize_horse_no_text(top_choice.get("horse_no", "")),
                "top_choice_horse_name": str(top_choice.get("horse_name", "") or ""),
                "top_choice_top3_prob_model": round(float(top_choice.get("top3_prob_model", 0.0) or 0.0), 6),
                "top_horses": [
                    {
                        "horse_no": normalize_horse_no_text(item.get("horse_no", "")),
                        "horse_name": str(item.get("horse_name", "") or ""),
                        "pred_rank": int(item.get("pred_rank", 0) or 0),
                        "top3_prob_model": round(float(item.get("top3_prob_model", 0.0) or 0.0), 6),
                        "rank_score_norm": round(float(item.get("rank_score_norm", 0.0) or 0.0), 6),
                        "win_odds": round(float(item.get("win_odds", 0.0) or 0.0), 6),
                        "place_odds": round(float(item.get("place_odds", 0.0) or 0.0), 6),
                    }
                    for item in top_slice
                ],
            }
        )
        for item in top_slice:
            horse_no = normalize_horse_no_text(item.get("horse_no", ""))
            if not horse_no:
                continue
            entry = consensus.setdefault(
                horse_no,
                {
                    "horse_no": horse_no,
                    "horse_name": str(item.get("horse_name", "") or ""),
                    "top1_votes": 0,
                    "top3_votes": 0,
                    "predictor_count": 0,
                    "pred_rank_total": 0.0,
                    "top3_prob_total": 0.0,
                    "rank_score_total": 0.0,
                    "win_odds": round(float(item.get("win_odds", 0.0) or 0.0), 6),
                    "place_odds": round(float(item.get("place_odds", 0.0) or 0.0), 6),
                    "predictors_support": [],
                },
            )
            entry["predictor_count"] += 1
            entry["pred_rank_total"] += float(item.get("pred_rank", 0) or 0)
            entry["top3_prob_total"] += float(item.get("top3_prob_model", 0.0) or 0.0)
            entry["rank_score_total"] += float(item.get("rank_score_norm", 0.0) or 0.0)
            if int(item.get("pred_rank", 99) or 99) == 1:
                entry["top1_votes"] += 1
            if int(item.get("pred_rank", 99) or 99) <= 3:
                entry["top3_votes"] += 1
            entry["predictors_support"].append(spec["label"])
    consensus_rows = []
    for horse_no, entry in consensus.items():
        count = max(1, int(entry.get("predictor_count", 0) or 0))
        consensus_rows.append(
            {
                "horse_no": horse_no,
                "horse_name": str(entry.get("horse_name", "") or ""),
                "top1_votes": int(entry.get("top1_votes", 0) or 0),
                "top3_votes": int(entry.get("top3_votes", 0) or 0),
                "predictor_count": count,
                "avg_pred_rank": round(float(entry.get("pred_rank_total", 0.0) or 0.0) / count, 4),
                "avg_top3_prob_model": round(float(entry.get("top3_prob_total", 0.0) or 0.0) / count, 6),
                "avg_rank_score_norm": round(float(entry.get("rank_score_total", 0.0) or 0.0) / count, 6),
                "win_odds": round(float(entry.get("win_odds", 0.0) or 0.0), 6),
                "place_odds": round(float(entry.get("place_odds", 0.0) or 0.0), 6),
                "predictors_support": list(entry.get("predictors_support", []) or []),
            }
        )
    consensus_rows.sort(
        key=lambda item: (
            -int(item.get("top1_votes", 0) or 0),
            -int(item.get("top3_votes", 0) or 0),
            float(item.get("avg_pred_rank", 999.0) or 999.0),
            str(item.get("horse_no", "")),
        )
    )
    return {
        "profiles": profiles,
        "summaries": summaries,
        "consensus": consensus_rows[:8],
        "performance": build_predictor_performance_context(scope_key, run_id, run_row, available_ids),
        "meta": {
            "available_predictor_ids": available_ids,
            "available_predictor_count": len(available_ids),
            "unique_top1_horses": sorted({horse for horse in top1_horses if horse}),
            "unique_top1_count": len({horse for horse in top1_horses if horse}),
            "consensus_top_horse_no": str(consensus_rows[0].get("horse_no", "") or "") if consensus_rows else "",
        },
    }


def build_policy_candidates(
    predictions,
    wide_odds_map,
    quinella_odds_map,
    exacta_odds_map,
    trio_odds_map,
    trifecta_odds_map,
    allowed_types,
):
    return web_policy_payload.build_policy_candidates(
        predictions,
        wide_odds_map,
        quinella_odds_map,
        exacta_odds_map,
        trio_odds_map,
        trifecta_odds_map,
        allowed_types,
    )


def build_pair_odds_top(candidate_lookup):
    return web_policy_payload.build_pair_odds_top(candidate_lookup)


def build_odds_full(win_rows, place_rows, wide_rows, quinella_rows, exacta_rows=None, trio_rows=None, trifecta_rows=None):
    return web_policy_payload.build_odds_full(
        win_rows,
        place_rows,
        wide_rows,
        quinella_rows,
        exacta_rows,
        trio_rows,
        trifecta_rows,
        to_float=to_float,
    )


def build_prediction_field_guide():
    return web_policy_payload.build_prediction_field_guide()


def build_policy_input_payload(
    scope_key,
    run_id,
    run_row,
    pred_path,
    odds_path,
    fuku_odds_path,
    wide_odds_path,
    quinella_odds_path,
    exacta_odds_path,
    trio_odds_path,
    trifecta_odds_path,
    policy_engine,
):
    return web_policy_payload.build_policy_input_payload(
        scope_key,
        run_id,
        run_row,
        pred_path,
        odds_path,
        fuku_odds_path,
        wide_odds_path,
        quinella_odds_path,
        exacta_odds_path,
        trio_odds_path,
        trifecta_odds_path,
        policy_engine,
        load_csv_rows_flexible=load_csv_rows_flexible,
        load_name_to_no=load_name_to_no,
        load_win_odds_map=load_win_odds_map,
        load_place_odds_map=load_place_odds_map,
        load_pair_odds_map=load_pair_odds_map,
        load_exacta_odds_map=load_exacta_odds_map,
        load_triple_odds_map=load_triple_odds_map,
        build_policy_prediction_rows_fn=build_policy_prediction_rows,
        extract_ledger_date=extract_ledger_date,
        summarize_bankroll=summarize_bankroll,
        base_dir=BASE_DIR,
        build_multi_predictor_context=build_multi_predictor_context,
        build_history_context=build_history_context,
        to_float=to_float,
    )


def apply_local_ticket_plan_fallback(output_dict, candidate_lookup, race_budget_yen):
    return web_policy_engine.apply_local_ticket_plan_fallback(output_dict, candidate_lookup, race_budget_yen)


def build_policy_ticket_rows(policy_output, candidate_lookup, horse_map, policy_engine):
    output_dict = dict(policy_output or {})
    tickets = []
    for item in list(output_dict.get("ticket_plan", []) or []):
        bet_type = str(item.get("bet_type", "") or "").strip().lower()
        legs = [str(x or "").strip() for x in list(item.get("legs", []) or []) if str(x or "").strip()]
        stake_yen = max(0, int(item.get("stake_yen", 0) or 0))
        if not bet_type or not legs or stake_yen <= 0:
            continue
        # Look up odds from candidate_lookup using the canonical key format
        # For unordered bet types (wide, quinella, trio), sort legs to match lookup key
        lookup_legs = sorted(legs, key=lambda x: int(x) if x.isdigit() else x) if bet_type in ("wide", "quinella", "trio") else legs
        ticket_key = f"{bet_type}:{'-'.join(lookup_legs)}"
        candidate = dict(candidate_lookup.get(ticket_key, {}) or {})
        if not candidate:
            continue
        odds_used = round(float(candidate.get("odds_used", 0.0) or 0.0), 6)
        p_hit = round(float(candidate.get("p_hit", 0.0) or 0.0), 6)
        ev = round(float(candidate.get("ev", 0.0) or 0.0), 6)
        horse_names = []
        for leg in legs:
            horse = dict(horse_map.get(leg, {}) or {})
            horse_names.append(str(horse.get("horse_name", "") or leg))
        expected_return_yen = int(round(stake_yen * max(0.0, odds_used))) if odds_used > 0 else 0
        tickets.append(
            {
                "ticket_id": ticket_key,
                "budget_yen": 0,
                "bet_type": bet_type,
                "horse_no": "-".join(legs),
                "horse_name": " / ".join(horse_names),
                "units": max(1, stake_yen // 100),
                "amount_yen": stake_yen,
                "hit_prob_est": p_hit,
                "hit_prob_se": "",
                "hit_prob_ci95_low": "",
                "hit_prob_ci95_high": "",
                "payout_mult": odds_used,
                "ev_ratio_est": round(p_hit * odds_used, 6) if p_hit > 0 else 0.0,
                "expected_return_yen": expected_return_yen,
                "odds_used": odds_used,
                "p_hit": p_hit,
                "edge": ev,
                "kelly_f": 0.0,
                "score": round(float(candidate.get("score", 0.0) or 0.0), 6),
                "stake_yen": stake_yen,
                "notes": "policy_pool=shared;policy={buy_style};decision={decision};construction={construction};reasons={reasons}".format(
                    buy_style=str(output_dict.get("buy_style", "") or ""),
                    decision=str(output_dict.get("bet_decision", "") or ""),
                    construction=str(output_dict.get("strategy_mode", "") or ""),
                    reasons=",".join(str(x) for x in list(output_dict.get("reason_codes", []) or [])),
                ),
                "strategy_text_ja": str(output_dict.get("strategy_text_ja", "") or ""),
                "bet_tendency_ja": str(output_dict.get("bet_tendency_ja", "") or ""),
                "policy_engine": policy_engine,
                "policy_buy_style": str(output_dict.get("buy_style", "") or ""),
                "policy_bet_decision": str(output_dict.get("bet_decision", "") or ""),
                "policy_construction_style": str(output_dict.get("strategy_mode", "") or ""),
            }
        )
    return tickets


def _normalize_output_ticket_plan_from_rows(ticket_rows):
    return web_policy_engine.normalize_output_ticket_plan_from_rows(ticket_rows)

def _append_output_warning(output_dict, code):
    return web_policy_engine.append_output_warning(output_dict, code)

def _set_output_no_bet(output_dict):
    return web_policy_engine.set_output_no_bet(output_dict)

def save_policy_payload(scope_key, run_id, race_id, payload, policy_engine):
    return web_policy_engine.save_policy_payload(
        base_dir=BASE_DIR,
        scope_key=scope_key,
        run_id=run_id,
        race_id=race_id,
        payload=payload,
        policy_engine=policy_engine,
        normalize_scope_key=normalize_scope_key,
        get_data_dir=get_data_dir,
        normalize_policy_engine=normalize_policy_engine,
    )

def execute_policy_buy(scope_key, run_row, run_id, policy_engine="gemini", policy_model=""):
    return web_policy_engine.execute_policy_buy(
        base_dir=BASE_DIR,
        scope_key=scope_key,
        run_row=run_row,
        run_id=run_id,
        policy_engine=policy_engine,
        policy_model=policy_model,
        normalize_scope_key=normalize_scope_key,
        normalize_policy_engine=normalize_policy_engine,
        resolve_policy_model=resolve_policy_model,
        default_gemini_model=DEFAULT_GEMINI_MODEL,
        resolve_pred_path=resolve_pred_path,
        resolve_run_asset_path=resolve_run_asset_path,
        build_policy_input_payload=build_policy_input_payload,
        call_policy=call_policy,
        resolve_policy_timeout=resolve_policy_timeout,
        get_last_call_meta=get_last_call_meta,
        reserve_run_tickets=reserve_run_tickets,
        summarize_bankroll=summarize_bankroll,
        update_run_row_fields=update_run_row_fields,
        get_data_dir=get_data_dir,
    )

def resolve_run_selection(scope_key, run_id):
    return web_policy_runtime.resolve_run_selection(
        scope_key=scope_key,
        run_id=run_id,
        normalize_scope_key=normalize_scope_key,
        infer_scope_and_run=infer_scope_and_run,
        resolve_run=resolve_run,
        normalize_race_id=normalize_race_id,
        resolve_latest_run_by_race_id=resolve_latest_run_by_race_id,
    )


def maybe_refresh_run_odds(scope_norm, run_row, run_id, refresh_enabled):
    return web_policy_runtime.maybe_refresh_run_odds(
        scope_norm=scope_norm,
        run_row=run_row,
        run_id=run_id,
        refresh_enabled=refresh_enabled,
        resolve_run_asset_path=resolve_run_asset_path,
        refresh_odds_for_run=refresh_odds_for_run,
    )


def resolve_policy_timeout(policy_engine):
    return web_policy_runtime.resolve_policy_timeout(
        policy_engine=policy_engine,
        normalize_policy_engine=normalize_policy_engine,
    )


def load_mark_recommendation_table(scope_key, run_id, run_row=None):
    return view_data.load_mark_recommendation_table(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        to_float,
        scope_key,
        run_id,
        run_row,
    )


def load_bet_engine_v3_cfg_summary(scope_key, run_id):
    scope_norm = normalize_scope_key(scope_key)
    if not scope_norm or not run_id:
        return {}
    data_dir = get_data_dir(BASE_DIR, scope_norm)
    path = data_dir / f"bet_engine_v3_cfg_{run_id}.json"
    return load_json_file(path)


def load_policy_payload(scope_key, run_id, run_row=None):
    return web_policy_html.load_policy_payload(
        scope_key,
        run_id,
        run_row,
        resolve_run_asset_path=resolve_run_asset_path,
        load_json_file=load_json_file,
    )


def load_policy_payloads(scope_key, run_id, run_row=None):
    return web_policy_html.load_policy_payloads(
        scope_key,
        run_id,
        run_row,
        resolve_run_asset_path=resolve_run_asset_path,
        load_json_file=load_json_file,
        normalize_policy_engine=normalize_policy_engine,
    )


def load_gemini_policy_payload(scope_key, run_id, run_row=None):
    return load_policy_payload(scope_key, run_id, run_row)


def _policy_chip_row(label, values, tone=""):
    return web_policy_html._policy_chip_row(label, values, tone=tone)


def _policy_detail_row(label, value, code=False):
    return web_policy_html._policy_detail_row(label, value, code=code)


def _policy_mark_html(marks):
    return web_policy_html._policy_mark_html(marks)


def _policy_ticket_html(rows):
    return web_policy_html._policy_ticket_html(rows)


def build_policy_html(payload):
    return web_policy_html.build_policy_html(payload, to_float=to_float)


def build_gemini_policy_html(payload):
    return web_policy_html.build_gemini_policy_html(payload, to_float=to_float)


def build_policy_workspace_html(payloads):
    return web_policy_html.build_policy_workspace_html(payloads, to_float=to_float)


def load_ability_marks_table(scope_key, run_id, run_row=None):
    return view_data.load_ability_marks_table(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        to_float,
        scope_key,
        run_id,
        run_row,
    )


def page_template(
    output_text="",
    error_text="",
    run_options="",
    view_run_options="",
    view_selected_run_id="",
    current_race_id="",
    top5_text="",
    top5_table_html="",
    mark_table_html="",
    mark_note_text="",
    llm_battle_html="",
    llm_note_text="",
    llm_compare_html="",
    gemini_policy_html="",
    daily_report_html="",
    daily_report_text="",
    weekly_report_html="",
    weekly_report_text="",
    summary_table_html="",
    stats_block="",
    default_scope="central_dirt",
    default_policy_engine="gemini",
    default_policy_model="",
    admin_token="",
    admin_enabled=False,
    admin_workspace_html="",
):
    return ui_page_template(
        output_text=output_text,
        error_text=error_text,
        run_options=run_options,
        view_run_options=view_run_options,
        view_selected_run_id=view_selected_run_id,
        current_race_id=current_race_id,
        top5_text=top5_text,
        top5_table_html=top5_table_html,
        mark_table_html=mark_table_html,
        mark_note_text=mark_note_text,
        llm_battle_html=llm_battle_html,
        llm_note_text=llm_note_text,
        llm_compare_html=llm_compare_html,
        gemini_policy_html=gemini_policy_html,
        daily_report_html=daily_report_html,
        daily_report_text=daily_report_text,
        weekly_report_html=weekly_report_html,
        weekly_report_text=weekly_report_text,
        summary_table_html=summary_table_html,
        stats_block=stats_block,
        default_scope=default_scope,
        default_policy_engine=default_policy_engine,
        default_policy_model=default_policy_model,
        admin_token=admin_token,
        admin_enabled=admin_enabled,
        admin_workspace_html=admin_workspace_html,
    )


def render_page(
    scope_key="central_dirt",
    output_text="",
    error_text="",
    top5_text="",
    summary_run_id="",
    selected_run_id="",
    admin_token="",
    admin_workspace_html="",
):
    return web_workspace.render_workspace_page(
        scope_key=scope_key,
        output_text=output_text,
        error_text=error_text,
        top5_text=top5_text,
        summary_run_id=summary_run_id,
        selected_run_id=selected_run_id,
        admin_token=admin_token,
        admin_workspace_html=admin_workspace_html,
        build_run_options=build_run_options,
        build_llm_compare_block=ui_build_llm_compare_block,
        build_stats_block=ui_build_stats_block,
        build_table_html=build_table_html,
        load_policy_bankroll_summary=load_policy_bankroll_summary,
        load_policy_daily_profit_summary=load_policy_daily_profit_summary,
        build_daily_profit_chart_html=ui_build_daily_profit_chart_html,
        resolve_run=resolve_run,
        parse_run_id=parse_run_id,
        normalize_race_id=normalize_race_id,
        resolve_predictor_paths=resolve_predictor_paths,
        load_top5_table=load_top5_table,
        load_ability_marks_table=load_ability_marks_table,
        load_mark_recommendation_table=load_mark_recommendation_table,
        load_text_file=load_text_file,
        build_mark_note_text=build_mark_note_text,
        load_prediction_summary=load_prediction_summary,
        load_bet_engine_v3_cfg_summary=load_bet_engine_v3_cfg_summary,
        load_policy_payloads=load_policy_payloads,
        load_policy_run_ticket_rows=load_policy_run_ticket_rows,
        build_policy_workspace_html=build_policy_workspace_html,
        build_llm_battle_bundle=_build_llm_battle_bundle,
        build_llm_daily_report_bundle=_build_llm_daily_report_bundle,
        build_llm_weekly_report_bundle=_build_llm_weekly_report_bundle,
        load_actual_result_map=_load_actual_result_map,
        normalize_policy_engine=normalize_policy_engine,
        resolve_policy_model=resolve_policy_model,
        default_gemini_model=DEFAULT_GEMINI_MODEL,
        prefix_public_html_routes=_prefix_public_html_routes,
        page_template=page_template,
        admin_token_enabled=_admin_token_enabled,
        load_predictor_summary=load_predictor_summary,
    )


def _admin_execution_denied(message, scope_key="", token="", selected_run_id="", summary_run_id=""):
    return render_page(
        scope_key,
        error_text=str(message or "管理员执行被拒绝。"),
        selected_run_id=selected_run_id,
        summary_run_id=summary_run_id,
        admin_token=token,
    )


LLM_BATTLE_ORDER = ("openai", "gemini", "deepseek", "grok")
LLM_BATTLE_LABELS = {
    "openai": "ChatGPT",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
}
LLM_NOTE_LABELS = {
    "openai": "ChatGPT",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
}
LLM_BATTLE_SHORT_LABELS = {
    "openai": "chatgpt",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "grok": "grok",
}
LLM_REPORT_SCOPE_KEYS = ("central_dirt", "central_turf", "local")
BET_TYPE_TEXT_MAP = {
    "win": "単勝",
    "place": "複勝",
    "wide": "ワイド",
    "quinella": "馬連",
    "exacta": "馬単",
    "trio": "三連複",
    "trifecta": "三連単",
}


def _scope_display_name(scope_key):
    mapping = {
        "central_dirt": "中央ダート",
        "central_turf": "中央芝",
        "local": "地方",
    }
    return mapping.get(str(scope_key or "").strip(), str(scope_key or "").strip() or "-")


def _safe_text(value):
    return str(value or "").strip()


def _policy_primary_budget(payload):
    budgets = list((payload or {}).get("budgets", []) or [])
    for item in budgets:
        if isinstance(item, dict):
            return item
    return {}


def _policy_primary_output(payload):
    item = _policy_primary_budget(payload)
    return dict(item.get("output", {}) or {})


def _policy_marks_map(payload):
    marks = {}
    for row in list(_policy_primary_output(payload).get("marks", []) or []):
        symbol = _safe_text(row.get("symbol"))
        horse_no = _safe_text(row.get("horse_no"))
        if symbol and horse_no and horse_no not in marks:
            marks[horse_no] = symbol
    return marks


def _format_bet_type_text(value):
    text = _safe_text(value).lower()
    return BET_TYPE_TEXT_MAP.get(text, text or "-")


def _format_ticket_target_text(bet_type, target):
    bet_type_text = _safe_text(bet_type).lower()
    text = _safe_text(target)
    if not text:
        return "-"
    parts = [part.strip() for part in text.split("-") if part.strip()]
    if bet_type_text == "exacta" and len(parts) == 2:
        return f"{parts[0]}→{parts[1]}"
    if bet_type_text == "trifecta" and len(parts) == 3:
        return f"{parts[0]}→{parts[1]}→{parts[2]}"
    return "-".join(parts) if parts else text


def _format_ticket_plan_text(ticket_rows, output):
    rows = list(ticket_rows or [])
    grouped = {}
    for row in rows:
        bet_type_key = _safe_text(row.get("bet_type")).lower()
        bet_type = _format_bet_type_text(bet_type_key)
        horse_no = _format_ticket_target_text(bet_type_key, row.get("horse_no"))
        amount = to_int_or_none(row.get("amount_yen"))
        if amount is None:
            amount = to_int_or_none(row.get("stake_yen"))
        amount_text = f"{amount}円" if amount is not None else "-"
        grouped.setdefault(bet_type, []).append(f"{horse_no}　{amount_text}")
    if grouped:
        lines = []
        for bet_type in [BET_TYPE_TEXT_MAP[key] for key in ("exacta", "quinella", "wide", "trio", "trifecta", "win", "place") if BET_TYPE_TEXT_MAP.get(key) in grouped]:
            lines.append(bet_type)
            lines.extend(grouped.get(bet_type, []))
            lines.append("")
        for bet_type, items in grouped.items():
            if bet_type in lines:
                continue
            lines.append(bet_type)
            lines.extend(items)
            lines.append("")
        return "\n".join(lines).strip()
    plan_rows = list(output.get("ticket_plan", []) or [])
    grouped = {}
    for row in plan_rows:
        ticket_id = _safe_text(row.get("id"))
        amount = to_int_or_none(row.get("stake_yen"))
        amount_text = f"{amount}円" if amount is not None else "-"
        if ticket_id:
            bet_type, _, target = ticket_id.partition(":")
        else:
            bet_type = _safe_text(row.get("bet_type")).lower()
            legs = [_safe_text(x) for x in list(row.get("legs", []) or []) if _safe_text(x)]
            if not bet_type or not legs:
                continue
            target = "-".join(legs)
        bet_type_label = _format_bet_type_text(bet_type)
        target_text = _format_ticket_target_text(bet_type, target or "-")
        grouped.setdefault(bet_type_label, []).append(f"{target_text}　{amount_text}")
    if grouped:
        lines = []
        for bet_type in [BET_TYPE_TEXT_MAP[key] for key in ("exacta", "quinella", "wide", "trio", "trifecta", "win", "place") if BET_TYPE_TEXT_MAP.get(key) in grouped]:
            lines.append(bet_type)
            lines.extend(grouped.get(bet_type, []))
            lines.append("")
        for bet_type, items in grouped.items():
            if bet_type in lines:
                continue
            lines.append(bet_type)
            lines.extend(items)
            lines.append("")
        return "\n".join(lines).strip()
    return "未生成"


def _format_marks_text(marks_map):
    if not marks_map:
        return "未生成"
    ordered = []
    symbol_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}
    for horse_no, symbol in marks_map.items():
        ordered.append((symbol_order.get(symbol, 99), horse_no, symbol))
    ordered.sort(key=lambda item: (item[0], to_int_or_none(item[1]) or 999, item[1]))
    return " ".join(f"{symbol}{horse_no}" for _, horse_no, symbol in ordered)


def _ticket_row_amount(row):
    amount = to_int_or_none(row.get("amount_yen"))
    if amount is None:
        amount = to_int_or_none(row.get("stake_yen"))
    return amount or 0


def _ticket_row_payout(row):
    payout = to_int_or_none(row.get("payout_yen"))
    if payout is None:
        payout = to_int_or_none(row.get("est_payout_yen"))
    return payout or 0


def _ticket_row_profit(row):
    profit = to_int_or_none(row.get("profit_yen"))
    if profit is not None:
        return profit
    return _ticket_row_payout(row) - _ticket_row_amount(row)


def _ticket_row_hit(row):
    text = _safe_text(row.get("hit"))
    return text in ("1", "true", "True", "yes", "Yes")


def _summarize_ticket_rows(ticket_rows):
    rows = list(ticket_rows or [])
    stake_yen = sum(_ticket_row_amount(row) for row in rows)
    payout_yen = sum(_ticket_row_payout(row) for row in rows)
    profit_yen = sum(_ticket_row_profit(row) for row in rows)
    hit_count = sum(1 for row in rows if _ticket_row_hit(row))
    status_values = {_safe_text(row.get("status")).lower() for row in rows if _safe_text(row.get("status"))}
    pending_count = sum(1 for value in status_values if value == "pending")
    settled_count = sum(1 for value in status_values if value == "settled")
    if rows and pending_count == 0 and (settled_count > 0 or not status_values):
        status = "settled"
    elif rows:
        status = "pending"
    else:
        status = "planned"
    roi = round(payout_yen / stake_yen, 4) if stake_yen > 0 else ""
    return {
        "stake_yen": stake_yen,
        "payout_yen": payout_yen,
        "profit_yen": profit_yen,
        "hit_count": hit_count,
        "ticket_count": len(rows),
        "status": status,
        "roi": roi,
    }


def _run_date_key(run_row):
    race_date = _safe_text((run_row or {}).get("race_date"))
    if race_date:
        return race_date
    timestamp = _safe_text((run_row or {}).get("timestamp"))
    return timestamp[:10] if len(timestamp) >= 10 else ""


def _parse_run_date(date_text):
    text = _safe_text(date_text)
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _format_jp_date_value(date_value):
    if not date_value:
        return ""
    return f"{date_value.year}年{date_value.month}月{date_value.day}日"


def _has_llm_policy_assets(run_row):
    row = dict(run_row or {})
    return bool(
        _safe_text(row.get("gemini_policy_path"))
        or _safe_text(row.get("deepseek_policy_path"))
        or _safe_text(row.get("siliconflow_policy_path"))
        or _safe_text(row.get("openai_policy_path"))
        or _safe_text(row.get("grok_policy_path"))
    )


def _report_scope_key_for_row(run_row, fallback_scope=""):
    row = dict(run_row or {})
    return normalize_scope_key(row.get("_report_scope_key") or row.get("scope_key") or fallback_scope) or ""


def _load_combined_llm_report_runs():
    return report_data.load_combined_llm_report_runs(load_runs, LLM_REPORT_SCOPE_KEYS)


def _payload_run_id(payload, fallback_run_id=""):
    return _safe_text((payload or {}).get("run_id")) or _safe_text(fallback_run_id)


def _race_no_text(race_id):
    digits = normalize_race_id(race_id)
    if len(digits) >= 2:
        try:
            return f"{int(digits[-2:])}R"
        except (TypeError, ValueError):
            return ""
    return ""


def _format_race_label(run_row):
    row = dict(run_row or {})
    venue = _safe_text(row.get("location")) or _safe_text(row.get("trigger_race"))
    race_no = _race_no_text(row.get("race_id"))
    if venue and race_no:
        return f"{venue}{race_no}"
    if venue:
        return venue
    if race_no:
        return race_no
    return _safe_text(row.get("race_id")) or _safe_text(row.get("run_id")) or "-"


def _format_jp_date_text(run_row):
    date_text = _safe_text((run_row or {}).get("race_date"))
    if not date_text:
        timestamp = _safe_text((run_row or {}).get("timestamp"))
        if len(timestamp) >= 10:
            date_text = timestamp[:10]
    if not date_text:
        return ""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_text)
    if not m:
        return date_text
    return f"{int(m.group(1))}年{int(m.group(2))}月{int(m.group(3))}日"


def _build_battle_title(run_row):
    race_label = _format_race_label(run_row)
    date_label = _format_jp_date_text(run_row)
    race_name = _safe_text((run_row or {}).get("trigger_race"))
    parts = ["第1回 いかいもAI競馬対決杯"]
    if date_label or race_label:
        detail = " ".join(part for part in [date_label, race_label] if part)
        if race_name:
            detail = f"{detail}（{race_name}）" if detail else f"（{race_name}）"
        if detail:
            parts.append(detail)
    return "｜".join(parts)


def _load_actual_result_map(scope_key):
    return report_data.load_actual_result_map(
        BASE_DIR,
        scope_key,
        get_data_dir=get_data_dir,
        load_csv_rows=load_csv_rows,
        canonical_predictor_id=canonical_predictor_id,
        load_race_jobs=load_race_jobs,
    )


def _find_actual_result_from_jobs(scope_key, run_id, run_row=None):
    return report_data.find_actual_result_from_jobs(
        BASE_DIR,
        scope_key,
        run_id,
        run_row,
        load_race_jobs=load_race_jobs,
    )


def _find_job_meta_for_run(scope_key, run_id, run_row=None):
    return report_data.find_job_meta_for_run(
        BASE_DIR,
        scope_key,
        run_id,
        run_row,
        load_race_jobs=load_race_jobs,
    )


def _load_name_to_no_map_for_run(scope_key, run_id, run_row):
    return report_data.load_name_to_no_map_for_run(
        scope_key,
        run_id,
        run_row,
        resolve_run_asset_path=resolve_run_asset_path,
        load_name_to_no=load_name_to_no,
        normalize_name=normalize_name,
        normalize_horse_no_text=normalize_horse_no_text,
    )


def _actual_result_snapshot(scope_key, run_id, run_row, actual_result_map):
    return report_data.actual_result_snapshot(
        BASE_DIR,
        scope_key,
        run_id,
        run_row,
        actual_result_map,
        load_race_jobs=load_race_jobs,
        resolve_run_asset_path=resolve_run_asset_path,
        load_name_to_no=load_name_to_no,
        normalize_name=normalize_name,
        normalize_horse_no_text=normalize_horse_no_text,
    )


def _marks_result_triplet(marks_map, actual_horse_nos):
    triplet = []
    for horse_no in list(actual_horse_nos or [])[:3]:
        triplet.append(marks_map.get(horse_no, "-") if horse_no else "-")
    while len(triplet) < 3:
        triplet.append("-")
    return triplet


def _format_triplet_text(triplet):
    values = [str(item or "-").strip() or "-" for item in list(triplet or [])[:3]]
    while len(values) < 3:
        values.append("-")
    return " ".join(values)


def _ratio_text(numerator, denominator):
    if not denominator:
        return "-"
    return f"{(float(numerator) / float(denominator)) * 100:.1f}%"


def _percent_text_from_ratio(value):
    if value in ("", None):
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _format_distance_label(value):
    text = _safe_text(value)
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    return f"{digits}m" if digits else text


def _format_confidence_text(value):
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "N/A"
    pct = max(0.0, min(100.0, ratio * 100.0))
    if pct >= 72.0:
        level = "HIGH"
    elif pct >= 58.0:
        level = "MID"
    else:
        level = "LOW"
    return f"{level} {pct:.0f}%"


def _public_trend_series(days=10):
    date_map = {}
    series = []
    for engine in LLM_BATTLE_ORDER:
        rows = load_policy_daily_profit_summary(days=days, policy_engine=engine)
        per_engine = {}
        for row in rows:
            date_text = _safe_text(row.get("date"))
            if not date_text:
                continue
            roi_ratio = row.get("roi", "")
            try:
                roi_value = round(float(roi_ratio or 0.0) * 100.0, 1)
            except (TypeError, ValueError):
                roi_value = None
            entry = {
                "date": date_text,
                "roi_value": roi_value,
                "roi_text": _percent_text_from_ratio(roi_ratio),
                "profit_yen": int(row.get("profit_yen", 0) or 0),
                "runs": int(row.get("runs", 0) or 0),
            }
            per_engine[date_text] = entry
            date_map[date_text] = True
        series.append(
            {
                "engine": engine,
                "label": LLM_BATTLE_LABELS.get(engine, engine),
                "points": per_engine,
            }
        )
    labels = sorted(date_map.keys())
    return {
        "labels": labels,
        "series": [
            {
                "engine": item["engine"],
                "label": item["label"],
                "points": [item["points"].get(label, {"date": label, "roi_value": None, "roi_text": "-", "profit_yen": 0, "runs": 0}) for label in labels],
            }
            for item in series
        ],
    }


def _public_all_time_roi_summary():
    return report_bundles.public_all_time_roi_summary(
        BASE_DIR,
        ledger_path=ledger_path,
        load_rows=load_rows,
    )


def _policy_primary_choice(output):
    return report_bundles._policy_primary_choice(output)

def _llm_agreement_summary(outputs):
    return report_bundles._llm_agreement_summary(outputs)

def _difficulty_summary(outputs, agreement_score):
    return report_bundles._difficulty_summary(outputs, agreement_score)

def _build_llm_battle_bundle(scope_key, run_id, run_row, payloads, actual_result_map):
    return report_bundles.build_llm_battle_bundle(
        scope_key,
        run_id,
        run_row,
        payloads,
        actual_result_map,
        normalize_policy_engine=normalize_policy_engine,
        actual_result_snapshot=_actual_result_snapshot,
        load_policy_run_ticket_rows=load_policy_run_ticket_rows,
        summarize_ticket_rows=_summarize_ticket_rows,
        marks_result_triplet=_marks_result_triplet,
        format_triplet_text=_format_triplet_text,
    )

def _build_llm_daily_report_bundle(scope_key, current_run_row, actual_result_map):
    return report_bundles.build_llm_daily_report_bundle(
        scope_key,
        current_run_row,
        actual_result_map,
        load_actual_result_map=_load_actual_result_map,
        load_combined_llm_report_runs=_load_combined_llm_report_runs,
        load_policy_payloads=load_policy_payloads,
        normalize_policy_engine=normalize_policy_engine,
        actual_result_snapshot=_actual_result_snapshot,
        load_policy_run_ticket_rows=load_policy_run_ticket_rows,
        summarize_ticket_rows=_summarize_ticket_rows,
        marks_result_triplet=_marks_result_triplet,
        format_triplet_text=_format_triplet_text,
        ratio_text=_ratio_text,
        percent_text_from_ratio=_percent_text_from_ratio,
        build_table_html=build_table_html,
    )

def _build_llm_weekly_report_bundle(scope_key, current_run_row, actual_result_map):
    return report_bundles.build_llm_weekly_report_bundle(
        scope_key,
        current_run_row,
        actual_result_map,
        load_actual_result_map=_load_actual_result_map,
        load_combined_llm_report_runs=_load_combined_llm_report_runs,
        load_policy_payloads=load_policy_payloads,
        normalize_policy_engine=normalize_policy_engine,
        actual_result_snapshot=_actual_result_snapshot,
        load_policy_run_ticket_rows=load_policy_run_ticket_rows,
        summarize_ticket_rows=_summarize_ticket_rows,
        marks_result_triplet=_marks_result_triplet,
        format_triplet_text=_format_triplet_text,
        percent_text_from_ratio=_percent_text_from_ratio,
        parse_run_date=_parse_run_date,
        build_table_html=build_table_html,
    )


def _jst_today_text():
    return (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")


def _normalize_report_date_text(date_text=""):
    text = _safe_text(date_text)
    if not text:
        return _jst_today_text()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    parsed = _parse_run_date(text)
    if parsed:
        return parsed.strftime("%Y-%m-%d")
    return _jst_today_text()


def _llm_today_scope_keys(scope_key=""):
    scope_norm = normalize_scope_key(scope_key)
    if scope_norm:
        return [scope_norm]
    return list(LLM_REPORT_SCOPE_KEYS)


def _resolve_llm_today_target_date(target_date="", scope_key=""):
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


def _llm_today_status_meta(ticket_summary, actual_names):
    status = _safe_text((ticket_summary or {}).get("status", "")).lower()
    actual_ready = any(_safe_text(name) for name in list(actual_names or []))
    if status == "settled":
        return "已结算", "settled"
    if status == "pending":
        return "待结算", "pending"
    if actual_ready:
        return "已录入结果", "result"
    return "待录入结果", "planned"


def _format_yen_text(value):
    try:
        amount = int(value or 0)
    except (TypeError, ValueError):
        amount = 0
    sign = "-" if amount < 0 else ""
    return f"{sign}{abs(amount):,}円"


def _format_percent_text(value):
    if value in ("", None):
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def build_llm_today_page(date_text="", scope_key=""):
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


def _llm_today_scope_label_ja(scope_key):
    mapping = {
        "central_dirt": "中央ダート",
        "central_turf": "中央芝",
        "local": "地方",
    }
    return mapping.get(str(scope_key or "").strip(), str(scope_key or "").strip() or "-")


def _llm_today_status_meta_ja(ticket_summary, actual_names):
    status = _safe_text((ticket_summary or {}).get("status", "")).lower()
    actual_ready = any(_safe_text(name) for name in list(actual_names or []))
    if status == "settled":
        return "確定", "settled"
    if status == "pending":
        return "発売中", "pending"
    if actual_ready:
        return "結果あり", "result"
    return "公開中", "planned"


def _format_yen_text_ja(value):
    try:
        amount = int(value or 0)
    except (TypeError, ValueError):
        amount = 0
    sign = "-" if amount < 0 else ""
    return f"{sign}¥{abs(amount):,}"


def _format_ticket_plan_text_ja(ticket_rows):
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
        bet_type = bet_type_map.get(_safe_text(row.get("bet_type")).lower(), _safe_text(row.get("bet_type")) or "-")
        horse_no = _safe_text(row.get("horse_no")) or "-"
        amount = _format_yen_text_ja(row.get("amount_yen") or row.get("stake_yen") or 0)
        horse_name = _safe_text(row.get("horse_name"))
        line = f"{bet_type} {horse_no} {amount}"
        if horse_name:
            line += f"  {horse_name}"
        lines.append(line)
    return "\n".join(lines)


def _format_result_triplet_text_ja(actual_names):
    names = [_safe_text(name) for name in list(actual_names or [])[:3] if _safe_text(name)]
    if not names:
        return "結果未登録"
    return " / ".join(f"{idx + 1}着 {name}" for idx, name in enumerate(names))


def build_llm_today_page_clean(date_text="", scope_key=""):
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


def _public_scope_label_ja(scope_key):
    mapping = {
        "central_dirt": "中央ダート",
        "central_turf": "中央芝",
        "local": "地方",
    }
    return mapping.get(str(scope_key or "").strip(), str(scope_key or "").strip() or "-")


def _public_status_meta_ja(ticket_summary, actual_names):
    return web_public_llm.public_status_meta_ja(ticket_summary, actual_names)

def _public_yen_text(value):
    return web_public_llm.public_yen_text(value)

def _public_ticket_plan_text(ticket_rows):
    return web_public_llm.public_ticket_plan_text(ticket_rows)

def _share_hashtag_race_label(run_row):
    venue = _safe_text((run_row or {}).get("location")) or _safe_text((run_row or {}).get("trigger_race"))
    race_no = _race_no_text((run_row or {}).get("race_id")) or ""
    venue = re.sub(r"\s+", "", venue)
    if venue and not venue.endswith("競馬"):
        venue = f"{venue}競馬"
    if venue and race_no:
        return f"#{venue} {race_no}"
    if venue:
        return f"#{venue}"
    if race_no:
        return race_no
    return "#競馬AI"


def _share_ticket_lines(ticket_rows):
    bet_type_map = {
        "win": "単勝",
        "place": "複勝",
        "wide": "ワイド",
        "quinella": "馬連",
        "exacta": "馬単",
        "trio": "三連複",
        "trifecta": "三連単",
    }
    lines = []
    for row in list(ticket_rows or []):
        bet_type = bet_type_map.get(_safe_text(row.get("bet_type")).lower(), _safe_text(row.get("bet_type")) or "-")
        horse_no = _safe_text(row.get("horse_no")) or "-"
        amount = to_int_or_none(row.get("amount_yen"))
        if amount is None:
            amount = to_int_or_none(row.get("stake_yen"))
        amount_text = f"¥{int(amount)}" if amount is not None else "-"
        lines.append(f"{bet_type} {horse_no} {amount_text}")
    return lines


def _share_marks_text(marks_map):
    if not marks_map:
        return "印なし"
    symbol_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}
    ordered = []
    for horse_no, symbol in dict(marks_map or {}).items():
        ordered.append((symbol_order.get(_safe_text(symbol), 99), to_int_or_none(horse_no) or 999, _safe_text(horse_no), _safe_text(symbol)))
    ordered.sort(key=lambda item: (item[0], item[1], item[2]))
    parts = [f"{symbol}{horse_no}" for _, _, horse_no, symbol in ordered if horse_no and symbol]
    return " ".join(parts) if parts else "印なし"


def _build_public_share_text(run_row, engine, marks_map, ticket_rows, max_chars=PUBLIC_SHARE_MAX_CHARS):
    return web_public_llm.build_public_share_text(
        run_row,
        engine,
        marks_map,
        ticket_rows,
        max_chars=max_chars,
        share_detail_label=PUBLIC_SHARE_DETAIL_LABEL,
        share_url=PUBLIC_SHARE_URL,
        share_hashtag=PUBLIC_SHARE_HASHTAG,
        to_int_or_none=to_int_or_none,
    )

def _public_result_triplet_text(actual_names):
    return web_public_llm.public_result_triplet_text(actual_names)

def _public_result_triplet_text_with_nos(actual_names, actual_horse_nos):
    return web_public_llm.public_result_triplet_text_with_nos(actual_names, actual_horse_nos)

def _public_date_label(date_text):
    return web_public_llm.public_date_label(date_text, parse_run_date=_parse_run_date)

LLM_BATTLE_ORDER = REPORT_LLM_BATTLE_ORDER
LLM_BATTLE_LABELS = REPORT_LLM_BATTLE_LABELS
LLM_NOTE_LABELS = REPORT_LLM_NOTE_LABELS
LLM_BATTLE_SHORT_LABELS = REPORT_LLM_BATTLE_SHORT_LABELS
LLM_REPORT_SCOPE_KEYS = REPORT_LLM_REPORT_SCOPE_KEYS
BET_TYPE_TEXT_MAP = REPORT_BET_TYPE_TEXT_MAP
_scope_display_name = report_scope_display_name
_safe_text = report_safe_text
_policy_primary_budget = report_policy_primary_budget
_policy_primary_output = report_policy_primary_output
_policy_marks_map = report_policy_marks_map
_format_ticket_plan_text = report_format_ticket_plan_text
_format_marks_text = report_format_marks_text
_run_date_key = report_run_date_key
_parse_run_date = report_parse_run_date
_format_jp_date_value = report_format_jp_date_value
_has_llm_policy_assets = report_has_llm_policy_assets
_report_scope_key_for_row = report_scope_key_for_row
_payload_run_id = report_payload_run_id
_race_no_text = report_race_no_text
_format_race_label = report_format_race_label
_format_jp_date_text = report_format_jp_date_text
_build_battle_title = report_build_battle_title
_jst_today_text = report_jst_today_text
_normalize_report_date_text = report_normalize_report_date_text
_llm_today_scope_keys = report_llm_today_scope_keys
_format_yen_text = report_format_yen_text
_format_percent_text = report_format_percent_text
_llm_today_scope_label_ja = report_llm_today_scope_label_ja
_public_scope_label_ja = report_public_scope_label_ja


def build_public_board_payload(date_text="", scope_key=""):
    return web_public_llm.build_public_board_payload(
        date_text=date_text,
        scope_key=scope_key,
        normalize_report_date_text=_normalize_report_date_text,
        llm_today_scope_keys=_llm_today_scope_keys,
        resolve_llm_today_target_date=_resolve_llm_today_target_date,
        load_actual_result_map=_load_actual_result_map,
        load_combined_llm_report_runs=_load_combined_llm_report_runs,
        find_job_meta_for_run=_find_job_meta_for_run,
        load_policy_payloads=load_policy_payloads,
        normalize_policy_engine=normalize_policy_engine,
        actual_result_snapshot=_actual_result_snapshot,
        load_policy_run_ticket_rows=load_policy_run_ticket_rows,
        summarize_ticket_rows=_summarize_ticket_rows,
        format_triplet_text=_format_triplet_text,
        marks_result_triplet=_marks_result_triplet,
        format_confidence_text=_format_confidence_text,
        format_distance_label=_format_distance_label,
        public_all_time_roi_summary=_public_all_time_roi_summary,
        public_trend_series=_public_trend_series,
        parse_run_date=_parse_run_date,
    )

def build_public_llm_page(date_text="", scope_key=""):
    payload = build_public_board_payload(date_text=date_text, scope_key=scope_key)
    return web_public_llm.build_public_llm_page(
        payload=payload,
        prefix_public_html_routes=_prefix_public_html_routes,
    )

_JOB_STEP_LABELS = {
    "odds": "赔率",
    "predictor": "预测",
    "policy": "LLM",
    "settlement": "结算",
}
_JOB_STEP_STATE_LABELS = {
    "idle": "未开始",
    "queued": "排队中",
    "running": "进行中",
    "succeeded": "完成",
    "failed": "失败",
}
_JOB_STEP_STATE_TONES = {
    "idle": "muted",
    "queued": "active",
    "running": "active",
    "succeeded": "good",
    "failed": "danger",
}


def _race_job_view(row):
    hydrated = hydrate_race_job_step_states(dict(row or {}))
    display = derive_race_job_display_state(hydrated)
    return hydrated, display


def _race_job_step_badges_html(row):
    hydrated, _ = _race_job_view(row)
    chips = []
    for step_name in ("odds", "predictor", "policy", "settlement"):
        state = str(hydrated.get(f"{step_name}_status", "idle") or "idle").strip().lower() or "idle"
        tone = _JOB_STEP_STATE_TONES.get(state, "muted")
        chips.append(
            f'<span class="hero-pill hero-pill--{html.escape(tone)}">{html.escape(_JOB_STEP_LABELS.get(step_name, step_name))}: {html.escape(_JOB_STEP_STATE_LABELS.get(state, state))}</span>'
        )
    return "".join(chips)


def _race_job_display_tone(row):
    _, display = _race_job_view(row)
    return str(display.get("tone", "muted") or "muted")


def _race_job_display_label(row):
    _, display = _race_job_view(row)
    return str(display.get("label", "-") or "-")


def _race_job_display_code(row):
    _, display = _race_job_view(row)
    return str(display.get("code", "") or "").strip().lower()


def _race_job_process_log_entries(row):
    raw = str((row or {}).get("last_process_output", "") or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    items = list(payload.get("process_log", []) or [])
    entries = []
    for item in items:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step", "") or "").strip()
        if not step:
            continue
        code = str(item.get("code", "") or "").strip()
        output = str(item.get("output", "") or "").strip()
        preview = ""
        if output:
            preview = output.splitlines()[0].strip()
            if len(preview) > 140:
                preview = preview[:137] + "..."
        entries.append({"step": step, "code": code, "preview": preview})
    return entries


def _race_job_process_log_html(row):
    entries = _race_job_process_log_entries(row)
    if not entries:
        return ""
    items = []
    for entry in entries:
        code_text = f"exit {entry['code']}" if entry.get("code", "") != "" else "exit -"
        preview_html = (
            f'<div class="job-process-preview">{html.escape(entry["preview"])}</div>'
            if entry.get("preview")
            else ""
        )
        items.append(
            f"""
            <article class="job-process-item">
              <div class="job-process-head">
                <strong>{html.escape(entry["step"])}</strong>
                <span>{html.escape(code_text)}</span>
              </div>
              {preview_html}
            </article>
            """
        )
    return f"""
    <section class="job-process-log">
      <div class="job-process-title">Process Log</div>
      <div class="job-process-list">
        {"".join(items)}
      </div>
    </section>
    """


def _race_job_action_buttons_v2(job_id, status, admin_token=""):
    buttons = []
    status_text = str(status or "").strip().lower()
    if status_text in ("scheduled", "queued_process", "queued_policy", "failed", "processing", "processing_policy"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/process_now">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">立即处理</button>
            </form>
            """
        )
    if status_text in ("ready", "queued_settle", "settling", "settled", "processing", "waiting_v5"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/process_now">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">重新执行</button>
            </form>
            """
        )
    if status_text in ("failed", "settled", "ready", "queued_settle", "processing", "processing_policy", "settling", "queued_process", "queued_policy", "waiting_v5"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/update">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="action" value="force_reset">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">强制回撤</button>
            </form>
            """
        )
    if status_text in ("queued_process", "queued_policy", "processing", "processing_policy", "queued_settle", "settling", "waiting_v5"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/update">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="action" value="mark_failed">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">标记失败</button>
            </form>
            """
        )
    if status_text in ("failed", "settled", "ready", "queued_settle"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/update">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="action" value="reset_schedule">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">回到排程</button>
            </form>
            """
        )
    buttons.append(
        f"""
        <form method="post" action="/console/tasks/delete">
          <input type="hidden" name="job_id" value="{html.escape(job_id)}">
          <input type="hidden" name="token" value="{html.escape(admin_token)}">
          <button type="submit">删除任务</button>
        </form>
        """
    )
    return "".join(buttons)


def _race_job_status_tone(status):
    text = str(status or "").strip().lower()
    if text in ("ready", "settled"):
        return "good"
    if text in ("queued_process", "processing", "waiting_v5", "queued_policy", "processing_policy", "queued_settle", "settling"):
        return "active"
    if text == "failed":
        return "danger"
    return "muted"


def _race_job_status_label(status):
    mapping = {
        "uploaded": "已上传",
        "scheduled": "待处理",
        "queued_process": "待执行",
        "processing": "处理中",
        "waiting_v5": "等待远程预测",
        "queued_policy": "等待 LLM",
        "processing_policy": "LLM 处理中",
        "ready": "预测已生成",
        "queued_settle": "已入结算队列",
        "settling": "结算中",
        "settled": "已结算",
        "failed": "失败",
    }
    text = str(status or "").strip().lower()
    return mapping.get(text, text or "-")


def _race_job_action_buttons(job_id, status, admin_token=""):
    buttons = []
    status_text = str(status or "").strip().lower()
    if status_text in ("scheduled", "queued_process", "queued_policy", "failed", "processing_policy"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/process_now">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">立即执行处理</button>
            </form>
            """
        )
    if status_text in ("ready", "queued_settle", "settling", "settled", "waiting_v5"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/process_now">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">重新生成预测</button>
            </form>
            """
        )
    if status_text in ("failed", "settled", "ready", "queued_settle"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/update">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="action" value="reset_schedule">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">重置为待命</button>
            </form>
            """
        )
    return "".join(buttons)


def _race_job_settle_form(row, admin_token=""):
    current_run_id = str((row or {}).get("current_run_id", "") or "").strip()
    if not current_run_id:
        return ""
    status_text = str((row or {}).get("status", "") or "").strip().lower()
    if status_text not in ("ready", "queued_settle", "settling", "settled"):
        return ""
    return f"""
    <section class="job-settle-panel">
      <div class="job-settle-head">
        <strong>第三步：录入赛果并结算</strong>
        <span>提交 1-3 着后可直接结算，也可以先保存到结算队列。</span>
      </div>
      <div class="job-settle-actions">
        <form method="post" action="/console/tasks/settle_now" class="job-settle-form">
          <input type="hidden" name="job_id" value="{html.escape(str((row or {}).get('job_id', '') or ''))}">
          <input type="hidden" name="token" value="{html.escape(admin_token)}">
          <input type="text" name="actual_top1" value="{html.escape(str((row or {}).get('actual_top1', '') or ''))}" placeholder="1着马名">
          <input type="text" name="actual_top2" value="{html.escape(str((row or {}).get('actual_top2', '') or ''))}" placeholder="2着马名">
          <input type="text" name="actual_top3" value="{html.escape(str((row or {}).get('actual_top3', '') or ''))}" placeholder="3着马名">
          <button type="submit">立即结算</button>
        </form>
        <form method="post" action="/console/tasks/queue_settle" class="job-settle-form">
          <input type="hidden" name="job_id" value="{html.escape(str((row or {}).get('job_id', '') or ''))}">
          <input type="hidden" name="token" value="{html.escape(admin_token)}">
          <input type="text" name="actual_top1" value="{html.escape(str((row or {}).get('actual_top1', '') or ''))}" placeholder="1着马名">
          <input type="text" name="actual_top2" value="{html.escape(str((row or {}).get('actual_top2', '') or ''))}" placeholder="2着马名">
          <input type="text" name="actual_top3" value="{html.escape(str((row or {}).get('actual_top3', '') or ''))}" placeholder="3着马名">
          <button type="submit">保存并加入结算队列</button>
        </form>
      </div>
    </section>
    """


def _admin_job_card_html(row, admin_token=""):
    row = dict(row or {})
    row, _ = _race_job_view(row)
    job_id = str(row.get("job_id", "") or "").strip()
    status = str(row.get("status", "") or "").strip()
    current_run_id = str(row.get("current_run_id", "") or "").strip()
    scope_key = str(row.get("scope_key", "") or "").strip()
    race_date = str(row.get("race_date", "") or "").strip()
    location = str(row.get("location", "") or "").strip()
    race_id = str(row.get("race_id", "") or "").strip()
    actual_top1 = str(row.get("actual_top1", "") or "").strip()
    actual_top2 = str(row.get("actual_top2", "") or "").strip()
    actual_top3 = str(row.get("actual_top3", "") or "").strip()
    notes = str(row.get("notes", "") or "").strip()
    artifacts = list(row.get("artifacts", []) or [])
    artifact_map = {str(item.get("artifact_type", "")).strip().lower(): dict(item) for item in artifacts}
    kachiuma_name = str((artifact_map.get("kachiuma") or {}).get("original_name", "") or "未上传")
    shutuba_name = str((artifact_map.get("shutuba") or {}).get("original_name", "") or "未上传")
    timing_items = []
    for label, key in (
        ("开赛", "scheduled_off_time"),
        ("筛选时间", "process_after_time"),
        ("已入队", "queued_process_at"),
        ("已出预测", "ready_at"),
        ("已结算", "settled_at"),
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
          <button type="submit" class="secondary-button">打开 Run</button>
        </form>
        <form method="get" action="/llm_today" class="stack-form">
          <input type="hidden" name="scope_key" value="{html.escape(scope_key)}">
          <input type="hidden" name="date" value="{html.escape(race_date)}">
          <button type="submit" class="secondary-button">打开看板</button>
        </form>
        """
    return f"""
    <section class="panel panel--tight">
      <div class="section-title">
        <div>
          <div class="eyebrow">任务</div>
          <h2>{html.escape((location + ' ' + race_id).strip() or job_id or '未命名任务')}</h2>
        </div>
        <span class="section-chip">{html.escape(_race_job_display_label(row))}</span>
      </div>
      <div class="copy-row">
        <span class="hero-pill">范围: {html.escape(_scope_display_name(scope_key))}</span>
        <span class="hero-pill">日期: {html.escape(race_date or '-')}</span>
        <span class="hero-pill">Run: {html.escape(current_run_id or '-')}</span>
        <span class="hero-pill">赛果: {html.escape(' / '.join(x for x in [actual_top1, actual_top2, actual_top3] if x) or '未录入')}</span>
      </div>
      <div class="copy-row">
        <span class="hero-pill">kachiuma: {html.escape(kachiuma_name)}</span>
        <span class="hero-pill">shutuba: {html.escape(shutuba_name)}</span>
        {''.join(timing_items)}
      </div>
      <p class="helper-text">{html.escape(notes or '无备注')}</p>
      <div class="copy-row">
        {_race_job_action_buttons_v2(job_id, status, admin_token=admin_token)}
        {open_links}
      </div>
      {_race_job_process_log_html(row)}
      {_race_job_settle_form(row, admin_token=admin_token)}
    </section>
    """


def build_admin_workspace_html(message_text="", error_text="", admin_token="", authorized=True):
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
            <p class="helper-text">{html.escape(error_text or '管理口令错误。')}</p>
          </section>
        </section>
        """

    jobs = load_race_jobs(BASE_DIR)
    summary = {
        "total": len(jobs),
        "scheduled": 0,
        "processing": 0,
        "ready": 0,
        "settled": 0,
    }
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        if status == "scheduled":
            summary["scheduled"] += 1
        elif status in ("queued_process", "processing", "queued_settle", "settling"):
            summary["processing"] += 1
        elif status == "ready":
            summary["ready"] += 1
        elif status == "settled":
            summary["settled"] += 1

    cards_html = "".join(_admin_job_card_html(job, admin_token=admin_token) for job in jobs)
    if not cards_html:
        cards_html = """
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Tasks</div>
              <h2>还没有任务</h2>
            </div>
            <span class="section-chip">empty</span>
          </div>
          <p class="helper-text">先上传 `kachiuma.csv` 和 `shutuba.csv`，再手动执行预测和结算。</p>
        </section>
        """

    message_panel = ""
    if message_text:
        message_panel = f"""
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Result</div>
              <h2>任务反馈</h2>
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
              <h2>任务异常</h2>
            </div>
            <span class="section-chip">error</span>
          </div>
          <pre>{html.escape(error_text)}</pre>
        </section>
        """

    default_dt = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%dT15:00")
    default_date = _default_job_race_date_text()
    default_date = _default_job_race_date_text()
    return f"""
    <section class="content-cluster" id="admin-zone">
      <div class="cluster-head">
        <div>
          <div class="eyebrow">Manual Admin Flow</div>
          <h2>任务后台</h2>
        </div>
        <p>后台现在按三步流工作：上传两个 CSV，手动执行预测，录入赛果并结算。</p>
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
            <span class="hero-pill">总任务: {summary['total']}</span>
            <span class="hero-pill">待处理: {summary['scheduled']}</span>
            <span class="hero-pill">处理中: {summary['processing']}</span>
            <span class="hero-pill">预测已生成: {summary['ready']}</span>
            <span class="hero-pill">已结算: {summary['settled']}</span>
          </div>
          <div class="copy-row">
            <form method="post" action="/console/tasks/scan_due" class="stack-form">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit" class="secondary-button">按时间筛选任务</button>
            </form>
            <form method="post" action="/console/tasks/run_due_now" class="stack-form">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit" class="secondary-button">执行已入队任务</button>
            </form>
            <a href="/llm_today" class="secondary-button">看 LLM 看板</a>
          </div>
          <p class="helper-text">开赛时间和提前分钟只作为记录与筛选参考，不会自动到点运行。</p>
        </section>
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Bankroll</div>
              <h2>统一追加资金</h2>
            </div>
            <span class="section-chip">budget</span>
          </div>
          <p class="helper-text">按账本日期给四个 LLM 同时追加当日预算，2026年3月17日起为 50000，此前为 10000。</p>
          <div class="copy-row">
            <form method="post" action="/console/tasks/topup_today_all_llm" class="stack-form">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit" class="secondary-button">所有LLM追加当日预算</button>
            </form>
          </div>
        </section>
        {message_panel}
        {error_panel}
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Import</div>
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
            <p class="helper-text">支持 `pipeline/data/...`、`data/...`，或者直接包含 `central_dirt`、`central_turf`、`local`、`_shared` 的 ZIP。</p>
            <button type="submit">导入到 Render Disk</button>
          </form>
        </section>
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Import</div>
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
            <p class="helper-text">支持上传 `pipeline/data/...`、`data/...`，或直接以 `central_dirt`、`central_turf`、`local`、`_shared` 开头的 ZIP。</p>
            <button type="submit">导入到 Render Disk</button>
          </form>
        </section>
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Step 1</div>
              <h2>上传两个 CSV</h2>
            </div>
            <span class="section-chip">upload</span>
          </div>
          <style>
            .admin-upload-grid > div:nth-child(7) {{ display: none; }}
          </style>
          <form class="stack-form" method="post" action="/console/tasks/create" enctype="multipart/form-data">
            <input type="hidden" name="token" value="{html.escape(admin_token)}">
            <div class="field-grid admin-upload-grid">
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
                <label>比赛日期（展示用）</label>
                <input type="date" name="race_date" value="{html.escape(default_date)}">
              </div>
              <div>
                <label>开赛时间（记录用）</label>
                <input type="datetime-local" name="scheduled_off_time" value="{html.escape(default_dt)}">
              </div>
              <div>
                <label>提前多少分钟（手动筛选参考）</label>
                <input type="number" name="lead_minutes" min="0" value="30">
              </div>
              <div>
                <label>本场距离</label>
                <input type="number" name="target_distance" min="100" step="100" value="1600" placeholder="1600">
              </div>
              <div>
                <label>本场马场状态</label>
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
              <textarea name="notes" placeholder="例如：前一天上传，比赛前手动点击执行预测"></textarea>
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


def _race_job_status_label_clean(status):
    return web_admin_jobs.race_job_status_label_clean(status)


def _race_job_action_buttons_clean(job_id, status, admin_token=""):
    buttons = []
    status_text = str(status or "").strip().lower()
    if status_text in ("scheduled", "queued_process", "queued_policy", "failed", "processing_policy"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/process_now">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">立即执行预测</button>
            </form>
            """
        )
    if status_text in ("ready", "queued_settle", "settling", "settled", "waiting_v5"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/process_now">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">重新生成预测</button>
            </form>
            """
        )
    if status_text in ("failed", "settled", "ready", "queued_settle"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/update">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="action" value="reset_schedule">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">重置为待处理</button>
            </form>
            """
        )
    return "".join(buttons)


def _race_job_settle_form_clean(row, admin_token=""):
    return web_admin_jobs.race_job_settle_form_clean(row, admin_token=admin_token)


def _race_job_edit_form_clean(row, admin_token="", default_job_race_date_text=None):
    return web_admin_jobs.race_job_edit_form_clean(
        row,
        admin_token=admin_token,
        default_job_race_date_text=default_job_race_date_text or _default_job_race_date_text,
    )


def _admin_job_card_html_clean(row, admin_token=""):
    return web_admin_jobs.admin_job_card_html_clean(
        row,
        admin_token=admin_token,
        scope_display_name=_scope_display_name,
        race_job_action_buttons_v2=_race_job_action_buttons_v2,
        race_job_status_label_clean=_race_job_status_label_clean,
        race_job_edit_form_clean=_race_job_edit_form_clean,
        race_job_settle_form_clean=_race_job_settle_form_clean,
        default_job_race_date_text=_default_job_race_date_text,
    )


def build_admin_filter_panel(admin_token="", show_settled=False):
    return web_admin_jobs.build_admin_filter_panel(admin_token=admin_token, show_settled=show_settled)


def build_admin_workspace_html_clean(message_text="", error_text="", admin_token="", authorized=True, show_settled=False):
    return web_admin_jobs.build_admin_workspace_html_clean(
        message_text=message_text,
        error_text=error_text,
        admin_token=admin_token,
        authorized=authorized,
        show_settled=show_settled,
        load_race_jobs=lambda: load_race_jobs(BASE_DIR),
        admin_job_card_html_clean=lambda row, admin_token="": _admin_job_card_html_clean(row, admin_token=admin_token),
        default_job_race_date_text=_default_job_race_date_text,
    )


def render_console_page(message_text="", error_text="", admin_token="", show_settled=False):
    return web_admin_console.render_console_page(
        message_text=message_text,
        error_text=error_text,
        admin_token=admin_token,
        show_settled=show_settled,
        render_page=render_page,
        build_import_archive_panel=build_import_archive_panel,
        build_admin_filter_panel=build_admin_filter_panel,
        build_admin_workspace_html_clean=build_admin_workspace_html_clean,
        admin_token_valid=_admin_token_valid,
    )


def _resolve_console_run_state(scope_key="", selected_run_id="", summary_run_id="", output_text=""):
    return web_admin_console.resolve_console_run_state(
        scope_key=scope_key,
        selected_run_id=selected_run_id,
        summary_run_id=summary_run_id,
        output_text=output_text,
        resolve_run=resolve_run,
        parse_run_id=parse_run_id,
        normalize_race_id=normalize_race_id,
    )


def _load_note_workspace(scope_key="", run_id="", run_row=None):
    return web_admin_console.load_note_workspace(
        scope_key=scope_key,
        run_id=run_id,
        run_row=run_row,
        load_bet_engine_v3_cfg_summary=load_bet_engine_v3_cfg_summary,
        load_actual_result_map=_load_actual_result_map,
        load_policy_payloads=load_policy_payloads,
        normalize_policy_engine=normalize_policy_engine,
        load_policy_bankroll_summary=load_policy_bankroll_summary,
        load_policy_run_ticket_rows=load_policy_run_ticket_rows,
        build_llm_battle_bundle=_build_llm_battle_bundle,
        build_llm_daily_report_bundle=_build_llm_daily_report_bundle,
        build_llm_weekly_report_bundle=_build_llm_weekly_report_bundle,
        resolve_predictor_paths=resolve_predictor_paths,
        load_ability_marks_table=load_ability_marks_table,
        load_mark_recommendation_table=load_mark_recommendation_table,
        load_text_file=load_text_file,
        build_mark_note_text=build_mark_note_text,
    )


def build_note_workspace_page(scope_key="", run_id="", admin_token=""):
    return web_admin_console.build_note_workspace_page(
        scope_key=scope_key,
        run_id=run_id,
        admin_token=admin_token,
        resolve_console_run_state=_resolve_console_run_state,
        load_note_workspace=_load_note_workspace,
        prefix_public_html_routes=_prefix_public_html_routes,
    )


def _target_surface_from_scope(scope_key):
    return "turf" if str(scope_key or "").strip() == "central_turf" else "dirt"


def _parse_job_dt_text(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _format_job_dt_text(value):
    if not value:
        return ""
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def _default_job_race_date_text():
    return (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")


def _import_history_zip(base_dir, archive_bytes, overwrite=False):
    return web_admin_tasks.import_history_zip(
        base_dir=base_dir,
        archive_bytes=archive_bytes,
        overwrite=overwrite,
    )


def build_import_archive_panel(admin_token=""):
    return web_admin_jobs.build_import_archive_panel(admin_token=admin_token)


def build_race_jobs_page(message_text="", error_text="", admin_token="", authorized=True):
    return web_admin_jobs.build_race_jobs_page(
        message_text=message_text,
        error_text=error_text,
        admin_token=admin_token,
        authorized=authorized,
        render_console_page=render_console_page,
    )


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url=PUBLIC_BASE_PATH, status_code=307)


@app.get(CONSOLE_BASE_PATH, response_class=HTMLResponse)
@app.get("/console", response_class=HTMLResponse)
def console_index(token: str = "", show_settled: str = ""):
    if _admin_token_enabled() and not _admin_token_valid(token):
        return build_console_gate_page(admin_token=token, error_text="管理口令无效。")
    show_settled_flag = str(show_settled or "").strip() in ("1", "true", "yes", "on")
    return render_console_page(admin_token=token, show_settled=show_settled_flag)


@app.get(f"{CONSOLE_BASE_PATH}/note", response_class=HTMLResponse)
@app.get("/console/note", response_class=HTMLResponse)
def console_note(scope_key: str = "central_dirt", run_id: str = "", token: str = ""):
    if _admin_token_enabled() and not _admin_token_valid(token):
        return build_console_gate_page(admin_token=token, error_text="管理口令无效。")
    return build_note_workspace_page(scope_key=scope_key, run_id=run_id, admin_token=token)


@app.get(PUBLIC_BASE_PATH, response_class=HTMLResponse)
@app.get("/llm_today", response_class=HTMLResponse)
def llm_today(date: str = "", scope_key: str = ""):
    return build_public_index_response()


@app.get("/ads.txt")
def ads_txt():
    return FileResponse(ADS_TXT_PATH, media_type="text/plain; charset=utf-8")


@app.get(f"{PUBLIC_API_BASE_PATH}/board")
@app.get("/api/public/board")
def public_board_api(date: str = "", scope_key: str = ""):
    return JSONResponse(build_public_board_payload(date_text=date, scope_key=scope_key))


@app.post(f"{CONSOLE_BASE_PATH}/tasks/create", response_class=HTMLResponse)
@app.post("/console/tasks/create", response_class=HTMLResponse)
def create_race_job_view(
    token: str = Form(""),
    scope_key: str = Form("central_dirt"),
    race_id: str = Form(""),
    location: str = Form(""),
    race_date: str = Form(""),
    scheduled_off_time: str = Form(""),
    target_distance: str = Form(""),
    target_track_condition: str = Form(""),
    lead_minutes: str = Form("30"),
    notes: str = Form(""),
    kachiuma_file: UploadFile = File(None),
    shutuba_file: UploadFile = File(None),
):
    return web_admin_tasks.create_race_job_response(
        base_dir=BASE_DIR,
        token=token,
        scope_key=scope_key,
        race_id=race_id,
        location=location,
        race_date=race_date,
        scheduled_off_time=scheduled_off_time,
        target_distance=target_distance,
        target_track_condition=target_track_condition,
        lead_minutes=lead_minutes,
        notes=notes,
        kachiuma_file=kachiuma_file,
        shutuba_file=shutuba_file,
        admin_token_valid=_admin_token_valid,
        build_race_jobs_page=build_race_jobs_page,
        normalize_race_id=normalize_race_id,
        normalize_scope_key=normalize_scope_key,
        default_job_race_date_text=_default_job_race_date_text,
        target_surface_from_scope=_target_surface_from_scope,
        create_race_job=create_race_job,
        save_race_job_artifact=save_race_job_artifact,
        update_race_job=update_race_job,
    )

@app.post(f"{CONSOLE_BASE_PATH}/tasks/import_archive", response_class=HTMLResponse)
@app.post("/console/tasks/import_archive", response_class=HTMLResponse)
async def import_history_archive(
    token: str = Form(""),
    overwrite: str = Form(""),
    archive_file: UploadFile = File(None),
):
    return await web_admin_tasks.import_history_archive_response(
        base_dir=BASE_DIR,
        token=token,
        overwrite=overwrite,
        archive_file=archive_file,
        admin_token_valid=_admin_token_valid,
        build_race_jobs_page=build_race_jobs_page,
    )

@app.post(f"{CONSOLE_BASE_PATH}/tasks/scan_due", response_class=HTMLResponse)
@app.post("/console/tasks/scan_due", response_class=HTMLResponse)
def scan_race_jobs_due(token: str = Form("")):
    return web_admin_tasks.scan_due_race_jobs_response(
        base_dir=BASE_DIR,
        token=token,
        admin_token_valid=_admin_token_valid,
        build_race_jobs_page=build_race_jobs_page,
        scan_due_race_jobs=scan_due_race_jobs,
    )

@app.post(f"{CONSOLE_BASE_PATH}/tasks/run_due_now", response_class=HTMLResponse)
@app.post("/console/tasks/run_due_now", response_class=HTMLResponse)
def run_due_race_jobs_now(token: str = Form("")):
    return web_admin_tasks.run_due_race_jobs_now_response(
        base_dir=BASE_DIR,
        token=token,
        admin_token_valid=_admin_token_valid,
        build_race_jobs_page=build_race_jobs_page,
        scan_due_race_jobs=scan_due_race_jobs,
        load_race_jobs=load_race_jobs,
    )

@app.get("/internal/run_due")
@app.post("/internal/run_due")
def internal_run_due(token: str = ""):
    return web_admin_tasks.internal_run_due_response(
        base_dir=BASE_DIR,
        token=token,
        admin_token_valid=_admin_token_valid,
        scan_due_race_jobs=scan_due_race_jobs,
        load_race_jobs=load_race_jobs,
    )

@app.post(f"{CONSOLE_BASE_PATH}/tasks/topup_today_all_llm", response_class=HTMLResponse)
@app.post("/console/tasks/topup_today_all_llm", response_class=HTMLResponse)
def topup_today_all_llm_budget(token: str = Form("")):
    return web_admin_tasks.topup_today_all_llm_budget_response(
        base_dir=BASE_DIR,
        token=token,
        admin_token_valid=_admin_token_valid,
        render_console_page=render_console_page,
        default_job_race_date_text=_default_job_race_date_text,
        resolve_daily_bankroll_yen=resolve_daily_bankroll_yen,
        add_bankroll_topup=add_bankroll_topup,
    )


@app.post("/console/tasks/edit", response_class=HTMLResponse)
def edit_race_job_details(
    token: str = Form(""),
    job_id: str = Form(""),
    race_id: str = Form(""),
    location: str = Form(""),
    race_date: str = Form(""),
    scheduled_off_time: str = Form(""),
    target_distance: str = Form(""),
    target_track_condition: str = Form(""),
    lead_minutes: str = Form("30"),
    notes: str = Form(""),
):
    return web_admin_ops.edit_race_job_details_response(
        base_dir=BASE_DIR,
        token=token,
        job_id=job_id,
        race_id=race_id,
        location=location,
        race_date=race_date,
        scheduled_off_time=scheduled_off_time,
        target_distance=target_distance,
        target_track_condition=target_track_condition,
        lead_minutes=lead_minutes,
        notes=notes,
        admin_token_valid=_admin_token_valid,
        build_race_jobs_page=build_race_jobs_page,
        normalize_race_id=normalize_race_id,
        default_job_race_date_text=_default_job_race_date_text,
        load_race_jobs=load_race_jobs,
        render_console_page=render_console_page,
        target_surface_from_scope=_target_surface_from_scope,
        parse_job_dt_text=_parse_job_dt_text,
        format_job_dt_text=_format_job_dt_text,
        update_race_job=update_race_job,
        compute_race_job_initial_status=compute_race_job_initial_status,
    )

@app.post(f"{CONSOLE_BASE_PATH}/tasks/update", response_class=HTMLResponse)
@app.post("/console/tasks/update", response_class=HTMLResponse)
def update_race_job_view(
    token: str = Form(""),
    job_id: str = Form(""),
    action: str = Form(""),
):
    return web_admin_tasks.update_race_job_response(
        base_dir=BASE_DIR,
        token=token,
        job_id=job_id,
        action=action,
        admin_token_valid=_admin_token_valid,
        build_race_jobs_page=build_race_jobs_page,
        apply_race_job_action=apply_race_job_action,
    )

@app.post(f"{CONSOLE_BASE_PATH}/tasks/delete", response_class=HTMLResponse)
@app.post("/console/tasks/delete", response_class=HTMLResponse)
def delete_race_job_view(
    token: str = Form(""),
    job_id: str = Form(""),
):
    return web_admin_tasks.delete_race_job_response(
        base_dir=BASE_DIR,
        token=token,
        job_id=job_id,
        admin_token_valid=_admin_token_valid,
        build_race_jobs_page=build_race_jobs_page,
        delete_race_job=delete_race_job,
    )

@app.post(f"{CONSOLE_BASE_PATH}/tasks/process_now", response_class=HTMLResponse)
@app.post("/console/tasks/process_now", response_class=HTMLResponse)
def process_race_job_now(
    token: str = Form(""),
    job_id: str = Form(""),
):
    return web_admin_tasks.process_race_job_now_response(
        base_dir=BASE_DIR,
        token=token,
        job_id=job_id,
        admin_token_valid=_admin_token_valid,
        build_race_jobs_page=build_race_jobs_page,
    )

@app.post(f"{CONSOLE_BASE_PATH}/tasks/queue_settle", response_class=HTMLResponse)
@app.post("/console/tasks/queue_settle", response_class=HTMLResponse)
def queue_race_job_settle(
    token: str = Form(""),
    job_id: str = Form(""),
    actual_top1: str = Form(""),
    actual_top2: str = Form(""),
    actual_top3: str = Form(""),
):
    return web_admin_tasks.queue_race_job_settle_response(
        base_dir=BASE_DIR,
        token=token,
        job_id=job_id,
        actual_top1=actual_top1,
        actual_top2=actual_top2,
        actual_top3=actual_top3,
        admin_token_valid=_admin_token_valid,
        build_race_jobs_page=build_race_jobs_page,
        initialize_race_job_step_fields=initialize_race_job_step_fields,
        set_race_job_step_state=set_race_job_step_state,
        update_race_job=update_race_job,
    )

@app.post(f"{CONSOLE_BASE_PATH}/tasks/settle_now", response_class=HTMLResponse)
@app.post("/console/tasks/settle_now", response_class=HTMLResponse)
def settle_race_job_now(
    token: str = Form(""),
    job_id: str = Form(""),
    actual_top1: str = Form(""),
    actual_top2: str = Form(""),
    actual_top3: str = Form(""),
):
    return web_admin_tasks.settle_race_job_now_response(
        base_dir=BASE_DIR,
        token=token,
        job_id=job_id,
        actual_top1=actual_top1,
        actual_top2=actual_top2,
        actual_top3=actual_top3,
        admin_token_valid=_admin_token_valid,
        render_console_page=render_console_page,
        initialize_race_job_step_fields=initialize_race_job_step_fields,
        set_race_job_step_state=set_race_job_step_state,
        update_race_job=update_race_job,
    )

@app.post("/view_run", response_class=HTMLResponse)
def view_run(
    run_id: str = Form(""),
    scope_key: str = Form(""),
    token: str = Form(""),
):
    return web_admin_ops.view_run_response(
        run_id=run_id,
        scope_key=scope_key,
        token=token,
        normalize_scope_key=normalize_scope_key,
        infer_scope_and_run=infer_scope_and_run,
        render_page=render_page,
        resolve_run=resolve_run,
        normalize_race_id=normalize_race_id,
        resolve_latest_run_by_race_id=resolve_latest_run_by_race_id,
        infer_run_id_from_row=infer_run_id_from_row,
        update_run_row_fields=update_run_row_fields,
    )

@app.get("/api/runs")
def api_runs(scope_key: str = "central_dirt", limit: int = DEFAULT_RUN_LIMIT, q: str = ""):
    return web_admin_ops.api_runs_response(
        scope_key=scope_key,
        limit=limit,
        query=q,
        normalize_scope_key=normalize_scope_key,
        default_limit=DEFAULT_RUN_LIMIT,
        max_limit=MAX_RUN_LIMIT,
        get_recent_runs=get_recent_runs,
    )

@app.post("/run_pipeline", response_class=HTMLResponse)
def run_pipeline(
    token: str = Form(""),
    race_id: str = Form(""),
    race_url: str = Form(""),
    history_url: str = Form(""),
    scope_key: str = Form(""),
    location: str = Form(""),
    race_date: str = Form(""),
    surface: str = Form(""),
    distance: str = Form(""),
    track_cond: str = Form(""),
):
    return web_admin_ops.run_pipeline_response(
        token=token,
        race_id=race_id,
        race_url=race_url,
        history_url=history_url,
        scope_key=scope_key,
        location=location,
        race_date=race_date,
        surface=surface,
        distance=distance,
        track_cond=track_cond,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        normalize_scope_key=normalize_scope_key,
        render_page=render_page,
        normalize_race_id=normalize_race_id,
        run_script=run_script,
        run_pipeline_path=RUN_PIPELINE,
        extract_top5=extract_top5,
        parse_run_id=parse_run_id,
    )

@app.post("/record_predictor", response_class=HTMLResponse)
def record_predictor(
    token: str = Form(""),
    scope_key: str = Form(""),
    run_id: str = Form(""),
    top1: str = Form(""),
    top2: str = Form(""),
    top3: str = Form(""),
):
    return web_admin_ops.record_predictor_response(
        token=token,
        scope_key=scope_key,
        run_id=run_id,
        top1=top1,
        top2=top2,
        top3=top3,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        normalize_scope_key=normalize_scope_key,
        infer_scope_and_run=infer_scope_and_run,
        resolve_run=resolve_run,
        normalize_race_id=normalize_race_id,
        resolve_latest_run_by_race_id=resolve_latest_run_by_race_id,
        render_page=render_page,
        resolve_run_asset_path=resolve_run_asset_path,
        refresh_odds_for_run=refresh_odds_for_run,
        run_script=run_script,
        record_predictor_path=RECORD_PREDICTOR,
    )

@app.post("/run_llm_buy", response_class=HTMLResponse)
def run_llm_buy(
    token: str = Form(""),
    scope_key: str = Form(""),
    run_id: str = Form(""),
    policy_engine: str = Form("gemini"),
    policy_model: str = Form(""),
    refresh_odds: str = Form("1"),
):
    return web_admin_ops.run_llm_buy_response(
        token=token,
        scope_key=scope_key,
        run_id=run_id,
        policy_engine=policy_engine,
        policy_model=policy_model,
        refresh_odds=refresh_odds,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        resolve_run_selection=resolve_run_selection,
        render_page=render_page,
        maybe_refresh_run_odds=maybe_refresh_run_odds,
        build_llm_buy_output=build_llm_buy_output,
        load_policy_bankroll_summary=load_policy_bankroll_summary,
        normalize_policy_engine=normalize_policy_engine,
        execute_policy_buy=execute_policy_buy,
    )

@app.post("/run_gemini_buy", response_class=HTMLResponse)
def run_gemini_buy(
    token: str = Form(""),
    scope_key: str = Form(""),
    run_id: str = Form(""),
    policy_model: str = Form(""),
    refresh_odds: str = Form("1"),
):
    return run_llm_buy(
        token=token,
        scope_key=scope_key,
        run_id=run_id,
        policy_engine="gemini",
        policy_model=policy_model,
        refresh_odds=refresh_odds,
    )


@app.post("/run_all_llm_buy", response_class=HTMLResponse)
def run_all_llm_buy(
    token: str = Form(""),
    scope_key: str = Form(""),
    run_id: str = Form(""),
    refresh_odds: str = Form("1"),
):
    return web_admin_ops.run_all_llm_buy_response(
        token=token,
        scope_key=scope_key,
        run_id=run_id,
        refresh_odds=refresh_odds,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        resolve_run_selection=resolve_run_selection,
        render_page=render_page,
        maybe_refresh_run_odds=maybe_refresh_run_odds,
        build_llm_buy_output=build_llm_buy_output,
        load_policy_bankroll_summary=load_policy_bankroll_summary,
        execute_policy_buy=execute_policy_buy,
    )

@app.post("/topup_all_llm_budget", response_class=HTMLResponse)
def topup_all_llm_budget(
    token: str = Form(""),
    scope_key: str = Form(""),
    run_id: str = Form(""),
):
    return web_admin_ops.topup_all_llm_budget_response(
        base_dir=BASE_DIR,
        token=token,
        scope_key=scope_key,
        run_id=run_id,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        resolve_run_selection=resolve_run_selection,
        render_page=render_page,
        extract_ledger_date=extract_ledger_date,
        resolve_daily_bankroll_yen=resolve_daily_bankroll_yen,
        add_bankroll_topup=add_bankroll_topup,
    )

@app.post("/reset_llm_state", response_class=HTMLResponse)
def reset_llm_state(token: str = Form("")):
    return web_admin_ops.reset_llm_state_response(
        base_dir=BASE_DIR,
        token=token,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        render_page=render_page,
        reset_llm_state_files=reset_llm_state_files,
    )

@app.post("/optimize_params", response_class=HTMLResponse)
def optimize_params(token: str = Form("")):
    return web_admin_ops.optimize_params_response(
        token=token,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        run_script=run_script,
        optimize_params_path=OPTIMIZE_PARAMS,
        render_page=render_page,
    )

@app.post("/optimize_predictor", response_class=HTMLResponse)
def optimize_predictor(token: str = Form("")):
    return web_admin_ops.optimize_predictor_response(
        token=token,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        run_script=run_script,
        optimize_predictor_path=OPTIMIZE_PREDICTOR,
        render_page=render_page,
    )

@app.post("/offline_eval", response_class=HTMLResponse)
def offline_eval(token: str = Form(""), window: str = Form("")):
    return web_admin_ops.offline_eval_response(
        token=token,
        window=window,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        run_script=run_script,
        offline_eval_path=OFFLINE_EVAL,
        render_page=render_page,
    )

@app.post("/init_update", response_class=HTMLResponse)
def init_update(token: str = Form("")):
    return web_admin_ops.init_update_response(
        token=token,
        reset=False,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        run_script=run_script,
        init_update_path=INIT_UPDATE,
        render_page=render_page,
    )

@app.post("/init_update_reset", response_class=HTMLResponse)
def init_update_reset(token: str = Form("")):
    return web_admin_ops.init_update_response(
        token=token,
        reset=True,
        admin_token_valid=_admin_token_valid,
        admin_execution_denied=_admin_execution_denied,
        run_script=run_script,
        init_update_path=INIT_UPDATE,
        render_page=render_page,
    )

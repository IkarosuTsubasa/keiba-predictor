import csv
import asyncio
import html
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
import base64
from io import BytesIO
from datetime import datetime, timedelta, timezone
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

from fetch_central_result import (
    build_result_url as build_official_result_url,
    fetch_html as fetch_official_result_html,
    parse_result_page as parse_official_result_page,
)
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
from web_admin import operations as web_admin_ops
from web_admin import remote_predictors as web_remote_predictors
from web_admin import task_routes as web_admin_tasks
from web_data import odds_service, run_resolver, run_store, summary_service, view_data
from web_helpers import (
    extract_section,
    format_path_mtime,
    get_env_timeout,
    is_run_id,
    load_csv_rows,
    load_csv_rows_flexible,
    load_json_file,
    load_text_file,
    normalize_race_id,
    parse_horse_no,
    run_script,
    to_float,
    to_int_or_none,
)
from web_public import (
    CONSOLE_BASE_PATH,
    PUBLIC_API_BASE_PATH,
    PUBLIC_BASE_PATH,
    PUBLIC_SITE_URL,
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
    normalize_report_date_text as report_normalize_report_date_text,
    parse_run_date as report_parse_run_date,
    payload_run_id as report_payload_run_id,
    policy_marks_map as report_policy_marks_map,
    policy_primary_budget as report_policy_primary_budget,
    policy_primary_output as report_policy_primary_output,
    race_no_text as report_race_no_text,
    report_scope_key_for_row as report_scope_key_for_row,
    run_date_key as report_run_date_key,
    safe_text as report_safe_text,
    scope_display_name as report_scope_display_name,
)


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ADS_TXT_PATH = BASE_DIR / "ads.txt"
ODDS_EXTRACT = ROOT_DIR / "odds_extract.py"
RECORD_PREDICTOR = BASE_DIR / "record_predictor_result.py"
DEFAULT_RUN_LIMIT = 200
MAX_RUN_LIMIT = 500
app = FastAPI()
load_local_env(BASE_DIR, override=False)
mount_public_assets(app)
register_public_static_routes(app)


def run_due_jobs_once():
    return web_admin_tasks.run_due_jobs_once(
        base_dir=BASE_DIR,
        scan_due_race_jobs=scan_due_race_jobs,
        load_race_jobs=load_race_jobs,
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
                rows, _ = run_store.normalize_csv_rows(rows, fieldnames)
                return rows
        except UnicodeDecodeError:
            continue
    return []

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
        lambda resolved_scope_key: run_store.load_runs_with_header(migrate_legacy_data, get_data_dir, BASE_DIR, resolved_scope_key),
        normalize_race_id,
        scope_key,
        run_row,
        updates,
    )


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
def load_policy_daily_profit_summary(days=30, policy_engine="gemini"):
    return web_policy_runtime.load_policy_daily_profit_summary(
        base_dir=BASE_DIR,
        days=days,
        policy_engine=policy_engine,
        load_daily_profit_rows=load_daily_profit_rows,
    )


def load_policy_run_ticket_rows(run_id, policy_engine="gemini"):
    return web_policy_runtime.load_policy_run_ticket_rows(
        base_dir=BASE_DIR,
        run_id=run_id,
        policy_engine=policy_engine,
        load_run_tickets=load_run_tickets,
    )
def build_llm_buy_output(summary_before, refresh_ok, refresh_message, refresh_warnings, script_output, policy_engine=""):
    return web_policy_runtime.build_llm_buy_output(
        summary_before,
        refresh_ok,
        refresh_message,
        refresh_warnings,
        script_output,
        policy_engine,
    )



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
        "central_turf": "中央芝",
        "central_dirt": "中央ダート",
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
                "notes": "policy_pool=shared;decision={decision};reasons={reasons}".format(
                    decision=str(output_dict.get("bet_decision", "") or ""),
                    reasons=",".join(str(x) for x in list(output_dict.get("reason_codes", []) or [])),
                ),
                "policy_engine": policy_engine,
                "policy_bet_decision": str(output_dict.get("bet_decision", "") or ""),
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


def build_public_share_text(run_row, engine, marks_map, ticket_rows, max_chars=PUBLIC_SHARE_MAX_CHARS):
    return _build_public_share_text(
        run_row,
        engine,
        marks_map,
        ticket_rows,
        max_chars=max_chars,
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


def _load_combined_llm_report_runs():
    return report_data.load_combined_llm_report_runs(load_runs, LLM_REPORT_SCOPE_KEYS)


def _load_actual_result_map(scope_key):
    return report_data.load_actual_result_map(
        BASE_DIR,
        scope_key,
        get_data_dir=get_data_dir,
        load_csv_rows=load_csv_rows,
        canonical_predictor_id=canonical_predictor_id,
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


def _official_result_source_for_scope(scope_key):
    return "local" if str(scope_key or "").strip() == "local" else "central"


def _fetch_official_result_payload(scope_key, race_id):
    race_id_text = normalize_race_id(race_id)
    if not race_id_text:
        raise ValueError("race_id required")
    result_url = build_official_result_url(
        race_id=race_id_text,
        source=_official_result_source_for_scope(scope_key),
    )
    html_bytes = fetch_official_result_html(result_url, timeout=30)
    payload = parse_official_result_page(html_bytes, source_url=result_url)
    if not payload.get("result_available"):
        raise RuntimeError(
            f"official result not available: race_id={race_id_text}"
            + (f" title={str(payload.get('page_title', '') or '').strip()}" if payload.get("page_title") else "")
        )
    return payload


def _fetch_official_top3_names(scope_key, race_id):
    payload = _fetch_official_result_payload(scope_key, race_id)
    names = [str(item.get("horse_name", "") or "").strip() for item in list(payload.get("top3", []) or [])[:3]]
    if len(names) < 3 or not all(names[:3]):
        raise RuntimeError("official top3 not available")
    return names[:3], payload


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


def _summarize_ticket_rows(rows):
    rows = list(rows or [])
    ticket_count = len(rows)
    stake_yen = sum(to_int_or_none(row.get("amount_yen")) or to_int_or_none(row.get("stake_yen")) or 0 for row in rows)
    payout_yen = sum(to_int_or_none(row.get("payout_yen")) or to_int_or_none(row.get("est_payout_yen")) or 0 for row in rows)
    profit_yen = sum(to_int_or_none(row.get("profit_yen")) or 0 for row in rows)
    hit_count = sum(1 for row in rows if (to_int_or_none(row.get("payout_yen")) or to_int_or_none(row.get("est_payout_yen")) or 0) > 0)
    statuses = {str(row.get("status", "") or "").strip().lower() for row in rows if str(row.get("status", "") or "").strip()}
    status = "pending"
    if statuses:
        if statuses == {"settled"}:
            status = "settled"
        elif "pending" in statuses:
            status = "pending"
        else:
            status = sorted(statuses)[0]
    roi = round(float(payout_yen) / float(stake_yen), 4) if stake_yen > 0 else ""
    return {
        "status": status,
        "ticket_count": ticket_count,
        "stake_yen": stake_yen,
        "payout_yen": payout_yen,
        "profit_yen": profit_yen,
        "hit_count": hit_count,
        "roi": roi,
    }


def _format_triplet_text(values):
    parts = [str(value or "").strip() for value in list(values or []) if str(value or "").strip()]
    return " / ".join(parts) if parts else "-"


def _marks_result_triplet(marks_map, actual_horse_nos):
    result = []
    for horse_no in list(actual_horse_nos or [])[:3]:
        text_no = str(horse_no or "").strip()
        if not text_no:
            continue
        symbol = str((marks_map or {}).get(text_no, "") or "").strip()
        result.append(f"{symbol}{text_no}" if symbol else text_no)
    return result


def _format_confidence_text(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _format_distance_label(value):
    text = str(value or "").strip()
    if not text:
        return "-"
    if text.endswith("m"):
        return text
    return f"{text}m"


def _public_all_time_roi_summary():
    return report_bundles.public_all_time_roi_summary(
        BASE_DIR,
        ledger_path=ledger_path,
        load_rows=load_rows,
    )


def _public_trend_series(days=10):
    del days
    return []

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
            return fallback_date, f"{target_date} は利用できないため、最新日付 {fallback_date} に切り替えました", scoped_rows
        return fallback_date, "", scoped_rows
    return target_date, "", scoped_rows


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
        share_detail_label=PUBLIC_SHARE_DETAIL_LABEL,
        share_url=PUBLIC_SHARE_URL,
        share_hashtag=PUBLIC_SHARE_HASHTAG,
        share_max_chars=PUBLIC_SHARE_MAX_CHARS,
        to_int_or_none=to_int_or_none,
    )


_DAILY_SUMMARY_BET_TYPE_LABELS = {
    "win": "単勝",
    "place": "複勝",
    "wide": "ワイド",
    "quinella": "馬連",
    "exacta": "馬単",
    "trio": "3連複",
    "trifecta": "3連単",
}


def _daily_summary_jst_date_text():
    return (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")


def _daily_summary_intent_url(text):
    from urllib.parse import quote

    return f"https://twitter.com/intent/tweet?text={quote(str(text or ''), safe='')}"


def _daily_summary_roi_value(item):
    stake_yen = int(item.get("stake_yen", 0) or 0)
    payout_yen = int(item.get("payout_yen", 0) or 0)
    return (float(payout_yen) / float(stake_yen)) if stake_yen > 0 else -1.0


def _daily_summary_ranked_cards(summary_cards):
    rows = [dict(item or {}) for item in list(summary_cards or []) if int((item or {}).get("races", 0) or 0) > 0]
    rows.sort(
        key=lambda item: (
            -_daily_summary_roi_value(item),
            -int(item.get("profit_yen", 0) or 0),
            -int(item.get("hit_races", 0) or 0),
            str(item.get("engine", "") or ""),
        )
    )
    return rows


def _daily_summary_parse_horse_nos(value):
    text = str(value or "").strip()
    if not text:
        return []
    out = []
    for part in re.split(r"[^0-9.]+", text):
        part = str(part or "").strip()
        if not part:
            continue
        normalized = normalize_horse_no_text(part)
        if normalized:
            out.append(normalized)
    return out


def _daily_summary_best_ticket(target_date):
    target_date = str(target_date or "").strip()
    if not target_date:
        return {}
    best = {}
    for run_row in _load_combined_llm_report_runs():
        if _run_date_key(run_row) != target_date:
            continue
        run_id = _safe_text((run_row or {}).get("run_id"))
        report_scope_key = _report_scope_key_for_row(run_row, "")
        if not run_id or report_scope_key not in LLM_REPORT_SCOPE_KEYS:
            continue
        for payload in load_policy_payloads(report_scope_key, run_id, run_row):
            engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
            if not engine:
                continue
            marks_map = report_policy_marks_map(payload)
            ticket_run_id = report_payload_run_id(payload, run_id)
            ticket_rows = load_policy_run_ticket_rows(ticket_run_id, policy_engine=engine) or list(
                report_policy_primary_budget(payload).get("tickets", []) or []
            )
            for row in list(ticket_rows or []):
                payout_yen = to_int_or_none(row.get("payout_yen")) or 0
                stake_yen = to_int_or_none(row.get("stake_yen")) or to_int_or_none(row.get("amount_yen")) or 0
                hit = to_int_or_none(row.get("hit")) or 0
                status = str(row.get("status", "") or "").strip().lower()
                if payout_yen <= 0:
                    continue
                if status and status != "settled" and hit <= 0:
                    continue
                if stake_yen <= 0:
                    continue
                horse_nos = _daily_summary_parse_horse_nos(row.get("horse_nos", "")) or _daily_summary_parse_horse_nos(row.get("horse_no", ""))
                if not horse_nos:
                    continue
                candidate = {
                    "engine": engine,
                    "race_title": _format_race_label(run_row),
                    "bet_type": str(row.get("bet_type", "") or "").strip().lower(),
                    "horse_nos": horse_nos,
                    "stake_yen": stake_yen,
                    "payout_yen": payout_yen,
                    "return_ratio": float(payout_yen) / float(stake_yen),
                    "marks_map": dict(marks_map or {}),
                }
                if not best:
                    best = candidate
                    continue
                if candidate["return_ratio"] > float(best.get("return_ratio", 0.0) or 0.0):
                    best = candidate
                    continue
                if candidate["return_ratio"] == float(best.get("return_ratio", 0.0) or 0.0) and candidate["payout_yen"] > int(best.get("payout_yen", 0) or 0):
                    best = candidate
    return best


def _daily_summary_best_ticket_lines(best_ticket):
    item = dict(best_ticket or {})
    if not item:
        return []
    race_title = str(item.get("race_title", "") or "").strip() or "-"
    horse_nos = [str(x or "").strip() for x in list(item.get("horse_nos", []) or []) if str(x or "").strip()]
    marks_map = dict(item.get("marks_map", {}) or {})
    head_no = horse_nos[0] if horse_nos else ""
    head_mark = str(marks_map.get(head_no, "") or "").strip() if head_no else ""
    horse_text = "-".join(horse_nos) if len(horse_nos) > 1 else (f"{head_mark}{head_no}" if head_mark and head_no else head_no)
    bet_label = _DAILY_SUMMARY_BET_TYPE_LABELS.get(str(item.get("bet_type", "") or "").strip().lower(), str(item.get("bet_type", "") or "").strip() or "的中")
    payout_text = f"{int(item.get('payout_yen', 0) or 0):,}円"
    if not horse_text:
        horse_text = "的中"
    return ["🔥 今日の神", race_title, f"{horse_text} → {bet_label}{payout_text}"]


def _build_daily_summary_text(*, target_date, ranked_cards, best_ticket, max_chars=140):
    rows = list(ranked_cards or [])
    if not rows:
        raise ValueError("daily summary cards not available")
    best = dict(rows[0] or {})
    best_label = str(best.get("label", "") or best.get("engine", "") or "ベストモデル").strip()
    settled_races = int(best.get("settled_races", 0) or 0)
    race_count = settled_races if settled_races > 0 else int(best.get("races", 0) or 0)
    hit_races = int(best.get("hit_races", 0) or 0)
    roi_pct = int(round(_daily_summary_roi_value(best) * 100.0)) if _daily_summary_roi_value(best) >= 0 else 0
    profit_yen = int(best.get("profit_yen", 0) or 0)
    profit_prefix = "+" if profit_yen > 0 else ""

    def compose(model_limit):
        medals = ["🏅", "🥈", "🥉"]
        model_lines = ["🤖 モデル別成績"]
        for index, item in enumerate(rows[:model_limit]):
            icon = medals[index] if index < len(medals) else ""
            roi_text = f"{int(round(_daily_summary_roi_value(item) * 100.0))}%" if _daily_summary_roi_value(item) >= 0 else "-"
            model_lines.append(f"{icon}{str(item.get('label', '') or item.get('engine', '') or '-').strip()}：{roi_text}")

        lines = [
            "【いかいもAI競馬 本日結果】",
            "",
            f"🎯 的中：{hit_races} / {race_count}レース（{best_label}）",
            "",
            f"💰 回収率：{roi_pct}%（{best_label}）",
            f"{profit_prefix}{profit_yen:,}円",
            "",
        ]
        hero_lines = _daily_summary_best_ticket_lines(best_ticket)
        if hero_lines:
            lines.extend(hero_lines)
            lines.append("")
        lines.extend(model_lines)
        lines.extend(["", "👉 全モデル・全買い目は無料公開中", PUBLIC_SHARE_URL])
        return "\n".join(lines)

    text = compose(4 if len(rows) >= 4 else len(rows))
    if len(text) <= int(max_chars):
        return text
    if len(rows) >= 4:
        text = compose(3)
        if len(text) <= int(max_chars):
            return text
    return text


def build_daily_summary_share_payload(date_text=""):
    requested_date = _normalize_report_date_text(date_text) or _daily_summary_jst_date_text()
    payload = build_public_board_payload(date_text=requested_date, scope_key="")
    target_date = str(payload.get("target_date", "") or "").strip()
    if not target_date:
        raise ValueError("daily target date not available")
    ranked_cards = _daily_summary_ranked_cards(payload.get("summary_cards", []))
    if not ranked_cards:
        raise ValueError("daily summary not available")
    summary_text = _build_daily_summary_text(
        target_date=target_date,
        ranked_cards=ranked_cards,
        best_ticket=_daily_summary_best_ticket(target_date),
        max_chars=140,
    )
    return {
        "requested_date": requested_date,
        "target_date": target_date,
        "target_date_label": str(payload.get("target_date_label", "") or "").strip(),
        "fallback_notice": str(payload.get("fallback_notice", "") or "").strip(),
        "summary_text": summary_text,
        "intent_url": _daily_summary_intent_url(summary_text),
        "best_engine": str((ranked_cards[0] or {}).get("engine", "") or "").strip(),
        "model_rankings": [
            {
                "engine": str(item.get("engine", "") or "").strip(),
                "label": str(item.get("label", "") or item.get("engine", "") or "").strip(),
                "roi_percent": int(round(_daily_summary_roi_value(item) * 100.0)) if _daily_summary_roi_value(item) >= 0 else None,
                "profit_yen": int(item.get("profit_yen", 0) or 0),
                "hit_races": int(item.get("hit_races", 0) or 0),
                "race_count": int(item.get("settled_races", 0) or item.get("races", 0) or 0),
            }
            for item in ranked_cards
        ],
    }

_JOB_STEP_LABELS = {
    "odds": "オッズ",
    "predictor": "予測",
    "policy": "LLM",
    "settlement": "精算",
}
_JOB_STEP_STATE_LABELS = {
    "idle": "未開始",
    "queued": "待機中",
    "running": "実行中",
    "succeeded": "成功",
    "failed": "失敗",
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


def _admin_supplied_token(request: Request, token: str = ""):
    auth_header = str(request.headers.get("authorization", "") or "").strip()
    if auth_header.lower().startswith("bearer "):
        bearer_token = auth_header[7:].strip()
        if bearer_token:
            return bearer_token
    return str(token or "").strip()


def _build_admin_jobs_payload(token: str = "", show_settled: bool = False):
    jobs = load_race_jobs(BASE_DIR)
    summary = {
        "total": len(jobs),
        "uploaded": 0,
        "scheduled": 0,
        "processing": 0,
        "ready": 0,
        "settled": 0,
        "failed": 0,
    }
    items = []

    for row in jobs:
        hydrated, display = _race_job_view(row)
        status = str(hydrated.get("status", "") or "").strip().lower()
        if status == "uploaded":
            summary["uploaded"] += 1
        elif status == "scheduled":
            summary["scheduled"] += 1
        elif status in ("queued_process", "processing", "waiting_v5", "queued_policy", "processing_policy", "queued_settle", "settling"):
            summary["processing"] += 1
        elif status == "ready":
            summary["ready"] += 1
        elif status == "settled":
            summary["settled"] += 1
        elif status == "failed":
            summary["failed"] += 1

        if not show_settled and status == "settled":
            continue

        step_badges = []
        for step_name in ("odds", "predictor", "policy", "settlement"):
            state = str(hydrated.get(f"{step_name}_status", "idle") or "idle").strip().lower() or "idle"
            step_badges.append(
                {
                    "step": step_name,
                    "label": _JOB_STEP_LABELS.get(step_name, step_name),
                    "state": state,
                    "state_label": _JOB_STEP_STATE_LABELS.get(state, state),
                    "tone": _JOB_STEP_STATE_TONES.get(state, "muted"),
                }
            )

        items.append(
            {
                "job_id": str(hydrated.get("job_id", "") or "").strip(),
                "status": status,
                "status_label": str(display.get("label", "-") or "-"),
                "status_tone": str(display.get("tone", "muted") or "muted"),
                "scope_key": str(hydrated.get("scope_key", "") or "").strip(),
                "scope_label": _scope_display_name(hydrated.get("scope_key", "")),
                "race_id": str(hydrated.get("race_id", "") or "").strip(),
                "race_date": str(hydrated.get("race_date", "") or "").strip(),
                "location": str(hydrated.get("location", "") or "").strip(),
                "scheduled_off_time": str(hydrated.get("scheduled_off_time", "") or "").strip(),
                "process_after_time": str(hydrated.get("process_after_time", "") or "").strip(),
                "current_run_id": str(hydrated.get("current_run_id", "") or "").strip(),
                "actual_top1": str(hydrated.get("actual_top1", "") or "").strip(),
                "actual_top2": str(hydrated.get("actual_top2", "") or "").strip(),
                "actual_top3": str(hydrated.get("actual_top3", "") or "").strip(),
                "ntfy_notify_status": str(hydrated.get("ntfy_notify_status", "") or "").strip(),
                "ntfy_notify_run_id": str(hydrated.get("ntfy_notify_run_id", "") or "").strip(),
                "ntfy_notify_engine": str(hydrated.get("ntfy_notify_engine", "") or "").strip(),
                "ntfy_notified_at": str(hydrated.get("ntfy_notified_at", "") or "").strip(),
                "ntfy_notify_error": str(hydrated.get("ntfy_notify_error", "") or "").strip(),
                "notes": str(hydrated.get("notes", "") or "").strip(),
                "step_badges": step_badges,
                "process_log": _race_job_process_log_entries(hydrated),
            }
        )

    return {
        "authorized": True,
        "show_settled": bool(show_settled),
        "summary": summary,
        "jobs": items,
    }


def _build_admin_workspace_payload(token: str = "", scope_key: str = "", run_id: str = ""):
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm:
        scope_norm = "central_dirt"
    if not run_row or not resolved_run_id:
        raise LookupError("run not found")

    odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "odds_path", "odds")
    fuku_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "fuku_odds_path", "fuku_odds")
    name_to_no_map = load_name_to_no(odds_path) if odds_path and Path(odds_path).exists() else {}
    win_odds_map = load_win_odds_map(odds_path) if odds_path and Path(odds_path).exists() else {}
    place_odds_map = load_place_odds_map(fuku_odds_path) if fuku_odds_path and Path(fuku_odds_path).exists() else {}
    win_snapshot = list(load_odds_snapshot(odds_path).values()) if odds_path and Path(odds_path).exists() else []
    place_snapshot = list(load_odds_snapshot(fuku_odds_path).values()) if fuku_odds_path and Path(fuku_odds_path).exists() else []
    win_snapshot.sort(key=odds_sort_key)
    place_snapshot.sort(key=odds_sort_key)

    recent_runs = []
    for item in get_recent_runs(scope_norm, limit=120):
        recent_runs.append(
            {
                "run_id": str(item.get("run_id", "") or "").strip(),
                "label": str(item.get("label", "") or item.get("run_id", "") or "").strip(),
                "race_id": str(item.get("race_id", "") or "").strip(),
                "timestamp": str(item.get("timestamp", "") or "").strip(),
                "location": str(item.get("location", "") or "").strip(),
                "date": str(item.get("date", "") or item.get("race_date", "") or "").strip(),
                "status": str(item.get("status", "") or "").strip(),
            }
        )

    predictor_sections = []
    for spec, pred_path in resolve_predictor_paths(scope_norm, resolved_run_id, run_row):
        if not pred_path or not pred_path.exists():
            continue
        predictor_run_row = dict(run_row or {})
        predictor_run_row["predictions_path"] = str(pred_path)
        top_rows, _ = load_top5_table(scope_norm, resolved_run_id, predictor_run_row)
        mark_rows, _ = load_ability_marks_table(scope_norm, resolved_run_id, predictor_run_row)
        if not mark_rows:
            mark_rows, _ = load_mark_recommendation_table(scope_norm, resolved_run_id, predictor_run_row)
        summary_rows = load_prediction_summary(scope_norm, resolved_run_id, predictor_run_row)
        predictor_sections.append(
            {
                "predictor_id": str(spec.get("id", "") or "").strip(),
                "label": str(spec.get("label", "") or "").strip(),
                "top5_rows": list(top_rows or []),
                "mark_rows": list(mark_rows or []),
                "summary_rows": list(summary_rows or []),
            }
        )

    actual_result_map = _load_actual_result_map(scope_norm)
    actual_snapshot = _actual_result_snapshot(scope_norm, resolved_run_id, run_row, actual_result_map)
    actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])

    policy_cards = []
    for payload in load_policy_payloads(scope_norm, resolved_run_id, run_row):
        engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
        output = report_policy_primary_output(payload)
        marks_map = report_policy_marks_map(payload)
        ticket_rows = load_policy_run_ticket_rows(resolved_run_id, policy_engine=engine) or list(
            report_policy_primary_budget(payload).get("tickets", []) or []
        )
        policy_cards.append(
            {
                "engine": engine,
                "engine_label": REPORT_LLM_BATTLE_LABELS.get(engine, engine),
                "model": str((payload or {}).get("policy_model", "") or (payload or {}).get("gemini_model", "") or "").strip(),
                "bet_decision": str(output.get("bet_decision", "") or "").strip(),
                "participation_level": str(output.get("participation_level", "") or "").strip(),
                "enabled_bet_types": list(output.get("enabled_bet_types", []) or []),
                "reason_codes": list(output.get("reason_codes", []) or []),
                "marks": list(output.get("marks", []) or []),
                "marks_text": report_format_marks_text(marks_map),
                "ticket_rows": list(ticket_rows or []),
                "ticket_plan_text": report_format_ticket_plan_text(ticket_rows, output),
                "result_triplet_text": _format_triplet_text(_marks_result_triplet(marks_map, actual_horse_nos)),
                "payload_preview_text": json.dumps(payload, ensure_ascii=False, indent=2)[:4000],
            }
        )

    predictor_overview = build_multi_predictor_context(
        scope_norm,
        resolved_run_id,
        run_row,
        name_to_no_map,
        win_odds_map,
        place_odds_map,
    )
    job_meta = _find_job_meta_for_run(scope_norm, resolved_run_id, run_row) or {}

    ledger_date = extract_ledger_date(resolved_run_id, str((run_row or {}).get("timestamp", "") or "").strip())
    portfolio_summaries = []
    for engine in REPORT_LLM_BATTLE_ORDER:
        bankroll = load_policy_bankroll_summary(
            run_id=resolved_run_id,
            timestamp=str((run_row or {}).get("timestamp", "") or "").strip(),
            policy_engine=engine,
        )
        history = build_history_context(
            BASE_DIR,
            ledger_date,
            lookback_days=14,
            recent_ticket_limit=5,
            policy_engine=engine,
        )
        portfolio_summaries.append(
            {
                "engine": engine,
                "engine_label": REPORT_LLM_BATTLE_LABELS.get(engine, engine),
                "today": dict(bankroll or {}),
                "lookback_summary": dict((history or {}).get("lookback_summary", {}) or {}),
                "recent_days": list((history or {}).get("recent_days", []) or [])[:7],
                "bet_type_breakdown": list((history or {}).get("bet_type_breakdown", []) or [])[:6],
                "recent_tickets": list((history or {}).get("recent_tickets", []) or [])[:5],
            }
        )

    run_result_summary = summary_service.load_run_result_summary(get_data_dir, BASE_DIR, load_csv_rows, scope_norm, resolved_run_id)
    run_bet_type_summary = summary_service.load_run_bet_type_summary(get_data_dir, BASE_DIR, load_csv_rows, scope_norm, resolved_run_id)
    run_bet_ticket_summary = summary_service.load_run_bet_ticket_summary(get_data_dir, BASE_DIR, load_csv_rows, scope_norm, resolved_run_id)
    run_predictor_summary = summary_service.load_run_predictor_summary(
        get_data_dir,
        BASE_DIR,
        load_csv_rows,
        to_float,
        compute_top5_hit_count,
        scope_norm,
        resolved_run_id,
    )
    run_context_rows = [
        {
            "scope_key": scope_norm,
            "scope_label": _scope_display_name(scope_norm),
            "run_id": resolved_run_id,
            "race_id": normalize_race_id((run_row or {}).get("race_id", "")),
            "location": str((run_row or {}).get("location", "") or "").strip(),
            "race_date": str((run_row or {}).get("date", "") or (run_row or {}).get("race_date", "") or "").strip(),
            "timestamp": str((run_row or {}).get("timestamp", "") or "").strip(),
            "off_time": str((run_row or {}).get("scheduled_off_time", "") or "").strip(),
        }
    ]
    run_asset_rows = []
    for field_name, label in (
        ("odds_path", "odds"),
        ("fuku_odds_path", "place_odds"),
        ("wide_odds_path", "wide_odds"),
        ("quinella_odds_path", "quinella_odds"),
        ("exacta_odds_path", "exacta_odds"),
        ("trio_odds_path", "trio_odds"),
        ("trifecta_odds_path", "trifecta_odds"),
        ("predictions_path", "predictor_v1"),
        ("predictions_v2_opus_path", "predictor_v2"),
        ("predictions_v3_premium_path", "predictor_v3"),
        ("predictions_v4_gemini_path", "predictor_v4"),
        ("predictions_v5_stacking_path", "predictor_v5"),
    ):
        value = str((run_row or {}).get(field_name, "") or "").strip()
        if not value:
            continue
        path_obj = Path(value)
        run_asset_rows.append(
            {
                "asset": label,
                "path": value,
                "exists": "yes" if path_obj.exists() else "no",
                "mtime": format_path_mtime(path_obj, label),
            }
        )

    return {
        "authorized": True,
        "job_id": str(job_meta.get("job_id", "") or "").strip(),
        "ntfy_notify_status": str(job_meta.get("ntfy_notify_status", "") or "").strip(),
        "ntfy_notify_run_id": str(job_meta.get("ntfy_notify_run_id", "") or "").strip(),
        "ntfy_notify_engine": str(job_meta.get("ntfy_notify_engine", "") or "").strip(),
        "ntfy_notified_at": str(job_meta.get("ntfy_notified_at", "") or "").strip(),
        "ntfy_notify_error": str(job_meta.get("ntfy_notify_error", "") or "").strip(),
        "scope_key": scope_norm,
        "scope_label": _scope_display_name(scope_norm),
        "run_id": resolved_run_id,
        "race_id": normalize_race_id((run_row or {}).get("race_id", "")),
        "location": str((run_row or {}).get("location", "") or "").strip(),
        "race_date": str((run_row or {}).get("date", "") or (run_row or {}).get("race_date", "") or "").strip(),
        "timestamp": str((run_row or {}).get("timestamp", "") or "").strip(),
        "official_result_url": build_official_result_url(
            race_id=normalize_race_id((run_row or {}).get("race_id", "")),
            source=_official_result_source_for_scope(scope_norm),
        ),
        "actual_result": {
            "actual_top1": str(actual_snapshot.get("actual_top1", "") or "").strip(),
            "actual_top2": str(actual_snapshot.get("actual_top2", "") or "").strip(),
            "actual_top3": str(actual_snapshot.get("actual_top3", "") or "").strip(),
            "actual_horse_nos": actual_horse_nos,
        },
        "available_scopes": [
            {"scope_key": "central_dirt", "label": _scope_display_name("central_dirt")},
            {"scope_key": "central_turf", "label": _scope_display_name("central_turf")},
            {"scope_key": "local", "label": _scope_display_name("local")},
        ],
        "available_runs": recent_runs,
        "predictors": predictor_sections,
        "predictor_overview": predictor_overview,
        "policies": policy_cards,
        "portfolio_summaries": portfolio_summaries,
        "run_context_rows": run_context_rows,
        "run_asset_rows": run_asset_rows,
        "run_result_summary": run_result_summary,
        "run_bet_type_summary": run_bet_type_summary,
        "run_bet_ticket_summary": run_bet_ticket_summary,
        "run_predictor_summary": run_predictor_summary,
        "odds_snapshots": {
            "win": win_snapshot[:12],
            "place": place_snapshot[:12],
        },
    }


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


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url=PUBLIC_BASE_PATH, status_code=307)


@app.get(CONSOLE_BASE_PATH, response_class=HTMLResponse)
def console_spa():
    return build_public_index_response(CONSOLE_BASE_PATH)


@app.get(f"{CONSOLE_BASE_PATH}/workspace", response_class=HTMLResponse)
def console_workspace_spa():
    return build_public_index_response(f"{CONSOLE_BASE_PATH}/workspace")


@app.get(PUBLIC_BASE_PATH, response_class=HTMLResponse)
def llm_today(date: str = "", scope_key: str = ""):
    return build_public_index_response(PUBLIC_BASE_PATH)


@app.get(f"{PUBLIC_BASE_PATH}/about", response_class=HTMLResponse)
@app.get(f"{PUBLIC_BASE_PATH}/guide", response_class=HTMLResponse)
@app.get(f"{PUBLIC_BASE_PATH}/methodology", response_class=HTMLResponse)
@app.get(f"{PUBLIC_BASE_PATH}/privacy", response_class=HTMLResponse)
@app.get(f"{PUBLIC_BASE_PATH}/terms", response_class=HTMLResponse)
@app.get(f"{PUBLIC_BASE_PATH}/disclaimer", response_class=HTMLResponse)
@app.get(f"{PUBLIC_BASE_PATH}/contact", response_class=HTMLResponse)
def public_static_pages(request: Request):
    return build_public_index_response(request.url.path)


@app.get("/ads.txt")
def ads_txt():
    return FileResponse(ADS_TXT_PATH, media_type="text/plain; charset=utf-8")


@app.get("/robots.txt")
def robots_txt():
    body = "\n".join(
        [
            "User-agent: *",
            "Allow: /",
            "Disallow: /keiba/console",
            "Disallow: /keiba/console/",
            "Disallow: /keiba/api/admin/",
            "Disallow: /keiba/internal/",
            "Disallow: /internal/",
            "Disallow: /api/admin/",
            f"Sitemap: {PUBLIC_SITE_URL}/sitemap.xml",
            "",
        ]
    )
    return Response(content=body, media_type="text/plain; charset=utf-8")


@app.get("/sitemap.xml")
def sitemap_xml():
    pages = [
        f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}",
        f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}/about",
        f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}/guide",
        f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}/methodology",
        f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}/privacy",
        f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}/terms",
        f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}/disclaimer",
        f"{PUBLIC_SITE_URL}{PUBLIC_BASE_PATH}/contact",
    ]
    lastmod = datetime.now(timezone.utc).date().isoformat()
    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for url in pages:
        body.extend(
            [
                "  <url>",
                f"    <loc>{html.escape(url)}</loc>",
                f"    <lastmod>{lastmod}</lastmod>",
                "  </url>",
            ]
        )
    body.append("</urlset>")
    return Response(content="\n".join(body), media_type="application/xml; charset=utf-8")


@app.get(f"{PUBLIC_API_BASE_PATH}/board")
@app.get("/api/public/board")
def public_board_api(date: str = "", scope_key: str = ""):
    return JSONResponse(build_public_board_payload(date_text=date, scope_key=scope_key))


@app.get(f"{PUBLIC_BASE_PATH}/api/admin/auth-check")
def admin_auth_check(request: Request, token: str = ""):
    supplied = _admin_supplied_token(request, token)
    enabled = _admin_token_enabled()
    valid = _admin_token_valid(supplied)
    return JSONResponse(
        {
            "enabled": enabled,
            "valid": valid,
            "console_url": CONSOLE_BASE_PATH,
        }
    )


@app.get(f"{PUBLIC_BASE_PATH}/api/admin/jobs")
def admin_jobs_api(request: Request, token: str = "", show_settled: str = ""):
    supplied = _admin_supplied_token(request, token)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"authorized": False, "error": "admin token invalid"}, status_code=403)
    show_settled_flag = str(show_settled or "").strip().lower() in ("1", "true", "yes", "on")
    return JSONResponse(_build_admin_jobs_payload(token=supplied, show_settled=show_settled_flag))


@app.get(f"{PUBLIC_BASE_PATH}/api/admin/runs")
def admin_runs_api(request: Request, token: str = "", scope_key: str = "central_dirt", limit: int = DEFAULT_RUN_LIMIT, q: str = ""):
    supplied = _admin_supplied_token(request, token)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"authorized": False, "error": "admin token invalid"}, status_code=403)
    return JSONResponse(
        {
            "authorized": True,
            "runs": web_admin_ops.api_runs_response(
                scope_key=scope_key,
                limit=limit,
                query=q,
                normalize_scope_key=normalize_scope_key,
                default_limit=DEFAULT_RUN_LIMIT,
                max_limit=MAX_RUN_LIMIT,
                get_recent_runs=get_recent_runs,
            ),
        }
    )


@app.get(f"{PUBLIC_BASE_PATH}/api/admin/workspace")
def admin_workspace_api(request: Request, token: str = "", scope_key: str = "", run_id: str = ""):
    supplied = _admin_supplied_token(request, token)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"authorized": False, "error": "admin token invalid"}, status_code=403)
    try:
        payload = _build_admin_workspace_payload(token=supplied, scope_key=scope_key, run_id=run_id)
    except LookupError:
        return JSONResponse({"authorized": True, "error": "run not found"}, status_code=404)
    return JSONResponse(payload)


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/ops/reset_llm_state")
async def admin_reset_llm_state_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    reset_llm_state_files(BASE_DIR)
    return JSONResponse({"ok": True, "output_text": "LLM state reset completed."})


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/workspace/run_llm_buy")
async def admin_workspace_run_llm_buy_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    scope_key = str((payload or {}).get("scope_key", "") or "").strip()
    run_id = str((payload or {}).get("run_id", "") or "").strip()
    policy_engine = normalize_policy_engine((payload or {}).get("policy_engine", "gemini"))
    policy_model = str((payload or {}).get("policy_model", "") or "").strip()
    refresh_enabled = bool((payload or {}).get("refresh_odds", True))
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return JSONResponse({"ok": False, "error": "run not found"}, status_code=404)
    refresh_ok, refresh_message, refresh_warnings = maybe_refresh_run_odds(scope_norm, run_row, resolved_run_id, refresh_enabled)
    if not refresh_ok:
        return JSONResponse(
            {
                "ok": False,
                "error": build_llm_buy_output(
                    load_policy_bankroll_summary(resolved_run_id, run_row.get("timestamp", ""), policy_engine=policy_engine),
                    refresh_ok,
                    refresh_message,
                    refresh_warnings,
                    "[llm_buy][blocked] odds refresh failed; skip policy execution.",
                    policy_engine=policy_engine,
                ),
            },
            status_code=400,
        )
    try:
        result = execute_policy_buy(scope_norm, run_row, resolved_run_id, policy_engine=policy_engine, policy_model=policy_model)
    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": build_llm_buy_output(
                    load_policy_bankroll_summary(resolved_run_id, run_row.get("timestamp", ""), policy_engine=policy_engine),
                    refresh_ok,
                    refresh_message,
                    refresh_warnings,
                    f"[llm_buy][error] {exc}",
                    policy_engine=policy_engine,
                ),
            },
            status_code=500,
        )
    return JSONResponse(
        {
            "ok": True,
            "scope_key": scope_norm,
            "run_id": resolved_run_id,
            "engine": result["engine"],
            "output_text": build_llm_buy_output(
                result["summary_before"],
                refresh_ok,
                refresh_message,
                refresh_warnings,
                result["output_text"],
                result["engine"],
            ),
        }
    )


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/workspace/run_all_llm_buy")
async def admin_workspace_run_all_llm_buy_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    scope_key = str((payload or {}).get("scope_key", "") or "").strip()
    run_id = str((payload or {}).get("run_id", "") or "").strip()
    refresh_enabled = bool((payload or {}).get("refresh_odds", True))
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return JSONResponse({"ok": False, "error": "run not found"}, status_code=404)
    refresh_ok, refresh_message, refresh_warnings = maybe_refresh_run_odds(scope_norm, run_row, resolved_run_id, refresh_enabled)
    if not refresh_ok:
        return JSONResponse(
            {
                "ok": False,
                "error": build_llm_buy_output(
                    load_policy_bankroll_summary(resolved_run_id, run_row.get("timestamp", ""), policy_engine="gemini"),
                    refresh_ok,
                    refresh_message,
                    refresh_warnings,
                    "[llm_buy][blocked] odds refresh failed; skip all policy engines.",
                    policy_engine="all",
                ),
            },
            status_code=400,
        )
    outputs = []
    errors = []
    for engine in ("gemini", "deepseek", "openai", "grok"):
        try:
            result = execute_policy_buy(scope_norm, run_row, resolved_run_id, policy_engine=engine, policy_model="")
            outputs.append(
                {
                    "engine": result["engine"],
                    "output_text": build_llm_buy_output(
                        result["summary_before"],
                        refresh_ok,
                        refresh_message,
                        refresh_warnings,
                        result["output_text"],
                        result["engine"],
                    ),
                }
            )
        except Exception as exc:
            errors.append(f"[llm_buy][{engine}] {exc}")
    if errors and not outputs:
        return JSONResponse({"ok": False, "error": "\n\n".join(errors)}, status_code=500)
    return JSONResponse({"ok": True, "scope_key": scope_norm, "run_id": resolved_run_id, "outputs": outputs, "errors": errors})


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/workspace/topup_all_llm_budget")
async def admin_workspace_topup_all_llm_budget_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    scope_key = str((payload or {}).get("scope_key", "") or "").strip()
    run_id = str((payload or {}).get("run_id", "") or "").strip()
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return JSONResponse({"ok": False, "error": "run not found"}, status_code=404)
    ledger_date = extract_ledger_date(resolved_run_id, run_row.get("timestamp", ""))
    amount_yen = resolve_daily_bankroll_yen(ledger_date)
    summaries = []
    for engine in ("gemini", "deepseek", "openai", "grok"):
        summary = add_bankroll_topup(BASE_DIR, ledger_date, amount_yen, policy_engine=engine)
        summaries.append(
            {
                "engine": engine,
                "available_bankroll_yen": int(summary.get("available_bankroll_yen", 0) or 0),
                "topup_yen": int(summary.get("topup_yen", 0) or 0),
            }
        )
    return JSONResponse(
        {
            "ok": True,
            "scope_key": scope_norm,
            "run_id": resolved_run_id,
            "ledger_date": ledger_date,
            "amount_yen": amount_yen,
            "summaries": summaries,
        }
    )


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/workspace/record_predictor")
async def admin_workspace_record_predictor_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    scope_key = str((payload or {}).get("scope_key", "") or "").strip()
    run_id = str((payload or {}).get("run_id", "") or "").strip()
    top1 = str((payload or {}).get("top1", "") or "").strip()
    top2 = str((payload or {}).get("top2", "") or "").strip()
    top3 = str((payload or {}).get("top3", "") or "").strip()
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return JSONResponse({"ok": False, "error": "run not found"}, status_code=404)
    if not top1 or not top2 or not top3:
        return JSONResponse({"ok": False, "error": "top1/top2/top3 required"}, status_code=400)
    job_meta = _find_job_meta_for_run(scope_norm, resolved_run_id, run_row) or {}
    job_id = str(job_meta.get("job_id", "") or "").strip()
    if not job_id:
        return JSONResponse({"ok": False, "error": "job not found for run"}, status_code=404)
    try:
        from race_job_runner import settle_race_job

        summary = settle_race_job(BASE_DIR, job_id, [top1, top2, top3])
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "scope_key": scope_norm,
            "run_id": resolved_run_id,
            "actual_top3": [top1, top2, top3],
            "summary": summary or {},
            "output_text": str((summary or {}).get("output", "") or ""),
        },
        status_code=200,
    )


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/workspace/fetch_result_and_settle")
async def admin_workspace_fetch_result_and_settle_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    scope_key = str((payload or {}).get("scope_key", "") or "").strip()
    run_id = str((payload or {}).get("run_id", "") or "").strip()
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return JSONResponse({"ok": False, "error": "run not found"}, status_code=404)
    job_meta = _find_job_meta_for_run(scope_norm, resolved_run_id, run_row) or {}
    job_id = str(job_meta.get("job_id", "") or "").strip()
    if not job_id:
        return JSONResponse({"ok": False, "error": "job not found for run"}, status_code=404)
    race_id = normalize_race_id((run_row or {}).get("race_id", ""))
    if not race_id:
        return JSONResponse({"ok": False, "error": "race_id missing for run"}, status_code=400)
    try:
        actual_top3, official_payload = _fetch_official_top3_names(scope_norm, race_id)
        from race_job_runner import settle_race_job

        summary = settle_race_job(BASE_DIR, job_id, actual_top3, official_result_payload=official_payload)
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc), "job_id": job_id, "run_id": resolved_run_id, "race_id": race_id},
            status_code=500,
        )
    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "scope_key": scope_norm,
            "run_id": resolved_run_id,
            "race_id": race_id,
            "actual_top3": actual_top3,
            "result_url": str(official_payload.get("source_url", "") or "").strip(),
            "summary": dict(summary or {}),
        }
    )


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/process_now")
async def admin_jobs_process_now_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    job_id = str((payload or {}).get("job_id", "") or "").strip()
    if not job_id:
        return JSONResponse({"ok": False, "error": "job_id required"}, status_code=400)
    try:
        from race_job_runner import process_race_job

        summary = process_race_job(BASE_DIR, job_id)
    except Exception as exc:
        try:
            from race_job_runner import fail_race_job

            fail_race_job(BASE_DIR, job_id, str(exc))
        except Exception:
            pass
        return JSONResponse({"ok": False, "error": str(exc), "job_id": job_id}, status_code=500)
    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "run_id": str((summary or {}).get("run_id", "") or "").strip(),
            "remote_predictor_task_id": str(
                (summary or {}).get("remote_predictor_task_id", "")
                or (summary or {}).get("v5_remote_task_id", "")
                or ""
            ).strip(),
        }
    )


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/delete")
async def admin_jobs_delete_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    job_id = str((payload or {}).get("job_id", "") or "").strip()
    if not job_id:
        return JSONResponse({"ok": False, "error": "job_id required"}, status_code=400)
    deleted = delete_race_job(BASE_DIR, job_id)
    if deleted is None:
        return JSONResponse({"ok": False, "error": "job not found", "job_id": job_id}, status_code=404)
    return JSONResponse({"ok": True, "job_id": job_id})


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/edit")
async def admin_jobs_edit_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    job_id = str((payload or {}).get("job_id", "") or "").strip()
    if not job_id:
        return JSONResponse({"ok": False, "error": "job_id required"}, status_code=400)

    race_id = normalize_race_id((payload or {}).get("race_id", ""))
    location = str((payload or {}).get("location", "") or "").strip()
    race_date = str((payload or {}).get("race_date", "") or "").strip() or _default_job_race_date_text()
    scheduled_off_time = str((payload or {}).get("scheduled_off_time", "") or "").strip()
    target_distance = str((payload or {}).get("target_distance", "") or "").strip()
    target_track_condition = str((payload or {}).get("target_track_condition", "") or "").strip()
    lead_minutes_text = str((payload or {}).get("lead_minutes", "30") or "30").strip() or "30"
    notes = str((payload or {}).get("notes", "") or "").strip()

    if not race_id:
        return JSONResponse({"ok": False, "error": "race_id required"}, status_code=400)
    if not scheduled_off_time:
        return JSONResponse({"ok": False, "error": "scheduled_off_time required"}, status_code=400)
    try:
        target_distance_value = int(target_distance)
    except ValueError:
        return JSONResponse({"ok": False, "error": "target_distance must be integer"}, status_code=400)
    if target_track_condition not in ("良", "稍重", "重", "不良"):
        return JSONResponse({"ok": False, "error": "invalid target_track_condition"}, status_code=400)
    try:
        lead_value = int(lead_minutes_text)
    except ValueError:
        lead_value = 30

    def _edit_job(row, now_text):
        row["race_id"] = race_id
        row["location"] = location
        row["race_date"] = race_date
        row["scheduled_off_time"] = scheduled_off_time
        row["process_after_time"] = ""
        row["target_distance"] = str(target_distance_value)
        row["target_track_condition"] = target_track_condition
        row["lead_minutes"] = lead_value
        row["notes"] = notes
        row["updated_at"] = now_text

    job = update_race_job(BASE_DIR, job_id, _edit_job)
    if job is None:
        return JSONResponse({"ok": False, "error": "job not found", "job_id": job_id}, status_code=404)
    return JSONResponse({"ok": True, "job_id": job_id})


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/update")
async def admin_jobs_update_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    job_id = str((payload or {}).get("job_id", "") or "").strip()
    action = str((payload or {}).get("action", "") or "").strip()
    if not job_id:
        return JSONResponse({"ok": False, "error": "job_id required"}, status_code=400)
    if not action:
        return JSONResponse({"ok": False, "error": "action required"}, status_code=400)
    job = apply_race_job_action(BASE_DIR, job_id, action)
    if job is None:
        return JSONResponse({"ok": False, "error": "job not found", "job_id": job_id}, status_code=404)
    return JSONResponse({"ok": True, "job_id": job_id, "action": action, "status": str(job.get("status", "") or "").strip()})


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/create")
async def admin_jobs_create_api(
    request: Request,
    scope_key: str = Form(""),
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
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)

    race_id = normalize_race_id(race_id)
    scope_norm = normalize_scope_key(scope_key)
    race_date = str(race_date or "").strip() or _default_job_race_date_text()
    if not scope_norm:
        return JSONResponse({"ok": False, "error": "scope_key required"}, status_code=400)
    if not race_id:
        return JSONResponse({"ok": False, "error": "race_id required"}, status_code=400)
    if not str(scheduled_off_time or "").strip():
        return JSONResponse({"ok": False, "error": "scheduled_off_time required"}, status_code=400)
    try:
        target_distance_value = int(str(target_distance or "").strip())
    except ValueError:
        return JSONResponse({"ok": False, "error": "target_distance must be integer"}, status_code=400)
    if target_distance_value <= 0:
        return JSONResponse({"ok": False, "error": "target_distance must be positive"}, status_code=400)
    target_track_condition = str(target_track_condition or "").strip()
    if target_track_condition not in ("良", "稍重", "重", "不良"):
        return JSONResponse({"ok": False, "error": "invalid target_track_condition"}, status_code=400)
    if kachiuma_file is None or not str(getattr(kachiuma_file, "filename", "") or "").strip():
        return JSONResponse({"ok": False, "error": "kachiuma_file required"}, status_code=400)
    if shutuba_file is None or not str(getattr(shutuba_file, "filename", "") or "").strip():
        return JSONResponse({"ok": False, "error": "shutuba_file required"}, status_code=400)
    try:
        lead_value = int(str(lead_minutes or "30").strip() or "30")
    except ValueError:
        lead_value = 30

    target_surface = _target_surface_from_scope(scope_norm)
    artifact_payloads = []
    job = create_race_job(
        BASE_DIR,
        race_id=race_id,
        scope_key=scope_norm,
        location=location,
        race_date=race_date,
        scheduled_off_time=scheduled_off_time,
        target_surface=target_surface,
        target_distance=str(target_distance_value),
        target_track_condition=target_track_condition,
        lead_minutes=lead_value,
        notes=notes,
        artifacts=[],
    )
    try:
        for artifact_type, upload in (("kachiuma", kachiuma_file), ("shutuba", shutuba_file)):
            if upload is None:
                continue
            payload = await upload.read()
            artifact_payloads.append(
                save_race_job_artifact(
                    BASE_DIR,
                    job["job_id"],
                    artifact_type,
                    upload.filename or f"{artifact_type}.csv",
                    payload,
                )
            )

        def _attach_artifacts(row, now_text):
            row["artifacts"] = artifact_payloads
            row["status"] = "scheduled"
            row["updated_at"] = now_text

        update_race_job(BASE_DIR, job["job_id"], _attach_artifacts)
    except Exception:
        try:
            delete_race_job(BASE_DIR, job["job_id"])
        except Exception:
            pass
        raise

    return JSONResponse({"ok": True, "job_id": str(job.get("job_id", "") or "").strip()})


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/import_archive")
async def admin_jobs_import_archive_api(
    request: Request,
    overwrite: str = Form(""),
    archive_file: UploadFile = File(None),
):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    if archive_file is None or not str(getattr(archive_file, "filename", "") or "").strip():
        return JSONResponse({"ok": False, "error": "archive_file required"}, status_code=400)
    filename = str(archive_file.filename or "").strip()
    if not filename.lower().endswith(".zip"):
        return JSONResponse({"ok": False, "error": "archive_file must be zip"}, status_code=400)
    try:
        archive_bytes = await archive_file.read()
        summary = _import_history_zip(BASE_DIR, archive_bytes, overwrite=str(overwrite or "").strip() == "1")
    except zipfile.BadZipFile:
        return JSONResponse({"ok": False, "error": "invalid zip archive"}, status_code=400)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, **summary})


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/scan_due")
async def admin_jobs_scan_due_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    changed = scan_due_race_jobs(BASE_DIR)
    return JSONResponse(
        {
            "ok": True,
            "queued_count": len(changed),
            "queued_job_ids": [str(item.get("job_id", "") or "").strip() for item in changed],
        }
    )


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/run_due_now")
def admin_jobs_run_due_now_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    summary = run_due_jobs_once()
    ok = not bool(list(summary.get("errors", []) or []))
    return JSONResponse({"ok": ok, **summary}, status_code=200 if ok else 500)


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/daily_summary_share")
async def admin_jobs_daily_summary_share_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    payload = await request.json()
    date_text = str((payload or {}).get("date_text", "") or "").strip()
    try:
        summary = build_daily_summary_share_payload(date_text=date_text)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True, **summary})


@app.post(f"{PUBLIC_BASE_PATH}/api/admin/jobs/topup_today_all_llm")
async def admin_jobs_topup_today_all_llm_api(request: Request):
    supplied = _admin_supplied_token(request)
    if _admin_token_enabled() and not _admin_token_valid(supplied):
        return JSONResponse({"ok": False, "error": "admin token invalid"}, status_code=403)
    ledger_date = datetime.now().strftime("%Y%m%d")
    amount_yen = resolve_daily_bankroll_yen(ledger_date)
    summaries = []
    for engine in REPORT_LLM_BATTLE_ORDER:
        summaries.append(add_bankroll_topup(BASE_DIR, ledger_date, amount_yen, policy_engine=engine))
    return JSONResponse(
        {
            "ok": True,
            "ledger_date": ledger_date,
            "amount_yen": amount_yen,
            "engines": list(REPORT_LLM_BATTLE_ORDER),
            "summaries": summaries,
        }
    )


@app.get("/internal/run_due")
@app.post("/internal/run_due")
def internal_run_due(request: Request):
    supplied = _admin_supplied_token(request)
    return web_admin_tasks.internal_run_due_response(
        base_dir=BASE_DIR,
        token=supplied,
        admin_token_valid=_admin_token_valid,
        scan_due_race_jobs=scan_due_race_jobs,
        load_race_jobs=load_race_jobs,
    )


def _remote_v5_result_dir():
    path = BASE_DIR / "data" / "_shared" / "remote_v5_results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _remote_v5_result_summary_path(scope_key, race_id, run_id):
    data_dir = get_data_dir(BASE_DIR, scope_key)
    race_dir = data_dir / str(race_id or "").strip() if str(race_id or "").strip() else data_dir
    race_dir.mkdir(parents=True, exist_ok=True)
    if str(race_id or "").strip():
        return race_dir / f"remote_predictor_summary_{run_id}_{race_id}.json"
    return race_dir / f"remote_predictor_summary_{run_id}.json"


def _apply_remote_v5_result(task, payload):
    task_row = dict(task or {})
    bundle_result = dict(payload.get("result") or {})
    content_base64 = str(bundle_result.get("content_base64", "") or "").strip()
    if not content_base64:
        raise ValueError("remote predictor result content_base64 missing")
    try:
        zip_bytes = base64.b64decode(content_base64)
    except Exception as exc:
        raise ValueError(f"invalid remote predictor base64: {exc}") from exc

    task_id = str(task_row.get("task_id", "") or "").strip()
    run_id = str(task_row.get("run_id", "") or "").strip()
    race_id = str(task_row.get("race_id", "") or "").strip()
    scope_key = str(task_row.get("scope_key", "") or "").strip()
    if not run_id or not scope_key:
        raise ValueError("remote predictor task missing run_id or scope_key")

    result_dir = _remote_v5_result_dir()
    result_zip_path = result_dir / f"{task_id}.zip"
    result_zip_path.write_bytes(zip_bytes)

    data_dir = get_data_dir(BASE_DIR, scope_key)
    updates = {}
    summary_path_text = ""
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())
        for spec in list_predictors():
            latest_name = str(spec.get("latest_filename", "") or "").strip()
            if not latest_name or latest_name not in names:
                continue
            dest_path = snapshot_prediction_path(data_dir, race_id, run_id, spec["id"])
            if dest_path is None:
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(zf.read(latest_name))
            updates[str(spec.get("run_field", "") or "").strip()] = str(dest_path)
        if "remote_predictor_summary.json" in names:
            summary_path = _remote_v5_result_summary_path(scope_key, race_id, run_id)
            summary_path.write_bytes(zf.read("remote_predictor_summary.json"))
            summary_path_text = str(summary_path)

    run_row = resolve_run(run_id, scope_key)
    if run_row is None:
        raise ValueError(f"run row not found for remote predictor callback: run_id={run_id}")
    if updates:
        update_run_row_fields(scope_key, run_row, updates)
    return {
        "run_id": run_id,
        "race_id": race_id,
        "scope_key": scope_key,
        "updated_fields": sorted([key for key in updates.keys() if key]),
        "result_zip_path": str(result_zip_path),
        "summary_path": summary_path_text,
    }


@app.get(f"{PUBLIC_BASE_PATH}/internal/v5_tasks/{{task_id}}/bundle")
def internal_v5_task_bundle(task_id: str, token: str = ""):
    task = get_v5_remote_task(BASE_DIR, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="remote task not found")
    expected = str((task or {}).get("bundle_token", "") or "").strip()
    supplied = str(token or "").strip()
    if not expected or not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="bundle token invalid")
    bundle_bytes = web_remote_predictors.remote_v5_bundle_zip_bytes(task)
    return Response(
        content=bundle_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{task_id}.zip"'},
    )


@app.post(f"{PUBLIC_BASE_PATH}/internal/v5_tasks/{{task_id}}/callback")
async def internal_v5_task_callback(task_id: str, request: Request):
    task = get_v5_remote_task(BASE_DIR, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="remote task not found")
    body = await request.body()
    signature = str(request.headers.get("X-Hub-Signature-256", "") or "").strip()
    if not _verify_callback_hmac(body, signature):
        raise HTTPException(status_code=403, detail="callback signature invalid")
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid callback json: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid callback payload")

    status = str(payload.get("status", "") or "").strip().lower()
    log_excerpt = str(payload.get("log_excerpt", "") or "").strip()
    summary_payload = payload.get("summary", {})
    if not isinstance(summary_payload, dict):
        summary_payload = {}
    job_id = str((task or {}).get("job_id", "") or "").strip()
    run_id = str((task or {}).get("run_id", "") or "").strip()

    if status == "succeeded":
        try:
            saved = _apply_remote_v5_result(task, payload)
        except Exception as exc:
            update_v5_remote_task(
                BASE_DIR,
                task_id,
                lambda row, now_text: row.update(
                    {
                        "status": "failed",
                        "finished_at": now_text,
                        "error_message": str(exc),
                        "result_summary": dict(summary_payload or {}),
                    }
                ),
            )
            if job_id:
                from race_job_runner import fail_race_job

                fail_race_job(BASE_DIR, job_id, str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        update_v5_remote_task(
            BASE_DIR,
            task_id,
            lambda row, now_text: row.update(
                {
                    "status": "succeeded",
                    "finished_at": now_text,
                    "error_message": "",
                    "result_path": str(saved.get("result_zip_path", "") or ""),
                    "result_summary": dict(summary_payload or {}),
                }
            ),
        )
        web_remote_predictors.promote_job_after_remote_v5(
            base_dir=BASE_DIR,
            job_id=job_id,
            run_id=run_id,
            task_id=task_id,
            log_output=log_excerpt,
            update_race_job=update_race_job,
            initialize_race_job_step_fields=initialize_race_job_step_fields,
            set_race_job_step_state=set_race_job_step_state,
        )
        if web_remote_predictors.remote_predictor_auto_continue_enabled() and job_id:
            web_remote_predictors.auto_continue_remote_policy(base_dir=BASE_DIR, job_id=job_id)
        return JSONResponse({"ok": True, "status": "succeeded", **saved})

    error_message = str(payload.get("error_message", "") or "remote predictor batch failed").strip()
    update_v5_remote_task(
        BASE_DIR,
        task_id,
        lambda row, now_text: row.update(
            {
                "status": "failed",
                "finished_at": now_text,
                "error_message": error_message,
                "result_summary": dict(summary_payload or {}),
            }
        ),
    )
    if job_id:
        from race_job_runner import fail_race_job

        fail_race_job(BASE_DIR, job_id, error_message)
    return JSONResponse(
        {
            "ok": True,
            "status": "failed",
            "task_id": task_id,
            "job_id": job_id,
            "run_id": run_id,
            "error_message": error_message,
            "log_excerpt": log_excerpt,
        },
        status_code=200,
    )

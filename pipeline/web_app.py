import csv
import html
import json
import math
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

from predictor_catalog import canonical_predictor_id, list_predictors, predictor_label, resolve_run_prediction_path
from gemini_portfolio import (
    build_history_context,
    extract_ledger_date,
    load_exacta_odds_map,
    load_daily_profit_rows,
    load_name_to_no,
    load_pair_odds_map,
    load_place_odds_map,
    load_run_tickets,
    load_triple_odds_map,
    load_win_odds_map,
    reserve_run_tickets,
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
from surface_scope import get_data_dir, migrate_legacy_data, normalize_scope_key
from web_data import odds_service, run_resolver, run_store, summary_service, view_data
from web_note import build_mark_note_text
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
load_local_env(BASE_DIR, override=True)


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
    race_url = str(run_row.get("race_url") or "").strip()
    race_id = normalize_race_id(run_row.get("race_id", ""))
    if not race_url and race_id:
        if scope_key in ("central_turf", "central_dirt"):
            base = "https://race.netkeiba.com/race/shutuba.html?race_id="
        else:
            base = "https://nar.netkeiba.com/race/shutuba.html?race_id="
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
    exacta_tmp = ROOT_DIR / "exacta_odds.csv"
    if exacta_odds_path and exacta_tmp.exists():
        try:
            Path(exacta_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exacta_tmp, exacta_odds_path)
        except Exception as exc:
            return False, f"Failed to update exacta odds file: {exc}", []
    elif exacta_odds_path:
        warnings.append("exacta_odds.csv not generated.")
    trio_tmp = ROOT_DIR / "trio_odds.csv"
    if trio_odds_path and trio_tmp.exists():
        try:
            Path(trio_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trio_tmp, trio_odds_path)
        except Exception as exc:
            return False, f"Failed to update trio odds file: {exc}", []
    elif trio_odds_path:
        warnings.append("trio_odds.csv not generated.")
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
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    if extra_env:
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


def load_text_file(path):
    if not path:
        return ""
    path = Path(path)
    if not path.exists():
        return ""
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def load_json_file(path):
    text = load_text_file(path)
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


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
    ledger_date = extract_ledger_date(run_id, timestamp)
    return summarize_bankroll(BASE_DIR, ledger_date, policy_engine=policy_engine)


def load_gemini_bankroll_summary(run_id="", timestamp=""):
    return load_policy_bankroll_summary(run_id, timestamp, policy_engine="gemini")


def load_policy_daily_profit_summary(days=30, policy_engine="gemini"):
    return load_daily_profit_rows(BASE_DIR, days=days, policy_engine=policy_engine)


def load_gemini_daily_profit_summary(days=30):
    return load_policy_daily_profit_summary(days=days, policy_engine="gemini")


def load_policy_run_ticket_rows(run_id, policy_engine="gemini"):
    rows = load_run_tickets(BASE_DIR, run_id, policy_engine=policy_engine)
    out = []
    for row in rows:
        out.append(
            {
                "status": row.get("status", ""),
                "bet_type": row.get("bet_type", ""),
                "horse_no": row.get("horse_nos", ""),
                "horse_name": row.get("horse_names", ""),
                "amount_yen": row.get("stake_yen", ""),
                "odds_used": row.get("odds_used", ""),
                "profit_yen": row.get("profit_yen", ""),
            }
        )
    return out


def load_gemini_run_ticket_rows(run_id):
    return load_policy_run_ticket_rows(run_id, policy_engine="gemini")


def build_llm_buy_output(summary_before, refresh_ok, refresh_message, refresh_warnings, script_output, policy_engine=""):
    parts = []
    if summary_before:
        parts.append(
            (
                "[bankroll_before] date={ledger_date} start_bankroll_yen={start_bankroll_yen} "
                "realized_profit_yen={realized_profit_yen} open_stake_yen={open_stake_yen} "
                "available_bankroll_yen={available_bankroll_yen} pending_tickets={pending_tickets}"
            ).format(**summary_before)
        )
    parts.append(f"[odds_update] status={'ok' if refresh_ok else 'fail'} message={refresh_message or ''}".strip())
    if refresh_warnings:
        parts.append("[odds_update][warnings] " + "; ".join(str(x) for x in refresh_warnings))
    if str(policy_engine or "").strip():
        parts.append(f"[policy] engine={policy_engine}")
    if script_output:
        parts.append(script_output.strip())
    return "\n".join([part for part in parts if str(part).strip()]).strip()


def build_gemini_buy_output(summary_before, refresh_ok, refresh_message, refresh_warnings, script_output):
    return build_llm_buy_output(summary_before, refresh_ok, refresh_message, refresh_warnings, script_output, "gemini")


def load_csv_rows_flexible(path):
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    return []


def parse_horse_no(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        pass
    digits = re.findall(r"\d+", text)
    if len(digits) == 1:
        return int(digits[0])
    return None


def _pick_first_value(row, keys, default=""):
    for key in keys:
        if key in row:
            value = row.get(key)
            text = str(value or "").strip()
            if text:
                return value
    return default


def _prediction_prob_value(row):
    for key in ("Top3Prob_model", "top3_prob_model", "Top3Prob", "top3_prob"):
        if key in row:
            return max(0.0, to_float(row.get(key)))
    return 0.0


def _prediction_rank_value(row, fallback_rank):
    explicit = to_int_or_none(_pick_first_value(row, ("pred_rank", "PredRank", "rank", "Rank"), ""))
    return explicit if explicit is not None and explicit > 0 else int(fallback_rank)


def _prediction_score_value(row):
    for key in ("rank_score_norm", "rank_score", "RankScore", "score"):
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
    items = []
    for idx, row in enumerate(list(pred_rows or []), start=1):
        horse_name = str(_pick_first_value(row, ("HorseName", "horse_name", "name"), "") or "").strip()
        if not horse_name:
            continue
        horse_no_raw = _pick_first_value(row, ("horse_no", "HorseNo", "umaban", "馬番"), "")
        horse_no = str(horse_no_raw or "").strip()
        if not horse_no:
            mapped = name_to_no_map.get(normalize_name(horse_name))
            if mapped is not None:
                horse_no = str(mapped)
        top3_prob = _prediction_prob_value(row)
        items.append(
            {
                "horse_no": horse_no,
                "horse_name": horse_name,
                "pred_rank": _prediction_rank_value(row, idx),
                "top3_prob_model": top3_prob,
                "confidence_score": max(0.0, to_float(row.get("confidence_score"))),
                "stability_score": max(0.0, to_float(row.get("stability_score"))),
                "risk_score": max(0.0, to_float(row.get("risk_score"))),
                "_raw_rank_score": _prediction_score_value(row),
                "source_row": dict(row),
            }
        )
    if not items:
        return []
    items.sort(
        key=lambda item: (
            int(item.get("pred_rank", 9999) or 9999),
            -float(item.get("top3_prob_model", 0.0) or 0.0),
            str(item.get("horse_no", "")),
            str(item.get("horse_name", "")),
        )
    )
    for idx, item in enumerate(items, start=1):
        item["pred_rank"] = idx
    _normalize_score_list(items)
    top3_sum = sum(max(float(item.get("top3_prob_model", 0.0) or 0.0), 0.000001) for item in items)
    for item in items:
        horse_name = str(item.get("horse_name", "") or "")
        horse_no_int = parse_horse_no(item.get("horse_no"))
        item["win_odds"] = round(float(win_odds_map.get(normalize_name(horse_name), 0.0) or 0.0), 6)
        item["place_odds"] = round(float(place_odds_map.get(horse_no_int, 0.0) or 0.0), 6) if horse_no_int else 0.0
        item["win_prob_est"] = round(
            min(
                float(item.get("top3_prob_model", 0.0) or 0.0),
                float(item.get("top3_prob_model", 0.0) or 0.0) / max(top3_sum, 1e-6),
            ),
            6,
        )
    return items


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
        top1_horses.append(str(top_choice.get("horse_no", "") or ""))
        summaries.append(
            {
                "predictor_id": spec["id"],
                "predictor_label": spec["label"],
                "top_choice_horse_no": str(top_choice.get("horse_no", "") or ""),
                "top_choice_horse_name": str(top_choice.get("horse_name", "") or ""),
                "top_choice_top3_prob_model": round(float(top_choice.get("top3_prob_model", 0.0) or 0.0), 6),
                "top_horses": [
                    {
                        "horse_no": str(item.get("horse_no", "") or ""),
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
            horse_no = str(item.get("horse_no", "") or "").strip()
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
    candidates = []
    candidate_lookup = {}
    horse_map = {str(item.get("horse_no", "") or ""): item for item in list(predictions or []) if str(item.get("horse_no", "") or "").strip()}
    for item in predictions:
        horse_no = str(item.get("horse_no", "") or "").strip()
        if not horse_no:
            continue
        top3_prob = max(0.0, float(item.get("top3_prob_model", 0.0) or 0.0))
        win_prob = max(0.01, float(item.get("win_prob_est", 0.0) or 0.0))
        if "win" in allowed_types:
            win_odds = float(item.get("win_odds", 0.0) or 0.0)
            if win_odds > 0:
                ev = round(win_prob * win_odds - 1.0, 6)
                candidate = {
                    "id": f"win:{horse_no}",
                    "bet_type": "win",
                    "legs": [horse_no],
                    "odds_used": round(win_odds, 6),
                    "p_hit": round(win_prob, 6),
                    "ev": ev,
                    "score": round(ev * math.sqrt(max(win_prob, 1e-6)), 6),
                }
                candidates.append(candidate)
                candidate_lookup[candidate["id"]] = candidate
        if "place" in allowed_types:
            place_odds = float(item.get("place_odds", 0.0) or 0.0)
            if place_odds > 0:
                place_prob = max(0.01, min(0.95, top3_prob))
                ev = round(place_prob * place_odds - 1.0, 6)
                candidate = {
                    "id": f"place:{horse_no}",
                    "bet_type": "place",
                    "legs": [horse_no],
                    "odds_used": round(place_odds, 6),
                    "p_hit": round(place_prob, 6),
                    "ev": ev,
                    "score": round(ev * math.sqrt(max(place_prob, 1e-6)), 6),
                }
                candidates.append(candidate)
                candidate_lookup[candidate["id"]] = candidate
    pair_source = list(predictions[:6])
    for idx, left in enumerate(pair_source):
        left_no = str(left.get("horse_no", "") or "").strip()
        if not left_no:
            continue
        for right in pair_source[idx + 1 :]:
            right_no = str(right.get("horse_no", "") or "").strip()
            if not right_no:
                continue
            a = parse_horse_no(left_no)
            b = parse_horse_no(right_no)
            if a is None or b is None:
                continue
            pair = (min(a, b), max(a, b))
            left_top3 = max(0.0, float(left.get("top3_prob_model", 0.0) or 0.0))
            right_top3 = max(0.0, float(right.get("top3_prob_model", 0.0) or 0.0))
            left_win = max(0.0, float(left.get("win_prob_est", 0.0) or 0.0))
            right_win = max(0.0, float(right.get("win_prob_est", 0.0) or 0.0))
            if "wide" in allowed_types:
                odds = float(wide_odds_map.get(pair, 0.0) or 0.0)
                if odds > 0:
                    p_hit = max(0.01, min(0.85, left_top3 * right_top3 * 0.9))
                    ev = round(p_hit * odds - 1.0, 6)
                    candidate = {
                        "id": f"wide:{pair[0]}-{pair[1]}",
                        "bet_type": "wide",
                        "legs": [str(pair[0]), str(pair[1])],
                        "odds_used": round(odds, 6),
                        "p_hit": round(p_hit, 6),
                        "ev": ev,
                        "score": round(ev * math.sqrt(max(p_hit, 1e-6)), 6),
                    }
                    candidates.append(candidate)
                    candidate_lookup[candidate["id"]] = candidate
            if "quinella" in allowed_types:
                odds = float(quinella_odds_map.get(pair, 0.0) or 0.0)
                if odds > 0:
                    p_hit = max(0.005, min(0.35, (left_win * max(right_top3, right_win) + right_win * max(left_top3, left_win)) * 0.5))
                    ev = round(p_hit * odds - 1.0, 6)
                    candidate = {
                        "id": f"quinella:{pair[0]}-{pair[1]}",
                        "bet_type": "quinella",
                        "legs": [str(pair[0]), str(pair[1])],
                        "odds_used": round(odds, 6),
                        "p_hit": round(p_hit, 6),
                        "ev": ev,
                        "score": round(ev * math.sqrt(max(p_hit, 1e-6)), 6),
                    }
                    candidates.append(candidate)
                    candidate_lookup[candidate["id"]] = candidate
            if "exacta" in allowed_types:
                for first, second, first_pred, second_pred in (
                    (a, b, left, right),
                    (b, a, right, left),
                ):
                    odds = float(exacta_odds_map.get((first, second), 0.0) or 0.0)
                    if odds <= 0:
                        continue
                    first_win = max(0.0, float(first_pred.get("win_prob_est", 0.0) or 0.0))
                    second_top3 = max(0.0, float(second_pred.get("top3_prob_model", 0.0) or 0.0))
                    p_hit = max(0.001, min(0.3, first_win * second_top3 * 0.7))
                    ev = round(p_hit * odds - 1.0, 6)
                    candidate = {
                        "id": f"exacta:{first}-{second}",
                        "bet_type": "exacta",
                        "legs": [str(first), str(second)],
                        "odds_used": round(odds, 6),
                        "p_hit": round(p_hit, 6),
                        "ev": ev,
                        "score": round(ev * math.sqrt(max(p_hit, 1e-6)), 6),
                    }
                    candidates.append(candidate)
                    candidate_lookup[candidate["id"]] = candidate

    triple_source = list(predictions[:5])
    for idx_a, first in enumerate(triple_source):
        first_no_text = str(first.get("horse_no", "") or "").strip()
        if not first_no_text:
            continue
        first_no = parse_horse_no(first_no_text)
        if first_no is None:
            continue
        for idx_b in range(idx_a + 1, len(triple_source)):
            second = triple_source[idx_b]
            second_no_text = str(second.get("horse_no", "") or "").strip()
            second_no = parse_horse_no(second_no_text)
            if second_no is None:
                continue
            for idx_c in range(idx_b + 1, len(triple_source)):
                third = triple_source[idx_c]
                third_no_text = str(third.get("horse_no", "") or "").strip()
                third_no = parse_horse_no(third_no_text)
                if third_no is None:
                    continue
                trio_key = tuple(sorted((first_no, second_no, third_no)))
                trio_prob = max(
                    0.0005,
                    min(
                        0.2,
                        float(first.get("top3_prob_model", 0.0) or 0.0)
                        * float(second.get("top3_prob_model", 0.0) or 0.0)
                        * float(third.get("top3_prob_model", 0.0) or 0.0)
                        * 0.85,
                    ),
                )
                if "trio" in allowed_types:
                    trio_odds = float(trio_odds_map.get(trio_key, 0.0) or 0.0)
                    if trio_odds > 0:
                        ev = round(trio_prob * trio_odds - 1.0, 6)
                        candidate = {
                            "id": f"trio:{trio_key[0]}-{trio_key[1]}-{trio_key[2]}",
                            "bet_type": "trio",
                            "legs": [str(trio_key[0]), str(trio_key[1]), str(trio_key[2])],
                            "odds_used": round(trio_odds, 6),
                            "p_hit": round(trio_prob, 6),
                            "ev": ev,
                            "score": round(ev * math.sqrt(max(trio_prob, 1e-6)), 6),
                        }
                        candidates.append(candidate)
                        candidate_lookup[candidate["id"]] = candidate
                if "trifecta" in allowed_types:
                    ordered_triples = [
                        (first_no, second_no, third_no, first, second, third),
                        (first_no, third_no, second_no, first, third, second),
                        (second_no, first_no, third_no, second, first, third),
                        (second_no, third_no, first_no, second, third, first),
                        (third_no, first_no, second_no, third, first, second),
                        (third_no, second_no, first_no, third, second, first),
                    ]
                    for a_no, b_no, c_no, a_pred, b_pred, c_pred in ordered_triples:
                        odds = float(trifecta_odds_map.get((a_no, b_no, c_no), 0.0) or 0.0)
                        if odds <= 0:
                            continue
                        p_hit = max(
                            0.0001,
                            min(
                                0.08,
                                float(a_pred.get("win_prob_est", 0.0) or 0.0)
                                * float(b_pred.get("top3_prob_model", 0.0) or 0.0)
                                * float(c_pred.get("top3_prob_model", 0.0) or 0.0)
                                * 0.55,
                            ),
                        )
                        ev = round(p_hit * odds - 1.0, 6)
                        candidate = {
                            "id": f"trifecta:{a_no}-{b_no}-{c_no}",
                            "bet_type": "trifecta",
                            "legs": [str(a_no), str(b_no), str(c_no)],
                            "odds_used": round(odds, 6),
                            "p_hit": round(p_hit, 6),
                            "ev": ev,
                            "score": round(ev * math.sqrt(max(p_hit, 1e-6)), 6),
                        }
                        candidates.append(candidate)
                        candidate_lookup[candidate["id"]] = candidate
    candidates.sort(key=lambda item: (-float(item.get("score", 0.0) or 0.0), -float(item.get("ev", 0.0) or 0.0), str(item.get("id", ""))))
    return candidates, candidate_lookup, horse_map


def build_pair_odds_top(candidate_lookup):
    rows = []
    for candidate in list(candidate_lookup.values()):
        if str(candidate.get("bet_type", "") or "") not in ("wide", "quinella", "exacta", "trio", "trifecta"):
            continue
        legs = list(candidate.get("legs", []) or [])
        if len(legs) < 2:
            continue
        rows.append(
            {
                "bet_type": str(candidate.get("bet_type", "") or ""),
                "pair": "-".join(str(x) for x in legs),
                "odds": round(float(candidate.get("odds_used", 0.0) or 0.0), 6),
                "score": float(candidate.get("score", 0.0) or 0.0),
            }
        )
    rows.sort(key=lambda item: (-float(item.get("score", 0.0) or 0.0), float(item.get("odds", 0.0) or 0.0)))
    return [{"bet_type": row["bet_type"], "pair": row["pair"], "odds": row["odds"]} for row in rows[:10]]


def build_odds_full(win_rows, place_rows, wide_rows, quinella_rows, exacta_rows=None, trio_rows=None, trifecta_rows=None):
    def _pair_rows(rows):
        out = []
        for row in list(rows or []):
            a = str(row.get("horse_no_a", "") or "").strip()
            b = str(row.get("horse_no_b", "") or "").strip()
            if not a or not b:
                continue
            odds = to_float(row.get("odds_mid", row.get("odds", 0)) or 0)
            out.append({"pair": f"{a}-{b}", "horse_no_a": a, "horse_no_b": b, "odds": round(odds, 6)})
        return out

    def _triple_rows(rows):
        out = []
        for row in list(rows or []):
            a = str(row.get("horse_no_a", "") or "").strip()
            b = str(row.get("horse_no_b", "") or "").strip()
            c = str(row.get("horse_no_c", "") or "").strip()
            if not a or not b or not c:
                continue
            odds = to_float(row.get("odds", 0) or 0)
            out.append(
                {
                    "triple": f"{a}-{b}-{c}",
                    "horse_no_a": a,
                    "horse_no_b": b,
                    "horse_no_c": c,
                    "odds": round(odds, 6),
                }
            )
        return out

    def _single_rows(rows, odds_key):
        out = []
        for row in list(rows or []):
            horse_no = str(row.get("horse_no", "") or "").strip()
            if not horse_no:
                continue
            out.append(
                {
                    "horse_no": horse_no,
                    "name": str(row.get("name", "") or "").strip(),
                    "odds": round(to_float(row.get(odds_key, 0) or 0), 6),
                }
            )
        return out

    return {
        "win": _single_rows(win_rows, "odds"),
        "place": _single_rows(place_rows, "odds_mid" if place_rows and "odds_mid" in place_rows[0] else "odds_low"),
        "wide": _pair_rows(wide_rows),
        "quinella": _pair_rows(quinella_rows),
        "exacta": _pair_rows(exacta_rows),
        "trio": _triple_rows(trio_rows),
        "trifecta": _triple_rows(trifecta_rows),
    }


def build_prediction_field_guide():
    return {
        "horse_no": "马番",
        "HorseName": "马名",
        "pred_rank": "模型排序",
        "Top3Prob_model": "模型给出的前三概率",
        "rank_score_norm": "归一化排序分数",
        "confidence_score": "置信度分数",
        "stability_score": "稳定性分数",
        "risk_score": "风险分数",
        "win_odds": "单胜赔率",
        "place_odds": "位置赔率",
    }


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
    pred_rows = load_csv_rows_flexible(pred_path)
    if not pred_rows:
        return None, "Prediction file is missing or empty."
    name_to_no_map = load_name_to_no(odds_path)
    win_odds_map = load_win_odds_map(odds_path)
    place_odds_map = load_place_odds_map(fuku_odds_path)
    wide_odds_map = load_pair_odds_map(wide_odds_path)
    quinella_odds_map = load_pair_odds_map(quinella_odds_path)
    exacta_odds_map = load_exacta_odds_map(exacta_odds_path)
    trio_odds_map = load_triple_odds_map(trio_odds_path, ordered=False)
    trifecta_odds_map = load_triple_odds_map(trifecta_odds_path, ordered=True)
    predictions = build_policy_prediction_rows(pred_rows, name_to_no_map, win_odds_map, place_odds_map)
    if not predictions:
        return None, "No valid prediction rows could be built for policy input."
    allowed_types = []
    if any(float(item.get("win_odds", 0.0) or 0.0) > 0 for item in predictions):
        allowed_types.append("win")
    if any(float(item.get("place_odds", 0.0) or 0.0) > 0 for item in predictions):
        allowed_types.append("place")
    if wide_odds_map:
        allowed_types.append("wide")
    if quinella_odds_map:
        allowed_types.append("quinella")
    if exacta_odds_map:
        allowed_types.append("exacta")
    if trio_odds_map:
        allowed_types.append("trio")
    if trifecta_odds_map:
        allowed_types.append("trifecta")
    if not allowed_types:
        return None, "No usable odds were found for LLM buy."
    candidates, candidate_lookup, horse_map = build_policy_candidates(
        predictions,
        wide_odds_map,
        quinella_odds_map,
        exacta_odds_map,
        trio_odds_map,
        trifecta_odds_map,
        allowed_types,
    )
    if not candidates:
        return None, "Candidate generation failed because odds data is incomplete."
    ledger_date = extract_ledger_date(run_id, (run_row or {}).get("timestamp", ""))
    bankroll = summarize_bankroll(BASE_DIR, ledger_date, policy_engine=policy_engine)
    bankroll_yen = max(0, int(bankroll.get("available_bankroll_yen", 0) or 0))
    multi_predictor = build_multi_predictor_context(scope_key, run_id, run_row, name_to_no_map, win_odds_map, place_odds_map)
    payload = {
        "race_id": str((run_row or {}).get("race_id", "") or ""),
        "scope_key": str(scope_key or ""),
        "field_size": len(predictions),
        "ai": {
            "gap": round(
                max(
                    0.0,
                    float(predictions[0].get("top3_prob_model", 0.0) or 0.0)
                    - float(predictions[1].get("top3_prob_model", 0.0) or 0.0 if len(predictions) > 1 else 0.0),
                ),
                6,
            ),
            "confidence_score": round(float(predictions[0].get("confidence_score", 0.5) or 0.5), 6),
            "stability_score": round(float(predictions[0].get("stability_score", 0.5) or 0.5), 6),
            "risk_score": round(float(predictions[0].get("risk_score", 0.5) or 0.5), 6),
        },
        "marks_top5": [
            {
                "horse_no": str(item.get("horse_no", "") or ""),
                "horse_name": str(item.get("horse_name", "") or ""),
                "pred_rank": int(item.get("pred_rank", 0) or 0),
                "top3_prob_model": round(float(item.get("top3_prob_model", 0.0) or 0.0), 6),
                "rank_score_norm": round(float(item.get("rank_score_norm", 0.0) or 0.0), 6),
            }
            for item in predictions[:5]
        ],
        "predictions": [
            {
                "horse_no": str(item.get("horse_no", "") or ""),
                "horse_name": str(item.get("horse_name", "") or ""),
                "pred_rank": int(item.get("pred_rank", 0) or 0),
                "top3_prob_model": round(float(item.get("top3_prob_model", 0.0) or 0.0), 6),
                "rank_score_norm": round(float(item.get("rank_score_norm", 0.0) or 0.0), 6),
                "win_odds": round(float(item.get("win_odds", 0.0) or 0.0), 6),
                "place_odds": round(float(item.get("place_odds", 0.0) or 0.0), 6),
            }
            for item in predictions[:10]
        ],
        "predictions_full": [
            {
                **dict(item.get("source_row", {}) or {}),
                "horse_no": str(item.get("horse_no", "") or ""),
                "HorseName": str(item.get("horse_name", "") or ""),
                "pred_rank": int(item.get("pred_rank", 0) or 0),
                "Top3Prob_model": round(float(item.get("top3_prob_model", 0.0) or 0.0), 6),
                "rank_score_norm": round(float(item.get("rank_score_norm", 0.0) or 0.0), 6),
                "win_odds": round(float(item.get("win_odds", 0.0) or 0.0), 6),
                "place_odds": round(float(item.get("place_odds", 0.0) or 0.0), 6),
            }
            for item in predictions
        ],
        "pair_odds_top": build_pair_odds_top(candidate_lookup),
        "odds_full": build_odds_full(
            load_csv_rows_flexible(odds_path),
            load_csv_rows_flexible(fuku_odds_path),
            load_csv_rows_flexible(wide_odds_path),
            load_csv_rows_flexible(quinella_odds_path),
            load_csv_rows_flexible(exacta_odds_path),
            load_csv_rows_flexible(trio_odds_path),
            load_csv_rows_flexible(trifecta_odds_path),
        ),
        "prediction_field_guide": build_prediction_field_guide(),
        "multi_predictor": multi_predictor,
        "portfolio_history": build_history_context(BASE_DIR, ledger_date, lookback_days=14, recent_ticket_limit=8, policy_engine=policy_engine),
        "candidates": candidates,
        "constraints": {
            "bankroll_yen": bankroll_yen,
            "race_budget_yen": bankroll_yen,
            "max_tickets_per_race": min(8, max(1, len(allowed_types) * 2)),
            "high_odds_threshold": 12.0,
            "allowed_types": allowed_types,
        },
    }
    return {
        "input": payload,
        "predictions": predictions,
        "candidate_lookup": candidate_lookup,
        "horse_map": horse_map,
        "summary_before": bankroll,
        "ledger_date": ledger_date,
    }, ""


def apply_local_ticket_plan_fallback(output_dict, candidate_lookup, race_budget_yen):
    payload = dict(output_dict or {})
    ticket_plan = list(payload.get("ticket_plan", []) or [])
    if ticket_plan or str(payload.get("bet_decision", "") or "") != "bet":
        return payload
    enabled_types = {str(item or "").strip() for item in list(payload.get("enabled_bet_types", []) or []) if str(item or "").strip()}
    key_horses = {
        str(item or "").strip()
        for item in list(payload.get("key_horses", []) or []) + list(payload.get("secondary_horses", []) or [])
        if str(item or "").strip()
    }
    ranked = []
    for candidate in list(candidate_lookup.values()):
        bet_type = str(candidate.get("bet_type", "") or "").strip()
        legs = [str(x or "").strip() for x in list(candidate.get("legs", []) or []) if str(x or "").strip()]
        if enabled_types and bet_type not in enabled_types:
            continue
        priority = 1 if key_horses and any(leg in key_horses for leg in legs) else 0
        ranked.append(
            (
                -priority,
                -float(candidate.get("score", 0.0) or 0.0),
                -float(candidate.get("ev", 0.0) or 0.0),
                str(candidate.get("id", "") or ""),
                candidate,
            )
        )
    ranked.sort()
    max_tickets = max(1, int(payload.get("max_ticket_count", 1) or 1))
    budget_cap = max(0, int(race_budget_yen or 0))
    remaining = max(0, (budget_cap // 100) * 100)
    selected = [item[-1] for item in ranked[:max_tickets]]
    fallback_plan = []
    stake_template = [500, 300, 200, 100]
    for idx, candidate in enumerate(selected):
        slots_left = len(selected) - idx
        if remaining < 100:
            break
        template_stake = stake_template[idx] if idx < len(stake_template) else 100
        reserve_floor = max(0, (slots_left - 1) * 100)
        stake = min(template_stake, remaining - reserve_floor)
        stake = max(100, (stake // 100) * 100)
        if stake > remaining:
            stake = max(100, (remaining // 100) * 100)
        if stake < 100:
            continue
        remaining -= stake
        fallback_plan.append({"id": str(candidate.get("id", "") or ""), "stake_yen": int(stake)})
    if fallback_plan:
        payload["ticket_plan"] = fallback_plan
        if not list(payload.get("pick_ids", []) or []):
            payload["pick_ids"] = [str(item["id"]) for item in fallback_plan]
        warnings = [str(item or "").strip() for item in list(payload.get("warnings", []) or []) if str(item or "").strip()]
        if "LOCAL_TICKET_PLAN_FALLBACK" not in warnings:
            warnings.append("LOCAL_TICKET_PLAN_FALLBACK")
        payload["warnings"] = warnings
    return payload


def build_policy_ticket_rows(policy_output, candidate_lookup, horse_map, policy_engine):
    output_dict = dict(policy_output or {})
    tickets = []
    for item in list(output_dict.get("ticket_plan", []) or []):
        candidate_id = str(item.get("id", "") or "").strip()
        stake_yen = max(0, int(item.get("stake_yen", 0) or 0))
        if not candidate_id or stake_yen <= 0:
            continue
        candidate = dict(candidate_lookup.get(candidate_id, {}) or {})
        if not candidate:
            continue
        legs = [str(x or "").strip() for x in list(candidate.get("legs", []) or []) if str(x or "").strip()]
        horse_names = []
        for leg in legs:
            horse = dict(horse_map.get(leg, {}) or {})
            horse_names.append(str(horse.get("horse_name", "") or leg))
        bet_type = str(candidate.get("bet_type", "") or "")
        odds_used = round(float(candidate.get("odds_used", 0.0) or 0.0), 6)
        p_hit = round(float(candidate.get("p_hit", 0.0) or 0.0), 6)
        ev = round(float(candidate.get("ev", 0.0) or 0.0), 6)
        expected_return_yen = int(round(stake_yen * max(0.0, p_hit * odds_used)))
        tickets.append(
            {
                "ticket_id": candidate_id,
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
                "ev_ratio_est": round(p_hit * odds_used, 6),
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


def save_policy_payload(scope_key, run_id, race_id, payload, policy_engine):
    scope_norm = normalize_scope_key(scope_key)
    if not scope_norm:
        return None
    race_dir = get_data_dir(BASE_DIR, scope_norm) / str(race_id or "")
    race_dir.mkdir(parents=True, exist_ok=True)
    engine = normalize_policy_engine(policy_engine)
    path = race_dir / f"{engine}_policy_{run_id}_{race_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def execute_policy_buy(scope_key, run_row, run_id, policy_engine="gemini", policy_model=""):
    scope_norm = normalize_scope_key(scope_key)
    engine = normalize_policy_engine(policy_engine)
    resolved_model = resolve_policy_model(engine, policy_model, DEFAULT_GEMINI_MODEL)
    pred_path = resolve_pred_path(scope_norm, run_id, run_row)
    odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "odds_path", "odds")
    fuku_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "fuku_odds_path", "fuku_odds")
    wide_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "wide_odds_path", "wide_odds")
    quinella_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "quinella_odds_path", "quinella_odds")
    exacta_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "exacta_odds_path", "exacta_odds")
    trio_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "trio_odds_path", "trio_odds")
    trifecta_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "trifecta_odds_path", "trifecta_odds")
    context, error = build_policy_input_payload(
        scope_norm,
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
        engine,
    )
    if error:
        raise ValueError(error)
    policy_output = call_policy(
        input=context["input"],
        policy_engine=engine,
        model=resolved_model,
        timeout_s=resolve_policy_timeout(engine),
        cache_enable=True,
    )
    meta = get_last_call_meta()
    output_dict = policy_output.model_dump() if hasattr(policy_output, "model_dump") else policy_output.dict()
    output_dict = apply_local_ticket_plan_fallback(
        output_dict,
        context["candidate_lookup"],
        context["input"]["constraints"]["race_budget_yen"],
    )
    tickets = build_policy_ticket_rows(output_dict, context["candidate_lookup"], context["horse_map"], engine)
    reserve_run_tickets(
        BASE_DIR,
        run_id=run_id,
        scope_key=scope_norm,
        race_id=str((run_row or {}).get("race_id", "") or ""),
        ledger_date=context["ledger_date"],
        tickets=tickets,
        policy_engine=engine,
    )
    summary_after = summarize_bankroll(BASE_DIR, context["ledger_date"], policy_engine=engine)
    payload = {
        "scope": scope_norm,
        "race_id": str((run_row or {}).get("race_id", "") or ""),
        "run_id": str(run_id or ""),
        "policy_engine": engine,
        "policy_model": resolved_model,
        "gemini_model": resolved_model if engine == "gemini" else "",
        "policy_budget_reuse": False,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "budgets": [
            {
                "budget_yen": 0,
                "shared_policy": True,
                "output": output_dict,
                "meta": meta,
                "portfolio": {
                    "ledger_date": context["ledger_date"],
                    "before": context["summary_before"],
                    "after": summary_after,
                },
                "tickets": tickets,
            }
        ],
    }
    path = save_policy_payload(scope_norm, run_id, (run_row or {}).get("race_id", ""), payload, engine)
    updates = {f"{engine}_policy_path": str(path or "")}
    if engine == "gemini":
        updates["gemini_policy_path"] = str(path or "")
        updates["tickets"] = str(len(tickets))
        updates["amount_yen"] = str(sum(int(ticket.get("amount_yen", 0) or 0) for ticket in tickets))
    elif engine == "openai":
        updates["openai_policy_path"] = str(path or "")
    elif engine == "grok":
        updates["grok_policy_path"] = str(path or "")
    update_run_row_fields(scope_norm, run_row, updates)
    output_lines = [
        f"[llm_buy] run_id={run_id} engine={engine} model={resolved_model}",
        f"[policy_save] path={path}" if path else "[policy_save] path=",
        (
            "[tickets] count={count} amount_yen={amount} decision={decision} buy_style={buy_style}".format(
                count=len(tickets),
                amount=sum(int(ticket.get("amount_yen", 0) or 0) for ticket in tickets),
                decision=str(output_dict.get("bet_decision", "") or ""),
                buy_style=str(output_dict.get("buy_style", "") or ""),
            )
        ),
        (
            "[policy_meta] cache_hit={cache_hit} llm_latency_ms={latency} fallback_reason={fallback} error_detail={detail}".format(
                cache_hit=int(bool(meta.get("cache_hit", False))),
                latency=int(meta.get("llm_latency_ms", 0) or 0),
                fallback=str(meta.get("fallback_reason", "") or ""),
                detail=str(meta.get("error_detail", "") or ""),
            )
        ),
    ]
    for ticket in tickets:
        output_lines.append(
            "[ticket] {bet_type} {horse_no} {horse_name} stake={stake_yen} odds={odds_used} p_hit={p_hit}".format(
                bet_type=str(ticket.get("bet_type", "") or ""),
                horse_no=str(ticket.get("horse_no", "") or ""),
                horse_name=str(ticket.get("horse_name", "") or ""),
                stake_yen=int(ticket.get("stake_yen", 0) or 0),
                odds_used=str(ticket.get("odds_used", "") or ""),
                p_hit=str(ticket.get("p_hit", "") or ""),
            )
        )
    return {
        "engine": engine,
        "model": resolved_model,
        "summary_before": context["summary_before"],
        "summary_after": summary_after,
        "payload_path": str(path or ""),
        "tickets": tickets,
        "meta": meta,
        "output_text": "\n".join(line for line in output_lines if str(line).strip()),
    }


def resolve_run_selection(scope_key, run_id):
    scope_norm = normalize_scope_key(scope_key)
    run_text = str(run_id or "").strip()
    run_row = None
    if not scope_norm:
        scope_norm, run_row = infer_scope_and_run(run_text)
    if run_row is None and scope_norm:
        run_row = resolve_run(run_text, scope_norm)
    if run_row is None and scope_norm:
        race_id = normalize_race_id(run_text)
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_norm)
    resolved_run_id = str((run_row or {}).get("run_id", "") or "").strip() or run_text
    return scope_norm, run_row, resolved_run_id


def maybe_refresh_run_odds(scope_norm, run_row, run_id, refresh_enabled):
    if not refresh_enabled:
        return True, "odds refresh skipped.", []
    if run_row is None or not scope_norm:
        return False, "Run row missing for odds update.", []
    odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "odds_path", "odds")
    wide_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "wide_odds_path", "wide_odds")
    fuku_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "fuku_odds_path", "fuku_odds")
    quinella_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "quinella_odds_path", "quinella_odds")
    exacta_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "exacta_odds_path", "exacta_odds")
    trio_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "trio_odds_path", "trio_odds")
    trifecta_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "trifecta_odds_path", "trifecta_odds")
    return refresh_odds_for_run(
        run_row,
        scope_norm,
        odds_path,
        wide_odds_path=wide_odds_path,
        fuku_odds_path=fuku_odds_path,
        quinella_odds_path=quinella_odds_path,
        exacta_odds_path=exacta_odds_path,
        trio_odds_path=trio_odds_path,
        trifecta_odds_path=trifecta_odds_path,
    )


def resolve_policy_timeout(policy_engine):
    engine = normalize_policy_engine(policy_engine)
    env_keys = []
    default_timeout = 20
    if engine == "siliconflow":
        env_keys = ["SILICONFLOW_POLICY_TIMEOUT", "POLICY_TIMEOUT_SILICONFLOW", "POLICY_TIMEOUT"]
        default_timeout = 75
    elif engine == "openai":
        env_keys = ["OPENAI_POLICY_TIMEOUT", "POLICY_TIMEOUT_OPENAI", "POLICY_TIMEOUT"]
        default_timeout = 90
    elif engine == "grok":
        env_keys = ["GROK_POLICY_TIMEOUT", "POLICY_TIMEOUT_GROK", "POLICY_TIMEOUT"]
        default_timeout = 120
    elif engine == "gemini":
        env_keys = ["GEMINI_POLICY_TIMEOUT", "POLICY_TIMEOUT_GEMINI", "POLICY_TIMEOUT"]
        default_timeout = 60
    for key in env_keys:
        raw = str(os.environ.get(key, "") or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return default_timeout


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
    candidates = [
        ("policy_path", "policy"),
        ("gemini_policy_path", "gemini_policy"),
        ("siliconflow_policy_path", "siliconflow_policy"),
        ("openai_policy_path", "openai_policy"),
        ("grok_policy_path", "grok_policy"),
    ]
    for field_name, prefix in candidates:
        path = resolve_run_asset_path(
            scope_key,
            run_id,
            run_row,
            field_name,
            prefix,
            ext=".json",
        )
        payload = load_json_file(path)
        if payload:
            return payload
    return {}


def load_policy_payloads(scope_key, run_id, run_row=None):
    payloads = []
    seen = set()
    candidates = [
        ("gemini", "gemini_policy_path", "gemini_policy"),
        ("siliconflow", "siliconflow_policy_path", "siliconflow_policy"),
        ("openai", "openai_policy_path", "openai_policy"),
        ("grok", "grok_policy_path", "grok_policy"),
        ("", "policy_path", "policy"),
    ]
    for default_engine, field_name, prefix in candidates:
        path = resolve_run_asset_path(
            scope_key,
            run_id,
            run_row,
            field_name,
            prefix,
            ext=".json",
        )
        payload = load_json_file(path)
        if not payload:
            continue
        item = dict(payload)
        policy_engine = normalize_policy_engine(item.get("policy_engine", "") or default_engine or "gemini")
        item["policy_engine"] = policy_engine
        key = (
            policy_engine,
            str(item.get("run_id", "") or run_id or ""),
            str(item.get("saved_at", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        payloads.append(item)
    return payloads


def load_gemini_policy_payload(scope_key, run_id, run_row=None):
    return load_policy_payload(scope_key, run_id, run_row)


def _policy_chip_row(label, values, tone=""):
    chips = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        tone_class = f" policy-chip--{tone}" if tone else ""
        chips.append(f'<span class="policy-chip{tone_class}">{html.escape(text)}</span>')
    if not chips:
        chips.append('<span class="policy-chip policy-chip--empty">none</span>')
    return (
        '<div class="policy-horse-group">'
        f'<div class="policy-label">{html.escape(label)}</div>'
        f'<div class="policy-chip-row">{"".join(chips)}</div>'
        "</div>"
    )


def _policy_detail_row(label, value, code=False):
    text = str(value or "").strip()
    if not text:
        return ""
    body = f"<code>{html.escape(text)}</code>" if code else f"<span>{html.escape(text)}</span>"
    return f'<div class="policy-detail-row"><strong>{html.escape(label)}</strong>{body}</div>'


def _policy_mark_html(marks):
    items = []
    for mark in list(marks or []):
        symbol = str(mark.get("symbol", "") or "").strip()
        horse_no = str(mark.get("horse_no", "") or "").strip()
        if not symbol or not horse_no:
            continue
        items.append(
            '<div class="policy-mark-item">'
            f'<span class="policy-mark-symbol">{html.escape(symbol)}</span>'
            f'<span class="policy-mark-horse">{html.escape(horse_no)}</span>'
            "</div>"
        )
    if not items:
        return ""
    return (
        '<section class="policy-marks">'
        '<div class="policy-label">Marks</div>'
        f'<div class="policy-mark-row">{"".join(items)}</div>'
        "</section>"
    )


def _policy_ticket_html(rows):
    if not rows:
        return ""
    body = []
    for row in rows:
        status = str(row.get("status", "") or "").strip() or "pending"
        body.append(
            "<tr>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(str(row.get('bet_type', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('horse_no', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('horse_name', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('amount_yen', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('odds_used', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('profit_yen', '') or ''))}</td>"
            "</tr>"
        )
    return (
        '<section class="policy-ticket-block">'
        '<div class="policy-label">Ticket Plan</div>'
        '<div class="table-wrap table-wrap--medium">'
        '<table class="data-table data-table--medium">'
        "<thead><tr><th>status</th><th>bet_type</th><th>horse_no</th><th>horse_name</th><th>amount_yen</th><th>odds_used</th><th>profit_yen</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def build_policy_html(payload):
    if not isinstance(payload, dict) or not payload:
        return ""
    budgets = list(payload.get("budgets", []) or [])
    model = str(payload.get("policy_model", "") or payload.get("gemini_model", "") or "")
    policy_engine = str(payload.get("policy_engine", "") or "")
    engine_label_map = {
        "gemini": "Gemini",
        "siliconflow": "DeepSeek",
        "openai": "OpenAI GPT-5",
        "grok": "xAI Grok",
    }
    panel_title = engine_label_map.get(policy_engine, policy_engine or "LLM")
    header_tags = []
    if model:
        header_tags.append(f'<span class="policy-meta-tag">{html.escape(model)}</span>')
    if policy_engine:
        header_tags.append(f'<span class="policy-meta-tag">{html.escape(policy_engine)}</span>')
    budget_sections = []
    for item in budgets:
        if not isinstance(item, dict):
            continue
        is_shared = bool(item.get("shared_policy", False))
        budget = int(to_float(item.get("budget_yen")))
        output = dict(item.get("output", {}) or {})
        meta = dict(item.get("meta", {}) or {})
        header = "[shared]" if is_shared else f"[{budget}]"
        decision = str(output.get("bet_decision", "") or "")
        participation = str(output.get("participation_level", "") or "")
        buy_style = str(output.get("buy_style", "") or "")
        strategy_mode = str(output.get("strategy_mode", "") or "")
        portfolio = dict(item.get("portfolio", {}) or {})
        portfolio_before = dict(portfolio.get("before", {}) or {})
        portfolio_after = dict(portfolio.get("after", {}) or {})
        summary_tags = "".join(
            f'<span class="policy-meta-tag">{html.escape(tag)}</span>'
            for tag in [header, decision, participation, buy_style]
            if str(tag or "").strip()
        )
        bankroll_cards = ""
        if portfolio_before or portfolio_after:
            bankroll_cards = (
                '<div class="policy-bankroll-grid">'
                '<article class="policy-text-card">'
                '<div class="policy-label">Available Before</div>'
                f"<p>{html.escape(str(portfolio_before.get('available_bankroll_yen', '')))} JPY</p>"
                "</article>"
                '<article class="policy-text-card">'
                '<div class="policy-label">Available After</div>'
                f"<p>{html.escape(str(portfolio_after.get('available_bankroll_yen', '')))} JPY</p>"
                "</article>"
                '<article class="policy-text-card">'
                '<div class="policy-label">Realized P/L Today</div>'
                f"<p>{html.escape(str(portfolio_after.get('realized_profit_yen', portfolio_before.get('realized_profit_yen', ''))))} JPY</p>"
                "</article>"
                '<article class="policy-text-card">'
                '<div class="policy-label">Pending Tickets Before</div>'
                f"<p>{html.escape(str(portfolio_before.get('pending_tickets', '')))}</p>"
                "</article>"
                '<article class="policy-text-card">'
                '<div class="policy-label">Pending Tickets After</div>'
                f"<p>{html.escape(str(portfolio_after.get('pending_tickets', '')))}</p>"
                "</article>"
                "</div>"
            )
        horse_groups = "".join(
            [
                _policy_chip_row("Key Horses", list(output.get("key_horses", []) or []), "key"),
                _policy_chip_row("Secondary", list(output.get("secondary_horses", []) or []), "secondary"),
                _policy_chip_row("Longshot", list(output.get("longshot_horses", []) or []), "longshot"),
            ]
        )
        mark_block = _policy_mark_html(list(output.get("marks", []) or []))
        ticket_block = _policy_ticket_html(list(item.get("tickets", []) or []))
        strategy_text = str(output.get("strategy_text_ja", "") or "").strip()
        tendency = str(output.get("bet_tendency_ja", "") or "").strip()
        text_cards = ""
        fallback_reason = str(meta.get("fallback_reason", "") or "").strip()
        error_detail = str(meta.get("error_detail", "") or "").strip()
        if strategy_text or tendency or fallback_reason or error_detail:
            if fallback_reason:
                error_lines = [fallback_reason]
                if error_detail:
                    error_lines.append(error_detail)
                for warn in list(output.get("warnings", []) or []):
                    text = str(warn or "").strip()
                    if text and text not in error_lines:
                        error_lines.append(text)
                error_html = (
                    '<article class="policy-text-card policy-text-card--primary">'
                    '<div class="policy-label">Error</div>'
                    f"<p>{html.escape(' | '.join(error_lines))}</p>"
                    "</article>"
                )
                text_cards = f'<div class="policy-text-grid">{error_html}</div>'
            else:
                strategy_html = (
                    '<article class="policy-text-card policy-text-card--primary">'
                    '<div class="policy-label">Strategy</div>'
                    f"<p>{html.escape(strategy_text or 'No strategy text.')}</p>"
                    "</article>"
                )
                tendency_html = (
                    '<article class="policy-text-card">'
                    '<div class="policy-label">Bet Tendency</div>'
                    f"<p>{html.escape(tendency or 'No tendency text.')}</p>"
                    "</article>"
                )
                text_cards = f'<div class="policy-text-grid">{strategy_html}{tendency_html}</div>'
        reason_codes = list(output.get("reason_codes", []) or [])
        warnings = list(output.get("warnings", []) or [])
        detail_rows = []
        detail_rows.append(
            _policy_detail_row(
                "enabled_bet_types",
                json.dumps(output.get("enabled_bet_types", []), ensure_ascii=False),
                code=True,
            )
        )
        detail_rows.append(
            _policy_detail_row(
                "max_ticket_count / risk_tilt",
                f"{int(to_float(output.get('max_ticket_count')))} / {output.get('risk_tilt', '')}",
                code=True,
            )
        )
        detail_rows.append(
            _policy_detail_row("pick_ids", json.dumps(output.get("pick_ids", []), ensure_ascii=False), code=True)
        )
        detail_rows.append(_policy_detail_row("strategy_mode", strategy_mode, code=True))
        if reason_codes:
            detail_rows.append(_policy_detail_row("reason_codes", ", ".join(str(x) for x in reason_codes), code=True))
        if warnings:
            detail_rows.append(_policy_detail_row("warnings", ", ".join(str(x) for x in warnings), code=True))
        detail_rows.append(
            _policy_detail_row(
                "meta",
                (
                    "cache_hit={cache_hit} llm_latency_ms={llm_latency_ms} fallback_reason={fallback_reason} "
                    "error_detail={error_detail} requested_budget_yen={requested_budget_yen} "
                    "requested_race_budget_yen={requested_race_budget_yen} "
                    "reused={reused} source_budget_yen={source_budget_yen} policy_version={policy_version}"
                ).format(
                    cache_hit=int(bool(meta.get("cache_hit", False))),
                    llm_latency_ms=int(meta.get("llm_latency_ms", 0) or 0),
                    fallback_reason=str(meta.get("fallback_reason", "") or ""),
                    error_detail=str(meta.get("error_detail", "") or ""),
                    requested_budget_yen=int(meta.get("requested_budget_yen", 0) or 0),
                    requested_race_budget_yen=int(meta.get("requested_race_budget_yen", 0) or 0),
                    reused=int(bool(meta.get("reused", False))),
                    source_budget_yen=int(meta.get("source_budget_yen", 0) or 0),
                    policy_version=str(meta.get("policy_version", "") or ""),
                ),
                code=True,
            )
        )
        detail_html = "".join(row for row in detail_rows if row)
        budget_sections.append(
            f"""
            <section class="policy-block">
              <div class="policy-summary">
                <div class="policy-summary-tags">{summary_tags}</div>
                <div class="policy-summary-line">{html.escape(f"strategy_mode={strategy_mode}" if strategy_mode else "")}</div>
              </div>
              {bankroll_cards}
              <div class="policy-horse-grid">{horse_groups}</div>
              {mark_block}
              {text_cards}
              {ticket_block}
              <details class="policy-fold">
                <summary>Other Fields</summary>
                <div class="policy-detail-grid">
                  {detail_html}
                </div>
              </details>
            </section>
            """
        )
    if not budget_sections:
        return ""
    return (
        '<section class="panel policy-panel">'
        '<div class="panel-title-row">'
        f'<div><div class="eyebrow">LLM</div><h2>{html.escape(panel_title)} Policy</h2></div>'
        f'<div class="policy-meta-row">{"".join(header_tags)}</div>'
        "</div>"
        f'{"".join(budget_sections)}'
        "</section>"
    )


def build_gemini_policy_html(payload):
    return build_policy_html(payload)


def build_policy_workspace_html(payloads):
    blocks = []
    for payload in list(payloads or []):
        block = build_policy_html(payload)
        if block:
            blocks.append(block)
    return "".join(blocks)


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
    llm_compare_html="",
    gemini_policy_html="",
    summary_table_html="",
    stats_block="",
    default_scope="central_dirt",
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
        llm_compare_html=llm_compare_html,
        gemini_policy_html=gemini_policy_html,
        summary_table_html=summary_table_html,
        stats_block=stats_block,
        default_scope=default_scope,
        default_policy_engine=normalize_policy_engine(os.environ.get("POLICY_ENGINE", "gemini") or "gemini"),
        default_policy_model=resolve_policy_model(
            normalize_policy_engine(os.environ.get("POLICY_ENGINE", "gemini") or "gemini"),
            os.environ.get("POLICY_MODEL", ""),
            os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        ),
    )


def render_page(
    scope_key="central_dirt",
    output_text="",
    error_text="",
    top5_text="",
    summary_run_id="",
    selected_run_id="",
):
    scope_norm = normalize_scope_key(scope_key)
    default_scope = scope_norm or "central_dirt"
    run_options = build_run_options(scope_norm or scope_key)
    selected_run_id = str(selected_run_id or "").strip()
    llm_compare_html = ui_build_llm_compare_block(
        build_table_html=build_table_html,
        load_policy_bankroll_summary=load_policy_bankroll_summary,
        load_policy_daily_profit_summary=load_policy_daily_profit_summary,
    )
    stats_block = ui_build_stats_block(
        scope_norm,
        load_predictor_summary=load_predictor_summary,
        build_table_html=build_table_html,
        load_policy_bankroll_summary=load_policy_bankroll_summary,
        load_policy_daily_profit_summary=load_policy_daily_profit_summary,
        build_daily_profit_chart_html=ui_build_daily_profit_chart_html,
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
    current_race_id = ""
    if run_row:
        current_race_id = normalize_race_id(run_row.get("race_id", ""))
    if not current_race_id:
        race_candidate = re.sub(r"\D", "", selected_run_id or summary_run_id or "")
        if re.fullmatch(r"\d{12}", race_candidate):
            current_race_id = race_candidate
    view_run_options = build_run_options(scope_norm or scope_key, view_selected_run_id)
    top5_table_html = ""
    mark_table_html = ""
    mark_note_text = ""
    gemini_policy_html = ""
    summary_table_html = ""
    if run_id:
        predictor_top_sections = []
        predictor_mark_sections = []
        predictor_summary_sections = []
        predictor_note_texts = []
        bet_engine_v3_summary = load_bet_engine_v3_cfg_summary(scope_norm or scope_key, run_id)
        policy_payloads = []
        for payload in load_policy_payloads(scope_norm or scope_key, run_id, run_row):
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
        gemini_policy_html = build_policy_workspace_html(policy_payloads)
        for spec, pred_path in resolve_predictor_paths(scope_norm or scope_key, run_id, run_row):
            if not pred_path or not pred_path.exists():
                continue
            predictor_run_row = dict(run_row or {})
            predictor_run_row["predictions_path"] = str(pred_path)
            top_rows, top_cols = load_top5_table(scope_norm or scope_key, run_id, predictor_run_row)
            if top_rows:
                predictor_top_sections.append(
                    build_table_html(top_rows, top_cols, f"Top5 Predictions - {spec['label']}")
                )
            ability_rows, ability_cols = load_ability_marks_table(scope_norm or scope_key, run_id, predictor_run_row)
            if ability_rows:
                predictor_mark_sections.append(
                    build_table_html(ability_rows, ability_cols, f"Ability Marks - {spec['label']}")
                )
            else:
                mark_rows, mark_cols = load_mark_recommendation_table(scope_norm or scope_key, run_id, predictor_run_row)
                ability_rows = mark_rows
                if mark_rows:
                    predictor_mark_sections.append(
                        build_table_html(mark_rows, mark_cols, f"Integrated Marks - {spec['label']}")
                    )
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
            summary_rows = load_prediction_summary(scope_norm or scope_key, run_id, predictor_run_row)
            if summary_rows:
                predictor_summary_sections.append(
                    build_table_html(summary_rows, ["metric", "value"], f"Model Status - {spec['label']}")
                )
        if predictor_top_sections:
            top5_table_html = "".join(predictor_top_sections)
        if predictor_mark_sections:
            mark_table_html = "".join(predictor_mark_sections)
        if predictor_note_texts:
            mark_note_text = "\n\n".join(predictor_note_texts)
        if predictor_summary_sections:
            summary_table_html = "".join(predictor_summary_sections)
    return page_template(
        output_text=output_text,
        error_text=error_text,
        run_options=run_options,
        view_run_options=view_run_options,
        view_selected_run_id=view_selected_run_id,
        current_race_id=current_race_id,
        top5_text=top5_text if not top5_table_html else "",
        top5_table_html=top5_table_html,
        mark_table_html=mark_table_html,
        mark_note_text=mark_note_text,
        llm_compare_html=llm_compare_html,
        gemini_policy_html=gemini_policy_html,
        summary_table_html=summary_table_html,
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
    resolved_run_id = str(run_row.get("run_id", "")).strip()
    if not resolved_run_id:
        resolved_run_id = infer_run_id_from_row(run_row)
        if resolved_run_id:
            update_run_row_fields(scope_key, run_row, {"run_id": resolved_run_id})
            run_row["run_id"] = resolved_run_id
    if not resolved_run_id:
        return render_page(
            scope_key,
            error_text="Run exists but run_id is missing; cannot resolve artifacts.",
            selected_run_id=run_id,
        )
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
    location: str = Form(""),
    race_date: str = Form(""),
    surface: str = Form(""),
    distance: str = Form(""),
    track_cond: str = Form(""),
):
    scope_key = normalize_scope_key(scope_key)
    if not scope_key:
        return render_page("", error_text="Please select a data scope.")
    race_id = normalize_race_id(race_id or race_url)
    history_url = history_url.strip()
    location = str(location or "").strip()
    race_date = str(race_date or "").strip()
    if not race_id or not history_url or not location:
        return render_page(
            scope_key,
            error_text="Race ID, History URL, and Location are required.",
        )
    if scope_key == "local":
        race_url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    else:
        race_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    if not surface.strip():
        if scope_key == "central_turf":
            surface = "1"
        else:
            surface = "2"
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
        location,
        race_date,
        surface,
        distance,
        track_cond,
    ]
    extra_env = {"SCOPE_KEY": scope_key}
    code, output = run_script(
        RUN_PIPELINE,
        inputs=inputs,
        extra_env=extra_env,
    )
    label = f"Exit code: {code}"
    top5_text = extract_top5(output)
    return render_page(
        scope_key,
        output_text=f"{label}\n{output}",
        top5_text=top5_text,
        summary_run_id=parse_run_id(output),
    )


@app.post("/record_predictor", response_class=HTMLResponse)
def record_predictor(
    scope_key: str = Form(""),
    run_id: str = Form(""),
    top1: str = Form(""),
    top2: str = Form(""),
    top3: str = Form(""),
):
    scope_norm = normalize_scope_key(scope_key)
    run_id = str(run_id or "").strip()
    run_row = None
    if not scope_norm:
        scope_norm, run_row = infer_scope_and_run(run_id)
    if run_row is None and scope_norm:
        run_row = resolve_run(run_id, scope_norm)
    if run_row is None and scope_norm:
        race_id = normalize_race_id(run_id)
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_norm)
    resolved_run_id = str((run_row or {}).get("run_id", "") or "").strip() or run_id
    if not top1 or not top2 or not top3:
        return render_page(
            scope_norm or scope_key,
            error_text="Top1/Top2/Top3 are required.",
            selected_run_id=resolved_run_id,
        )
    refresh_ok = False
    refresh_message = "Run row missing for odds update."
    refresh_warnings = []
    if run_row is not None and scope_norm:
        odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "odds_path", "odds")
        wide_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "wide_odds_path", "wide_odds")
        fuku_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "fuku_odds_path", "fuku_odds")
        quinella_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "quinella_odds_path", "quinella_odds")
        exacta_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "exacta_odds_path", "exacta_odds")
        trio_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "trio_odds_path", "trio_odds")
        trifecta_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "trifecta_odds_path", "trifecta_odds")
        refresh_ok, refresh_message, refresh_warnings = refresh_odds_for_run(
            run_row,
            scope_norm,
            odds_path,
            wide_odds_path=wide_odds_path,
            fuku_odds_path=fuku_odds_path,
            quinella_odds_path=quinella_odds_path,
            exacta_odds_path=exacta_odds_path,
            trio_odds_path=trio_odds_path,
            trifecta_odds_path=trifecta_odds_path,
        )
    inputs = [resolved_run_id, top1, top2, top3]
    code, output = run_script(
        RECORD_PREDICTOR,
        inputs=inputs,
        extra_blanks=2,
        extra_env={"SCOPE_KEY": scope_norm or scope_key or "central_dirt"},
    )
    label = f"Exit code: {code}"
    output_parts = [f"[odds_update] status={'ok' if refresh_ok else 'fail'} message={refresh_message or ''}".strip()]
    if refresh_warnings:
        output_parts.append("[odds_update][warnings] " + "; ".join(str(x) for x in refresh_warnings))
    output_parts.append(label)
    output_parts.append(output)
    return render_page(
        scope_norm or scope_key,
        output_text="\n".join(part for part in output_parts if str(part).strip()),
        selected_run_id=resolved_run_id,
    )


@app.post("/run_llm_buy", response_class=HTMLResponse)
def run_llm_buy(
    scope_key: str = Form(""),
    run_id: str = Form(""),
    policy_engine: str = Form("gemini"),
    policy_model: str = Form(""),
    refresh_odds: str = Form("1"),
):
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return render_page(
            scope_norm or scope_key,
            error_text="Run ID / Race ID not found for LLM buy.",
            selected_run_id=resolved_run_id or str(run_id or "").strip(),
        )
    refresh_enabled = str(refresh_odds or "").strip() not in ("", "0", "false", "False", "off")
    refresh_ok, refresh_message, refresh_warnings = maybe_refresh_run_odds(scope_norm, run_row, resolved_run_id, refresh_enabled)
    try:
        result = execute_policy_buy(
            scope_norm,
            run_row,
            resolved_run_id,
            policy_engine=policy_engine,
            policy_model=policy_model,
        )
    except Exception as exc:
        return render_page(
            scope_norm,
            error_text=build_llm_buy_output(
                load_policy_bankroll_summary(resolved_run_id, run_row.get("timestamp", ""), policy_engine=policy_engine),
                refresh_ok,
                refresh_message,
                refresh_warnings,
                f"[llm_buy][error] {exc}",
                policy_engine=normalize_policy_engine(policy_engine),
            ),
            selected_run_id=resolved_run_id,
        )
    return render_page(
        scope_norm,
        output_text=build_llm_buy_output(
            result["summary_before"],
            refresh_ok,
            refresh_message,
            refresh_warnings,
            result["output_text"],
            result["engine"],
        ),
        selected_run_id=resolved_run_id,
    )


@app.post("/run_gemini_buy", response_class=HTMLResponse)
def run_gemini_buy(
    scope_key: str = Form(""),
    run_id: str = Form(""),
    policy_model: str = Form(""),
    refresh_odds: str = Form("1"),
):
    return run_llm_buy(
        scope_key=scope_key,
        run_id=run_id,
        policy_engine="gemini",
        policy_model=policy_model,
        refresh_odds=refresh_odds,
    )


@app.post("/run_all_llm_buy", response_class=HTMLResponse)
def run_all_llm_buy(
    scope_key: str = Form(""),
    run_id: str = Form(""),
    refresh_odds: str = Form("1"),
):
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return render_page(
            scope_norm or scope_key,
            error_text="Run ID / Race ID not found for LLM buy.",
            selected_run_id=resolved_run_id or str(run_id or "").strip(),
        )
    refresh_enabled = str(refresh_odds or "").strip() not in ("", "0", "false", "False", "off")
    refresh_ok, refresh_message, refresh_warnings = maybe_refresh_run_odds(scope_norm, run_row, resolved_run_id, refresh_enabled)
    result_blocks = []
    error_blocks = []
    for engine in ("gemini", "siliconflow", "openai", "grok"):
        try:
            result = execute_policy_buy(scope_norm, run_row, resolved_run_id, policy_engine=engine, policy_model="")
            result_blocks.append(
                build_llm_buy_output(
                    result["summary_before"],
                    refresh_ok,
                    refresh_message,
                    refresh_warnings,
                    result["output_text"],
                    result["engine"],
                )
            )
        except Exception as exc:
            error_blocks.append(f"[llm_buy][{engine}] {exc}")
    if error_blocks and not result_blocks:
        return render_page(
            scope_norm,
            error_text="\n\n".join(error_blocks),
            selected_run_id=resolved_run_id,
        )
    output_text = "\n\n".join(block for block in result_blocks if str(block).strip())
    if error_blocks:
        output_text = "\n\n".join([output_text] + error_blocks if output_text else error_blocks)
    return render_page(
        scope_norm,
        output_text=output_text,
        selected_run_id=resolved_run_id,
    )


@app.post("/reset_llm_state", response_class=HTMLResponse)
def reset_llm_state():
    summary = reset_llm_state_files(BASE_DIR)
    return render_page(
        "",
        output_text=json.dumps(summary, ensure_ascii=False, indent=2),
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

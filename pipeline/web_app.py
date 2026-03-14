import csv
import html
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from predictor_catalog import canonical_predictor_id, list_predictors, predictor_label, resolve_run_prediction_path
from gemini_portfolio import (
    add_bankroll_topup,
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
from race_job_store import (
    apply_job_action as apply_race_job_action,
    create_job as create_race_job,
    load_jobs as load_race_jobs,
    save_artifact as save_race_job_artifact,
    scan_due_jobs as scan_due_race_jobs,
    update_job as update_race_job,
)
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
load_local_env(BASE_DIR, override=False)


def _admin_token_expected():
    return str(os.environ.get("ADMIN_TOKEN", "") or "").strip()


def _admin_token_enabled():
    return bool(_admin_token_expected())


def _admin_token_valid(token=""):
    expected = _admin_token_expected()
    if not expected:
        return True
    supplied = str(token or "").strip()
    return bool(supplied) and secrets.compare_digest(supplied, expected)


def _pick_next_process_job_id():
    jobs = load_race_jobs(BASE_DIR)
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        if status == "queued_process":
            return str(job.get("job_id", "") or "").strip()
    return ""


def _pick_next_settle_job_id():
    jobs = load_race_jobs(BASE_DIR)
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        actual_top1 = str(job.get("actual_top1", "") or "").strip()
        actual_top2 = str(job.get("actual_top2", "") or "").strip()
        actual_top3 = str(job.get("actual_top3", "") or "").strip()
        if status == "queued_settle" and actual_top1 and actual_top2 and actual_top3:
            return str(job.get("job_id", "") or "").strip()
    return ""


def run_due_jobs_once():
    changed = scan_due_race_jobs(BASE_DIR)
    process_results = []
    settle_results = []
    errors = []

    while True:
        job_id = _pick_next_process_job_id()
        if not job_id:
            break
        try:
            from race_job_runner import process_race_job

            process_results.append(process_race_job(BASE_DIR, job_id))
        except Exception as exc:
            try:
                from race_job_runner import fail_race_job

                fail_race_job(BASE_DIR, job_id, str(exc))
            except Exception:
                pass
            errors.append({"kind": "process", "job_id": job_id, "error": str(exc)})

    while True:
        job_id = _pick_next_settle_job_id()
        if not job_id:
            break
        job = next((item for item in load_race_jobs(BASE_DIR) if str(item.get("job_id", "")).strip() == job_id), {})
        actual_top3 = [
            str(job.get("actual_top1", "") or "").strip(),
            str(job.get("actual_top2", "") or "").strip(),
            str(job.get("actual_top3", "") or "").strip(),
        ]
        try:
            from race_job_runner import settle_race_job

            settle_results.append(settle_race_job(BASE_DIR, job_id, actual_top3))
        except Exception as exc:
            try:
                from race_job_runner import fail_race_job

                fail_race_job(BASE_DIR, job_id, str(exc))
            except Exception:
                pass
            errors.append({"kind": "settle", "job_id": job_id, "error": str(exc)})

    return {
        "queued_count": len(changed),
        "queued_job_ids": [str(item.get("job_id", "") or "").strip() for item in changed],
        "processed_count": len(process_results),
        "processed_job_ids": [str(item.get("job_id", "") or "").strip() for item in process_results],
        "settled_count": len(settle_results),
        "settled_job_ids": [str(item.get("job_id", "") or "").strip() for item in settle_results],
        "errors": errors,
    }


def build_console_gate_page(admin_token="", error_text=""):
    error_block = ""
    if error_text:
        error_block = f'<section class="job-flash job-flash--error">{html.escape(error_text)}</section>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>控制台验证</title>
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
      <h1>控制台验证</h1>
      <p>这里是你的后台。输入正确的 `ADMIN_TOKEN` 后才能进入 `/console`。</p>
      <form method="get" action="/console">
        <input type="password" name="token" placeholder="ADMIN_TOKEN" value="{html.escape(admin_token)}">
        <div class="actions">
          <button type="submit">进入控制台</button>
          <a href="/llm_today">返回前台</a>
        </div>
      </form>
    </section>
  </main>
</body>
</html>"""


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
                "ticket_id": row.get("ticket_id", ""),
                "bet_type": row.get("bet_type", ""),
                "horse_no": row.get("horse_nos", ""),
                "horse_name": row.get("horse_names", ""),
                "amount_yen": row.get("stake_yen", ""),
                "odds_used": row.get("odds_used", ""),
                "hit": row.get("hit", ""),
                "payout_yen": row.get("payout_yen", ""),
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
    items = []
    has_explicit_rank = False
    for idx, row in enumerate(list(pred_rows or []), start=1):
        horse_name = str(_pick_first_value(row, ("HorseName", "horse_name", "name"), "") or "").strip()
        if not horse_name:
            continue
        horse_no_raw = _pick_first_value(row, ("horse_no", "HorseNo", "umaban", "馬番"), "")
        horse_no = normalize_horse_no_text(horse_no_raw)
        if not horse_no:
            mapped = name_to_no_map.get(normalize_name(horse_name))
            if mapped is not None:
                horse_no = normalize_horse_no_text(mapped)
        top3_prob = _prediction_prob_value(row)
        explicit_rank = to_int_or_none(_pick_first_value(row, ("pred_rank", "PredRank", "rank", "Rank"), ""))
        if explicit_rank is not None and explicit_rank > 0:
            has_explicit_rank = True
        items.append(
            {
                "horse_no": horse_no,
                "horse_name": horse_name,
                "pred_rank": explicit_rank if explicit_rank is not None and explicit_rank > 0 else int(idx),
                "top3_prob_model": top3_prob,
                "confidence_score": max(0.0, to_float(row.get("confidence_score"))),
                "stability_score": max(0.0, to_float(row.get("stability_score"))),
                "risk_score": max(0.0, to_float(row.get("risk_score"))),
                "_raw_rank_score": _prediction_score_value(row),
                "_input_order": int(idx),
                "source_row": dict(row),
            }
        )
    if not items:
        return []
    if has_explicit_rank:
        items.sort(
            key=lambda item: (
                int(item.get("pred_rank", 9999) or 9999),
                -float(item.get("top3_prob_model", 0.0) or 0.0),
                str(item.get("horse_no", "")),
                str(item.get("horse_name", "")),
            )
        )
    else:
        items.sort(
            key=lambda item: (
                -float(item.get("top3_prob_model", 0.0) or 0.0),
                -float(item.get("_raw_rank_score", 0.0) or 0.0),
                str(item.get("horse_no", "")),
                str(item.get("horse_name", "")),
                int(item.get("_input_order", 9999) or 9999),
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
    candidates = []
    candidate_lookup = {}
    horse_map = {str(item.get("horse_no", "") or ""): item for item in list(predictions or []) if str(item.get("horse_no", "") or "").strip()}
    bet_type_order = {"win": 0, "place": 1, "wide": 2, "quinella": 3, "exacta": 4, "trio": 5, "trifecta": 6}
    for item in predictions:
        horse_no = str(item.get("horse_no", "") or "").strip()
        if not horse_no:
            continue
        top3_prob = max(0.0, float(item.get("top3_prob_model", 0.0) or 0.0))
        if "win" in allowed_types:
            win_odds = float(item.get("win_odds", 0.0) or 0.0)
            if win_odds > 0:
                candidate = {
                    "id": f"win:{horse_no}",
                    "bet_type": "win",
                    "legs": [horse_no],
                    "odds_used": round(win_odds, 6),
                    "p_hit": 0.0,
                    "ev": 0.0,
                    "score": 0.0,
                }
                candidates.append(candidate)
                candidate_lookup[candidate["id"]] = candidate
        if "place" in allowed_types:
            place_odds = float(item.get("place_odds", 0.0) or 0.0)
            if place_odds > 0:
                candidate = {
                    "id": f"place:{horse_no}",
                    "bet_type": "place",
                    "legs": [horse_no],
                    "odds_used": round(place_odds, 6),
                    "p_hit": 0.0,
                    "ev": 0.0,
                    "score": 0.0,
                }
                candidates.append(candidate)
                candidate_lookup[candidate["id"]] = candidate
    if "wide" in allowed_types:
        for pair, odds in sorted(wide_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {
                "id": f"wide:{pair[0]}-{pair[1]}",
                "bet_type": "wide",
                "legs": [str(pair[0]), str(pair[1])],
                "odds_used": round(float(odds), 6),
                "p_hit": 0.0,
                "ev": 0.0,
                "score": 0.0,
            }
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    if "quinella" in allowed_types:
        for pair, odds in sorted(quinella_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {
                "id": f"quinella:{pair[0]}-{pair[1]}",
                "bet_type": "quinella",
                "legs": [str(pair[0]), str(pair[1])],
                "odds_used": round(float(odds), 6),
                "p_hit": 0.0,
                "ev": 0.0,
                "score": 0.0,
            }
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    if "exacta" in allowed_types:
        for pair, odds in sorted(exacta_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {
                "id": f"exacta:{pair[0]}-{pair[1]}",
                "bet_type": "exacta",
                "legs": [str(pair[0]), str(pair[1])],
                "odds_used": round(float(odds), 6),
                "p_hit": 0.0,
                "ev": 0.0,
                "score": 0.0,
            }
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    if "trio" in allowed_types:
        for trio_key, odds in sorted(trio_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {
                "id": f"trio:{trio_key[0]}-{trio_key[1]}-{trio_key[2]}",
                "bet_type": "trio",
                "legs": [str(trio_key[0]), str(trio_key[1]), str(trio_key[2])],
                "odds_used": round(float(odds), 6),
                "p_hit": 0.0,
                "ev": 0.0,
                "score": 0.0,
            }
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    if "trifecta" in allowed_types:
        for trio_key, odds in sorted(trifecta_odds_map.items()):
            if float(odds or 0.0) <= 0:
                continue
            candidate = {
                "id": f"trifecta:{trio_key[0]}-{trio_key[1]}-{trio_key[2]}",
                "bet_type": "trifecta",
                "legs": [str(trio_key[0]), str(trio_key[1]), str(trio_key[2])],
                "odds_used": round(float(odds), 6),
                "p_hit": 0.0,
                "ev": 0.0,
                "score": 0.0,
            }
            candidates.append(candidate)
            candidate_lookup[candidate["id"]] = candidate
    candidates.sort(
        key=lambda item: (
            bet_type_order.get(str(item.get("bet_type", "") or ""), 99),
            float(item.get("odds_used", 0.0) or 0.0),
            str(item.get("id", "")),
        )
    )
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
    rows.sort(key=lambda item: (float(item.get("odds", 0.0) or 0.0), str(item.get("bet_type", "") or ""), str(item.get("pair", "") or "")))
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
        "place": _single_rows(place_rows, "odds_low" if place_rows and "odds_low" in place_rows[0] else "odds_mid"),
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
    return dict(output_dict or {})


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
    out = []
    for ticket in list(ticket_rows or []):
        bet_type = str(ticket.get("bet_type", "") or "").strip().lower()
        legs = [str(x).strip() for x in str(ticket.get("horse_no", "") or "").split("-") if str(x).strip()]
        stake_yen = int(ticket.get("stake_yen", ticket.get("amount_yen", 0)) or 0)
        if not bet_type or not legs or stake_yen <= 0:
            continue
        out.append({"bet_type": bet_type, "legs": legs, "stake_yen": stake_yen})
    return out


def _append_output_warning(output_dict, code):
    warnings = [str(x).strip() for x in list((output_dict or {}).get("warnings", []) or []) if str(x).strip()]
    text = str(code or "").strip()
    if text and text not in warnings:
        warnings.append(text)
    output_dict["warnings"] = warnings
    return output_dict


def _set_output_no_bet(output_dict):
    output_dict["bet_decision"] = "no_bet"
    output_dict["participation_level"] = "no_bet"
    output_dict["buy_style"] = "no_bet"
    output_dict["strategy_mode"] = "no_bet"
    output_dict["enabled_bet_types"] = []
    output_dict["key_horses"] = []
    output_dict["secondary_horses"] = []
    output_dict["longshot_horses"] = []
    output_dict["max_ticket_count"] = 0
    output_dict["ticket_plan"] = []
    return output_dict


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
    requested_ticket_count = len(list(output_dict.get("ticket_plan", []) or []))
    tickets = build_policy_ticket_rows(output_dict, context["candidate_lookup"], context["horse_map"], engine)
    output_dict["ticket_plan"] = _normalize_output_ticket_plan_from_rows(tickets)
    if len(tickets) < requested_ticket_count:
        output_dict = _append_output_warning(output_dict, "INVALID_TICKET_DROPPED")
    if str(output_dict.get("bet_decision", "") or "").strip().lower() == "bet" and not tickets:
        output_dict = _append_output_warning(output_dict, "NO_EXECUTABLE_TICKETS")
        output_dict = _set_output_no_bet(output_dict)
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
        "openai": "ChatGPT",
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


LLM_BATTLE_ORDER = ("openai", "gemini", "siliconflow", "grok")
LLM_BATTLE_LABELS = {
    "openai": "ChatGPT",
    "gemini": "Gemini",
    "siliconflow": "DeepSeek",
    "grok": "Grok",
}
LLM_NOTE_LABELS = {
    "openai": "ChatGPT",
    "gemini": "Gemini",
    "siliconflow": "DeepSeek",
    "grok": "Grok",
}
LLM_BATTLE_SHORT_LABELS = {
    "openai": "chatgpt",
    "gemini": "gemini",
    "siliconflow": "deepseek",
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
        or _safe_text(row.get("siliconflow_policy_path"))
        or _safe_text(row.get("openai_policy_path"))
        or _safe_text(row.get("grok_policy_path"))
    )


def _report_scope_key_for_row(run_row, fallback_scope=""):
    row = dict(run_row or {})
    return normalize_scope_key(row.get("_report_scope_key") or row.get("scope_key") or fallback_scope) or ""


def _load_combined_llm_report_runs():
    runs = []
    for report_scope_key in LLM_REPORT_SCOPE_KEYS:
        for row in load_runs(report_scope_key):
            if not _has_llm_policy_assets(row):
                continue
            item = dict(row)
            item["_report_scope_key"] = report_scope_key
            runs.append(item)
    return runs


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
    path = get_data_dir(BASE_DIR, scope_key) / "predictor_results.csv"
    rows = load_csv_rows(path) if path.exists() else []
    result_map = {}
    for row in rows:
        run_id = _safe_text(row.get("run_id"))
        if not run_id:
            continue
        predictor_id = canonical_predictor_id(row.get("predictor_id"))
        current = result_map.get(run_id)
        item = {
            "actual_top1": _safe_text(row.get("actual_top1")),
            "actual_top2": _safe_text(row.get("actual_top2")),
            "actual_top3": _safe_text(row.get("actual_top3")),
        }
        if current is None or predictor_id == "main":
            result_map[run_id] = item
    return result_map


def _load_name_to_no_map_for_run(scope_key, run_id, run_row):
    odds_path = resolve_run_asset_path(scope_key, run_id, run_row, "odds_path", "odds")
    if not odds_path or not Path(odds_path).exists():
        return {}
    raw = load_name_to_no(odds_path) or {}
    out = {}
    for name, horse_no in raw.items():
        norm_name = normalize_name(name)
        text_no = normalize_horse_no_text(horse_no)
        if norm_name and text_no:
            out[norm_name] = text_no
    return out


def _actual_result_snapshot(scope_key, run_id, run_row, actual_result_map):
    actual = dict((actual_result_map or {}).get(run_id, {}) or {})
    actual_names = [
        _safe_text(actual.get("actual_top1")),
        _safe_text(actual.get("actual_top2")),
        _safe_text(actual.get("actual_top3")),
    ]
    name_to_no = _load_name_to_no_map_for_run(scope_key, run_id, run_row)
    actual_horse_nos = []
    for name in actual_names:
        actual_horse_nos.append(name_to_no.get(normalize_name(name), "") if name else "")
    return {
        "actual_names": actual_names,
        "actual_horse_nos": actual_horse_nos,
    }


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


def _policy_primary_choice(output):
    marks = list((output or {}).get("marks", []) or [])
    symbol_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}
    best = None
    for row in marks:
        symbol = _safe_text(row.get("symbol"))
        horse_no = _safe_text(row.get("horse_no"))
        if not symbol or not horse_no:
            continue
        score = symbol_order.get(symbol, 99)
        if best is None or score < best[0]:
            best = (score, horse_no)
    if best is not None:
        return best[1]
    key_horses = list((output or {}).get("key_horses", []) or [])
    for horse_no in key_horses:
        text = _safe_text(horse_no)
        if text:
            return text
    return ""


def _llm_agreement_summary(outputs):
    top_choices = []
    for output in list(outputs or []):
        horse_no = _policy_primary_choice(output)
        if horse_no:
            top_choices.append(horse_no)
    if not top_choices:
        return {"label": "不明", "score": 0.5}
    counts = {}
    for horse_no in top_choices:
        counts[horse_no] = counts.get(horse_no, 0) + 1
    max_count = max(counts.values()) if counts else 0
    unique_count = len(counts)
    if max_count >= 4:
        return {"label": "高", "score": 1.0}
    if max_count == 3:
        return {"label": "やや高", "score": 0.8}
    if max_count == 2 and unique_count <= 2:
        return {"label": "中", "score": 0.5}
    if max_count == 2:
        return {"label": "やや低", "score": 0.3}
    return {"label": "低", "score": 0.0}


def _difficulty_summary(outputs, agreement_score):
    outputs = list(outputs or [])
    if not outputs:
        return "★★★☆☆"
    score = 2.0
    mixed_count = 0
    low_conf_count = 0
    low_stability_count = 0
    one_shot_count = 0
    strong_favorite_count = 0
    place_focus_count = 0
    risk_total = 0.0
    for output in outputs:
        reason_codes = {str(item or "").strip() for item in list(output.get("reason_codes", []) or []) if str(item or "").strip()}
        if "MIXED_FIELD" in reason_codes:
            mixed_count += 1
        if "LOW_CONFIDENCE" in reason_codes:
            low_conf_count += 1
        if "LOW_STABILITY" in reason_codes:
            low_stability_count += 1
        if "HIGH_ODDS_ONE_SHOT" in reason_codes:
            one_shot_count += 1
        if "STRONG_FAVORITE" in reason_codes:
            strong_favorite_count += 1
        if "PLACE_FOCUS" in reason_codes:
            place_focus_count += 1
        risk_tilt = _safe_text(output.get("risk_tilt")).lower()
        if risk_tilt == "high":
            risk_total += 0.8
        elif risk_tilt == "medium":
            risk_total += 0.4
        elif risk_tilt == "low":
            risk_total += 0.1
    total = float(len(outputs))
    score += (mixed_count / total) * 2.0
    score += (low_conf_count / total) * 1.0
    score += (low_stability_count / total) * 0.8
    score += (one_shot_count / total) * 0.5
    score -= (strong_favorite_count / total) * 1.1
    score -= (place_focus_count / total) * 0.3
    score += risk_total / total
    score += (1.0 - float(agreement_score or 0.0)) * 1.6
    star_count = max(1, min(5, int(round(score))))
    return "★" * star_count + "☆" * (5 - star_count)


def _build_llm_battle_bundle(scope_key, run_id, run_row, payloads, actual_result_map):
    payload_map = {}
    for payload in list(payloads or []):
        engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
        if engine:
            payload_map[engine] = payload
    if not payload_map:
        return {"html": "", "note_text": ""}

    actual_snapshot = _actual_result_snapshot(scope_key, run_id, run_row, actual_result_map)
    actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])
    race_label = _format_race_label(run_row)
    battle_title = _build_battle_title(run_row)
    outputs = []
    cards = []
    premium_lines = ["【AI予想の考え方】", ""]
    note_lines = [battle_title, "", f"レース：{race_label}"]

    for engine in LLM_BATTLE_ORDER:
        payload = payload_map.get(engine)
        if not payload:
            continue
        output = _policy_primary_output(payload)
        outputs.append(output)
        marks_map = _policy_marks_map(payload)
        ticket_rows = load_policy_run_ticket_rows(run_id, policy_engine=engine) or list(
            _policy_primary_budget(payload).get("tickets", []) or []
        )
        ticket_summary = _summarize_ticket_rows(ticket_rows)
        result_triplet = _marks_result_triplet(marks_map, actual_horse_nos)
        marks_text = _format_marks_text(marks_map)
        ticket_plan_text = _format_ticket_plan_text(ticket_rows, output)
        tendency_text = _safe_text(output.get("bet_tendency_ja")) or "未生成"
        strategy_text = _safe_text(output.get("strategy_text_ja")) or "未生成"
        meta_tags = [
            f"状態 {ticket_summary['status']}",
            f"投資 {ticket_summary['stake_yen']}円" if ticket_summary["stake_yen"] else "未購入",
        ]
        if ticket_summary["roi"] != "":
            meta_tags.append(f"ROI {ticket_summary['roi']:.4f}")
        cards.append(
            {
                "engine": engine,
                "label": LLM_BATTLE_LABELS.get(engine, engine),
                "marks_text": marks_text,
                "ticket_plan_text": ticket_plan_text,
                "tendency_text": tendency_text,
                "strategy_text": strategy_text,
                "result_triplet_text": _format_triplet_text(result_triplet),
                "meta_tags": meta_tags,
            }
        )
        premium_lines.extend(
            [
                f"{LLM_NOTE_LABELS.get(engine, LLM_BATTLE_LABELS.get(engine, engine))}",
                strategy_text,
                "",
            ]
        )

    agreement = _llm_agreement_summary(outputs)
    difficulty = _difficulty_summary(outputs, agreement.get("score", 0.5))
    note_lines.extend(
        [
            f"難易度：{difficulty}",
            f"一致度：{agreement.get('label', '不明')}",
            "",
        ]
    )

    card_html = []
    premium_html = []
    for card in cards:
        tag_html = "".join(
            f'<span class="battle-meta-chip">{html.escape(tag)}</span>' for tag in list(card.get("meta_tags", []) or [])
        )
        card_html.append(
            f"""
            <article class="battle-card">
              <div class="battle-card-head">
                <div>
                  <div class="eyebrow">LLM Battle</div>
                  <h3>{html.escape(card['label'])}</h3>
                </div>
                <div class="battle-meta-row">{tag_html}</div>
              </div>
              <div class="battle-mark-band">
                <span class="battle-band-label">印</span>
                <strong>{html.escape(card['marks_text'])}</strong>
              </div>
              <div class="battle-public-grid">
                <section class="battle-copy-card">
                  <div class="policy-label">買い目</div>
                  <p>{html.escape(card['ticket_plan_text'])}</p>
                </section>
                <section class="battle-copy-card">
                  <div class="policy-label">買い目傾向</div>
                  <p>{html.escape(card['tendency_text'])}</p>
                </section>
              </div>
              <div class="battle-result-row">
                <span class="policy-label">着順対応印</span>
                <code>{html.escape(card['result_triplet_text'])}</code>
              </div>
            </article>
            """
        )
        note_lines.extend(
            [
                f"{LLM_NOTE_LABELS.get(card['engine'], card['label'])}",
                f"印：{card['marks_text']}",
                "",
                "買い目",
                "",
                card["ticket_plan_text"],
                "",
                f"買い目傾向：{card['tendency_text']}",
                "",
            ]
        )
        premium_html.append(
            f"""
            <section class="battle-copy-card">
              <div class="policy-label">{html.escape(card['label'])} Strategy</div>
              <p>{html.escape(card['strategy_text'])}</p>
            </section>
            """
        )
    summary_html = (
        '<section class="panel battle-hero-panel">'
        '<div class="panel-title-row">'
        f'<div><div class="eyebrow">Note Layout</div><h2>{html.escape(battle_title)}</h2></div>'
        f'<span class="section-chip">{html.escape(race_label)}</span>'
        "</div>"
        '<div class="battle-overview-grid">'
        f'<section class="battle-overview-card"><span class="battle-band-label">レース</span><strong>{html.escape(race_label)}</strong></section>'
        f'<section class="battle-overview-card"><span class="battle-band-label">難易度</span><strong>{html.escape(difficulty)}</strong></section>'
        f'<section class="battle-overview-card"><span class="battle-band-label">一致度</span><strong>{html.escape(str(agreement.get("label", "不明")))}</strong></section>'
        "</div>"
        '<p class="helper-text">難易度は混戦度が高いほど星が増える想定です。一致度は 4 LLM の本命がどれだけ揃っているかを見ています。</p>'
        f'<div class="battle-grid">{"".join(card_html)}</div>'
        '<section class="panel battle-hero-panel" style="margin-top:14px;">'
        '<div class="panel-title-row">'
        '<div><div class="eyebrow">Paid Block</div><h2>AI予想の考え方</h2></div>'
        '<span class="section-chip">note footer</span>'
        "</div>"
        '<p class="helper-text">このブロックは note 記事の最下部に置く前提で、4 LLM の Strategy をひとまとめにしています。</p>'
        f'<div class="battle-public-grid">{"".join(premium_html)}</div>'
        "</section>"
        "</section>"
    )
    note_text = "\n".join(line for line in note_lines if line is not None).strip()
    premium_text = "\n".join(line for line in premium_lines if line is not None).strip()
    if premium_text:
        note_text = f"{note_text}\n\n{premium_text}".strip()
    return {"html": summary_html, "note_text": note_text}


def _build_llm_daily_report_bundle(scope_key, current_run_row, actual_result_map):
    target_date = _run_date_key(current_run_row)
    if not target_date:
        return {"html": "", "text": ""}

    actual_result_maps = {report_scope_key: _load_actual_result_map(report_scope_key) for report_scope_key in LLM_REPORT_SCOPE_KEYS}
    runs = [row for row in _load_combined_llm_report_runs() if _run_date_key(row) == target_date]
    if not runs:
        return {"html": "", "text": ""}

    runs.sort(key=lambda row: (_safe_text(row.get("race_id")), _safe_text(row.get("timestamp")), _safe_text(row.get("run_id"))))
    metric_rows = {
        engine: {
            "model": LLM_BATTLE_LABELS.get(engine, engine),
            "runs": 0,
            "race_hit_count": 0,
            "ticket_hit_count": 0,
            "ticket_count": 0,
            "honmei_hit_count": 0,
            "mark_hit_count": 0,
            "mark_total_count": 0,
            "stake_yen": 0,
            "payout_yen": 0,
            "profit_yen": 0,
        }
        for engine in LLM_BATTLE_ORDER
    }
    race_rows = []
    report_lines = [f"{target_date} 第1回 いかいもAI競馬対決杯｜ 日報", ""]

    for run_row in runs:
        run_id = _safe_text(run_row.get("run_id"))
        report_scope_key = _report_scope_key_for_row(run_row, scope_key)
        payload_map = {}
        for payload in load_policy_payloads(report_scope_key, run_id, run_row):
            engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
            if engine:
                payload_map[engine] = payload
        if not payload_map:
            continue
        race_run_id = run_id
        if not race_run_id:
            for payload in payload_map.values():
                race_run_id = _payload_run_id(payload, race_run_id)
                if race_run_id:
                    break
        if not race_run_id:
            continue
        actual_snapshot = _actual_result_snapshot(
            report_scope_key,
            race_run_id,
            run_row,
            actual_result_maps.get(report_scope_key, {}),
        )
        actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])
        race_label = _format_race_label(run_row)
        race_row = {"race": race_label}
        report_lines.append(race_label)
        for engine in LLM_BATTLE_ORDER:
            payload = payload_map.get(engine)
            triplet_text = "- - -"
            report_line_text = triplet_text
            if payload:
                output = _policy_primary_output(payload)
                marks_map = _policy_marks_map(payload)
                triplet = _marks_result_triplet(marks_map, actual_horse_nos)
                triplet_text = _format_triplet_text(triplet)
                bet_decision = _safe_text(output.get("bet_decision")).lower()
                ticket_run_id = _payload_run_id(payload, race_run_id)
                ticket_rows = load_policy_run_ticket_rows(ticket_run_id, policy_engine=engine)
                ticket_summary = _summarize_ticket_rows(ticket_rows)
                if bet_decision == "no_bet" or int(ticket_summary.get("ticket_count", 0) or 0) <= 0:
                    report_line_text = "見"
                else:
                    recovery_rate_text = _percent_text_from_ratio(ticket_summary.get("roi", ""))
                    report_line_text = triplet_text if recovery_rate_text == "-" else f"{triplet_text} 回収率：{recovery_rate_text}"
                stats = metric_rows[engine]
                stats["runs"] += 1
                stats["race_hit_count"] += 1 if ticket_summary["hit_count"] > 0 else 0
                stats["ticket_hit_count"] += int(ticket_summary["hit_count"])
                stats["ticket_count"] += int(ticket_summary["ticket_count"])
                stats["stake_yen"] += int(ticket_summary["stake_yen"])
                stats["payout_yen"] += int(ticket_summary["payout_yen"])
                stats["profit_yen"] += int(ticket_summary["profit_yen"])
                if actual_horse_nos:
                    if triplet and triplet[0] == "◎":
                        stats["honmei_hit_count"] += 1
                    actual_set = {horse_no for horse_no in actual_horse_nos if horse_no}
                    stats["mark_hit_count"] += len(set(marks_map.keys()) & actual_set)
                    stats["mark_total_count"] += len(set(marks_map.keys()))
            race_row[LLM_BATTLE_SHORT_LABELS.get(engine, engine)] = triplet_text
            report_lines.append(f"{LLM_BATTLE_SHORT_LABELS.get(engine, engine)} {report_line_text}")
        race_rows.append(race_row)
        report_lines.append("")

    summary_rows = []
    for engine in LLM_BATTLE_ORDER:
        stats = metric_rows[engine]
        if stats["runs"] <= 0:
            continue
        stake_yen = int(stats["stake_yen"])
        payout_yen = int(stats["payout_yen"])
        recovery_rate_value = round(payout_yen / stake_yen, 4) if stake_yen > 0 else ""
        stats["recovery_rate_value"] = recovery_rate_value
        stats["honmei_hit_rate"] = _ratio_text(stats["honmei_hit_count"], stats["runs"])
        stats["mark_hit_rate"] = _ratio_text(stats["mark_hit_count"], stats["mark_total_count"])
        stats["race_hit_rate"] = _ratio_text(stats["race_hit_count"], stats["runs"])
        stats["ticket_hit_rate"] = _ratio_text(stats["ticket_hit_count"], stats["ticket_count"])
        summary_rows.append(
            {
                "model": stats["model"],
                "レース数": stats["runs"],
                "的中": f"{stats['runs']}レース中 {stats['race_hit_count']}レース的中",
                "的中率": stats["race_hit_rate"],
                "本命命中率": stats["honmei_hit_rate"],
                "印命中率": stats["mark_hit_rate"],
                "honmei_hit_rate_value": round(float(stats["honmei_hit_count"]) / float(stats["runs"]), 6)
                if stats["runs"]
                else "",
                "mark_hit_rate_value": round(float(stats["mark_hit_count"]) / float(stats["mark_total_count"]), 6)
                if stats["mark_total_count"]
                else "",
                "hit_rate_value": round(float(stats["race_hit_count"]) / float(stats["runs"]), 6)
                if stats["runs"]
                else "",
                "投資円": stake_yen,
                "回収円": payout_yen,
                "収支円": int(stats["profit_yen"]),
                "回収率": _percent_text_from_ratio(recovery_rate_value),
                "recovery_rate_value": recovery_rate_value,
            }
        )

    if not race_rows and not summary_rows:
        return {"html": "", "text": ""}

    report_lines.append("【当日成績】")
    report_lines.append("")
    for row in summary_rows:
        report_lines.append(str(row.get("model", "-") or "-"))
        report_lines.append(
            "{hit_summary} 的中率 {hit_rate} 投資{stake_yen}円 回収{payout_yen}円 収支{profit_yen}円 回収率 {recovery_rate} 本命命中率 {honmei_hit_rate} 印命中率 {mark_hit_rate}".format(
                hit_summary=row.get("的中", "-"),
                hit_rate=row.get("的中率", "-"),
                stake_yen=row.get("投資円", 0),
                payout_yen=row.get("回収円", 0),
                profit_yen=row.get("収支円", 0),
                recovery_rate=row.get("回収率", "-"),
                honmei_hit_rate=row.get("本命命中率", "-"),
                mark_hit_rate=row.get("印命中率", "-"),
            )
        )
        report_lines.append("")

    def _leader_rows(metric_key, title):
        ordered = sorted(
            summary_rows,
            key=lambda row: (
                float(row.get(metric_key)) if row.get(metric_key, "") not in ("", None) else -1.0,
                float(_safe_text(row.get("収支円")) or 0.0),
            ),
            reverse=True,
        )
        out = []
        for idx, row in enumerate(ordered[:4], start=1):
            value = row.get(metric_key, "")
            if metric_key in ("recovery_rate_value", "honmei_hit_rate_value", "mark_hit_rate_value", "hit_rate_value"):
                value_text = _percent_text_from_ratio(value)
            else:
                value_text = _safe_text(value) or "-"
            out.append({"順位": idx, "model": row.get("model", ""), "value": value_text})
        report_lines.append("")
        report_lines.append(f"【{title}】")
        for item in out:
            report_lines.append(f"{item['順位']}. {item['model']} {item['value']}")
        return out

    leaderboard_hit = _leader_rows("hit_rate_value", "的中率ランキング")
    leaderboard_honmei = _leader_rows("honmei_hit_rate_value", "本命命中率ランキング")
    leaderboard_recovery = _leader_rows("recovery_rate_value", "回収率ランキング")
    leaderboard_mark = _leader_rows("mark_hit_rate_value", "印命中率ランキング")

    race_table_html = build_table_html(
        race_rows,
        ["race", "chatgpt", "gemini", "deepseek", "grok"],
        "当日レース別対戦",
    )
    summary_table_html = build_table_html(
        summary_rows,
        [
            "model",
            "レース数",
            "的中",
            "的中率",
            "本命命中率",
            "印命中率",
            "投資円",
            "回収円",
            "収支円",
            "回収率",
        ],
        "当日収支と命中",
    )
    leaderboard_html = "".join(
        [
            build_table_html(leaderboard_hit, ["順位", "model", "value"], "的中率ランキング"),
            build_table_html(leaderboard_honmei, ["順位", "model", "value"], "本命命中率ランキング"),
            build_table_html(leaderboard_recovery, ["順位", "model", "value"], "回収率ランキング"),
            build_table_html(leaderboard_mark, ["順位", "model", "value"], "印命中率ランキング"),
        ]
    )
    header_html = (
        '<section class="panel battle-hero-panel">'
        '<div class="panel-title-row">'
        '<div><div class="eyebrow">Daily Report</div><h2>LLM 当日戦績</h2></div>'
        f'<span class="section-chip">{html.escape(target_date)}</span>'
        "</div>"
        '<p class="helper-text">対象：中央・地方合算。各レースでは実際の1着から3着に対応した印を並べ、そのまま日報へ転記できる形にしています。</p>'
        "</section>"
    )
    return {
        "html": f"{header_html}{race_table_html}{summary_table_html}{leaderboard_html}",
        "text": "\n".join(report_lines).strip(),
    }


def _build_llm_weekly_report_bundle(scope_key, current_run_row, actual_result_map):
    target_date = _parse_run_date(_run_date_key(current_run_row))
    if not target_date:
        return {"html": "", "text": ""}

    week_start = target_date - timedelta(days=target_date.weekday())
    week_end = week_start + timedelta(days=6)

    actual_result_maps = {report_scope_key: _load_actual_result_map(report_scope_key) for report_scope_key in LLM_REPORT_SCOPE_KEYS}
    runs = []
    for row in _load_combined_llm_report_runs():
        run_date = _parse_run_date(_run_date_key(row))
        if run_date is None:
            continue
        if week_start <= run_date <= week_end:
            runs.append((run_date, row))
    if not runs:
        return {"html": "", "text": ""}

    runs.sort(
        key=lambda item: (
            item[0].isoformat(),
            _safe_text(item[1].get("race_id")),
            _safe_text(item[1].get("timestamp")),
            _safe_text(item[1].get("run_id")),
        )
    )
    metric_rows = {
        engine: {
            "model": LLM_BATTLE_LABELS.get(engine, engine),
            "runs": 0,
            "race_hit_count": 0,
            "stake_yen": 0,
            "payout_yen": 0,
            "profit_yen": 0,
        }
        for engine in LLM_BATTLE_ORDER
    }
    best_entry = None
    week_label = f"{_format_jp_date_value(week_start)}〜{_format_jp_date_value(week_end)}"
    report_lines = [f"【LLM 週報】{week_label} 中央・地方合算", ""]

    for run_date, run_row in runs:
        run_id = _safe_text(run_row.get("run_id"))
        report_scope_key = _report_scope_key_for_row(run_row, scope_key)
        payload_map = {}
        for payload in load_policy_payloads(report_scope_key, run_id, run_row):
            engine = normalize_policy_engine((payload or {}).get("policy_engine", ""))
            if engine:
                payload_map[engine] = payload
        if not payload_map:
            continue
        race_run_id = run_id
        if not race_run_id:
            for payload in payload_map.values():
                race_run_id = _payload_run_id(payload, race_run_id)
                if race_run_id:
                    break
        if not race_run_id:
            continue
        actual_snapshot = _actual_result_snapshot(
            report_scope_key,
            race_run_id,
            run_row,
            actual_result_maps.get(report_scope_key, {}),
        )
        actual_horse_nos = list(actual_snapshot.get("actual_horse_nos", []) or [])
        race_label = _format_race_label(run_row)
        race_date_text = _format_jp_date_text(run_row) or _format_jp_date_value(run_date)
        for engine in LLM_BATTLE_ORDER:
            payload = payload_map.get(engine)
            if not payload:
                continue
            marks_map = _policy_marks_map(payload)
            triplet = _marks_result_triplet(marks_map, actual_horse_nos)
            triplet_text = _format_triplet_text(triplet)
            ticket_run_id = _payload_run_id(payload, race_run_id)
            ticket_rows = load_policy_run_ticket_rows(ticket_run_id, policy_engine=engine)
            ticket_summary = _summarize_ticket_rows(ticket_rows)
            stats = metric_rows[engine]
            stats["runs"] += 1
            stats["race_hit_count"] += 1 if ticket_summary["hit_count"] > 0 else 0
            stats["stake_yen"] += int(ticket_summary["stake_yen"])
            stats["payout_yen"] += int(ticket_summary["payout_yen"])
            stats["profit_yen"] += int(ticket_summary["profit_yen"])
            if ticket_summary["stake_yen"] > 0:
                recovery_rate_value = round(ticket_summary["payout_yen"] / ticket_summary["stake_yen"], 4)
                candidate = {
                    "model": LLM_BATTLE_LABELS.get(engine, engine),
                    "race": race_label,
                    "date": race_date_text,
                    "的中印": triplet_text,
                    "投資円": int(ticket_summary["stake_yen"]),
                    "回収円": int(ticket_summary["payout_yen"]),
                    "収支円": int(ticket_summary["profit_yen"]),
                    "回収率": _percent_text_from_ratio(recovery_rate_value),
                    "recovery_rate_value": recovery_rate_value,
                    "status_rank": 1 if ticket_summary.get("status") == "settled" else 0,
                }
                if best_entry is None or (
                    candidate["recovery_rate_value"],
                    candidate["回収円"],
                    candidate["status_rank"],
                ) > (
                    best_entry["recovery_rate_value"],
                    best_entry["回収円"],
                    best_entry["status_rank"],
                ):
                    best_entry = candidate

    summary_rows = []
    for engine in LLM_BATTLE_ORDER:
        stats = metric_rows[engine]
        if stats["runs"] <= 0:
            continue
        stake_yen = int(stats["stake_yen"])
        payout_yen = int(stats["payout_yen"])
        profit_yen = int(stats["profit_yen"])
        recovery_rate_value = round(payout_yen / stake_yen, 4) if stake_yen > 0 else ""
        hit_rate_value = round(float(stats["race_hit_count"]) / float(stats["runs"]), 6) if stats["runs"] else ""
        summary_rows.append(
            {
                "model": stats["model"],
                "レース数": stats["runs"],
                "的中": f"{stats['runs']}レース中 {stats['race_hit_count']}レース的中",
                "的中率": _percent_text_from_ratio(hit_rate_value),
                "投資円": stake_yen,
                "回収円": payout_yen,
                "収支円": profit_yen,
                "回収率": _percent_text_from_ratio(recovery_rate_value),
                "stake_value": stake_yen,
                "payout_value": payout_yen,
                "profit_value": profit_yen,
                "hit_rate_value": hit_rate_value,
                "recovery_rate_value": recovery_rate_value,
            }
        )

    if not summary_rows:
        return {"html": "", "text": ""}

    report_lines.append("【週間成績】")
    for row in summary_rows:
        report_lines.append(
            "{model} {hit_summary} 的中率 {hit_rate} 投資{stake_yen}円 回収{payout_yen}円 収支{profit_yen}円 回収率 {recovery_rate}".format(
                model=row.get("model", "-"),
                hit_summary=row.get("的中", "-"),
                hit_rate=row.get("的中率", "-"),
                stake_yen=row.get("投資円", 0),
                payout_yen=row.get("回収円", 0),
                profit_yen=row.get("収支円", 0),
                recovery_rate=row.get("回収率", "-"),
            )
        )

    def _leader_rows(metric_key, title, value_kind):
        ordered = sorted(
            summary_rows,
            key=lambda row: (
                float(row.get(metric_key)) if row.get(metric_key, "") not in ("", None) else -1.0,
                float(_safe_text(row.get("収支円")) or 0.0),
            ),
            reverse=True,
        )
        out = []
        for idx, row in enumerate(ordered[:4], start=1):
            value = row.get(metric_key, "")
            if value_kind == "percent":
                value_text = _percent_text_from_ratio(value)
            else:
                value_text = f"{int(value)}円" if value not in ("", None) else "-"
            out.append({"順位": idx, "model": row.get("model", ""), "value": value_text})
        report_lines.append("")
        report_lines.append(f"【{title}】")
        for item in out:
            report_lines.append(f"{item['順位']}. {item['model']} {item['value']}")
        return out

    leaderboard_stake = _leader_rows("stake_value", "投資額ランキング", "yen")
    leaderboard_payout = _leader_rows("payout_value", "回収額ランキング", "yen")
    leaderboard_profit = _leader_rows("profit_value", "収支ランキング", "yen")
    leaderboard_recovery = _leader_rows("recovery_rate_value", "回収率ランキング", "percent")
    leaderboard_hit = _leader_rows("hit_rate_value", "的中率ランキング", "percent")

    if best_entry:
        report_lines.extend(
            [
                "",
                "【今週のベスト】",
                "{model} {date} {race} 的中印 {triplet} 投資{stake_yen}円 回収{payout_yen}円 収支{profit_yen}円 回収率 {recovery_rate}".format(
                    model=best_entry.get("model", "-"),
                    date=best_entry.get("date", "-"),
                    race=best_entry.get("race", "-"),
                    triplet=best_entry.get("的中印", "- - -"),
                    stake_yen=best_entry.get("投資円", 0),
                    payout_yen=best_entry.get("回収円", 0),
                    profit_yen=best_entry.get("収支円", 0),
                    recovery_rate=best_entry.get("回収率", "-"),
                ),
            ]
        )

    summary_table_html = build_table_html(
        summary_rows,
        ["model", "レース数", "的中", "的中率", "投資円", "回収円", "収支円", "回収率"],
        "週間収支サマリー",
    )
    leaderboard_html = "".join(
        [
            build_table_html(leaderboard_stake, ["順位", "model", "value"], "投資額ランキング"),
            build_table_html(leaderboard_payout, ["順位", "model", "value"], "回収額ランキング"),
            build_table_html(leaderboard_profit, ["順位", "model", "value"], "収支ランキング"),
            build_table_html(leaderboard_recovery, ["順位", "model", "value"], "回収率ランキング"),
            build_table_html(leaderboard_hit, ["順位", "model", "value"], "的中率ランキング"),
        ]
    )
    best_html = ""
    if best_entry:
        best_table_rows = [
            {
                "model": best_entry.get("model", "-"),
                "date": best_entry.get("date", "-"),
                "race": best_entry.get("race", "-"),
                "的中印": best_entry.get("的中印", "- - -"),
                "投資円": best_entry.get("投資円", 0),
                "回収円": best_entry.get("回収円", 0),
                "収支円": best_entry.get("収支円", 0),
                "回収率": best_entry.get("回収率", "-"),
            }
        ]
        best_html = (
            '<section class="panel battle-hero-panel">'
            '<div class="panel-title-row">'
            '<div><div class="eyebrow">Best Of Week</div><h2>今週のベスト</h2></div>'
            f'<span class="section-chip">{html.escape(best_entry.get("model", "-"))}</span>'
            "</div>"
            '<p class="helper-text">今週の中で回収率が最も高かったレースです。</p>'
            f"{build_table_html(best_table_rows, ['model', 'date', 'race', '的中印', '投資円', '回収円', '収支円', '回収率'], '今週のベスト')}"
            "</section>"
        )
    header_html = (
        '<section class="panel battle-hero-panel">'
        '<div class="panel-title-row">'
        '<div><div class="eyebrow">Weekly Report</div><h2>LLM 週報</h2></div>'
        f'<span class="section-chip">{html.escape(week_label)}</span>'
        "</div>"
        '<p class="helper-text">対象：中央・地方合算。週単位で投資、回収、収支、回収率、的中率をまとめています。</p>'
        "</section>"
    )
    return {
        "html": f"{header_html}{summary_table_html}{leaderboard_html}{best_html}",
        "text": "\n".join(report_lines).strip(),
    }


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
        for engine in LLM_BATTLE_ORDER:
            payload = payload_map.get(engine)
            if not payload:
                continue
            output = _policy_primary_output(payload)
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
                    <span>ROI {_format_percent_text(ticket_summary.get("roi", ""))}</span>
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
                <span>ROI {_format_percent_text(roi)}</span>
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
                    <span>ROI {_format_percent_text(ticket_summary.get("roi", ""))}</span>
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
                <span>ROI {_format_percent_text(roi)}</span>
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


def _race_job_status_tone(status):
    text = str(status or "").strip().lower()
    if text in ("ready", "settled"):
        return "good"
    if text in ("queued_process", "processing", "queued_settle", "settling"):
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
    if status_text in ("scheduled", "queued_process", "failed"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/process_now">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">立即执行处理</button>
            </form>
            """
        )
    if status_text in ("ready", "queued_settle", "settling", "settled"):
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
        <span class="section-chip">{html.escape(_race_job_status_label(status))}</span>
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
        {_race_job_action_buttons(job_id, status, admin_token=admin_token)}
        {open_links}
      </div>
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
                <label>比赛日期（展示用）</label>
                <input type="date" name="race_date">
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
    mapping = {
        "uploaded": "已上传",
        "scheduled": "待处理",
        "queued_process": "待执行",
        "processing": "处理中",
        "ready": "预测已生成",
        "queued_settle": "待结算",
        "settling": "结算中",
        "settled": "已结算",
        "failed": "失败",
    }
    text = str(status or "").strip().lower()
    return mapping.get(text, text or "-")


def _race_job_action_buttons_clean(job_id, status, admin_token=""):
    buttons = []
    status_text = str(status or "").strip().lower()
    if status_text in ("scheduled", "queued_process", "failed"):
        buttons.append(
            f"""
            <form method="post" action="/console/tasks/process_now">
              <input type="hidden" name="job_id" value="{html.escape(job_id)}">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit">立即执行预测</button>
            </form>
            """
        )
    if status_text in ("ready", "queued_settle", "settling", "settled"):
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
    current_run_id = str((row or {}).get("current_run_id", "") or "").strip()
    if not current_run_id:
        return ""
    status_text = str((row or {}).get("status", "") or "").strip().lower()
    if status_text not in ("ready", "queued_settle", "settling", "settled"):
        return ""
    job_id = str((row or {}).get("job_id", "") or "").strip()
    top1 = str((row or {}).get("actual_top1", "") or "").strip()
    top2 = str((row or {}).get("actual_top2", "") or "").strip()
    top3 = str((row or {}).get("actual_top3", "") or "").strip()
    return f"""
    <section class="job-settle-panel">
      <div class="job-settle-head">
        <strong>Step 3: 录入赛果并结算</strong>
        <span>填写 1-3 着马名后，可以直接结算，也可以先加入结算队列。</span>
      </div>
      <div class="job-settle-actions">
        <form method="post" action="/console/tasks/settle_now" class="job-settle-form">
          <input type="hidden" name="job_id" value="{html.escape(job_id)}">
          <input type="hidden" name="token" value="{html.escape(admin_token)}">
          <input type="text" name="actual_top1" value="{html.escape(top1)}" placeholder="1着马名">
          <input type="text" name="actual_top2" value="{html.escape(top2)}" placeholder="2着马名">
          <input type="text" name="actual_top3" value="{html.escape(top3)}" placeholder="3着马名">
          <button type="submit">立即结算</button>
        </form>
        <form method="post" action="/console/tasks/queue_settle" class="job-settle-form">
          <input type="hidden" name="job_id" value="{html.escape(job_id)}">
          <input type="hidden" name="token" value="{html.escape(admin_token)}">
          <input type="text" name="actual_top1" value="{html.escape(top1)}" placeholder="1着马名">
          <input type="text" name="actual_top2" value="{html.escape(top2)}" placeholder="2着马名">
          <input type="text" name="actual_top3" value="{html.escape(top3)}" placeholder="3着马名">
          <button type="submit">加入结算队列</button>
        </form>
      </div>
    </section>
    """


def _admin_job_card_html_clean(row, admin_token=""):
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
        ("预计开始处理", "process_after_time"),
        ("入处理队列", "queued_process_at"),
        ("预测完成", "ready_at"),
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
          <button type="submit" class="secondary-button">打开 Run</button>
        </form>
        <form method="get" action="/llm_today" class="stack-form">
          <input type="hidden" name="scope_key" value="{html.escape(scope_key)}">
          <input type="hidden" name="date" value="{html.escape(race_date)}">
          <button type="submit" class="secondary-button">查看当日看板</button>
        </form>
        """
    return f"""
    <section class="panel panel--tight">
      <div class="section-title">
        <div>
          <div class="eyebrow">Task</div>
          <h2>{html.escape((location + ' ' + race_id).strip() or job_id or '未命名任务')}</h2>
        </div>
        <span class="section-chip">{html.escape(_race_job_status_label_clean(status))}</span>
      </div>
      <div class="copy-row">
        <span class="hero-pill">范围 {html.escape(_scope_display_name(scope_key))}</span>
        <span class="hero-pill">日期 {html.escape(race_date or '-')}</span>
        <span class="hero-pill">条件 {html.escape(str(row.get('target_surface', '') or '-'))} / {html.escape(str(row.get('target_distance', '') or '-'))}m / {html.escape(str(row.get('target_track_condition', '') or '-'))}</span>
        <span class="hero-pill">Run {html.escape(current_run_id or '-')}</span>
        <span class="hero-pill">赛果 {html.escape(' / '.join(x for x in [actual_top1, actual_top2, actual_top3] if x) or '未录入')}</span>
      </div>
      <div class="copy-row">
        <span class="hero-pill">kachiuma: {html.escape(kachiuma_name)}</span>
        <span class="hero-pill">shutuba: {html.escape(shutuba_name)}</span>
        {''.join(timing_items)}
      </div>
      <p class="helper-text">{html.escape(notes or '无备注')}</p>
      <div class="copy-row">
        {_race_job_action_buttons_clean(job_id, status, admin_token=admin_token)}
        {open_links}
      </div>
      {_race_job_settle_form_clean(row, admin_token=admin_token)}
    </section>
    """


def build_admin_workspace_html_clean(message_text="", error_text="", admin_token="", authorized=True):
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
            <p class="helper-text">{html.escape(error_text or "管理口令无效。")}</p>
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

    cards_html = "".join(_admin_job_card_html_clean(job, admin_token=admin_token) for job in jobs)
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
          <p class="helper-text">先上传一场比赛的 `kachiuma.csv` 和 `shutuba.csv`，再执行预测流程。</p>
        </section>
        """

    message_panel = ""
    if message_text:
        message_panel = f"""
        <section class="panel panel--tight">
          <div class="section-title">
            <div>
              <div class="eyebrow">Result</div>
              <h2>操作结果</h2>
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
    return f"""
    <section class="content-cluster" id="admin-zone">
      <div class="cluster-head">
        <div>
          <div class="eyebrow">Admin Workspace</div>
          <h2>任务后台</h2>
        </div>
        <p>当前后台只做三件事：上传输入、执行预测、录入赛果并结算。流程已经跑通，这里主要服务日常运营。</p>
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
            <span class="hero-pill">待处理 {summary['scheduled']}</span>
            <span class="hero-pill">处理中 {summary['processing']}</span>
            <span class="hero-pill">预测已生成 {summary['ready']}</span>
            <span class="hero-pill">已结算 {summary['settled']}</span>
          </div>
          <div class="copy-row">
            <form method="post" action="/console/tasks/scan_due" class="stack-form">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit" class="secondary-button">扫描到点任务</button>
            </form>
            <form method="post" action="/console/tasks/run_due_now" class="stack-form">
              <input type="hidden" name="token" value="{html.escape(admin_token)}">
              <button type="submit" class="secondary-button">执行已入队任务</button>
            </form>
            <a href="/llm_today" class="secondary-button">查看前台看板</a>
          </div>
          <p class="helper-text">如果暂时不做自动调度，就按这个节奏操作：上传两份 CSV，手动执行，赛后录入前三名并结算。</p>
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
                <label>赛道范围</label>
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
                <label>比赛地点</label>
                <input type="text" name="location" placeholder="中山">
              </div>
              <div>
                <label>比赛日期</label>
                <input type="date" name="race_date">
              </div>
              <div>
                <label>开赛时间</label>
                <input type="datetime-local" name="scheduled_off_time" value="{html.escape(default_dt)}">
              </div>
              <div>
                <label>提前执行分钟数</label>
                <input type="number" name="lead_minutes" min="0" value="30">
              </div>
              <div>
                <label>本场场地</label>
                <select name="target_surface">
                  <option value="dirt">dirt</option>
                  <option value="turf">turf</option>
                </select>
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
              <textarea name="notes" placeholder="可记录这场比赛的说明、上传来源或临时备注。"></textarea>
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


def render_console_page(message_text="", error_text="", admin_token=""):
    return render_page(
        "",
        admin_token=admin_token,
        admin_workspace_html=(
            build_import_archive_panel(admin_token=admin_token)
            + build_admin_workspace_html_clean(
                message_text=message_text,
                error_text=error_text,
                admin_token=admin_token,
                authorized=_admin_token_valid(admin_token),
            )
        ),
    )


def _resolve_console_run_state(scope_key="", selected_run_id="", summary_run_id="", output_text=""):
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


def _load_note_workspace(scope_key="", run_id="", run_row=None):
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
    actual_result_map = _load_actual_result_map(scope_norm)
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
    battle_bundle = _build_llm_battle_bundle(
        scope_norm,
        run_id,
        run_row,
        policy_payloads,
        actual_result_map,
    )
    llm_note_text = str(battle_bundle.get("note_text", "") or "").strip()
    daily_report_text = str(_build_llm_daily_report_bundle(scope_norm, run_row, actual_result_map).get("text", "") or "").strip()
    weekly_report_text = str(_build_llm_weekly_report_bundle(scope_norm, run_row, actual_result_map).get("text", "") or "").strip()

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


def build_note_workspace_page(scope_key="", run_id="", admin_token=""):
    state = _resolve_console_run_state(scope_key=scope_key, selected_run_id=run_id)
    scope_norm = normalize_scope_key(state["scope_key"]) or "central_dirt"
    note_bundle = _load_note_workspace(scope_norm, state["run_id"], state["run_row"])
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
              <p class="note-empty">当前没有可复制的内容。</p>
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
            <button type="button" class="note-copy-button" data-copy-target="{html.escape(dom_id)}" data-copy-status="{html.escape(dom_id)}-status">复制</button>
            <span id="{html.escape(dom_id)}-status" class="note-copy-status"></span>
          </div>
          <details class="note-preview" open>
            <summary>预览</summary>
            <pre>{html.escape(text_value)}</pre>
          </details>
          <textarea id="{html.escape(dom_id)}" class="note-source" readonly>{html.escape(text_value)}</textarea>
        </section>
        """

    blocks_html = "".join(
        [
            _note_block("单场 LLM Note", note_bundle["llm_note_text"], "note-llm", tone="accent"),
            _note_block("Predictor Note", note_bundle["mark_note_text"], "note-predictor"),
            _note_block("Daily Report", note_bundle["daily_report_text"], "note-daily"),
            _note_block("Weekly Report", note_bundle["weekly_report_text"], "note-weekly"),
        ]
    )

    run_label = state["run_id"] or "未指定"
    race_label = state["current_race_id"] or "-"
    return f"""<!doctype html>
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
        <a href="{html.escape(back_href)}">返回控制台</a>
      </div>
    </section>
    <section class="note-tools">
      <form method="get" action="/console/note">
        <input type="hidden" name="token" value="{html.escape(admin_token)}">
        <div class="note-form-grid">
          <label>范围
            <select name="scope_key">
              <option value="central_dirt"{' selected' if scope_norm == 'central_dirt' else ''}>central_dirt</option>
              <option value="central_turf"{' selected' if scope_norm == 'central_turf' else ''}>central_turf</option>
              <option value="local"{' selected' if scope_norm == 'local' else ''}>local</option>
            </select>
          </label>
          <label>Run ID / Race ID
            <input type="text" name="run_id" value="{html.escape(run_value)}" placeholder="202501010101 或 20250101_123456">
          </label>
        </div>
        <div class="note-links">
          <button type="submit">打开这场 Note</button>
          <a href="{html.escape(note_href)}">刷新当前页面</a>
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
          if (status) status.textContent = "已复制";
        }} catch (error) {{
          target.focus();
          target.select();
          document.execCommand("copy");
          if (status) status.textContent = "已复制";
        }}
      }});
    }});
  </script>
</body>
</html>"""


def _target_surface_from_scope(scope_key):
    return "turf" if str(scope_key or "").strip() == "central_turf" else "dirt"


def _import_history_zip(base_dir, archive_bytes, overwrite=False):
    data_root = Path(base_dir) / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    imported_paths = []
    with zipfile.ZipFile(BytesIO(archive_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            raw_name = str(info.filename or "").replace("\\", "/").strip("/")
            if not raw_name:
                continue
            parts = [part for part in raw_name.split("/") if part and part not in (".", "..")]
            if not parts:
                continue
            rel_parts = None
            if len(parts) >= 2 and parts[0] == "pipeline" and parts[1] == "data":
                rel_parts = parts[2:]
            elif parts[0] == "data":
                rel_parts = parts[1:]
            elif parts[0] in ("_shared", "central_dirt", "central_turf", "local"):
                rel_parts = parts
            if not rel_parts:
                skipped += 1
                continue
            dest = data_root.joinpath(*rel_parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            file_bytes = zf.read(info)
            dest.write_bytes(file_bytes)
            written += 1
            imported_paths.append(str(dest.relative_to(data_root)).replace("\\", "/"))
    return {
        "written": written,
        "skipped": skipped,
        "sample_paths": imported_paths[:8],
    }


def build_import_archive_panel(admin_token=""):
    return f"""
    <section class="content-cluster" id="import-zone">
      <div class="cluster-head">
        <div>
          <div class="eyebrow">Import</div>
          <h2>历史数据导入</h2>
        </div>
        <p>把你本地 3 月 14 日跑过的数据打成 ZIP，直接导入到 Render 的持久磁盘。</p>
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
            <p class="helper-text">支持 `pipeline/data/...`、`data/...`，或者直接包含 `central_dirt`、`central_turf`、`local`、`_shared` 的 ZIP。</p>
            <button type="submit">导入到 Render Disk</button>
          </form>
        </section>
      </div>
    </section>
    """


def build_race_jobs_page(message_text="", error_text="", admin_token="", authorized=True):
    if authorized:
        return render_console_page(message_text=message_text, error_text=error_text, admin_token=admin_token)
    return render_console_page(message_text=message_text, error_text=error_text, admin_token=admin_token)

    if not authorized:
        error_block = ""
        if error_text:
            error_block = f'<section class="job-flash job-flash--error">{html.escape(error_text)}</section>'
        helper_text = "已启用管理口令。输入正确 token 后才能上传文件、扫描到点任务或修改状态。"
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Race Job 调度看板</title>
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
      <h1>Race Job 调度看板</h1>
      <p>{html.escape(helper_text)}</p>
      <form method="get" action="/console">
        <input type="password" name="token" placeholder="ADMIN_TOKEN" value="{html.escape(admin_token)}">
        <div class="actions">
          <button type="submit">进入管理页</button>
          <a href="/llm_today">仅看公开看板</a>
        </div>
      </form>
    </section>
  </main>
</body>
</html>"""

    jobs = load_race_jobs(BASE_DIR)
    summary = {
        "total": len(jobs),
        "scheduled": 0,
        "processing": 0,
        "ready": 0,
        "settled": 0,
    }
    for job in jobs:
        status = str(job.get("status", "")).strip().lower()
        if status == "scheduled":
            summary["scheduled"] += 1
        elif status in ("queued_process", "processing", "queued_settle", "settling"):
            summary["processing"] += 1
        elif status == "ready":
            summary["ready"] += 1
        elif status == "settled":
            summary["settled"] += 1

    job_cards = []
    for job in jobs:
        row = dict(job)
        job_id = str(row.get("job_id", "") or "").strip()
        status = str(row.get("status", "") or "").strip()
        tone = _race_job_status_tone(status)
        artifacts = list(row.get("artifacts", []) or [])
        artifact_map = {str(item.get("artifact_type", "")).strip().lower(): dict(item) for item in artifacts}
        kachiuma = artifact_map.get("kachiuma", {})
        shutuba = artifact_map.get("shutuba", {})
        notes = str(row.get("notes", "") or "").strip()
        current_run_id = str(row.get("current_run_id", "") or "").strip()
        actual_top1 = str(row.get("actual_top1", "") or "").strip()
        actual_top2 = str(row.get("actual_top2", "") or "").strip()
        actual_top3 = str(row.get("actual_top3", "") or "").strip()
        timing_tags = []
        for label, key in (
            ("开赛", "scheduled_off_time"),
            ("开始处理", "process_after_time"),
            ("处理队列", "queued_process_at"),
            ("已就绪", "ready_at"),
            ("已结算", "settled_at"),
        ):
            value = str(row.get(key, "") or "").strip()
            if value:
                timing_tags.append(f'<span>{html.escape(label)} {html.escape(value)}</span>')
        job_cards.append(
            f"""
            <article class="job-card job-card--{html.escape(tone)}">
              <div class="job-card-head">
                <div>
                  <div class="job-eyebrow">{html.escape(_scope_display_name(row.get('scope_key', '')))}</div>
                  <h3>{html.escape(str(row.get('location', '') or '-') + ' ' + str(row.get('race_id', '') or '-'))}</h3>
                </div>
                <span class="job-badge job-badge--{html.escape(tone)}">{html.escape(_race_job_status_label(status))}</span>
              </div>
              <div class="job-meta-row">
                <span>比赛日 {html.escape(str(row.get('race_date', '') or '-'))}</span>
                <span>Job {html.escape(job_id or '-')}</span>
                <span>提前 {html.escape(str(row.get('lead_minutes', 30) or 30))} 分钟启动</span>
                <span>条件 {html.escape(str(row.get('target_surface', '') or '-'))} / {html.escape(str(row.get('target_distance', '') or '-'))}m / {html.escape(str(row.get('target_track_condition', '') or '-'))}</span>
                <span>Run {html.escape(current_run_id or '-')}</span>
                <span>赛果 {html.escape(' / '.join(x for x in [actual_top1, actual_top2, actual_top3] if x) or '未录入')}</span>
              </div>
              <div class="job-file-grid">
                <section>
                  <h4>kachiuma.csv</h4>
                  <p>{html.escape(str(kachiuma.get('original_name', '') or '未上传'))}</p>
                </section>
                <section>
                  <h4>shutuba.csv</h4>
                  <p>{html.escape(str(shutuba.get('original_name', '') or '未上传'))}</p>
                </section>
              </div>
              <div class="job-meta-row">
                {''.join(timing_tags) if timing_tags else '<span>还没有时间节点</span>'}
              </div>
              <div class="job-notes">{html.escape(notes or '无备注')}</div>
              {_race_job_settle_form(row, admin_token=admin_token)}
              <div class="job-actions">
                {_race_job_action_buttons(job_id, status, admin_token=admin_token)}
                {
                    f'''
                    <form method="post" action="/view_run">
                      <input type="hidden" name="scope_key" value="{html.escape(str(row.get("scope_key", "") or ""))}">
                      <input type="hidden" name="run_id" value="{html.escape(current_run_id)}">
                      <button type="submit">打开 Run</button>
                    </form>
                    <form method="get" action="/llm_today">
                      <input type="hidden" name="scope_key" value="{html.escape(str(row.get("scope_key", "") or ""))}">
                      <input type="hidden" name="date" value="{html.escape(str(row.get("race_date", "") or ""))}">
                      <button type="submit">打开当日看板</button>
                    </form>
                    '''
                    if current_run_id
                    else ""
                }
              </div>
            </article>
            """
        )

    empty_state = ""
    if not job_cards:
        empty_state = """
        <section class="job-empty">
          <h2>还没有调度任务</h2>
          <p>先上传某一场比赛的 kachiuma.csv 和 shutuba.csv，再设置开赛时间。页面会自动计算“开赛前 30 分钟开始处理”。</p>
        </section>
        """

    message_block = ""
    if message_text:
        message_block = f'<section class="job-flash job-flash--ok">{html.escape(message_text)}</section>'
    error_block = ""
    if error_text:
        error_block = f'<section class="job-flash job-flash--error">{html.escape(error_text)}</section>'
    auth_notice = ""
    if _admin_token_enabled():
        auth_notice = '<span>管理口令：已启用</span>'
    else:
        auth_notice = '<span>管理口令：未启用</span>'

    default_dt = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%dT15:00")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Race Job 调度看板</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --paper: rgba(255, 250, 244, 0.92);
      --line: rgba(23, 31, 26, 0.1);
      --ink: #17201a;
      --muted: #617064;
      --accent: #145846;
      --accent-soft: rgba(20, 88, 70, 0.12);
      --warn: #a36a18;
      --warn-soft: rgba(163, 106, 24, 0.12);
      --danger: #ad4d3b;
      --danger-soft: rgba(173, 77, 59, 0.12);
      --shadow: 0 18px 50px rgba(28, 33, 29, 0.09);
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
        radial-gradient(circle at 100% 0%, rgba(198, 221, 210, 0.65), transparent 30%),
        linear-gradient(180deg, #faf6f0 0%, var(--bg) 100%);
    }}
    .job-page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 20px 40px;
      display: grid;
      gap: 22px;
    }}
    .job-hero, .job-panel, .job-card, .job-empty, .job-flash {{
      background: var(--paper);
      border: 1px solid rgba(255,255,255,0.7);
      border-radius: 26px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .job-hero, .job-panel, .job-empty, .job-flash {{ padding: 24px; }}
    .job-eyebrow {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .job-hero h1, .job-panel h2, .job-card h3, .job-empty h2 {{
      margin: 0;
      font-family: var(--title-font);
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    .job-hero h1 {{ font-size: clamp(34px, 5vw, 52px); line-height: 0.96; }}
    .job-hero p, .job-empty p {{
      margin: 0;
      max-width: 72ch;
      color: var(--muted);
      line-height: 1.65;
    }}
    .job-flash--ok {{ border-color: rgba(20, 88, 70, 0.2); }}
    .job-flash--error {{ border-color: rgba(173, 77, 59, 0.25); background: rgba(255, 245, 242, 0.95); }}
    .job-summary-grid, .job-board-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .job-summary-card, .job-card {{
      padding: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.78);
      border-radius: 20px;
      display: grid;
      gap: 12px;
    }}
    .job-summary-card strong {{ font-size: 28px; font-family: var(--title-font); }}
    .job-tools {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
    }}
    .job-tools a, .job-tools button, .job-actions button, .job-upload button {{
      min-height: 40px;
      padding: 0 14px;
      border-radius: 999px;
      border: 0;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
    }}
    .job-upload {{
      display: grid;
      gap: 12px;
    }}
    .job-upload-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .job-upload label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .job-upload input, .job-upload select, .job-upload textarea {{
      width: 100%;
      min-height: 42px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.9);
      font: inherit;
      color: var(--ink);
    }}
    .job-upload textarea {{ min-height: 90px; resize: vertical; }}
    .job-card--good {{ border-color: rgba(20, 88, 70, 0.2); }}
    .job-card--active {{ border-color: rgba(163, 106, 24, 0.22); }}
    .job-card--danger {{ border-color: rgba(173, 77, 59, 0.24); }}
    .job-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .job-badge {{
      padding: 7px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .job-badge--good {{ background: var(--accent-soft); color: var(--accent); }}
    .job-badge--active {{ background: var(--warn-soft); color: var(--warn); }}
    .job-badge--danger {{ background: var(--danger-soft); color: var(--danger); }}
    .job-badge--muted {{ background: rgba(23,31,26,0.06); color: var(--muted); }}
    .job-meta-row, .job-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .job-meta-row span {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(23,31,26,0.06);
      color: var(--muted);
      font-size: 12px;
    }}
    .job-file-grid {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .job-file-grid section {{
      padding: 12px;
      border-radius: 16px;
      background: rgba(23,31,26,0.04);
    }}
    .job-file-grid h4 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .job-file-grid p, .job-notes {{
      margin: 0;
      line-height: 1.55;
      word-break: break-word;
    }}
    .job-settle-panel {{
      padding: 12px;
      border-radius: 16px;
      background: rgba(20, 88, 70, 0.06);
      display: grid;
      gap: 10px;
    }}
    .job-settle-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: space-between;
      align-items: baseline;
    }}
    .job-settle-head span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .job-settle-actions {{
      display: grid;
      gap: 10px;
    }}
    .job-settle-form {{
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(3, minmax(0, 1fr)) auto;
    }}
    .job-settle-form input {{
      width: 100%;
      min-height: 40px;
      padding: 0 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.95);
      color: var(--ink);
      font: inherit;
    }}
    .job-actions form {{ margin: 0; }}
    @media (max-width: 760px) {{
      .job-page {{ padding: 18px 14px 30px; }}
      .job-hero, .job-panel, .job-empty, .job-flash {{ padding: 18px; }}
      .job-card-head {{ flex-direction: column; }}
      .job-tools {{ align-items: stretch; }}
      .job-settle-form {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="job-page">
    <section class="job-hero">
      <div class="job-eyebrow">Manual Admin Flow</div>
      <h1>任务后台</h1>
      <p>这里按手动三步流来管理：先上传 `kachiuma.csv` 和 `shutuba.csv`，再手动执行预测，最后录入赛果并结算。开赛时间和提前分钟现在只作为记录和手动筛选参考，不会自动到点运行。</p>
      <div class="job-tools">
        <div class="job-meta-row">
          <span>总任务 {summary['total']}</span>
          <span>待处理 {summary['scheduled']}</span>
          <span>处理中 {summary['processing']}</span>
          <span>预测已生成 {summary['ready']}</span>
          <span>已结算 {summary['settled']}</span>
          {auth_notice}
        </div>
        <div class="job-actions">
          <form method="post" action="/console/tasks/scan_due">
            <input type="hidden" name="token" value="{html.escape(admin_token)}">
            <button type="submit">按时间筛选任务</button>
          </form>
          <form method="post" action="/console/tasks/run_due_now">
            <input type="hidden" name="token" value="{html.escape(admin_token)}">
            <button type="submit">执行已入队任务</button>
          </form>
          <a href="/llm_today">看 LLM 看板</a>
          <a href="/console">回主控制台</a>
        </div>
      </div>
    </section>
    {message_block}
    {error_block}
    <section class="job-panel">
      <div class="job-eyebrow">Step 1</div>
      <h2>上传两个 CSV</h2>
      <form class="job-upload" method="post" action="/console/tasks/create" enctype="multipart/form-data">
        <input type="hidden" name="token" value="{html.escape(admin_token)}">
        <div class="job-upload-grid">
          <label>范围
            <select name="scope_key">
              <option value="central_dirt">中央 Dirt</option>
              <option value="central_turf">中央 Turf</option>
              <option value="local">地方</option>
            </select>
          </label>
          <label>Race ID
            <input type="text" name="race_id" placeholder="202606010109">
          </label>
          <label>场地
            <input type="text" name="location" placeholder="中山">
          </label>
          <label>比赛日期（展示用）
            <input type="date" name="race_date">
          </label>
          <label>开赛时间（记录用）
            <input type="datetime-local" name="scheduled_off_time" value="{html.escape(default_dt)}">
          </label>
          <label>提前多少分钟（手动筛选参考）
            <input type="number" name="lead_minutes" min="0" value="30">
          </label>
          <label>kachiuma.csv
            <input type="file" name="kachiuma_file" accept=".csv">
          </label>
          <label>shutuba.csv
            <input type="file" name="shutuba_file" accept=".csv">
          </label>
        </div>
        <label>备注
          <textarea name="notes" placeholder="例如：前一天上传，比赛前手动点击执行预测"></textarea>
        </label>
        <div class="job-actions">
          <button type="submit">创建任务</button>
        </div>
      </form>
    </section>
    {empty_state}
    <section class="job-panel">
      <div class="job-eyebrow">Step 2 / Step 3</div>
      <h2>预测与结算任务</h2>
      <div class="job-board-grid">
        {''.join(job_cards)}
      </div>
    </section>
  </main>
</body>
</html>"""


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
    llm_battle_html = ""
    llm_note_text = ""
    gemini_policy_html = ""
    daily_report_html = ""
    daily_report_text = ""
    weekly_report_html = ""
    weekly_report_text = ""
    summary_table_html = ""
    if run_id:
        predictor_top_sections = []
        predictor_mark_sections = []
        predictor_summary_sections = []
        predictor_note_texts = []
        bet_engine_v3_summary = load_bet_engine_v3_cfg_summary(scope_norm or scope_key, run_id)
        policy_payloads = []
        actual_result_map = _load_actual_result_map(scope_norm or scope_key)
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
        battle_bundle = _build_llm_battle_bundle(
            scope_norm or scope_key,
            run_id,
            run_row,
            policy_payloads,
            actual_result_map,
        )
        llm_battle_html = battle_bundle.get("html", "")
        llm_note_text = battle_bundle.get("note_text", "")
        daily_bundle = _build_llm_daily_report_bundle(scope_norm or scope_key, run_row, actual_result_map)
        daily_report_html = daily_bundle.get("html", "")
        daily_report_text = daily_bundle.get("text", "")
        weekly_bundle = _build_llm_weekly_report_bundle(scope_norm or scope_key, run_row, actual_result_map)
        weekly_report_html = weekly_bundle.get("html", "")
        weekly_report_text = weekly_bundle.get("text", "")
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
        default_policy_engine=normalize_policy_engine(os.environ.get("POLICY_ENGINE", "gemini") or "gemini"),
        default_policy_model=resolve_policy_model(
            normalize_policy_engine(os.environ.get("POLICY_ENGINE", "gemini") or "gemini"),
            os.environ.get("POLICY_MODEL", ""),
            os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        ),
        admin_token=admin_token,
        admin_enabled=_admin_token_enabled(),
        admin_workspace_html=admin_workspace_html,
    )


def _admin_execution_denied(message, scope_key="", token="", selected_run_id="", summary_run_id=""):
    return render_page(
        scope_key,
        error_text=str(message or "管理口令错误，不能执行该操作。"),
        selected_run_id=selected_run_id,
        summary_run_id=summary_run_id,
        admin_token=token,
    )


@app.get("/", response_class=HTMLResponse)
def index(token: str = ""):
    return build_llm_today_page_clean()


@app.get("/console", response_class=HTMLResponse)
def console_index(token: str = ""):
    if _admin_token_enabled() and not _admin_token_valid(token):
        return build_console_gate_page(admin_token=token, error_text="管理口令无效。")
    return render_console_page(admin_token=token)


@app.get("/console/note", response_class=HTMLResponse)
def console_note(scope_key: str = "central_dirt", run_id: str = "", token: str = ""):
    if _admin_token_enabled() and not _admin_token_valid(token):
        return build_console_gate_page(admin_token=token, error_text="管理口令无效。")
    return build_note_workspace_page(scope_key=scope_key, run_id=run_id, admin_token=token)


@app.get("/llm_today", response_class=HTMLResponse)
def llm_today(date: str = "", scope_key: str = ""):
    return build_llm_today_page_clean(date_text=date, scope_key=scope_key)


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
    if not _admin_token_valid(token):
        return build_race_jobs_page(
            admin_token=token,
            authorized=False,
            error_text="管理口令无效，无法创建任务。",
        )
    race_id = normalize_race_id(race_id)
    scope_norm = normalize_scope_key(scope_key)
    if not scope_norm:
        return build_race_jobs_page(admin_token=token, error_text="范围无效。")
    if not race_id:
        return build_race_jobs_page(admin_token=token, error_text="Race ID 不能为空。")
    if not scheduled_off_time.strip():
        return build_race_jobs_page(admin_token=token, error_text="请填写开赛时间。")
    target_surface = _target_surface_from_scope(scope_norm)
    if False:
        return build_race_jobs_page(admin_token=token, error_text="请填写本场场地：turf 或 dirt。")
    target_distance = str(target_distance or "").strip()
    try:
        target_distance_value = int(target_distance)
    except ValueError:
        return build_race_jobs_page(admin_token=token, error_text="请填写本场距离，例如 1200 或 1800。")
    if target_distance_value <= 0:
        return build_race_jobs_page(admin_token=token, error_text="距离必须大于 0。")
    target_track_condition = str(target_track_condition or "").strip()
    if target_track_condition not in ("良", "稍重", "重", "不良"):
        return build_race_jobs_page(admin_token=token, error_text="请填写本场马场状态：良 / 稍重 / 重 / 不良。")
    if kachiuma_file is None or not str(getattr(kachiuma_file, "filename", "") or "").strip():
        return build_race_jobs_page(admin_token=token, error_text="请上传 kachiuma.csv。")
    if shutuba_file is None or not str(getattr(shutuba_file, "filename", "") or "").strip():
        return build_race_jobs_page(admin_token=token, error_text="请上传 shutuba.csv。")
    try:
        lead_value = int(str(lead_minutes or "30").strip() or "30")
    except ValueError:
        lead_value = 30
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
    for artifact_type, upload in (("kachiuma", kachiuma_file), ("shutuba", shutuba_file)):
        if upload is None:
            continue
        payload = upload.file.read()
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
    return build_race_jobs_page(admin_token=token, message_text=f"已创建任务 {job['job_id']}。")


@app.post("/console/tasks/import_archive", response_class=HTMLResponse)
async def import_history_archive(
    token: str = Form(""),
    overwrite: str = Form(""),
    archive_file: UploadFile = File(None),
):
    if not _admin_token_valid(token):
        return build_race_jobs_page(
            admin_token=token,
            authorized=False,
            error_text="管理口令无效，不能导入历史数据。",
        )
    if archive_file is None or not str(getattr(archive_file, "filename", "") or "").strip():
        return build_race_jobs_page(admin_token=token, error_text="请上传 ZIP 文件。")
    filename = str(archive_file.filename or "").strip()
    if not filename.lower().endswith(".zip"):
        return build_race_jobs_page(admin_token=token, error_text="只支持 ZIP 文件。")
    try:
        archive_bytes = await archive_file.read()
        summary = _import_history_zip(
            BASE_DIR,
            archive_bytes,
            overwrite=str(overwrite or "").strip() == "1",
        )
    except zipfile.BadZipFile:
        return build_race_jobs_page(admin_token=token, error_text="ZIP 文件损坏或格式无效。")
    except Exception as exc:
        return build_race_jobs_page(admin_token=token, error_text=f"导入失败：{exc}")
    message = f"已导入 {summary['written']} 个文件，跳过 {summary['skipped']} 个。"
    sample_paths = list(summary.get("sample_paths", []) or [])
    if sample_paths:
        message += " 示例: " + ", ".join(sample_paths)
    return build_race_jobs_page(admin_token=token, message_text=message)


@app.post("/console/tasks/scan_due", response_class=HTMLResponse)
def scan_race_jobs_due(token: str = Form("")):
    if not _admin_token_valid(token):
        return build_race_jobs_page(
            admin_token=token,
            authorized=False,
            error_text="管理口令无效，无法扫描任务。",
        )
    changed = scan_due_race_jobs(BASE_DIR)
    if changed:
        return build_race_jobs_page(admin_token=token, message_text=f"已将 {len(changed)} 场比赛加入处理队列。")
    return build_race_jobs_page(admin_token=token, message_text="当前没有到点任务。")


@app.post("/console/tasks/run_due_now", response_class=HTMLResponse)
def run_due_race_jobs_now(token: str = Form("")):
    if not _admin_token_valid(token):
        return build_race_jobs_page(
            admin_token=token,
            authorized=False,
            error_text="管理口令错误，不能执行到点任务。",
        )
    summary = run_due_jobs_once()
    message_parts = [
        f"queued={int(summary.get('queued_count', 0) or 0)}",
        f"processed={int(summary.get('processed_count', 0) or 0)}",
        f"settled={int(summary.get('settled_count', 0) or 0)}",
        f"errors={len(list(summary.get('errors', []) or []))}",
    ]
    error_items = list(summary.get("errors", []) or [])
    if error_items:
        error_text = "\n".join(
            f"[{item.get('kind', 'job')}] {item.get('job_id', '-')}: {item.get('error', '')}"
            for item in error_items
        )
        return build_race_jobs_page(
            admin_token=token,
            message_text="已执行到点任务：" + ", ".join(message_parts),
            error_text=error_text,
        )
    return build_race_jobs_page(
        admin_token=token,
        message_text="已执行到点任务：" + ", ".join(message_parts),
    )


@app.get("/internal/run_due")
@app.post("/internal/run_due")
def internal_run_due(token: str = ""):
    if not _admin_token_valid(token):
        return JSONResponse({"ok": False, "error": "invalid_admin_token"}, status_code=403)
    summary = run_due_jobs_once()
    ok = not bool(list(summary.get("errors", []) or []))
    return JSONResponse({"ok": ok, **summary}, status_code=200 if ok else 500)


@app.post("/console/tasks/update", response_class=HTMLResponse)
def update_race_job_view(
    token: str = Form(""),
    job_id: str = Form(""),
    action: str = Form(""),
):
    if not _admin_token_valid(token):
        return build_race_jobs_page(
            admin_token=token,
            authorized=False,
            error_text="管理口令无效，无法修改任务。",
        )
    job = apply_race_job_action(BASE_DIR, job_id, action)
    if job is None:
        return build_race_jobs_page(admin_token=token, error_text="找不到对应的任务。")
    return build_race_jobs_page(admin_token=token, message_text=f"{job_id} 已执行动作：{action}。")


@app.post("/console/tasks/process_now", response_class=HTMLResponse)
def process_race_job_now(
    token: str = Form(""),
    job_id: str = Form(""),
):
    if not _admin_token_valid(token):
        return build_race_jobs_page(
            admin_token=token,
            authorized=False,
            error_text="管理口令无效，无法执行任务。",
        )
    try:
        from race_job_runner import process_race_job

        summary = process_race_job(BASE_DIR, job_id)
    except Exception as exc:
        try:
            from race_job_runner import fail_race_job

            fail_race_job(BASE_DIR, job_id, str(exc))
        except Exception:
            pass
        return build_race_jobs_page(admin_token=token, error_text=f"{job_id} 执行失败：{exc}")
    run_id = str((summary or {}).get("run_id", "") or "").strip()
    engine_count = len(list((summary or {}).get("policy_engines", []) or []))
    return build_race_jobs_page(
        admin_token=token,
        message_text=f"{job_id} 已完成处理。run_id={run_id or '-'} engines={engine_count}",
    )


@app.post("/console/tasks/queue_settle", response_class=HTMLResponse)
def queue_race_job_settle(
    token: str = Form(""),
    job_id: str = Form(""),
    actual_top1: str = Form(""),
    actual_top2: str = Form(""),
    actual_top3: str = Form(""),
):
    if not _admin_token_valid(token):
        return build_race_jobs_page(
            admin_token=token,
            authorized=False,
            error_text="管理口令无效，无法加入结算队列。",
        )
    names = [str(actual_top1 or "").strip(), str(actual_top2 or "").strip(), str(actual_top3 or "").strip()]
    if not all(names):
        return build_race_jobs_page(admin_token=token, error_text="请完整填写 1-3 着马名。")

    def _queue_settle(row, now_text):
        row["actual_top1"] = names[0]
        row["actual_top2"] = names[1]
        row["actual_top3"] = names[2]
        row["status"] = "queued_settle"
        row["queued_settle_at"] = now_text
        row["error_message"] = ""

    job = update_race_job(BASE_DIR, job_id, _queue_settle)
    if job is None:
        return build_race_jobs_page(admin_token=token, error_text="找不到对应的任务。")
    return build_race_jobs_page(admin_token=token, message_text=f"{job_id} 已保存赛果并加入结算队列。")


@app.post("/console/tasks/settle_now", response_class=HTMLResponse)
def settle_race_job_now(
    token: str = Form(""),
    job_id: str = Form(""),
    actual_top1: str = Form(""),
    actual_top2: str = Form(""),
    actual_top3: str = Form(""),
):
    if not _admin_token_valid(token):
        return build_race_jobs_page(
            admin_token=token,
            authorized=False,
            error_text="管理口令无效，无法结算任务。",
        )
    names = [str(actual_top1 or "").strip(), str(actual_top2 or "").strip(), str(actual_top3 or "").strip()]
    if not all(names):
        return build_race_jobs_page(admin_token=token, error_text="请完整填写 1-3 着马名。")
    try:
        from race_job_runner import settle_race_job

        summary = settle_race_job(BASE_DIR, job_id, names)
    except Exception as exc:
        try:
            from race_job_runner import fail_race_job

            fail_race_job(BASE_DIR, job_id, str(exc))
        except Exception:
            pass
        return build_race_jobs_page(admin_token=token, error_text=f"{job_id} 结算失败：{exc}")
    return build_race_jobs_page(
        admin_token=token,
        message_text=f"{job_id} 已完成结算。run_id={str((summary or {}).get('run_id', '') or '-')}",
    )


@app.post("/view_run", response_class=HTMLResponse)
def view_run(
    run_id: str = Form(""),
    scope_key: str = Form(""),
    token: str = Form(""),
):
    run_id = run_id.strip()
    scope_key = normalize_scope_key(scope_key)
    run_row = None
    if not scope_key:
        scope_key, run_row = infer_scope_and_run(run_id)
    if not scope_key:
        return render_page("", error_text="Enter Run ID or Race ID to view history.", admin_token=token)
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
            admin_token=token,
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
            admin_token=token,
        )
    return render_page(
        scope_key,
        selected_run_id=resolved_run_id,
        summary_run_id=resolved_run_id,
        admin_token=token,
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
    if not _admin_token_valid(token):
        return _admin_execution_denied("管理口令错误，不能执行 Pipeline。", scope_key=scope_key, token=token)
    scope_key = normalize_scope_key(scope_key)
    if not scope_key:
        return render_page("", error_text="Please select a data scope.", admin_token=token)
    race_id = normalize_race_id(race_id or race_url)
    history_url = history_url.strip()
    location = str(location or "").strip()
    race_date = str(race_date or "").strip()
    if not race_id or not history_url or not location:
        return render_page(
            scope_key,
            error_text="Race ID, History URL, and Location are required.",
            admin_token=token,
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
        admin_token=token,
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
    if not _admin_token_valid(token):
        return _admin_execution_denied(
            "管理口令错误，不能录入 Predictor 结果。",
            scope_key=scope_key,
            token=token,
            selected_run_id=str(run_id or "").strip(),
        )
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
            admin_token=token,
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
        admin_token=token,
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
    if not _admin_token_valid(token):
        return _admin_execution_denied(
            "管理口令错误，不能执行 LLM buy。",
            scope_key=scope_key,
            token=token,
            selected_run_id=str(run_id or "").strip(),
        )
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return render_page(
            scope_norm or scope_key,
            error_text="Run ID / Race ID not found for LLM buy.",
            selected_run_id=resolved_run_id or str(run_id or "").strip(),
            admin_token=token,
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
            admin_token=token,
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
        admin_token=token,
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
    if not _admin_token_valid(token):
        return _admin_execution_denied(
            "管理口令错误，不能执行全部 LLM。",
            scope_key=scope_key,
            token=token,
            selected_run_id=str(run_id or "").strip(),
        )
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return render_page(
            scope_norm or scope_key,
            error_text="Run ID / Race ID not found for LLM buy.",
            selected_run_id=resolved_run_id or str(run_id or "").strip(),
            admin_token=token,
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
            admin_token=token,
        )
    output_text = "\n\n".join(block for block in result_blocks if str(block).strip())
    if error_blocks:
        output_text = "\n\n".join([output_text] + error_blocks if output_text else error_blocks)
    return render_page(
        scope_norm,
        output_text=output_text,
        selected_run_id=resolved_run_id,
        admin_token=token,
    )


@app.post("/topup_all_llm_budget", response_class=HTMLResponse)
def topup_all_llm_budget(
    token: str = Form(""),
    scope_key: str = Form(""),
    run_id: str = Form(""),
):
    if not _admin_token_valid(token):
        return _admin_execution_denied(
            "管理口令错误，不能执行资金充值。",
            scope_key=scope_key,
            token=token,
            selected_run_id=str(run_id or "").strip(),
        )
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return render_page(
            scope_norm or scope_key,
            error_text="Run ID / Race ID not found for LLM bankroll top-up.",
            selected_run_id=resolved_run_id or str(run_id or "").strip(),
            admin_token=token,
        )
    ledger_date = extract_ledger_date(resolved_run_id, run_row.get("timestamp", ""))
    amount_yen = 10000
    lines = [f"[llm_budget_topup] ledger_date={ledger_date} amount_yen={amount_yen} engines=4"]
    for engine in ("gemini", "siliconflow", "openai", "grok"):
        summary = add_bankroll_topup(BASE_DIR, ledger_date, amount_yen, policy_engine=engine)
        lines.append(
            "[topup][{engine}] available_bankroll_yen={available} topup_yen={topup}".format(
                engine=engine,
                available=int(summary.get("available_bankroll_yen", 0) or 0),
                topup=int(summary.get("topup_yen", 0) or 0),
            )
        )
    return render_page(
        scope_norm,
        output_text="\n".join(lines),
        selected_run_id=resolved_run_id,
        admin_token=token,
    )


@app.post("/reset_llm_state", response_class=HTMLResponse)
def reset_llm_state(token: str = Form("")):
    if not _admin_token_valid(token):
        return _admin_execution_denied("管理口令错误，不能重置 LLM 状态。", token=token)
    summary = reset_llm_state_files(BASE_DIR)
    return render_page(
        "",
        output_text=json.dumps(summary, ensure_ascii=False, indent=2),
        admin_token=token,
    )


@app.post("/optimize_params", response_class=HTMLResponse)
def optimize_params(token: str = Form("")):
    if not _admin_token_valid(token):
        return _admin_execution_denied("管理口令错误，不能执行 optimize_params。", token=token)
    code, output = run_script(OPTIMIZE_PARAMS, inputs=[""], extra_blanks=1)
    label = f"Exit code: {code}"
    return render_page("", output_text=f"{label}\n{output}", admin_token=token)


@app.post("/optimize_predictor", response_class=HTMLResponse)
def optimize_predictor(token: str = Form("")):
    if not _admin_token_valid(token):
        return _admin_execution_denied("管理口令错误，不能执行 optimize_predictor。", token=token)
    code, output = run_script(OPTIMIZE_PREDICTOR, inputs=[""], extra_blanks=1)
    label = f"Exit code: {code}"
    return render_page("", output_text=f"{label}\n{output}", admin_token=token)


@app.post("/offline_eval", response_class=HTMLResponse)
def offline_eval(token: str = Form(""), window: str = Form("")):
    if not _admin_token_valid(token):
        return _admin_execution_denied("管理口令错误，不能执行 offline_eval。", token=token)
    inputs = [window, ""]
    code, output = run_script(OFFLINE_EVAL, inputs=inputs, extra_blanks=1)
    label = f"Exit code: {code}"
    return render_page("", output_text=f"{label}\n{output}", admin_token=token)


@app.post("/init_update", response_class=HTMLResponse)
def init_update(token: str = Form("")):
    if not _admin_token_valid(token):
        return _admin_execution_denied("管理口令错误，不能执行 init_update。", token=token)
    code, output = run_script(INIT_UPDATE, inputs=[""], extra_blanks=1)
    label = f"Exit code: {code}"
    return render_page("", output_text=f"{label}\n{output}", admin_token=token)


@app.post("/init_update_reset", response_class=HTMLResponse)
def init_update_reset(token: str = Form("")):
    if not _admin_token_valid(token):
        return _admin_execution_denied("管理口令错误，不能执行 init_update --reset。", token=token)
    code, output = run_script(INIT_UPDATE, inputs=[""], args=["--reset"], extra_blanks=1)
    label = f"Exit code: {code}"
    return render_page("", output_text=f"{label}\n{output}", admin_token=token)

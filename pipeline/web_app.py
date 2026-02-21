import csv
import html
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from surface_scope import get_data_dir, migrate_legacy_data, normalize_scope_key


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
    if not fieldnames:
        return []
    return [name.lstrip("\ufeff") if name else name for name in fieldnames]


def normalize_csv_rows(rows, fieldnames):
    norm_fieldnames = normalize_csv_fieldnames(fieldnames)
    if norm_fieldnames == fieldnames:
        return rows, fieldnames
    mapping = dict(zip(fieldnames, norm_fieldnames))
    out = []
    for row in rows:
        new_row = {}
        for key, value in row.items():
            new_key = mapping.get(key, key.lstrip("\ufeff") if key else key)
            new_row[new_key] = value
        out.append(new_row)
    return out, norm_fieldnames


def load_runs_with_header(scope_key):
    migrate_legacy_data(BASE_DIR, scope_key)
    runs_path = get_data_dir(BASE_DIR, scope_key) / "runs.csv"
    if not runs_path.exists():
        return [], [], "utf-8"
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            with open(runs_path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
                rows, fieldnames = normalize_csv_rows(rows, fieldnames)
                return rows, fieldnames, enc
        except UnicodeDecodeError:
            continue
    return [], [], "utf-8"


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
        return '<option value="" disabled>请选择数据范围</option>'
    for item in get_recent_runs(scope_key, limit=DEFAULT_RUN_LIMIT):
        value = html.escape(item["run_id"])
        label = html.escape(item["label"])
        selected_attr = ' selected' if selected_run_id and item["run_id"] == selected_run_id else ""
        options.append(f'<option value="{value}"{selected_attr}>{label}</option>')
    if not options:
        options.append('<option value="" disabled>无记录</option>')
    return "\n".join(options)

def resolve_run(run_id, scope_key):
    runs = load_runs(scope_key)
    if not runs:
        return None
    if not run_id:
        return runs[-1]
    for row in runs:
        if row.get("run_id") == run_id:
            return row
    return None


def resolve_latest_run_by_race_id(race_id, scope_key):
    race_id = normalize_race_id(race_id)
    if not race_id:
        return None
    runs = load_runs(scope_key)
    for row in reversed(runs):
        if normalize_race_id(row.get("race_id", "")) == race_id:
            return row
    return None


def infer_run_id_from_path(path):
    if not path:
        return ""
    match = re.search(r"(\d{8}_\d{6})", str(path))
    return match.group(1) if match else ""


def infer_run_id_from_row(run_row):
    if not run_row:
        return ""
    for key in (
        "predictions_path",
        "odds_path",
        "wide_odds_path",
        "fuku_odds_path",
        "quinella_odds_path",
        "trifecta_odds_path",
        "plan_path",
    ):
        run_id = infer_run_id_from_path(run_row.get(key, ""))
        if run_id:
            return run_id
    return ""


def update_run_row_fields(scope_key, run_row, updates):
    if not run_row or not updates:
        return False
    runs_path = get_data_dir(BASE_DIR, scope_key) / "runs.csv"
    rows, fieldnames, enc = load_runs_with_header(scope_key)
    if not rows or not fieldnames:
        return False
    for key in updates:
        if key not in fieldnames:
            fieldnames.append(key)
    target_run_id = str(run_row.get("run_id", "")).strip()
    target_race_id = normalize_race_id(run_row.get("race_id", ""))
    target_timestamp = str(run_row.get("timestamp", "")).strip()
    target_pred_path = str(run_row.get("predictions_path", "")).strip()
    matched = False
    for row in rows:
        row_run_id = str(row.get("run_id", "")).strip()
        if target_run_id and row_run_id == target_run_id:
            row.update(updates)
            matched = True
            break
        row_race_id = normalize_race_id(row.get("race_id", ""))
        row_ts = str(row.get("timestamp", "")).strip()
        if target_race_id and target_timestamp:
            if row_race_id != target_race_id or row_ts != target_timestamp:
                continue
            if target_pred_path:
                row_pred = str(row.get("predictions_path", "")).strip()
                if row_pred != target_pred_path:
                    continue
            row.update(updates)
            matched = True
            break
    if not matched:
        return False
    with open(runs_path, "w", newline="", encoding=enc, errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return True


def resolve_plan_path(scope_key, run_id, run_row):
    path = str(run_row.get("plan_path", "")).strip() if run_row else ""
    if path:
        return Path(path)
    race_id = str(run_row.get("race_id", "") or "") if run_row else ""
    if race_id:
        race_dir = get_data_dir(BASE_DIR, scope_key) / race_id
        return race_dir / f"bet_plan_{run_id}_{race_id}.csv"
    return get_data_dir(BASE_DIR, scope_key) / f"bet_plan_{run_id}.csv"


def update_run_plan_path(scope_key, run_id, plan_path):
    runs_path = get_data_dir(BASE_DIR, scope_key) / "runs.csv"
    rows, fieldnames, enc = load_runs_with_header(scope_key)
    if not rows or not fieldnames:
        return False
    updated = False
    if "plan_path" not in fieldnames:
        fieldnames.append("plan_path")
    for row in rows:
        if row.get("run_id") == run_id:
            row["plan_path"] = str(plan_path)
            updated = True
            break
    if not updated:
        return False
    with open(runs_path, "w", newline="", encoding=enc, errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return True


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
            return False, f"Failed to update fuku odds file: {exc}", []
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
    if not scope_key or not id_text:
        return None
    runs = load_runs(scope_key)
    if not runs:
        return None
    id_text = str(id_text).strip()
    race_id = normalize_race_id(id_text)
    for row in reversed(runs):
        run_id = str(row.get("run_id", "")).strip()
        if run_id and run_id == id_text:
            return row
        if race_id and normalize_race_id(row.get("race_id", "")) == race_id:
            return row
    return None


def infer_scope_and_run(id_text):
    id_text = str(id_text or "").strip()
    if not id_text:
        return "", None
    for scope_key in ("central_dirt", "central_turf", "local"):
        run_row = find_run_in_scope(scope_key, id_text)
        if run_row:
            return scope_key, run_row
    return "", None


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
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_odds_snapshot(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                out = {}
                for row in reader:
                    horse_no = str(row.get("horse_no", "")).strip()
                    name = str(row.get("name", "")).strip()
                    odds_raw = str(row.get("odds", "")).strip()
                    key = ""
                    if horse_no:
                        key = f"no:{horse_no}"
                    elif name:
                        key = f"name:{normalize_name(name)}"
                    if not key:
                        continue
                    out[key] = {
                        "horse_no": horse_no,
                        "name": name,
                        "odds_raw": odds_raw,
                        "odds_val": parse_odds_value(odds_raw),
                    }
                return out
        except UnicodeDecodeError:
            continue
    return {}


def odds_changed(prev_item, curr_item):
    if prev_item.get("odds_val") is not None and curr_item.get("odds_val") is not None:
        return abs(prev_item["odds_val"] - curr_item["odds_val"]) > 1e-9
    return str(prev_item.get("odds_raw", "")) != str(curr_item.get("odds_raw", ""))


def odds_item_label(item):
    horse_no = str(item.get("horse_no", "")).strip()
    name = str(item.get("name", "")).strip()
    if horse_no and name:
        return f"{horse_no} {name}"
    return horse_no or name or "(unknown)"


def odds_sort_key(item):
    horse_no = str(item.get("horse_no", "")).strip()
    name = str(item.get("name", "")).strip()
    try:
        return (0, int(float(horse_no)), name)
    except (TypeError, ValueError):
        return (1, horse_no, name)


def format_odds_diff(prev_snapshot, curr_snapshot, limit=50):
    if not prev_snapshot:
        return ["odds_diff: no previous odds"]
    if not curr_snapshot:
        return ["odds_diff: no new odds"]
    added = []
    removed = []
    changed = []
    for key, curr in curr_snapshot.items():
        prev = prev_snapshot.get(key)
        if not prev:
            added.append(curr)
        elif odds_changed(prev, curr):
            changed.append((prev, curr))
    for key, prev in prev_snapshot.items():
        if key not in curr_snapshot:
            removed.append(prev)
    lines = [
        f"odds_diff: changed={len(changed)} added={len(added)} removed={len(removed)}"
    ]
    for prev, curr in sorted(changed, key=lambda pair: odds_sort_key(pair[1])):
        lines.append(
            f"~ {odds_item_label(curr)}: {prev.get('odds_raw', '')} -> {curr.get('odds_raw', '')}"
        )
    for item in sorted(added, key=odds_sort_key):
        lines.append(f"+ {odds_item_label(item)}: {item.get('odds_raw', '')}")
    for item in sorted(removed, key=odds_sort_key):
        lines.append(f"- {odds_item_label(item)}: {item.get('odds_raw', '')}")
    if len(lines) > limit + 1:
        more = len(lines) - (limit + 1)
        return lines[: limit + 1] + [f"... truncated {more} lines"]
    return lines


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
    return build_table_html(rows, ["指标", "数值"], title)


def load_top5_table(scope_key, run_id, run_row=None):
    path = ""
    if run_row:
        path = run_row.get("predictions_path", "")
    if not path:
        path = str(get_data_dir(BASE_DIR, scope_key) / f"predictions_{run_id}.csv")
    path = Path(path)
    rows = load_csv_rows(path)
    if not rows:
        return [], []
    score_key = pick_score_key(rows)
    if score_key:
        rows_sorted = sorted(rows, key=lambda r: to_float(r.get(score_key)), reverse=True)
    else:
        rows_sorted = rows
    top_rows = rows_sorted[:5]
    preferred_cols = [
        "HorseName",
        "Top3Prob_model",
        "Top3Prob_lgbm",
        "Top3Prob_lr",
        "agg_score",
        "best_TimeIndexEff",
        "avg_TimeIndexEff",
        "dist_close",
    ]
    if score_key and score_key not in preferred_cols:
        preferred_cols.append(score_key)
    global_cols = {"confidence_score", "rank_ema", "ev_ema", "risk_score"}
    columns = [col for col in preferred_cols if col in rows[0]]
    if not columns:
        columns = [col for col in rows[0].keys() if col not in global_cols][:6]
    else:
        columns = columns[:6]
    return top_rows, columns


def load_prediction_summary(scope_key, run_id, run_row=None):
    path = ""
    if run_row:
        path = run_row.get("predictions_path", "")
    if not path:
        path = str(get_data_dir(BASE_DIR, scope_key) / f"predictions_{run_id}.csv")
    path = Path(path)
    rows = load_csv_rows(path)
    if not rows:
        return []
    row = rows[0]
    summary_keys = ["confidence_score", "rank_ema", "ev_ema", "risk_score"]
    summary = []
    for key in summary_keys:
        if key in row:
            summary.append({"指标": key, "数值": row.get(key, "")})
    summary.extend(load_mc_uncertainty_summary(scope_key, run_id, run_row))
    return summary


def load_mc_uncertainty_summary(scope_key, run_id, run_row=None):
    path = ""
    if run_row:
        path = run_row.get("plan_path", "")
    if not path:
        path = str(get_data_dir(BASE_DIR, scope_key) / f"bet_plan_{run_id}.csv")
    rows = load_csv_rows(Path(path))
    if not rows:
        return []
    filtered = [
        row
        for row in rows
        if str(row.get("bet_type", "")).strip().lower() != "trifecta_rec"
    ]
    filtered = [
        row
        for row in filtered
        if row.get("hit_prob_se") or row.get("hit_prob_ci95_low") or row.get("hit_prob_ci95_high")
    ]
    if not filtered:
        return []
    target = max(filtered, key=lambda r: to_float(r.get("hit_prob_est")))
    bet_type = str(target.get("bet_type", "")).strip()
    horse_no = str(target.get("horse_no", "")).strip()
    label = f"{bet_type} {horse_no}".strip()
    summary = [
        {"指标": f"MC SE ({label})", "数值": target.get("hit_prob_se", "")},
        {"指标": f"MC CI95 Low ({label})", "数值": target.get("hit_prob_ci95_low", "")},
        {"指标": f"MC CI95 High ({label})", "数值": target.get("hit_prob_ci95_high", "")},
    ]


    return summary


def load_profit_summary(scope_key):
    path = get_data_dir(BASE_DIR, scope_key) / "results.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    latest = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id:
            latest[run_id] = row
    rows = list(latest.values())
    total_profit = 0
    total_base = 0
    sample_count = 0
    for row in rows:
        try:
            profit = int(float(row.get("profit_yen", 0)))
        except (TypeError, ValueError):
            profit = 0
        try:
            base = int(float(row.get("base_amount", 0)))
        except (TypeError, ValueError):
            base = 0
        total_profit += profit
        total_base += base
        sample_count += 1
    roi = ""
    if total_base > 0:
        roi = round((total_base + total_profit) / total_base, 4)
    return [
        {"指标": "记录场次", "数值": sample_count},
        {"指标": "累计投入 (yen)", "数值": total_base},
        {"指标": "累计盈亏 (yen)", "数值": total_profit},
        {"指标": "累计 ROI", "数值": roi},
    ]


def extract_date_prefix(value):
    match = re.match(r"(\d{8})", str(value or ""))
    if not match:
        return None
    raw = match.group(1)
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def extract_run_date(run_id):
    return extract_date_prefix(run_id)


def extract_year_prefix(value):
    match = re.match(r"(\d{4})", str(value or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def build_run_race_map(scope_key):
    runs = load_runs(scope_key)
    return {row.get("run_id", ""): row.get("race_id", "") for row in runs if row.get("run_id")}


def load_daily_profit_summary(scope_key, days=30):
    path = get_data_dir(BASE_DIR, scope_key) / "results.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    latest = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id:
            latest[run_id] = row
    rows = list(latest.values())
    run_race_map = build_run_race_map(scope_key)
    cutoff = None
    if days is not None:
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 30
        if days > 0:
            cutoff = datetime.now().date() - timedelta(days=days - 1)
    daily = {}
    for row in rows:
        run_id = row.get("run_id", "")
        race_id = run_race_map.get(run_id, "")
        race_year = extract_year_prefix(race_id)
        if race_year is not None and race_year < MIN_RACE_YEAR:
            continue
        date_obj = extract_run_date(run_id)
        if not date_obj:
            continue
        if cutoff and date_obj < cutoff:
            continue
        try:
            profit = int(float(row.get("profit_yen", 0)))
        except (TypeError, ValueError):
            profit = 0
        try:
            base = int(float(row.get("base_amount", 0)))
        except (TypeError, ValueError):
            base = 0
        item = daily.setdefault(date_obj, {"runs": 0, "profit": 0, "base": 0})
        item["runs"] += 1
        item["profit"] += profit
        item["base"] += base
    if not daily:
        return []
    items = sorted(daily.items(), key=lambda pair: pair[0], reverse=True)
    out = []
    for date_obj, item in items:
        base = item["base"]
        roi = round((base + item["profit"]) / base, 4) if base > 0 else ""
        out.append(
            {
                "date": date_obj.strftime("%Y-%m-%d"),
                "runs": item["runs"],
                "profit_yen": item["profit"],
                "base_amount": base,
                "roi": roi,
            }
        )
    return out


def normalize_name(value):
    return "".join(str(value or "").split())


def pick_score_key(rows):
    if not rows:
        return ""
    sample = rows[0]
    for key in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if key in sample:
            return key
    return ""


def load_top5_names(path):
    rows = load_csv_rows(path)
    if not rows:
        return []
    score_key = pick_score_key(rows)
    if score_key:
        rows = sorted(rows, key=lambda r: to_float(r.get(score_key)), reverse=True)
    names = []
    seen = set()
    for row in rows:
        name = row.get("HorseName") or row.get("name")
        if not name:
            continue
        norm = normalize_name(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        names.append(name)
        if len(names) >= 5:
            break
    return names


def load_odds_name_to_no(path):
    rows = load_csv_rows(path)
    if not rows:
        return {}
    out = {}
    for row in rows:
        name = row.get("name") or row.get("HorseName") or row.get("horse_name")
        horse_no = row.get("horse_no") or row.get("horse")
        if not name or horse_no is None:
            continue
        try:
            horse_no = int(float(horse_no))
        except (TypeError, ValueError):
            continue
        out[normalize_name(name)] = horse_no
    return out


def load_wide_odds_map(path):
    rows = load_csv_rows(path)
    if not rows:
        return {}
    odds_key = "odds_mid" if "odds_mid" in rows[0] else "odds"
    out = {}
    for row in rows:
        a = row.get("horse_no_a") or row.get("horse_a")
        b = row.get("horse_no_b") or row.get("horse_b")
        if a is None or b is None:
            continue
        try:
            a_i = int(float(a))
            b_i = int(float(b))
        except (TypeError, ValueError):
            continue
        try:
            odds = float(row.get(odds_key, 0))
        except (TypeError, ValueError):
            odds = 0.0
        if odds <= 0:
            continue
        if a_i > b_i:
            a_i, b_i = b_i, a_i
        out[(a_i, b_i)] = odds
    return out


def resolve_pred_path(scope_key, run_id, run_row):
    path = run_row.get("predictions_path", "") if run_row else ""
    if not path:
        path = str(get_data_dir(BASE_DIR, scope_key) / f"predictions_{run_id}.csv")
    return Path(path)


def resolve_odds_path(scope_key, run_id, run_row):
    path = run_row.get("odds_path", "") if run_row else ""
    if path:
        return Path(path)
    race_id = str(run_row.get("race_id", "") or "")
    if race_id:
        race_dir = get_data_dir(BASE_DIR, scope_key) / race_id
        return race_dir / f"odds_{run_id}_{race_id}.csv"
    return None


def resolve_wide_odds_path(scope_key, run_id, run_row):
    path = run_row.get("wide_odds_path", "") if run_row else ""
    if path:
        return Path(path)
    race_id = str(run_row.get("race_id", "") or "")
    if race_id:
        race_dir = get_data_dir(BASE_DIR, scope_key) / race_id
        return race_dir / f"wide_odds_{run_id}_{race_id}.csv"
    return None


def resolve_run_asset_path(scope_key, run_id, run_row, field_name, prefix):
    path = str(run_row.get(field_name, "")).strip() if run_row else ""
    if path:
        return Path(path)
    race_id = str(run_row.get("race_id", "") or "") if run_row else ""
    if race_id:
        race_dir = get_data_dir(BASE_DIR, scope_key) / race_id
        return race_dir / f"{prefix}_{run_id}_{race_id}.csv"
    return None


def load_race_results(scope_key):
    path = get_data_dir(BASE_DIR, scope_key) / "race_results.csv"
    rows = load_csv_rows(path)
    if not rows:
        return {}
    return {row.get("run_id", ""): row for row in rows if row.get("run_id")}


def compute_wide_box_profit(scope_key, run_row, race_row, budget_yen=1000):
    run_id = run_row.get("run_id", "")
    if not run_id:
        return None
    pred_path = resolve_pred_path(scope_key, run_id, run_row)
    if not pred_path.exists():
        return None
    top5_names = load_top5_names(pred_path)
    if len(top5_names) < 5:
        return None
    odds_path = resolve_odds_path(scope_key, run_id, run_row)
    if not odds_path or not odds_path.exists():
        return None
    name_to_no = load_odds_name_to_no(odds_path)
    if not name_to_no:
        return None
    wide_path = resolve_wide_odds_path(scope_key, run_id, run_row)
    if not wide_path or not wide_path.exists():
        return None
    wide_map = load_wide_odds_map(wide_path)
    if not wide_map:
        return None

    actual_names = [
        race_row.get("actual_top1"),
        race_row.get("actual_top2"),
        race_row.get("actual_top3"),
    ]
    actual_norm = [normalize_name(n) for n in actual_names if n]
    if len(actual_norm) < 3:
        return None
    try:
        actual_nos = {name_to_no[n] for n in actual_norm}
    except KeyError:
        return None
    if len(actual_nos) < 2:
        return None

    pred_norm = [normalize_name(n) for n in top5_names]
    try:
        pred_nos = [name_to_no[n] for n in pred_norm]
    except KeyError:
        return None
    combos = list(combinations(pred_nos, 2))
    if not combos:
        return None
    per_ticket = int(budget_yen // len(combos))
    if per_ticket <= 0:
        return None
    total_amount = per_ticket * len(combos)
    total_payout = 0.0
    for a, b in combos:
        if a not in actual_nos or b not in actual_nos:
            continue
        key = (a, b) if a <= b else (b, a)
        odds = wide_map.get(key, 0.0)
        if odds:
            total_payout += per_ticket * odds
    profit = int(round(total_payout - total_amount))
    return {"amount": total_amount, "profit": profit}


def load_wide_box_daily_profit_summary(scope_key, days=30, budget_yen=1000):
    rows = load_csv_rows(get_data_dir(BASE_DIR, scope_key) / "wide_box_results.csv")
    run_race_map = build_run_race_map(scope_key)
    if rows:
        cutoff = None
        if days is not None:
            try:
                days = int(days)
            except (TypeError, ValueError):
                days = 30
            if days > 0:
                cutoff = datetime.now().date() - timedelta(days=days - 1)
        daily = {}
        for row in rows:
            run_id = row.get("run_id", "")
            race_id = run_race_map.get(run_id, "")
            race_year = extract_year_prefix(race_id)
            if race_year is not None and race_year < MIN_RACE_YEAR:
                continue
            date_obj = extract_run_date(run_id)
            if not date_obj:
                continue
            if cutoff and date_obj < cutoff:
                continue
            try:
                profit = int(float(row.get("profit_yen", 0)))
            except (TypeError, ValueError):
                profit = 0
            try:
                base = int(float(row.get("amount_yen", 0)))
            except (TypeError, ValueError):
                base = 0
            item = daily.setdefault(date_obj, {"runs": 0, "profit": 0, "base": 0})
            item["runs"] += 1
            item["profit"] += profit
            item["base"] += base
        if not daily:
            return []
        items = sorted(daily.items(), key=lambda pair: pair[0], reverse=True)
        out = []
        for date_obj, item in items:
            base = item["base"]
            roi = round((base + item["profit"]) / base, 4) if base > 0 else ""
            out.append(
                {
                    "date": date_obj.strftime("%Y-%m-%d"),
                    "runs": item["runs"],
                    "profit_yen": item["profit"],
                    "base_amount": base,
                    "roi": roi,
                }
            )
        return out

    runs = load_runs(scope_key)
    if not runs:
        return []
    results = load_race_results(scope_key)
    cutoff = None
    if days is not None:
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 30
        if days > 0:
            cutoff = datetime.now().date() - timedelta(days=days - 1)
    daily = {}
    for run in runs:
        run_id = run.get("run_id", "")
        race_id = run.get("race_id", "")
        race_year = extract_year_prefix(race_id)
        if race_year is not None and race_year < MIN_RACE_YEAR:
            continue
        date_obj = extract_run_date(run_id)
        if not date_obj:
            continue
        if cutoff and date_obj < cutoff:
            continue
        race_row = results.get(run_id)
        if not race_row:
            continue
        info = compute_wide_box_profit(scope_key, run, race_row, budget_yen=budget_yen)
        if not info:
            continue
        item = daily.setdefault(date_obj, {"runs": 0, "profit": 0, "base": 0})
        item["runs"] += 1
        item["profit"] += info["profit"]
        item["base"] += info["amount"]
    if not daily:
        return []
    items = sorted(daily.items(), key=lambda pair: pair[0], reverse=True)
    out = []
    for date_obj, item in items:
        base = item["base"]
        roi = round((base + item["profit"]) / base, 4) if base > 0 else ""
        out.append(
            {
                "date": date_obj.strftime("%Y-%m-%d"),
                "runs": item["runs"],
                "profit_yen": item["profit"],
                "base_amount": base,
                "roi": roi,
            }
        )
    return out


def load_bet_type_summary(scope_key):
    path = get_data_dir(BASE_DIR, scope_key) / "bet_type_stats.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    stats = {}
    for row in rows:
        bet_type = str(row.get("bet_type", "")).strip() or "unknown"
        bets = int(float(row.get("bets", 0) or 0))
        hits = int(float(row.get("hits", 0) or 0))
        amount = int(float(row.get("amount_yen", 0) or 0))
        est_profit = int(float(row.get("est_profit_yen", 0) or 0))
        item = stats.setdefault(bet_type, {"bets": 0, "hits": 0, "amount": 0, "est_profit": 0})
        item["bets"] += bets
        item["hits"] += hits
        item["amount"] += amount
        item["est_profit"] += est_profit
    out = []
    for bet_type, item in sorted(stats.items()):
        hit_rate = round(item["hits"] / item["bets"], 4) if item["bets"] else ""
        out.append(
            {
                "bet_type": bet_type,
                "bets": item["bets"],
                "hits": item["hits"],
                "hit_rate": hit_rate,
                "amount_yen": item["amount"],
                "est_profit_yen": item["est_profit"],
            }
        )
    return out


def load_bet_type_profit_summary(scope_key):
    path = get_data_dir(BASE_DIR, scope_key) / "bet_type_stats.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    run_race_map = build_run_race_map(scope_key)
    labels = {"win": "win (单胜)", "place": "place (复胜)", "wide": "wide"}
    totals = {key: {"amount": 0, "profit": 0} for key in labels}
    for row in rows:
        bet_type = str(row.get("bet_type", "")).strip().lower()
        if bet_type not in totals:
            continue
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        race_id = run_race_map.get(run_id, "")
        race_year = extract_year_prefix(race_id)
        if race_year is None or race_year < MIN_RACE_YEAR:
            continue
        try:
            amount = int(float(row.get("amount_yen", 0) or 0))
        except (TypeError, ValueError):
            amount = 0
        try:
            profit = int(float(row.get("est_profit_yen", 0) or 0))
        except (TypeError, ValueError):
            profit = 0
        totals[bet_type]["amount"] += amount
        totals[bet_type]["profit"] += profit
    out = []
    for bet_type in ("win", "place", "wide"):
        item = totals.get(bet_type)
        if not item:
            continue
        amount = item["amount"]
        profit = item["profit"]
        roi = round((amount + profit) / amount, 4) if amount > 0 else ""
        out.append(
            {
                "bet_type": labels[bet_type],
                "amount_yen": amount,
                "est_profit_yen": profit,
                "roi": roi,
            }
        )
    return out


def compute_top5_hit_count(scope_key, row):
    hit_count = to_int_or_none(row.get("top5_hit_count"))
    if hit_count is not None:
        return hit_count
    run_id = str(row.get("run_id", "") or "")
    pred_path = row.get("predictions_path", "")
    if not pred_path and run_id:
        pred_path = str(get_data_dir(BASE_DIR, scope_key) / f"predictions_{run_id}.csv")
    if not pred_path:
        return None
    pred_path = Path(pred_path)
    if not pred_path.exists():
        return None
    top5_names = load_top5_names(pred_path)
    if not top5_names:
        return None
    actual_names = [
        row.get("actual_top1"),
        row.get("actual_top2"),
        row.get("actual_top3"),
    ]
    actual_norm = [normalize_name(n) for n in actual_names if n]
    if len(actual_norm) < 3:
        return None
    pred_norm = [normalize_name(n) for n in top5_names if n]
    return len(set(pred_norm) & set(actual_norm))


def load_predictor_summary(scope_key):
    path = get_data_dir(BASE_DIR, scope_key) / "predictor_results.csv"
    rows = load_csv_rows(path)
    if not rows:
        return []
    total = len(rows)
    top1_hit = sum(int(float(r.get("top1_hit", 0) or 0)) for r in rows)
    top1_in_top3 = sum(int(float(r.get("top1_in_top3", 0) or 0)) for r in rows)
    top3_exact = sum(int(float(r.get("top3_exact", 0) or 0)) for r in rows)
    top3_hit = sum(int(float(r.get("top3_hit_count", 0) or 0)) for r in rows)
    top5_hit = 0
    top5_total = 0
    for row in rows:
        hit_count = compute_top5_hit_count(scope_key, row)
        if hit_count is None:
            continue
        top5_hit += hit_count
        top5_total += 1
    top1_rate = round(top1_hit / total, 4) if total else ""
    top1_in_top3_rate = round(top1_in_top3 / total, 4) if total else ""
    top3_exact_rate = round(top3_exact / total, 4) if total else ""
    top3_hit_rate = round(top3_hit / (3 * total), 4) if total else ""
    top5_hit_rate = round(top5_hit / (3 * top5_total), 4) if top5_total else ""
    summary = [
        {"指标": "样本场次", "数值": total},
        {"指标": "Top3 命中率", "数值": top3_hit_rate},
        {"指标": "Top1 命中率", "数值": top1_rate},
        {"指标": "Top1 入 Top3", "数值": top1_in_top3_rate},
        {"指标": "Top3 全中率", "数值": top3_exact_rate},
    ]
    if top5_hit_rate != "":
        summary.insert(2, {"指标": "Top5 入 Top3 命中率", "数值": top5_hit_rate})
    return summary


def load_run_result_summary(scope_key, run_id):
    path = get_data_dir(BASE_DIR, scope_key) / "results.csv"
    rows = load_csv_rows(path)
    row = None
    for item in rows:
        if item.get("run_id") == run_id:
            row = item
    if not row:
        return []
    return [
        {"指标": "本场盈亏 (yen)", "数值": row.get("profit_yen", "")},
        {"指标": "本场投入 (yen)", "数值": row.get("base_amount", "")},
        {"指标": "本场 ROI", "数值": row.get("roi", "")},
    ]


def load_run_bet_type_summary(scope_key, run_id):
    path = get_data_dir(BASE_DIR, scope_key) / "bet_type_stats.csv"
    rows = [r for r in load_csv_rows(path) if r.get("run_id") == run_id]
    if not rows:
        return []
    out = []
    for row in rows:
        out.append(
            {
                "bet_type": row.get("bet_type", ""),
                "bets": row.get("bets", ""),
                "hits": row.get("hits", ""),
                "hit_rate": row.get("hit_rate", ""),
                "amount_yen": row.get("amount_yen", ""),
                "est_profit_yen": row.get("est_profit_yen", ""),
            }
        )
    return out


def load_run_bet_ticket_summary(scope_key, run_id):
    path = get_data_dir(BASE_DIR, scope_key) / "bet_ticket_results.csv"
    rows = [r for r in load_csv_rows(path) if r.get("run_id") == run_id]
    if not rows:
        return []
    out = []
    for row in rows:
        try:
            amount = int(float(row.get("amount_yen", 0) or 0))
        except ValueError:
            amount = 0
        try:
            est_payout = int(float(row.get("est_payout_yen", 0) or 0))
        except ValueError:
            est_payout = 0
        out.append(
            {
                "bet_type": row.get("bet_type", ""),
                "horse_no": row.get("horse_no", ""),
                "horse_name": row.get("horse_name", ""),
                "amount_yen": amount,
                "hit": row.get("hit", ""),
                "est_payout_yen": est_payout,
                "profit_yen": est_payout - amount,
            }
        )
    return out


def load_run_predictor_summary(scope_key, run_id):
    path = get_data_dir(BASE_DIR, scope_key) / "predictor_results.csv"
    rows = load_csv_rows(path)
    row = next((r for r in rows if r.get("run_id") == run_id), None)
    if not row:
        return []
    top3_hit = to_float(row.get("top3_hit_count")) / 3.0 if row.get("top3_hit_count") is not None else ""
    top5_hit_count = compute_top5_hit_count(scope_key, row)
    top5_hit = round(top5_hit_count / 3.0, 4) if top5_hit_count is not None else ""
    summary = [
        {"指标": "本场 Top3 命中率", "数值": round(top3_hit, 4) if top3_hit != "" else ""},
        {"指标": "本场 Top1 命中", "数值": row.get("top1_hit", "")},
        {"指标": "Top1 入 Top3", "数值": row.get("top1_in_top3", "")},
        {"指标": "Top3 全中", "数值": row.get("top3_exact", "")},
    ]
    if top5_hit != "":
        summary.insert(1, {"指标": "本场 Top5 入 Top3 命中率", "数值": top5_hit})
    return summary


def load_bet_plan_table(scope_key, run_id, run_row=None):
    path = ""
    if run_row:
        path = run_row.get("plan_path", "")
    if not path:
        path = str(get_data_dir(BASE_DIR, scope_key) / f"bet_plan_{run_id}.csv")
    path = Path(path)
    rows = load_csv_rows(path)
    if not rows:
        return [], []
    columns = [
        "bet_type",
        "horse_no",
        "horse_name",
        "amount_yen",
        "expected_return_yen",
        "hit_prob_est",
        "hit_prob_se",
        "hit_prob_ci95_low",
        "hit_prob_ci95_high",
        "units",
        "gate_status",
        "risk_note",
        "gate_reason",
    ]
    columns = [col for col in columns if col in rows[0]] or list(rows[0].keys())
    return rows, columns


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
            "High risk: soft gate failed; showing tickets anyway."
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
        return f"[WARN] PASS_GATE_SOFT: HIGH_RISK (showing tickets){reason_text}"
    if status == "hard_fail":
        return f"[WARN] PASS_GATE_HARD: BLOCKED{reason_text}"
    return ""


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
  <title>赛马本地控制台</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --ink: #1f1f1c;
      --accent: #2e6a4f;
      --accent-2: #d46f4d;
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
      position: relative;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      background-image: repeating-linear-gradient(
        135deg,
        rgba(0, 0, 0, 0.025) 0px,
        rgba(0, 0, 0, 0.025) 1px,
        transparent 1px,
        transparent 10px
      );
      opacity: 0.35;
      pointer-events: none;
    }}
    header {{
      padding: 28px 24px 6px;
      max-width: 980px;
      margin: 0 auto;
      position: relative;
    }}
    h1 {{
      margin: 0;
      font-size: 32px;
      letter-spacing: 0.4px;
    }}
    .subtitle {{
      color: var(--muted);
      margin-top: 6px;
      font-style: italic;
    }}
    .wrap {{
      max-width: 980px;
      margin: 0 auto;
      padding: 12px 24px 48px;
      display: grid;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(4px);
      animation: fadeIn 0.6s ease both;
    }}
    .panel:nth-of-type(2) {{ animation-delay: 0.05s; }}
    .panel:nth-of-type(3) {{ animation-delay: 0.1s; }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 1px;
    }}
    .alert {{
      padding: 10px 12px;
      border-radius: 12px;
      background: #ffe8e3;
      border: 1px solid #f1b7aa;
      color: #7a2e1d;
      font-size: 13px;
    }}
    .alert strong {{
      display: block;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }}
    form {{
      display: grid;
      gap: 12px;
    }}
    label {{
      font-size: 14px;
      color: var(--muted);
    }}
    input, select, textarea {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid #e2d3c2;
      border-radius: 12px;
      font-size: 14px;
      background: #fffdfa;
      box-shadow: inset 0 1px 2px rgba(0,0,0,0.04);
    }}
    input:focus, select:focus, textarea:focus {{
      outline: 2px solid rgba(46, 106, 79, 0.25);
      border-color: rgba(46, 106, 79, 0.45);
    }}
    textarea {{
      min-height: 70px;
      resize: vertical;
    }}
    .grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .radio-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 680px;
    }}
    .data-table th,
    .data-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-variant-numeric: tabular-nums;
    }}
    .data-table th {{
      background: #f3e9de;
      color: #2f2a24;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .data-table tbody tr:nth-child(even) {{
      background: #fbf6ef;
    }}
    #scope-radio, #record-scope, #view-scope {{
      flex-wrap: nowrap;
      overflow-x: auto;
      padding-bottom: 2px;
    }}
    #scope-radio .radio-option, #record-scope .radio-option, #view-scope .radio-option {{
      white-space: nowrap;
    }}
    #scope-radio {{
      gap: 12px;
    }}
    #scope-radio .radio-option {{
      padding: 0;
      border: none;
      background: transparent;
    }}
    #scope-radio .radio-option .radio-text {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-width: 108px;
      padding: 12px 16px;
      min-height: 40px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: #f7efe6;
      font-size: 12px;
      line-height: 1.3;
      text-align: center;
      transition: transform 0.2s ease, border 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
    }}
    #scope-radio .radio-option input {{
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }}
    #scope-radio .radio-option input:checked + .radio-text {{
      background: #e8f1ea;
      border-color: rgba(46, 106, 79, 0.6);
      color: #1f4b39;
      box-shadow: 0 6px 14px rgba(46, 106, 79, 0.18);
      transform: translateY(-1px);
    }}
    #scope-radio .radio-option input:focus + .radio-text {{
      outline: 2px solid rgba(46, 106, 79, 0.25);
      outline-offset: 2px;
    }}
    #scope-radio .radio-option:hover {{
      transform: none;
      border-color: transparent;
      background: transparent;
    }}
    #track-cond {{
      gap: 10px;
    }}
    #track-cond .radio-option {{
      padding: 0;
      border: none;
      background: transparent;
    }}
    #track-cond .radio-option .radio-text {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-width: 64px;
      padding: 8px 10px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #f7efe6;
      font-size: 12px;
      line-height: 1.2;
      text-align: center;
      transition: transform 0.2s ease, border 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
    }}
    #track-cond .radio-option input {{
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }}
    #track-cond .radio-option input:checked + .radio-text {{
      background: #e8f1ea;
      border-color: rgba(46, 106, 79, 0.6);
      color: #1f4b39;
      box-shadow: 0 6px 14px rgba(46, 106, 79, 0.18);
      transform: translateY(-1px);
    }}
    #track-cond .radio-option input:focus + .radio-text {{
      outline: 2px solid rgba(46, 106, 79, 0.25);
      outline-offset: 2px;
    }}
    #track-cond .radio-option:hover {{
      transform: none;
      border-color: transparent;
      background: transparent;
    }}
    #action-type {{
      gap: 12px;
    }}
    #action-type .radio-option {{
      padding: 0;
      border: none;
      background: transparent;
    }}
    #action-type .radio-option .radio-text {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-width: 108px;
      padding: 10px 14px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: #f7efe6;
      font-size: 12px;
      line-height: 1.2;
      text-align: center;
      transition: transform 0.2s ease, border 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
    }}
    #action-type .radio-option input {{
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }}
    #action-type .radio-option input:checked + .radio-text {{
      background: #e8f1ea;
      border-color: rgba(46, 106, 79, 0.6);
      color: #1f4b39;
      box-shadow: 0 6px 14px rgba(46, 106, 79, 0.18);
      transform: translateY(-1px);
    }}
    #action-type .radio-option input:focus + .radio-text {{
      outline: 2px solid rgba(46, 106, 79, 0.25);
      outline-offset: 2px;
    }}
    #action-type .radio-option:hover {{
      transform: none;
      border-color: transparent;
      background: transparent;
    }}
    .radio-option {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #f7efe6;
      cursor: pointer;
      font-size: 12px;
      transition: transform 0.2s ease, border 0.2s ease, background 0.2s ease;
    }}
    .radio-option input {{
      accent-color: var(--accent);
      transform: scale(0.9);
    }}
    .radio-option:hover {{
      transform: translateY(-1px);
      border-color: rgba(46, 106, 79, 0.4);
      background: #f2e6db;
    }}
    button {{
      background: linear-gradient(120deg, var(--accent), #1f4b39);
      border: none;
      color: white;
      padding: 11px 16px;
      font-size: 14px;
      border-radius: 12px;
      cursor: pointer;
      box-shadow: 0 10px 18px rgba(34, 79, 60, 0.22);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 14px 24px rgba(34, 79, 60, 0.3);
    }}
    pre {{
      white-space: pre-wrap;
      background: #f4efe9;
      padding: 12px;
      border-radius: 10px;
      border: 1px dashed var(--border);
      margin: 0;
      max-height: 320px;
      overflow: auto;
    }}
    .error {{
      border-color: #d98b6b;
      background: #fff6f0;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 720px) {{
      header {{ padding: 24px 18px 6px; }}
      .wrap {{ padding: 10px 18px 36px; }}
      h1 {{ font-size: 26px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>赛马本地控制台</h1>
    <div class="subtitle">在网页上运行流程与记录结果。</div>
  </header>
  <main class="wrap">
    <section class="panel">
      <h2>运行流程</h2>
      <form action="/run_pipeline" method="post">
        <label>Race ID</label>
        <input name="race_id" inputmode="numeric" pattern="[0-9]*" placeholder="e.g. 202501010101">
        <label>历史查询链接</label>
        <input name="history_url" placeholder="https://db.netkeiba.com/...">
        <label>数据范围</label>
        <div class="radio-group" id="scope-radio">
          <label class="radio-option">
            <input type="radio" name="scope_key" value="central_dirt">
            <span class="radio-text">中央・沙地</span>
          </label>
          <label class="radio-option">
            <input type="radio" name="scope_key" value="central_turf">
            <span class="radio-text">中央・草地</span>
          </label>
          <label class="radio-option">
            <input type="radio" name="scope_key" value="local">
            <span class="radio-text">地方</span>
          </label>
        </div>
        <div class="grid">
          <div>
            <label>距离（米）</label>
            <input name="distance" placeholder="1600">
          </div>
          <div>
            <label>马场（良/稍重/重/不良）</label>
            <div class="radio-group" id="track-cond">
              <label class="radio-option">
                <input type="radio" name="track_cond" value="良" checked>
                <span class="radio-text">良</span>
              </label>
              <label class="radio-option">
                <input type="radio" name="track_cond" value="稍重">
                <span class="radio-text">稍重</span>
              </label>
              <label class="radio-option">
                <input type="radio" name="track_cond" value="重">
                <span class="radio-text">重</span>
              </label>
              <label class="radio-option">
                <input type="radio" name="track_cond" value="不良">
                <span class="radio-text">不良</span>
              </label>
            </div>
          </div>
          <div>
            <label>预算（日元）</label>
            <input name="budget" placeholder="2000">
          </div>
          <div>
            <label>风格</label>
            <select name="style">
              <option value="">（默认）</option>
              <option value="steady">稳健</option>
              <option value="balanced">平衡</option>
              <option value="aggressive">进取</option>
            </select>
          </div>
        </div>
        <button type="submit" {run_button_attr}>运行</button>
      </form>
    </section>

    
    <section class="panel">
      <h2>单场入口</h2>
      <form id="single-action-form" action="/view_run" method="post">
        <label>运行ID / 比赛ID</label>
        <input id="action_id_input" inputmode="text" pattern="[0-9_]*" placeholder="例如 202501010101 或 20250101_123456">
        <input type="hidden" id="action_run_id" name="run_id">
        <input type="hidden" id="action_race_id" name="race_id">
        <label>操作类型</label>
        <div class="radio-group" id="action-type">
          <label class="radio-option">
            <input type="radio" name="action_type" value="view" checked>
            <span class="radio-text">查看</span>
          </label>
          <label class="radio-option">
            <input type="radio" name="action_type" value="update">
            <span class="radio-text">更新投注计划</span>
          </label>
          <label class="radio-option">
            <input type="radio" name="action_type" value="record">
            <span class="radio-text">赛后记录</span>
          </label>
        </div>
        <div class="subtitle">数据范围会根据 ID 自动判断。</div>
        <div class="grid" id="action-update-fields" style="display:none;">
          <div>
            <label>预算（日元，可空）</label>
            <input name="budget" placeholder="2000">
          </div>
          <div>
            <label>投注风格</label>
            <select name="style">
              <option value="">自动</option>
              <option value="steady">steady</option>
              <option value="balanced">balanced</option>
              <option value="aggressive">aggressive</option>
            </select>
          </div>
        </div>
        <div class="grid" id="action-record-fields" style="display:none;">
          <div>
            <label>实际第1名</label>
            <input name="top1">
          </div>
          <div>
            <label>实际第2名</label>
            <input name="top2">
          </div>
          <div>
            <label>实际第3名</label>
            <input name="top3">
          </div>
        </div>
        <button type="submit" id="action-submit">执行</button>
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
      if (actionRunId) {{
        actionRunId.value = value;
      }}
      if (actionRaceId) {{
        actionRaceId.value = value;
      }}
    }}

    function getActionType() {{
      const checked = document.querySelector('input[name="action_type"]:checked');
      return checked ? checked.value : "view";
    }}

    function refreshActionUI() {{
      const actionType = getActionType();
      if (updateFields) {{
        updateFields.style.display = actionType === "update" ? "grid" : "none";
      }}
      if (recordFields) {{
        recordFields.style.display = actionType === "record" ? "grid" : "none";
      }}
      if (actionForm) {{
        if (actionType === "update") {{
          actionForm.action = "/update_bet_plan";
          if (actionSubmit) {{
            actionSubmit.textContent = "更新";
          }}
        }} else if (actionType === "record") {{
          actionForm.action = "/record_pipeline";
          if (actionSubmit) {{
            actionSubmit.textContent = "记录";
          }}
        }} else {{
          actionForm.action = "/view_run";
          if (actionSubmit) {{
            actionSubmit.textContent = "查看";
          }}
        }}
      }}
    }}

    if (actionInput) {{
      actionInput.addEventListener("input", syncActionIds);
      syncActionIds();
    }}
    if (actionTypeRadios.length) {{
      actionTypeRadios.forEach((radio) => {{
        radio.addEventListener("change", refreshActionUI);
      }});
    }}
    refreshActionUI();
  </script>
</body>
</html>
"""


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
    stats_block = ""
    if scope_norm:
        profit_rows = load_profit_summary(scope_norm)
        daily_profit_rows = load_daily_profit_summary(scope_norm)
        wide_box_rows = load_wide_box_daily_profit_summary(scope_norm)
        bet_type_profit_rows = load_bet_type_profit_summary(scope_norm)
        bet_type_rows = load_bet_type_summary(scope_norm)
        predictor_rows = load_predictor_summary(scope_norm)
        parts = []
        if profit_rows:
            parts.append(build_metric_table(profit_rows, "累计盈亏"))
        if daily_profit_rows:
            parts.append(
                build_table_html(
                    daily_profit_rows,
                    ["date", "runs", "profit_yen", "base_amount", "roi"],
                    "Daily Profit",
                )
            )
        if wide_box_rows:
            parts.append(
                build_table_html(
                    wide_box_rows,
                    ["date", "runs", "profit_yen", "base_amount", "roi"],
                    "Wide Box Profit (Top5, 1000 yen)",
                )
            )
        if bet_type_profit_rows:
            parts.append(
                build_table_html(
                    bet_type_profit_rows,
                    ["bet_type", "amount_yen", "est_profit_yen", "roi"],
                    "Win/Place/Wide Profit (2026+)",
                )
            )
        if bet_type_rows:
            parts.append(build_table_html(bet_type_rows, ["bet_type", "bets", "hits", "hit_rate", "amount_yen", "est_profit_yen"], "Bet Type 命中率"))
        if predictor_rows:
            parts.append(build_metric_table(predictor_rows, "预测命中率"))
        stats_block = "\n".join(parts)
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
            summary_table_html = build_table_html(summary_rows, ["指标", "数值"], "模型状态")
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
    if summary_id and scope_norm:
        parts = []
        result_rows = load_run_result_summary(scope_norm, summary_id)
        if result_rows:
            parts.append(build_metric_table(result_rows, "本场盈亏"))
        bet_ticket_rows = load_run_bet_ticket_summary(scope_norm, summary_id)
        if bet_ticket_rows:
            parts.append(
                build_table_html(
                    bet_ticket_rows,
                    [
                        "bet_type",
                        "horse_no",
                        "horse_name",
                        "amount_yen",
                        "hit",
                        "est_payout_yen",
                        "profit_yen",
                    ],
                    "本场 Bet Plan 盈亏明细",
                )
            )
        predictor_rows = load_run_predictor_summary(scope_norm, summary_id)
        if predictor_rows:
            parts.append(build_metric_table(predictor_rows, "本场预测命中率"))
        bet_type_rows = load_run_bet_type_summary(scope_norm, summary_id)
        if bet_type_rows:
            parts.append(
                build_table_html(
                    bet_type_rows,
                    ["bet_type", "bets", "hits", "hit_rate", "amount_yen", "est_profit_yen"],
                    "本场 Bet Type 命中率",
                )
            )
        if parts:
            run_summary_block = "\n".join(parts)
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
        return render_page("", error_text="请输入 Run ID / Race ID 以查看历史记录。")
    if run_row is None:
        run_row = resolve_run(run_id, scope_key)
    if run_row is None:
        race_id = normalize_race_id(run_id)
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_key)
    if run_row is None:
        return render_page(
            scope_key,
            error_text="找不到对应的 Run ID / Race ID。",
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
        return render_page("", error_text="请选择数据范围。")
    race_id = normalize_race_id(race_id or race_url)
    history_url = history_url.strip()
    if not race_id or not history_url:
        return render_page(
            scope_key,
            error_text="Race ID and history URL are required.",
        )
    if scope_key == "local":
        race_url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    else:
        race_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    if not surface.strip():
        if scope_key == "central_turf":
            surface = "1"
        elif scope_key == "central_dirt":
            surface = "2"
        else:
            surface = "1"
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
        return render_page("", error_text="请输入 Run ID / Race ID 以更新。")
    race_id = "" if is_run_id(raw_id) else normalize_race_id(raw_id)
    if run_row is None:
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_key)
        elif raw_id:
            run_row = resolve_run(raw_id, scope_key)
    if run_row is None:
        return render_page(scope_key, error_text="找不到对应的 Run ID / Race ID。")
    if not race_id:
        race_id = normalize_race_id(run_row.get("race_id", ""))
    if not race_id:
        return render_page(scope_key, error_text="Race ID 缺失，无法更新。")
    run_id = str(run_row.get("run_id", "")).strip()
    if not run_id:
        run_id = infer_run_id_from_row(run_row)
        if run_id:
            update_run_row_fields(scope_key, run_row, {"run_id": run_id})
            run_row["run_id"] = run_id
    if not run_id:
        return render_page(scope_key, error_text="Run ID missing for the selected race_id.")
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
            error_text="必须填写实际第1/2/3名。",
        )
    run_id = run_id.strip()
    scope_key = normalize_scope_key(scope_key)
    run_row = None
    if not scope_key:
        scope_key, run_row = infer_scope_and_run(run_id)
    if not scope_key:
        return render_page("", error_text="请输入 Run ID / Race ID 以记录。")
    if run_row is None:
        run_row = resolve_run(run_id, scope_key)
    if run_row is None:
        race_id = normalize_race_id(run_id)
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_key)
    if run_row is None:
        return render_page(
            scope_key,
            error_text="找不到对应的 Run ID / Race ID。",
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

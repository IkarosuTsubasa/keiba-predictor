import atexit
import csv
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import json

from local_env import load_local_env
from surface_scope import (
    get_config_path,
    get_data_dir,
    get_predictor_config_path,
    get_scope_key,
    migrate_legacy_data,
)
from predictor_catalog import latest_prediction_path, list_predictors, snapshot_prediction_path


BASE_DIR = Path(__file__).resolve().parent
load_local_env(BASE_DIR, override=False)
ROOT_DIR = BASE_DIR.parent
DATA_DIR = None
CONFIG_PATH = None
PRED_CONFIG_PATH = None
PRED_RESULTS_PATH = None
SCRAPE_DELAY_RANGE = (1.5, 3.5)
DEFAULT_BUDGETS = [2000, 5000, 10000, 50000]


def configure_utf8_io():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_io()


def sleep_between_scrapes():
    raw = os.environ.get("PIPELINE_SCRAPE_DELAY", "").strip()
    if raw:
        try:
            seconds = float(raw)
        except ValueError:
            return
        if seconds > 0:
            time.sleep(seconds)
        return
    time.sleep(random.uniform(*SCRAPE_DELAY_RANGE))


def find_chrome_exe():
    env_keys = ("CHROME_BIN", "GOOGLE_CHROME_BIN", "CHROME_PATH")
    for key in env_keys:
        value = os.environ.get(key)
        if not value:
            continue
        if Path(value).exists():
            return value
        resolved = shutil.which(value)
        if resolved:
            return resolved
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for path in candidates:
        if Path(path).exists():
            return str(path)
    for cmd in ("chrome", "google-chrome", "chromium", "chromium-browser"):
        resolved = shutil.which(cmd)
        if resolved:
            return resolved
    return ""


def pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_port(port, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


def start_shared_chrome():
    disabled = os.environ.get("PIPELINE_SHARED_CHROME", "").strip().lower()
    if disabled in ("0", "false", "no", "off"):
        return
    chrome_exe = find_chrome_exe()
    if not chrome_exe:
        print("[WARN] Chrome binary not found; starting per-script Chrome.")
        return
    headless_env = os.environ.get("PIPELINE_HEADLESS", "").strip().lower()
    if headless_env in ("1", "true", "yes", "on"):
        headless = True
    elif headless_env in ("0", "false", "no", "off"):
        headless = False
    else:
        headless = False
    profile_env = os.environ.get("PIPELINE_CHROME_PROFILE", "").strip()
    using_temp_profile = not profile_env
    if using_temp_profile:
        profile_dir = tempfile.mkdtemp(prefix="keiba_chrome_")
    else:
        profile_dir = profile_env
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
    port = pick_free_port()
    args = [
        chrome_exe,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--lang=ja-JP",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if headless:
        args.extend(["--headless", "--disable-gpu"])
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[WARN] Failed to start shared Chrome: {exc}")
        if using_temp_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)
        return
    if not wait_for_port(port, timeout=5.0):
        print("[WARN] Shared Chrome did not start; fallback to per-script Chrome.")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        if using_temp_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)
        return
    debugger_address = f"127.0.0.1:{port}"
    os.environ["CHROME_DEBUGGER_ADDRESS"] = debugger_address

    def cleanup():
        os.environ.pop("CHROME_DEBUGGER_ADDRESS", None)
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        if using_temp_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)

    atexit.register(cleanup)

def init_scope():
    scope_key = get_scope_key()
    migrate_legacy_data(BASE_DIR, scope_key)
    global DATA_DIR, CONFIG_PATH, PRED_CONFIG_PATH, PRED_RESULTS_PATH
    DATA_DIR = get_data_dir(BASE_DIR, scope_key)
    CONFIG_PATH = get_config_path(BASE_DIR, scope_key)
    PRED_CONFIG_PATH = get_predictor_config_path(BASE_DIR, scope_key)
    PRED_RESULTS_PATH = DATA_DIR / "predictor_results.csv"
    return scope_key


def require_value(label):
    value = input(label).strip()
    if not value:
        print("Missing input, aborting.")
        sys.exit(1)
    return value


def extract_race_id(url):
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        race_id = parse_qs(parsed.query).get("race_id", [""])[0]
        race_id = re.sub(r"\D", "", race_id)
        return race_id
    except Exception:
        return ""


def map_surface(value):
    raw = (value or "").strip().lower()
    if raw in ("2", "d", "dirt", "ダ"):
        return "ダ"
    if raw in ("1", "s", "shiba", "turf", "芝"):
        return "芝"
    return ""


def map_distance(value):
    raw = (value or "").strip()
    if not raw:
        return ""
    try:
        int(raw)
        return raw
    except ValueError:
        return ""


def normalize_name(value):
    return "".join(str(value or "").split())


def _load_entry_sets(path, name_fields, no_fields):
    if not path.exists():
        return None, None, f"{path} not found."
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        name_field = next((field for field in name_fields if field in fieldnames), None)
        no_field = next((field for field in no_fields if field in fieldnames), None)
        if name_field is None and no_field is None:
            expected = ", ".join(sorted(set(name_fields + no_fields)))
            return None, None, f"{path} missing columns: {expected}"
        names = set()
        numbers = set()
        for row in reader:
            if name_field:
                name = normalize_name(row.get(name_field, ""))
                if name:
                    names.add(name)
            if no_field:
                horse_no = str(row.get(no_field, "") or "").strip()
                if horse_no:
                    numbers.add(horse_no)
    names = {name for name in names if name}
    if not names and not numbers:
        return None, None, f"{path} has no usable rows."
    return names, numbers, ""


def _format_entry_mismatch(left, right, label):
    missing = sorted(left - right)
    extra = sorted(right - left)
    details = []
    if missing:
        details.append(f"missing_{label}={','.join(missing[:8])}")
    if extra:
        details.append(f"extra_{label}={','.join(extra[:8])}")
    return "; ".join(details)


def validate_odds_predictions(odds_path, pred_path):
    odds_names, odds_numbers, err = _load_entry_sets(
        odds_path,
        ["name", "HorseName", "horse_name"],
        ["horse_no", "HorseNo", "horse_number", "馬番"],
    )
    if err:
        return False, err
    pred_names, pred_numbers, err = _load_entry_sets(
        pred_path,
        ["HorseName", "horse_name", "name"],
        ["horse_no", "HorseNo", "horse_number", "馬番"],
    )
    if err:
        return False, err

    if odds_numbers and pred_numbers:
        if odds_numbers != pred_numbers:
            mismatch = _format_entry_mismatch(odds_numbers, pred_numbers, "horse_no")
            return False, f"odds/predictions horse_no mismatch: {mismatch}"
        return True, ""

    if not odds_names or not pred_names:
        return False, "odds/predictions missing comparable entrant fields."
    if odds_names != pred_names:
        mismatch = _format_entry_mismatch(odds_names, pred_names, "horse_name")
        return False, f"odds/predictions horse_name mismatch: {mismatch}"
    return True, ""


def csv_has_rows(path, min_rows=1):
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        count = 0
        for _ in reader:
            count += 1
            if count >= min_rows:
                return True
    return False


def validate_odds_outputs(start_ts, required_paths):
    missing = []
    stale = []
    empty = []
    for label, path in required_paths.items():
        if not path.exists():
            missing.append(label)
            continue
        try:
            if path.stat().st_mtime < start_ts - 1:
                stale.append(label)
        except OSError:
            stale.append(label)
        if not csv_has_rows(path):
            empty.append(label)
    if missing or stale or empty:
        parts = []
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        if stale:
            parts.append(f"not updated: {', '.join(stale)}")
        if empty:
            parts.append(f"empty: {', '.join(empty)}")
        return False, "; ".join(parts)
    return True, ""


def normalize_track_condition_label(value):
    raw = str(value or "").strip()
    raw_lower = raw.lower()
    mapping = {
        "good": "良",
        "firm": "良",
        "slightly_heavy": "稍重",
        "slightly heavy": "稍重",
        "heavy": "重",
        "bad": "不良",
    }
    return mapping.get(raw_lower, raw or "良")


def surface_cli_token(surface_value):
    return "dirt" if surface_value == map_surface("2") else "turf"


def validate_prediction_output(start_ts, pred_path, odds_path):
    if not pred_path.exists():
        return False, f"{pred_path.name} not generated."
    try:
        if pred_path.stat().st_mtime < start_ts - 1:
            return False, f"{pred_path.name} not updated."
    except OSError:
        return False, f"{pred_path.name} stat unavailable."
    if not csv_has_rows(pred_path):
        return False, f"{pred_path.name} has no rows."
    ok, msg = validate_odds_predictions(odds_path, pred_path)
    if not ok:
        return False, msg
    return True, ""


def run_script(script_path, inputs, label, cwd, extra_env=None, extra_lines=0, script_args=None):
    payload_inputs = list(inputs) + ([""] * max(0, int(extra_lines)))
    payload = "\n".join(payload_inputs) + "\n"
    print(f"\n=== {label} ===")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    if extra_env:
        env.update(extra_env)
    subprocess.run(
        [sys.executable, str(script_path)] + [str(arg) for arg in (script_args or [])],
        input=payload,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        cwd=cwd,
        env=env,
    )


def ensure_csv_header(path, fieldnames):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = reader.fieldnames or []
        rows = list(reader)
    if existing == fieldnames:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_csv(path, fieldnames, row):
    ensure_csv_header(path, fieldnames)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        sanitized = {
            name: str(row.get(name, "")).encode("utf-8", "replace").decode("utf-8")
            for name in fieldnames
        }
        writer.writerow(sanitized)


def load_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_config():
    cfg_path = CONFIG_PATH
    if not cfg_path.exists():
        stem = cfg_path.stem.replace("config_", "")
        scope = stem.split("_", 1)[0] if stem else "central_dirt"
        fallback = []
        if scope == "central_turf":
            fallback.extend(
                [
                    BASE_DIR / "config_turf_default.json",
                    BASE_DIR / "config_turf.json",
                ]
            )
        elif scope == "central_dirt":
            fallback.extend(
                [
                    BASE_DIR / "config_dirt_default.json",
                    BASE_DIR / "config_dirt.json",
                ]
            )
        elif scope == "local":
            fallback.extend(
                [
                    BASE_DIR / "config_central_dirt.json",
                    BASE_DIR / "config_dirt_default.json",
                    BASE_DIR / "config_dirt.json",
                ]
            )
        fallback.append(BASE_DIR / "config.json")
        for legacy in fallback:
            if legacy.exists():
                data = json.loads(legacy.read_text(encoding="utf-8"))
                cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return data
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_predictor_config():
    if not PRED_CONFIG_PATH.exists():
        stem = PRED_CONFIG_PATH.stem.replace("predictor_config_", "")
        scope = stem.split("_", 1)[0] if stem else "central_dirt"
        fallback = []
        if scope == "central_turf":
            fallback.extend(
                [
                    BASE_DIR / "predictor_config_turf_default.json",
                    BASE_DIR / "predictor_config_turf.json",
                ]
            )
        elif scope == "central_dirt":
            fallback.extend(
                [
                    BASE_DIR / "predictor_config_dirt_default.json",
                    BASE_DIR / "predictor_config_dirt.json",
                ]
            )
        elif scope == "local":
            fallback.extend(
                [
                    BASE_DIR / "predictor_config_central_dirt.json",
                    BASE_DIR / "predictor_config_dirt_default.json",
                    BASE_DIR / "predictor_config_dirt.json",
                ]
            )
        fallback.append(BASE_DIR / "predictor_config.json")
        for legacy in fallback:
            if legacy.exists():
                data = json.loads(legacy.read_text(encoding="utf-8"))
                PRED_CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return data
    try:
        return json.loads(PRED_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_config_version(config):
    if not config:
        return 1
    return config.get("version", 1)


def choose_strategy(config):
    strategies = config.get("strategies", {})
    if not strategies:
        return "", "none", {}

    selector = config.get("selector", {})
    default_name = config.get("active_strategy", "")
    if default_name not in strategies:
        default_name = next(iter(strategies))

    runs = load_csv(DATA_DIR / "runs.csv")
    results = load_csv(DATA_DIR / "results.csv")
    run_strategy = {row.get("run_id"): row.get("strategy") for row in runs}

    stats = {}
    for name in strategies:
        stats[name] = {"count": 0, "avg_roi": 0.0}

    roi_sum = {name: 0.0 for name in strategies}
    for row in results:
        budget_raw = str(row.get("budget_yen", "")).strip()
        if budget_raw:
            try:
                budget_val = int(float(budget_raw))
            except (TypeError, ValueError):
                budget_val = None
            if budget_val is not None and budget_val != 2000:
                continue
        run_id = row.get("run_id")
        strategy = run_strategy.get(run_id)
        if strategy not in strategies:
            continue
        try:
            roi = float(row.get("roi", ""))
        except (TypeError, ValueError):
            continue
        roi_sum[strategy] += roi
        stats[strategy]["count"] += 1

    for name in strategies:
        count = stats[name]["count"]
        if count > 0:
            stats[name]["avg_roi"] = roi_sum[name] / count

    mode = selector.get("mode", "epsilon_greedy")
    epsilon = float(selector.get("epsilon", 0.2))
    min_samples = int(selector.get("min_samples", 3))

    if mode != "epsilon_greedy":
        return default_name, "default", stats

    if random.random() < epsilon:
        return random.choice(list(strategies)), "explore", stats

    eligible = [name for name, value in stats.items() if value["count"] >= min_samples]
    if not eligible:
        return default_name, "default", stats

    best = max(eligible, key=lambda name: stats[name]["avg_roi"])
    return best, "best_roi", stats


def choose_predictor_strategy(config):
    strategies = config.get("strategies", {})
    if not strategies:
        return "", "none", {}

    selector = config.get("selector", {})
    default_name = config.get("active_strategy", "")
    if default_name not in strategies:
        default_name = next(iter(strategies))

    results = load_csv(PRED_RESULTS_PATH)
    stats = {}
    for name in strategies:
        stats[name] = {"count": 0, "avg_score": 0.0}

    score_sum = {name: 0.0 for name in strategies}
    for row in results:
        strategy = row.get("strategy")
        if strategy not in strategies:
            continue
        try:
            score = float(row.get("score", ""))
        except (TypeError, ValueError):
            continue
        score_sum[strategy] += score
        stats[strategy]["count"] += 1

    for name in strategies:
        count = stats[name]["count"]
        if count > 0:
            stats[name]["avg_score"] = score_sum[name] / count

    mode = selector.get("mode", "epsilon_greedy")
    epsilon = float(selector.get("epsilon", 0.2))
    min_samples = int(selector.get("min_samples", 3))

    if mode != "epsilon_greedy":
        return default_name, "default", stats

    if random.random() < epsilon:
        return random.choice(list(strategies)), "explore", stats

    eligible = [name for name, value in stats.items() if value["count"] >= min_samples]
    if not eligible:
        return default_name, "default", stats

    best = max(eligible, key=lambda name: stats[name]["avg_score"])
    return best, "best_score", stats


def main():
    start_shared_chrome()
    scope_key = init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    pred_config = load_predictor_config()

    race_url = require_value("Race URL (for race_card/odds_extract): ")
    race_id = extract_race_id(race_url)
    if not race_id:
        print("Missing race_id in Race URL. Please provide a URL with race_id.")
        sys.exit(1)
    history_url = require_value("History search URL (for new_history): ")
    target_location = require_value("Location (e.g. 中山): ")
    default_race_date = datetime.now().strftime("%Y-%m-%d")
    race_date = input(f"Race date [YYYY-MM-DD] [{default_race_date}]: ").strip() or default_race_date

    surface_raw = input("Surface 1=turf 2=dirt [auto]: ").strip()
    if not surface_raw:
        if scope_key == "central_turf":
            surface_raw = "1"
        elif scope_key == "central_dirt":
            surface_raw = "2"
        else:
            surface_raw = "2"
    surface = map_surface(surface_raw)
    distance = map_distance(input("Distance meters [1600]: "))
    track_cond = input("Track condition (良/稍重/重/不良) [良]: ").strip()

    strategy_name, strategy_reason, _ = choose_strategy(config)
    if strategy_name:
        print(f"Selected strategy: {strategy_name} ({strategy_reason})")

    predictor_strategy, predictor_reason, _ = choose_predictor_strategy(pred_config)
    if predictor_strategy:
        print(f"Selected predictor strategy: {predictor_strategy} ({predictor_reason})")

    os.environ["SCOPE_KEY"] = scope_key
    predictor_specs = list_predictors()
    race_card_env = None
    run_script(
        ROOT_DIR / "race_card.py",
        [race_url],
        "race_card",
        ROOT_DIR,
        race_card_env,
    )
    sleep_between_scrapes()
    run_script(ROOT_DIR / "new_history.py", [history_url], "new_history", ROOT_DIR)
    sleep_between_scrapes()
    odds_extract_start = time.time()
    run_script(ROOT_DIR / "odds_extract.py", [race_url], "odds_extract", ROOT_DIR)
    odds_required = {
        "odds.csv": ROOT_DIR / "odds.csv",
        "fuku_odds.csv": ROOT_DIR / "fuku_odds.csv",
        "wide_odds.csv": ROOT_DIR / "wide_odds.csv",
        "quinella_odds.csv": ROOT_DIR / "quinella_odds.csv",
        "exacta_odds.csv": ROOT_DIR / "exacta_odds.csv",
        "trio_odds.csv": ROOT_DIR / "trio_odds.csv",
        "trifecta_odds.csv": ROOT_DIR / "trifecta_odds.csv",
    }
    ok, msg = validate_odds_outputs(odds_extract_start, odds_required)
    if not ok:
        print(f"[WARN] odds_extract incomplete: {msg}")
        print("Abort before bet plan to avoid incorrect predictions.")
        sys.exit(1)
    odds_src = ROOT_DIR / "odds.csv"
    track_cond_label = normalize_track_condition_label(track_cond)
    surface_token = surface_cli_token(surface)
    latest_prediction_paths = {}
    for spec in predictor_specs:
        pred_latest_path = latest_prediction_path(ROOT_DIR, spec["id"])
        latest_prediction_paths[spec["id"]] = pred_latest_path
        predictor_start = time.time()
        if spec["id"] == "main":
            env = {"SCOPE_KEY": scope_key}
            if predictor_strategy:
                env["PREDICTOR_STRATEGY"] = predictor_strategy
            run_script(
                ROOT_DIR / spec["script_name"],
                [surface, distance, track_cond],
                spec["label"],
                ROOT_DIR,
                env,
                extra_lines=1,
            )
        elif spec["id"] == "v2_opus":
            run_script(
                ROOT_DIR / spec["script_name"],
                [surface_token, distance, track_cond_label],
                spec["label"],
                ROOT_DIR,
                {
                    "SCOPE_KEY": scope_key,
                    "PREDICTIONS_OUTPUT": str(pred_latest_path),
                    "PREDICTOR_NO_PROMPT": "1",
                    "PREDICTOR_NO_WAIT": "1",
                },
                extra_lines=1,
            )
        elif spec["id"] == "v3_premium":
            run_script(
                ROOT_DIR / spec["script_name"],
                [],
                spec["label"],
                ROOT_DIR,
                {"SCOPE_KEY": scope_key},
                script_args=[
                    "--base-dir",
                    str(ROOT_DIR),
                    "--output",
                    spec["latest_filename"],
                    "--race-venue",
                    target_location or "",
                    "--race-surface",
                    surface,
                    "--race-distance",
                    distance or "1800",
                    "--race-going",
                    track_cond_label,
                    "--no-prompt",
                    "--no-wait",
                ],
            )
        elif spec["id"] == "v4_gemini":
            run_script(
                ROOT_DIR / spec["script_name"],
                [],
                spec["label"],
                ROOT_DIR,
                {
                    "SCOPE_KEY": scope_key,
                    "PREDICTIONS_OUTPUT": str(pred_latest_path),
                    "PREDICTOR_TARGET_SURFACE": surface,
                    "PREDICTOR_TARGET_DISTANCE": distance or "1800",
                    "PREDICTOR_TARGET_CONDITION": track_cond_label,
                },
            )
        elif spec["id"] == "v5_stacking":
            run_script(
                ROOT_DIR / spec["script_name"],
                [],
                spec["label"],
                ROOT_DIR,
                {
                    "SCOPE_KEY": scope_key,
                    "PREDICTIONS_OUTPUT": str(pred_latest_path),
                    "PREDICTOR_TARGET_LOCATION": target_location,
                    "PREDICTOR_TARGET_SURFACE": surface,
                    "PREDICTOR_TARGET_DISTANCE": distance or "1800",
                    "PREDICTOR_TARGET_CONDITION": track_cond_label,
                    "PREDICTOR_TARGET_DATE": race_date,
                    "PREDICTOR_NO_PROMPT": "1",
                    "ODDS_PATH": str(ROOT_DIR / "odds.csv"),
                    "FUKU_ODDS_PATH": str(ROOT_DIR / "fuku_odds.csv"),
                    "WIDE_ODDS_PATH": str(ROOT_DIR / "wide_odds.csv"),
                    "QUINELLA_ODDS_PATH": str(ROOT_DIR / "quinella_odds.csv"),
                    "EXACTA_ODDS_PATH": str(ROOT_DIR / "exacta_odds.csv"),
                    "TRIO_ODDS_PATH": str(ROOT_DIR / "trio_odds.csv"),
                    "TRIFECTA_ODDS_PATH": str(ROOT_DIR / "trifecta_odds.csv"),
                },
            )
        else:
            continue
        ok, msg = validate_prediction_output(predictor_start, pred_latest_path, odds_src)
        if not ok:
            print(f"[ERROR] {spec['label']}: {msg}")
            print("Abort before recording run to avoid polluting data.")
            sys.exit(1)
        sleep_between_scrapes()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    race_suffix = f"_{race_id}"
    race_dir = DATA_DIR / race_id
    race_dir.mkdir(parents=True, exist_ok=True)
    prediction_snapshot_paths = {}
    for spec in predictor_specs:
        pred_src = latest_prediction_paths.get(spec["id"])
        if not pred_src or not pred_src.exists():
            prediction_snapshot_paths[spec["id"]] = ""
            continue
        pred_dest = snapshot_prediction_path(DATA_DIR, race_id, run_id, spec["id"])
        pred_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred_src, pred_dest)
        prediction_snapshot_paths[spec["id"]] = str(pred_dest)
    predictions_path = prediction_snapshot_paths.get("main", "")

    odds_path = ""
    if odds_src.exists():
        odds_dest = race_dir / f"odds_{run_id}{race_suffix}.csv"
        shutil.copy2(odds_src, odds_dest)
        odds_path = str(odds_dest)

    wide_odds_path = ""
    wide_src = ROOT_DIR / "wide_odds.csv"
    if wide_src.exists():
        wide_dest = race_dir / f"wide_odds_{run_id}{race_suffix}.csv"
        shutil.copy2(wide_src, wide_dest)
        wide_odds_path = str(wide_dest)

    fuku_odds_path = ""
    fuku_src = ROOT_DIR / "fuku_odds.csv"
    if fuku_src.exists():
        fuku_dest = race_dir / f"fuku_odds_{run_id}{race_suffix}.csv"
        shutil.copy2(fuku_src, fuku_dest)
        fuku_odds_path = str(fuku_dest)

    quinella_odds_path = ""
    quinella_src = ROOT_DIR / "quinella_odds.csv"
    if quinella_src.exists():
        quinella_dest = race_dir / f"quinella_odds_{run_id}{race_suffix}.csv"
        shutil.copy2(quinella_src, quinella_dest)
        quinella_odds_path = str(quinella_dest)

    exacta_odds_path = ""
    exacta_src = ROOT_DIR / "exacta_odds.csv"
    if exacta_src.exists():
        exacta_dest = race_dir / f"exacta_odds_{run_id}{race_suffix}.csv"
        shutil.copy2(exacta_src, exacta_dest)
        exacta_odds_path = str(exacta_dest)

    trio_odds_path = ""
    trio_src = ROOT_DIR / "trio_odds.csv"
    if trio_src.exists():
        trio_dest = race_dir / f"trio_odds_{run_id}{race_suffix}.csv"
        shutil.copy2(trio_src, trio_dest)
        trio_odds_path = str(trio_dest)

    trifecta_odds_path = ""
    trifecta_src = ROOT_DIR / "trifecta_odds.csv"
    if trifecta_src.exists():
        trifecta_dest = race_dir / f"trifecta_odds_{run_id}{race_suffix}.csv"
        shutil.copy2(trifecta_src, trifecta_dest)
        trifecta_odds_path = str(trifecta_dest)

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "race_url": race_url,
        "race_id": race_id,
        "history_url": history_url,
        "trigger_race": "",
        "scope": scope_key,
        "location": target_location,
        "race_date": race_date,
        "surface": surface,
        "distance": distance,
        "track_condition": track_cond_label,
        "budget_yen": ",".join(str(v) for v in DEFAULT_BUDGETS),
        "style": "auto",
        "strategy": strategy_name,
        "strategy_reason": strategy_reason,
        "predictor_strategy": predictor_strategy,
        "predictor_reason": predictor_reason,
        "config_version": load_config_version(config),
        "predictions_path": predictions_path,
        "predictions_v2_opus_path": prediction_snapshot_paths.get("v2_opus", ""),
        "predictions_v3_premium_path": prediction_snapshot_paths.get("v3_premium", ""),
        "predictions_v4_gemini_path": prediction_snapshot_paths.get("v4_gemini", ""),
        "predictions_v5_stacking_path": prediction_snapshot_paths.get("v5_stacking", ""),
        "odds_path": odds_path,
        "wide_odds_path": wide_odds_path,
        "fuku_odds_path": fuku_odds_path,
        "quinella_odds_path": quinella_odds_path,
        "exacta_odds_path": exacta_odds_path,
        "trio_odds_path": trio_odds_path,
        "trifecta_odds_path": trifecta_odds_path,
        "plan_path": "",
        "gemini_policy_path": "",
        "deepseek_policy_path": "",
        "openai_policy_path": "",
        "grok_policy_path": "",
        "tickets": "",
        "amount_yen": "",
    }
    append_csv(DATA_DIR / "runs.csv", list(row.keys()), row)
    print(f"\nLogged run: {run_id}")


if __name__ == "__main__":
    pipeline_start = time.perf_counter()

    def report_pipeline_elapsed():
        elapsed = time.perf_counter() - pipeline_start
        print(f"[INFO] pipeline elapsed: {elapsed:.1f}s")

    atexit.register(report_pipeline_elapsed)
    main()

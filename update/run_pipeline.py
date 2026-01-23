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

from surface_scope import (
    get_config_path,
    get_data_dir,
    get_predictor_config_path,
    get_scope_key,
    migrate_legacy_data,
)


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = None
CONFIG_PATH = None
PRED_CONFIG_PATH = None
PRED_RESULTS_PATH = None
SCRAPE_DELAY_RANGE = (1.5, 3.5)


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
        if "PIPELINE_SKIP_COOKIE_INJECTION" not in os.environ:
            os.environ["PIPELINE_SKIP_COOKIE_INJECTION"] = "1"
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


def load_name_set(path, field):
    if not path.exists():
        return None, f"{path} not found."
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if field not in fieldnames:
            return None, f"{path} missing column: {field}"
        names = {normalize_name(row.get(field, "")) for row in reader if row.get(field)}
    names = {name for name in names if name}
    if not names:
        return None, f"{path} has no rows for {field}"
    return names, ""


def validate_odds_predictions(odds_path, pred_path):
    odds_names, err = load_name_set(odds_path, "name")
    if odds_names is None:
        return False, err
    pred_names, err = load_name_set(pred_path, "HorseName")
    if pred_names is None:
        return False, err
    matches = odds_names & pred_names
    base = min(len(odds_names), len(pred_names))
    ratio = (len(matches) / base) if base else 0.0
    if len(matches) < 3 or ratio < 0.6:
        return (
            False,
            f"odds/predictions mismatch: matches={len(matches)} ratio={ratio:.2f}",
        )
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


def run_script(script_path, inputs, label, cwd, extra_env=None, extra_lines=0):
    payload_inputs = list(inputs) + ([""] * max(0, int(extra_lines)))
    payload = "\n".join(payload_inputs) + "\n"
    print(f"\n=== {label} ===")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    if extra_env:
        env.update(extra_env)
    subprocess.run(
        [sys.executable, str(script_path)],
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
    trigger_race = require_value("Trigger race name (for new_history): ")

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

    budget = input("Budget yen [2000]: ").strip()
    style = input("Bet style (steady/balanced/aggressive) [balanced]: ").strip()

    strategy_name, strategy_reason, _ = choose_strategy(config)
    if strategy_name:
        print(f"Selected strategy: {strategy_name} ({strategy_reason})")

    predictor_strategy, predictor_reason, _ = choose_predictor_strategy(pred_config)
    if predictor_strategy:
        print(f"Selected predictor strategy: {predictor_strategy} ({predictor_reason})")

    os.environ["SCOPE_KEY"] = scope_key
    history_exclude_trigger = (
        os.environ.get("HISTORY_EXCLUDE_TRIGGER", "").strip().lower()
        in ("1", "true", "yes", "on")
    )

    race_card_env = None
    if history_exclude_trigger:
        race_card_env = {
            "RACE_CARD_EXCLUDE_TRIGGER": "1",
            "TRIGGER_RACE": trigger_race,
        }
    run_script(
        ROOT_DIR / "race_card.py",
        [race_url],
        "race_card",
        ROOT_DIR,
        race_card_env,
    )
    sleep_between_scrapes()
    run_script(ROOT_DIR / "new_history.py", [history_url, trigger_race], "new_history", ROOT_DIR)
    sleep_between_scrapes()
    run_script(
        ROOT_DIR / "predictor.py",
        [surface, distance, track_cond],
        "predictor",
        ROOT_DIR,
        {"PREDICTOR_STRATEGY": predictor_strategy} if predictor_strategy else None,
        extra_lines=1,
    )
    sleep_between_scrapes()
    odds_extract_start = time.time()
    run_script(ROOT_DIR / "odds_extract.py", [race_url], "odds_extract", ROOT_DIR)
    odds_required = {
        "odds.csv": ROOT_DIR / "odds.csv",
        "fuku_odds.csv": ROOT_DIR / "fuku_odds.csv",
        "wide_odds.csv": ROOT_DIR / "wide_odds.csv",
        "quinella_odds.csv": ROOT_DIR / "quinella_odds.csv",
    }
    ok, msg = validate_odds_outputs(odds_extract_start, odds_required)
    if not ok:
        print(f"[WARN] odds_extract incomplete: {msg}")
        print("Abort before bet plan to avoid incorrect predictions.")
        sys.exit(1)
    pred_src = ROOT_DIR / "predictions.csv"
    odds_src = ROOT_DIR / "odds.csv"
    ok, msg = validate_odds_predictions(odds_src, pred_src)
    if not ok:
        print(f"[ERROR] {msg}")
        print("Abort before recording run to avoid polluting data.")
        sys.exit(1)
    bet_plan_env = {"RACE_ID": race_id}
    if strategy_name:
        bet_plan_env["BET_STRATEGY"] = strategy_name
    run_script(
        BASE_DIR / "bet_plan_update.py",
        [budget, style],
        "bet_plan_update",
        BASE_DIR,
        bet_plan_env,
        extra_lines=1,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    race_suffix = f"_{race_id}"
    race_dir = DATA_DIR / race_id
    race_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = ""
    if pred_src.exists():
        pred_dest = race_dir / f"predictions_{run_id}{race_suffix}.csv"
        shutil.copy2(pred_src, pred_dest)
        predictions_path = str(pred_dest)

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

    trifecta_odds_path = ""
    trifecta_src = ROOT_DIR / "trifecta_odds.csv"
    if trifecta_src.exists():
        trifecta_dest = race_dir / f"trifecta_odds_{run_id}{race_suffix}.csv"
        shutil.copy2(trifecta_src, trifecta_dest)
        trifecta_odds_path = str(trifecta_dest)

    plan_path = BASE_DIR / "bet_plan_update.csv"
    plan_log_path = ""
    tickets = 0
    amount_yen = 0
    if plan_path.exists():
        plan_dest = race_dir / f"bet_plan_{run_id}{race_suffix}.csv"
        shutil.copy2(plan_path, plan_dest)
        plan_log_path = str(plan_dest)
        df = None
        try:
            import pandas as pd

            df = pd.read_csv(plan_path, encoding="utf-8-sig")
        except Exception:
            df = None
        if df is not None and not df.empty:
            hard_blocked = False
            if "gate_status" in df.columns:
                hard_blocked = (
                    df["gate_status"].astype(str).str.lower().eq("hard_fail").any()
                )
            if hard_blocked:
                tickets = 0
                amount_yen = 0
            else:
                tickets = int(len(df))
                if "amount_yen" in df.columns:
                    amount_yen = int(df["amount_yen"].sum())

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "race_url": race_url,
        "race_id": race_id,
        "history_url": history_url,
        "trigger_race": trigger_race,
        "scope": scope_key,
        "surface": surface,
        "distance": distance,
        "budget_yen": budget or "2000",
        "style": style or "balanced",
        "strategy": strategy_name,
        "strategy_reason": strategy_reason,
        "predictor_strategy": predictor_strategy,
        "predictor_reason": predictor_reason,
        "config_version": load_config_version(config),
        "predictions_path": predictions_path,
        "odds_path": odds_path,
        "wide_odds_path": wide_odds_path,
        "fuku_odds_path": fuku_odds_path,
        "quinella_odds_path": quinella_odds_path,
        "trifecta_odds_path": trifecta_odds_path,
        "plan_path": plan_log_path or str(plan_path),
        "tickets": tickets,
        "amount_yen": amount_yen,
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

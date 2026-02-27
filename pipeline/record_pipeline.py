import atexit
import csv
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

from surface_scope import (
    get_data_dir,
    get_predictor_config_path,
    get_scope_key,
    migrate_legacy_data,
)


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = None
PRED_CONFIG_PATH = None
RUNS_PATH = None
RESULTS_PATH = None
PRED_RESULTS_PATH = None
RACE_RESULTS_PATH = None
BET_TICKET_PATH = None
BET_TYPE_PATH = None
WIDE_BOX_PATH = None
ODDS_EXTRACT_PATH = ROOT_DIR / "odds_extract.py"

def configure_utf8_io():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_io()


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
    os.environ["SCOPE_KEY"] = scope_key
    global DATA_DIR, PRED_CONFIG_PATH, RUNS_PATH, RESULTS_PATH, PRED_RESULTS_PATH
    global RACE_RESULTS_PATH, BET_TICKET_PATH, BET_TYPE_PATH, WIDE_BOX_PATH
    DATA_DIR = get_data_dir(BASE_DIR, scope_key)
    PRED_CONFIG_PATH = get_predictor_config_path(BASE_DIR, scope_key)
    RUNS_PATH = DATA_DIR / "runs.csv"
    RESULTS_PATH = DATA_DIR / "results.csv"
    PRED_RESULTS_PATH = DATA_DIR / "predictor_results.csv"
    RACE_RESULTS_PATH = DATA_DIR / "race_results.csv"
    BET_TICKET_PATH = DATA_DIR / "bet_ticket_results.csv"
    BET_TYPE_PATH = DATA_DIR / "bet_type_stats.csv"
    WIDE_BOX_PATH = DATA_DIR / "wide_box_results.csv"
    return scope_key


def load_runs():
    if not RUNS_PATH.exists():
        return []
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            with open(RUNS_PATH, "r", encoding=enc) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    return []


def resolve_pred_path_for_run(run, run_id, race_id, race_dir, last_run_id):
    candidates = []
    run_path = str(run.get("predictions_path") or "").strip()
    if run_path:
        candidates.append(Path(run_path))
    if race_id:
        candidates.append(race_dir / f"predictions_{run_id}_{race_id}.csv")
    candidates.append(DATA_DIR / f"predictions_{run_id}.csv")
    for path in candidates:
        if path.exists():
            return path
    if last_run_id and run_id == last_run_id:
        latest = ROOT_DIR / "predictions.csv"
        if latest.exists():
            return latest
    return candidates[0] if candidates else None


def refresh_odds_for_run(
    run_row,
    odds_path,
    wide_odds_path,
    fuku_odds_path,
    quinella_odds_path,
    trifecta_odds_path,
):
    race_url = (run_row.get("race_url") or "").strip()
    race_id = str(run_row.get("race_id") or "").strip()
    if not race_url and race_id:
        scope = str(run_row.get("scope") or "").strip().lower()
        if scope in ("central_turf", "central_dirt"):
            base = "https://race.netkeiba.com/race/shutuba.html?race_id="
        else:
            base = "https://nar.netkeiba.com/race/shutuba.html?race_id="
        race_url = f"{base}{race_id}"
    if not race_url:
        return False, "Race URL missing in run log."
    if not ODDS_EXTRACT_PATH.exists():
        return False, "odds_extract.py not found."
    try:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        result = subprocess.run(
            [sys.executable, str(ODDS_EXTRACT_PATH)],
            input=f"{race_url}\n",
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=str(ROOT_DIR),
            check=False,
            env=env,
        )
    except Exception as exc:
        return False, f"odds_extract failed: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, f"odds_extract failed: {detail}"
    if "Saved: odds.csv" not in (result.stdout or ""):
        return False, "odds_extract produced no new odds."
    tmp_path = ROOT_DIR / "odds.csv"
    if not tmp_path.exists():
        return False, "odds.csv not generated."
    if odds_path:
        try:
            Path(odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_path, odds_path)
        except Exception as exc:
            return False, f"Failed to update odds file: {exc}"
    wide_tmp = ROOT_DIR / "wide_odds.csv"
    if wide_tmp.exists() and wide_odds_path:
        try:
            Path(wide_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wide_tmp, wide_odds_path)
        except Exception as exc:
            return False, f"Failed to update wide odds file: {exc}"
    fuku_tmp = ROOT_DIR / "fuku_odds.csv"
    if fuku_tmp.exists() and fuku_odds_path:
        try:
            Path(fuku_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fuku_tmp, fuku_odds_path)
        except Exception as exc:
            return False, f"Failed to update fuku odds file: {exc}"
    quinella_tmp = ROOT_DIR / "quinella_odds.csv"
    if quinella_tmp.exists() and quinella_odds_path:
        try:
            Path(quinella_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(quinella_tmp, quinella_odds_path)
        except Exception as exc:
            return False, f"Failed to update quinella odds file: {exc}"
    trifecta_tmp = ROOT_DIR / "trifecta_odds.csv"
    if trifecta_tmp.exists() and trifecta_odds_path:
        try:
            Path(trifecta_odds_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trifecta_tmp, trifecta_odds_path)
        except Exception as exc:
            return False, f"Failed to update trifecta odds file: {exc}"
    return True, ""


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
            writer.writerow(sanitize_row(fieldnames, row))


def sanitize_row(fieldnames, row):
    return {
        name: sanitize_value(row.get(name, ""))
        for name in fieldnames
    }


def sanitize_value(value):
    if value is None:
        return ""
    text = str(value)
    return text.encode("utf-8", "replace").decode("utf-8")


def append_csv(path, fieldnames, row):
    ensure_csv_header(path, fieldnames)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(sanitize_row(fieldnames, row))


def replace_rows_for_run(path, fieldnames, run_id, new_rows):
    if isinstance(new_rows, dict):
        new_rows = [new_rows]
    keep = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            keep = [row for row in reader if row.get("run_id") != run_id]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in keep:
            writer.writerow(sanitize_row(fieldnames, row))
        for row in new_rows:
            writer.writerow(sanitize_row(fieldnames, row))


def load_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_name(value):
    return "".join(str(value or "").split())


def parse_horse_no(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        pass
    digits = re.findall(r"\d+", text)
    if len(digits) == 1:
        return int(digits[0])
    return None


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def load_odds_map(odds_path):
    if not odds_path or not Path(odds_path).exists():
        return {}
    df = pd.read_csv(odds_path, encoding="utf-8-sig")
    if "name" not in df.columns or "odds" not in df.columns:
        return {}
    df["name_key"] = df["name"].apply(normalize_name)
    odds_map = {}
    for _, row in df.iterrows():
        val = safe_float(row.get("odds"))
        if val > 0:
            odds_map[str(row.get("name_key"))] = val
    return odds_map


def load_wide_odds_map(path):
    if not path or not Path(path).exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return {}
    if "horse_no_a" not in df.columns or "horse_no_b" not in df.columns:
        return {}
    odds_col = "odds_mid" if "odds_mid" in df.columns else None
    if not odds_col:
        return {}
    out = {}
    for _, row in df.iterrows():
        a_i = parse_horse_no(row.get("horse_no_a", ""))
        b_i = parse_horse_no(row.get("horse_no_b", ""))
        if a_i is None or b_i is None:
            continue
        try:
            val = float(row.get(odds_col, 0))
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        if a_i > b_i:
            a_i, b_i = b_i, a_i
        out[(a_i, b_i)] = val
    return out


def load_fuku_odds_map(path):
    if not path or not Path(path).exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return {}
    if "horse_no" not in df.columns:
        return {}
    if "odds_low" in df.columns:
        odds_col = "odds_low"
    else:
        odds_col = "odds_mid" if "odds_mid" in df.columns else None
    if not odds_col:
        return {}
    out = {}
    for _, row in df.iterrows():
        horse_no = parse_horse_no(row.get("horse_no", ""))
        if horse_no is None:
            continue
        try:
            val = float(row.get(odds_col, 0))
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        out[horse_no] = val
    return out


def load_quinella_odds_map(path):
    if not path or not Path(path).exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return {}
    if "horse_no_a" not in df.columns or "horse_no_b" not in df.columns:
        return {}
    odds_col = "odds" if "odds" in df.columns else "odds_mid"
    out = {}
    for _, row in df.iterrows():
        a_i = parse_horse_no(row.get("horse_no_a", ""))
        b_i = parse_horse_no(row.get("horse_no_b", ""))
        if a_i is None or b_i is None:
            continue
        try:
            val = float(row.get(odds_col, 0))
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        if a_i > b_i:
            a_i, b_i = b_i, a_i
        out[(a_i, b_i)] = val
    return out


def load_trifecta_odds_map(path):
    if not path or not Path(path).exists():
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return {}
    if (
        "horse_no_a" not in df.columns
        or "horse_no_b" not in df.columns
        or "horse_no_c" not in df.columns
    ):
        return {}
    odds_col = "odds" if "odds" in df.columns else "odds_mid"
    out = {}
    for _, row in df.iterrows():
        a_i = parse_horse_no(row.get("horse_no_a", ""))
        b_i = parse_horse_no(row.get("horse_no_b", ""))
        c_i = parse_horse_no(row.get("horse_no_c", ""))
        if a_i is None or b_i is None or c_i is None:
            continue
        try:
            val = float(row.get(odds_col, 0))
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        nums = sorted([a_i, b_i, c_i])
        if len(set(nums)) != 3:
            continue
        out[tuple(nums)] = val
    return out


def pick_score_column(columns):
    for key in ("Top3Prob_model", "Top3Prob_est", "Top3Prob", "agg_score", "score"):
        if key in columns:
            return key
    return ""


def load_top5_names(path):
    if not path or not Path(path).exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "HorseName" not in df.columns:
        return []
    score_key = pick_score_column(df.columns)
    if score_key:
        df = df.sort_values(score_key, ascending=False)
    names = []
    seen = set()
    for value in df["HorseName"].tolist():
        name = str(value or "").strip()
        norm = normalize_name(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        names.append(name)
        if len(names) >= 5:
            break
    return names


def load_name_to_no(path):
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "horse_no" not in df.columns or "name" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        name = row.get("name")
        horse_no = row.get("horse_no")
        if name is None or horse_no is None:
            continue
        try:
            horse_no = int(float(horse_no))
        except (TypeError, ValueError):
            continue
        norm = normalize_name(name)
        if not norm:
            continue
        out[norm] = horse_no
    return out


def compute_wide_box_profit(top5_names, actual_names, name_to_no, wide_odds_map, budget_yen=1000):
    if len(top5_names) < 5 or not wide_odds_map:
        return None
    try:
        pred_nos = [name_to_no[normalize_name(n)] for n in top5_names]
    except KeyError:
        return None
    try:
        actual_nos = {name_to_no[normalize_name(n)] for n in actual_names if n}
    except KeyError:
        return None
    if len(actual_nos) < 2:
        return None
    combos = list(combinations(pred_nos, 2))
    if not combos:
        return None
    per_ticket = int(budget_yen // len(combos))
    if per_ticket <= 0:
        return None
    amount = per_ticket * len(combos)
    total_payout = 0.0
    for a, b in combos:
        if a not in actual_nos or b not in actual_nos:
            continue
        key = (a, b) if a <= b else (b, a)
        odds = wide_odds_map.get(key, 0.0)
        if odds:
            total_payout += per_ticket * odds
    profit = int(round(total_payout - amount))
    roi = round((amount + profit) / amount, 4) if amount > 0 else ""
    return {
        "amount": amount,
        "profit": profit,
        "roi": roi,
        "tickets": len(combos),
        "per_ticket": per_ticket,
    }


def estimate_payout_multiplier(
    bet_type,
    horse_names,
    odds_map,
    wide_odds_map=None,
    fuku_odds_map=None,
    quinella_odds_map=None,
    trifecta_odds_map=None,
    horse_nos=None,
):
    if bet_type == "win" and odds_map and horse_names:
        val = odds_map.get(normalize_name(horse_names[0]), 0)
        if val:
            return max(1.0, float(val))
    if bet_type == "place" and fuku_odds_map and horse_nos:
        try:
            horse_no = int(horse_nos[0])
        except (TypeError, ValueError, IndexError):
            horse_no = None
        if horse_no is not None:
            val = fuku_odds_map.get(horse_no)
            if val:
                return max(1.0, float(val))
    if bet_type == "wide" and wide_odds_map and horse_nos and len(horse_nos) >= 2:
        try:
            a = int(horse_nos[0])
            b = int(horse_nos[1])
        except (TypeError, ValueError):
            a = b = None
        if a is not None and b is not None:
            if a > b:
                a, b = b, a
            val = wide_odds_map.get((a, b))
            if val:
                return max(1.0, float(val))
    if bet_type == "quinella" and quinella_odds_map and horse_nos and len(horse_nos) >= 2:
        try:
            a = int(horse_nos[0])
            b = int(horse_nos[1])
        except (TypeError, ValueError):
            a = b = None
        if a is not None and b is not None:
            if a > b:
                a, b = b, a
            val = quinella_odds_map.get((a, b))
            if val:
                return max(1.0, float(val))
    if bet_type == "trifecta" and trifecta_odds_map and horse_nos and len(horse_nos) >= 3:
        try:
            nums = [int(horse_nos[0]), int(horse_nos[1]), int(horse_nos[2])]
        except (TypeError, ValueError):
            nums = []
        if len(nums) == 3 and len(set(nums)) == 3:
            nums.sort()
            val = trifecta_odds_map.get(tuple(nums))
            if val:
                return max(1.0, float(val))
    odds = []
    for name in horse_names:
        val = odds_map.get(normalize_name(name), 0)
        if val > 0:
            odds.append(val)
    base = sum(odds) / len(odds) if odds else 1.0
    factors = {
        "win": 1.0,
        "place": 0.35,
        "wide": 0.25,
        "quinella": 0.6,
        "exacta": 0.9,
        "trifecta": 1.4,
    }
    factor = factors.get(bet_type, 1.0)
    return max(1.0, base * factor)


def ndcg_at3(pred_names, actual_order):
    if not actual_order or len(actual_order) < 3:
        return 0.0
    rel_map = {
        actual_order[0]: 3,
        actual_order[1]: 2,
        actual_order[2]: 1,
    }
    dcg = 0.0
    for idx, name in enumerate(pred_names[:3]):
        rel = rel_map.get(name, 0)
        if rel <= 0:
            continue
        dcg += (2 ** rel - 1) / math.log2(idx + 2)
    idcg = 0.0
    for idx, rel in enumerate([3, 2, 1]):
        idcg += (2 ** rel - 1) / math.log2(idx + 2)
    return dcg / idcg if idcg > 0 else 0.0


def ev_score_from_odds(pred_items, odds_map):
    ev_values = []
    for name, prob in pred_items:
        odds = odds_map.get(normalize_name(name), 0)
        if odds > 0:
            ev_values.append(float(prob) * float(odds) - 1.0)
    if not ev_values:
        return 0.0
    ev_mean = sum(ev_values) / len(ev_values)
    return (math.tanh(ev_mean) + 1.0) / 2.0


def load_predictor_state():
    try:
        cfg = json.loads(PRED_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"rank_ema": 0.5, "ev_ema": 0.5, "risk_score": 0.5}
    state = cfg.get("state", {})
    return {
        "rank_ema": clamp(float(state.get("rank_ema", 0.5)), 0.0, 1.0),
        "ev_ema": clamp(float(state.get("ev_ema", 0.5)), 0.0, 1.0),
        "risk_score": clamp(float(state.get("risk_score", 0.5)), 0.0, 1.0),
    }


def compute_confidence(pred_items, state):
    rank_ema = state.get("rank_ema", 0.5)
    ev_ema = state.get("ev_ema", 0.5)
    risk_score = state.get("risk_score", 0.5)
    validity = clamp(0.6 * rank_ema + 0.4 * ev_ema, 0.0, 1.0)
    stability = clamp(risk_score, 0.0, 1.0)
    consistency = 0.0
    if pred_items:
        probs = [safe_float(p) for _, p in pred_items]
        if len(probs) >= 3:
            gap = probs[0] - probs[2]
        elif len(probs) >= 2:
            gap = probs[0] - probs[1]
        else:
            gap = probs[0]
        consistency = clamp(gap / 0.15, 0.0, 1.0)
    confidence = math.sqrt(stability * validity) * consistency
    return {
        "confidence_score": round(clamp(confidence, 0.0, 1.0), 4),
        "stability_score": round(stability, 4),
        "validity_score": round(validity, 4),
        "consistency_score": round(consistency, 4),
        "rank_ema": round(rank_ema, 4),
        "ev_ema": round(ev_ema, 4),
        "risk_score": round(risk_score, 4),
    }


def prompt_horse_name(label):
    while True:
        value = input(label).strip()
        if value:
            return value
        print("Please enter a horse name.")


def prompt_actual_top3():
    name1 = prompt_horse_name("Enter 1st place horse name: ")
    name2 = prompt_horse_name("Enter 2nd place horse name: ")
    name3 = prompt_horse_name("Enter 3rd place horse name: ")
    return [name1, name2, name3]


def split_names(raw):
    if not raw:
        return []
    cleaned = raw.replace("／", "/")
    parts = [p.strip() for p in cleaned.split("/") if p.strip()]
    return parts


def split_numbers(raw):
    if not raw:
        return []
    nums = re.findall(r"\d+", str(raw))
    return [int(n) for n in nums]



def eval_ticket(bet_type, horse_names, actual_order):
    if not horse_names:
        return 0
    top3 = actual_order[:3]
    top2 = actual_order[:2]
    if bet_type == "win":
        return 1 if horse_names[0] == actual_order[0] else 0
    if bet_type == "place":
        return 1 if horse_names[0] in top3 else 0
    if bet_type == "wide":
        return 1 if all(n in top3 for n in horse_names[:2]) else 0
    if bet_type == "quinella":
        return 1 if len(horse_names) >= 2 and horse_names[0] in top2 and horse_names[1] in top2 else 0
    if bet_type == "exacta":
        return 1 if len(horse_names) >= 2 and horse_names[0] == top2[0] and horse_names[1] == top2[1] else 0
    if bet_type == "trifecta":
        return 1 if all(n in top3 for n in horse_names[:3]) else 0
    return 0


def prompt_profit(allow_blank=False):
    while True:
        try:
            raw = input("Profit/Loss yen (blank=estimate): ").strip()
        except EOFError:
            if allow_blank:
                return None
            if not sys.stdin.isatty():
                print("No stdin available; defaulting profit/loss to 0.")
                return 0
            print("No input available; defaulting profit/loss to 0.")
            return 0
        if not raw:
            if allow_blank:
                return None
            if not sys.stdin.isatty():
                print("No stdin available; defaulting profit/loss to 0.")
                return 0
            print("Please enter profit/loss amount.")
            continue
        try:
            return int(float(raw))
        except ValueError:
            print("Invalid profit value.")


def run_tool(path, input_text=None):
    if path.exists():
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        subprocess.run(
            [sys.executable, str(path)],
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
        )


def main():
    start_shared_chrome()
    init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No runs found. Run pipeline/run_pipeline.py first.")
        sys.exit(1)

    last_run = runs[-1]["run_id"]
    run_id = input(f"Run ID [{last_run}]: ").strip() or last_run
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        print("Run ID not found.")
        sys.exit(1)
    race_id = str(run.get("race_id") or "").strip()
    if not race_id:
        print("Missing race_id for this run. Please fix runs.csv.")
        sys.exit(1)

    race_dir = DATA_DIR / race_id
    odds_path = run.get("odds_path") or str(race_dir / f"odds_{run_id}_{race_id}.csv")
    wide_odds_path = run.get("wide_odds_path") or str(race_dir / f"wide_odds_{run_id}_{race_id}.csv")
    fuku_odds_path = run.get("fuku_odds_path") or str(race_dir / f"fuku_odds_{run_id}_{race_id}.csv")
    quinella_odds_path = run.get("quinella_odds_path") or str(race_dir / f"quinella_odds_{run_id}_{race_id}.csv")
    trifecta_odds_path = run.get("trifecta_odds_path") or str(
        race_dir / f"trifecta_odds_{run_id}_{race_id}.csv"
    )
    odds_path_run = Path(odds_path)
    wide_odds_path_run = Path(wide_odds_path)
    pred_path_run = resolve_pred_path_for_run(run, run_id, race_id, race_dir, last_run)
    updated, odds_msg = refresh_odds_for_run(
        run,
        odds_path,
        wide_odds_path,
        fuku_odds_path,
        quinella_odds_path,
        trifecta_odds_path,
    )
    if updated:
        print("Updated odds for this run.")
    elif odds_msg:
        print(f"Odds update skipped: {odds_msg}")
    if not Path(odds_path).exists():
        odds_path = str(ROOT_DIR / "odds.csv")
    odds_map = load_odds_map(odds_path)
    if not Path(wide_odds_path).exists():
        wide_odds_path = str(ROOT_DIR / "wide_odds.csv")
    wide_odds_map = load_wide_odds_map(wide_odds_path)
    if not Path(fuku_odds_path).exists():
        fuku_odds_path = str(ROOT_DIR / "fuku_odds.csv")
    fuku_odds_map = load_fuku_odds_map(fuku_odds_path)
    if not Path(quinella_odds_path).exists():
        quinella_odds_path = str(ROOT_DIR / "quinella_odds.csv")
    quinella_odds_map = load_quinella_odds_map(quinella_odds_path)
    if not Path(trifecta_odds_path).exists():
        trifecta_odds_path = str(ROOT_DIR / "trifecta_odds.csv")
    trifecta_odds_map = load_trifecta_odds_map(trifecta_odds_path)

    strategy = run.get("strategy", "")
    if strategy:
        print(f"Bet strategy: {strategy}")

    profit_yen = prompt_profit(allow_blank=True)
    note = input("Notes (optional): ").strip()

    predictor_strategy = run.get("predictor_strategy", "")
    if predictor_strategy:
        print(f"Predictor strategy: {predictor_strategy}")

    pred_names_raw = []
    pred_names = []
    pred_items = []
    pred_top5_names = []
    pred_name_set = None
    pred_file = None
    pred_path = str(pred_path_run) if pred_path_run else ""
    pred_file = Path(pred_path) if pred_path else None
    if pred_file and pred_file.exists():
        df = pd.read_csv(pred_file, encoding="utf-8-sig")
        score_key = pick_score_column(df.columns)
        if "HorseName" in df.columns and score_key:
            pred_name_set = {
                normalize_name(n)
                for n in df["HorseName"].dropna().astype(str)
                if normalize_name(n)
            }
            pred_top = df.sort_values(score_key, ascending=False).head(3)
            pred_names_raw = pred_top["HorseName"].tolist()
            pred_items = list(zip(pred_top["HorseName"].tolist(), pred_top[score_key].tolist()))
            pred_names = [normalize_name(n) for n in pred_names_raw]
            pred_top5 = df.sort_values(score_key, ascending=False).head(5)
            pred_top5_names_raw = pred_top5["HorseName"].tolist()
            pred_top5_names = [normalize_name(n) for n in pred_top5_names_raw]
            print("\nPredicted Top3:")
            for i, name in enumerate(pred_names_raw, 1):
                print(f"{i}. {name}")
        else:
            print("Predictions file missing columns: HorseName / Top3Prob")
    else:
        if pred_path_run:
            print(f"Predictions file not found: {pred_path_run}")
        else:
            print("Predictions file not found for this run.")
        print("Skip predictor record.")

    actual_names_raw = prompt_actual_top3()
    actual_names = [normalize_name(n) for n in actual_names_raw]
    if pred_name_set:
        missing = [
            raw
            for raw, norm in zip(actual_names_raw, actual_names)
            if norm and norm not in pred_name_set
        ]
        if missing:
            print(f"[ERROR] Actual Top3 names not found in predictions: {', '.join(missing)}")
            sys.exit(1)
    race_row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "actual_top1": actual_names_raw[0],
        "actual_top2": actual_names_raw[1],
        "actual_top3": actual_names_raw[2],
    }
    replace_rows_for_run(RACE_RESULTS_PATH, list(race_row.keys()), run_id, race_row)

    if pred_names_raw and pred_file and pred_file.exists():
        hit_count = len(set(pred_names) & set(actual_names))
        top5_hit_count = len(set(pred_top5_names) & set(actual_names)) if pred_top5_names else 0
        top1_hit = 1 if pred_names and actual_names and pred_names[0] == actual_names[0] else 0
        top1_in_top3 = 1 if pred_names and pred_names[0] in actual_names else 0
        top3_exact = 1 if set(pred_names) == set(actual_names) else 0
        score = hit_count + top1_hit
        rank_score = ndcg_at3(pred_names, actual_names)
        hit_rate = hit_count / 3.0
        ev_score = ev_score_from_odds(pred_items, odds_map)
        score_total = 0.4 * rank_score + 0.4 * ev_score + 0.2 * hit_rate
        conf_state = load_predictor_state()
        conf = compute_confidence(pred_items, conf_state)

        pred_row = {
            "run_id": run_id,
            "strategy": predictor_strategy,
            "predictions_path": str(pred_file),
            "pred_top1": pred_names_raw[0] if len(pred_names_raw) > 0 else "",
            "pred_top2": pred_names_raw[1] if len(pred_names_raw) > 1 else "",
            "pred_top3": pred_names_raw[2] if len(pred_names_raw) > 2 else "",
            "actual_top1": actual_names_raw[0],
            "actual_top2": actual_names_raw[1],
            "actual_top3": actual_names_raw[2],
            "top3_hit_count": hit_count,
            "top5_hit_count": top5_hit_count,
            "top1_hit": top1_hit,
            "top1_in_top3": top1_in_top3,
            "top3_exact": top3_exact,
            "score": score,
            "rank_score": round(rank_score, 4),
            "ev_score": round(ev_score, 4),
            "hit_rate": round(hit_rate, 4),
            "score_total": round(score_total, 4),
            "confidence_score": conf["confidence_score"],
            "stability_score": conf["stability_score"],
            "validity_score": conf["validity_score"],
            "consistency_score": conf["consistency_score"],
            "rank_ema": conf["rank_ema"],
            "ev_ema": conf["ev_ema"],
            "risk_score": conf["risk_score"],
        }
        replace_rows_for_run(PRED_RESULTS_PATH, list(pred_row.keys()), run_id, pred_row)
        print(f"Recorded predictor result for {run_id}")

    wide_box_info = None
    if pred_path_run and pred_path_run.exists() and odds_path_run.exists() and wide_odds_path_run.exists():
        top5_names = load_top5_names(pred_path_run)
        name_to_no = load_name_to_no(str(odds_path_run))
        wide_box_map = load_wide_odds_map(str(wide_odds_path_run))
        wide_box_info = compute_wide_box_profit(
            top5_names,
            actual_names_raw,
            name_to_no,
            wide_box_map,
            budget_yen=1000,
        )
    else:
        missing = []
        if not pred_path_run or not pred_path_run.exists():
            missing.append("predictions")
        if not odds_path_run.exists():
            missing.append("odds")
        if not wide_odds_path_run.exists():
            missing.append("wide_odds")
        if missing:
            print(f"Skip wide box: missing {', '.join(missing)} for {run_id}.")
    if wide_box_info:
        wide_row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "budget_yen": 1000,
            "tickets": wide_box_info["tickets"],
            "per_ticket": wide_box_info["per_ticket"],
            "amount_yen": wide_box_info["amount"],
            "profit_yen": wide_box_info["profit"],
            "roi": wide_box_info["roi"],
            "method": "top5_wide_box",
        }
        existing = load_csv(WIDE_BOX_PATH)
        if existing:
            keep = [
                row
                for row in existing
                if not (
                    row.get("run_id") == run_id
                    and row.get("method") == wide_row["method"]
                )
            ]
            with open(WIDE_BOX_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(wide_row.keys()))
                writer.writeheader()
                for row in keep:
                    writer.writerow(sanitize_row(list(wide_row.keys()), row))
                writer.writerow(sanitize_row(list(wide_row.keys()), wide_row))
        else:
            append_csv(WIDE_BOX_PATH, list(wide_row.keys()), wide_row)
        print(f"Recorded wide box result for {run_id}")

    plan_path = run.get("plan_path") or str(race_dir / f"bet_plan_{run_id}_{race_id}.csv")
    if not Path(plan_path).exists():
        plan_path = str(BASE_DIR / "bet_plan_update.csv")
    plan_file = Path(plan_path)
    budget_totals = {}
    if not plan_file.exists():
        print(f"Plan file not found: {plan_file}")
    else:
        df_plan = pd.read_csv(plan_file, encoding="utf-8-sig")
        bet_type_stats = {}
        ticket_rows = []
        for _, row in df_plan.iterrows():
            bet_type = str(row.get("bet_type", "")).strip().lower()
            if bet_type == "trifecta_rec":
                continue
            budget_raw = row.get("budget_yen", "")
            try:
                budget_yen = int(float(budget_raw))
            except (TypeError, ValueError):
                budget_yen = 2000
            horse_no = str(row.get("horse_no", "")).strip()
            horse_name_raw = str(row.get("horse_name", "")).strip()
            names = [normalize_name(n) for n in split_names(horse_name_raw)]
            hit = eval_ticket(bet_type, names, actual_names)
            amount_yen = int(float(row.get("amount_yen", 0)))
            horse_nums = split_numbers(horse_no)
            payout_mult = estimate_payout_multiplier(
                bet_type,
                names,
                odds_map,
                wide_odds_map=wide_odds_map,
                fuku_odds_map=fuku_odds_map,
                quinella_odds_map=quinella_odds_map,
                trifecta_odds_map=trifecta_odds_map,
                horse_nos=horse_nums,
            )
            est_payout = amount_yen * payout_mult if hit else 0.0
            budget_item = budget_totals.setdefault(
                budget_yen,
                {"amount": 0, "est_payout": 0.0},
            )
            budget_item["amount"] += amount_yen
            budget_item["est_payout"] += est_payout

            ticket_row = {
                "run_id": run_id,
                "budget_yen": budget_yen,
                "bet_type": bet_type,
                "horse_no": horse_no,
                "horse_name": horse_name_raw,
                "amount_yen": amount_yen,
                "hit": hit,
                "est_payout_yen": int(round(est_payout)),
            }
            ticket_rows.append(ticket_row)

            stats = bet_type_stats.setdefault(
                (budget_yen, bet_type),
                {"bets": 0, "hits": 0, "amount": 0, "est_payout": 0.0},
            )
            stats["bets"] += 1
            stats["hits"] += hit
            stats["amount"] += amount_yen
            stats["est_payout"] += est_payout

        stat_rows = []
        for (budget_yen, bet_type), stats in bet_type_stats.items():
            hit_rate = stats["hits"] / stats["bets"] if stats["bets"] else 0
            stat_row = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "budget_yen": budget_yen,
                "bet_type": bet_type,
                "bets": stats["bets"],
                "hits": stats["hits"],
                "hit_rate": round(hit_rate, 4),
                "amount_yen": stats["amount"],
                "est_payout_yen": int(round(stats["est_payout"])),
                "est_profit_yen": int(round(stats["est_payout"] - stats["amount"])),
            }
            stat_rows.append(stat_row)
        if ticket_rows:
            replace_rows_for_run(
                BET_TICKET_PATH, list(ticket_rows[0].keys()), run_id, ticket_rows
            )
        if stat_rows:
            replace_rows_for_run(
                BET_TYPE_PATH, list(stat_rows[0].keys()), run_id, stat_rows
            )
            print("Recorded bet-type stats.")

    if not budget_totals:
        total_amount_yen = 0
        total_est_payout = 0.0
    else:
        total_amount_yen = sum(item["amount"] for item in budget_totals.values())
        total_est_payout = sum(item["est_payout"] for item in budget_totals.values())

    if profit_yen is None:
        has_odds_data = bool(
            odds_map or wide_odds_map or fuku_odds_map or quinella_odds_map or trifecta_odds_map
        )
        if total_amount_yen > 0 and has_odds_data:
            profit_yen = int(round(total_est_payout - total_amount_yen))
            if note:
                note = f"{note} | est_by_odds"
            else:
                note = "est_by_odds"
            print(f"Estimated profit (odds-based): {profit_yen} yen")
        else:
            print("Unable to estimate profit; please enter it manually.")
            profit_yen = prompt_profit(allow_blank=False)

    result_rows = []
    if budget_totals:
        if profit_yen is not None:
            print("[WARN] Multi-budget plan detected; manual profit is ignored. Using odds-based estimate per budget.")
        for budget_yen in sorted(budget_totals.keys()):
            base_amount = int(budget_totals[budget_yen]["amount"])
            est_profit = int(round(budget_totals[budget_yen]["est_payout"] - base_amount))
            roi = round((base_amount + est_profit) / base_amount, 4) if base_amount > 0 else ""
            result_rows.append(
                {
                    "run_id": run_id,
                    "strategy": strategy,
                    "budget_yen": budget_yen,
                    "profit_yen": est_profit,
                    "base_amount": base_amount,
                    "roi": roi,
                    "note": note if note else "est_by_odds",
                }
            )
    else:
        base_amount = total_amount_yen
        if base_amount <= 0:
            try:
                base_amount = int(float(run.get("amount_yen", 0)))
            except ValueError:
                base_amount = 0
        if base_amount <= 0:
            try:
                base_amount = int(float(run.get("budget_yen", 0)))
            except ValueError:
                base_amount = 0
        roi = round((base_amount + profit_yen) / base_amount, 4) if base_amount > 0 else ""
        result_rows.append(
            {
                "run_id": run_id,
                "strategy": strategy,
                "budget_yen": 2000,
                "profit_yen": profit_yen,
                "base_amount": base_amount,
                "roi": roi,
                "note": note,
            }
        )

    replace_rows_for_run(RESULTS_PATH, list(result_rows[0].keys()), run_id, result_rows)
    print(f"Recorded result for {run_id}")

    run_tool(BASE_DIR / "optimize_params.py")
    run_tool(BASE_DIR / "optimize_predictor_params.py")
    run_tool(BASE_DIR / "offline_eval.py", input_text="\n\n")


if __name__ == "__main__":
    main()

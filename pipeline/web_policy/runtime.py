import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def load_policy_bankroll_summary(*, base_dir, run_id="", timestamp="", policy_engine="gemini", extract_ledger_date, summarize_bankroll):
    ledger_date = extract_ledger_date(run_id, timestamp)
    return summarize_bankroll(base_dir, ledger_date, policy_engine=policy_engine)


def load_policy_daily_profit_summary(*, base_dir, days=30, policy_engine="gemini", load_daily_profit_rows):
    return load_daily_profit_rows(base_dir, days=days, policy_engine=policy_engine)


def load_policy_run_ticket_rows(*, base_dir, run_id, policy_engine="gemini", load_run_tickets):
    rows = load_run_tickets(base_dir, run_id, policy_engine=policy_engine)
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
        parts.append(str(script_output).strip())
    return "\n".join([part for part in parts if str(part).strip()]).strip()


def resolve_run_selection(
    *,
    scope_key,
    run_id,
    normalize_scope_key,
    infer_scope_and_run,
    resolve_run,
    normalize_race_id,
    resolve_latest_run_by_race_id,
):
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


def maybe_refresh_run_odds(
    *,
    scope_norm,
    run_row,
    run_id,
    refresh_enabled,
    resolve_run_asset_path,
    refresh_odds_for_run,
):
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
    return refresh_odds_for_run(
        run_row,
        scope_norm,
        odds_path,
        wide_odds_path=wide_odds_path,
        fuku_odds_path=fuku_odds_path,
        quinella_odds_path=quinella_odds_path,
        exacta_odds_path=exacta_odds_path,
        trio_odds_path=trio_odds_path,
    )


def resolve_policy_timeout(*, policy_engine, normalize_policy_engine):
    engine = normalize_policy_engine(policy_engine)
    env_keys = []
    default_timeout = 20
    if engine == "deepseek":
        env_keys = ["DEEPSEEK_POLICY_TIMEOUT", "POLICY_TIMEOUT_DEEPSEEK", "POLICY_TIMEOUT"]
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
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return default_timeout


def strict_llm_odds_gate_enabled():
    raw = os.environ.get("PIPELINE_BLOCK_LLM_ON_ODDS_WARNING", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def expected_odds_output_names(scope_key):
    return [
        "odds.csv",
        "fuku_odds.csv",
        "wide_odds.csv",
        "quinella_odds.csv",
        "exacta_odds.csv",
        "trio_odds.csv",
    ]


def capture_output_mtimes(root_dir, names):
    mtimes = {}
    for name in list(names or []):
        path = Path(root_dir) / name
        try:
            mtimes[name] = path.stat().st_mtime
        except OSError:
            mtimes[name] = None
    return mtimes


def is_fresh_output(path, previous_mtime, started_at):
    path = Path(path)
    if not path.exists():
        return False
    try:
        current_mtime = path.stat().st_mtime
    except OSError:
        return False
    if previous_mtime is None:
        return current_mtime >= (float(started_at) - 1.0)
    return current_mtime > float(previous_mtime)


def copy_fresh_odds_output(tmp_path, dest_path, before_mtimes, started_at, warnings):
    tmp_name = Path(tmp_path).name
    if not dest_path:
        return True, ""
    if not is_fresh_output(tmp_path, before_mtimes.get(tmp_name), started_at):
        warnings.append(f"{tmp_name} not freshly generated.")
        return True, ""
    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path, dest_path)
    except Exception as exc:
        return False, f"Failed to update {tmp_name}: {exc}"
    return True, ""


def refresh_odds_for_run(
    run_row,
    scope_key,
    odds_path,
    wide_odds_path=None,
    fuku_odds_path=None,
    quinella_odds_path=None,
    exacta_odds_path=None,
    trio_odds_path=None,
    *,
    get_env_timeout,
    normalize_race_id,
    odds_extract_path,
    root_dir,
):
    odds_timeout_seconds = get_env_timeout("PIPELINE_ODDS_EXTRACT_TIMEOUT", 300)
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
    if not Path(odds_extract_path).exists():
        return False, "odds_extract.py not found.", []
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PIPELINE_HEADLESS", "0")
    expected_names = expected_odds_output_names(scope_key)
    before_mtimes = capture_output_mtimes(root_dir, expected_names)
    started_at = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(odds_extract_path)],
            input=f"{race_url}\n",
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=str(root_dir),
            env=env,
            check=False,
            timeout=odds_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        detail = f"odds_extract timeout after {odds_timeout_seconds}s"
        stdout_text = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        stderr_text = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        if stdout_text:
            detail = f"{detail}\n{stdout_text}"
        if stderr_text:
            detail = f"{detail}\n[stderr]\n{stderr_text}"
        return False, detail, []
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, f"odds_extract failed: {detail}", []
    if "Saved: odds.csv" not in (result.stdout or ""):
        return False, "odds_extract produced no new odds.", []
    tmp_path = Path(root_dir) / "odds.csv"
    if not is_fresh_output(tmp_path, before_mtimes.get("odds.csv"), started_at):
        return False, "odds.csv not freshly generated.", []
    warnings = []
    try:
        Path(odds_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path, odds_path)
    except Exception as exc:
        return False, f"Failed to update odds file: {exc}", []
    for tmp_name, dest in (
        ("wide_odds.csv", wide_odds_path),
        ("fuku_odds.csv", fuku_odds_path),
        ("quinella_odds.csv", quinella_odds_path),
        ("exacta_odds.csv", exacta_odds_path),
        ("trio_odds.csv", trio_odds_path),
    ):
        ok, message = copy_fresh_odds_output(Path(root_dir) / tmp_name, dest, before_mtimes, started_at, warnings)
        if not ok:
            return False, message, []
    if strict_llm_odds_gate_enabled() and warnings:
        return False, "Incomplete odds refresh: " + "; ".join(warnings), warnings
    return True, "", warnings

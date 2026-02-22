import re
from pathlib import Path


def resolve_run(load_runs, run_id, scope_key):
    runs = load_runs(scope_key)
    if not runs:
        return None
    if not run_id:
        return runs[-1]
    for row in runs:
        if row.get("run_id") == run_id:
            return row
    return None


def resolve_latest_run_by_race_id(load_runs, normalize_race_id, race_id, scope_key):
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


def resolve_plan_path(get_data_dir, base_dir, scope_key, run_id, run_row):
    path = str(run_row.get("plan_path", "")).strip() if run_row else ""
    if path:
        return Path(path)
    race_id = str(run_row.get("race_id", "") or "") if run_row else ""
    if race_id:
        race_dir = get_data_dir(base_dir, scope_key) / race_id
        return race_dir / f"bet_plan_{run_id}_{race_id}.csv"
    return get_data_dir(base_dir, scope_key) / f"bet_plan_{run_id}.csv"


def find_run_in_scope(load_runs, normalize_race_id, scope_key, id_text):
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


def infer_scope_and_run(load_runs, normalize_race_id, id_text):
    id_text = str(id_text or "").strip()
    if not id_text:
        return "", None
    for scope_key in ("central_dirt", "central_turf", "local"):
        run_row = find_run_in_scope(load_runs, normalize_race_id, scope_key, id_text)
        if run_row:
            return scope_key, run_row
    return "", None


def resolve_pred_path(get_data_dir, base_dir, scope_key, run_id, run_row):
    path = run_row.get("predictions_path", "") if run_row else ""
    if not path:
        path = str(get_data_dir(base_dir, scope_key) / f"predictions_{run_id}.csv")
    return Path(path)


def resolve_odds_path(get_data_dir, base_dir, scope_key, run_id, run_row):
    path = run_row.get("odds_path", "") if run_row else ""
    if path:
        return Path(path)
    race_id = str(run_row.get("race_id", "") or "") if run_row else ""
    if race_id:
        race_dir = get_data_dir(base_dir, scope_key) / race_id
        return race_dir / f"odds_{run_id}_{race_id}.csv"
    return None


def resolve_wide_odds_path(get_data_dir, base_dir, scope_key, run_id, run_row):
    path = run_row.get("wide_odds_path", "") if run_row else ""
    if path:
        return Path(path)
    race_id = str(run_row.get("race_id", "") or "") if run_row else ""
    if race_id:
        race_dir = get_data_dir(base_dir, scope_key) / race_id
        return race_dir / f"wide_odds_{run_id}_{race_id}.csv"
    return None


def resolve_run_asset_path(get_data_dir, base_dir, scope_key, run_id, run_row, field_name, prefix):
    path = str(run_row.get(field_name, "")).strip() if run_row else ""
    if path:
        return Path(path)
    race_id = str(run_row.get("race_id", "") or "") if run_row else ""
    if race_id:
        race_dir = get_data_dir(base_dir, scope_key) / race_id
        return race_dir / f"{prefix}_{run_id}_{race_id}.csv"
    return None

from pathlib import Path

from surface_scope import normalize_scope_key
from web_report.helpers import has_llm_policy_assets, report_scope_key_for_row, safe_text


def load_combined_llm_report_runs(load_runs, scope_keys):
    runs = []
    for report_scope_key in list(scope_keys or []):
        for row in load_runs(report_scope_key):
            if not has_llm_policy_assets(row):
                continue
            item = dict(row)
            item["_report_scope_key"] = report_scope_key
            runs.append(item)
    return runs


def load_actual_result_map(
    base_dir,
    scope_key,
    *,
    get_data_dir,
    load_csv_rows,
    canonical_predictor_id,
    load_race_jobs,
):
    path = get_data_dir(base_dir, scope_key) / "predictor_results.csv"
    rows = load_csv_rows(path) if Path(path).exists() else []
    result_map = {}
    for row in rows:
        run_id = safe_text(row.get("run_id"))
        if not run_id:
            continue
        predictor_id = canonical_predictor_id(row.get("predictor_id"))
        current = result_map.get(run_id)
        item = {
            "actual_top1": safe_text(row.get("actual_top1")),
            "actual_top2": safe_text(row.get("actual_top2")),
            "actual_top3": safe_text(row.get("actual_top3")),
        }
        if current is None or predictor_id == "main":
            result_map[run_id] = item
    scope_norm = normalize_scope_key(scope_key)
    for job in load_race_jobs(base_dir):
        if normalize_scope_key(safe_text(job.get("scope_key"))) != scope_norm:
            continue
        run_id = safe_text(job.get("current_run_id"))
        if not run_id:
            continue
        item = {
            "actual_top1": safe_text(job.get("actual_top1")),
            "actual_top2": safe_text(job.get("actual_top2")),
            "actual_top3": safe_text(job.get("actual_top3")),
        }
        if not any(item.values()):
            continue
        current = dict(result_map.get(run_id, {}) or {})
        if not any(safe_text(current.get(key)) for key in ("actual_top1", "actual_top2", "actual_top3")):
            result_map[run_id] = item
    return result_map


def find_actual_result_from_jobs(base_dir, scope_key, run_id, run_row=None, *, load_race_jobs):
    scope_norm = normalize_scope_key(scope_key)
    run_id_text = safe_text(run_id)
    race_id_text = safe_text((run_row or {}).get("race_id"))
    best = {}
    best_time = ""
    for job in load_race_jobs(base_dir):
        if normalize_scope_key(safe_text(job.get("scope_key"))) != scope_norm:
            continue
        job_run_id = safe_text(job.get("current_run_id"))
        job_race_id = safe_text(job.get("race_id"))
        matched = bool((run_id_text and job_run_id == run_id_text) or (race_id_text and job_race_id == race_id_text))
        if not matched:
            continue
        item = {
            "actual_top1": safe_text(job.get("actual_top1")),
            "actual_top2": safe_text(job.get("actual_top2")),
            "actual_top3": safe_text(job.get("actual_top3")),
        }
        if not any(item.values()):
            continue
        settled_at = safe_text(job.get("settled_at")) or safe_text(job.get("updated_at")) or safe_text(job.get("created_at"))
        if not best or settled_at >= best_time:
            best = item
            best_time = settled_at
    return best


def find_job_meta_for_run(base_dir, scope_key, run_id, run_row=None, *, load_race_jobs):
    scope_norm = normalize_scope_key(scope_key)
    run_id_text = safe_text(run_id)
    race_id_text = safe_text((run_row or {}).get("race_id"))
    best = {}
    best_time = ""
    for job in load_race_jobs(base_dir):
        if normalize_scope_key(safe_text(job.get("scope_key"))) != scope_norm:
            continue
        job_run_id = safe_text(job.get("current_run_id"))
        job_race_id = safe_text(job.get("race_id"))
        matched = bool((run_id_text and job_run_id == run_id_text) or (race_id_text and job_race_id == race_id_text))
        if not matched:
            continue
        updated_at = safe_text(job.get("updated_at")) or safe_text(job.get("created_at"))
        if not best or updated_at >= best_time:
            best = {
                "job_id": safe_text(job.get("job_id")),
                "location": safe_text(job.get("location")),
                "scheduled_off_time": safe_text(job.get("scheduled_off_time")),
                "target_distance": safe_text(job.get("target_distance")),
                "target_track_condition": safe_text(job.get("target_track_condition")),
                "race_date": safe_text(job.get("race_date")),
            }
            best_time = updated_at
    return best


def load_name_to_no_map_for_run(
    scope_key,
    run_id,
    run_row,
    *,
    resolve_run_asset_path,
    load_name_to_no,
    normalize_name,
    normalize_horse_no_text,
):
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


def actual_result_snapshot(
    base_dir,
    scope_key,
    run_id,
    run_row,
    actual_result_map,
    *,
    load_race_jobs,
    resolve_run_asset_path,
    load_name_to_no,
    normalize_name,
    normalize_horse_no_text,
):
    actual = dict((actual_result_map or {}).get(run_id, {}) or {})
    if not any(safe_text(actual.get(key)) for key in ("actual_top1", "actual_top2", "actual_top3")):
        actual = find_actual_result_from_jobs(
            base_dir,
            scope_key,
            run_id,
            run_row,
            load_race_jobs=load_race_jobs,
        )
    actual_names = [
        safe_text(actual.get("actual_top1")),
        safe_text(actual.get("actual_top2")),
        safe_text(actual.get("actual_top3")),
    ]
    name_to_no = load_name_to_no_map_for_run(
        scope_key,
        run_id,
        run_row,
        resolve_run_asset_path=resolve_run_asset_path,
        load_name_to_no=load_name_to_no,
        normalize_name=normalize_name,
        normalize_horse_no_text=normalize_horse_no_text,
    )
    actual_horse_nos = []
    for name in actual_names:
        actual_horse_nos.append(name_to_no.get(normalize_name(name), "") if name else "")
    return {
        "actual_names": actual_names,
        "actual_horse_nos": actual_horse_nos,
    }

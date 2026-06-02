from __future__ import annotations

from datetime import date
from pathlib import Path

from keiba_llm_agent.schemas.race_data import RaceData


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _is_abnormal_recent_race_id(race_id: str | None) -> bool:
    if not race_id:
        return False
    return (not race_id.isdigit()) or len(race_id) < 10


def validate_race_data(race_data: RaceData) -> dict[str, object]:
    race_id = race_data.race_info.race_id or "unknown"
    errors: list[str] = []
    warnings: list[str] = []

    horse_count = len(race_data.horses)
    odds_missing_count = sum(1 for horse in race_data.horses if horse.odds is None)
    popularity_missing_count = sum(1 for horse in race_data.horses if horse.popularity is None)
    recent_runs_missing_count = sum(1 for horse in race_data.horses if not horse.recent_runs)
    total_recent_runs = sum(len(horse.recent_runs) for horse in race_data.horses)
    pedigree_missing_count = sum(
        1
        for horse in race_data.horses
        if not getattr(horse, "horse_id", None)
    )

    race_date_text = race_data.race_info.race_date
    race_date = _parse_date(race_date_text)
    future_race_leakage_count = 0
    null_course_count = 0
    abnormal_recent_race_id_count = 0
    null_finish_count = 0
    null_field_size_count = 0

    for horse in race_data.horses:
        for recent_run in horse.recent_runs:
            recent_date = _parse_date(recent_run.date)
            if race_date is not None and recent_date is not None and recent_date >= race_date:
                future_race_leakage_count += 1
            if recent_run.course is None:
                null_course_count += 1
            if _is_abnormal_recent_race_id(recent_run.race_id):
                abnormal_recent_race_id_count += 1
            if recent_run.finish is None:
                null_finish_count += 1
            if recent_run.field_size is None:
                null_field_size_count += 1

    odds_missing_rate = (odds_missing_count / horse_count) if horse_count else 1.0
    popularity_missing_rate = (popularity_missing_count / horse_count) if horse_count else 1.0
    recent_runs_missing_rate = (recent_runs_missing_count / horse_count) if horse_count else 1.0
    incomplete_recent_runs_count = (
        null_course_count
        + abnormal_recent_race_id_count
        + null_finish_count
        + null_field_size_count
    )

    if not race_data.race_info.race_id:
        errors.append("race_info.race_id is missing")
    if not race_date_text:
        errors.append("race_info.race_date is missing")
    if horse_count == 0:
        errors.append("horses are empty")
    if horse_count > 0 and recent_runs_missing_count == horse_count:
        errors.append("recent_runs are empty for all horses")
    if future_race_leakage_count > 0:
        errors.append("future race leakage detected in recent_runs")

    if 0 < horse_count < 5:
        warnings.append("horse count is less than 5")
    if odds_missing_rate > 0.5:
        warnings.append("odds missing rate is above 0.5")
    if popularity_missing_rate > 0.5:
        warnings.append("popularity missing rate is above 0.5")
    if horse_count and (pedigree_missing_count / horse_count) > 0:
        warnings.append("pedigree missing rate is above 0.0")
    if null_course_count > 0:
        warnings.append("recent_runs contain course=null")
    if abnormal_recent_race_id_count > 0:
        warnings.append("recent_runs contain abnormal race_id")

    status = "ERROR" if errors else "WARNING" if warnings else "OK"
    return {
        "race_id": race_id,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "horse_count": horse_count,
            "odds_missing_rate": round(odds_missing_rate, 4),
            "popularity_missing_rate": round(popularity_missing_rate, 4),
            "recent_runs_missing_rate": round(recent_runs_missing_rate, 4),
            "pedigree_missing_rate": round((pedigree_missing_count / horse_count) if horse_count else 1.0, 4),
            "total_recent_runs": total_recent_runs,
            "incomplete_recent_runs_count": incomplete_recent_runs_count,
            "future_race_leakage_count": future_race_leakage_count,
            "null_course_count": null_course_count,
            "abnormal_recent_race_id_count": abnormal_recent_race_id_count,
            "null_finish_count": null_finish_count,
            "null_field_size_count": null_field_size_count,
        },
    }


def validate_race_data_file(path: str | Path) -> dict[str, object]:
    return validate_race_data(RaceData.from_json_file(path))

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import requests

from keiba_llm_agent.parsers.netkeiba_horse_parser import (
    has_recent_runs_table,
    parse_horse_recent_runs,
)
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData


BASE_DIR = Path(__file__).resolve().parents[1]
HORSE_HTML_CACHE_DIR = BASE_DIR / "data" / "horse_html_cache"
DB_NETKEIBA_ENCODING = "euc_jp"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def get_horse_cache_path(horse_id: str) -> Path:
    return HORSE_HTML_CACHE_DIR / f"{horse_id}.html"


def get_horse_result_url(horse_id: str) -> str:
    return f"https://db.netkeiba.com/horse/result/{horse_id}/"


def fetch_horse_html(horse_id: str, force_refresh: bool = False) -> str:
    cache_path = get_horse_cache_path(horse_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        html = cache_path.read_text(encoding="utf-8")
        if has_recent_runs_table(html):
            return html
        warnings.warn(
            f"stale horse cache detected, refetching result page: horse_id={horse_id}",
            stacklevel=2,
        )

    url = get_horse_result_url(horse_id)
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"failed to fetch netkeiba horse page: {horse_id}") from exc

    response.encoding = DB_NETKEIBA_ENCODING
    html = response.text
    cache_path.write_text(html, encoding="utf-8")
    return html


def parse_iso_date_safe(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def filter_recent_runs_for_target_race(
    recent_runs,
    target_race_date: str | None,
    target_race_id: str | None,
    horse_id: str | None,
    limit: int | None,
):
    if limit is not None and limit <= 0:
        return []

    if not target_race_date:
        warnings.warn(
            "race_date is missing; recent_runs may contain target race result",
            stacklevel=2,
        )
        return []

    target_date = parse_iso_date_safe(target_race_date)
    if target_date is None:
        warnings.warn(
            f"invalid race_date format; recent_runs skipped: race_date={target_race_date}",
            stacklevel=2,
        )
        return []

    filtered = []
    for recent_run in recent_runs:
        if target_race_id and recent_run.race_id == target_race_id:
            continue
        run_date = parse_iso_date_safe(recent_run.date)
        if run_date is None:
            warnings.warn(
                f"recent run date missing or invalid, excluded: horse_id={horse_id}",
                stacklevel=2,
            )
            continue
        if run_date >= target_date:
            continue
        filtered.append(recent_run)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def enrich_race_data_with_recent_runs(
    race_data: RaceData,
    limit: int | None = None,
    force_refresh: bool = False,
) -> RaceData:
    enriched_horses: list[HorseEntry] = []
    target_race_date = race_data.race_info.race_date
    target_race_id = race_data.race_info.race_id
    for horse in race_data.horses:
        if not horse.horse_id:
            enriched_horses.append(horse.model_copy(update={"recent_runs": []}))
            continue

        recent_runs = []
        try:
            html = fetch_horse_html(horse.horse_id, force_refresh=force_refresh)
            parsed_runs = parse_horse_recent_runs(html, limit=None)
            recent_runs = filter_recent_runs_for_target_race(
                parsed_runs,
                target_race_date=target_race_date,
                target_race_id=target_race_id,
                horse_id=horse.horse_id,
                limit=limit,
            )
            if not recent_runs:
                warnings.warn(
                    f"no recent runs parsed: horse_id={horse.horse_id}",
                    stacklevel=2,
                )
        except Exception as exc:
            warnings.warn(
                f"failed to enrich horse recent runs: horse_id={horse.horse_id}: {exc}",
                stacklevel=2,
            )
        enriched_horses.append(horse.model_copy(update={"recent_runs": recent_runs}))

    return race_data.model_copy(update={"horses": enriched_horses})

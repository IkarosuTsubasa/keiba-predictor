from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import requests

from keiba_llm_agent.parsers.netkeiba_race_parser import parse_netkeiba_shutuba_html
from keiba_llm_agent.parsers.netkeiba_url_parser import extract_race_id
from keiba_llm_agent.schemas.race_data import RaceData


BASE_DIR = Path(__file__).resolve().parents[1]
HTML_CACHE_DIR = BASE_DIR / "data" / "html_cache"
RACE_DATA_DIR = BASE_DIR / "data" / "race_data"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def get_html_cache_path(race_id: str) -> Path:
    return HTML_CACHE_DIR / f"{race_id}.html"


def get_race_data_path(race_id: str) -> Path:
    return RACE_DATA_DIR / f"{race_id}.json"


def fetch_netkeiba_html(url: str, force_refresh: bool = False) -> tuple[str, str]:
    race_id = extract_race_id(url)
    cache_path = get_html_cache_path(race_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        return race_id, cache_path.read_text(encoding="utf-8")

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"failed to fetch netkeiba URL: {url}") from exc

    if getattr(response, "apparent_encoding", None):
        response.encoding = response.apparent_encoding
    html = response.text
    cache_path.write_text(html, encoding="utf-8")
    return race_id, html


def parse_current_odds_response(response_text: str) -> dict[int, tuple[float | None, int | None]]:
    payload_text = response_text.strip()
    if payload_text.startswith("(") and payload_text.endswith(")"):
        payload_text = payload_text[1:-1]
    payload = json.loads(payload_text)
    compressed = payload.get("data")
    if not compressed:
        return {}
    decoded = zlib.decompress(base64.b64decode(compressed))
    body = json.loads(decoded)
    odds_rows = body.get("odds", {}).get("1", {})
    odds_map: dict[int, tuple[float | None, int | None]] = {}
    for horse_no_key, row in odds_rows.items():
        try:
            horse_no = int(horse_no_key)
        except ValueError:
            continue
        odds_value = None
        popularity_value = None
        if isinstance(row, list) and row:
            try:
                odds_value = float(row[0]) if row[0] not in ("", "---.-", "--") else None
            except (TypeError, ValueError):
                odds_value = None
            try:
                popularity_value = int(row[2]) if len(row) > 2 else None
            except (TypeError, ValueError):
                popularity_value = None
        odds_map[horse_no] = (odds_value, popularity_value)
    return odds_map


def fetch_netkeiba_current_odds(race_id: str) -> dict[int, tuple[float | None, int | None]]:
    params = {
        "pid": "api_get_jra_odds",
        "input": "UTF-8",
        "output": "jsonp",
        "race_id": race_id,
        "type": "1",
        "action": "init",
        "sort": "odds",
        "compress": "1",
    }
    headers = dict(DEFAULT_HEADERS)
    headers["Referer"] = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        response = requests.get(
            "https://race.netkeiba.com/api/api_get_jra_odds.html",
            headers=headers,
            params=params,
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"failed to fetch current netkeiba odds for race_id={race_id}") from exc
    return parse_current_odds_response(response.text)


def enrich_race_data_with_current_odds(race_data: RaceData) -> RaceData:
    odds_map = fetch_netkeiba_current_odds(race_data.race_info.race_id)
    if not odds_map:
        return race_data
    for horse in race_data.horses:
        current_values = odds_map.get(horse.horse_no)
        if current_values is None:
            continue
        current_odds, current_popularity = current_values
        if current_odds is not None:
            horse.odds = current_odds
        if current_popularity is not None:
            horse.popularity = current_popularity
    return race_data


def fetch_and_parse_netkeiba_race(url: str, force_refresh: bool = False) -> RaceData:
    race_id, html = fetch_netkeiba_html(url, force_refresh=force_refresh)
    race_data = parse_netkeiba_shutuba_html(html, race_id=race_id)
    if race_data.horses and all(
        horse.odds is None and horse.popularity is None
        for horse in race_data.horses
    ):
        try:
            race_data = enrich_race_data_with_current_odds(race_data)
        except RuntimeError:
            pass
    return race_data


def save_race_data(race_data: RaceData) -> Path:
    race_id = race_data.race_info.race_id
    output_path = get_race_data_path(race_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(race_data.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path

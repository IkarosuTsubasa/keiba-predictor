from __future__ import annotations

import json
from pathlib import Path

import requests

from keiba_llm_agent.parsers.netkeiba_result_parser import parse_netkeiba_result_html
from keiba_llm_agent.parsers.netkeiba_url_parser import extract_race_id
from keiba_llm_agent.schemas.result import ResultData


BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_HTML_CACHE_DIR = BASE_DIR / "data" / "result_html_cache"
RESULTS_DIR = BASE_DIR / "data" / "results"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def get_result_cache_path(race_id: str) -> Path:
    return RESULT_HTML_CACHE_DIR / f"{race_id}.html"


def get_result_data_path(race_id: str) -> Path:
    return RESULTS_DIR / f"{race_id}.json"


def fetch_netkeiba_result_html(url: str, force_refresh: bool = False) -> tuple[str, str]:
    race_id = extract_race_id(url)
    cache_path = get_result_cache_path(race_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        return race_id, cache_path.read_text(encoding="utf-8")

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"failed to fetch netkeiba result URL: {url}") from exc

    if getattr(response, "apparent_encoding", None):
        response.encoding = response.apparent_encoding
    html = response.text
    cache_path.write_text(html, encoding="utf-8")
    return race_id, html


def fetch_and_parse_netkeiba_result(url: str, force_refresh: bool = False) -> ResultData:
    race_id, html = fetch_netkeiba_result_html(url, force_refresh=force_refresh)
    return parse_netkeiba_result_html(html, race_id=race_id)


def save_result_data(result_data: ResultData) -> Path:
    output_path = get_result_data_path(result_data.race_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_data.model_dump(by_alias=True), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path

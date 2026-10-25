"""Capture public workout evidence without changing prediction inputs or scores."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import requests
from bs4 import BeautifulSoup

from keiba_llm_agent.fetchers.netkeiba_fetcher import DEFAULT_HEADERS
from keiba_llm_agent.schemas.race_data import RaceData

OBSERVATION_DIR = Path(__file__).resolve().parents[1] / "data" / "workout_observations"
CACHE_SECONDS = 600
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
SCHEMA_VERSION = "netkeiba_public_workout_observation_v1"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _race_id_from_url(url: str) -> str | None:
    return parse_qs(urlparse(url).query).get("race_id", [None])[0]


def parse_workout_page(html: str, race_data: RaceData) -> dict:
    """Validate page and runner identity before exposing whitelisted public fields."""
    soup = BeautifulSoup(html, "html.parser")
    canonical = soup.select_one('link[rel="canonical"]')
    page_id = _race_id_from_url(str(canonical.get("href", ""))) if canonical else None
    if page_id != race_data.race_info.race_id:
        return {"status": "page_identity_unverified", "horses": {}}
    table = soup.select_one("table.OikiriTable")
    if table is None:
        return {"status": "table_unavailable", "horses": {}}
    expected = {str(h.horse_no): str(h.horse_id or "") for h in race_data.horses}
    if len(expected) != len(race_data.horses):
        return {"status": "roster_mismatch", "horses": {}}
    horses = {}
    for row in table.select("tr.HorseList"):
        number_cell = row.select_one("td.Umaban")
        if number_cell is None:
            continue
        number = number_cell.get_text(strip=True)
        if not number.isdigit():
            continue
        number = str(int(number))
        link = row.select_one('a[href*="/horse/"]')
        match = re.search(r"/horse/(\d+)", str(link.get("href", ""))) if link else None
        horse_id = match.group(1) if match else None
        if (number in horses or number not in expected or not horse_id
                or not expected[number] or horse_id != expected[number]):
            return {"status": "roster_mismatch", "horses": {}}
        grade_cell = row.select_one('td[class*="Rank_"]')
        grade = grade_cell.get_text(strip=True) if grade_cell else None
        critic_cell = row.select_one("td.Training_Critic")
        horses[number] = {
            "horse_id": horse_id,
            "grade": grade if grade in ("A", "B", "C", "D") else None,
            "critic": critic_cell.get_text(strip=True) if critic_cell else None,
        }
    available = sum(h["grade"] is not None for h in horses.values())
    return {
        "status": "collected" if available == len(expected) and expected else "partial",
        "horses": horses,
        "expected_horses": len(expected),
        "matched_horses": len(horses),
        "available_grades": available,
    }


def _is_pre_race(race_data: RaceData, received_at: str) -> bool | None:
    try:
        start = datetime.fromisoformat(race_data.race_info.scheduled_start_at or "")
        if start.tzinfo is None:
            return None
        return datetime.fromisoformat(received_at) < start
    except (ValueError, TypeError):
        return None


def _capture(race_data: RaceData, output_dir: Path, persist: bool) -> dict:
    info = race_data.race_info
    race_id = str(info.race_id)
    observation = {
        "schema_version": SCHEMA_VERSION,
        "race_id": race_id,
        "capture_requested_at": _now().isoformat(),
        "source_published_at": None,
        "used_for_scoring": False,
        "used_for_llm": False,
        "horses": {},
    }
    if os.environ.get("KEIBA_CAPTURE_PUBLIC_WORKOUTS", "1").lower() in ("0", "false", "off"):
        return {**observation, "status": "disabled"}
    if (not re.fullmatch(r"\d{12}", race_id) or not 1 <= int(race_id[4:6]) <= 10
            or str(info.source or "").lower() == "local"
            or str(info.scope_key or "").lower().startswith("local")
            or info.surface not in ("芝", "ダート") or not race_data.horses):
        return {**observation, "status": "unsupported"}

    roster = sorted((h.horse_no, h.horse_id) for h in race_data.horses)
    roster_hash = hashlib.sha256(json.dumps(roster).encode()).hexdigest()
    observation["roster_sha256"] = roster_hash
    archive_dir = Path(output_dir) / race_id
    # Per-worker cooldown also applies to failed responses. Never overwrite evidence.
    if persist and archive_dir.exists():
        previous = sorted(archive_dir.glob("*.json.gz"), reverse=True)
        if previous:
            try:
                cached = json.loads(gzip.decompress(previous[0].read_bytes()))
                age = (_now() - datetime.fromisoformat(cached["capture_requested_at"])).total_seconds()
                if (cached.get("schema_version") == SCHEMA_VERSION
                        and cached.get("race_id") == race_id
                        and cached.get("roster_sha256") == roster_hash and 0 <= age < CACHE_SECONDS):
                    return {**cached, "reused_at": observation["capture_requested_at"]}
            except (OSError, ValueError, KeyError, TypeError, EOFError):
                pass

    url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    observation["source_url"] = url
    observation["request_started_at"] = _now().isoformat()
    try:
        # No login, redirects, retries, private endpoints, or paid workout times.
        started = time.monotonic()
        with requests.get(url, headers=DEFAULT_HEADERS, timeout=(3, 6),
                          allow_redirects=False, stream=True) as response:
            observation["http_status"] = response.status_code
            if response.status_code in (401, 403, 429):
                observation["status"] = "restricted"
            elif response.status_code != 200:
                observation["status"] = "http_error"
            else:
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=16384):
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES or time.monotonic() - started > 12:
                        raise ValueError("response_limit")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                observation["received_at"] = _now().isoformat()
                observation["received_before_scheduled_start"] = _is_pre_race(race_data, observation["received_at"])
                observation["scheduled_start_at"] = info.scheduled_start_at
                # Keep exact response bytes. The parser only reads public grades/critics.
                observation["raw_response"] = {
                    "encoding": "gzip+base64", "text_encoding": "utf-8",
                    "sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw),
                    "data": base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii"),
                }
                observation.update(parse_workout_page(raw.decode("utf-8", errors="strict"), race_data))
    except Exception as exc:
        observation.update(status="fetch_failed", error_type=type(exc).__name__)
    observation["completed_at"] = _now().isoformat()
    if persist:
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            name = _now().strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex + ".json.gz"
            with (archive_dir / name).open("xb") as handle:
                handle.write(gzip.compress(json.dumps(observation, ensure_ascii=False).encode("utf-8"), mtime=0))
        except OSError as exc:
            # The observation is still embedded in the prediction audit/callback.
            observation["archive_error_type"] = type(exc).__name__
    return observation


def capture_public_workout(race_data: RaceData, *, output_dir: Path | None = None,
                           persist: bool = True) -> dict:
    """Best-effort online collection; never invoked by offline scoring/backtests."""
    try:
        return _capture(race_data, Path(output_dir) if output_dir is not None else OBSERVATION_DIR, persist)
    except Exception as exc:
        return {"schema_version": SCHEMA_VERSION, "race_id": race_data.race_info.race_id,
                "capture_requested_at": _now().isoformat(), "status": "capture_failed",
                "error_type": type(exc).__name__, "used_for_scoring": False, "used_for_llm": False,
                "horses": {}}

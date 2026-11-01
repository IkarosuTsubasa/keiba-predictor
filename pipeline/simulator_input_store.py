"""Versioned, market-free simulator inputs. Agent source data is read-only here."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import time
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

MODEL_VERSION = "illustrative-terrain-v2"
SCHEMA_VERSION = 1
JST = timezone(timedelta(hours=9))
_PUBLISH_LOCK = threading.RLock()
INFO_FIELDS = ("race_id", "race_date", "race_name", "course", "surface", "distance", "track_condition", "weather", "source", "scope_key", "course_layout", "scheduled_start_at")
HORSE_FIELDS = ("horse_no", "frame_no", "horse_id", "horse_name", "jockey", "carried_weight", "entry_status")
RUN_FIELDS = ("race_id", "date", "course", "surface", "distance", "track_condition", "finish", "field_size", "corner_positions", "final_3f", "time_seconds")


def race_identity(race_id, race_date):
    rid = str(race_id or "")
    day = str(race_date or "")
    if not re.fullmatch(r"\d{12}", rid):
        raise ValueError("invalid_race_id")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("invalid_date")
    date.fromisoformat(day)
    return rid, day


def now_text():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(path):
    try:
        raw = path.read_bytes()
        return json.loads((gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8-sig"))
    except (OSError, ValueError, EOFError):
        return None


def _write(path, payload, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temporary files publish complete versions without replacing them.
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".publish-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        if immutable:
            try:
                if os.name == "nt":
                    os.rename(temp, path)  # Windows rename never overwrites.
                else:
                    os.link(temp, path)
            except FileExistsError:
                pass
        else:
            for attempt in range(8):
                try:
                    os.replace(temp, path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(.01 * (attempt + 1))
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _fields(row, fields):
    return {key: row.get(key) for key in fields}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _seconds(value):
    try:
        if isinstance(value, str) and ":" in value:
            minute, second = value.split(":", 1)
            return int(minute) * 60 + float(second)
        return float(value)
    except (ValueError, TypeError):
        return None


def _historical_clock(base_dir, horse, run):
    """Match only the requested prior race; ambiguous identities stay missing."""
    rid = str(run.get("race_id") or "")
    if not re.fullmatch(r"\d{12}", rid):
        return
    result = _json(Path(base_dir) / "data/agent_results" / f"{rid}.json") or {}
    rows = result.get("finish_order") or result.get("results") or []
    if not isinstance(rows, list):
        return
    # A result without independently verified date and distance is insufficient.
    info = result.get("race_info") or result
    historical = _json(Path(base_dir).parent / "keiba_llm_agent/data/race_data" / f"{rid}.json") or {}
    if not historical:
        audit = _json(Path(base_dir) / "data/agent_backtest_audit" / f"{rid}.json.gz") or {}
        historical = (audit.get("inputs") or {}).get("race_data") or {}
    if historical:
        info = historical.get("race_info") or {}
    if (info.get("race_date") or info.get("date")) != run.get("date") or info.get("distance") != run.get("distance"):
        return
    hid = horse.get("horse_id")
    if historical:
        identities = [h for h in historical.get("horses", []) if (str(h.get("horse_id")) == str(hid) if hid and h.get("horse_id") else h.get("horse_name") == horse.get("horse_name"))]
        if len(identities) != 1:
            return
        matched = identities[0]
        rows = [r for r in rows if str(r.get("horse_no")) == str(matched.get("horse_no")) and r.get("horse_name") == matched.get("horse_name")]
    matches = [r for r in rows if isinstance(r, dict) and (
        str(r.get("horse_id")) == str(hid) if hid and r.get("horse_id") else r.get("horse_name") == horse.get("horse_name")
    )]
    if len(matches) != 1 or matches[0].get("finish") != run.get("finish"):
        return
    row = matches[0]
    value = _seconds(row.get("time_seconds") or row.get("time"))
    if value and 7 < run["distance"] / value < 22:
        run["time_seconds"] = value


def sanitize_race(raw, *, base_dir=None):
    if not isinstance(raw, dict) or not isinstance(raw.get("race_info"), dict):
        raise ValueError("invalid_race_input")
    info = _fields(raw.get("race_info") or {}, INFO_FIELDS)
    if any(v is not None and not isinstance(v, str) for k, v in info.items() if k != "distance"):
        raise ValueError("invalid_race_fields")
    rid, day = race_identity(info["race_id"], info["race_date"])
    distance = info.get("distance")
    if not _number(distance) or not 100 <= distance <= 6000:
        raise ValueError("invalid_distance")
    horses, seen, missing = [], set(), []
    for original in raw.get("horses") or []:
        if not isinstance(original, dict):
            raise ValueError("invalid_horse")
        horse = _fields(original, HORSE_FIELDS)
        if any(horse.get(k) is not None and not isinstance(horse[k], str) for k in ("horse_id", "horse_name", "jockey", "entry_status")):
            raise ValueError("invalid_horse_fields")
        for key in ("frame_no", "carried_weight"):
            if not _number(horse.get(key)):
                horse[key] = None
        no = horse.get("horse_no")
        if not isinstance(no, int) or isinstance(no, bool) or not 1 <= no <= 30 or no in seen or not horse.get("horse_name"):
            raise ValueError("invalid_horse_identity")
        seen.add(no)
        status = str(horse.get("entry_status") or "active")
        horse["entry_status"] = status if status in ("active", "scratched", "excluded") else "unknown"
        runs = []
        for original_run in original.get("recent_runs") or []:
            if not isinstance(original_run, dict):
                continue
            run = _fields(original_run, RUN_FIELDS)
            if any(v is not None and not isinstance(v, str) for k, v in run.items() if k in ("race_id", "date", "course", "surface", "track_condition")):
                continue
            try:
                valid = date.fromisoformat(str(run.get("date"))) < date.fromisoformat(day)
            except ValueError:
                valid = False
            if not valid or str(run.get("race_id")) == rid:
                continue
            for key in ("distance", "finish", "field_size", "final_3f", "time_seconds"):
                if not _number(run.get(key)):
                    run[key] = None
            if not _number(run.get("distance")) or run["distance"] <= 0:
                continue
            positions = run.get("corner_positions")
            run["corner_positions"] = [v for v in (positions if isinstance(positions, list) else []) if isinstance(v, int) and not isinstance(v, bool) and v > 0]
            if base_dir and run.get("time_seconds") is None:
                _historical_clock(base_dir, horse, run)
            runs.append(run)
        horse["recent_runs"] = runs
        if horse["entry_status"] == "active":
            if not runs:
                missing.append(f"horses.{no}.recent_runs")
            if not any(r.get("final_3f") for r in runs):
                missing.append(f"horses.{no}.final_3f")
        horses.append(horse)
    if not horses:
        raise ValueError("empty_entries")
    for key in ("track_condition", "weather", "course_layout"):
        if not info.get(key):
            missing.append(f"race_info.{key}")
    supported = info.get("surface") in ("芝", "ダート") and info.get("course") != "帯広" and "障" not in str((raw.get("race_info") or {}).get("race_name") or "")
    availability = "unsupported" if not supported else "partial" if missing else "ready"
    if supported and len([h for h in horses if h["entry_status"] == "active"]) < 2:
        availability = "unavailable"
    return {"race_info": info, "horses": sorted(horses, key=lambda h: h["horse_no"]), "availability": availability, "missing_fields": missing}


def _root(base_dir, rid, day):
    return Path(base_dir) / "data/simulator_inputs" / day / rid


@contextmanager
def _index_lock(root):
    # Coordinate both worker threads and independent collector/server processes.
    root.mkdir(parents=True, exist_ok=True)
    with _PUBLISH_LOCK, (root / ".index.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            for attempt in range(500):
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if attempt == 499:
                        raise
                    time.sleep(.01)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def publish_race(base_dir, raw, *, source="collector", source_as_of=None):
    clean = sanitize_race(raw, base_dir=base_dir)
    info = clean["race_info"]
    rid, day = race_identity(info["race_id"], info["race_date"])
    content = {"schema_version": SCHEMA_VERSION, **clean}
    digest = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    version = digest[:24]
    root = _root(base_dir, rid, day)
    existing = _json(root / "latest.json")
    if existing and existing.get("input_version") == version:
        return read_race(base_dir, rid, day, version)
    published = now_text()
    start = (raw.get("race_info") or {}).get("scheduled_start_at")
    # Times are metadata, never inferred from the absence of a result.
    if start:
        try:
            parsed = datetime.fromisoformat(start)
            if parsed.date().isoformat() != day:
                start = None
            elif parsed.tzinfo is None:
                start = parsed.replace(tzinfo=JST).isoformat()
        except (ValueError, TypeError):
            start = None
    payload = {**content, "race_id": rid, "race_date": day, "input_version": version, "input_hash": digest,
               "published_at": published, "source_as_of": source_as_of, "scheduled_off_at": start,
               "active_model_version": MODEL_VERSION, "default_seed": int(digest[:8], 16),
               "provenance": {"source": source, "uses_market": False, "uses_target_result": False},
               "freeze_evidence": "source_timestamp_unverified"}
    _write(root / f"{version}.json", payload, immutable=True)
    saved = _json(root / f"{version}.json")
    # Audit compatibility imports must not roll back a newer collected card.
    with _index_lock(root):
        prior_stamp = (_json(root / "latest.json") or {}).get("source_as_of")
        if not prior_stamp or source_as_of and str(source_as_of) >= str(prior_stamp):
            _write(root / "latest.json", summary(saved))
    return saved


def summary(payload):
    info = payload["race_info"]
    return {key: payload.get(key) for key in ("schema_version", "race_id", "race_date", "input_version", "availability", "published_at", "source_as_of", "scheduled_off_at", "active_model_version", "default_seed")} | {
        "course": info.get("course"), "surface": info.get("surface"), "distance": info.get("distance"),
        "scope_key": info.get("scope_key"), "race_name": info.get("race_name"),
        "horse_count": len([h for h in payload["horses"] if h["entry_status"] == "active"])}


def read_race(base_dir, rid, day, version=""):
    rid, day = race_identity(rid, day)
    if version and not re.fullmatch(r"[a-f0-9]{24}", version):
        raise ValueError("invalid_input_version")
    root = _root(base_dir, rid, day)
    if not version:
        latest = _json(root / "latest.json") or {}
        version = latest.get("input_version", "")
    return _json(root / f"{version}.json") if version else None


def import_saved_race(base_dir, rid, day):
    """Compatibility only; no network or prediction is triggered by a page view."""
    rid, day = race_identity(rid, day)
    root = Path(base_dir)
    candidates = [root.parent / "keiba_llm_agent/data/race_data" / f"{rid}.json",
                  root / "data/agent_backtest_audit" / f"{rid}.json.gz"]
    for path in candidates:
        raw = _json(path) or {}
        if path.suffix == ".gz":
            raw = (raw.get("inputs") or {}).get("race_data") or {}
        if (raw.get("race_info") or {}).get("race_date") == day:
            try:
                return publish_race(root, raw, source="saved_agent_input", source_as_of=None)
            except ValueError:
                continue
    return None


def list_races(base_dir, day, scheduled=(), *, scope_key="", state="all"):
    date.fromisoformat(day)
    entries = {}
    for row in scheduled:
        rid = str(row.get("race_id") or "")
        if str(row.get("race_date") or row.get("date") or "") != day or not re.fullmatch(r"\d{12}", rid):
            continue
        entries[rid] = {"race_id": rid, "race_date": day, "course": row.get("location") or row.get("course"),
                        "race_name": row.get("race_name") or row.get("race_title"), "scope_key": row.get("scope_key"),
                        "scheduled_off_at": row.get("scheduled_off_time"), "availability": "preparing", "horse_count": None}
    for path in (Path(base_dir) / "data/simulator_inputs" / day).glob("*/latest.json"):
        row = _json(path)
        if row:
            prior = entries.get(row["race_id"], {})
            entries[row["race_id"]] = {**prior, **row, "scheduled_off_at": row.get("scheduled_off_at") or prior.get("scheduled_off_at")}
    current = datetime.now(JST)
    output = []
    for row in entries.values():
        if scope_key and not str(row.get("scope_key") or "").startswith(scope_key):
            continue
        start = row.get("scheduled_off_at")
        try:
            off = datetime.fromisoformat(start)
            if off.tzinfo is None:
                off = off.replace(tzinfo=JST)
            row["race_state"] = "scheduled" if off > current else "off_time_passed"
        except (ValueError, TypeError):
            row["race_state"] = "unknown"
        if state == "upcoming" and row["race_state"] != "scheduled":
            continue
        output.append(row)
    return sorted(output, key=lambda r: (r.get("scheduled_off_at") or "9999", r["race_id"]))

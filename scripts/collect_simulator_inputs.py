"""Collect every scheduled race independently of LLM analysis. No prediction runs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from pipeline.race_discovery import discover_races_for_date
from pipeline.race_meta_fetcher import build_shutuba_url, infer_location_from_race_id, parse_race_page
from pipeline.simulator_input_store import _write, _json, publish_race, sanitize_race, JST


def callback(payload):
    url, secret = os.environ.get("SIMULATOR_COLLECTION_CALLBACK_URL"), os.environ.get("PIPELINE_CALLBACK_SECRET")
    if not url:
        return
    if not secret:
        raise ValueError("callback secret required")
    body = json.dumps(payload, ensure_ascii=False).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    req = Request(url, data=body, headers={"Content-Type": "application/json", "X-Hub-Signature-256": signature}, method="POST")
    with urlopen(req, timeout=30) as response:
        response.read()


def collect(row, day, base_dir, force):
    from keiba_llm_agent.fetchers.netkeiba_fetcher import fetch_and_parse_netkeiba_race
    from keiba_llm_agent.fetchers.netkeiba_horse_fetcher import enrich_race_data_with_recent_runs
    race = fetch_and_parse_netkeiba_race(build_shutuba_url(row["race_id"]), force_refresh=force)
    if race.race_info.race_date != day:
        raise ValueError("collected date mismatch")
    race = enrich_race_data_with_recent_runs(race, force_refresh=force)
    raw = race.model_dump()
    from keiba_llm_agent.fetchers.netkeiba_fetcher import get_html_cache_path
    meta = parse_race_page(get_html_cache_path(row['race_id']).read_text(encoding='utf-8'), race_id=row['race_id'])
    start = meta.get('scheduled_off_time') or ''
    if 'T' in start:
        start = day + 'T' + start.split('T', 1)[1]
    raw['race_info']['scheduled_start_at'] = raw['race_info'].get('scheduled_start_at') or start or None
    raw["race_info"]["scheduled_start_at"] = raw["race_info"].get("scheduled_start_at") or row.get("scheduled_off_time")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    saved = publish_race(base_dir, raw, source="schedule_collector", source_as_of=stamp)
    callback({"race_data": raw, "source_as_of": stamp})
    return {"race_id": row["race_id"], "input_version": saved["input_version"], "availability": saved["availability"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(JST).date().isoformat())
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--import-saved", action="store_true", help="Read saved Agent inputs only; no network")
    args = parser.parse_args()
    base = ROOT / "pipeline"
    if args.import_saved:
        count = 0
        for path in (ROOT / "keiba_llm_agent/data/race_data").glob("*.json"):
            raw = _json(path) or {}
            if (raw.get("race_info") or {}).get("race_date") == args.date:
                try:
                    publish_race(base, raw, source="saved_agent_input", source_as_of=None)
                    count += 1
                except ValueError:
                    continue
        print(json.dumps({"imported": count, "race_date": args.date}))
        return
    discovery = discover_races_for_date(args.date)
    rows = [{**r, "race_date": args.date, "course": infer_location_from_race_id(r['race_id']), "scope_key": r.get('scope_key') or r.get('source')} for r in discovery["races"]]
    _write(base / "data/simulator_schedule" / f"{args.date}.json", rows)
    callback({"race_date": args.date, "schedule": rows})
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 3))) as pool:
        futures = {pool.submit(collect, row, args.date, base, args.force_refresh): row for row in rows}
        for future in as_completed(futures):
            try:
                print(json.dumps(future.result(), ensure_ascii=False))
            except Exception as error:
                print(json.dumps({"race_id": futures[future]["race_id"], "status": "preparing", "error_type": type(error).__name__}))


if __name__ == "__main__":
    main()

"""Bounded, read-only NAR acquisition check. Never creates prediction jobs."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from race_discovery import LOCAL_RACE_LIST_URL, parse_race_list_html
from race_meta_fetcher import LOCAL_VENUE_MAP, fetch_html
from keiba_llm_agent import main
from keiba_llm_agent.fetchers import netkeiba_fetcher, netkeiba_horse_fetcher
from keiba_llm_agent.parsers.netkeiba_horse_parser import parse_horse_recent_runs


def check(start_date):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    dates = []
    sample = None
    for offset in (0, 1):
        day = start + timedelta(days=offset)
        compact = day.strftime("%Y%m%d")
        url = LOCAL_RACE_LIST_URL + "?kaisai_date=" + compact
        html = fetch_html(url, timeout=12)
        races = [r for r in parse_race_list_html(html, base_url=url, source="local")
                 if len(r["race_id"]) == 12 and r["race_id"][4:6] in LOCAL_VENUE_MAP
                 and r["race_id"][6:10] == compact[4:]]
        dates.append({"date": day.isoformat(), "listed_races": len(races),
                      "venues": sorted({LOCAL_VENUE_MAP[r["race_id"][4:6]] for r in races})})
        if races and sample is None:
            sample = races[0]
        time.sleep(1)
    if sample is None:
        return {"status": "not_listed", "dates": dates, "purpose": "acquisition_only"}
    with tempfile.TemporaryDirectory(prefix="local-acquisition-") as temp:
        with (patch.object(netkeiba_fetcher, "HTML_CACHE_DIR", Path(temp) / "cards"),
              patch.object(netkeiba_horse_fetcher, "HORSE_HTML_CACHE_DIR", Path(temp) / "horses"),
              patch.object(main, "capture_public_workout", side_effect=AssertionError("local workout request")) as capture):
            race = netkeiba_fetcher.fetch_and_parse_netkeiba_race(sample["shutuba_url"], force_refresh=True)
            assert race.race_info.source == "local" and race.race_info.scope_key == "local"
            assert race.horses and race.race_info.race_date
            assert main._capture_prediction_supplements(race) is None
            capture.assert_not_called()
            horse = next(h for h in race.horses if h.horse_id)
            time.sleep(1)
            html = netkeiba_horse_fetcher.fetch_horse_html(horse.horse_id, force_refresh=True)
            parsed = parse_horse_recent_runs(html)
            prior = netkeiba_horse_fetcher.filter_recent_runs_for_target_race(
                parsed, race.race_info.race_date, race.race_info.race_id, horse.horse_id, 5)
            assert prior, "Sample horse prior runs unavailable"
            return {"status": "passed", "dates": dates, "purpose": "acquisition_only",
                    "sample": {"race_id": race.race_info.race_id, "race_date": race.race_info.race_date,
                               "course": race.race_info.course, "source": race.race_info.source,
                               "horses": len(race.horses), "horse_ids": sum(bool(h.horse_id) for h in race.horses),
                               "sample_horse_id": horse.horse_id, "prior_runs": len(prior)},
                    "workout_requests": 0, "prediction_jobs_created": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(timezone(timedelta(hours=9))).date().isoformat())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.date)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "passed" else 2)

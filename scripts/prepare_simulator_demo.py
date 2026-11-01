"""Build a small, market-free simulator fixture from saved Agent LLM inputs.

Reads source files only. Never updates predictions, results, or audit snapshots.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ("202604010201", "202604010202", "202604010203")
RUN_FIELDS = (
    "date", "course", "surface", "distance", "track_condition", "finish",
    "field_size", "corner_positions", "final_3f", "time_seconds",
)


def build_race(race_id: str, race_dir: Path, prediction_dir: Path) -> dict:
    race = json.loads((race_dir / f"{race_id}.json").read_text(encoding="utf-8-sig"))
    prediction = json.loads((prediction_dir / f"{race_id}.json").read_text(encoding="utf-8-sig"))
    race_date = date.fromisoformat(race["race_info"]["race_date"])
    pace_by_no = {h["horse_no"]: h for h in prediction.get("pace_analyses", [])}
    horses = []
    excluded = 0
    for horse in race["horses"]:
        runs = []
        for run in horse.get("recent_runs", []):
            try:
                before_race = date.fromisoformat(run.get("date", "")) < race_date
            except (TypeError, ValueError):
                before_race = False
            if not before_race or run.get("race_id") == race_id:
                excluded += 1
                continue
            runs.append({k: run.get(k) for k in RUN_FIELDS})
        pace = pace_by_no.get(horse["horse_no"], {})
        horses.append({
            "horse_no": horse["horse_no"], "frame_no": horse.get("frame_no"),
            "horse_name": horse["horse_name"], "jockey": horse.get("jockey"),
            "carried_weight": horse.get("carried_weight"), "recent_runs": runs,
            "running_style": pace.get("running_style", "不明"),
            "position_stability": pace.get("position_stability", "不明"),
        })
    return {
        "race_info": race["race_info"], "horses": horses,
        "provenance": {
            "source": "saved_agent_inputs", "race_data": f"keiba_llm_agent/data/race_data/{race_id}.json",
            "pace_analysis": f"pipeline/data/agent_predictions/{race_id}.json",
            "excluded_runs": excluded, "uses_results": False, "uses_market": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-ids", nargs="+", default=list(DEFAULT_IDS))
    parser.add_argument("--race-dir", type=Path, default=ROOT / "keiba_llm_agent/data/race_data")
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "pipeline/data/agent_predictions")
    parser.add_argument("--output", type=Path, default=ROOT / "frontend/src/fixtures/raceSimulatorDemo.json")
    args = parser.parse_args()
    payload = {"schema_version": 1, "races": [build_race(i, args.race_dir, args.prediction_dir) for i in args.race_ids]}
    protected = [ROOT / p for p in ("pipeline/data", "keiba_llm_agent/data", "analysis_outputs")]
    output = args.output.resolve()
    if any(output.is_relative_to(p.resolve()) for p in protected):
        parser.error("Output must not be inside a source or audit data directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(payload['races'])} races: {output}")


if __name__ == "__main__":
    main()

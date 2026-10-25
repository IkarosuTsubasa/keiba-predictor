"""Read-only, one-race-one-vote report of frozen prospective comparisons."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keiba_llm_agent.backtest.input_selection import select_effective_backtest_audit_paths
from pipeline.workout_shadow_metrics import compare_workout_shadow


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def summarize(pairs):
    output = {"races": len(pairs)}
    for key in ("baseline", "candidate"):
        output[key] = {name: sum(p[key][name] for p in pairs) for name in (
            "covered_slots", "all3", "atleast2", "winner_top3", "winner_top5", "winner_rank")}
        output[key]["recall_pct"] = 100 * output[key]["covered_slots"] / (3 * len(pairs)) if pairs else None
        winner_rank_sum = output[key].pop("winner_rank")
        output[key]["winner_mean_rank"] = winner_rank_sum / len(pairs) if pairs else None
    output["delta_covered_slots"] = output["candidate"]["covered_slots"] - output["baseline"]["covered_slots"]
    return output


def report(snapshot):
    excluded = Counter()
    pairs = []
    paths = select_effective_backtest_audit_paths(snapshot / "agent_backtest_audit")
    for rid, path in paths.items():
        try:
            audit = read(path)
            result_path = snapshot / "agent_results" / (rid + ".json")
            result = read(result_path) if result_path.exists() else audit.get("outcomes", {}).get("result")
            pair = compare_workout_shadow(audit, result)
            if pair and pair["status"] == "settled":
                pairs.append(pair)
            else:
                excluded[pair.get("reason_code", pair["status"]) if pair else "shadow_missing"] += 1
        except (ValueError, KeyError, OSError, TypeError):
            excluded["invalid_audit"] += 1
    if len({(p["model_sha256"], p["baseline_model_sha256"]) for p in pairs}) > 1:
        raise ValueError("Mixed model contracts; report separately")
    return {"selected_audits": len(paths), "excluded_or_pending": dict(excluded), "overall": summarize(pairs),
            "strata": {**{key: summarize([p for p in pairs if p["strata"].get(key)]) for key in (
                "empty_history", "graded", "field_13plus", "field_16plus")},
                "with_history": summarize([p for p in pairs if not p["strata"].get("empty_history")])},
            "pairs": pairs, "note": "Prospective descriptive comparison; small groups are not significance evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.snapshot.resolve())
    if args.output:
        path = args.output.resolve()
        if not path.is_relative_to(ROOT / "analysis_outputs"):
            parser.error("Use analysis_outputs for derived reports")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, ensure_ascii=False, indent=2))

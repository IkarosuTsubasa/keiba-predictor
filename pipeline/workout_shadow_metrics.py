"""Pure settlement checks for the prospective public-workout comparison."""
from datetime import datetime

MODEL_VERSION = "central_turf_workout_shadow_v1_20261026"


def compare_workout_shadow(audit, result):
    shadow = audit.get("shadow_predictions", {}).get("public_workout")
    if shadow is None:
        return None
    output = {"schema_version": "public_workout_comparison_v1", "status": "excluded",
              "race_id": audit.get("race_id"), "model_version": shadow.get("model_version")}
    try:
        if (shadow.get("model_version") != MODEL_VERSION or shadow.get("status") != "predicted"
                or shadow.get("authoritative") is not False or audit.get("audit_kind") != "full"):
            output["reason_code"] = "shadow_not_predicted"
            return output
        info = audit["inputs"]["race_data"]["race_info"]
        prediction = audit["outputs"]["prediction"]
        if (info["surface"] != "芝" or info.get("source") == "local"
                or str(info.get("scope_key", "")).startswith("local")
                or shadow["race_id"] != output["race_id"]
                or prediction["race_id"] != output["race_id"]
                or info["race_date"] <= shadow["training_end"]
                or not shadow["timing"]["eligible_at_generation"]):
            output["reason_code"] = "not_prospective"
            return output
        times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in (
            shadow["timing"]["scheduled_start_at"], shadow["generated_at"],
            shadow["workout_received_at"], prediction["prediction_received_at"])]
        start, generated, captured, received = times
        if any(t.tzinfo is None for t in times) or not captured <= generated < start or received >= start:
            output["reason_code"] = "not_received_before_start"
            return output
        if not result:
            output.update(status="pending", reason_code="unsettled")
            return output
        if result.get("race_id") != output["race_id"]:
            raise ValueError("Result identity mismatch")
        top3 = [result["result"][k] for k in ("1st", "2nd", "3rd")]
        if len(set(top3)) != 3 or any(type(n) is not int for n in top3):
            raise ValueError("Invalid podium")
        baseline = shadow["baseline_ranking"]
        candidate = shadow["candidate_ranking"]
        roster = shadow["horse_numbers"]
        if (not roster or len(set(roster)) != len(roster) or sorted(baseline) != sorted(roster)
                or sorted(candidate) != sorted(roster) or not set(top3).issubset(roster)
                or baseline != prediction["scope_reranker_result"]["reranked_ranking"]
                or baseline[:5] != list(prediction["marks"].values())):
            raise ValueError("Ranking mismatch")
        output.update(status="settled", date=info["race_date"], model_sha256=shadow["model_sha256"],
                      baseline_model_sha256=shadow["baseline_model_sha256"], strata=shadow["strata"], top3=top3)
        for key, ranking in (("baseline", baseline), ("candidate", candidate)):
            hits = len(set(ranking[:5]) & set(top3))
            winner = ranking.index(top3[0]) + 1
            output[key] = {"covered_slots": hits, "all3": int(hits == 3), "atleast2": int(hits >= 2),
                           "winner_top3": int(winner <= 3), "winner_top5": int(winner <= 5), "winner_rank": winner}
        output["delta_covered_slots"] = output["candidate"]["covered_slots"] - output["baseline"]["covered_slots"]
    except (KeyError, ValueError, TypeError, AttributeError):
        output.update(status="excluded", reason_code="invalid_evidence_or_timing")
    return output


def attach_workout_comparison(audit, result):
    comparison = compare_workout_shadow(audit, result)
    if comparison is not None:
        audit.setdefault("outcomes", {})["public_workout_comparison"] = comparison

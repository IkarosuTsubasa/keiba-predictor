"""Fit the already evaluated candidate for prospective testing, never activation."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from keiba_llm_agent.scoring import workout_shadow as shadow, turf_place_model as active

TRIAL = ROOT / "analysis_outputs/netkeiba_feature_trial_20261026_v1"
PREPARED = ROOT / "analysis_outputs/turf_correction_recheck_20261025_v1/prepared.joblib"
ENRICH = ROOT / "analysis_outputs/turf_enrichment_20261025_v1/supplement_manifest.json"


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    manifest = read(TRIAL / "workout_evaluation/manifest.json")
    for name, expected in {**manifest["input_hashes"], **manifest["protected_hashes"]}.items():
        assert sha(Path(name)) == expected, name
    template, baseline = active.load_bundle()
    assert baseline["prepared_sha256"] == sha(PREPARED)
    rows = joblib.load(PREPARED)["rows"]
    enrich = {r["race_id"]: r for r in read(ENRICH)["races"]}
    history = {}
    matrices = {}
    by_day = defaultdict(list)
    for row in rows:
        by_day[row["date"]].append(row)
    for day in sorted(by_day):
        updates = {}
        for row in by_day[day]:
            rid = row["race_id"]
            source = Path(enrich[rid]["enriched"])
            assert sha(source) == enrich[rid]["enriched_sha256"]
            horses = read(source)["horses"]
            assert [h["horse_no"] for h in horses] == row["numbers"]
            ids = [h["horse_id"] for h in horses]
            grades = read(TRIAL / "workout_pages" / (rid + ".json"))["horses"]
            extra = shadow.workout_matrix(row["numbers"], ids, day, grades, history)
            matrices[rid] = np.column_stack((row["matrices"]["class_context"], extra))
            assert matrices[rid].shape == (len(row["numbers"]), 324)
            for number, horse_id in zip(row["numbers"], ids, strict=True):
                grade = grades.get(str(number), {}).get("grade")
                if horse_id and grade in shadow.GRADE_VALUES:
                    updates[horse_id] = {"date": day, "grade": grade}
        history.update(updates)
    return rows, matrices, history, template, baseline


def release():
    rows, matrices, history, template, baseline = inputs()
    x = np.concatenate([matrices[r["race_id"]] for r in rows])
    y = np.asarray([int(n in r["top3"]) for r in rows for n in r["numbers"]])
    scale = sum((len(r["numbers"])-3) + max(len(r["numbers"])-3, 1) for r in rows) / len(y)
    assert len(rows) == 630 and x.shape == (8324, 324)
    model = clone(template)
    with threadpool_limits(limits=2):
        model.fit(x, y, sample_weight=np.full(len(y), scale))
    destination = shadow.MODEL_DIRECTORY / (shadow.MODEL_VERSION + ".joblib")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        joblib.dump({"model": model, "history": history}, handle, compress=3)
    meta = {
        "model_version": shadow.MODEL_VERSION, "mode": "shadow_only", "authoritative": False,
        "market_features_used": False, "feature_names": list(shadow.FEATURE_NAMES),
        "feature_schema_sha256": active.json_hash(list(shadow.FEATURE_NAMES)),
        "source_hashes": shadow.source_hashes(), "sklearn_version": sklearn.__version__,
        "model_sha256": sha(destination), "baseline_model_version": active.MODEL_VERSION,
        "baseline_model_sha256": baseline["model_sha256"], "training_races": len(rows),
        "training_runner_observations": len(y), "training_start": min(r["date"] for r in rows),
        "training_end": max(r["date"] for r in rows), "created_at": datetime.now(timezone.utc).isoformat(),
        "estimator_parameters": model.get_params(), "frozen_candidate": "workout",
        "training_target": "actual_top3_equal_runner_weight", "ranking_policy": "direct_top5",
        "history_policy": "frozen_training_cohort", "prior_grade_horses": len(history),
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (
            PREPARED, ENRICH, TRIAL / "workout_evaluation/results.json", TRIAL / "workout_evaluation/manifest.json")},
        "validation": "Existing 630-race repeated OOF and 522-race chronological evidence; this final fit is not an accuracy test.",
        "source_timing": "Historical training grades were retrospectively fetched; forward comparison requires frozen pre-off evidence.",
    }
    with destination.with_suffix(".json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    shadow.load_bundle()
    print(json.dumps({k: meta[k] for k in ("model_version", "model_sha256", "training_races", "training_runner_observations", "training_end")}))


if __name__ == "__main__":
    release()

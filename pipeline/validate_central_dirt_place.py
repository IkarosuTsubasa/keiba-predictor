from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import experiment_agent_place_models as exp

MIN_FIELD_SIZE = 13
MODEL_VERSION = "central_dirt_place_context_v1_20260908"
FEATURE_SET = "place_context_v1"


def fit_place_model(rows):
    x = np.concatenate([r["fresh_features"]["context"] for r in rows])
    y = np.concatenate([
        np.asarray([int(n in r["top3"]) for n in r["numbers"]]) for r in rows
    ])
    # Keep the reference loss scale for L2, but give every runner equal weight.
    total = sum((len(r["numbers"]) - 3) + max(len(r["numbers"]) - 3, 1) for r in rows)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=180, max_leaf_nodes=7,
        min_samples_leaf=40, l2_regularization=2.0, random_state=42,
    )
    return model.fit(x, y, sample_weight=np.full(len(y), total / len(y)))


def evaluate(model, rows):
    probabilities = model.predict_proba(np.concatenate([
        r["fresh_features"]["context"] for r in rows
    ]))[:, 1]
    baseline, candidate, records = [], [], []
    offset = 0
    for row in rows:
        numbers = row["numbers"]
        values = probabilities[offset:offset + len(numbers)]
        offset += len(numbers)
        old = row["fresh_raw"]
        new = sorted(numbers, key=lambda n: (-float(values[numbers.index(n)]), n))
        active = len(numbers) >= MIN_FIELD_SIZE
        proposed = new if active else old
        baseline.append(exp.metric_vector(row, old))
        candidate.append(exp.metric_vector(row, proposed))
        records.append({
            "race_id": row["race_id"], "date": row["date"], "numbers": numbers,
            "top3": row["top3"], "winner": row["winner"], "active": active,
            "old_ranking": old, "candidate_ranking": proposed,
            "model_scores": list(map(float, values)),
        })
    return np.asarray(baseline), np.asarray(candidate), records


def validate(rows, output):
    old, new, records = [], [], []
    for seed in exp.SEEDS:
        bm, cm = np.zeros((len(rows), 8)), np.zeros((len(rows), 8))
        seen = np.zeros(len(rows), dtype=int)
        for fold, (train, test) in enumerate(exp.repeated_splits(rows, seed)):
            training, testing = [rows[i] for i in train], [rows[i] for i in test]
            model = fit_place_model(training)
            b, c, rr = evaluate(model, testing)
            bm[test], cm[test] = b, c
            seen[test] += 1
            records.extend({"seed": seed, "fold": fold, **r} for r in rr)
        assert (seen == 1).all()
        old.append(bm)
        new.append(cm)
        print(f"seed={seed}: additional slots={sum(cm[:, 0] - bm[:, 0])}", flush=True)
    old, new = np.asarray(old), np.asarray(new)
    small = np.asarray([len(r["numbers"]) < MIN_FIELD_SIZE for r in rows])
    np.testing.assert_array_equal(old[:, small], new[:, small])
    result = exp.compare(rows, old, new)
    result["every_race_oof_per_repeat"] = True
    result["small_fields_exactly_unchanged"] = True
    np.savez_compressed(output / "oof_metrics.npz", old=old, candidate=new)
    (output / "oof_races.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    future, bm, cm, periods, time_records = [], [], [], [], []
    for block in np.array_split(sorted({r["date"] for r in rows}), 5)[1:]:
        train = [r for r in rows if r["date"] < block[0]]
        test = [r for r in rows if r["date"] in block]
        b, c, rr = evaluate(fit_place_model(train), test)
        future.extend(test)
        bm.extend(b)
        cm.extend(c)
        time_records.extend(rr)
        periods.append({"train_through": max(r["date"] for r in train),
                        "test_from": str(block[0]), "test_through": str(block[-1]),
                        "training_races": len(train), "test_races": len(test)})
    result["chronological_diagnostic"] = {
        "blocks": periods,
        "comparison": exp.compare(future, np.asarray([bm]), np.asarray([cm])),
    }
    (output / "chronological_races.json").write_text(json.dumps(time_records, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "analysis_outputs"):
        parser.error("output must be a new derived directory under analysis_outputs")
    output.mkdir(parents=True, exist_ok=False)
    cache = joblib.load(args.prepared)
    rows = [r for r in cache["rows"] if r["scope"] == "central_dirt"]
    if cache["preparation"]["errors"]:
        raise ValueError("full recomputation contains errors")
    with threadpool_limits(limits=2):
        result = validate(rows, output)
        model = fit_place_model(rows)
    joblib.dump(model, output / f"{MODEL_VERSION}.joblib", compress=3)
    metadata = {
        "model_version": MODEL_VERSION, "feature_set": FEATURE_SET,
        "training_target": "actual_top3_equal_runner_weight",
        "min_field_size": MIN_FIELD_SIZE, "blend_weight": 1.0, "preserve_top_n": 0,
        "training_races": len(rows), "training_runners": sum(len(r["numbers"]) for r in rows),
        "training_start": min(r["date"] for r in rows), "training_end": max(r["date"] for r in rows),
        "training_race_ids": [r["race_id"] for r in rows],
        "n_features": int(model.n_features_in_), "snapshot": cache["snapshot"],
        "scikit_learn_version": sklearn.__version__,
        "source_code_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "pipeline/experiment_agent_place_models.py", "pipeline/validate_central_dirt_place.py",
                "pipeline/train_scope_reranker.py", "keiba_llm_agent/scoring/place_context.py",
                "keiba_llm_agent/scoring/recent_run_scorer.py", "keiba_llm_agent/scoring/scope_reranker.py",
            )
        },
        "prepared_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((output / f"{MODEL_VERSION}.joblib").read_bytes()).hexdigest(),
        "market_features_used": False,
        "frozen_inputs": ["pedigree_analyses", "used_lessons"],
        "refreshed_inputs": ["deep_analyses", "race_level_analyses", "pace_analyses"],
        "selection_caveat": "Fixed exploratory gate; repeated CV is not an untouched forward test.",
        "preparation_counts": cache["preparation"]["counts"], "validation": result,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "comparison": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()

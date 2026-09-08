from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

from pipeline import experiment_agent_place_models as exp
from pipeline.validate_central_dirt_place import fit_place_model
from keiba_llm_agent.backtest.audit_snapshot import build_backtest_audit_payload
from keiba_llm_agent.schemas.prediction import Prediction, ScopeRerankerResult
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.scoring import recent_run_scorer as scorer
from keiba_llm_agent.scoring import scope_reranker as runtime
from web_data.agent_predictions import _custom_prediction_race_item


def fingerprints(snapshot):
    return {
        str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("agent_predictions", "agent_results", "agent_backtest_audit")
        for path in sorted((snapshot / directory).rglob("*")) if path.is_file()
    }


def verify(args):
    cache = joblib.load(args.prepared)
    snapshot = Path(cache["snapshot"])
    manifest = fingerprints(snapshot)
    rows = {r["race_id"]: r for r in cache["rows"]}
    baseline_source = subprocess.check_output([
        "git", "show", f"{args.baseline_ref}:keiba_llm_agent/scoring/scope_reranker.py",
    ], cwd=ROOT)
    old = types.ModuleType("baseline_scope_reranker")
    old.__file__ = str(ROOT / "keiba_llm_agent/scoring/scope_reranker.py")
    exec(compile(baseline_source, old.__file__, "exec"), old.__dict__)
    if "central_dirt" in old.MODEL_SPECS:
        raise ValueError("baseline ref must precede the central dirt candidate")
    metadata = json.loads((args.validation / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["snapshot"] == str(snapshot)
    assert metadata["prepared_sha256"] == hashlib.sha256(args.prepared.read_bytes()).hexdigest()
    model_path = runtime._model_directory() / runtime.MODEL_SPECS["central_dirt"]["filename"]
    assert metadata["model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    expected_records = {(r["seed"], r["race_id"]): r for r in
                        json.loads((args.validation / "oof_races.json").read_text(encoding="utf-8"))}
    counts, dirt_inputs = Counter(), {}
    with threadpool_limits(limits=2), patch.dict(os.environ, {"KEIBA_CENTRAL_DIRT_PLACE_MODE": "active"}):
        for payload in exp.preferred_audits(snapshot).values():
            if payload.get("audit_kind") != "full" or exp.winner_from_audit(payload) is None:
                continue
            race, removed = exp.sanitize_race(RaceData.model_validate(payload["inputs"]["race_data"]))
            assert removed == 0
            saved = Prediction.model_validate(payload["outputs"]["prediction"])
            scope = exp.scope_key(race)
            refreshed = exp.refreshed_prediction(race, saved)
            scores = exp.recompute_horse_scores(race, refreshed, exp.scoring_config_for_scope(scope))
            immutable = [s.model_dump() for s in scores]
            previous, _, _ = old.apply_scope_reranker(race, scores, enabled=True)
            current, config, result = runtime.apply_scope_reranker(race, scores, enabled=True)
            assert immutable == [s.model_dump() for s in scores]
            before = {"race_info": race.race_info.model_dump(), "horse_scores": [s.model_dump() for s in previous]}
            after = {**before, "horse_scores": [s.model_dump() for s in current]}
            assert _custom_prediction_race_item(before) == _custom_prediction_race_item(after)
            counts["my_payload_identical"] += 1
            if scope != "central_dirt" or len(race.horses) < 13:
                assert previous == current
                counts[scope + "_unchanged"] += 1
            if scope == "central_dirt":
                row = rows[race.race_info.race_id]
                matrix, numbers = runtime.build_scope_reranker_feature_matrix(race, scores, "place_context_v1")
                assert numbers == row["numbers"]
                np.testing.assert_allclose(matrix, row["fresh_features"]["context"], equal_nan=True, atol=0, rtol=0)
                assert [s.horse_no for s in scores] == row["fresh_raw"]
                dirt_inputs[row["race_id"]] = (race, scores)
                counts["dirt_exact_training_runtime_features"] += 1
                if len(numbers) >= 13:
                    assert result.applied and result.candidate_ranking == result.reranked_ranking
                    model = runtime._load_dirt_model(str(model_path))
                    p = model.predict_proba(matrix)[:, 1]
                    expected = sorted(numbers, key=lambda n: (-float(p[numbers.index(n)]), n))
                    assert [s.horse_no for s in current] == expected
                    with patch.object(scorer, "build_pedigree_analyses_for_race", return_value=saved.pedigree_analyses):
                        prediction = scorer.build_prediction_from_recent_runs_with_scoring_config(
                            race, saved.used_lessons, scoring_profile=saved.scoring_profile,
                            scoring_config=exp.scoring_config_for_scope(scope), scope_reranker_enabled=True,
                        )
                    assert [s.horse_no for s in prediction.horse_scores] == expected
                    assert list(prediction.marks.values()) == expected[:5]
                    audit = build_backtest_audit_payload(race_data=race, prediction=prediction)
                    restored = ScopeRerankerResult.model_validate_json(json.dumps(audit["analyses"]["scope_reranker_result"]))
                    assert restored == prediction.scope_reranker_result
                    counts["dirt_full_prediction_and_audit_parity"] += 1
            counts["full_races_verified"] += 1
            if counts["full_races_verified"] % 400 == 0:
                print(dict(counts), flush=True)

        dirt_rows = [r for r in cache["rows"] if r["scope"] == "central_dirt"]
        for seed in exp.SEEDS:
            for fold, (train, test) in enumerate(exp.repeated_splits(dirt_rows, seed)):
                model = fit_place_model([dirt_rows[i] for i in train])
                training_dates = {dirt_rows[i]["date"] for i in train}
                with patch.object(runtime, "_load_dirt_model", return_value=model):
                    for i in test:
                        row = dirt_rows[i]
                        assert row["date"] not in training_dates
                        race, scores = dirt_inputs[row["race_id"]]
                        ranked, _, _ = runtime.apply_scope_reranker(race, scores, enabled=True)
                        expected = expected_records[(seed, row["race_id"])]
                        assert expected["fold"] == fold
                        assert [s.horse_no for s in ranked] == expected["candidate_ranking"]
                        counts["runtime_oof_ranking_parity"] += 1
            print(f"runtime OOF seed {seed} verified", flush=True)
    assert counts["runtime_oof_ranking_parity"] == 5 * len(dirt_rows)
    independent_old, independent_new = [], []
    for seed in exp.SEEDS:
        old_slots, new_slots, new_all3, new_atleast2 = 0, 0, 0, 0
        for row in dirt_rows:
            record = expected_records[(seed, row["race_id"])]
            assert record["top3"] == row["top3"]
            assert record["old_ranking"] == row["fresh_raw"]
            old_slots += sum(n in row["top3"] for n in record["old_ranking"][:5])
            slots = sum(n in row["top3"] for n in record["candidate_ranking"][:5])
            new_slots += slots
            new_all3 += slots == 3
            new_atleast2 += slots >= 2
        independent_old.append(old_slots)
        independent_new.append((new_slots, new_all3, new_atleast2))
    expected_summary = metadata["validation"]["candidate_mean"]
    np.testing.assert_allclose(np.mean(independent_new, axis=0),
                               [expected_summary[k] for k in ("slots", "all3", "atleast2")])
    assert metadata["validation"]["current_mean"]["slots"] == np.mean(independent_old)
    np.testing.assert_allclose(expected_summary["recall_pct"],
                               np.mean(independent_new, axis=0)[0] / (3 * len(dirt_rows)) * 100, atol=1e-6)
    assert manifest == fingerprints(snapshot)
    assert metadata["training_race_ids"] == [r["race_id"] for r in dirt_rows]
    report = {
        "snapshot": str(snapshot), "validation": str(args.validation), "counts": dict(counts),
        "baseline_ref": subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip(),
        "baseline_runtime_sha256": hashlib.sha256(baseline_source).hexdigest(),
        "all_checks_passed": True, "source_files_unchanged": len(manifest),
        "independent_primary_metric_check": True,
        "static_replay_used_only_for_integration_not_accuracy": True,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="HEAD")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "analysis_outputs"):
        parser.error("output must be under analysis_outputs")
    verify(args)


if __name__ == "__main__":
    main()

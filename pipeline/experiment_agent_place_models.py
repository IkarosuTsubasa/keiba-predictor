from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from keiba_llm_agent.analysis.race_deep_analyzer import analyze_race_deeply
from keiba_llm_agent.analysis.race_level_analyzer import analyze_race_level_for_race
from keiba_llm_agent.analysis.pace_analyzer import analyze_pace_for_race
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.scoring import scope_reranker as reranker
from pipeline.train_scope_reranker import (
    preferred_audits, recompute_horse_scores, scope_key,
    scoring_config_for_scope, top3_from_audit, winner_from_audit,
)

SCOPES = ("central_turf", "central_dirt", "local")
SEEDS = (7, 19, 42, 73, 101)
CANDIDATES = (
    "refresh_analysis", "missing_context", "rank_native",
    "rank_context", "large_field_rank", "linear_components",
)
POLICIES = {
    "central_turf": ("base", "course_geometry", 1, 0.5, 1.0),
    "central_dirt": ("base", "course_geometry", 1, 0.5, 1.0),
    "local": ("extended", "local_course_geometry", 2, 1.0, 2.0),
}
METRICS = (
    "slots", "all3", "atleast2", "top1_win", "top1_place",
    "winner_top3", "winner_top5", "winner_rank",
)


def parsed_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value or "")
    except ValueError:
        return None


def sanitize_race(race: RaceData) -> tuple[RaceData, int]:
    clean = race.model_copy(deep=True)
    target_date = parsed_date(clean.race_info.race_date)
    removed = 0
    for horse in clean.horses:
        horse.odds = None
        horse.popularity = None
        prior = []
        for run in horse.recent_runs:
            run_date = parsed_date(run.date)
            if run.race_id == clean.race_info.race_id or (
                target_date and run_date and run_date >= target_date
            ):
                removed += 1
                continue
            run.odds = None
            run.popularity = None
            prior.append(run)
        horse.recent_runs = prior
    return clean, removed


def refreshed_prediction(race: RaceData, saved: Prediction) -> Prediction:
    # Pedigree profiles and lessons remain frozen at the captured prediction.
    pace, projection = analyze_pace_for_race(race.horses, race.race_info)
    return saved.model_copy(update={
        "deep_analyses": analyze_race_deeply(race.horses, race.race_info),
        "race_level_analyses": analyze_race_level_for_race(race.horses, race.race_info),
        "pace_analyses": pace,
        "race_pace_projection": projection,
    })


def context_matrix(race: RaceData, scores: list[Any], native: np.ndarray) -> np.ndarray:
    rows = []
    race_date = parsed_date(race.race_info.race_date)
    score_map = {s.horse_no: s for s in scores}
    for horse in race.horses:
        runs = [r for r in horse.recent_runs[:8]
                if r.finish is not None and r.field_size is not None and r.field_size > 1][:5]
        features = reranker.build_run_features(horse, race)
        if not runs:
            features = {name: np.nan for name in reranker.RUN_FEATURE_NAMES}
            features["run_count"] = 0.0
        last = runs[0] if runs else None
        last_date = parsed_date(last.date) if last else None
        rest_missing = not (race_date and last_date)
        distance_missing = not (last and last.distance and race.race_info.distance)
        margin_missing = not last or last.margin is None
        corners_missing = not last or not last.corner_positions
        final3f_missing = sum(r.final_3f is not None for r in runs) < 2
        if rest_missing:
            features["rest_days_preference"] = np.nan
        if distance_missing:
            features["distance_change_penalty"] = np.nan
        if margin_missing:
            features["last_margin_closeness"] = np.nan
        if corners_missing:
            features["last_position_gain"] = np.nan
            features["last_early_position"] = np.nan
        if final3f_missing:
            features["last_final3f_vs_history"] = np.nan
        for name, predicate in (
            ("same_distance_best", lambda r: r.distance == race.race_info.distance and r.distance is not None),
            ("same_course_best", lambda r: r.course == race.race_info.course and bool(r.course)),
            ("same_surface_best", lambda r: r.surface == race.race_info.surface and bool(r.surface)),
            ("large_field_best", lambda r: r.field_size >= 13),
            ("same_jockey_best", lambda r: r.jockey == horse.jockey and bool(r.jockey)),
        ):
            if not any(predicate(r) for r in runs):
                features[name] = np.nan
        absolute = [
            float(len(race.horses)), float(len(race.horses) >= 13),
            float(race.race_info.distance or np.nan),
            float(len(horse.recent_runs)),
            float((race_date - last_date).days) if not rest_missing else np.nan,
            float(horse.frame_no) if horse.frame_no else np.nan,
            float(horse.carried_weight) if horse.carried_weight else np.nan,
            float(score_map[horse.horse_no].scores.ability_score),
            float(score_map[horse.horse_no].scores.recent_quality_score),
        ]
        missing = [float(not runs), float(rest_missing), float(distance_missing),
                   float(margin_missing), float(corners_missing), float(final3f_missing)]
        rows.append([features[n] for n in reranker.RUN_FEATURE_NAMES] + absolute + missing)
    return np.concatenate((native, np.asarray(rows, dtype=float)), axis=1)


def feature_matrices(race: RaceData, scores: list[Any], scope: str) -> dict[str, np.ndarray]:
    winner_name, place_name, *_ = POLICIES[scope]
    winner, numbers = reranker.build_scope_reranker_feature_matrix(race, scores, winner_name)
    native, place_numbers = reranker.build_scope_reranker_feature_matrix(race, scores, place_name)
    if numbers != place_numbers:
        raise ValueError("feature horse order mismatch")
    return {"winner": winner, "native": native,
            "context": context_matrix(race, scores, native), "components": native[:, :13]}


def metric_vector(row: dict[str, Any], ranking: list[int]) -> np.ndarray:
    if len(ranking) != len(set(ranking)) or set(ranking) != set(row["numbers"]):
        raise ValueError("ranking is not a permutation of the runners")
    slots = len(set(ranking[:5]) & set(row["top3"]))
    winner_rank = ranking.index(row["winner"]) + 1
    return np.asarray([slots, slots == 3, slots >= 2, ranking[0] == row["winner"],
                       ranking[0] in row["top3"], winner_rank <= 3, winner_rank <= 5,
                       winner_rank], dtype=float)


def summarize_vectors(vectors: np.ndarray) -> dict[str, float]:
    n = len(vectors)
    total = vectors.sum(axis=0)
    output = dict(zip(METRICS, map(float, total), strict=True))
    output["races"] = n
    output["recall_pct"] = 100.0 * total[0] / (3 * n) if n else 0.0
    output["mean_winner_rank"] = total[-1] / n if n else 0.0
    return {k: round(v, 6) for k, v in output.items()}


def metric_delta(current: dict, candidate: dict) -> dict:
    return {k: round(candidate[k] - current[k], 6) for k in current if k != "races"}


def prepare(snapshot: Path) -> tuple[list[dict], dict]:
    rows, errors = [], []
    counts = Counter()
    static = {scope: {key: [] for key in ("saved", "current", "refreshed")}
              for scope in (*SCOPES, "central_other")}
    for payload in preferred_audits(snapshot).values():
        kind = payload.get("audit_kind", "unknown")
        settled = winner_from_audit(payload) is not None
        counts[f"{kind}_{'settled' if settled else 'unsettled'}"] += 1
        if kind != "full" or not settled:
            continue
        race_payload = (payload.get("inputs") or {}).get("race_data")
        saved_payload = (payload.get("outputs") or {}).get("prediction")
        try:
            race, removed = sanitize_race(RaceData.model_validate(race_payload))
            counts["removed_nonhistorical_runs"] += removed
            saved = Prediction.model_validate(saved_payload)
            scope = scope_key(race)
            numbers = [h.horse_no for h in race.horses]
            top3 = top3_from_audit(payload)
            if len(top3) != 3 or not top3 <= set(numbers):
                raise ValueError("incomplete or nonmatching podium")
            if len(numbers) != len(set(numbers)):
                raise ValueError("duplicate horse number")
            if parsed_date(race.race_info.race_date) is None:
                raise ValueError("missing race date")
            config = scoring_config_for_scope(scope)
            current_scores = recompute_horse_scores(race, saved, config)
            refreshed = refreshed_prediction(race, saved)
            fresh_scores = recompute_horse_scores(race, refreshed, config)
            row = {"race_id": race.race_info.race_id, "date": race.race_info.race_date,
                   "scope": scope, "numbers": numbers, "winner": winner_from_audit(payload),
                   "top3": sorted(top3), "raw": [s.horse_no for s in current_scores],
                   "fresh_raw": [s.horse_no for s in fresh_scores]}
            current_order = [s.horse_no for s in reranker.apply_scope_reranker(race, current_scores, enabled=True)[0]]
            fresh_order = [s.horse_no for s in reranker.apply_scope_reranker(race, fresh_scores, enabled=True)[0]]
            saved_order = [s.horse_no for s in saved.horse_scores]
            static[scope]["current"].append(metric_vector(row, current_order))
            static[scope]["refreshed"].append(metric_vector(row, fresh_order))
            if set(saved_order) == set(numbers):
                static[scope]["saved"].append(metric_vector(row, saved_order))
            counts[f"{scope}_full"] += 1
            if scope not in SCOPES:
                continue
            row["features"] = feature_matrices(race, current_scores, scope)
            row["fresh_features"] = feature_matrices(race, fresh_scores, scope)
            rows.append(row)
            if len(rows) % 300 == 0:
                print(f"prepared {len(rows)} full races", flush=True)
        except Exception as exc:
            errors.append({"race_id": payload.get("race_id"), "error": str(exc)})
    rows.sort(key=lambda r: (r["scope"], r["date"], r["race_id"]))
    return rows, {"counts": dict(counts), "errors": errors,
                  "static_replay_not_oof": {s: {k: summarize_vectors(np.asarray(v)) for k, v in group.items() if v}
                                            for s, group in static.items()}}


def blend(base: list[int], signal: np.ndarray, numbers: list[int], preserve: int, weight: float) -> list[int]:
    return reranker._blend_model_ranking(base, dict(zip(numbers, map(float, signal), strict=True)),
                                         preserve_top_n=preserve, blend_weight=weight)[0]


def fit_model(rows: list[dict], feature: str, *, kind: str = "classifier", target: str = "top3",
              fresh: bool = False, multiplier: float = 1.0) -> Any:
    feature_key = "fresh_features" if fresh else "features"
    matrices, labels, weights, groups = [], [], [], []
    for row in rows:
        x = row[feature_key][feature]
        y = np.asarray([int(n in row["top3"] if target == "top3" else n == row["winner"])
                        for n in row["numbers"]])
        if kind == "linear":
            positives, negatives = x[y == 1], x[y == 0]
            differences = (positives[:, None, :] - negatives[None, :, :]).reshape(-1, x.shape[1])
            if not len(differences):
                continue
            matrices.extend((differences, -differences))
            labels.extend((np.ones(len(differences)), np.zeros(len(differences))))
            weights.extend((np.full(len(differences), 0.5 / len(differences)),) * 2)
            continue
        matrices.append(x)
        labels.append(y)
        positive_weight = max(len(y) - sum(y), 1) / max(sum(y), 1)
        w = np.where(y == 1, positive_weight, 1.0)
        if target == "top3":
            w[np.asarray(row["numbers"]) == row["winner"]] *= multiplier
        weights.append(w)
        groups.append(len(y))
    x, y, weight = np.concatenate(matrices), np.concatenate(labels), np.concatenate(weights)
    if kind == "linear":
        model = LogisticRegression(C=1.0, fit_intercept=False, max_iter=500, random_state=42)
        return model.fit(x, y, sample_weight=weight)
    if kind == "ranker":
        from lightgbm import LGBMRanker
        model = LGBMRanker(objective="lambdarank", n_estimators=180, learning_rate=0.04,
                           num_leaves=7, min_child_samples=40, reg_lambda=5.0,
                           label_gain=[0, 1], lambdarank_truncation_level=8,
                           random_state=42, n_jobs=2, verbosity=-1, deterministic=True,
                           force_col_wise=True)
        return model.fit(x, y, group=groups)
    model = HistGradientBoostingClassifier(learning_rate=0.06, max_iter=180, max_leaf_nodes=7,
                                           min_samples_leaf=40, l2_regularization=2.0,
                                           random_state=42)
    return model.fit(x, y, sample_weight=weight)


def signal(model: Any, matrix: np.ndarray, kind: str = "classifier") -> np.ndarray:
    if kind == "ranker":
        return model.booster_.predict(matrix, num_threads=2)
    return model.predict_proba(matrix)[:, 1]


def evaluate_fold(training: list[dict], testing: list[dict], scope: str) -> dict[str, np.ndarray]:
    _, _, preserve, strength, multiplier = POLICIES[scope]
    active = scope != "central_dirt"
    winner = fit_model(training, "winner", target="winner") if active else None
    place = fit_model(training, "native", multiplier=multiplier) if active else None
    fresh_winner = fit_model(training, "winner", target="winner", fresh=True) if active else None
    fresh_place = fit_model(training, "native", multiplier=multiplier, fresh=True) if active else None
    context = fit_model(training, "context", multiplier=multiplier)
    rank_native = fit_model(training, "native", kind="ranker")
    rank_context = fit_model(training, "context", kind="ranker")
    large_training = [r for r in training if len(r["numbers"]) >= 13]
    large = fit_model(large_training, "context", kind="ranker") if len(large_training) >= 60 else None
    linear = fit_model(training, "components", kind="linear")
    batch = {name: np.concatenate([r["features"][name] for r in testing])
             for name in ("winner", "native", "context", "components")}
    fresh_batch = {name: np.concatenate([r["fresh_features"][name] for r in testing])
                   for name in ("winner", "native")}
    predictions = {
        "context": signal(context, batch["context"]),
        "rank_native": signal(rank_native, batch["native"], "ranker"),
        "rank_context": signal(rank_context, batch["context"], "ranker"),
        "linear": signal(linear, batch["components"]),
    }
    if large is not None:
        predictions["large"] = signal(large, batch["context"], "ranker")
    if active:
        predictions.update({"winner": signal(winner, batch["winner"]),
                            "place": signal(place, batch["native"]),
                            "fresh_winner": signal(fresh_winner, fresh_batch["winner"]),
                            "fresh_place": signal(fresh_place, fresh_batch["native"])})
    results = {name: [] for name in ("current", *CANDIDATES)}
    offset = 0
    for row in testing:
        numbers = row["numbers"]
        end = offset + len(numbers)
        signals = {name: values[offset:end] for name, values in predictions.items()}
        offset = end
        base = blend(row["raw"], signals["winner"], numbers, 0, 0.5) if active else row["raw"]
        current = blend(base, signals["place"], numbers, preserve, strength) if active else base
        fresh_base = blend(row["fresh_raw"], signals["fresh_winner"], numbers, 0, 0.5) if active else row["fresh_raw"]
        orders = {
            "current": current,
            "refresh_analysis": blend(fresh_base, signals["fresh_place"], numbers, preserve, strength) if active else fresh_base,
            "missing_context": blend(base, signals["context"], numbers, preserve, strength),
            "rank_native": blend(base, signals["rank_native"], numbers, preserve, strength),
            "rank_context": blend(base, signals["rank_context"], numbers, preserve, strength),
            "large_field_rank": blend(current, signals["large"], numbers, preserve, 0.35)
                                if large is not None and len(numbers) >= 13 else current,
            "linear_components": blend(current, signals["linear"], numbers, preserve, 0.25),
        }
        for name, order in orders.items():
            if name != "refresh_analysis" and order[0] != current[0]:
                raise AssertionError("candidate changed the protected Top1")
            results[name].append(metric_vector(row, order))
    return {name: np.asarray(values) for name, values in results.items()}


def repeated_splits(rows: list[dict], seed: int):
    groups = np.asarray([r["date"] for r in rows])
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in splitter.split(np.arange(len(rows)), groups=groups):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("race date leaked between folds")
        yield train, test


def bootstrap_ci(rows: list[dict], differences: np.ndarray) -> list[float]:
    # Repeats are not independent samples; average per race, then resample dates.
    dates = sorted({r["date"] for r in rows})
    sums = np.asarray([sum(d for r, d in zip(rows, differences, strict=True) if r["date"] == day)
                       for day in dates])
    counts = np.asarray([sum(r["date"] == day for r in rows) for day in dates])
    rng = np.random.default_rng(20260908)
    choices = rng.integers(0, len(dates), size=(2000, len(dates)))
    gains = sums[choices].sum(axis=1) / counts[choices].sum(axis=1) / 3.0 * 100.0
    return np.quantile(gains, [0.025, 0.975]).round(4).tolist()


def compare(rows: list[dict], baseline: np.ndarray, candidate: np.ndarray) -> dict:
    repeats = []
    for before, after in zip(baseline, candidate, strict=True):
        b, c = summarize_vectors(before), summarize_vectors(after)
        repeats.append({"current": b, "candidate": c, "delta": metric_delta(b, c)})
    current = summarize_vectors(baseline.mean(axis=0))
    proposed = summarize_vectors(candidate.mean(axis=0))
    diff = (candidate[:, :, 0] - baseline[:, :, 0]).mean(axis=0)
    strata = {}
    for name, predicate in (("le8", lambda n: n <= 8), ("9to12", lambda n: 9 <= n <= 12),
                            ("ge13", lambda n: n >= 13)):
        indexes = [i for i, row in enumerate(rows) if predicate(len(row["numbers"]))]
        if indexes:
            bm = summarize_vectors(baseline[:, indexes].mean(axis=0))
            cm = summarize_vectors(candidate[:, indexes].mean(axis=0))
            random_recall = np.mean([min(5, len(rows[i]["numbers"])) / len(rows[i]["numbers"]) for i in indexes]) * 100
            strata[name] = {"current": bm, "candidate": cm, "delta": metric_delta(bm, cm),
                            "random_recall_pct": round(random_recall, 4)}
    return {"current_mean": current, "candidate_mean": proposed,
            "mean_delta": metric_delta(current, proposed), "repeats": repeats,
            "positive_repeats": sum(r["delta"]["slots"] > 0 for r in repeats),
            "recall_delta_95ci_date_clustered": bootstrap_ci(rows, diff), "field_size": strata}


def run_scope(rows: list[dict], scope: str, output: Path, seeds: tuple[int, ...]) -> dict:
    repeated = {name: [] for name in ("current", *CANDIDATES)}
    assignments = []
    for seed in seeds:
        vectors = {name: np.zeros((len(rows), len(METRICS))) for name in repeated}
        seen = np.zeros(len(rows), dtype=int)
        for fold, (train, test) in enumerate(repeated_splits(rows, seed)):
            result = evaluate_fold([rows[i] for i in train], [rows[i] for i in test], scope)
            for name, values in result.items():
                vectors[name][test] = values
            seen[test] += 1
            assignments.extend({"seed": seed, "fold": fold, "race_id": rows[i]["race_id"]} for i in test)
            print(f"{scope} seed={seed} fold={fold + 1}/5 complete", flush=True)
        if not np.all(seen == 1):
            raise AssertionError("every eligible race must be evaluated exactly once per repeat")
        for name, values in vectors.items():
            repeated[name].append(values)
    arrays = {name: np.asarray(values) for name, values in repeated.items()}
    np.savez_compressed(output / f"{scope}_oof_metrics.npz", **arrays)
    report = {"races": len(rows), "date_groups": len({r["date"] for r in rows}),
              "every_race_out_of_fold": True, "seeds": seeds,
              "candidates": {name: compare(rows, arrays["current"], arrays[name]) for name in CANDIDATES}}
    # Four non-overlapping future blocks are diagnostic, never a seven-day selection gate.
    days = sorted({r["date"] for r in rows})
    blocks = np.array_split(days, 5)
    forward_rows, forward_vectors, periods = [], {name: [] for name in repeated}, []
    for block in blocks[1:]:
        training = [r for r in rows if r["date"] < block[0]]
        testing = [r for r in rows if r["date"] in block]
        result = evaluate_fold(training, testing, scope)
        current = summarize_vectors(result["current"])
        periods.append({"start": str(block[0]), "end": str(block[-1]), "training_races": len(training),
                        "test_races": len(testing), "candidate_deltas": {
                            name: metric_delta(current, summarize_vectors(result[name])) for name in CANDIDATES}})
        forward_rows.extend(testing)
        for name in repeated:
            forward_vectors[name].extend(result[name])
        print(f"{scope} chronological block through {block[-1]} complete", flush=True)
    forward_arrays = {k: np.asarray([v]) for k, v in forward_vectors.items()}
    report["chronological_diagnostic"] = {"blocks": periods,
        "candidates": {name: compare(forward_rows, forward_arrays["current"], forward_arrays[name])
                       for name in CANDIDATES}}
    (output / f"{scope}_folds.json").write_text(json.dumps(assignments, indent=2), encoding="utf-8")
    (output / f"{scope}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Agent full-audit ranking experiments")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, action="append")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    snapshot, output = args.snapshot.resolve(), args.output_dir.resolve()
    analysis_root = (WORKSPACE / "analysis_outputs").resolve()
    if not output.is_relative_to(analysis_root) or output == snapshot or output.is_relative_to(snapshot):
        raise SystemExit("Output must be a separate derived directory under analysis_outputs")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "prepared.joblib"
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with threadpool_limits(limits=2):
        if cache.exists():
            payload = joblib.load(cache)
            if payload["snapshot"] != str(snapshot) or payload["source_hash"] != source_hash:
                raise SystemExit("Cache provenance mismatch; use a new output directory")
            rows, preparation = payload["rows"], payload["preparation"]
        else:
            rows, preparation = prepare(snapshot)
            joblib.dump({"snapshot": str(snapshot), "source_hash": source_hash,
                         "rows": rows, "preparation": preparation}, cache, compress=3)
        (output / "preparation.json").write_text(json.dumps(preparation, indent=2), encoding="utf-8")
        if args.prepare_only:
            print(json.dumps(preparation, indent=2))
            return
        report = {"snapshot": str(snapshot), "market_features_used": False,
                  "primary_metric": "Top5 recall of the actual Top3", "source_hash": source_hash,
                  "frozen_inputs": ["pedigree_analyses", "lessons"],
                  "refreshed_inputs": ["deep_analyses", "race_level_analyses", "pace_analyses"],
                  "selection_caveat": "Fixed exploratory candidates; repeated CV is not an untouched external test.",
                  "preparation": preparation, "scopes": {}}
        for scope in args.scope or SCOPES:
            group = [r for r in rows if r["scope"] == scope]
            report["scopes"][scope] = run_scope(group, scope, output, tuple(args.seeds))
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({s: {n: c["mean_delta"] for n, c in data["candidates"].items()}
                          for s, data in report["scopes"].items()}, indent=2))


if __name__ == "__main__":
    main()

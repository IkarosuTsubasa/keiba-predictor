"""Fit bounded simulator coefficients, freeze them, then evaluate unseen dates.

Uses a compiled implementation of the frontend dynamics with a 1 m terrain
lookup during fitting. Final published metrics come from the exact JS engine.
Run with the optional isolated numba runtime installed under analysis_outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "analysis_outputs/simulator_calibration_runtime"
if RUNTIME.exists(): sys.path.insert(0, str(RUNTIME))
import numpy as np
from numba import njit, prange, set_num_threads
from scipy.optimize import least_squares

NAMES = ["turfSpeed", "dirtSpeed", "localOffset", "distanceSlope", "abilitySpeed", "earlySpeed", "kickBase", "kickClosing", "kickStyle", "energyDrain", "staminaSaving", "fatigueScale"]
LOWER = np.array([13., 12., -3., -3., 0., 0., 0., 0., 0., .3, 0., 2.])
UPPER = np.array([19., 18., 1., 1., 5., 3., 2., 4., 1., 1.3, .8, 10.])
SCALE = np.array([1., 1., 1., 1., 1.5, 1.5, .75, 1.5, .5, .3, .3, 3.])


@njit(cache=True)
def clip(x, lo=0., hi=1.):
    return min(hi, max(lo, x))


@njit(cache=True, parallel=True)
def dynamics(parameters, context, profiles, counts, terrain_ids, terrain):
    """Same fixed 0.2 s update, lane/pressure/energy and finish interpolation."""
    out = np.zeros((len(context), profiles.shape[1], 2))
    for r in prange(len(context)):
        length, turf, local = context[r]
        n = counts[r]
        base = (parameters[0] if turf else parameters[1]) + local * parameters[2] + parameters[3] * math.log(length / 1600.)
        nominal = length / base
        front_count = 0
        for h in range(n):
            if profiles[r, h, 4] < 2: front_count += 1
        pressure = clip((front_count - 2) / 6., 0., .8)
        d = np.zeros(n); v = np.zeros(n); energy = np.ones(n)
        lanes = np.array([.7 + h * 6.5 / max(1, n - 1) for h in range(n)])
        finished = np.zeros(n, dtype=np.bool_); closing_start = np.full(n, -1.)
        for tick in range(1, 6000):
            t = tick * .2
            leader = max(d)
            for h in range(n):
                if finished[h]: continue
                ability, early, closing, stamina, style = profiles[r, h]
                progress = d[h] / length
                late = clip((progress - .66) / .25)
                early_phase = 1. - clip(progress / .42)
                front_style = 1.15 if style == 0 else .7 if style == 1 else -.6 if style == 3 else -.15
                early_drive = (front_style + (early - .5) * parameters[5]) * early_phase
                flow = pressure * .55 * (1. - late * .75)
                back = style == 2 or style == 3
                kick = late * (parameters[6] + closing * parameters[7] + (parameters[8] if back else 0.))
                fatigue = max(0., .36 - energy[h]) * (parameters[11] - stamina * 2.)
                target_lane = lanes[h] if progress < .12 else (.85 if style == 0 else 1.3 + (h % 4) * .65) + late * (1.2 if back else .15)
                lane = lanes[h] + clip(target_lane - lanes[h], -.07, .07)
                ix = int(d[h]); frac = d[h] - ix
                grade = terrain[terrain_ids[r], ix, 0] * (1. - frac) + terrain[terrain_ids[r], ix + 1, 0] * frac
                bend = terrain[terrain_ids[r], ix, 1] > .5
                outer = lane * .055 if bend else 0.
                target = base + (ability - .5) * parameters[4] + early_drive + flow + kick - fatigue - outer - grade * 18.
                acceleration = 2.8 + early if t < 6. else .85
                speed = clip(v[h] + clip(target - v[h], -.3, acceleration * .2), 0., 23.)
                effort = max(.6, speed / base)
                lead = pressure * .15 if d[h] >= leader - 4 else 0.
                terrain_energy = clip(1. + max(0., grade) * 16. + min(0., grade) * 4., .85, 1.8)
                drain = parameters[9] / nominal * effort ** 3.2 * (1.14 - stamina * parameters[10] + lead) * terrain_energy
                energy[h] = clip(energy[h] - drain * .2)
                new_d = d[h] + speed * .2
                if closing_start[h] < 0 and new_d >= length - 600.:
                    closing_start[h] = t - .2 + (length - 600. - d[h]) / max(speed, .01)
                if new_d >= length:
                    out[r, h, 0] = t - .2 + (length - d[h]) / max(speed, .01)
                    out[r, h, 1] = out[r, h, 0] - closing_start[h]
                    finished[h] = True
                d[h] = min(length, new_d); v[h] = speed; lanes[h] = lane
            if np.all(finished): break
    return out


def arrays(prepared):
    cases = prepared["cases"]; n = len(cases); width = max(len(c["profiles"]) for c in cases)
    context = np.zeros((n, 3)); profiles = np.zeros((n, width, 5)); counts = np.zeros(n, dtype=np.int64)
    target = np.full((n, width, 3), np.nan)
    keys = list(prepared["terrain"]); terrain_ids = np.array([keys.index(c["terrainKey"]) for c in cases])
    maxdist = max(len(c) for c in prepared["terrain"].values()) + 1
    terrain = np.zeros((len(keys), maxdist, 2))
    for i, key in enumerate(keys):
        curve = np.array(prepared["terrain"][key]); terrain[i, :len(curve)] = curve; terrain[i, len(curve):] = curve[-1]
    for r, c in enumerate(cases):
        info = c["race"]["race_info"]; context[r] = [info["distance"], info["surface"] == "芝", c["scope"] == "local"]
        counts[r] = len(c["profiles"]); labels = {x["horse_no"]: x for x in c["labels"]}
        for h, p in enumerate(c["profiles"]):
            profiles[r, h] = [p["ability"], p["early"], p["closing"], p["stamina"], ["逃げ", "先行", "差し", "追込", "不明"].index(p["style"])]
            lab = labels.get(p["horse_no"], {})
            target[r, h] = [lab.get("time") or np.nan, lab.get("final_3f") or np.nan, lab.get("finish") or np.nan]
    return context, profiles, counts, terrain_ids, terrain, target


def summarize(cases, predictions):
    rows = []
    by_id = {r["race_id"]: r["predictions"] for r in predictions}
    for c in cases:
        info = c["race"]["race_info"]; pred = by_id[info["race_id"]]
        pred = sorted(pred, key=lambda p: (p["time"], p["horse_no"]))
        labels = {p["horse_no"]: p for p in c["labels"]}
        top3 = {p["horse_no"] for p in c["labels"] if p["finish"] <= 3}
        winner = next(p["horse_no"] for p in c["labels"] if p["finish"] == 1)
        covered = len({p["horse_no"] for p in pred[:5]} & top3)
        ranks = {p["horse_no"]: i + 1 for i, p in enumerate(pred)}
        time_errors = [abs(p["time"] - labels[p["horse_no"]]["time"]) for p in pred if labels.get(p["horse_no"], {}).get("time")]
        close_errors = [abs(p["final_3f"] - labels[p["horse_no"]]["final_3f"]) for p in pred if labels.get(p["horse_no"], {}).get("final_3f")]
        rank_errors = [abs(ranks[no] - lab["finish"]) for no, lab in labels.items() if no in ranks]
        rows.append({"race_id": info["race_id"], "date": info["race_date"], "scope": c["scope"], "surface": info["surface"], "split": c["split"], "covered": covered, "winner_rank": ranks[winner], "winner_correct": pred[0]["horse_no"] == winner, "time_error_sum": sum(time_errors), "time_n": len(time_errors), "closing_error_sum": sum(close_errors), "closing_n": len(close_errors), "rank_error_sum": sum(rank_errors), "rank_n": len(rank_errors), "field": len(pred)})
    return rows


def aggregate(rows):
    n = len(rows)
    if not n: return {"races": 0}
    time_n = sum(r["time_n"] for r in rows); close_n = sum(r["closing_n"] for r in rows); rank_n = sum(r["rank_n"] for r in rows)
    return {"races": n, "covered_seats": sum(r["covered"] for r in rows), "top3_recall": sum(r["covered"] for r in rows) / (3 * n), "all3_races": sum(r["covered"] == 3 for r in rows), "at_least2_races": sum(r["covered"] >= 2 for r in rows), "winner_top3": sum(r["winner_rank"] <= 3 for r in rows) / n, "winner_top5": sum(r["winner_rank"] <= 5 for r in rows) / n, "winner_mean_rank": sum(r["winner_rank"] for r in rows) / n, "winner_accuracy": sum(r["winner_correct"] for r in rows) / n, "time_mae": sum(r["time_error_sum"] for r in rows) / time_n if time_n else None, "time_n": time_n, "closing_mae": sum(r["closing_error_sum"] for r in rows) / close_n if close_n else None, "closing_n": close_n, "rank_mae": sum(r["rank_error_sum"] for r in rows) / rank_n if rank_n else None, "rank_n": rank_n}


def predictions_from_array(cases, values):
    return [{"race_id": c["race"]["race_info"]["race_id"], "predictions": [{"horse_no": p["horse_no"], "time": float(values[r, h, 0]), "final_3f": float(values[r, h, 1])} for h, p in enumerate(c["profiles"])]} for r, c in enumerate(cases)]


def fit(prepared, directory):
    if (directory / 'frozen_model.json').exists():
        raise ValueError('A model is already frozen. Use a new experiment directory; do not overwrite validation evidence.')
    cases = prepared["cases"]; base = np.array([prepared["parameters"][k] for k in NAMES])
    ctx, prof, counts, tids, terrain, target = arrays(prepared)
    dev = np.array([i for i, c in enumerate(cases) if c["split"] == "development"])
    config = {"names": NAMES, "bounds": [LOWER.tolist(), UPPER.tolist()], "prior_scale": SCALE.tolist(), "objective": "soft_l1: finish seconds/3 + final600 seconds/2 + actual top3 vs others pairwise softplus + prior shrinkage; equal race weight", "max_nfev": 18, "difference_step": .02, "blend_candidates": [0., .5, 1.], "stochastic_seeds": [20261101, 42, 8675309, 20261031, 71], "final_selection": "development OOF Top5 coverage, then all3, at_least2, lower time MAE; holdout never selects parameters", "parameter_interpretation": "energy/stamina remain latent, jointly regularized; terrain response and acceleration remain fixed", "prepared_sha256": hashlib.sha256((directory / "prepared.json").read_bytes()).hexdigest()}
    config["dependencies"] = {k: importlib.metadata.version(k) for k in ("numpy", "numba", "scipy")}
    config["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("frontend/src/lib/raceSimulator.js", "frontend/src/lib/raceCourses.js", "frontend/src/fixtures/raceCourses.json", "scripts/calibrate_simulator.py", "scripts/simulator_calibration_bridge.mjs")}
    (directory / "fit_protocol.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    def solve(indices, label):
        assert all(cases[i]["split"] == "development" for i in indices)
        args = (ctx[indices], prof[indices], counts[indices], tids[indices], terrain)
        y = target[indices]; clock_mask = np.isfinite(y[:, :, 0]); close_mask = np.isfinite(y[:, :, 1])
        clock_w = np.sqrt(1. / np.maximum(1, clock_mask.sum(axis=1)))[:, None]
        close_w = np.sqrt(1. / np.maximum(1, close_mask.sum(axis=1)))[:, None]
        pair_r, pair_a, pair_b = [], [], []
        for r in range(len(indices)):
            for a in range(counts[indices[r]]):
                if y[r, a, 2] <= 3:
                    for b in range(counts[indices[r]]):
                        if y[r, b, 2] > 3:
                            pair_r.append(r); pair_a.append(a); pair_b.append(b)
        pair_r = np.array(pair_r, dtype=int); pair_a = np.array(pair_a, dtype=int); pair_b = np.array(pair_b, dtype=int)
        pair_w = 1. / np.sqrt(np.maximum(1, np.bincount(pair_r, minlength=len(indices)))[pair_r])
        evaluations = 0; started = time.monotonic()
        def residual(x):
            nonlocal evaluations
            values = dynamics(x, *args); evaluations += 1
            clock = ((values[:, :, 0] - y[:, :, 0]) / 3. * clock_w)[clock_mask]
            closing = ((values[:, :, 1] - y[:, :, 1]) / 2. * close_w)[close_mask]
            gap = (values[pair_r, pair_a, 0] - values[pair_r, pair_b, 0]) / 1.5
            ranking = np.logaddexp(0., gap) * pair_w
            prior = (x - base) / SCALE * math.sqrt(len(indices) * .02)
            if evaluations % 30 == 0: print(f"{label}: {evaluations} evaluations, {time.monotonic()-started:.1f}s", flush=True)
            return np.concatenate((clock, closing, ranking, prior))
        result = least_squares(residual, base, bounds=(LOWER, UPPER), loss="soft_l1", f_scale=1., diff_step=.02, x_scale=SCALE, max_nfev=18, ftol=.005, xtol=.005, gtol=.005)
        print(label, dict(zip(NAMES, result.x.tolist())), "evaluations", evaluations, flush=True)
        return result.x
    # Warm compilation before timing the repeated optimization.
    start = time.monotonic(); dynamics(base, ctx[:2], prof[:2], counts[:2], tids[:2], terrain)
    print(f"Compiled in {time.monotonic()-start:.1f}s; fitting {len(dev)} development races", flush=True)
    fold_parameters = {}; oof = {a: np.zeros((len(cases), prof.shape[1], 2)) for a in config["blend_candidates"]}
    for fold in range(5):
        train = np.array([i for i in dev if cases[i]["fold"] != fold]); test = np.array([i for i in dev if cases[i]["fold"] == fold])
        x = solve(train, f"fold {fold}"); fold_parameters[fold] = x.tolist()
        for a in oof:
            oof[a][test] = dynamics(base * (1-a) + x * a, ctx[test], prof[test], counts[test], tids[test], terrain)
    dev_cases = [cases[i] for i in dev]; selection = {}
    for a in oof:
        pred = predictions_from_array(dev_cases, oof[a][dev]); rows = summarize(dev_cases, pred)
        selection[a] = aggregate(rows)
        (directory / f"oof_{a}.json").write_text(json.dumps(pred), encoding="utf-8")
    approximate_selection = dict(selection)
    # Select blends with the exact browser engine on development folds only.
    (directory / "development_cases.json").write_text(json.dumps(dev_cases, ensure_ascii=False), encoding="utf-8")
    for a in selection:
        by_race = {c["race"]["race_info"]["race_id"]: dict(zip(NAMES, (base*(1-a)+np.array(fold_parameters[c["fold"]])*a).tolist())) for c in dev_cases}
        model_file = directory / f"oof_model_{a}.json"; model_file.write_text(json.dumps({"byRace": by_race}), encoding="utf-8")
        prediction_file = directory / f"exact_oof_{a}.json"
        subprocess.run(["node", str(ROOT / "scripts/simulator_calibration_bridge.mjs"), "evaluate", str(directory / "development_cases.json"), str(prediction_file), str(model_file)], cwd=ROOT, check=True)
        selection[a] = aggregate(summarize(dev_cases, json.loads(prediction_file.read_text(encoding="utf-8"))))
    # Reject a candidate with materially worse winner coverage as a guardrail.
    eligible = [a for a in selection if selection[a]["winner_top5"] >= selection[0.]["winner_top5"] - .01]
    chosen = max(eligible, key=lambda a: (selection[a]["covered_seats"], selection[a]["all3_races"], selection[a]["at_least2_races"], -selection[a]["time_mae"]))
    x = solve(dev, "all development")
    frozen = {"parameters": dict(zip(NAMES, (base * (1-chosen) + x * chosen).tolist())), "fitted_parameters": dict(zip(NAMES, x.tolist())), "chosen_blend": chosen, "development_oof": selection, "approximate_development_oof": approximate_selection, "fold_parameters": fold_parameters, "holdout_used_for_selection": False}
    # Write before invoking any holdout evaluation. Baseline may win; still retain
    # the fitted candidate to report its held-out tradeoff, without choosing on it.
    (directory / "frozen_model.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (directory / "fitted_candidate.json").write_text(json.dumps({"parameters": frozen["fitted_parameters"]}, indent=2), encoding="utf-8")
    (directory / "oof_model.json").write_text(json.dumps({"byRace": {c["race"]["race_info"]["race_id"]: dict(zip(NAMES, (base*(1-chosen)+np.array(fold_parameters[c["fold"]])*chosen).tolist())) for c in dev_cases}}), encoding="utf-8")
    print("Frozen", json.dumps({"blend": chosen, "OOF": selection}), flush=True)


def evaluate(directory):
    prepared = json.loads((directory / "prepared.json").read_text(encoding="utf-8")); cases = prepared["cases"]
    frozen = json.loads((directory / "frozen_model.json").read_text(encoding="utf-8"))
    bridge = ROOT / "scripts/simulator_calibration_bridge.mjs"
    files = [("baseline", None), ("calibrated", directory / "frozen_model.json"), ("fitted_candidate", directory / "fitted_candidate.json")]
    outputs = {}
    cached = {}
    for label, parameters in files:
        identity = json.dumps(json.loads(parameters.read_text(encoding="utf-8"))["parameters"] if parameters else prepared["parameters"], sort_keys=True)
        if identity in cached:
            outputs[label] = outputs[cached[identity]]
            continue
        command = ["node", str(bridge), "evaluate", str(directory / "cases.json"), str(directory / f"exact_{label}.json")]
        if parameters: command.append(str(parameters))
        subprocess.run(command, cwd=ROOT, check=True)
        outputs[label] = json.loads((directory / f"exact_{label}.json").read_text(encoding="utf-8"))
        cached[identity] = label
    # The published development numbers use fold-specific models, not all-dev fit.
    dev_cases = [c for c in cases if c["split"] == "development"]
    (directory / "development_cases.json").write_text(json.dumps(dev_cases, ensure_ascii=False), encoding="utf-8")
    outputs["oof"] = json.loads((directory / f"exact_oof_{frozen['chosen_blend']}.json").read_text(encoding="utf-8"))
    rows = {label: summarize(cases if label != "oof" else dev_cases, values) for label, values in outputs.items()}
    summary = {"holdout": {}, "development_oof": {"baseline": aggregate([r for r in rows["baseline"] if r["split"] == "development"]), "calibrated": aggregate(rows["oof"])}, "holdout_by_scope": {}}
    for label in ("baseline", "calibrated", "fitted_candidate"):
        summary["holdout"][label] = aggregate([r for r in rows[label] if r["split"] == "holdout"])
        summary["holdout_by_scope"][label] = {scope: aggregate([r for r in rows[label] if r["split"] == "holdout" and r["scope"] == scope]) for scope in ("central", "local")}
    # Date-block bootstrap respects within-day race correlation. Positive recall
    # delta and negative MAE delta favor the calibrated model.
    b = [r for r in rows["baseline"] if r["split"] == "holdout"]; c = [r for r in rows["calibrated"] if r["split"] == "holdout"]
    days = sorted({r["date"] for r in b}); rng = np.random.default_rng(20261101); deltas = []
    for _ in range(2000):
        sample = rng.choice(days, len(days)); bb = [r for d in sample for r in b if r["date"] == d]; cc = [r for d in sample for r in c if r["date"] == d]
        am, cm = aggregate(bb), aggregate(cc)
        deltas.append([cm["top3_recall"]-am["top3_recall"], cm["time_mae"]-am["time_mae"], cm["closing_mae"]-am["closing_mae"] if cm["closing_mae"] is not None and am["closing_mae"] is not None else np.nan])
    summary["paired_date_bootstrap_95_ci"] = {name: np.nanquantile(np.array(deltas)[:, i], [.025, .975]).tolist() for i, name in enumerate(("recall_delta", "time_mae_delta", "closing_mae_delta"))}
    summary["holdout_days"] = len(days)
    # Predeclared random seeds evaluate individual animation runs; they do not
    # select coefficients and their variation is not a confidence interval.
    hold_cases = [c for c in cases if c["split"] == "holdout"]
    hold_file = directory / "holdout_cases.json"; hold_file.write_text(json.dumps(hold_cases, ensure_ascii=False), encoding="utf-8")
    summary["stochastic_holdout"] = {}
    config = json.loads((directory / "fit_protocol.json").read_text(encoding="utf-8"))
    for label in ("baseline", "fitted_candidate"):
        metrics = []
        parameters = prepared["parameters"] if label == "baseline" else frozen["fitted_parameters"]
        for seed in config["stochastic_seeds"]:
            model_file = directory / f"random_{label}_{seed}_model.json"; model_file.write_text(json.dumps({"parameters": parameters, "seed": seed, "variation": 1}), encoding="utf-8")
            pred_file = directory / f"random_{label}_{seed}.json"
            subprocess.run(["node", str(bridge), "evaluate", str(hold_file), str(pred_file), str(model_file)], cwd=ROOT, check=True)
            metrics.append({"seed": seed, **aggregate(summarize(hold_cases, json.loads(pred_file.read_text(encoding="utf-8"))))})
        summary["stochastic_holdout"][label] = metrics
    # Independent implementation agreement is checked across all available cases.
    ctx, prof, counts, tids, terrain, _ = arrays(prepared)
    p = np.array([frozen["parameters"][k] for k in NAMES]); approx = dynamics(p, ctx, prof, counts, tids, terrain)
    exact_map = {r["race_id"]: {h["horse_no"]: h for h in r["predictions"]} for r in outputs["calibrated"]}
    difference = [abs(approx[r,h,0] - exact_map[c["race"]["race_info"]["race_id"]][v["horse_no"]]["time"]) for r,c in enumerate(cases) for h,v in enumerate(c["profiles"])]
    summary["compiled_vs_exact_time_difference"] = {"mean_seconds": float(np.mean(difference)), "max_seconds": float(max(difference)), "horse_count": len(difference)}
    (directory / "race_metrics.json").write_text(json.dumps(rows), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "evaluate")); parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args(); set_num_threads(min(8, os.cpu_count() or 1))
    if args.mode == "fit": fit(json.loads((args.directory / "prepared.json").read_text(encoding="utf-8")), args.directory)
    else: evaluate(args.directory)

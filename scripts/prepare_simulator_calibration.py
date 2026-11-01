"""Create immutable, market-free inputs and separate labels from Agent audits.

Does not write to any source directory. Later audit histories may supply outcome
labels (last 600 m), but are NEVER merged into an earlier race's inputs.
"""
from __future__ import annotations

import argparse
import collections
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUN_FIELDS = ("race_id", "date", "surface", "distance", "finish", "field_size", "corner_positions", "final_3f")


def read_json(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw), hashlib.sha256(raw).hexdigest()


def candidates(root, overlay=None):
    """Same final/version/mtime precedence as input_selection, without reading losers."""
    groups = collections.defaultdict(list)
    paths = {str(p.relative_to(root)): p for p in list(root.glob("*.json.gz")) + list((root / "_versions").glob("*/*/*.json.gz"))}
    if overlay:
        paths.update({str(p.relative_to(overlay)): p for p in list(overlay.glob("*.json.gz")) + list((overlay / "_versions").glob("*/*/*.json.gz"))})
    for p in paths.values():
        if p.parent == root or (overlay and p.parent == overlay):
            groups[p.name.removesuffix(".json.gz")].append(p)
            continue
        if p.parent.name in ("early", "final"):
            groups[p.parent.parent.name].append(p)
    for rid, paths in sorted(groups.items()):
        finals = [p for p in paths if p.parent.name == "final"]
        if finals:
            p = max(finals, key=lambda x: (x.stat().st_mtime_ns, x.name))
            x, sha = read_json(p)
            if str(x.get("race_id")) != rid:
                raise ValueError(f"Version identity mismatch: {p}")
            yield rid, p, x, sha
            continue
        loaded = []
        for p in paths:
            x, sha = read_json(p)
            stage = x.get("prediction_stage") or x.get("outputs", {}).get("prediction", {}).get("prediction_stage")
            stage = p.parent.name if p.parent.name in ("early", "final") else stage
            loaded.append(((stage in ("final", "final_prediction", "finalized"), p.parent.name in ("early", "final"), p.stat().st_mtime_ns, p.name), p, x, sha))
        _, p, x, sha = max(loaded, key=lambda row: row[0])
        yield rid, p, x, sha


def seconds(value):
    if not isinstance(value, str):
        return None
    try:
        parts = [float(v) for v in value.split(":")]
        total = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
        return total if total > 15 else None
    except ValueError:
        return None


def valid_date(value):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def prepare(snapshot, output, overlay=None, before_date=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; existing evidence is never overwritten")
    for protected in (snapshot, ROOT / "pipeline/data", ROOT / "keiba_llm_agent/data", *([overlay] if overlay else [])):
        if output.resolve().is_relative_to(protected.resolve()):
            raise ValueError("Output overlaps protected source")
    quality = collections.Counter()
    prediction_ids = {p.stem for p in (snapshot / "agent_predictions").glob("*.json") if re.fullmatch(r"\d{12}", p.stem)}
    result_ids = {p.stem for p in (snapshot / "agent_results").glob("*.json") if re.fullmatch(r"\d{12}", p.stem)}
    if overlay:
        verified = json.loads((overlay / "verification.json").read_text())
        if not verified.get("complete"): raise ValueError("Overlay is not verified")
        result_ids |= {p.stem for p in (overlay / "agent_results").glob("*.json")}
    quality["prediction_races"] = len(prediction_ids)
    quality["result_races"] = len(result_ids)
    cases, manifest, history_labels = [], [], collections.defaultdict(dict)
    selected_ids = set()
    for rid, path, audit, sha in candidates(snapshot / "agent_backtest_audit", overlay / "agent_backtest_audit" if overlay else None):
        quality["selected_audits"] += 1
        selected_ids.add(rid)
        quality["kind_" + str(audit.get("audit_kind"))] += 1
        raw = audit.get("inputs", {}).get("race_data") or {}
        info = raw.get("race_info") or audit.get("race_info") or {}
        race_day = valid_date(info.get("race_date"))
        try:
            captured_day = (datetime.fromisoformat(audit.get("created_at", "").replace("Z", "+00:00")) + timedelta(hours=9)).date()
            quality["captured_after_race_day"] += bool(race_day and captured_day > race_day)
        except (ValueError, TypeError):
            quality["capture_time_unknown"] += 1
        horses = raw.get("horses") or []
        # Build a label-only index before selecting the modelling population.
        for h in horses:
            identity = str(h.get("horse_id") or h.get("horse_name") or "")
            for run in h.get("recent_runs") or []:
                prior_day = valid_date(run.get("date"))
                if identity and race_day and prior_day and prior_day < race_day and run.get("race_id"):
                    key = (str(run["race_id"]), identity)
                    fingerprint = json.dumps({k: run.get(k) for k in ("date", "finish", "field_size", "final_3f", "corner_positions", "distance")}, sort_keys=True)
                    history_labels[key][fingerprint] = {k: run.get(k) for k in ("date", "finish", "field_size", "final_3f", "corner_positions", "distance")}
        result = audit.get("outcomes", {}).get("result") or {}
        result_path = snapshot / "agent_results" / f"{rid}.json"
        if overlay and (overlay / "agent_results" / f"{rid}.json").exists(): result_path = overlay / "agent_results" / f"{rid}.json"
        if result_path.exists():
            result, result_sha = read_json(result_path)
        else:
            result_sha = None
        reason = None
        if not re.fullmatch(r"\d{12}", rid): reason = "invalid_id"
        elif audit.get("audit_kind") != "full": reason = "prediction_only"
        elif not horses: reason = "missing_race_data"
        elif not race_day: reason = "invalid_race_date"
        elif before_date and race_day >= before_date: reason = "in_progress_day"
        elif info.get("surface") not in ("芝", "ダート"): reason = "non_flat_surface"
        elif not isinstance(info.get("distance"), (int, float)) or not 800 <= info["distance"] <= 3600: reason = "unsupported_distance"
        elif not result.get("finish_order"): reason = "missing_result"
        if reason:
            quality["excluded_" + reason] += 1
            continue
        numbers = [h.get("horse_no") for h in horses]
        if len(set(numbers)) != len(numbers) or len(horses) < 5:
            quality["excluded_invalid_field"] += 1
            continue
        labels = []
        for row in result["finish_order"]:
            if row.get("horse_no") not in numbers or not isinstance(row.get("finish"), int) or row["finish"] < 1:
                continue
            t = seconds(row.get("time"))
            if t and not 7 < info["distance"] / t < 22: t = None
            labels.append({"horse_no": row["horse_no"], "finish": row["finish"], "time": t})
        if len({r["horse_no"] for r in labels if r["finish"] <= 3}) != 3:
            quality["excluded_incomplete_top3"] += 1
            continue
        clean_horses = []
        for h in horses:
            runs = []
            for run in h.get("recent_runs") or []:
                prior_day = valid_date(run.get("date"))
                if not prior_day or prior_day >= race_day or str(run.get("race_id")) == rid:
                    quality["history_excluded_unknown_or_not_prior"] += 1
                    continue
                runs.append({k: run.get(k) for k in RUN_FIELDS})
            quality["horses"] += 1
            quality["horses_no_history"] += not bool(runs)
            clean_horses.append({"horse_no": h["horse_no"], "horse_id": str(h.get("horse_id") or h.get("horse_name") or ""), "horse_name": h.get("horse_name", ""), "frame_no": h.get("frame_no"), "recent_runs": runs})
        scope = "local" if info.get("source") == "local" or str(info.get("scope_key", "")).startswith("local") else "central"
        clean_info = {k: info.get(k) for k in ("race_id", "race_date", "course", "surface", "distance", "track_condition", "course_layout", "layout", "source", "scope_key")}
        clean_info["race_id"] = rid
        cases.append({"race": {"race_info": clean_info, "horses": clean_horses}, "labels": labels, "scope": scope, "captured_at": audit.get("created_at")})
        manifest.append({"race_id": rid, "audit": str(path.relative_to(snapshot)) if path.is_relative_to(snapshot) else str(path.resolve()), "sha256": sha, "result_sha256": result_sha, "missing_inputs": audit.get("missing_inputs", []), "scope": scope, "race_date": race_day.isoformat()})
        if len(cases) % 500 == 0: print(f"Prepared {len(cases)} races", flush=True)
    # Join post-outcome observations exclusively to the label table, checking IDs,
    # date, distance, actual placing and field size. Conflicts remain missing.
    for c in cases:
        info = c["race"]["race_info"]
        by_no = {h["horse_no"]: h for h in c["race"]["horses"]}
        for label in c["labels"]:
            h = by_no[label["horse_no"]]
            values = list(history_labels.get((info["race_id"], h["horse_id"]), {}).values())
            values = [r for r in values if r["date"] == info["race_date"] and r["distance"] == info["distance"] and r["finish"] == label["finish"] and r["field_size"] == len(by_no)]
            closing = {r["final_3f"] for r in values if isinstance(r["final_3f"], (float, int)) and 25 < r["final_3f"] < 65}
            label["final_3f"] = next(iter(closing)) if len(closing) == 1 else None
            quality["closing_label_conflicts"] += len(closing) > 1
            quality["closing_labels"] += label["final_3f"] is not None
            quality["time_labels"] += label["time"] is not None
            quality["finish_labels"] += 1
        quality["races_with_closing_label"] += any(r["final_3f"] is not None for r in c["labels"])
    cases.sort(key=lambda c: (c["race"]["race_info"]["race_date"], c["race"]["race_info"]["race_id"]))
    quality["settled_prediction_races_without_audit"] = len((prediction_ids & result_ids) - selected_ids)
    days = sorted({c["race"]["race_info"]["race_date"] for c in cases})
    # Freeze a latest 20%-of-days chronological holdout before any parameter fit.
    cutoff = days[int(len(days) * .8)]
    dev_days = [d for d in days if d < cutoff]
    for c in cases:
        d = c["race"]["race_info"]["race_date"]
        c["split"] = "holdout" if d >= cutoff else "development"
        c["fold"] = None if d >= cutoff else dev_days.index(d) % 5
    protocol = {"version": 1, "snapshot": str(snapshot.resolve()), "created_at": datetime.now(timezone.utc).isoformat(), "holdout_first_date": cutoff, "folds": 5, "fold_group": "race_date", "split_counts": dict(collections.Counter(c["split"] for c in cases)), "seed": 20261101, "selection_metric": "Top5 covers actual Top3 seats; time and final600 MAE secondary", "features": "prior raw runs only; no saved scores, analyses, odds, popularity or target outcomes", "labels": "official finish_order time and finish; final600 recovered from later Agent audit histories by race and horse identity, label-only", "snapshot_caveat": "Raw prior runs reconstructed by strict date; no guarantee every archived snapshot was frozen before post time", "quality": dict(quality), "manifest": manifest}
    protocol["verified_overlay"] = str(overlay.resolve()) if overlay else None
    protocol["complete_days_before"] = before_date.isoformat() if before_date else None
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("cases.json", cases), ("protocol.json", protocol)):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: protocol[k] for k in ("holdout_first_date", "split_counts", "quality")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--before-date", type=date.fromisoformat)
    args = parser.parse_args()
    prepare(args.snapshot.resolve(), args.output.resolve(), args.overlay.resolve() if args.overlay else None, args.before_date)

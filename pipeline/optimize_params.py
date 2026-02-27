import csv
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from surface_scope import (
    get_config_path,
    get_data_dir,
    get_scope_key,
    migrate_legacy_data,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = None
CONFIG_PATH = None
PREV_CONFIG_PATH = None
BET_TYPE_STATS_PATH = None


def pause_exit(message="Press Enter to exit..."):
    if sys.stdin and sys.stdin.isatty():
        try:
            input(message)
        except EOFError:
            pass


def init_scope():
    scope_key = get_scope_key()
    migrate_legacy_data(BASE_DIR, scope_key)
    os.environ["SCOPE_KEY"] = scope_key
    global DATA_DIR, CONFIG_PATH, PREV_CONFIG_PATH, BET_TYPE_STATS_PATH
    DATA_DIR = get_data_dir(BASE_DIR, scope_key)
    CONFIG_PATH = get_config_path(BASE_DIR, scope_key)
    PREV_CONFIG_PATH = DATA_DIR / "config_prev.json"
    BET_TYPE_STATS_PATH = DATA_DIR / "bet_type_stats.csv"
    return scope_key


def load_config():
    if not CONFIG_PATH.exists():
        stem = CONFIG_PATH.stem.replace("config_", "")
        scope = stem.split("_", 1)[0] if stem else "central_dirt"
        fallback = []
        if scope == "central_turf":
            fallback.extend(
                [
                    BASE_DIR / "config_turf_default.json",
                    BASE_DIR / "config_turf.json",
                ]
            )
        elif scope == "central_dirt":
            fallback.extend(
                [
                    BASE_DIR / "config_dirt_default.json",
                    BASE_DIR / "config_dirt.json",
                ]
            )
        elif scope == "local":
            fallback.extend(
                [
                    BASE_DIR / "config_central_dirt.json",
                    BASE_DIR / "config_dirt_default.json",
                    BASE_DIR / "config_dirt.json",
                ]
            )
        fallback.append(BASE_DIR / "config.json")
        for legacy in fallback:
            if legacy.exists():
                data = json.loads(legacy.read_text(encoding="utf-8"))
                CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return data
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_config(data):
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_csv_header(path, fieldnames):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = reader.fieldnames or []
        rows = list(reader)
    if existing == fieldnames:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_csv(path, fieldnames, row):
    ensure_csv_header(path, fieldnames)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def choose_param_group(cursor):
    groups = ["global", "hit_floors", "type_adjust"]
    return groups[int(cursor) % len(groups)]


def main():
    init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = load_csv(DATA_DIR / "results.csv")
    if not results:
        print("No results found.")
        pause_exit()
        return

    window = int(os.environ.get("BET_OPT_WINDOW", 10))
    min_samples = int(os.environ.get("BET_OPT_MIN_SAMPLES", 5))
    max_roi_std = float(os.environ.get("BET_OPT_MAX_ROI_STD", 0.8))
    target_budget = int(os.environ.get("BET_OPT_BUDGET", 2000))
    filtered = []
    for row in results:
        raw_budget = str(row.get("budget_yen", "")).strip()
        if not raw_budget:
            filtered.append(row)
            continue
        try:
            budget = int(float(raw_budget))
        except (TypeError, ValueError):
            continue
        if budget == target_budget:
            filtered.append(row)
    recent = filtered[-window:]
    recent_run_ids = [row.get("run_id") for row in recent if row.get("run_id")]

    # --- Compute ROI statistics (optionally stake-weighted) ---
    rois = []
    weighted_sum = 0.0
    stake_sum = 0.0
    stake_key = None

    # Try to detect stake field (total_bet / total_stake / stake).
    if recent:
        for cand in ("total_bet", "total_stake", "stake"):
            if cand in recent[0]:
                stake_key = cand
                break

    for row in recent:
        try:
            roi_val = float(row.get("roi", 0))
        except ValueError:
            continue
        rois.append(roi_val)

        # Use stake-weighted ROI when stake is available.
        if stake_key:
            try:
                stake = float(row.get(stake_key) or 0)
            except ValueError:
                stake = 0.0
            if stake > 0:
                weighted_sum += roi_val * stake
                stake_sum += stake

    if not rois:
        print("No valid ROI values.")
        pause_exit()
        return
    if len(rois) < min_samples:
        print(f"Not enough samples for update ({len(rois)}/{min_samples}).")
        pause_exit()
        return

    # Use stake-weighted ROI if possible.
    if stake_key and stake_sum > 0:
        avg_roi = weighted_sum / stake_sum
    else:
        avg_roi = sum(rois) / len(rois)

    # Volatility still uses simple (unweighted) std.
    roi_std = math.sqrt(sum((v - avg_roi) ** 2 for v in rois) / len(rois))

    # --- Volatility gate + step size ---
    gate_reason = []
    step_mult = 1.0
    volatility_freeze = False
    extreme_std = max_roi_std * 1.5

    if roi_std > extreme_std:
        # Freeze parameters under extreme volatility.
        print(
            f"ROI volatility EXTREME (std={roi_std:.4f} > {extreme_std:.4f}); "
            "freezing parameters for this run."
        )
        volatility_freeze = True
        gate_reason.append("volatility_extreme")
    elif roi_std > max_roi_std:
        print(
            f"ROI volatility high (std={roi_std:.4f} > {max_roi_std}); reducing step size."
        )
        step_mult *= 0.2
        # NOTE: Keep volatility from triggering a full freeze.

    cfg = load_config()
    prev_snapshot = json.loads(json.dumps(cfg))
    old_cfg = json.loads(json.dumps(cfg))
    state = dict(cfg.get("state", {}))
    tune_cursor = int(state.get("tune_cursor", 0))
    group = choose_param_group(tune_cursor)
    reason_parts = [f"group_{group}"]
    adjusted_types = []

    hit_weight = float(old_cfg.get("hit_weight", 1.3))
    payout_weight = float(old_cfg.get("payout_weight", 0.25))
    coverage = float(old_cfg.get("coverage_target", 0.65))
    hit_floors = dict(old_cfg.get("hit_floors", {}))
    type_adjust = dict(old_cfg.get("type_weight_adjust", {}))

    step_hw = 0.05 * step_mult
    step_cov = 0.05 * step_mult
    step_pw = 0.03 * step_mult
    step_floor = 0.03 * step_mult

    if avg_roi < 1.0:
        reason_parts.append("roi_low")
        if group == "global":
            hit_weight = clamp(hit_weight + step_hw, 0.8, 2.0)
            coverage = clamp(coverage + step_cov, 0.5, 0.9)
            payout_weight = clamp(payout_weight - step_pw, 0.1, 0.5)
        elif group == "hit_floors":
            for k in hit_floors:
                hit_floors[k] = clamp(float(hit_floors[k]) + step_floor, 0.2, 0.85)
    elif avg_roi > 1.1:
        reason_parts.append("roi_high")
        if group == "global":
            hit_weight = clamp(hit_weight - step_hw, 0.8, 2.0)
            coverage = clamp(coverage - step_cov, 0.5, 0.9)
            payout_weight = clamp(payout_weight + step_pw, 0.1, 0.5)
        elif group == "hit_floors":
            for k in hit_floors:
                hit_floors[k] = clamp(float(hit_floors[k]) - step_floor, 0.2, 0.85)
    else:
        reason_parts.append("roi_mid")

    next_hit_weight = round(hit_weight, 3)
    next_payout_weight = round(payout_weight, 3)
    next_coverage = round(coverage, 3)

    # Bet-type hit rate adjustments
    if group == "type_adjust" and not gate_reason:
        bet_stats = load_csv(BET_TYPE_STATS_PATH)
        base_targets = {
            "win": 0.35,
            "place": 0.45,
            "wide": 0.3,
            "quinella": 0.2,
        }
        target_bounds = {
            "win": (0.25, 0.55),
            "place": (0.35, 0.65),
            "wide": (0.2, 0.5),
            "quinella": (0.12, 0.35),
        }
        sample_ratio = min(1.0, len(rois) / float(window))
        roi_pressure = clamp(1.0 - avg_roi, -0.2, 0.3)
        vol_pressure = clamp(roi_std - 0.15, -0.05, 0.15)
        risk_pressure = roi_pressure + vol_pressure
        target_factor = clamp(1.0 + risk_pressure * 0.6, 0.85, 1.2)
        adjust_scale = clamp((0.03 + sample_ratio * 0.04) * step_mult, 0.01, 0.08)
        if risk_pressure >= 0:
            adjust_up = 1.0 + adjust_scale * 0.8
            adjust_down = 1.0 - adjust_scale * 1.2
        else:
            adjust_up = 1.0 + adjust_scale * 1.2
            adjust_down = 1.0 - adjust_scale * 0.8

        # Hit-rate safety margin.
        margin = clamp(0.05 + (1.0 - sample_ratio) * 0.03, 0.05, 0.08)

        # Default minimum samples.
        default_min_bets = 6 if sample_ratio >= 0.8 else 8
        per_type_min_bets = {
            "win": default_min_bets,
            "place": default_min_bets,
            "wide": max(default_min_bets + 2, 8),
            "quinella": max(default_min_bets + 4, 10),
        }

        targets = {}
        for bet_type, base in base_targets.items():
            lo, hi = target_bounds.get(bet_type, (0.05, 0.8))
            targets[bet_type] = clamp(base * target_factor, lo, hi)
            if bet_type not in type_adjust:
                type_adjust[bet_type] = 1.0

        if bet_stats and recent_run_ids:
            totals = {k: {"bets": 0, "hits": 0} for k in targets}
            for row in bet_stats:
                if row.get("run_id") not in recent_run_ids:
                    continue
                bet_type = str(row.get("bet_type", "")).strip().lower()
                if bet_type not in totals:
                    continue
                try:
                    totals[bet_type]["bets"] += int(float(row.get("bets", 0)))
                    totals[bet_type]["hits"] += int(float(row.get("hits", 0)))
                except ValueError:
                    continue

            for bet_type, data in totals.items():
                bets = data["bets"]
                hits = data["hits"]

                min_bets = per_type_min_bets.get(bet_type, default_min_bets)
                if bets < min_bets:
                    continue
                hit_rate = hits / bets if bets else 0
                current = float(type_adjust.get(bet_type, 1.0))
                updated = current
                if hit_rate < targets[bet_type] - margin:
                    updated = clamp(current * adjust_down, 0.5, 1.5)
                elif hit_rate > targets[bet_type] + margin:
                    updated = clamp(current * adjust_up, 0.5, 1.5)

                if updated != current and avg_roi > 1.05:
                    jitter = 1.0 + (random.random() - 0.5) * 0.02 * step_mult
                    updated = clamp(updated * jitter, 0.5, 1.5)

                if updated != current:
                    adjusted_types.append(bet_type)
                type_adjust[bet_type] = round(updated, 3)

        if adjusted_types:
            reason_parts.append("type_adjust")

    params_changed = (
        next_hit_weight != old_cfg.get("hit_weight")
        or next_payout_weight != old_cfg.get("payout_weight")
        or next_coverage != old_cfg.get("coverage_target")
        or hit_floors != old_cfg.get("hit_floors", {})
        or type_adjust != old_cfg.get("type_weight_adjust", {})
    )
    if gate_reason:
        action = "freeze"
    else:
        action = "update" if params_changed else "no_change"

    reasons = []
    reasons.extend(gate_reason)
    reasons.extend(reason_parts)
    reason = "+".join(reasons) if reasons else "none"

    if action == "update":
        next_cursor = tune_cursor + 1
        new_state = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "tune_cursor": next_cursor,
        }
        new_cfg = json.loads(json.dumps(old_cfg))
        new_cfg["state"] = new_state
        PREV_CONFIG_PATH.write_text(json.dumps(prev_snapshot, indent=2), encoding="utf-8")
        new_cfg["hit_weight"] = next_hit_weight
        new_cfg["payout_weight"] = next_payout_weight
        new_cfg["coverage_target"] = next_coverage
        new_cfg["hit_floors"] = hit_floors
        new_cfg["type_weight_adjust"] = type_adjust
        new_cfg["version"] = int(old_cfg.get("version", 1)) + 1
        cfg = new_cfg
        save_config(cfg)
    else:
        cfg = old_cfg

    history_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "reason": reason,
        "avg_roi": round(avg_roi, 4),
        "roi_std": round(roi_std, 4),
        "step_mult": round(step_mult, 3),
        "window": window,
        "roi_samples": len(rois),
        "tune_group": group,
        "adjusted_types": ",".join(adjusted_types),
        "hit_weight": cfg["hit_weight"],
        "payout_weight": cfg["payout_weight"],
        "coverage_target": cfg["coverage_target"],
        "hit_floors_json": json.dumps(cfg.get("hit_floors", {}), ensure_ascii=False),
        "type_adjust_json": json.dumps(cfg.get("type_weight_adjust", {}), ensure_ascii=False),
        "version": cfg["version"],
    }
    append_csv(DATA_DIR / "config_history.csv", list(history_row.keys()), history_row)

    if action == "update":
        print(f"Updated {CONFIG_PATH.name}")
    elif action == "freeze":
        print("Freeze: config not updated.")
    else:
        print("No parameter changes.")
    print(f"Avg ROI (last {len(rois)}): {avg_roi:.4f}")
    print(f"ROI std: {roi_std:.4f}")
    print(f"Reason: {reason}")
    if action == "update":
        changes = []
        if old_cfg.get("hit_weight") != cfg.get("hit_weight"):
            changes.append(
                f"hit_weight: {old_cfg.get('hit_weight')} -> {cfg.get('hit_weight')}"
            )
        if old_cfg.get("payout_weight") != cfg.get("payout_weight"):
            changes.append(
                f"payout_weight: {old_cfg.get('payout_weight')} -> {cfg.get('payout_weight')}"
            )
        if old_cfg.get("coverage_target") != cfg.get("coverage_target"):
            changes.append(
                f"coverage_target: {old_cfg.get('coverage_target')} -> {cfg.get('coverage_target')}"
            )
        old_floors = old_cfg.get("hit_floors", {})
        new_floors = cfg.get("hit_floors", {})
        for key in sorted(set(old_floors) | set(new_floors)):
            if old_floors.get(key) != new_floors.get(key):
                changes.append(
                    f"hit_floors.{key}: {old_floors.get(key)} -> {new_floors.get(key)}"
                )
        old_adjust = old_cfg.get("type_weight_adjust", {})
        new_adjust = cfg.get("type_weight_adjust", {})
        for key in sorted(set(old_adjust) | set(new_adjust)):
            if old_adjust.get(key) != new_adjust.get(key):
                changes.append(
                    f"type_weight_adjust.{key}: {old_adjust.get(key)} -> {new_adjust.get(key)}"
                )
        if changes:
            print("Changes:")
            for line in changes:
                print(f"- {line}")
    pause_exit()


if __name__ == "__main__":
    main()

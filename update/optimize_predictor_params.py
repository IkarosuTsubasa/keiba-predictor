import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from surface_scope import (
    get_data_dir,
    get_predictor_config_path,
    get_predictor_prev_path,
    get_scope_key,
    migrate_legacy_data,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = None
CONFIG_PATH = None
PREV_CONFIG_PATH = None
RESULTS_PATH = None
HISTORY_PATH = None


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
    global DATA_DIR, CONFIG_PATH, PREV_CONFIG_PATH, RESULTS_PATH, HISTORY_PATH
    DATA_DIR = get_data_dir(BASE_DIR, scope_key)
    CONFIG_PATH = get_predictor_config_path(BASE_DIR, scope_key)
    PREV_CONFIG_PATH = get_predictor_prev_path(BASE_DIR, scope_key)
    RESULTS_PATH = DATA_DIR / "predictor_results.csv"
    HISTORY_PATH = DATA_DIR / "predictor_config_history.csv"
    return scope_key


def load_config():
    if not CONFIG_PATH.exists():
        stem = CONFIG_PATH.stem.replace("predictor_config_", "")
        scope = stem.split("_", 1)[0] if stem else "central_dirt"
        fallback = []
        if scope == "central_turf":
            fallback.extend(
                [
                    BASE_DIR / "predictor_config_turf_default.json",
                    BASE_DIR / "predictor_config_turf.json",
                ]
            )
        elif scope == "central_dirt":
            fallback.extend(
                [
                    BASE_DIR / "predictor_config_dirt_default.json",
                    BASE_DIR / "predictor_config_dirt.json",
                ]
            )
        elif scope == "local":
            fallback.extend(
                [
                    BASE_DIR / "predictor_config_central_dirt.json",
                    BASE_DIR / "predictor_config_dirt_default.json",
                    BASE_DIR / "predictor_config_dirt.json",
                ]
            )
        fallback.append(BASE_DIR / "predictor_config.json")
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


def signal_dir(value, target, margin):
    if value < target - margin:
        return -1
    if value > target + margin:
        return 1
    return 0


def signal_strength(score, target, margin):
    # Strength is 0 near target, approaches 1 as distance grows.
    d = abs(score - target)
    if d <= margin:
        return 0.0
    # Keep scale fixed so strengths are comparable across signals.
    denom = max(1e-9, 0.4)
    return min(1.0, (d - margin) / denom)


def int_step(step_mult):
    # Keep integer params fixed when step_mult < 0.5.
    if step_mult < 0.5:
        return 0
    # Coordinate descent moves integer params by 1.
    return 1



_LAST_ADJUST_NO_OP = False

def adjust_params(params, rank_score, ev_score, risk_score, sample_ratio, group, step_mult):
    global _LAST_ADJUST_NO_OP
    _LAST_ADJUST_NO_OP = False
    # base step sizes (scaled by sample_ratio and step_mult)
    w_step = (0.03 + 0.02 * sample_ratio) * step_mult
    p_step = 0.05 * step_mult
    s_step = 0.10 * step_mult
    c_step = int_step(step_mult)

    # signals: direction (-1/0/+1) + strength (0..1)
    rank_dir = signal_dir(rank_score, 0.70, 0.05)
    ev_dir = signal_dir(ev_score, 0.50, 0.05)
    risk_dir = signal_dir(risk_score, 0.60, 0.05)

    rank_str = signal_strength(rank_score, 0.70, 0.05)
    ev_str = signal_strength(ev_score, 0.50, 0.05)
    risk_str = signal_strength(risk_score, 0.60, 0.05)

    # debug header
    print(
        "[debug] adjust_params "
        f"group={group} sample_ratio={sample_ratio:.2f} step_mult={step_mult:.2f} "
        f"w_step={w_step:.4f} p_step={p_step:.4f} s_step={s_step:.4f} c_step={c_step} "
        f"rank={rank_score:.4f} ev={ev_score:.4f} risk={risk_score:.4f} "
        f"dirs(r,e,k)=({rank_dir},{ev_dir},{risk_dir}) "
        f"str(r,e,k)=({rank_str:.2f},{ev_str:.2f},{risk_str:.2f})"
    )

    # group-specific update (coordinate descent)
    if group == "weights":
        # Risk-driven: unstable (risk_dir<0) => increase base, decrease match
        # Stable (risk_dir>0) => allow more match weight
        base_delta = ((-risk_dir) * risk_str + (-rank_dir) * 0.25 * rank_str) * w_step
        match_delta = -base_delta

        old_b = params["record_weight_base"]
        old_m = params["record_weight_match"]
        target_sum = old_b + old_m
        base_min = max(0.60, target_sum - 0.40)
        base_max = min(0.85, target_sum - 0.15)
        if base_min > base_max:
            base_min, base_max = 0.60, 0.85
        new_b_raw = clamp(old_b + base_delta, base_min, base_max)
        new_m_raw = target_sum - new_b_raw
        new_b = round(new_b_raw, 3)
        new_m = round(new_m_raw, 3)

        params["record_weight_base"] = new_b
        params["record_weight_match"] = new_m

        print(
            f"[debug] weights base {old_b:.3f}->{new_b:.3f} (d{base_delta:+.4f}), "
            f"match {old_m:.3f}->{new_m:.3f} (d{match_delta:+.4f})"
        )

    elif group == "recent_race_count":
        # Rank-driven only: avoid risk cancellation.
        # rank_dir == -1 and step_mult >= 0.5 => +1.
        if c_step == 0 or rank_dir == 0:
            _LAST_ADJUST_NO_OP = True
            print("[debug] recent_race_count no-op (c_step==0 or rank_dir==0)")
        else:
            old = params["recent_race_count"]
            delta = (-rank_dir) * c_step
            new = int(clamp(old + delta, 4, 7))
            params["recent_race_count"] = new
            print(f"[debug] recent_race_count {old}->{new} (d{delta:+d})")

    elif group == "top_score_count":
        # EV-driven: EV high => concentrate => fewer picks; EV low => spread => more picks
        # ev_dir == +1 and step_mult >= 0.5 => -1.
        if c_step == 0 or ev_dir == 0:
            _LAST_ADJUST_NO_OP = True
            print("[debug] top_score_count no-op (c_step==0 or ev_dir==0)")
        else:
            old = params["top_score_count"]
            delta = (-ev_dir) * c_step
            new = int(clamp(old + delta, 2, 4))
            params["top_score_count"] = new
            print(f"[debug] top_score_count {old}->{new} (d{delta:+d})")

    elif group == "smooth_p":
        # Risk-driven smoothing: unstable => increase smooth_p, stable => decrease slightly
        old = params["smooth_p"]
        if risk_dir == 0:
            _LAST_ADJUST_NO_OP = True
            print("[debug] smooth_p no-op (risk_dir==0)")
        else:
            delta = ((-risk_dir) * risk_str) * p_step
            proposed = old + delta
            clamped = clamp(proposed, 1.00, 1.60)
            new = round(clamped, 2)
            params["smooth_p"] = new
            print(
                f"[debug] smooth_p {old:.2f}->{new:.2f} "
                f"(proposed {proposed:.4f}, d{delta:+.4f})"
            )

    elif group == "top3_scale":
        # Calibration proxy: use EV as weak proxy until you add logloss/brier feedback
        old = params["top3_scale"]
        if ev_dir == 0:
            _LAST_ADJUST_NO_OP = True
            print("[debug] top3_scale no-op (ev_dir==0)")
        else:
            delta = (-ev_dir) * ev_str * s_step
            proposed = old + delta
            new = round(clamp(proposed, 2.40, 3.60), 2)
            params["top3_scale"] = new
            print(
                f"[debug] top3_scale {old:.2f}->{new:.2f} "
                f"(proposed {proposed:.4f}, d{delta:+.4f})"
            )

    # reasons (explainability)
    reasons = [f"group_{group}"] if group else []
    if rank_dir < 0:
        reasons.append("rank_low")
    elif rank_dir > 0:
        reasons.append("rank_high")
    if ev_dir < 0:
        reasons.append("ev_low")
    elif ev_dir > 0:
        reasons.append("ev_high")
    if risk_dir < 0:
        reasons.append("risk_high")
    elif risk_dir > 0:
        reasons.append("risk_low")

    return "+".join(reasons) if reasons else "none"


def choose_param_group(cursor):
    groups = ["smooth_p", "recent_race_count", "top_score_count", "top3_scale", "weights"]
    if not groups:
        return ""
    return groups[int(cursor) % len(groups)]


def main():
    init_scope()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = load_csv(RESULTS_PATH)
    if not results:
        print("No predictor results found.")
        pause_exit()
        return

    window = int(os.environ.get("PRED_OPT_WINDOW", 10))
    min_samples = int(os.environ.get("PRED_OPT_MIN_SAMPLES", 5))
    max_score_std = float(os.environ.get("PRED_OPT_MAX_STD", 1.0))
    recent = results[-window:]
    scores = []
    rank_scores = []
    ev_scores = []
    hit_rates = []
    for row in recent:
        try:
            rank_score = float(row.get("rank_score", 0))
        except ValueError:
            rank_score = 0.0
        try:
            ev_score = float(row.get("ev_score", 0))
        except ValueError:
            ev_score = 0.0
        try:
            hit_rate = float(row.get("hit_rate", 0))
        except ValueError:
            hit_rate = 0.0
        quality = 0.55 * rank_score + 0.35 * ev_score + 0.10 * hit_rate
        scores.append(quality)
        rank_scores.append(rank_score)
        ev_scores.append(ev_score)
        hit_rates.append(hit_rate)
    if not scores:
        print("No valid predictor scores.")
        pause_exit()
        return
    if len(scores) < min_samples:
        print(f"Not enough samples for update ({len(scores)}/{min_samples}).")
        pause_exit()
        return

    alpha = 0.3
    ema = None
    for value in scores:
        ema = value if ema is None else (alpha * value + (1 - alpha) * ema)
    avg_quality = ema if ema is not None else sum(scores) / len(scores)
    score_std = math.sqrt(sum((v - avg_quality) ** 2 for v in scores) / len(scores))
    sample_ratio = min(1.0, len(scores) / float(window))
    dd_limit = 0.8 - 0.2 * sample_ratio
    cum = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in scores:
        cum += value
        if cum > peak:
            peak = cum
        drawdown = peak - cum
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    max_streak = 0
    streak = 0
    for value in hit_rates:
        if value <= 0:
            streak += 1
        else:
            max_streak = max(max_streak, streak)
            streak = 0
    max_streak = max(max_streak, streak)

    gate_reason = []
    step_mult = 1.0
    if score_std > max_score_std:
        print(
            f"Score volatility high (std={score_std:.4f} > {max_score_std}). "
            "Reducing step size."
        )
        step_mult *= 0.5
        gate_reason.append("volatility_high")
    if max_drawdown > dd_limit:
        print(
            f"Score drawdown high (max_dd={max_drawdown:.4f} > {dd_limit:.4f}). "
            "Reducing step size."
        )
        step_mult *= 0.5
        gate_reason.append("drawdown_high")
    rank_ema = None
    for value in rank_scores:
        rank_ema = value if rank_ema is None else (alpha * value + (1 - alpha) * rank_ema)
    rank_ema = rank_ema if rank_ema is not None else 0.0

    ev_ema = None
    for value in ev_scores:
        ev_ema = value if ev_ema is None else (alpha * value + (1 - alpha) * ev_ema)
    ev_ema = ev_ema if ev_ema is not None else 0.0

    std_norm = clamp(score_std / max_score_std, 0.0, 1.0)
    dd_norm = clamp(max_drawdown / dd_limit, 0.0, 1.0) if dd_limit > 0 else 0.0
    streak_norm = clamp(max_streak / 5.0, 0.0, 1.0)
    risk_penalty = 0.4 * std_norm + 0.4 * dd_norm + 0.2 * streak_norm
    risk_score = clamp(1.0 - risk_penalty, 0.0, 1.0)

    config = load_config()
    prev_snapshot = json.loads(json.dumps(config))
    params = dict(config.get("params", {}))
    # Normalize types for stable comparisons.
    for key in ("record_weight_base", "record_weight_match", "top3_scale", "smooth_p"):
        if key in params:
            try:
                params[key] = float(params[key])
            except (TypeError, ValueError):
                pass
    for key in ("recent_race_count", "top_score_count"):
        if key in params:
            try:
                params[key] = int(params[key])
            except (TypeError, ValueError):
                pass
    old_params = dict(params)
    state = dict(config.get("state", {}))
    tune_cursor = int(state.get("tune_cursor", 0))
    group = choose_param_group(tune_cursor)
    adjust_no_op = False

    risk_slow_threshold = float(os.environ.get("PRED_OPT_RISK_SLOW", 0.4))
    risk_freeze_threshold = float(os.environ.get("PRED_OPT_RISK_FREEZE", 0.2))

    if risk_score < risk_freeze_threshold:
        action = "freeze"
        reasons = list(gate_reason)
        reasons.append("risk_score_very_low")
        reason = "+".join(reasons)
    else:
        if risk_score < risk_slow_threshold:
            print(
                f"Risk score low ({risk_score:.4f} < {risk_slow_threshold}). "
                "Reducing step size further."
            )
            step_mult *= 0.3
            gate_reason.append("risk_score_low")

        adjust_reason = adjust_params(
            params,
            rank_ema,
            ev_ema,
            risk_score,
            sample_ratio,
            group,
            step_mult,
        )
        adjust_no_op = _LAST_ADJUST_NO_OP
        reasons = []
        reasons.extend(gate_reason)
        if adjust_reason:
            reasons.append(adjust_reason)
        reason = "+".join(reasons) if reasons else "none"
        action = "update" if params != old_params else "no_change"

    advance_cursor = action == "update" or (action == "no_change" and adjust_no_op)
    next_cursor = tune_cursor + 1 if advance_cursor else tune_cursor
    new_state = {
        "rank_ema": round(rank_ema, 4),
        "ev_ema": round(ev_ema, 4),
        "risk_score": round(risk_score, 4),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "tune_cursor": next_cursor,
    }

    if action == "update":
        PREV_CONFIG_PATH.write_text(json.dumps(prev_snapshot, indent=2), encoding="utf-8")
        config["params"] = params
        config["version"] = int(config.get("version", 1)) + 1
        config["state"] = new_state
        save_config(config)
    elif action == "no_change" and adjust_no_op:
        config["state"] = new_state
        save_config(config)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "avg_score": round(avg_quality, 4),
        "avg_quality": round(avg_quality, 4),
        "action": action,
        "reason": reason,
        "group": group,
        "sample_ratio": round(sample_ratio, 4),
        "step_mult": round(step_mult, 4),
        "ema_score": round(avg_quality, 4),
        "ema_quality": round(avg_quality, 4),
        "rank_ema": round(rank_ema, 4),
        "ev_ema": round(ev_ema, 4),
        "risk_score": round(risk_score, 4),
        "score_std": round(score_std, 4),
        "max_drawdown": round(max_drawdown, 4),
        "record_weight_base": params.get("record_weight_base"),
        "record_weight_match": params.get("record_weight_match"),
        "recent_race_count": params.get("recent_race_count"),
        "top_score_count": params.get("top_score_count"),
        "smooth_p": params.get("smooth_p"),
        "top3_scale": params.get("top3_scale"),
    }
    append_csv(HISTORY_PATH, list(row.keys()), row)
    if action == "update":
        print("Updated predictor params.")
    elif action == "freeze":
        print("Freeze: predictor params not updated.")
    else:
        print("No parameter changes.")
    print(f"Reason: {reason}")
    print(f"EMA quality (last {len(scores)}): {avg_quality:.4f}")
    print(f"Rank EMA: {rank_ema:.4f} | EV EMA: {ev_ema:.4f} | Risk score: {risk_score:.4f}")
    print(f"Score std: {score_std:.4f}")
    print(f"Max drawdown: {max_drawdown:.4f}")
    if action == "update":
        changes = []
        keys = [
            "record_weight_base",
            "record_weight_match",
            "recent_race_count",
            "top_score_count",
            "smooth_p",
            "top3_scale",
        ]
        for key in keys:
            if old_params.get(key) != params.get(key):
                changes.append(f"{key}: {old_params.get(key)} -> {params.get(key)}")
        if changes:
            print("Changes:")
            for line in changes:
                print(f"- {line}")
    pause_exit()


if __name__ == "__main__":
    main()

import json
import os


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    os.environ.setdefault("SCOPE_KEY", "local")

    from bet_plan_update import (  # noqa: WPS433
        apply_bet_profile_to_v3,
        build_bet_engine_v3_audit_summary,
        get_bet_engine_v3_config,
        load_predictor_config,
    )

    predictor_cfg = load_predictor_config()
    merged_cfg = get_bet_engine_v3_config(predictor_cfg)
    final_cfg = apply_bet_profile_to_v3(merged_cfg, "publish")

    expected = {
        "kelly_scale": 1.0,
        "min_p_hit_per_ticket": 0.04,
        "min_p_win_per_ticket": 0.03,
        "min_edge_per_ticket": 0.00,
        "fallback_max_odds_place": 15.0,
    }
    for key, val in expected.items():
        got = _as_float(final_cfg.get(key))
        assert abs(got - float(val)) < 1e-9, f"{key} expected={val} got={final_cfg.get(key)}"

    summary = build_bet_engine_v3_audit_summary(final_cfg)
    print("OK: publish profile merged config")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


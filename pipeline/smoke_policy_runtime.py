import os
import time

from llm.policy_runtime import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROK_MODEL,
    DEFAULT_SILICONFLOW_MODEL,
    RacePolicyInput,
    call_policy,
    get_last_call_meta,
    get_policy_cache_key,
)


def _build_input():
    race_id = f"SMOKE_POLICY_{int(time.time())}"
    payload = {
        "race_id": race_id,
        "scope_key": "central_dirt",
        "field_size": 8,
        "ai": {
            "gap": 0.041,
            "confidence_score": 0.58,
            "stability_score": 0.54,
            "risk_score": 0.49,
        },
        "marks_top5": [
            {"horse_no": "1", "horse_name": "A", "pred_rank": 1, "top3_prob_model": 0.42, "rank_score_norm": 1.0},
            {"horse_no": "2", "horse_name": "B", "pred_rank": 2, "top3_prob_model": 0.35, "rank_score_norm": 0.83},
        ],
        "predictions": [
            {
                "horse_no": "1",
                "horse_name": "A",
                "pred_rank": 1,
                "top3_prob_model": 0.42,
                "rank_score_norm": 1.0,
                "win_odds": 3.4,
                "place_odds": 1.8,
            },
            {
                "horse_no": "2",
                "horse_name": "B",
                "pred_rank": 2,
                "top3_prob_model": 0.35,
                "rank_score_norm": 0.83,
                "win_odds": 5.6,
                "place_odds": 2.2,
            },
        ],
        "predictions_full": [
            {"horse_no": "1", "HorseName": "A", "pred_rank": 1, "Top3Prob_model": 0.42, "rank_score": 0.92},
            {"horse_no": "2", "HorseName": "B", "pred_rank": 2, "Top3Prob_model": 0.35, "rank_score": 0.78},
        ],
        "pair_odds_top": [{"bet_type": "wide", "pair": "1-2", "odds": 6.5}],
        "odds_full": {
            "win": [{"horse_no": "1", "odds": 3.4}, {"horse_no": "2", "odds": 5.6}],
            "place": [{"horse_no": "1", "odds": 1.8}, {"horse_no": "2", "odds": 2.2}],
            "wide": [{"pair": "1-2", "horse_no_a": "1", "horse_no_b": "2", "odds": 6.5}],
            "quinella": [{"pair": "1-2", "horse_no_a": "1", "horse_no_b": "2", "odds": 11.4}],
            "exacta": [{"pair": "1-2", "horse_no_a": "1", "horse_no_b": "2", "odds": 17.8}],
            "trio": [{"triple": "1-2-3", "horse_no_a": "1", "horse_no_b": "2", "horse_no_c": "3", "odds": 39.2}],
            "trifecta": [{"triple": "1-2-3", "horse_no_a": "1", "horse_no_b": "2", "horse_no_c": "3", "odds": 121.0}],
        },
        "prediction_field_guide": {"Top3Prob_model": "3着内確率", "rank_score": "順位スコア"},
        "multi_predictor": {},
        "portfolio_history": {},
        "candidates": [
            {"id": "place:1", "bet_type": "place", "legs": ["1"], "odds_used": 1.8, "p_hit": 0.42, "ev": -0.244, "score": -0.158},
            {"id": "place:2", "bet_type": "place", "legs": ["2"], "odds_used": 2.2, "p_hit": 0.35, "ev": -0.23, "score": -0.136},
            {"id": "wide:1-2", "bet_type": "wide", "legs": ["1", "2"], "odds_used": 6.5, "p_hit": 0.18, "ev": 0.17, "score": 0.072},
            {"id": "exacta:1-2", "bet_type": "exacta", "legs": ["1", "2"], "odds_used": 17.8, "p_hit": 0.07, "ev": 0.246, "score": 0.065},
            {"id": "trio:1-2-3", "bet_type": "trio", "legs": ["1", "2", "3"], "odds_used": 39.2, "p_hit": 0.03, "ev": 0.176, "score": 0.03},
        ],
        "constraints": {
            "bankroll_yen": 10000,
            "race_budget_yen": 2400,
            "max_tickets_per_race": 3,
            "high_odds_threshold": 10.0,
            "allowed_types": ["place", "wide", "exacta", "trio", "trifecta"],
        },
    }
    return RacePolicyInput(**payload)


def _assert_output(output, meta, expected_engine):
    assert output is not None
    assert str(output.bet_decision) in ("bet", "no_bet")
    assert str(output.participation_level) in ("no_bet", "small_bet", "normal_bet")
    assert str(meta.get("policy_engine", "")) == expected_engine
    assert str(meta.get("policy_model", "")).strip()


def main():
    policy_input = _build_input()

    os.environ["GEMINI_POLICY_MOCK"] = "1"
    os.environ["SILICONFLOW_POLICY_MOCK"] = "1"
    os.environ["GROK_POLICY_MOCK"] = "1"
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("SILICONFLOW_API_KEY", None)
    os.environ.pop("XAI_API_KEY", None)

    gemini_key = get_policy_cache_key(policy_input, policy_engine="gemini", model=DEFAULT_GEMINI_MODEL)
    siliconflow_key = get_policy_cache_key(
        policy_input,
        policy_engine="siliconflow",
        model=DEFAULT_SILICONFLOW_MODEL,
    )
    grok_key = get_policy_cache_key(
        policy_input,
        policy_engine="grok",
        model=DEFAULT_GROK_MODEL,
    )
    assert gemini_key != siliconflow_key
    assert grok_key != siliconflow_key

    gemini_output = call_policy(
        input=policy_input,
        policy_engine="gemini",
        model=DEFAULT_GEMINI_MODEL,
        timeout_s=5,
        cache_enable=True,
    )
    gemini_meta = get_last_call_meta()
    _assert_output(gemini_output, gemini_meta, "gemini")

    siliconflow_output = call_policy(
        input=policy_input,
        policy_engine="siliconflow",
        model=DEFAULT_SILICONFLOW_MODEL,
        timeout_s=5,
        cache_enable=True,
    )
    siliconflow_meta = get_last_call_meta()
    _assert_output(siliconflow_output, siliconflow_meta, "siliconflow")

    grok_output = call_policy(
        input=policy_input,
        policy_engine="grok",
        model=DEFAULT_GROK_MODEL,
        timeout_s=5,
        cache_enable=True,
    )
    grok_meta = get_last_call_meta()
    _assert_output(grok_output, grok_meta, "grok")

    print("OK: smoke_policy_runtime passed")
    print(
        "gemini_cache={g} siliconflow_cache={s} grok_cache={k} gemini_style={gs} siliconflow_style={ss} grok_style={ks}".format(
            g=gemini_key[:12],
            s=siliconflow_key[:12],
            k=grok_key[:12],
            gs=str(gemini_output.buy_style),
            ss=str(siliconflow_output.buy_style),
            ks=str(grok_output.buy_style),
        )
    )


if __name__ == "__main__":
    main()

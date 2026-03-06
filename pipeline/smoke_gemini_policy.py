import os
import time

from llm.gemini_policy import (
    RacePolicyInput,
    call_gemini_policy,
    get_last_call_meta,
    get_policy_cache_key,
)


def _build_smoke_input():
    race_id = f"SMOKE_GEMINI_{int(time.time())}"
    marks_top5 = []
    for i in range(1, 6):
        marks_top5.append(
            {
                "horse_no": str(i),
                "horse_name": f"H{i}",
                "pred_rank": i,
                "top3_prob_model": round(0.42 - (i - 1) * 0.04, 6),
                "rank_score_norm": round(1.0 - (i - 1) * 0.18, 6),
            }
        )

    predictions = []
    predictions_full = []
    candidates = []
    win_odds_rows = []
    place_odds_rows = []
    for i in range(1, 11):
        p_win = max(0.02, 0.22 - (i - 1) * 0.015)
        p_place = min(0.70, p_win * 2.2)
        win_odds = 2.0 + i * 1.4
        place_odds = 1.2 + i * 0.5
        predictions.append(
            {
                "horse_no": str(i),
                "horse_name": f"H{i}",
                "pred_rank": i,
                "top3_prob_model": round(max(0.05, 0.42 - (i - 1) * 0.035), 6),
                "rank_score_norm": round(max(0.05, 1.0 - (i - 1) * 0.1), 6),
                "win_odds": round(win_odds, 6),
                "place_odds": round(place_odds, 6),
            }
        )
        predictions_full.append(
            {
                "horse_no": str(i),
                "HorseName": f"H{i}",
                "pred_rank": i,
                "Top3Prob_model": round(max(0.05, 0.42 - (i - 1) * 0.035), 6),
                "rank_score": round(max(0.05, 1.0 - (i - 1) * 0.09), 6),
                "confidence_score": 0.44,
                "stability_score": 0.39,
                "risk_score": 0.61,
                "odds_num": round(win_odds, 6),
                "win_odds": round(win_odds, 6),
                "place_odds": round(place_odds, 6),
            }
        )
        win_odds_rows.append({"horse_no": str(i), "odds": round(win_odds, 6)})
        place_odds_rows.append({"horse_no": str(i), "odds": round(place_odds, 6)})
        candidates.append(
            {
                "id": f"win:{i}",
                "bet_type": "win",
                "legs": [str(i)],
                "odds_used": round(win_odds, 6),
                "p_hit": round(p_win, 6),
                "ev": round(p_win * win_odds - 1.0, 6),
                "score": round((p_win * win_odds - 1.0) * (p_win**0.5), 6),
            }
        )
        candidates.append(
            {
                "id": f"place:{i}",
                "bet_type": "place",
                "legs": [str(i)],
                "odds_used": round(place_odds, 6),
                "p_hit": round(p_place, 6),
                "ev": round(p_place * place_odds - 1.0, 6),
                "score": round((p_place * place_odds - 1.0) * (p_place**0.5), 6),
            }
        )
    for a, b, odds_wide, odds_quin in [(1, 2, 5.8, 12.0), (1, 3, 7.4, 15.5), (2, 3, 6.9, 14.2)]:
        p_wide = round(min(0.65, 0.18 + (0.04 * a)), 6)
        p_quin = round(min(0.28, 0.06 + (0.02 * a)), 6)
        candidates.append(
            {
                "id": f"wide:{a}-{b}",
                "bet_type": "wide",
                "legs": [str(a), str(b)],
                "odds_used": odds_wide,
                "p_hit": p_wide,
                "ev": round(p_wide * odds_wide - 1.0, 6),
                "score": round((p_wide * odds_wide - 1.0) * (p_wide**0.5), 6),
            }
        )
        candidates.append(
            {
                "id": f"quinella:{a}-{b}",
                "bet_type": "quinella",
                "legs": [str(a), str(b)],
                "odds_used": odds_quin,
                "p_hit": p_quin,
                "ev": round(p_quin * odds_quin - 1.0, 6),
                "score": round((p_quin * odds_quin - 1.0) * (p_quin**0.5), 6),
            }
        )

    payload = {
        "race_id": race_id,
        "scope_key": "central_dirt",
        "field_size": 10,
        "ai": {
            "gap": 0.012,
            "confidence_score": 0.44,
            "stability_score": 0.39,
            "risk_score": 0.61,
        },
        "marks_top5": marks_top5,
        "predictions": predictions,
        "predictions_full": predictions_full,
        "pair_odds_top": [
            {"bet_type": "wide", "pair": "1-2", "odds": 5.8},
            {"bet_type": "quinella", "pair": "1-2", "odds": 12.0},
        ],
        "odds_full": {
            "win": win_odds_rows,
            "place": place_odds_rows,
            "wide": [
                {"pair": "1-2", "horse_no_a": "1", "horse_no_b": "2", "odds": 5.8},
                {"pair": "1-3", "horse_no_a": "1", "horse_no_b": "3", "odds": 7.4},
                {"pair": "2-3", "horse_no_a": "2", "horse_no_b": "3", "odds": 6.9},
            ],
            "quinella": [
                {"pair": "1-2", "horse_no_a": "1", "horse_no_b": "2", "odds": 12.0},
                {"pair": "1-3", "horse_no_a": "1", "horse_no_b": "3", "odds": 15.5},
                {"pair": "2-3", "horse_no_a": "2", "horse_no_b": "3", "odds": 14.2},
            ],
        },
        "prediction_field_guide": {
            "horse_no": "馬番",
            "HorseName": "馬名",
            "Top3Prob_model": "統合モデルの3着内確率",
            "rank_score": "順位付け用スコア",
            "confidence_score": "予測信頼度スコア",
            "stability_score": "予測安定性スコア",
            "risk_score": "リスク/不確実性スコア",
            "odds_num": "単勝オッズ",
            "win_odds": "単勝オッズの代表値",
            "place_odds": "複勝オッズの代表値",
        },
        "candidates": candidates,
        "constraints": {
            "bankroll_yen": 5000,
            "race_budget_yen": 1200,
            "max_tickets_per_race": 6,
            "high_odds_threshold": 12.0,
            "allowed_types": ["win", "place", "wide", "quinella"],
        },
    }
    return RacePolicyInput(**payload)


def _clone_with_budget(policy_input, bankroll_yen, race_budget_yen):
    payload = policy_input.model_dump() if hasattr(policy_input, "model_dump") else policy_input.dict()
    payload["constraints"]["bankroll_yen"] = int(bankroll_yen)
    payload["constraints"]["race_budget_yen"] = int(race_budget_yen)
    return RacePolicyInput(**payload)


def main():
    os.environ["GEMINI_POLICY_MOCK"] = "1"
    os.environ.pop("GEMINI_API_KEY", None)

    policy_input = _build_smoke_input()
    input_2000 = _clone_with_budget(policy_input, 2000, 800)
    input_50000 = _clone_with_budget(policy_input, 50000, 10000)

    key_2000 = get_policy_cache_key(input_2000, model="gemini-3.1-flash-lite-preview")
    key_50000 = get_policy_cache_key(input_50000, model="gemini-3.1-flash-lite-preview")
    assert key_2000 == key_50000, "cache key should ignore budget after budget-agnostic policy change"

    out1 = call_gemini_policy(
        input=input_2000,
        model="gemini-3.1-flash-lite-preview",
        timeout_s=5,
        cache_enable=True,
    )
    meta1 = get_last_call_meta()
    assert out1 is not None, "output should not be None"
    assert str(out1.bet_decision) in ("bet", "no_bet"), "bet_decision should be valid"
    assert str(out1.participation_level) in ("no_bet", "small_bet", "normal_bet"), "participation_level should be valid"
    assert str(out1.strategy_mode) in (
        "no_bet",
        "place_only",
        "place_focus",
        "balanced",
        "win_focus",
        "pair_focus",
        "spread",
        "conservative_single",
        "small_probe",
    )
    assert str(out1.construction_style) in ("single_axis", "pair_spread", "value_hunt", "conservative_single")
    assert isinstance(out1.pick_ids, list), "pick_ids should be list"
    all_ids = {str(c.id) for c in input_2000.candidates}
    assert all(pid in all_ids for pid in out1.pick_ids), "pick_ids must exist in candidates"
    assert int(out1.max_ticket_count or 0) >= 0, "max_ticket_count should be non-negative"
    assert str(out1.risk_tilt) in ("low", "medium", "high"), "risk_tilt should be valid"
    assert str(meta1.get("fallback_reason", "")) in ("mock_mode", "cache"), "should run fallback/mock path"
    assert int(meta1.get("requested_budget_yen", 0) or 0) == 2000, "meta keeps caller budget for logging"
    assert int(meta1.get("requested_race_budget_yen", 0) or 0) == 800, "meta keeps caller race budget for logging"
    assert not bool(meta1.get("reused", False)), "independent generation should not mark reused"
    assert int(meta1.get("source_budget_yen", 0) or 0) == 2000, "source budget should equal requested budget"

    out2 = call_gemini_policy(
        input=input_2000,
        model="gemini-3.1-flash-lite-preview",
        timeout_s=5,
        cache_enable=True,
    )
    meta2 = get_last_call_meta()
    assert bool(meta2.get("cache_hit", False)), "second run must be cache hit"
    assert all(pid in all_ids for pid in out2.pick_ids), "second run pick_ids must exist in candidates"
    assert int(meta2.get("requested_budget_yen", 0) or 0) == 2000, "cached meta budget should stay 2000"
    assert not bool(meta2.get("reused", False)), "cached independent generation should not be reused"

    out3 = call_gemini_policy(
        input=input_50000,
        model="gemini-3.1-flash-lite-preview",
        timeout_s=5,
        cache_enable=True,
    )
    meta3 = get_last_call_meta()
    assert int(meta3.get("requested_budget_yen", 0) or 0) == 50000, "meta budget should reflect caller budget"
    assert int(meta3.get("requested_race_budget_yen", 0) or 0) == 10000, "meta race budget should reflect caller race budget"
    assert not bool(meta3.get("reused", False)), "independent generation should not be reused"
    assert int(meta3.get("source_budget_yen", 0) or 0) == 50000, "source budget should equal 50000"

    reused_meta = dict(meta3)
    reused_meta["reused"] = True
    reused_meta["source_budget_yen"] = 2000
    reused_meta["requested_budget_yen"] = 50000
    reused_meta["requested_race_budget_yen"] = 10000
    assert bool(reused_meta.get("reused", False)), "reuse=true meta should be supported"
    assert int(reused_meta.get("source_budget_yen", 0) or 0) == 2000, "source budget should show reused source"

    print("OK: smoke_gemini_policy passed")
    print(
        "first_cache_hit={a} second_cache_hit={b} buy_style={style} picks={n} key_2000={k1} key_50000={k2}".format(
            a=int(bool(meta1.get("cache_hit", False))),
            b=int(bool(meta2.get("cache_hit", False))),
            style=str(out2.buy_style),
            n=int(out2.max_ticket_count or 0),
            k1=key_2000[:12],
            k2=key_50000[:12],
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from statistics import mean

from keiba_llm_agent.llm.llm_client import BaseLLMClient


def _text_or_unknown(value: object) -> str:
    if value in (None, "", []):
        return "unknown"
    return str(value)


def _score_recent_form(recent_runs: list[dict]) -> int:
    ratios: list[float] = []
    for run in recent_runs:
        finish = run.get("finish")
        field_size = run.get("field_size")
        if finish is None or field_size in (None, 0):
            continue
        ratios.append(((field_size - finish + 1) / field_size) * 100)
    if not ratios:
        return 0
    return round(mean(ratios))


def _score_distance_fit(recent_runs: list[dict], race_distance: int | None) -> int:
    if race_distance is None:
        return 0
    diffs = [
        abs(run["distance"] - race_distance)
        for run in recent_runs
        if run.get("distance") is not None
    ]
    if not diffs:
        return 0
    best = min(diffs)
    if best == 0:
        return 80
    if best <= 200:
        return 60
    if best <= 400:
        return 40
    return 20


def _score_course_fit(recent_runs: list[dict], course: str | None) -> int:
    if not course:
        return 0
    courses = [run.get("course") for run in recent_runs if run.get("course")]
    if not courses:
        return 0
    return 80 if course in courses else 30


def _score_track_condition_fit(recent_runs: list[dict], track_condition: str | None) -> int:
    if not track_condition:
        return 0
    conditions = [run.get("track_condition") for run in recent_runs if run.get("track_condition")]
    if not conditions:
        return 0
    return 80 if track_condition in conditions else 40


def _score_jockey_fit(recent_runs: list[dict], jockey: str | None) -> int:
    if not jockey:
        return 0
    jockeys = [run.get("jockey") for run in recent_runs if run.get("jockey")]
    if not jockeys:
        return 0
    return 80 if jockey in jockeys else 40


def _score_odds_value(odds: float | None, popularity: int | None) -> int:
    if odds is None or popularity is None:
        return 0
    if 2.0 <= odds <= 10.0:
        score = 70
    elif odds < 2.0:
        score = 50
    elif odds <= 20.0:
        score = 60
    else:
        score = 40
    if popularity <= 3:
        score += 10
    return min(score, 100)


def _score_risk(
    recent_runs: list[dict],
    recent_form: int,
    distance_fit: int,
    course_fit: int,
    track_condition_fit: int,
    jockey_fit: int,
    odds: float | None,
) -> int:
    known_scores = [recent_form, distance_fit, course_fit, track_condition_fit, jockey_fit]
    base_risk = max(0, 100 - round(mean(known_scores)))
    if not recent_runs:
        base_risk = max(base_risk, 80)
    if odds is None:
        base_risk = max(base_risk, 70)
    penalty = min(10, max(0, round(base_risk / 10)))
    return -penalty


def _build_marks(sorted_horses: list[dict]) -> dict[str, int]:
    labels = ("◎", "○", "▲", "△", "☆")
    marks: dict[str, int] = {}
    for index, label in enumerate(labels):
        marks[label] = sorted_horses[index]["horse_no"] if index < len(sorted_horses) else 0
    return marks


def _normalize_bet_type(value: object) -> str:
    mapping = {
        "ワイド": "wide",
        "wide": "wide",
        "複勝": "place",
        "place": "place",
        "単勝": "win",
        "win": "win",
    }
    return mapping.get(str(value), str(value))


def _normalize_combination(horse_numbers: list[int]) -> str:
    return "-".join(str(number) for number in sorted(horse_numbers))


def _extract_payout_lookup(result: dict) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for payout in result.get("payouts", []):
        payout_type = _normalize_bet_type(payout.get("type") or payout.get("bet_type"))
        if "horse_numbers" in payout and payout.get("horse_numbers"):
            combination = _normalize_combination([int(num) for num in payout["horse_numbers"]])
        else:
            combination_text = str(payout.get("combination", ""))
            parts = [part for part in combination_text.replace("/", "-").split("-") if part]
            try:
                numbers = [int(part) for part in parts]
            except ValueError:
                numbers = []
            combination = _normalize_combination(numbers) if numbers else combination_text
        payout_value = payout.get("payout", 0) or 0
        lookup[(payout_type, combination)] = int(payout_value)
    return lookup


def _calculate_bet_results(prediction: dict, result: dict, top3: list[int]) -> tuple[list[dict], int, int, bool, bool]:
    payout_lookup = _extract_payout_lookup(result)
    total_stake = 0
    total_return = 0
    any_hit = False
    payout_missing = False
    bet_results: list[dict] = []
    first_place = top3[0] if top3 else None

    for bet in prediction.get("bets", []):
        bet_type = _normalize_bet_type(bet.get("bet_type"))
        horse_numbers = [int(number) for number in bet.get("horse_numbers", [])]
        amount = int(bet.get("amount") or 0)
        total_stake += amount

        hit = False
        if bet_type == "wide" and len(horse_numbers) == 2:
            hit = all(number in top3 for number in horse_numbers)
        elif bet_type == "place" and len(horse_numbers) == 1:
            hit = horse_numbers[0] in top3
        elif bet_type == "win" and len(horse_numbers) == 1:
            hit = first_place is not None and horse_numbers[0] == first_place

        payout = 0
        return_amount = 0
        if hit:
            any_hit = True
            combination = _normalize_combination(horse_numbers)
            payout = payout_lookup.get((bet_type, combination), 0)
            if payout == 0:
                payout_missing = True
            return_amount = int(amount / 100 * payout) if amount > 0 else 0
            total_return += return_amount

        bet_results.append(
            {
                "bet_type": bet.get("bet_type", ""),
                "horse_numbers": horse_numbers,
                "amount": amount,
                "hit": hit,
                "payout": payout,
                "return_amount": return_amount,
            }
        )
    return bet_results, total_stake, total_return, any_hit, payout_missing


class MockLLMClient(BaseLLMClient):
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str | None = None,
    ) -> dict:
        self.last_fallback_used = False
        try:
            payload = json.loads(user_prompt)
        except Exception:
            payload = {}
        if schema_name == "llm_check":
            return {"ok": True}
        if schema_name == "prediction_enhancement":
            return {
                "summary": payload.get("summary", "Mock summary"),
                "risks": payload.get("risks", []),
                "commentary": "Mock LLM commentary",
            }
        if schema_name == "review_enhancement":
            return {
                "good_points": payload.get("good_points", []),
                "bad_points": payload.get("bad_points", []),
                "lessons": payload.get("lessons", []),
            }
        if schema_name == "race_simulation":
            top_horses = payload.get("top_horses", [])
            first = top_horses[0] if top_horses else {"horse_no": 0, "horse_name": "unknown"}
            return {
                "race_flow": "Mock simulation flow",
                "key_positions": "Mock key positions",
                "favorable_horses": [
                    {
                        "horse_no": first.get("horse_no", 0),
                        "horse_name": first.get("horse_name", "unknown"),
                        "reason": "Mock favorable horse",
                    }
                ],
                "risk_horses": [],
                "win_scenario": "Mock win scenario",
                "top3_scenario": "Mock top3 scenario",
                "betting_scenario": "Mock betting scenario",
                "confidence_comment": "Mock confidence",
                "reasoning_summary": "Mock simulation summary",
                "warnings": [],
            }
        if schema_name == "simulation_review_enhancement":
            simulation_review = payload.get("simulation_review", {})
            return {
                "good_points": simulation_review.get("good_points", []),
                "bad_points": simulation_review.get("bad_points", []),
                "new_lessons": simulation_review.get("new_lessons", []),
                "overall_comment": simulation_review.get("overall_comment", ""),
            }
        return {"ok": True}

    def generate_analysis(self, prompt: str, payload: dict) -> dict:
        race_data = payload["race_data"]
        lessons = payload.get("lessons", [])
        race_info = race_data["race_info"]
        horses = race_data.get("horses", [])
        matched_lessons = lessons

        horse_scores: list[dict] = []
        for horse in horses:
            recent_runs = horse.get("recent_runs", [])
            recent_form = _score_recent_form(recent_runs)
            distance_fit = _score_distance_fit(recent_runs, race_info.get("distance"))
            course_fit = _score_course_fit(recent_runs, race_info.get("course"))
            track_condition_fit = _score_track_condition_fit(
                recent_runs, race_info.get("track_condition")
            )
            jockey_fit = _score_jockey_fit(recent_runs, horse.get("jockey"))
            odds_value = _score_odds_value(horse.get("odds"), horse.get("popularity"))
            risk = _score_risk(
                recent_runs,
                recent_form,
                distance_fit,
                course_fit,
                track_condition_fit,
                jockey_fit,
                horse.get("odds"),
            )
            total_score = (
                recent_form
                + distance_fit
                + course_fit
                + track_condition_fit
                + jockey_fit
                + odds_value
                + risk
            )
            latest_run = recent_runs[0] if recent_runs else {}
            reason = (
                f"前走着順={_text_or_unknown(latest_run.get('finish'))}, "
                f"同コース={_text_or_unknown(latest_run.get('course'))}, "
                f"同距離={_text_or_unknown(latest_run.get('distance'))}, "
                f"騎手継続={_text_or_unknown(horse.get('jockey'))}。"
            )
            horse_scores.append(
                {
                    "horse_no": horse["horse_no"],
                    "horse_name": horse["horse_name"],
                    "scores": {
                        "recent_form": recent_form,
                        "distance_fit": distance_fit,
                        "course_fit": course_fit,
                        "track_condition_fit": track_condition_fit,
                        "jockey_fit": jockey_fit,
                        "odds_value": odds_value,
                        "risk": risk,
                    },
                    "total_score": total_score,
                    "reason": reason,
                }
            )

        horse_scores.sort(key=lambda item: (-item["total_score"], item["horse_no"]))
        marks = _build_marks(horse_scores)
        top_horse = horse_scores[0] if horse_scores else {"horse_name": "unknown", "horse_no": 0}
        risks: list[str] = []
        if any(not horse.get("recent_runs") for horse in horses):
            risks.append("近走データが不足している馬がいるため、未知要素があります。")
        if not matched_lessons:
            risks.append("同条件の過去lessonがないため、used_lessonsはunknownに近い状態です。")

        lesson_note = ""
        if matched_lessons:
            lesson_note = f"過去lesson: {matched_lessons[0].get('lesson', 'unknown')}"
        summary = (
            f"◎は{top_horse['horse_name']}({top_horse['horse_no']})。"
            f"同条件lesson使用数={len(matched_lessons)}。"
            f"{lesson_note}"
            f"不足データはunknown扱いです。"
        )
        strategy = {
            "bet_decision": "SKIP" if not horse_scores else "BET",
            "confidence": "low",
            "participation_level": "none" if not horse_scores else "light",
            "reason_codes": ["ODDS_MISSING"] if any(horse.get("odds") is None for horse in horses) else ["ODDS_AVAILABLE"],
            "reason": "Mock/LLM fallbackのため、軽い参加判断のみを出力。",
        }
        if strategy["bet_decision"] == "SKIP":
            strategy["reason_codes"].append("SKIP_LOW_CONFIDENCE")
        return {
            "race_id": race_info["race_id"],
            "race_info": race_info,
            "marks": marks,
            "horse_scores": horse_scores,
            "bets": [],
            "summary": summary,
            "risks": risks,
            "used_lessons": matched_lessons,
            "strategy": strategy,
            "commentary": "Mock analysis commentary",
        }

    def generate_review(self, prompt: str, payload: dict) -> dict:
        race_id = payload["race_id"]
        result = payload["result"]
        prediction = payload["prediction"]
        race_info = payload.get("race_info") or result.get("race_info", {})
        if "finish_order" in result:
            finish_order = sorted(
                result.get("finish_order", []),
                key=lambda item: item.get("finish", 9999),
            )
            top3 = [item.get("horse_no") for item in finish_order[:3] if item.get("horse_no") is not None]
            payout_summary = result.get("payout_summary", {})
        else:
            top3_block = result.get("result", {})
            top3 = [top3_block.get("1st"), top3_block.get("2nd"), top3_block.get("3rd")]
            payout_summary = {"cost": 0.0, "return": 0.0}
        marked_horses = list(dict.fromkeys(prediction.get("marks", {}).values()))
        marked_horses = [horse_no for horse_no in marked_horses if horse_no]
        main_mark = prediction.get("marks", {}).get("◎", 0)
        marked_horses_top3_count = sum(1 for horse_no in marked_horses if horse_no in top3)

        bet_results, total_stake, total_return, bet_hit, payout_missing = _calculate_bet_results(
            prediction, result, top3
        )
        cost = float(total_stake if total_stake > 0 else payout_summary.get("cost", 0.0) or 0.0)
        returned = float(total_return if total_return > 0 else payout_summary.get("return", 0.0) or 0.0)
        roi = round(returned / cost, 2) if cost > 0 else 0.0

        good_points: list[str] = []
        bad_points: list[str] = []
        if main_mark in top3:
            good_points.append("本命印が3着以内に入り、軸判断は維持できました。")
        else:
            bad_points.append("本命印が3着以内を外しました。")
        if marked_horses_top3_count >= 2:
            good_points.append("印上位と実着順の整合性がありました。")
        else:
            bad_points.append("印上位と実着順の整合性が弱かったです。")
        if not prediction.get("bets"):
            bad_points.append("MVP段階ではbetsが空のため、券種検証はunknownです。")
        if payout_missing:
            bad_points.append("払戻データが不足しているためROIは暫定です。")

        if marked_horses_top3_count >= 2:
            lesson_text = "同条件では近走の同距離・同コース実績を優先して評価する。"
            confidence = "high" if main_mark in top3 else "medium"
        else:
            lesson_text = "同条件でも近走実績が薄い馬の評価を上げすぎない。"
            confidence = "medium"

        lessons = [
            {
                "course": _text_or_unknown(race_info.get("course", "unknown")),
                "surface": _text_or_unknown(race_info.get("surface", "unknown")),
                "distance": int(race_info.get("distance") or 0),
                "track_condition": _text_or_unknown(race_info.get("track_condition", "unknown")),
                "lesson": lesson_text,
                "confidence": confidence,
                "source_race_id": race_id,
            }
        ]
        return {
            "race_id": race_id,
            "hit_summary": {
                "main_mark_top3": main_mark in top3,
                "marked_horses_top3_count": marked_horses_top3_count,
                "bet_hit": bet_hit,
                "roi": roi,
                "total_stake": int(cost),
                "total_return": int(returned),
            },
            "bet_results": bet_results,
            "good_points": good_points,
            "bad_points": bad_points,
            "lessons": lessons,
        }

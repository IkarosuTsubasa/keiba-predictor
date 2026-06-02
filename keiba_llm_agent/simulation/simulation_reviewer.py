from __future__ import annotations

import json
import warnings

from keiba_llm_agent.llm.llm_client import BaseLLMClient
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.review import (
    FavorableHorseResult,
    LessonItem,
    Review,
    RiskHorseResult,
    SimulationReview,
)


SIMULATION_REVIEW_PROMPT = """あなたは競馬のレースシミュレーション回顧補助LLMです。
返答は必ずJSONのみで、good_points / bad_points / new_lessons / overall_comment を含めてください。
predictionの印・score・strategyは変更してはいけません。
与えられた結果にない事実を作らないでください。
"""


def _is_mock_like_llm(llm_client: BaseLLMClient | None) -> bool:
    if llm_client is None:
        return True
    class_name = llm_client.__class__.__name__.lower()
    module_name = llm_client.__class__.__module__.lower()
    return "mock" in class_name or "mock" in module_name


def _extract_top3(result: dict) -> list[int]:
    if "finish_order" in result:
        finish_order = sorted(
            result.get("finish_order", []),
            key=lambda item: item.get("finish", 9999),
        )
        return [item.get("horse_no") for item in finish_order[:3] if item.get("horse_no") is not None]
    top3_block = result.get("result", {})
    return [top3_block.get("1st"), top3_block.get("2nd"), top3_block.get("3rd")]


def _extract_finish_map(result: dict) -> dict[int, int]:
    finish_map: dict[int, int] = {}
    if "finish_order" in result:
        for item in result.get("finish_order", []):
            horse_no = item.get("horse_no")
            finish = item.get("finish")
            if isinstance(horse_no, int) and isinstance(finish, int):
                finish_map[horse_no] = finish
        return finish_map

    top3 = result.get("result", {})
    alias_map = {"1st": 1, "2nd": 2, "3rd": 3}
    for key, finish in alias_map.items():
        horse_no = top3.get(key)
        if isinstance(horse_no, int):
            finish_map[horse_no] = finish
    return finish_map


def _pace_result_bias(prediction: Prediction, top3: list[int]) -> str:
    pace_map = {analysis.horse_no: analysis for analysis in prediction.pace_analyses}
    front_count = 0
    closer_count = 0
    for horse_no in top3:
        analysis = pace_map.get(horse_no)
        if analysis is None:
            continue
        if analysis.running_style in {"逃げ", "先行"}:
            front_count += 1
        elif analysis.running_style in {"差し", "追込"}:
            closer_count += 1
    if front_count >= 2:
        return "front"
    if closer_count >= 2:
        return "close"
    return "neutral"


def _finish_status(finish: int | None) -> str:
    if finish is None:
        return "unknown"
    if finish <= 3:
        return "top3"
    if finish <= 5:
        return "close"
    if finish <= 8:
        return "mid"
    return "failed"


def _build_favorable_results(prediction: Prediction, finish_map: dict[int, int]) -> list[FavorableHorseResult]:
    simulation = prediction.race_simulation
    if simulation is None:
        return []
    results: list[FavorableHorseResult] = []
    for horse in simulation.favorable_horses:
        finish = finish_map.get(horse.horse_no)
        status = _finish_status(finish)
        if finish is None:
            result = "unknown"
            comment = "着順情報が限定的で評価保留。"
        elif finish <= 3:
            result = "hit"
            comment = "有利馬想定は的中。"
        elif finish <= 5:
            result = "partial"
            comment = "3着には届かなかったが4〜5着で、大きく外れてはいない。"
        elif finish <= 8:
            result = "miss"
            comment = "掲示板外まで下がり、上位争いまでは届かなかった。"
        else:
            result = "miss"
            comment = "大敗し、有利馬想定ほどの結果にはつながらなかった。"
        results.append(
            FavorableHorseResult(
                horse_no=horse.horse_no,
                horse_name=horse.horse_name,
                predicted_reason=horse.reason,
                finish=finish,
                status=status,
                result=result,
                comment=comment,
            )
        )
    return results


def _build_risk_results(prediction: Prediction, finish_map: dict[int, int]) -> list[RiskHorseResult]:
    simulation = prediction.race_simulation
    if simulation is None:
        return []
    results: list[RiskHorseResult] = []
    for horse in simulation.risk_horses:
        finish = finish_map.get(horse.horse_no)
        status = _finish_status(finish)
        if finish is None:
            result = "unknown"
            comment = "着順情報が限定的でリスク検証は保留。"
        elif finish > 5:
            result = "risk_materialized"
            comment = "想定したリスクが結果に表れた。"
        elif finish <= 3:
            result = "risk_not_materialized"
            comment = "リスク想定を上回る好走。"
        else:
            result = "unknown"
            comment = "4〜5着で、リスク評価がそのまま失敗とは言い切れない。"
        results.append(
            RiskHorseResult(
                horse_no=horse.horse_no,
                horse_name=horse.horse_name,
                predicted_risk=horse.reason,
                finish=finish,
                status=status,
                result=result,
                comment=comment,
            )
        )
    return results


def _build_new_lessons(
    prediction: Prediction,
    review: Review,
    top3_hit_count: int,
) -> list[LessonItem]:
    race_info = prediction.race_info
    if race_info is None:
        return []
    if top3_hit_count >= 2:
        lesson_text = "展開シミュレーションでは上位印と脚質の整合性を維持して評価する。"
        confidence = "medium"
    else:
        lesson_text = "展開シミュレーションは本命だけでなく相手候補の取りこぼしも点検する。"
        confidence = "medium"
    return [
        LessonItem(
            course=race_info.course or "unknown",
            surface=race_info.surface or "unknown",
            distance=race_info.distance or 0,
            track_condition=race_info.track_condition or "unknown",
            lesson=lesson_text,
            confidence=confidence,
            source_race_id=prediction.race_id,
        )
    ]


def _fallback_simulation_review(prediction: Prediction, result: dict, review: Review) -> SimulationReview:
    simulation = prediction.race_simulation
    if simulation is None:
        return SimulationReview(
            race_id=prediction.race_id,
            pace_prediction_review="事前シミュレーションは未生成。",
            scenario_hit_level="unknown",
            favorable_horses_result=[],
            risk_horses_result=[],
            win_scenario_review="検証対象なし。",
            top3_scenario_review="検証対象なし。",
            betting_scenario_review="検証対象なし。",
            good_points=[],
            bad_points=["事前シミュレーションが存在しない。"],
            new_lessons=[],
            overall_comment="シミュレーション回顧は未実施。",
        )

    top3 = _extract_top3(result)
    finish_map = _extract_finish_map(result)
    favorable_results = _build_favorable_results(prediction, finish_map)
    risk_results = _build_risk_results(prediction, finish_map)
    mark_top3 = [prediction.marks.get(mark, 0) for mark in ("◎", "○", "▲")]
    top3_hit_count = sum(1 for horse_no in mark_top3 if horse_no in top3)
    if top3_hit_count >= 3:
        scenario_level = "high"
    elif top3_hit_count >= 2:
        scenario_level = "medium"
    else:
        scenario_level = "low"

    if review.hit_summary.bet_hit:
        betting_scenario_review = "買い方は結果的に成功。"
    elif prediction.strategy and prediction.strategy.bet_decision == "SKIP":
        betting_scenario_review = "見送り判断。"
    else:
        betting_scenario_review = "買い判断は外れ。"

    bias = _pace_result_bias(prediction, top3)
    favorable_styles = prediction.race_pace_projection.favorable_styles if prediction.race_pace_projection else []
    if bias == "front":
        if any(style in favorable_styles for style in ("逃げ", "先行")):
            pace_prediction_review = "前残り傾向で、展開想定は概ね合っていた。"
        else:
            pace_prediction_review = "前残り傾向となり、展開想定との差が出た。"
    elif bias == "close":
        if any(style in favorable_styles for style in ("差し", "追込")):
            pace_prediction_review = "差し決着寄りで、展開想定は概ね合っていた。"
        else:
            pace_prediction_review = "差し決着寄りとなり、展開想定との差が出た。"
    else:
        pace_prediction_review = "脚質傾向は混在し、展開評価は中立。"

    top_horse = prediction.marks.get("◎", 0)
    top_horse_name = next((score.horse_name for score in prediction.horse_scores if score.horse_no == top_horse), "unknown")
    if top_horse in top3:
        win_scenario_review = f"◎{top_horse} {top_horse_name}の勝ち筋想定は一定程度機能した。"
    else:
        win_scenario_review = f"◎{top_horse} {top_horse_name}の勝ち筋想定は結果につながらなかった。"

    if top3_hit_count >= 2:
        top3_scenario_review = "◎○▲のうち複数が上位に入り、3着内シナリオは概ね的中。"
    else:
        top3_scenario_review = "◎○▲の上位進出は限定的で、3着内シナリオは弱かった。"

    good_points: list[str] = []
    bad_points: list[str] = []
    if scenario_level in {"high", "medium"}:
        good_points.append("上位印のシナリオ整合性は一定保てた。")
    else:
        bad_points.append("上位印のシナリオ整合性が弱かった。")
    if any(item.result == "hit" for item in favorable_results):
        good_points.append("有利馬想定の一部は結果に結びついた。")
    if any(item.result == "risk_not_materialized" for item in risk_results):
        bad_points.append("リスク馬想定の一部は見立てより走った。")
    if review.hit_summary.bet_hit:
        good_points.append("買い方シナリオは回収に結びついた。")
    elif prediction.strategy and prediction.strategy.bet_decision == "BET":
        bad_points.append("買い方シナリオは結果に結びつかなかった。")

    new_lessons = _build_new_lessons(prediction, review, top3_hit_count)
    overall_comment = (
        f"展開評価は{pace_prediction_review} "
        f"シナリオ総合評価は{scenario_level}。"
    )
    return SimulationReview(
        race_id=prediction.race_id,
        pace_prediction_review=pace_prediction_review,
        scenario_hit_level=scenario_level,
        favorable_horses_result=favorable_results,
        risk_horses_result=risk_results,
        win_scenario_review=win_scenario_review,
        top3_scenario_review=top3_scenario_review,
        betting_scenario_review=betting_scenario_review,
        good_points=good_points,
        bad_points=bad_points,
        new_lessons=new_lessons,
        overall_comment=overall_comment,
    )


def review_race_simulation(
    prediction: Prediction,
    result: dict,
    review: Review,
    llm_client: BaseLLMClient | None = None,
) -> SimulationReview:
    base_review = _fallback_simulation_review(prediction, result, review)
    if _is_mock_like_llm(llm_client):
        return base_review

    payload = {
        "prediction": {
            "race_id": prediction.race_id,
            "marks": prediction.marks,
            "strategy": prediction.strategy.model_dump() if prediction.strategy else None,
            "race_simulation": prediction.race_simulation.model_dump() if prediction.race_simulation else None,
        },
        "result": result,
        "hit_summary": review.hit_summary.model_dump(),
        "simulation_review": base_review.model_dump(),
    }
    try:
        response = llm_client.generate_json(
            SIMULATION_REVIEW_PROMPT,
            json.dumps(payload, ensure_ascii=False, indent=2),
            schema_name="simulation_review_enhancement",
        )
        good_points = response.get("good_points")
        bad_points = response.get("bad_points")
        new_lessons = response.get("new_lessons")
        overall_comment = response.get("overall_comment")
        if isinstance(good_points, list) and all(isinstance(item, str) for item in good_points):
            base_review.good_points = good_points
        if isinstance(bad_points, list) and all(isinstance(item, str) for item in bad_points):
            base_review.bad_points = bad_points
        if isinstance(new_lessons, list):
            valid_lessons: list[LessonItem] = []
            for lesson in new_lessons:
                try:
                    valid_lessons.append(LessonItem.model_validate(lesson))
                except Exception:
                    continue
            if valid_lessons:
                base_review.new_lessons = valid_lessons
        if isinstance(overall_comment, str) and overall_comment.strip():
            base_review.overall_comment = overall_comment
    except Exception as exc:
        warnings.warn(f"simulation review enhancement failed: {exc}", stacklevel=2)
    return base_review

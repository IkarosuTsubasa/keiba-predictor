from __future__ import annotations

import json
import warnings

from keiba_llm_agent.llm.llm_client import BaseLLMClient
from keiba_llm_agent.schemas.deep_analysis import HorseDeepAnalysis
from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis
from keiba_llm_agent.schemas.prediction import BetSuggestion, HorseScore, StrategyDecision
from keiba_llm_agent.schemas.race_data import RaceInfo
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis
from keiba_llm_agent.schemas.race_simulation import FavorableHorse, RaceSimulation, RiskHorse


SIMULATION_PROMPT = """あなたは競馬の展開シミュレーション補助LLMです。
出力は必ずJSONのみ。日本語で書いてください。
horse_scores、印、strategy、買い目を変更してはいけません。
与えられた構造化情報だけを使い、事実のない内容を作らないでください。
情報不足があれば warnings に明記してください。
reasoning_summary は公開ページにも出るため、展開がどう馬券判断に効くかを1文で書いてください。「シミュレーションでは平均寄り」「平均寄り」のような定型句だけで終えないでください。
口調は競馬分析師だが、過度に断定しないでください。
"""

PACE_LABEL_MAP = {
    "slow": "スロー",
    "average": "平均",
    "fast": "ハイ",
    "unknown": "不明",
}


def _is_mock_like_llm(llm_client: BaseLLMClient | None) -> bool:
    if llm_client is None:
        return True
    try:
        from keiba_llm_agent.llm.mock_llm_client import MockLLMClient

        if isinstance(llm_client, MockLLMClient):
            return True
    except Exception:
        pass
    class_name = llm_client.__class__.__name__.lower()
    module_name = llm_client.__class__.__module__.lower()
    return "mock" in class_name or "mock" in module_name


def _horse_map(items: list[object]) -> dict[int, object]:
    return {getattr(item, "horse_no"): item for item in items}


def _select_top_horses(horse_scores: list[HorseScore], limit: int = 7) -> list[HorseScore]:
    return sorted(horse_scores, key=lambda item: (-item.total_score, item.horse_no))[:limit]


def _mark_map(horse_scores: list[HorseScore]) -> dict[int, str]:
    labels = ("◎", "○", "▲", "△", "☆")
    marks: dict[int, str] = {}
    for index, horse_score in enumerate(_select_top_horses(horse_scores, limit=5)):
        marks[horse_score.horse_no] = labels[index]
    return marks


def _compact_text(text: str, limit: int = 48) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _flag_to_risk_phrase(flags: list[str], running_style: str | None = None) -> str:
    flag_set = set(flags)
    if "RECENT_FORM_DECLINING" in flag_set and "JOCKEY_CHANGE" in flag_set:
        return "近走下降傾向と騎手替わりがリスク。"
    if "DISTANCE_UNKNOWN" in flag_set and "COURSE_UNKNOWN" in flag_set:
        return "距離・コース適性に未知要素が残る。"
    if "COURSE_UNKNOWN" in flag_set and "JOCKEY_CHANGE" in flag_set:
        return "コース適性と騎手替わりに注意。"
    if "PEDIGREE_DISTANCE_RISK" in flag_set:
        return "血統面で距離延長への裏付けが弱い。"
    if "HEAD_TO_HEAD_NEGATIVE" in flag_set:
        return "再戦比較ではやや劣勢。"
    if "POPULAR_DISAPPOINTMENT" in flag_set:
        return "人気先行で取りこぼしリスクがある。"
    if "PACE_MISMATCH" in flag_set:
        style_text = running_style or "脚質"
        return f"{style_text}が想定ペースと噛み合わない懸念。"
    if "PACE_DATA_INCOMPLETE" in flag_set:
        return "脚質データが不足しており展開面は慎重。"
    if "TRACK_CONDITION_UNKNOWN" in flag_set:
        return "馬場適性の裏付けが薄い。"
    if "PEDIGREE_DATA_INCOMPLETE" in flag_set and len(flag_set) == 1:
        return "血統情報が不足しており上積み評価は控えたい。"
    if "DATA_INCOMPLETE" in flag_set:
        return "判断材料が不足しており評価は慎重。"
    if flags:
        return "・".join(flags) + " に注意。"
    return "明確なリスク材料は限定的。"


def _build_reasoning_summary(projected_pace: str, style_text: str) -> str:
    style = style_text if style_text and style_text != "不明" else "上位勢"
    if projected_pace == "slow":
        return f"序盤が落ち着くなら、{style}の位置取りと直線での反応を重く見る。"
    if projected_pace == "fast":
        return f"前半から流れる想定で、{style}の持続力と早めに脚を使った後の粘りを問う。"
    if projected_pace == "average":
        return f"流れが極端に偏らないぶん、{style}の持続力とロスの少なさが明暗を分ける。"
    return f"展開の読みは絞り切れず、{style}が道中で不利を受けないかを重く見る。"


def _build_top_payload(
    horse_scores: list[HorseScore],
    deep_analyses: list[HorseDeepAnalysis],
    pedigree_analyses: list[PedigreeAnalysis],
    race_level_analyses: list[RaceLevelAnalysis],
    pace_analyses: list[HorsePaceAnalysis],
) -> list[dict]:
    deep_map = _horse_map(deep_analyses)
    pedigree_map = _horse_map(pedigree_analyses)
    race_level_map = _horse_map(race_level_analyses)
    pace_map = _horse_map(pace_analyses)
    top_horses = _select_top_horses(horse_scores, limit=7)
    mark_by_horse = _mark_map(horse_scores)

    payload: list[dict] = []
    for horse_score in top_horses:
        deep = deep_map.get(horse_score.horse_no)
        pedigree = pedigree_map.get(horse_score.horse_no)
        race_level = race_level_map.get(horse_score.horse_no)
        pace = pace_map.get(horse_score.horse_no)
        payload.append(
            {
                "mark": mark_by_horse.get(horse_score.horse_no, ""),
                "horse_no": horse_score.horse_no,
                "horse_name": horse_score.horse_name,
                "total_score": horse_score.total_score,
                "reason": horse_score.reason,
                "deep_summary": {
                    "positive_flags": deep.positive_flags if deep else [],
                    "risk_flags": deep.risk_flags if deep else [],
                    "overall_comment": deep.overall_comment if deep else "",
                },
                "pedigree_summary": {
                    "positive_flags": pedigree.positive_flags if pedigree else [],
                    "risk_flags": pedigree.risk_flags if pedigree else [],
                    "overall_comment": pedigree.overall_comment if pedigree else "",
                },
                "race_level_summary": {
                    "positive_flags": race_level.positive_flags if race_level else [],
                    "risk_flags": race_level.risk_flags if race_level else [],
                    "overall_comment": race_level.overall_comment if race_level else "",
                },
                "pace_summary": {
                    "running_style": pace.running_style if pace else "不明",
                    "positive_flags": pace.positive_flags if pace else [],
                    "risk_flags": pace.risk_flags if pace else [],
                    "overall_comment": pace.overall_comment if pace else "",
                },
            }
        )
    return payload


def _to_favorable_horses(payload: list[dict]) -> list[FavorableHorse]:
    favorable: list[FavorableHorse] = []
    for item in payload[:3]:
        reasons: list[str] = []
        deep_flags = item["deep_summary"]["positive_flags"]
        pedigree_flags = item["pedigree_summary"]["positive_flags"]
        pace_style = item["pace_summary"]["running_style"]
        if deep_flags:
            reasons.append("近走内容を評価")
        if pedigree_flags:
            reasons.append("血統面に後押し")
        if pace_style in {"逃げ", "先行", "差し"}:
            reasons.append(f"{pace_style}想定")
        favorable.append(
            FavorableHorse(
                horse_no=item["horse_no"],
                horse_name=item["horse_name"],
                reason="、".join(reasons) if reasons else "上位評価の一角。",
            )
        )
    return favorable


def _to_risk_horses(payload: list[dict]) -> list[RiskHorse]:
    risk_horses: list[RiskHorse] = []
    for item in payload:
        risks = (
            item["deep_summary"]["risk_flags"]
            + item["pedigree_summary"]["risk_flags"]
            + item["race_level_summary"]["risk_flags"]
            + item["pace_summary"]["risk_flags"]
        )
        if not risks:
            continue
        running_style = item["pace_summary"].get("running_style")
        risk_horses.append(
            RiskHorse(
                horse_no=item["horse_no"],
                horse_name=item["horse_name"],
                reason=_flag_to_risk_phrase(risks, running_style=running_style),
            )
        )
        if len(risk_horses) >= 3:
            break
    return risk_horses


def _build_template_simulation(
    race_info: RaceInfo,
    marks: dict[str, int],
    horse_scores: list[HorseScore],
    bets: list[BetSuggestion],
    deep_analyses: list[HorseDeepAnalysis],
    pedigree_analyses: list[PedigreeAnalysis],
    race_level_analyses: list[RaceLevelAnalysis],
    pace_analyses: list[HorsePaceAnalysis],
    race_pace_projection: RacePaceProjection | None,
    strategy: StrategyDecision | None,
    warnings_list: list[str] | None = None,
) -> RaceSimulation:
    top_payload = _build_top_payload(
        horse_scores,
        deep_analyses,
        pedigree_analyses,
        race_level_analyses,
        pace_analyses,
    )
    top_horses = _select_top_horses(horse_scores, limit=3)
    projected_pace = race_pace_projection.projected_pace if race_pace_projection is not None else "unknown"
    pace_comment = race_pace_projection.pace_comment if race_pace_projection is not None else "展開情報は限定的。"
    favorable_styles = race_pace_projection.favorable_styles if race_pace_projection is not None else []

    favorable_horses = _to_favorable_horses(top_payload)
    risk_horses = _to_risk_horses(top_payload)
    if len(favorable_horses) < 3:
        for item in top_payload:
            if any(horse.horse_no == item["horse_no"] for horse in favorable_horses):
                continue
            favorable_horses.append(
                FavorableHorse(
                    horse_no=item["horse_no"],
                    horse_name=item["horse_name"],
                    reason="上位印の一角として評価。",
                )
            )
            if len(favorable_horses) >= 3:
                break
    top_names = " → ".join(f"{horse.horse_no}{horse.horse_name}" for horse in top_horses) if top_horses else "unknown"
    style_text = "〜".join(favorable_styles[:2]) if favorable_styles else "不明"
    strategy_text = strategy.reason if strategy is not None else "買い判断は保留。"
    warnings_output = list(warnings_list or [])
    mark_horse_names = {
        mark: next((score.horse_name for score in horse_scores if score.horse_no == horse_no), "unknown")
        for mark, horse_no in marks.items()
        if horse_no
    }

    key_positions = (
        f"序盤は{style_text}勢が主導しやすく、◎{marks.get('◎', 0)}{mark_horse_names.get('◎', 'unknown')}は好位を確保したい。上位想定は {top_names}。"
        if top_horses
        else "位置取りの想定材料は限定的。"
    )
    win_scenario = (
        f"◎{marks.get('◎', 0)}{mark_horse_names.get('◎', 'unknown')}が想定通りの流れで運べれば押し切り。"
        if top_horses
        else "勝ち筋の特定材料は不足。"
    )
    top3_scenario = (
        f"◎{marks.get('◎', 0)}{mark_horse_names.get('◎', 'unknown')}、○{marks.get('○', 0)}{mark_horse_names.get('○', 'unknown')}、▲{marks.get('▲', 0)}{mark_horse_names.get('▲', 'unknown')}が3着内候補の中心。"
        if top_horses
        else "3着内シナリオは不明。"
    )
    if bets:
        first_bet = bets[0]
        horse_numbers = "-".join(str(number) for number in first_bet.horse_numbers)
        amount_text = f"{first_bet.amount}円" if first_bet.amount is not None else "金額未設定"
        bet_reason = first_bet.reason or strategy_text
        betting_scenario = f"買い目は{first_bet.bet_type}{horse_numbers}を{amount_text}。{bet_reason}"
    elif strategy is not None and strategy.bet_decision == "SKIP":
        betting_scenario = "買い目なし。見送り判断。"
    else:
        betting_scenario = f"戦略は {strategy.bet_decision if strategy else 'unknown'}。{strategy_text}"
    confidence_comment = (
        f"confidence={strategy.confidence}。展開面は{PACE_LABEL_MAP.get(projected_pace, projected_pace)}想定。"
        if strategy is not None
        else f"展開面は{PACE_LABEL_MAP.get(projected_pace, projected_pace)}想定。"
    )
    reasoning_summary = _build_reasoning_summary(projected_pace, style_text)
    return RaceSimulation(
        race_id=race_info.race_id,
        projected_pace=projected_pace,
        race_flow=pace_comment,
        key_positions=key_positions,
        favorable_horses=favorable_horses,
        risk_horses=risk_horses,
        win_scenario=win_scenario,
        top3_scenario=top3_scenario,
        betting_scenario=betting_scenario,
        confidence_comment=confidence_comment,
        reasoning_summary=reasoning_summary,
        warnings=warnings_output,
    )


def _build_llm_payload(
    race_info: RaceInfo,
    marks: dict[str, int],
    horse_scores: list[HorseScore],
    bets: list[BetSuggestion],
    deep_analyses: list[HorseDeepAnalysis],
    pedigree_analyses: list[PedigreeAnalysis],
    race_level_analyses: list[RaceLevelAnalysis],
    pace_analyses: list[HorsePaceAnalysis],
    race_pace_projection: RacePaceProjection | None,
    strategy: StrategyDecision | None,
) -> dict:
    return {
        "race_info": race_info.model_dump(),
        "marks": marks,
        "bets": [bet.model_dump() for bet in bets],
        "top_horses": _build_top_payload(
            horse_scores,
            deep_analyses,
            pedigree_analyses,
            race_level_analyses,
            pace_analyses,
        ),
        "race_pace_projection": race_pace_projection.model_dump() if race_pace_projection is not None else None,
        "strategy": strategy.model_dump() if strategy is not None else None,
    }


def _normalize_simulation_response(response: dict, race_info: RaceInfo, projected_pace: str) -> RaceSimulation:
    payload = dict(response)
    payload.setdefault("race_id", race_info.race_id)
    payload.setdefault("projected_pace", projected_pace)
    payload.setdefault("race_flow", "")
    payload.setdefault("key_positions", "")
    payload.setdefault("favorable_horses", [])
    payload.setdefault("risk_horses", [])
    payload.setdefault("win_scenario", "")
    payload.setdefault("top3_scenario", "")
    payload.setdefault("betting_scenario", "")
    payload.setdefault("confidence_comment", "")
    payload.setdefault("reasoning_summary", "")
    payload.setdefault("warnings", [])
    return RaceSimulation.model_validate(payload)


def _simulation_warnings_from_client(llm_client: BaseLLMClient | None) -> list[str]:
    if llm_client is not None and getattr(llm_client, "last_fallback_used", False):
        return ["LLM simulation fallback used."]
    return []


def _is_incomplete_simulation(simulation: RaceSimulation) -> bool:
    required_texts = [
        simulation.race_flow,
        simulation.key_positions,
        simulation.win_scenario,
        simulation.top3_scenario,
        simulation.betting_scenario,
        simulation.reasoning_summary,
    ]
    if any(not text or not text.strip() for text in required_texts):
        return True
    if not simulation.favorable_horses:
        return True
    return False


def simulate_race(
    race_info: RaceInfo,
    marks: dict[str, int],
    horse_scores: list[HorseScore],
    bets: list[BetSuggestion],
    deep_analyses: list[HorseDeepAnalysis],
    pedigree_analyses: list[PedigreeAnalysis],
    race_level_analyses: list[RaceLevelAnalysis],
    pace_analyses: list[HorsePaceAnalysis],
    race_pace_projection: RacePaceProjection | None,
    strategy: StrategyDecision | None,
    llm_client: BaseLLMClient | None = None,
) -> RaceSimulation:
    if _is_mock_like_llm(llm_client):
        return _build_template_simulation(
            race_info,
            marks,
            horse_scores,
            bets,
            deep_analyses,
            pedigree_analyses,
            race_level_analyses,
            pace_analyses,
            race_pace_projection,
            strategy,
            warnings_list=["LLM simulation fallback used."],
        )

    payload = _build_llm_payload(
        race_info,
        marks,
        horse_scores,
        bets,
        deep_analyses,
        pedigree_analyses,
        race_level_analyses,
        pace_analyses,
        race_pace_projection,
        strategy,
    )
    try:
        response = llm_client.generate_json(
            SIMULATION_PROMPT,
            json.dumps(payload, ensure_ascii=False, indent=2),
            schema_name="race_simulation",
        )
        simulation = _normalize_simulation_response(
            response,
            race_info=race_info,
            projected_pace=race_pace_projection.projected_pace if race_pace_projection is not None else "unknown",
        )
        if _is_incomplete_simulation(simulation):
            fallback_simulation = _build_template_simulation(
                race_info,
                marks,
                horse_scores,
                bets,
                deep_analyses,
                pedigree_analyses,
                race_level_analyses,
                pace_analyses,
                race_pace_projection,
                strategy,
                warnings_list=["LLM simulation output was incomplete; fallback used."],
            )
            fallback_warnings = _simulation_warnings_from_client(llm_client)
            for warning_text in fallback_warnings:
                if warning_text not in fallback_simulation.warnings:
                    fallback_simulation.warnings.append(warning_text)
            return fallback_simulation
        if not simulation.warnings:
            simulation.warnings = _simulation_warnings_from_client(llm_client)
        return simulation
    except Exception as exc:
        warnings.warn(f"race simulation failed: {exc}", stacklevel=2)
        return _build_template_simulation(
            race_info,
            marks,
            horse_scores,
            bets,
            deep_analyses,
            pedigree_analyses,
            race_level_analyses,
            pace_analyses,
            race_pace_projection,
            strategy,
            warnings_list=["LLM simulation fallback used."],
        )

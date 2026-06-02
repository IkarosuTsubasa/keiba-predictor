from __future__ import annotations

from pathlib import Path

from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData, RaceInfo
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


MARK_ORDER = ("◎", "○", "▲", "△", "☆")


def _race_title(race_info: RaceInfo | None, fallback_race_id: str, suffix: str) -> str:
    race_name = race_info.race_name if race_info and race_info.race_name else fallback_race_id
    return f"# {race_name} {suffix}"


def _overview_lines(race_info: RaceInfo | None, fallback_race_id: str) -> list[str]:
    race_id = race_info.race_id if race_info else fallback_race_id
    race_date = race_info.race_date if race_info and race_info.race_date else "unknown"
    course = race_info.course if race_info and race_info.course else "unknown"
    surface = race_info.surface if race_info and race_info.surface else "unknown"
    distance = race_info.distance if race_info and race_info.distance else "unknown"
    track_condition = (
        race_info.track_condition if race_info and race_info.track_condition else "unknown"
    )
    weather = race_info.weather if race_info and race_info.weather else "unknown"
    return [
        "## レース概要",
        f"- race_id: {race_id}",
        f"- 日付: {race_date}",
        f"- コース: {course}",
        f"- 条件: {surface} {distance}m / 馬場={track_condition} / 天候={weather}",
    ]


def _horse_score_map(prediction: Prediction) -> dict[int, object]:
    return {horse_score.horse_no: horse_score for horse_score in prediction.horse_scores}


def _race_horse_map(race_data: RaceData | None) -> dict[int, object]:
    if race_data is None:
        return {}
    return {horse.horse_no: horse for horse in race_data.horses}


def _horse_name_map(prediction: Prediction, race_data: RaceData | None = None) -> dict[int, str]:
    horse_names = {horse_score.horse_no: horse_score.horse_name for horse_score in prediction.horse_scores}
    for horse_no, horse in _race_horse_map(race_data).items():
        horse_names.setdefault(horse_no, horse.horse_name)
    return horse_names


def _format_mark_lines(prediction: Prediction) -> list[str]:
    horse_map = _horse_score_map(prediction)
    lines = ["## 印"]
    for mark in MARK_ORDER:
        horse_no = prediction.marks.get(mark, 0)
        horse_score = horse_map.get(horse_no)
        horse_name = horse_score.horse_name if horse_score else "unknown"
        lines.append(f"- {mark} {horse_no} {horse_name}")
    return lines


def _format_scoring_config(prediction: Prediction) -> list[str]:
    config = prediction.scoring_config
    market_config = prediction.market_signal_config
    return [
        "## Scoring Profile",
        f"- scoring_profile: {prediction.scoring_profile}",
        f"- scoring_mode: {prediction.scoring_mode}",
        f"- pedigree_weight: {config.pedigree_weight}",
        f"- race_level_weight: {config.race_level_weight}",
        f"- pace_weight: {config.pace_weight}",
        f"- conditional_weight_profile: {config.conditional_weight_profile}",
        f"- use_market_score_in_ranking: {str(market_config.use_market_score_in_ranking).lower()}",
        f"- market_signal_weight: {market_config.market_signal_weight}",
        f"- borderline_recovery_enabled: {str(prediction.borderline_recovery_enabled).lower()}",
        "- odds / 人気: 参考情報（core ranking scoreには未使用）",
    ]


def _format_bets(prediction: Prediction) -> list[str]:
    lines = ["## 買い目"]
    if not prediction.bets:
        lines.append("- なし")
        return lines

    for bet in prediction.bets:
        horse_numbers = "-".join(str(number) for number in bet.horse_numbers)
        amount = f"{bet.amount}円" if bet.amount is not None else "金額未設定"
        reason = f" ({bet.reason})" if bet.reason else ""
        lines.append(f"- {bet.bet_type} {horse_numbers} {amount}{reason}")
    return lines


def _format_risks(prediction: Prediction) -> list[str]:
    lines = ["## リスク"]
    if not prediction.risks:
        lines.append("- なし")
        return lines

    lines.extend(f"- {risk}" for risk in prediction.risks)
    return lines


def _format_lessons(prediction: Prediction) -> list[str]:
    lines = ["## 使用 lesson"]
    if not prediction.used_lessons:
        lines.append("- なし")
        return lines

    for lesson in prediction.used_lessons:
        lines.append(
            "- "
            f"{lesson.course}/{lesson.surface}/{lesson.distance}m/{lesson.track_condition} "
            f"[{lesson.confidence}] {lesson.lesson} (source={lesson.source_race_id})"
        )
    return lines


def _deep_analysis_map(prediction: Prediction) -> dict[int, object]:
    return {analysis.horse_no: analysis for analysis in prediction.deep_analyses}


def _format_deep_analyses(prediction: Prediction) -> list[str]:
    lines = ["## 深掘り分析"]
    if not prediction.deep_analyses:
        lines.append("- なし")
        return lines

    mark_by_horse_no = {horse_no: mark for mark, horse_no in prediction.marks.items()}
    deep_map = _deep_analysis_map(prediction)
    for horse_score in sorted(prediction.horse_scores, key=lambda item: item.total_score, reverse=True)[:5]:
        analysis = deep_map.get(horse_score.horse_no)
        if analysis is None:
            continue
        mark = mark_by_horse_no.get(horse_score.horse_no, "")
        positive = ", ".join(analysis.positive_flags) if analysis.positive_flags else "なし"
        risks = ", ".join(analysis.risk_flags) if analysis.risk_flags else "なし"
        lines.extend(
            [
                f"### {mark} {horse_score.horse_no} {horse_score.horse_name}".strip(),
                f"- Positive: {positive}",
                f"- Risk: {risks}",
                f"- Comment: {analysis.overall_comment}",
            ]
        )
    return lines


def _pedigree_analysis_map(prediction: Prediction) -> dict[int, object]:
    return {analysis.horse_no: analysis for analysis in prediction.pedigree_analyses}


def _race_level_analysis_map(prediction: Prediction) -> dict[int, object]:
    return {analysis.horse_no: analysis for analysis in prediction.race_level_analyses}


def _pace_analysis_map(prediction: Prediction) -> dict[int, object]:
    return {analysis.horse_no: analysis for analysis in prediction.pace_analyses}


def _format_pace_analyses(prediction: Prediction) -> list[str]:
    lines = ["## 展開・脚質分析"]
    projection = prediction.race_pace_projection
    if projection is None:
        lines.append("- projected_pace: unknown")
        return lines

    lines.extend(
        [
            f"- projected_pace: {projection.projected_pace}",
            f"- pace_comment: {projection.pace_comment}",
            f"- favorable_styles: {', '.join(projection.favorable_styles) if projection.favorable_styles else 'なし'}",
            f"- risk_styles: {', '.join(projection.risk_styles) if projection.risk_styles else 'なし'}",
        ]
    )
    if not prediction.pace_analyses:
        return lines

    mark_by_horse_no = {horse_no: mark for mark, horse_no in prediction.marks.items()}
    pace_map = _pace_analysis_map(prediction)
    for horse_score in sorted(prediction.horse_scores, key=lambda item: item.total_score, reverse=True)[:5]:
        analysis = pace_map.get(horse_score.horse_no)
        if analysis is None:
            continue
        positive = ", ".join(analysis.positive_flags) if analysis.positive_flags else "なし"
        risks = ", ".join(analysis.risk_flags) if analysis.risk_flags else "なし"
        mark = mark_by_horse_no.get(horse_score.horse_no, "")
        lines.extend(
            [
                f"### {mark} {horse_score.horse_no} {horse_score.horse_name}".strip(),
                f"- Running style: {analysis.running_style}",
                f"- Positive: {positive}",
                f"- Risk: {risks}",
                f"- Comment: {analysis.overall_comment}",
            ]
        )
    return lines


def _format_race_simulation(prediction: Prediction) -> list[str]:
    lines = ["## レースシミュレーション"]
    simulation = prediction.race_simulation
    if simulation is None:
        lines.append("- なし")
        return lines

    lines.extend(
        [
            f"- 展開想定: {simulation.projected_pace}",
            f"- レースの流れ: {simulation.race_flow}",
            f"- 位置取り: {simulation.key_positions}",
            f"- 勝ち筋: {simulation.win_scenario}",
            f"- 3着内シナリオ: {simulation.top3_scenario}",
            f"- 買い方シナリオ: {simulation.betting_scenario}",
            f"- Confidence comment: {simulation.confidence_comment}",
        ]
    )
    lines.append("- 有利になりそうな馬:")
    if simulation.favorable_horses:
        lines.extend(
            f"  - {horse.horse_no} {horse.horse_name}: {horse.reason}"
            for horse in simulation.favorable_horses
        )
    else:
        lines.append("  - なし")
    lines.append("- リスク馬:")
    if simulation.risk_horses:
        lines.extend(
            f"  - {horse.horse_no} {horse.horse_name}: {horse.reason}"
            for horse in simulation.risk_horses
        )
    else:
        lines.append("  - なし")
    if simulation.warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {warning}" for warning in simulation.warnings)
    return lines


def _format_borderline_recovery(prediction: Prediction) -> list[str]:
    lines = ["## Top5境界補正"]
    if not prediction.borderline_recovery_result.recovery_applied:
        lines.append("Top5境界補正の対象馬なし。")
        return lines
    lines.extend(
        [
            "| 馬番 | 馬名 | original_rank | new_rank | score_gap | recovery_bonus | reasons |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for case in prediction.borderline_recovery_result.recovery_cases:
        lines.append(
            f"| {case.horse_no} | {case.horse_name} | {case.original_rank} | {case.new_rank} | "
            f"{case.score_gap_to_top5:.1f} | {case.recovery_bonus:.1f} | {', '.join(case.recovery_reasons)} |"
        )
    return lines


def _format_race_level_analyses(prediction: Prediction) -> list[str]:
    lines = ["## レースレベル・相手関係分析"]
    if not prediction.race_level_analyses:
        lines.append("- なし")
        return lines

    mark_by_horse_no = {horse_no: mark for mark, horse_no in prediction.marks.items()}
    race_level_map = _race_level_analysis_map(prediction)
    for horse_score in sorted(prediction.horse_scores, key=lambda item: item.total_score, reverse=True)[:5]:
        analysis = race_level_map.get(horse_score.horse_no)
        if analysis is None:
            continue
        positive = ", ".join(analysis.positive_flags) if analysis.positive_flags else "なし"
        risks = ", ".join(analysis.risk_flags) if analysis.risk_flags else "なし"
        mark = mark_by_horse_no.get(horse_score.horse_no, "")
        lines.extend(
            [
                f"### {mark} {horse_score.horse_no} {horse_score.horse_name}".strip(),
                f"- Positive: {positive}",
                f"- Risk: {risks}",
                f"- Head-to-head: {analysis.head_to_head_summary}",
                f"- Race level: {analysis.race_level_summary}",
                f"- Comment: {analysis.overall_comment}",
            ]
        )
    return lines


def _format_pedigree_analyses(prediction: Prediction) -> list[str]:
    lines = ["## 血統分析"]
    if not prediction.pedigree_analyses:
        lines.append("- なし")
        return lines

    mark_by_horse_no = {horse_no: mark for mark, horse_no in prediction.marks.items()}
    pedigree_map = _pedigree_analysis_map(prediction)
    for horse_score in sorted(prediction.horse_scores, key=lambda item: item.total_score, reverse=True)[:5]:
        analysis = pedigree_map.get(horse_score.horse_no)
        if analysis is None:
            continue
        positive = ", ".join(analysis.positive_flags) if analysis.positive_flags else "なし"
        risks = ", ".join(analysis.risk_flags) if analysis.risk_flags else "なし"
        mark = mark_by_horse_no.get(horse_score.horse_no, "")
        lines.extend(
            [
                f"### {mark} {horse_score.horse_no} {horse_score.horse_name}".strip(),
                f"- 父: {analysis.sire or 'unknown'}",
                f"- 母父: {analysis.damsire or 'unknown'}",
                f"- Positive: {positive}",
                f"- Risk: {risks}",
                f"- Comment: {analysis.overall_comment}",
            ]
        )
    return lines


def _format_prediction_strategy(prediction: Prediction) -> list[str]:
    lines = ["## 買い判断"]
    if prediction.strategy is None:
        lines.append("- 判定: unknown")
        return lines

    lines.extend(
        [
            f"- 判定: {prediction.strategy.bet_decision}",
            f"- Confidence: {prediction.strategy.confidence}",
            f"- Participation: {prediction.strategy.participation_level}",
            (
                "- Reason Codes: "
                + (", ".join(prediction.strategy.reason_codes) if prediction.strategy.reason_codes else "なし")
            ),
            f"- 理由: {prediction.strategy.reason}",
        ]
    )
    return lines


def _format_top_horses_table(prediction: Prediction, race_data: RaceData | None = None) -> list[str]:
    lines = [
        "## 上位5頭評価",
        "- odds / 人気は参考情報であり、scoring reasonではない。",
        "| 印 | 馬番 | 馬名 | base | pedigree | raceLv | pace | border_adj | total | odds(参考) | 人気(参考) | reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    mark_by_horse_no = {horse_no: mark for mark, horse_no in prediction.marks.items()}
    race_horse_map = _race_horse_map(race_data)
    sorted_horses = sorted(
        prediction.horse_scores,
        key=lambda item: item.total_score,
        reverse=True,
    )[:5]
    for horse_score in sorted_horses:
        mark = mark_by_horse_no.get(horse_score.horse_no, "")
        race_horse = race_horse_map.get(horse_score.horse_no)
        odds = race_horse.odds if race_horse is not None else None
        popularity = race_horse.popularity if race_horse is not None else None
        odds_text = f"{odds:.1f}" if isinstance(odds, (int, float)) else "-"
        popularity_text = str(popularity) if popularity is not None else "-"
        base_score_text = f"{horse_score.base_total_score:.1f}" if horse_score.base_total_score else f"{horse_score.total_score:.1f}"
        pedigree_adj_text = f"{horse_score.score_breakdown.pedigree_adjustment_weighted:+.1f}"
        race_level_adj_text = f"{horse_score.score_breakdown.race_level_adjustment_weighted:+.1f}"
        pace_adj_text = f"{horse_score.score_breakdown.pace_adjustment_weighted:+.1f}"
        border_adj_text = f"{horse_score.score_breakdown.borderline_recovery_bonus:+.1f}"
        lines.append(
            f"| {mark} | {horse_score.horse_no} | {horse_score.horse_name} | "
            f"{base_score_text} | {pedigree_adj_text} | {race_level_adj_text} | {pace_adj_text} | {border_adj_text} | "
            f"{horse_score.total_score:.1f} | {odds_text} | {popularity_text} | {horse_score.reason} |"
        )
    return lines


def _format_review_results(review: Review, result_data: ResultData | None = None) -> list[str]:
    summary = review.hit_summary
    payout_status = "未確認"
    if result_data is not None:
        if result_data.payouts:
            payout_status = f"解析済み ({len(result_data.payouts)}件)"
        else:
            payout_status = "未解析"
    lines = [
        "## 予想結果",
        f"- ◎ 的中: {'的中' if summary.main_mark_top3 else '不的中'}",
        f"- 印内Top3数: {summary.marked_horses_top3_count}",
        f"- bet_hit: {summary.bet_hit}",
        f"- ROI: {summary.roi:.2f}",
        f"- total_stake: {summary.total_stake}",
        f"- total_return: {summary.total_return}",
        f"- 払戻情報: {payout_status}",
        f"- payout_warning: {'true' if review.payout_warning else 'false'}",
        f"- ROI reliability: {'unreliable' if review.payout_warning else 'reliable'}",
    ]
    if review.review_warnings:
        lines.extend(f"- warning: {warning}" for warning in review.review_warnings)
    if result_data is not None and result_data.warnings:
        lines.extend(f"- result_warning: {warning}" for warning in result_data.warnings)
    return lines


def _format_actual_result_lines(
    prediction: Prediction,
    result_data: ResultData | None = None,
    race_data: RaceData | None = None,
) -> list[str]:
    if result_data is None:
        return [
            "## 実際結果",
            "- 1着: review.jsonに実馬番は未保存",
            "- 2着: review.jsonに実馬番は未保存",
            "- 3着: review.jsonに実馬番は未保存",
        ]

    horse_name_map = _horse_name_map(prediction, race_data)
    first = result_data.result.first
    second = result_data.result.second
    third = result_data.result.third

    def _label(horse_no: int) -> str:
        horse_name = horse_name_map.get(horse_no)
        if horse_name:
            return f"{horse_no} {horse_name}"
        return str(horse_no)

    return [
        "## 実際結果",
        f"- 1着: {_label(first)}",
        f"- 2着: {_label(second)}",
        f"- 3着: {_label(third)}",
    ]


def _format_review_bet_table(review: Review) -> list[str]:
    lines = [
        "## 買い目結果",
        "| 券種 | 馬番 | 金額 | 的中 | 払戻 | 回収 |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    if not review.bet_results:
        lines.append("| なし | - | 0 | - | 0 | 0 |")
        return lines

    for bet_result in review.bet_results:
        horse_numbers = "-".join(str(number) for number in bet_result.horse_numbers)
        lines.append(
            f"| {bet_result.bet_type} | {horse_numbers} | {bet_result.amount} | "
            f"{'的中' if bet_result.hit else '不的中'} | {bet_result.payout} | {bet_result.return_amount} |"
        )
    return lines


def _format_simple_list(title: str, items: list[str]) -> list[str]:
    lines = [title]
    if not items:
        lines.append("- なし")
        return lines
    lines.extend(f"- {item}" for item in items)
    return lines


def _format_simulation_review(review: Review) -> list[str]:
    lines = ["## レースシミュレーション回顧"]
    simulation_review = review.simulation_review
    if simulation_review is None:
        lines.append("- なし")
        return lines

    lines.extend(
        [
            f"- 展開予測の評価: {simulation_review.pace_prediction_review}",
            f"- 勝ち筋の評価: {simulation_review.win_scenario_review}",
            f"- 3着内シナリオの評価: {simulation_review.top3_scenario_review}",
            f"- 買い方シナリオの評価: {simulation_review.betting_scenario_review}",
            f"- Scenario hit level: {simulation_review.scenario_hit_level}",
        ]
    )
    lines.append("- 有利馬の結果:")
    if simulation_review.favorable_horses_result:
        lines.extend(
            f"  - {item.horse_no} {item.horse_name}: finish={item.finish if item.finish is not None else 'unknown'} / status={item.status} / {item.result} / {item.comment}"
            for item in simulation_review.favorable_horses_result
        )
    else:
        lines.append("  - なし")
    lines.append("- リスク馬の結果:")
    if simulation_review.risk_horses_result:
        lines.extend(
            f"  - {item.horse_no} {item.horse_name}: finish={item.finish if item.finish is not None else 'unknown'} / status={item.status} / {item.result} / {item.comment}"
            for item in simulation_review.risk_horses_result
        )
    else:
        lines.append("  - なし")
    lines.extend(_format_simple_list("### 良かった点", simulation_review.good_points))
    lines.extend(_format_simple_list("### 反省点", simulation_review.bad_points))
    lines.extend(
        _format_simple_list(
            "### 次回 lesson",
            [
                f"{lesson.course}/{lesson.surface}/{lesson.distance}m/{lesson.track_condition} "
                f"[{lesson.confidence}] {lesson.lesson}"
                for lesson in simulation_review.new_lessons
            ],
        )
    )
    lines.append(f"- Overall: {simulation_review.overall_comment}")
    return lines


def generate_prediction_report(prediction: Prediction, race_data: RaceData | None = None) -> str:
    sections: list[str] = []
    sections.append(_race_title(prediction.race_info, prediction.race_id, "予想レポート"))
    sections.extend(_overview_lines(prediction.race_info, prediction.race_id))
    sections.append("")
    sections.extend(_format_scoring_config(prediction))
    sections.append("")
    sections.extend(_format_mark_lines(prediction))
    sections.append("")
    sections.extend(_format_prediction_strategy(prediction))
    sections.append("")
    sections.extend(_format_bets(prediction))
    sections.append("")
    sections.extend(_format_top_horses_table(prediction, race_data=race_data))
    sections.append("")
    sections.extend(_format_borderline_recovery(prediction))
    sections.append("")
    sections.extend(_format_deep_analyses(prediction))
    sections.append("")
    sections.extend(_format_pace_analyses(prediction))
    sections.append("")
    sections.extend(_format_race_simulation(prediction))
    sections.append("")
    sections.extend(_format_race_level_analyses(prediction))
    sections.append("")
    sections.extend(_format_pedigree_analyses(prediction))
    sections.append("")
    sections.extend(_format_risks(prediction))
    sections.append("")
    sections.extend(_format_lessons(prediction))
    return "\n".join(sections) + "\n"


def generate_review_report(
    prediction: Prediction,
    review: Review,
    result_data: ResultData | None = None,
    race_data: RaceData | None = None,
) -> str:
    sections: list[str] = []
    sections.append(_race_title(prediction.race_info, prediction.race_id, "回顧レポート"))
    sections.extend(_overview_lines(prediction.race_info, prediction.race_id))
    sections.append("")
    sections.extend(_format_mark_lines(prediction))
    sections.append("")
    sections.extend(_format_actual_result_lines(prediction, result_data=result_data, race_data=race_data))
    sections.append("")
    sections.extend(_format_review_results(review, result_data=result_data))
    sections.append("")
    sections.extend(_format_review_bet_table(review))
    sections.append("")
    sections.extend(_format_simulation_review(review))
    sections.append("")
    sections.extend(_format_simple_list("## 良かった点", review.good_points))
    sections.append("")
    sections.extend(_format_simple_list("## 反省点", review.bad_points))
    sections.append("")
    sections.extend(
        _format_simple_list(
            "## 次回 lesson",
            [
                f"{lesson.course}/{lesson.surface}/{lesson.distance}m/{lesson.track_condition} "
                f"[{lesson.confidence}] {lesson.lesson}"
                for lesson in review.lessons
            ],
        )
    )
    return "\n".join(sections) + "\n"


def save_report(markdown_text: str, output_path: str | Path) -> Path:
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(markdown_text, encoding="utf-8")
    return final_path

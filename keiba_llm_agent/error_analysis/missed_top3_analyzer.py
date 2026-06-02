from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from keiba_llm_agent.backtest.scoring_comparator import calculate_weighted_score, result_top3_list
from keiba_llm_agent.config.scoring_config import effective_scoring_weights, resolve_scoring_config
from keiba_llm_agent.scoring.borderline_recovery import apply_top5_borderline_recovery
from keiba_llm_agent.schemas.prediction import HorseScore, Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _load_prediction(path: Path) -> Prediction:
    return Prediction.model_validate_json(path.read_text(encoding="utf-8"))


def _load_result(path: Path) -> ResultData | None:
    if not path.exists():
        return None
    return ResultData.model_validate_json(path.read_text(encoding="utf-8"))


def _load_review(path: Path) -> Review | None:
    if not path.exists():
        return None
    return Review.model_validate_json(path.read_text(encoding="utf-8"))


def _load_race_data(path: Path) -> RaceData | None:
    if not path.exists():
        return None
    return RaceData.from_json_file(path)


def _collect_predictions_in_period(
    predictions_dir: Path,
    from_date: str,
    to_date: str,
) -> tuple[list[Prediction], list[str]]:
    predictions: list[Prediction] = []
    warnings: list[str] = []
    if not predictions_dir.exists():
        return predictions, warnings

    for path in sorted(predictions_dir.glob("*.json")):
        prediction = _load_prediction(path)
        if prediction.race_info is None or not prediction.race_info.race_date:
            warnings.append(f"prediction skipped because race_info.race_date is missing: race_id={prediction.race_id}")
            continue
        if from_date <= prediction.race_info.race_date <= to_date:
            predictions.append(prediction)
    return predictions, warnings


def _build_race_horse_map(race_data: RaceData | None) -> dict[int, object]:
    if race_data is None:
        return {}
    return {horse.horse_no: horse for horse in race_data.horses}


def _score_horse(horse_score: HorseScore, scoring_config: object) -> float:
    if isinstance(scoring_config, dict):
        pedigree_weight = scoring_config["pedigree_weight"]
        race_level_weight = scoring_config["race_level_weight"]
        pace_weight = scoring_config["pace_weight"]
    else:
        pedigree_weight = scoring_config.pedigree_weight
        race_level_weight = scoring_config.race_level_weight
        pace_weight = scoring_config.pace_weight
    return calculate_weighted_score(
        horse_score,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
    )


def _rank_horses(horse_scores: list[HorseScore], scoring_config: object) -> list[tuple[int, HorseScore, float]]:
    ranked: list[tuple[int, HorseScore, float]] = []
    for horse_score in horse_scores:
        ranked.append((horse_score.horse_no, horse_score, _score_horse(horse_score, scoring_config)))
    ranked.sort(key=lambda item: (-item[2], item[0]))
    return ranked


def _make_scored_horse_scores(prediction: Prediction, scoring_config: object) -> list[HorseScore]:
    scored_horses: list[HorseScore] = []
    for horse_score in prediction.horse_scores:
        scored_horses.append(horse_score.model_copy(update={"total_score": _score_horse(horse_score, scoring_config)}))
    return scored_horses


def _effective_weights_for_prediction(prediction: Prediction, scoring_config: object) -> dict[str, float]:
    return effective_scoring_weights(
        scoring_config,
        surface=prediction.race_info.surface if prediction.race_info else None,
        field_size=len(prediction.horse_scores),
    )


def _find_analysis(prediction: Prediction, field_name: str, horse_no: int) -> object | None:
    analyses = getattr(prediction, field_name, [])
    for analysis in analyses:
        if getattr(analysis, "horse_no", None) == horse_no:
            return analysis
    return None


def _is_filtered_out(popularity: int | None, finish: int, min_popularity: int | None, finish_filter: int | None) -> bool:
    if min_popularity is not None:
        if popularity is None or popularity < min_popularity:
            return True
    if finish_filter is not None and finish != finish_filter:
        return True
    return False


def _build_miss_categories(
    *,
    predicted_rank: int | None,
    score_gap_to_top5: float | None,
    popularity: int | None,
    finish: int,
    deep_analysis: object | None,
    pedigree_analysis: object | None,
    race_level_analysis: object | None,
    pace_analysis: object | None,
    top_n: int,
    data_missing: bool,
    horse_score: HorseScore | None,
) -> list[str]:
    categories: list[str] = []
    if data_missing:
        categories.append("DATA_MISSING")
        return categories

    deep_positive = set(getattr(deep_analysis, "positive_flags", []))
    deep_risk = set(getattr(deep_analysis, "risk_flags", []))
    pedigree_positive = set(getattr(pedigree_analysis, "positive_flags", []))
    pedigree_risk = set(getattr(pedigree_analysis, "risk_flags", []))
    race_level_positive = set(getattr(race_level_analysis, "positive_flags", []))
    race_level_risk = set(getattr(race_level_analysis, "risk_flags", []))
    pace_positive = set(getattr(pace_analysis, "positive_flags", []))
    pace_risk = set(getattr(pace_analysis, "risk_flags", []))

    if predicted_rank is not None and (predicted_rank > max(8, top_n + 3) or (score_gap_to_top5 is not None and score_gap_to_top5 > 2.0)):
        categories.append("LOW_SCORE_OVERALL")
    if predicted_rank == top_n + 1 or (score_gap_to_top5 is not None and score_gap_to_top5 <= 1.0):
        categories.append("JUST_BELOW_TOP5")
    if popularity is not None and popularity >= 6 and finish <= 3:
        categories.append("ODDS_UNDERESTIMATED")
    if popularity is not None and popularity <= 3:
        categories.append("POPULAR_BUT_MISSED")
    if {"RECENT_FORM_STRONG", "RECENT_FORM_STABLE"} & deep_positive:
        categories.append("RECENT_FORM_UNDERESTIMATED")
    if "DISTANCE_FIT" in deep_positive:
        categories.append("DISTANCE_FIT_UNDERESTIMATED")
    if "COURSE_FIT" in deep_positive:
        categories.append("COURSE_FIT_UNDERESTIMATED")
    if {"PEDIGREE_DISTANCE_FIT", "PEDIGREE_STAMINA_FIT", "PEDIGREE_SURFACE_FIT"} & pedigree_positive:
        categories.append("PEDIGREE_FIT_UNDERESTIMATED")
    if {"HEAD_TO_HEAD_POSITIVE", "LARGE_FIELD_GOOD_RUN", "UNDERVALUED_GOOD_RUN", "VALUE_WIN"} & race_level_positive:
        categories.append("RACE_LEVEL_UNDERESTIMATED")
    if {"PACE_FIT", "STALKER_ADVANTAGE", "CLOSING_SPEED"} & pace_positive:
        categories.append("PACE_FIT_UNDERESTIMATED")
    strong_risk = (
        (horse_score is not None and horse_score.scores.risk <= -4)
        or len(deep_risk) >= 4
        or len(pace_risk) >= 3
    )
    positive_signal_count = sum(
        1
        for condition in (
            len(deep_positive) >= 2,
            len(pedigree_positive) >= 2,
            len(race_level_positive) >= 1,
            len(pace_positive) >= 1,
            popularity is not None and popularity <= 3,
        )
        if condition
    )
    if strong_risk and positive_signal_count >= 2:
        categories.append("RISK_OVER_PENALIZED")

    return list(dict.fromkeys(categories))


def _build_analysis_comment(categories: list[str]) -> str:
    category_set = set(categories)
    if "DATA_MISSING" in category_set:
        return "判断材料が不足しており分析は限定的。"
    if "JUST_BELOW_TOP5" in category_set:
        return "TopN直下で、軽微な補正で拾えた可能性。"
    if "ODDS_UNDERESTIMATED" in category_set and "RACE_LEVEL_UNDERESTIMATED" in category_set:
        return "人気薄ながら実際は好走。race_levelまたは近走内容の評価不足の可能性。"
    if "PEDIGREE_FIT_UNDERESTIMATED" in category_set:
        return "血統・距離適性はあったがスコアに十分反映されなかった。"
    if "PACE_FIT_UNDERESTIMATED" in category_set:
        return "展開適性はあったが順位評価に十分つながらなかった。"
    if "RACE_LEVEL_UNDERESTIMATED" in category_set:
        return "相手関係やレースレベル面の加点が不足した可能性。"
    if "POPULAR_BUT_MISSED" in category_set:
        return "市場評価の高い馬を取りこぼしており、基礎評価の見直し余地。"
    if "RISK_OVER_PENALIZED" in category_set:
        return "リスク要因を強く見すぎた可能性。"
    if "LOW_SCORE_OVERALL" in category_set:
        return "全体スコアが低く、根本評価の不足が大きい。"
    return "総合評価のあと一歩が足りず、TopNに届かなかった。"


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def run_missed_top3_analysis(
    *,
    from_date: str,
    to_date: str,
    predictions_dir: Path,
    results_dir: Path,
    reviews_dir: Path,
    race_data_dir: Path | None = None,
    scoring_mode: str = "candidate_default",
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    min_popularity: int | None = None,
    finish_filter: int | None = None,
    top_n: int = 5,
    simulate_borderline_recovery: bool = False,
) -> dict[str, object]:
    scoring_config, scoring_warnings = resolve_scoring_config(
        scoring_mode=scoring_mode,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
    )
    warnings = list(scoring_warnings)
    predictions, prediction_warnings = _collect_predictions_in_period(predictions_dir, from_date, to_date)
    warnings.extend(prediction_warnings)

    race_count = len(predictions)
    reviewed_race_count = 0
    total_top3_horses = 0
    captured_top3_horses = 0
    missed_cases: list[dict[str, object]] = []
    race_details: list[dict[str, object]] = []
    category_counter: Counter[str] = Counter()
    borderline_rank6_count = 0
    recovery_candidate_count = 0
    theoretically_recoverable_top3_count = 0

    for prediction in predictions:
        result_data = _load_result(results_dir / f"{prediction.race_id}.json")
        review = _load_review(reviews_dir / f"{prediction.race_id}.json")
        if result_data is None:
            warnings.append(f"result missing for race_id={prediction.race_id}")
            continue
        if review is None:
            warnings.append(f"review missing for race_id={prediction.race_id}")
            continue

        reviewed_race_count += 1
        race_data = _load_race_data((race_data_dir or Path()) / f"{prediction.race_id}.json") if race_data_dir is not None else None
        race_horse_map = _build_race_horse_map(race_data)
        effective_weights = _effective_weights_for_prediction(prediction, scoring_config)
        ranked = _rank_horses(prediction.horse_scores, effective_weights)
        rank_map = {horse_no: index + 1 for index, (horse_no, _, _) in enumerate(ranked)}
        score_map = {horse_no: score for horse_no, _, score in ranked}
        horse_score_map = {horse_score.horse_no: horse_score for horse_score in prediction.horse_scores}
        predicted_top_n = [horse_no for horse_no, _, _ in ranked[:top_n]]
        top_n_cutoff_score = ranked[top_n - 1][2] if len(ranked) >= top_n else (ranked[-1][2] if ranked else None)
        recovery_result = {
            "recovery_applied": False,
            "recovery_cases": [],
            "adjusted_horse_scores": _make_scored_horse_scores(prediction, effective_weights),
        }
        if simulate_borderline_recovery:
            recovery_result = apply_top5_borderline_recovery(
                recovery_result["adjusted_horse_scores"],
                prediction.deep_analyses,
                prediction.pedigree_analyses,
                prediction.race_level_analyses,
                prediction.pace_analyses,
                prediction.race_info,
                prediction.scoring_config,
                enabled=True,
            )
        recovered_top_n = [
            horse_score.horse_no
            for horse_score in sorted(
                recovery_result["adjusted_horse_scores"],
                key=lambda item: (-item.total_score, item.horse_no),
            )[:top_n]
        ]

        result_top3 = result_top3_list(result_data)
        filtered_result_top3: list[int] = []
        filtered_missed: list[int] = []
        for finish, horse_no in enumerate(result_top3, start=1):
            horse_entry = race_horse_map.get(horse_no)
            popularity = getattr(horse_entry, "popularity", None)
            if _is_filtered_out(popularity, finish, min_popularity, finish_filter):
                continue
            filtered_result_top3.append(horse_no)
            total_top3_horses += 1
            if horse_no in predicted_top_n:
                captured_top3_horses += 1
                continue
            filtered_missed.append(horse_no)
            horse_score = horse_score_map.get(horse_no)
            horse_name = (
                horse_score.horse_name if horse_score is not None
                else getattr(horse_entry, "horse_name", None)
                or f"Horse{horse_no}"
            )
            predicted_rank = rank_map.get(horse_no)
            predicted_score = score_map.get(horse_no)
            score_gap = (
                round(top_n_cutoff_score - predicted_score, 1)
                if top_n_cutoff_score is not None and predicted_score is not None and predicted_rank and predicted_rank > top_n
                else None
            )
            if predicted_rank == 6 and score_gap is not None and score_gap <= 1.0:
                borderline_rank6_count += 1
            deep_analysis = _find_analysis(prediction, "deep_analyses", horse_no)
            pedigree_analysis = _find_analysis(prediction, "pedigree_analyses", horse_no)
            race_level_analysis = _find_analysis(prediction, "race_level_analyses", horse_no)
            pace_analysis = _find_analysis(prediction, "pace_analyses", horse_no)
            categories = _build_miss_categories(
                predicted_rank=predicted_rank,
                score_gap_to_top5=score_gap,
                popularity=popularity,
                finish=finish,
                deep_analysis=deep_analysis,
                pedigree_analysis=pedigree_analysis,
                race_level_analysis=race_level_analysis,
                pace_analysis=pace_analysis,
                top_n=top_n,
                data_missing=horse_score is None or predicted_score is None,
                horse_score=horse_score,
            )
            category_counter.update(categories)
            recovery_candidate = False
            if simulate_borderline_recovery and recovery_result["recovery_applied"]:
                recovery_candidate = any(
                    (case["horse_no"] if isinstance(case, dict) else case.horse_no) == horse_no
                    for case in recovery_result["recovery_cases"]
                )
                if recovery_candidate:
                    recovery_candidate_count += 1
            missed_cases.append(
                {
                    "race_id": prediction.race_id,
                    "race_name": prediction.race_info.race_name if prediction.race_info and prediction.race_info.race_name else prediction.race_id,
                    "horse_no": horse_no,
                    "horse_name": horse_name,
                    "finish": finish,
                    "predicted_rank": predicted_rank,
                    "predicted_score": predicted_score,
                    "top5_cutoff_score": top_n_cutoff_score,
                    "score_gap_to_top5": score_gap,
                    "odds": getattr(horse_entry, "odds", None),
                    "popularity": popularity,
                    "miss_categories": categories,
                    "recovery_candidate": recovery_candidate,
                    "analysis_comment": _build_analysis_comment(categories),
                }
            )

        predicted_capture_count = len(set(predicted_top_n).intersection(filtered_result_top3))
        recovered_capture_count = len(set(recovered_top_n).intersection(filtered_result_top3))
        theoretically_recoverable_top3_count += max(0, recovered_capture_count - predicted_capture_count)

        race_details.append(
            {
                "race_id": prediction.race_id,
                "race_name": prediction.race_info.race_name if prediction.race_info and prediction.race_info.race_name else prediction.race_id,
                "result_top3": filtered_result_top3,
                "predicted_topN": predicted_top_n,
                "missed_top3": filtered_missed,
            }
        )

    missed_top3_horses = len(missed_cases)
    avg_captured = round(captured_top3_horses / reviewed_race_count, 2) if reviewed_race_count else 0.0
    capture_rate = round(captured_top3_horses / total_top3_horses, 3) if total_top3_horses else 0.0
    category_counts = dict(sorted(category_counter.items(), key=lambda item: (-item[1], item[0])))
    expected_avg_after_recovery = (
        round((captured_top3_horses + theoretically_recoverable_top3_count) / reviewed_race_count, 2)
        if reviewed_race_count
        else 0.0
    )
    risk_over_penalized_count = category_counts.get("RISK_OVER_PENALIZED", 0)
    if missed_top3_horses > 0 and (risk_over_penalized_count / missed_top3_horses) > 0.7:
        warnings.append("RISK_OVER_PENALIZED classification may still be too broad.")

    return {
        "period": {"from": from_date, "to": to_date},
        "analysis_config": {
            "scoring_mode": scoring_config.scoring_mode,
            "pedigree_weight": scoring_config.pedigree_weight,
            "race_level_weight": scoring_config.race_level_weight,
            "pace_weight": scoring_config.pace_weight,
            "conditional_weight_profile": scoring_config.conditional_weight_profile,
            "top_n": top_n,
            "min_popularity": min_popularity,
            "finish_filter": finish_filter,
        },
        "summary": {
            "race_count": race_count,
            "reviewed_race_count": reviewed_race_count,
            "total_top3_horses": total_top3_horses,
            "captured_top3_horses": captured_top3_horses,
            "missed_top3_horses": missed_top3_horses,
            "avg_captured_top3_per_race": avg_captured,
            "capture_rate": capture_rate,
            "top_n": top_n,
            "borderline_rank6_count": borderline_rank6_count,
            "recovery_candidate_count": recovery_candidate_count,
            "theoretically_recoverable_top3_count": theoretically_recoverable_top3_count,
            "expected_avg_captured_top3_per_race_after_recovery": expected_avg_after_recovery,
        },
        "category_counts": category_counts,
        "missed_cases": missed_cases,
        "race_details": race_details,
        "warnings": warnings,
    }


def generate_missed_top3_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    config = report["analysis_config"]
    missed_cases = report["missed_cases"]
    total_missed = max(int(summary["missed_top3_horses"]), 1)
    lines = [
        "# Missed Top3 Error Analysis",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}" if "period" in report else "",
        f"- reviewed races: {summary['reviewed_race_count']}",
        f"- top_n: {summary['top_n']}",
        f"- scoring_mode: {config['scoring_mode']}",
        f"- conditional_weight_profile: {config.get('conditional_weight_profile', 'none')}",
        f"- total top3 horses: {summary['total_top3_horses']}",
        f"- captured top3 horses: {summary['captured_top3_horses']}",
        f"- missed top3 horses: {summary['missed_top3_horses']}",
        f"- avg captured top3 per race: {summary['avg_captured_top3_per_race']:.2f}",
        f"- capture rate: {_format_rate(summary['capture_rate'])}",
        f"- borderline rank=6 / gap<=1.0: {summary.get('borderline_rank6_count', 0)}",
        f"- recovery candidates: {summary.get('recovery_candidate_count', 0)}",
        f"- theoretically recoverable top3: {summary.get('theoretically_recoverable_top3_count', 0)}",
        f"- expected avg after recovery: {summary.get('expected_avg_captured_top3_per_race_after_recovery', 0.0):.2f}",
        "",
        "## Miss Category Ranking",
        "| category | count | ratio |",
        "| --- | ---: | ---: |",
    ]
    for category, count in report["category_counts"].items():
        lines.append(f"| {category} | {count} | {_format_rate(count / total_missed)} |")
    if not report["category_counts"]:
        lines.append("| なし | 0 | 0.0% |")

    high_priority = [
        case
        for case in missed_cases
        if case["finish"] == 1
        or case["predicted_rank"] == summary["top_n"] + 1
        or (case["popularity"] is not None and case["popularity"] <= 3)
        or (case["score_gap_to_top5"] is not None and case["score_gap_to_top5"] <= 1.0)
    ]
    high_priority.sort(
        key=lambda case: (
            case["finish"],
            case["predicted_rank"] if case["predicted_rank"] is not None else 999,
            case["score_gap_to_top5"] if case["score_gap_to_top5"] is not None else 999.0,
            case["race_id"],
            case["horse_no"],
        )
    )
    lines.extend(
        [
            "",
            "## High Priority Misses",
            "| race_id | horse_no | horse_name | finish | predicted_rank | odds | popularity | categories | comment |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    if high_priority:
        for case in high_priority:
            odds_text = f"{case['odds']:.1f}" if isinstance(case["odds"], (int, float)) else "-"
            popularity_text = str(case["popularity"]) if case["popularity"] is not None else "-"
            rank_text = str(case["predicted_rank"]) if case["predicted_rank"] is not None else "-"
            lines.append(
                f"| {case['race_id']} | {case['horse_no']} | {case['horse_name']} | {case['finish']} | "
                f"{rank_text} | {odds_text} | {popularity_text} | "
                f"{', '.join(case['miss_categories']) if case['miss_categories'] else 'なし'} | {case['analysis_comment']} |"
            )
    else:
        lines.append("| なし | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Race Details",
            f"| race_id | result_top3 | predicted_top{summary['top_n']} | missed_top3 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for detail in report["race_details"]:
        lines.append(
            f"| {detail['race_id']} | "
            f"{'→'.join(str(item) for item in detail['result_top3']) if detail['result_top3'] else '-'} | "
            f"{'→'.join(str(item) for item in detail['predicted_topN']) if detail['predicted_topN'] else '-'} | "
            f"{'→'.join(str(item) for item in detail['missed_top3']) if detail['missed_top3'] else '-'} |"
        )

    findings: list[str] = []
    category_counts = report["category_counts"]
    if category_counts.get("JUST_BELOW_TOP5", 0) > 0:
        findings.append(
            f"JUST_BELOW_TOP5 は {category_counts.get('JUST_BELOW_TOP5', 0)}件。predicted_rank=6 かつ score_gap<=1.0 は {summary.get('borderline_rank6_count', 0)}件。"
        )
    if category_counts.get("JUST_BELOW_TOP5", 0) > 0:
        findings.append("Top5 cutoff 付近の取りこぼしが多く、軽量補正またはTop6評価の活用余地あり。")
    if category_counts.get("ODDS_UNDERESTIMATED", 0) > 0:
        findings.append("人気薄の激走を拾えていない。穴馬評価ロジックの追加検討。")
    if category_counts.get("RACE_LEVEL_UNDERESTIMATED", 0) > 0:
        findings.append("race_level の評価は有効だが、Top5反映が弱い可能性。")
    if category_counts.get("PEDIGREE_FIT_UNDERESTIMATED", 0) > 0:
        findings.append("血統適性はあるが score への反映が弱い可能性。")
    if category_counts.get("PACE_FIT_UNDERESTIMATED", 0) > 0:
        findings.append("展開適性の評価を report だけでなく ranking に活用する余地あり。")
    if category_counts.get("POPULAR_BUT_MISSED", 0) > 0:
        findings.append("市場評価の高い馬を過小評価している可能性。")
    if summary.get("recovery_candidate_count", 0) > 0:
        findings.append(
            f"Top5境界補正の候補レースは {summary['recovery_candidate_count']}件。理論上の追加captureは {summary['theoretically_recoverable_top3_count']}頭、"
            f"avg captured top3 per race は {summary['expected_avg_captured_top3_per_race_after_recovery']:.2f} まで改善余地。"
        )
    lines.extend(["", "## Findings"])
    if findings:
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.append("- 明確な偏りはまだ少なく、追加サンプルの蓄積が必要。")

    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def save_missed_top3_json(report: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_missed_top3_markdown(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

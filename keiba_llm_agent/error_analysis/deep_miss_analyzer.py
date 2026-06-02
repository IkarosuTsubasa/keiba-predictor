from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from keiba_llm_agent.error_analysis.missed_top3_analyzer import (
    _build_analysis_comment,
    _build_miss_categories,
    _build_race_horse_map,
    _collect_predictions_in_period,
    _effective_weights_for_prediction,
    _find_analysis,
    _format_rate,
    _is_filtered_out,
    _load_race_data,
    _load_result,
    _load_review,
    _rank_horses,
)
from keiba_llm_agent.backtest.scoring_comparator import result_top3_list
from keiba_llm_agent.config.scoring_config import resolve_scoring_config


def _severity_for_rank(predicted_rank: int) -> str:
    if predicted_rank <= 8:
        return "light"
    if predicted_rank <= 12:
        return "moderate"
    return "deep"


def _severity_label(severity: str) -> str:
    return {
        "light": "rank 7-8",
        "moderate": "rank 9-12",
        "deep": "rank 13+",
    }.get(severity, severity)


def _popularity_bucket(popularity: int | None) -> str:
    if popularity is None:
        return "unknown"
    if popularity <= 3:
        return "1-3"
    if popularity <= 6:
        return "4-6"
    if popularity <= 9:
        return "7-9"
    return "10+"


def _score_gap_bucket(score_gap: float | None) -> str:
    if score_gap is None:
        return "unknown"
    if score_gap <= 1.0:
        return "<=1.0"
    if score_gap <= 2.0:
        return "1.1-2.0"
    if score_gap <= 4.0:
        return "2.1-4.0"
    return ">4.0"


def _distance_bucket(distance: int | None) -> str:
    if distance is None:
        return "unknown"
    if distance <= 1400:
        return "short"
    if distance <= 1800:
        return "mile"
    if distance <= 2200:
        return "middle"
    return "long"


def _build_findings(report: dict[str, object]) -> list[str]:
    summary = report["summary"]
    severity_counts = report["severity_counts"]
    category_counts = report["category_counts"]
    findings: list[str] = []

    if severity_counts.get("light", 0) > 0:
        findings.append(
            f"軽度漏馬(rank 7-8) は {severity_counts['light']}件。Top5境界ではなく、Top6-8帯の評価反映を見直す余地。"
        )
    if severity_counts.get("moderate", 0) > 0:
        findings.append(
            f"中度漏馬(rank 9-12) は {severity_counts['moderate']}件。単純な境界補正ではなく、構造的な加点不足の可能性。"
        )
    if severity_counts.get("deep", 0) > 0:
        findings.append(
            f"深度漏馬(rank 13+) は {severity_counts['deep']}件。高波動要因か、基礎評価の大幅不足かを切り分ける必要。"
        )
    if category_counts.get("RACE_LEVEL_UNDERESTIMATED", 0) > 0:
        findings.append("race_level positive があるのに低順位に沈む例があり、加点反映の粒度を再点検したい。")
    if category_counts.get("POPULAR_BUT_MISSED", 0) > 0:
        findings.append("人気馬の取りこぼしがあり、人気上位の安全弁を検討する余地。")
    if category_counts.get("RECENT_FORM_UNDERESTIMATED", 0) > 0:
        findings.append("近走安定・好調シグナルが十分に順位へ反映されていない可能性。")
    if category_counts.get("DISTANCE_FIT_UNDERESTIMATED", 0) > 0:
        findings.append("距離適性の正面シグナルが、低順位漏馬の中で見落とされている。")
    if category_counts.get("PEDIGREE_FIT_UNDERESTIMATED", 0) > 0:
        findings.append("血統適性は説明には出るが、低順位漏馬の救済にはまだ弱い可能性。")
    if category_counts.get("PACE_FIT_UNDERESTIMATED", 0) > 0:
        findings.append("展開適性はあるが、現状は default score に強く入れていないため低順位漏馬として残りやすい。")
    if summary.get("low_rank_top3_horses", 0) == 0:
        findings.append("rank>=7 の実Top3漏馬は確認されず、深い低評価ミスは限定的。")
    return findings


def run_deep_miss_analysis(
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
    low_rank_threshold: int = 7,
    top_n_reference: int = 5,
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
    deep_miss_cases: list[dict[str, object]] = []
    race_details: list[dict[str, object]] = []
    category_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()
    severity_category_counter: dict[str, Counter[str]] = {
        "light": Counter(),
        "moderate": Counter(),
        "deep": Counter(),
    }
    popularity_bucket_counter: Counter[str] = Counter()
    score_gap_bucket_counter: Counter[str] = Counter()
    distance_bucket_counter: Counter[str] = Counter()

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
        predicted_top_n = [horse_no for horse_no, _, _ in ranked[:top_n_reference]]
        top_n_cutoff_score = ranked[top_n_reference - 1][2] if len(ranked) >= top_n_reference else (ranked[-1][2] if ranked else None)
        result_top3 = result_top3_list(result_data)

        filtered_result_top3: list[int] = []
        low_rank_top3: list[int] = []
        for finish, horse_no in enumerate(result_top3, start=1):
            horse_entry = race_horse_map.get(horse_no)
            popularity = getattr(horse_entry, "popularity", None)
            if _is_filtered_out(popularity, finish, min_popularity, finish_filter):
                continue
            filtered_result_top3.append(horse_no)
            total_top3_horses += 1
            if horse_no in predicted_top_n:
                captured_top3_horses += 1

            predicted_rank = rank_map.get(horse_no)
            if predicted_rank is None:
                warnings.append(f"predicted_rank missing for race_id={prediction.race_id}, horse_no={horse_no}")
                continue
            if predicted_rank < low_rank_threshold:
                continue

            low_rank_top3.append(horse_no)
            horse_score = horse_score_map.get(horse_no)
            predicted_score = score_map.get(horse_no)
            score_gap = (
                round(top_n_cutoff_score - predicted_score, 1)
                if top_n_cutoff_score is not None and predicted_score is not None and predicted_rank > top_n_reference
                else None
            )
            severity = _severity_for_rank(predicted_rank)
            horse_name = (
                horse_score.horse_name if horse_score is not None
                else getattr(horse_entry, "horse_name", None)
                or f"Horse{horse_no}"
            )
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
                top_n=top_n_reference,
                data_missing=horse_score is None or predicted_score is None,
                horse_score=horse_score,
            )
            category_counter.update(categories)
            severity_counter.update([severity])
            severity_category_counter[severity].update(categories)
            popularity_bucket_counter.update([_popularity_bucket(popularity)])
            score_gap_bucket_counter.update([_score_gap_bucket(score_gap)])
            distance_bucket_counter.update([_distance_bucket(prediction.race_info.distance if prediction.race_info else None)])
            deep_miss_cases.append(
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
                    "severity": severity,
                    "odds": getattr(horse_entry, "odds", None),
                    "popularity": popularity,
                    "miss_categories": categories,
                    "analysis_comment": _build_analysis_comment(categories),
                }
            )

        race_details.append(
            {
                "race_id": prediction.race_id,
                "race_name": prediction.race_info.race_name if prediction.race_info and prediction.race_info.race_name else prediction.race_id,
                "result_top3": filtered_result_top3,
                "predicted_top5": predicted_top_n,
                "low_rank_top3": low_rank_top3,
                "light_count": sum(
                    1 for horse_no in low_rank_top3 if (rank_map.get(horse_no) or 0) in {7, 8}
                ),
                "moderate_count": sum(
                    1 for horse_no in low_rank_top3 if 9 <= (rank_map.get(horse_no) or 0) <= 12
                ),
                "deep_count": sum(
                    1 for horse_no in low_rank_top3 if (rank_map.get(horse_no) or 0) >= 13
                ),
            }
        )

    low_rank_top3_horses = len(deep_miss_cases)
    avg_captured = round(captured_top3_horses / reviewed_race_count, 2) if reviewed_race_count else 0.0
    capture_rate = round(captured_top3_horses / total_top3_horses, 3) if total_top3_horses else 0.0

    report = {
        "period": {"from": from_date, "to": to_date},
        "analysis_config": {
            "scoring_mode": scoring_config.scoring_mode,
            "pedigree_weight": scoring_config.pedigree_weight,
            "race_level_weight": scoring_config.race_level_weight,
            "pace_weight": scoring_config.pace_weight,
            "conditional_weight_profile": scoring_config.conditional_weight_profile,
            "low_rank_threshold": low_rank_threshold,
            "top_n_reference": top_n_reference,
            "min_popularity": min_popularity,
            "finish_filter": finish_filter,
        },
        "summary": {
            "race_count": race_count,
            "reviewed_race_count": reviewed_race_count,
            "total_top3_horses": total_top3_horses,
            "captured_top3_horses": captured_top3_horses,
            "low_rank_top3_horses": low_rank_top3_horses,
            "avg_captured_top3_per_race": avg_captured,
            "capture_rate": capture_rate,
        },
        "severity_counts": dict(severity_counter),
        "category_counts": dict(sorted(category_counter.items(), key=lambda item: (-item[1], item[0]))),
        "severity_category_counts": {
            severity: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
            for severity, counter in severity_category_counter.items()
        },
        "popularity_bucket_counts": dict(sorted(popularity_bucket_counter.items(), key=lambda item: item[0])),
        "score_gap_bucket_counts": dict(sorted(score_gap_bucket_counter.items(), key=lambda item: item[0])),
        "distance_bucket_counts": dict(sorted(distance_bucket_counter.items(), key=lambda item: item[0])),
        "deep_miss_cases": deep_miss_cases,
        "race_details": race_details,
        "warnings": warnings,
    }
    report["findings"] = _build_findings(report)
    return report


def generate_deep_miss_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    config = report["analysis_config"]
    cases = report["deep_miss_cases"]
    total_cases = max(int(summary["low_rank_top3_horses"]), 1)
    lines = [
        "# Deep Miss Analysis",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}",
        f"- reviewed races: {summary['reviewed_race_count']}",
        f"- scoring_mode: {config['scoring_mode']}",
        f"- conditional_weight_profile: {config.get('conditional_weight_profile', 'none')}",
        f"- low-rank threshold: rank>={config['low_rank_threshold']}",
        f"- total top3 horses: {summary['total_top3_horses']}",
        f"- captured top3 horses: {summary['captured_top3_horses']}",
        f"- low-rank top3 horses: {summary['low_rank_top3_horses']}",
        f"- avg captured top3 per race: {summary['avg_captured_top3_per_race']:.2f}",
        f"- capture rate: {_format_rate(summary['capture_rate'])}",
        "",
        "## Severity Breakdown",
        "| severity | count | ratio |",
        "| --- | ---: | ---: |",
    ]
    for severity in ("light", "moderate", "deep"):
        count = report["severity_counts"].get(severity, 0)
        lines.append(f"| {_severity_label(severity)} | {count} | {_format_rate(count / total_cases)} |")

    lines.extend(
        [
            "",
            "## Miss Category Ranking",
            "| category | count | ratio |",
            "| --- | ---: | ---: |",
        ]
    )
    if report["category_counts"]:
        for category, count in report["category_counts"].items():
            lines.append(f"| {category} | {count} | {_format_rate(count / total_cases)} |")
    else:
        lines.append("| なし | 0 | 0.0% |")

    lines.extend(
        [
            "",
            "## High Priority Deep Misses",
            "| race_id | horse_no | horse_name | finish | predicted_rank | severity | odds | popularity | categories | comment |",
            "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    high_priority = sorted(
        cases,
        key=lambda case: (
            case["predicted_rank"] if case["predicted_rank"] is not None else 999,
            case["finish"],
            case["race_id"],
            case["horse_no"],
        ),
    )
    if high_priority:
        for case in high_priority:
            odds_text = f"{case['odds']:.1f}" if isinstance(case["odds"], (int, float)) else "-"
            pop_text = str(case["popularity"]) if case["popularity"] is not None else "-"
            lines.append(
                f"| {case['race_id']} | {case['horse_no']} | {case['horse_name']} | {case['finish']} | "
                f"{case['predicted_rank']} | {_severity_label(case['severity'])} | {odds_text} | {pop_text} | "
                f"{', '.join(case['miss_categories']) if case['miss_categories'] else 'なし'} | {case['analysis_comment']} |"
            )
    else:
        lines.append("| なし | - | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Support Buckets",
            "| bucket | count |",
            "| --- | ---: |",
        ]
    )
    for bucket_name, counts in (
        ("popularity", report["popularity_bucket_counts"]),
        ("score_gap", report["score_gap_bucket_counts"]),
        ("distance", report["distance_bucket_counts"]),
    ):
        for key, count in counts.items():
            lines.append(f"| {bucket_name}:{key} | {count} |")

    lines.extend(
        [
            "",
            "## Race Details",
            "| race_id | result_top3 | predicted_top5 | low_rank_top3 | light | moderate | deep |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for detail in report["race_details"]:
        lines.append(
            f"| {detail['race_id']} | "
            f"{'→'.join(str(item) for item in detail['result_top3']) if detail['result_top3'] else '-'} | "
            f"{'→'.join(str(item) for item in detail['predicted_top5']) if detail['predicted_top5'] else '-'} | "
            f"{'→'.join(str(item) for item in detail['low_rank_top3']) if detail['low_rank_top3'] else '-'} | "
            f"{detail['light_count']} | {detail['moderate_count']} | {detail['deep_count']} |"
        )

    lines.extend(["", "## Findings"])
    findings = report.get("findings", [])
    if findings:
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.append("- 低排位漏馬は確認されず、深い構造的ミスは限定的。")

    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])

    return "\n".join(lines) + "\n"


def save_deep_miss_json(report: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_deep_miss_markdown(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from keiba_llm_agent.backtest.scoring_comparator import result_top3_list
from keiba_llm_agent.error_analysis.missed_top3_analyzer import (
    _collect_predictions_in_period,
    _load_result,
    _load_review,
)
from keiba_llm_agent.schemas.prediction import HorseScore, Prediction


SCORE_FACTORS = [
    "recent_form",
    "distance_fit",
    "course_fit",
    "track_condition_fit",
    "jockey_fit",
    "risk",
    "base_total_score",
    "pedigree_adjustment_raw",
    "pedigree_adjustment_weighted",
    "race_level_adjustment_raw",
    "race_level_adjustment_weighted",
    "pace_adjustment_raw",
    "pace_adjustment_weighted",
    "borderline_recovery_bonus",
    "total_score",
]


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def _factor_values(horse_score: HorseScore) -> dict[str, float]:
    scores = horse_score.scores
    breakdown = horse_score.score_breakdown
    return {
        "recent_form": float(scores.recent_form),
        "distance_fit": float(scores.distance_fit),
        "course_fit": float(scores.course_fit),
        "track_condition_fit": float(scores.track_condition_fit),
        "jockey_fit": float(scores.jockey_fit),
        "risk": float(scores.risk),
        "base_total_score": float(horse_score.base_total_score),
        "pedigree_adjustment_raw": float(horse_score.pedigree_adjustment.pedigree_adjustment),
        "pedigree_adjustment_weighted": float(breakdown.pedigree_adjustment_weighted),
        "race_level_adjustment_raw": float(horse_score.race_level_adjustment.adjustment),
        "race_level_adjustment_weighted": float(breakdown.race_level_adjustment_weighted),
        "pace_adjustment_raw": float(horse_score.pace_adjustment.adjustment),
        "pace_adjustment_weighted": float(breakdown.pace_adjustment_weighted),
        "borderline_recovery_bonus": float(breakdown.borderline_recovery_bonus),
        "total_score": float(horse_score.total_score),
    }


def _rank_map(prediction: Prediction) -> dict[int, int]:
    ordered = sorted(prediction.horse_scores, key=lambda item: (-item.total_score, item.horse_no))
    return {horse_score.horse_no: index + 1 for index, horse_score in enumerate(ordered)}


def _top5(prediction: Prediction) -> list[int]:
    return [horse_score.horse_no for horse_score in sorted(prediction.horse_scores, key=lambda item: (-item.total_score, item.horse_no))[:5]]


def _new_group_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "top3_count": 0,
        "winner_count": 0,
        "missed_top3_count": 0,
        "values": {factor: [] for factor in SCORE_FACTORS},
    }


def _add_to_group(group: dict[str, Any], factor_values: dict[str, float], *, top3: bool, winner: bool, missed_top3: bool) -> None:
    group["count"] += 1
    group["top3_count"] += 1 if top3 else 0
    group["winner_count"] += 1 if winner else 0
    group["missed_top3_count"] += 1 if missed_top3 else 0
    for factor, value in factor_values.items():
        group["values"][factor].append(value)


def _summarize_group(group: dict[str, Any]) -> dict[str, Any]:
    count = int(group["count"])
    return {
        "count": count,
        "top3_count": int(group["top3_count"]),
        "winner_count": int(group["winner_count"]),
        "missed_top3_count": int(group["missed_top3_count"]),
        "top3_rate": _safe_rate(int(group["top3_count"]), count),
        "winner_rate": _safe_rate(int(group["winner_count"]), count),
        "factor_means": {
            factor: _safe_mean([float(value) for value in values])
            for factor, values in group["values"].items()
        },
    }


def _bucket_for_value(value: float, factor: str) -> str:
    if factor in {"risk"}:
        if value <= -7:
            return "<=-7"
        if value <= -5:
            return "-6..-5"
        if value <= -3:
            return "-4..-3"
        return ">=-2"
    if factor.endswith("_adjustment_raw") or factor.endswith("_adjustment_weighted") or factor == "borderline_recovery_bonus":
        if value < 0:
            return "<0"
        if value == 0:
            return "0"
        if value <= 0.3:
            return "0.1..0.3"
        if value <= 0.7:
            return "0.4..0.7"
        return ">=0.8"
    if factor in {"base_total_score", "total_score"}:
        if value < 10:
            return "<10"
        if value < 20:
            return "10..19.9"
        if value < 30:
            return "20..29.9"
        if value < 40:
            return "30..39.9"
        return ">=40"
    if value <= 2:
        return "0..2"
    if value <= 4:
        return "3..4"
    if value <= 6:
        return "5..6"
    if value <= 8:
        return "7..8"
    return "9..10"


def _bucket_order(bucket: str) -> tuple[int, str]:
    order = {
        "<=-7": 0,
        "-6..-5": 1,
        "-4..-3": 2,
        ">=-2": 3,
        "<0": 0,
        "0": 1,
        "0.1..0.3": 2,
        "0.4..0.7": 3,
        ">=0.8": 4,
        "<10": 0,
        "10..19.9": 1,
        "20..29.9": 2,
        "30..39.9": 3,
        ">=40": 4,
        "0..2": 0,
        "3..4": 1,
        "5..6": 2,
        "7..8": 3,
        "9..10": 4,
    }
    return order.get(bucket, 999), bucket


def _summarize_bucket_groups(bucket_groups: dict[str, dict[str, dict[str, int]]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for factor, buckets in bucket_groups.items():
        rows: list[dict[str, Any]] = []
        for bucket, counters in buckets.items():
            count = counters["count"]
            top3_count = counters["top3_count"]
            rows.append(
                {
                    "bucket": bucket,
                    "count": count,
                    "top3_count": top3_count,
                    "top3_rate": _safe_rate(top3_count, count),
                    "winner_count": counters["winner_count"],
                    "winner_rate": _safe_rate(counters["winner_count"], count),
                    "missed_top3_count": counters["missed_top3_count"],
                }
            )
        output[factor] = sorted(rows, key=lambda row: _bucket_order(str(row["bucket"])))
    return output


def _factor_comparison(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    top3_means = groups["top3"]["factor_means"]
    non_top3_means = groups["non_top3"]["factor_means"]
    winner_means = groups["winner"]["factor_means"]
    missed_means = groups["missed_top3"]["factor_means"]
    captured_means = groups["captured_top3"]["factor_means"]
    rows = []
    for factor in SCORE_FACTORS:
        top3_mean = top3_means.get(factor, 0.0)
        non_top3_mean = non_top3_means.get(factor, 0.0)
        winner_mean = winner_means.get(factor, 0.0)
        missed_mean = missed_means.get(factor, 0.0)
        captured_mean = captured_means.get(factor, 0.0)
        rows.append(
            {
                "factor": factor,
                "top3_mean": top3_mean,
                "non_top3_mean": non_top3_mean,
                "top3_minus_non_top3": round(top3_mean - non_top3_mean, 4),
                "winner_mean": winner_mean,
                "winner_minus_non_top3": round(winner_mean - non_top3_mean, 4),
                "captured_top3_mean": captured_mean,
                "missed_top3_mean": missed_mean,
                "missed_minus_captured_top3": round(missed_mean - captured_mean, 4),
            }
        )
    return sorted(rows, key=lambda row: abs(row["top3_minus_non_top3"]), reverse=True)


def _build_findings(report: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    rows = {row["factor"]: row for row in report["factor_comparison"]}
    bucket_stats = report["bucket_stats"]

    pace_raw = rows.get("pace_adjustment_raw")
    pace_weighted = rows.get("pace_adjustment_weighted")
    if pace_raw and pace_raw["top3_minus_non_top3"] > 0.05 and pace_weighted and abs(pace_weighted["top3_minus_non_top3"]) < 0.001:
        findings.append("pace_adjustment_raw はTop3側で高いが、現在 pace_weight=0 のため ranking には反映されていない。小幅weightの再検証余地。")

    race_level = rows.get("race_level_adjustment_weighted")
    if race_level and race_level["top3_minus_non_top3"] > 0.03:
        findings.append("race_level_adjustment はTop3側で高く、現行の主要補正源として一定の妥当性がある。")

    pedigree = rows.get("pedigree_adjustment_raw")
    if pedigree and pedigree["top3_minus_non_top3"] > 0.05:
        findings.append("pedigree raw はTop3側で高い。現行0.2 weightのままか、軽量範囲で再検証する価値がある。")

    course = rows.get("course_fit")
    if course and abs(course["top3_minus_non_top3"]) < 0.2:
        findings.append("course_fit のTop3識別力は限定的。未経験コースを強く下げすぎていないか確認余地。")

    jockey = rows.get("jockey_fit")
    if jockey and abs(jockey["top3_minus_non_top3"]) < 0.2:
        findings.append("jockey_fit のTop3識別力は限定的。騎手継続だけでは分離力が弱い可能性。")

    risk = rows.get("risk")
    if risk and risk["top3_minus_non_top3"] > 0.2:
        findings.append("risk はTop3側で軽く、基礎risk判定は一定程度機能している。")

    missed = rows.get("missed_top3_mean")
    distance = rows.get("distance_fit")
    if distance and distance["missed_minus_captured_top3"] > 0.3:
        findings.append("missed Top3 の distance_fit が captured Top3 より高く、距離適性の拾い漏れが残る可能性。")

    total_score_buckets = bucket_stats.get("total_score", [])
    if total_score_buckets:
        low_score_top3 = sum(row["top3_count"] for row in total_score_buckets if row["bucket"] in {"<10", "10..19.9"})
        if low_score_top3 > 0:
            findings.append(f"total_score 20未満にもTop3が {low_score_top3}頭あり、低スコア激走は別枠watchlistで扱うべき。")

    if not findings:
        findings.append("明確な単独factorはまだ限定的。複合条件での検証が必要。")
    return findings


def run_score_factor_analysis(
    *,
    from_date: str,
    to_date: str,
    predictions_dir: Path,
    results_dir: Path,
    reviews_dir: Path,
    top_n: int = 5,
) -> dict[str, Any]:
    predictions, prediction_warnings = _collect_predictions_in_period(predictions_dir, from_date, to_date)
    warnings = list(prediction_warnings)
    groups_raw = {
        "all": _new_group_stats(),
        "top3": _new_group_stats(),
        "non_top3": _new_group_stats(),
        "winner": _new_group_stats(),
        "captured_top3": _new_group_stats(),
        "missed_top3": _new_group_stats(),
    }
    bucket_groups: dict[str, dict[str, dict[str, int]]] = {
        factor: defaultdict(lambda: {"count": 0, "top3_count": 0, "winner_count": 0, "missed_top3_count": 0})
        for factor in SCORE_FACTORS
    }
    race_details: list[dict[str, Any]] = []
    race_count = len(predictions)
    reviewed_race_count = 0
    skipped_race_count = 0
    total_horse_count = 0
    total_top3_count = 0
    captured_top3_count = 0
    missed_top3_count = 0

    for prediction in predictions:
        result_data = _load_result(results_dir / f"{prediction.race_id}.json")
        review = _load_review(reviews_dir / f"{prediction.race_id}.json")
        if result_data is None:
            warnings.append(f"result missing for race_id={prediction.race_id}")
            skipped_race_count += 1
            continue
        if review is None:
            warnings.append(f"review missing for race_id={prediction.race_id}")
            skipped_race_count += 1
            continue

        reviewed_race_count += 1
        result_top3 = result_top3_list(result_data)
        result_top3_set = set(result_top3)
        winner = result_data.result.first
        predicted_top_n = _top5(prediction)[:top_n]
        rank_map = _rank_map(prediction)
        race_top3_captured = len(set(predicted_top_n) & result_top3_set)
        total_top3_count += len(result_top3)
        captured_top3_count += race_top3_captured
        missed_top3_count += len([horse_no for horse_no in result_top3 if horse_no not in predicted_top_n])
        total_horse_count += len(prediction.horse_scores)

        for horse_score in prediction.horse_scores:
            values = _factor_values(horse_score)
            is_top3 = horse_score.horse_no in result_top3_set
            is_winner = horse_score.horse_no == winner
            is_captured_top3 = is_top3 and horse_score.horse_no in predicted_top_n
            is_missed_top3 = is_top3 and horse_score.horse_no not in predicted_top_n
            _add_to_group(groups_raw["all"], values, top3=is_top3, winner=is_winner, missed_top3=is_missed_top3)
            if is_top3:
                _add_to_group(groups_raw["top3"], values, top3=True, winner=is_winner, missed_top3=is_missed_top3)
            else:
                _add_to_group(groups_raw["non_top3"], values, top3=False, winner=False, missed_top3=False)
            if is_winner:
                _add_to_group(groups_raw["winner"], values, top3=True, winner=True, missed_top3=is_missed_top3)
            if is_captured_top3:
                _add_to_group(groups_raw["captured_top3"], values, top3=True, winner=is_winner, missed_top3=False)
            if is_missed_top3:
                _add_to_group(groups_raw["missed_top3"], values, top3=True, winner=is_winner, missed_top3=True)

            for factor, value in values.items():
                bucket = _bucket_for_value(value, factor)
                bucket_groups[factor][bucket]["count"] += 1
                bucket_groups[factor][bucket]["top3_count"] += 1 if is_top3 else 0
                bucket_groups[factor][bucket]["winner_count"] += 1 if is_winner else 0
                bucket_groups[factor][bucket]["missed_top3_count"] += 1 if is_missed_top3 else 0

        race_details.append(
            {
                "race_id": prediction.race_id,
                "race_name": prediction.race_info.race_name if prediction.race_info and prediction.race_info.race_name else prediction.race_id,
                "result_top3": result_top3,
                f"predicted_top{top_n}": predicted_top_n,
                "captured_top3_count": race_top3_captured,
                "missed_top3": [horse_no for horse_no in result_top3 if horse_no not in predicted_top_n],
                "winner_rank": rank_map.get(winner),
                "top3_ranks": [rank_map.get(horse_no) for horse_no in result_top3],
            }
        )

    groups = {group_name: _summarize_group(group) for group_name, group in groups_raw.items()}
    report = {
        "period": {"from": from_date, "to": to_date},
        "analysis_config": {"top_n": top_n},
        "summary": {
            "race_count": race_count,
            "reviewed_race_count": reviewed_race_count,
            "skipped_race_count": skipped_race_count,
            "horse_count": total_horse_count,
            "total_top3_count": total_top3_count,
            "captured_top3_count": captured_top3_count,
            "missed_top3_count": missed_top3_count,
            "avg_captured_top3_per_race": round(captured_top3_count / reviewed_race_count, 4) if reviewed_race_count else 0.0,
            "capture_rate": _safe_rate(captured_top3_count, total_top3_count),
        },
        "group_stats": groups,
        "factor_comparison": _factor_comparison(groups),
        "bucket_stats": _summarize_bucket_groups(bucket_groups),
        "race_details": race_details,
        "warnings": warnings,
    }
    report["findings"] = _build_findings(report)
    return report


def generate_score_factor_analysis_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Score Factor Analysis",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}",
        f"- reviewed races: {summary['reviewed_race_count']}",
        f"- horse_count: {summary['horse_count']}",
        f"- total_top3_count: {summary['total_top3_count']}",
        f"- captured_top3_count: {summary['captured_top3_count']}",
        f"- missed_top3_count: {summary['missed_top3_count']}",
        f"- avg captured top3 per race: {summary['avg_captured_top3_per_race']:.2f}",
        f"- capture rate: {_format_rate(summary['capture_rate'])}",
        "",
        "## Factor Comparison",
        "| factor | Top3 avg | NonTop3 avg | diff | Winner avg | MissedTop3 avg | CapturedTop3 avg | missed-captured |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["factor_comparison"]:
        lines.append(
            f"| {row['factor']} | {row['top3_mean']:.3f} | {row['non_top3_mean']:.3f} | {row['top3_minus_non_top3']:.3f} | "
            f"{row['winner_mean']:.3f} | {row['missed_top3_mean']:.3f} | {row['captured_top3_mean']:.3f} | {row['missed_minus_captured_top3']:.3f} |"
        )

    key_factors = [
        "recent_form",
        "distance_fit",
        "course_fit",
        "track_condition_fit",
        "jockey_fit",
        "risk",
        "race_level_adjustment_raw",
        "pace_adjustment_raw",
        "pedigree_adjustment_raw",
        "total_score",
    ]
    lines.extend(["", "## Bucket Top3 Rates"])
    for factor in key_factors:
        rows = report["bucket_stats"].get(factor, [])
        lines.extend(
            [
                f"### {factor}",
                "| bucket | count | Top3 | Top3率 | Winner率 | MissedTop3 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['bucket']} | {row['count']} | {row['top3_count']} | {_format_rate(row['top3_rate'])} | "
                f"{_format_rate(row['winner_rate'])} | {row['missed_top3_count']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Race Details",
            "| race_id | result_top3 | predicted_topN | captured | missed_top3 | winner_rank | top3_ranks |",
            "| --- | --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    top_n_key = f"predicted_top{report['analysis_config']['top_n']}"
    for detail in report["race_details"]:
        lines.append(
            f"| {detail['race_id']} | {'→'.join(str(item) for item in detail['result_top3'])} | "
            f"{'→'.join(str(item) for item in detail[top_n_key])} | {detail['captured_top3_count']} | "
            f"{'→'.join(str(item) for item in detail['missed_top3']) if detail['missed_top3'] else '-'} | "
            f"{detail['winner_rank'] if detail['winner_rank'] is not None else '-'} | "
            f"{'→'.join(str(item) for item in detail['top3_ranks'])} |"
        )

    lines.extend(["", "## Findings"])
    for finding in report.get("findings", []):
        lines.append(f"- {finding}")

    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def save_score_factor_analysis_json(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_score_factor_analysis_markdown(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

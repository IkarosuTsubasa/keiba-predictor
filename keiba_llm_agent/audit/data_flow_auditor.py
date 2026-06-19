from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    normalized = value.replace("/", "-").strip()
    try:
        return date.fromisoformat(normalized[:10])
    except ValueError:
        return None


def _issue(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def _round(value: float) -> float:
    return round(value, 4)


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _load_model(model_cls: type, path: Path) -> tuple[Any | None, dict[str, Any]]:
    info = _file_info(path)
    if not path.exists():
        info["parse_ok"] = False
        info["error"] = "missing"
        return None, info
    try:
        if model_cls is RaceData:
            model = RaceData.from_json_file(path)
        else:
            model = model_cls.model_validate_json(path.read_text(encoding="utf-8"))
        info["parse_ok"] = True
        return model, info
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        info["parse_ok"] = False
        info["error"] = str(exc)
        return None, info


def _audit_race_data(race_id: str, race_data: RaceData | None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if race_data is None:
        return {"source": "netkeiba shutuba + horse recent runs enrichment", "status": "missing"}, [
            _issue("error", "RACE_DATA_MISSING", "race_data.json が見つからない。")
        ]

    race_info = race_data.race_info
    race_date = _parse_date(race_info.race_date)
    horse_count = len(race_data.horses)
    horse_nos = [horse.horse_no for horse in race_data.horses]
    duplicate_horse_nos = sorted({horse_no for horse_no in horse_nos if horse_nos.count(horse_no) > 1})
    missing_horse_id = [horse.horse_no for horse in race_data.horses if not horse.horse_id]
    missing_odds = [horse.horse_no for horse in race_data.horses if horse.odds is None]
    missing_popularity = [horse.horse_no for horse in race_data.horses if horse.popularity is None]
    missing_jockey = [horse.horse_no for horse in race_data.horses if not horse.jockey]
    empty_recent_runs = [horse.horse_no for horse in race_data.horses if not horse.recent_runs]
    recent_run_counts = [len(horse.recent_runs) for horse in race_data.horses]
    future_leakage: list[dict[str, Any]] = []
    invalid_recent_run_ids: list[dict[str, Any]] = []

    for horse in race_data.horses:
        for run in horse.recent_runs:
            run_date = _parse_date(run.date)
            if race_date is not None and run_date is not None and run_date >= race_date:
                future_leakage.append(
                    {
                        "horse_no": horse.horse_no,
                        "horse_name": horse.horse_name,
                        "recent_run_race_id": run.race_id,
                        "recent_run_date": run.date,
                    }
                )
            if run.race_id is not None and len(str(run.race_id)) < 8:
                invalid_recent_run_ids.append(
                    {
                        "horse_no": horse.horse_no,
                        "horse_name": horse.horse_name,
                        "recent_run_race_id": run.race_id,
                    }
                )

    if race_info.race_id != race_id:
        issues.append(_issue("error", "RACE_ID_MISMATCH", f"race_data.race_info.race_id={race_info.race_id} が race_id と一致しない。"))
    if race_info.race_date is None:
        issues.append(_issue("error", "RACE_DATE_MISSING", "race_data.race_info.race_date が欠損。"))
    if horse_count == 0:
        issues.append(_issue("error", "HORSES_EMPTY", "race_data.horses が空。"))
    if duplicate_horse_nos:
        issues.append(_issue("error", "DUPLICATE_HORSE_NO", f"重複馬番: {duplicate_horse_nos}"))
    if empty_recent_runs:
        issues.append(_issue("warning", "RECENT_RUNS_MISSING", f"recent_runs 空: {empty_recent_runs}"))
    if future_leakage:
        issues.append(_issue("error", "RECENT_RUN_DATE_LEAKAGE", "recent_runs に本レース日以降のデータが含まれる。"))
    if invalid_recent_run_ids:
        issues.append(_issue("warning", "RECENT_RUN_RACE_ID_INVALID", "recent_runs に短すぎる race_id が含まれる。"))

    return (
        {
            "source": "netkeiba shutuba page; recent_runs from horse pages before prediction",
            "status": "ok",
            "race_info": race_info.model_dump(),
            "horse_count": horse_count,
            "missing_horse_id_count": len(missing_horse_id),
            "missing_horse_id_horses": missing_horse_id,
            "missing_odds_count": len(missing_odds),
            "missing_popularity_count": len(missing_popularity),
            "missing_jockey_count": len(missing_jockey),
            "empty_recent_runs_count": len(empty_recent_runs),
            "avg_recent_runs": _round(sum(recent_run_counts) / horse_count) if horse_count else 0.0,
            "future_leakage_count": len(future_leakage),
            "future_leakage_examples": future_leakage[:10],
            "invalid_recent_run_id_count": len(invalid_recent_run_ids),
            "field_sources": {
                "race_info": "netkeiba shutuba parser",
                "horses": "netkeiba shutuba parser",
                "odds_popularity": "netkeiba shutuba page snapshot; reference only unless market_signal is enabled",
                "recent_runs": "netkeiba horse page enrichment",
            },
        },
        issues,
    )


def _audit_prediction(
    race_id: str,
    prediction: Prediction | None,
    race_data: RaceData | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if prediction is None:
        return {"source": "rule scoring pipeline + optional LLM text generation", "status": "missing"}, [
            _issue("error", "PREDICTION_MISSING", "prediction.json が見つからない。")
        ]

    if prediction.race_id != race_id:
        issues.append(_issue("error", "PREDICTION_RACE_ID_MISMATCH", f"prediction.race_id={prediction.race_id} が race_id と一致しない。"))
    horse_scores = prediction.horse_scores
    score_horse_nos = {horse.horse_no for horse in horse_scores}
    marks_missing = [mark for mark in ("◎", "○", "▲", "△", "☆") if mark not in prediction.marks]
    mark_horses_missing = [
        horse_no for horse_no in prediction.marks.values()
        if horse_no and horse_no not in score_horse_nos
    ]
    sorted_by_total = sorted(horse_scores, key=lambda item: (-item.total_score, item.horse_no))
    expected_top5 = [horse.horse_no for horse in sorted_by_total[:5]]
    expected_top5_by_mark_rule = expected_top5
    actual_marks_top5 = [prediction.marks.get(mark, 0) for mark in ("◎", "○", "▲", "△", "☆")]
    score_breakdown_mismatches: list[dict[str, Any]] = []
    for horse in horse_scores:
        breakdown = horse.score_breakdown
        expected_total = _round(
            breakdown.base_total_score
            + breakdown.pedigree_adjustment_weighted
            + breakdown.race_level_adjustment_weighted
            + breakdown.pace_adjustment_weighted
            + breakdown.borderline_recovery_bonus
        )
        actual_total = _round(horse.total_score)
        breakdown_total = _round(breakdown.total_score_after_recovery or breakdown.total_score)
        if abs(expected_total - actual_total) > 0.11 and abs(breakdown_total - actual_total) > 0.11:
            score_breakdown_mismatches.append(
                {
                    "horse_no": horse.horse_no,
                    "horse_name": horse.horse_name,
                    "expected_from_breakdown": expected_total,
                    "total_score": actual_total,
                    "breakdown_total": breakdown_total,
                }
            )

    race_horse_count = len(race_data.horses) if race_data is not None else None
    if race_horse_count is not None and race_horse_count != len(horse_scores):
        issues.append(_issue("warning", "HORSE_SCORE_COUNT_MISMATCH", f"race_data horses={race_horse_count}, horse_scores={len(horse_scores)}"))
    if marks_missing:
        issues.append(_issue("error", "MARKS_MISSING", f"marks 欠損: {marks_missing}"))
    if mark_horses_missing:
        issues.append(_issue("error", "MARK_HORSE_NOT_IN_SCORE", f"marks の馬番が horse_scores に存在しない: {mark_horses_missing}"))
    if expected_top5 and actual_marks_top5 != expected_top5:
        issues.append(_issue("warning", "MARKS_NOT_TOTAL_SCORE_ORDER", f"marks={actual_marks_top5}, total_score順top5={expected_top5}"))
    if score_breakdown_mismatches:
        issues.append(_issue("warning", "SCORE_BREAKDOWN_MISMATCH", "score_breakdown と total_score が一致しない馬がいる。"))

    analysis_counts = {
        "deep_analyses": len(prediction.deep_analyses),
        "pedigree_analyses": len(prediction.pedigree_analyses),
        "race_level_analyses": len(prediction.race_level_analyses),
        "pace_analyses": len(prediction.pace_analyses),
    }
    for name, count in analysis_counts.items():
        if horse_scores and count == 0:
            issues.append(_issue("warning", f"{name.upper()}_EMPTY", f"{name} が空。"))

    core_simulation_fields_empty: list[str] = []
    if prediction.race_simulation is None:
        issues.append(_issue("warning", "RACE_SIMULATION_MISSING", "race_simulation が欠損。"))
    else:
        for field_name in (
            "race_flow",
            "key_positions",
            "win_scenario",
            "top3_scenario",
            "betting_scenario",
            "reasoning_summary",
        ):
            if not getattr(prediction.race_simulation, field_name, ""):
                core_simulation_fields_empty.append(field_name)
        if core_simulation_fields_empty:
            issues.append(_issue("warning", "RACE_SIMULATION_EMPTY_FIELD", f"race_simulation 空字段: {core_simulation_fields_empty}"))

    scoring_config = prediction.scoring_config.model_dump()
    market_config = prediction.market_signal_config.model_dump()
    if market_config.get("use_market_score_in_ranking") or market_config.get("market_signal_weight", 0.0) > 0:
        issues.append(_issue("warning", "MARKET_SCORE_ENABLED", "odds/popularity が ranking score に入る設定。現在の期待値は default false。"))

    return (
        {
            "source": "rule scoring pipeline; LLM only for text/simulation/review explanations",
            "status": "ok",
            "scoring_profile": prediction.scoring_profile,
            "scoring_mode": prediction.scoring_mode,
            "scoring_config": scoring_config,
            "market_signal_config": market_config,
            "borderline_recovery_enabled": prediction.borderline_recovery_enabled,
            "borderline_recovery_applied": prediction.borderline_recovery_result.recovery_applied,
            "horse_scores_count": len(horse_scores),
            "marks": prediction.marks,
            "expected_top5_by_total_score": expected_top5,
            "expected_top5_by_current_mark_rule": expected_top5_by_mark_rule,
            "marks_match_total_score_order": actual_marks_top5 == expected_top5,
            "marks_match_current_rule_order": actual_marks_top5 == expected_top5_by_mark_rule,
            "analysis_counts": analysis_counts,
            "score_breakdown_mismatch_count": len(score_breakdown_mismatches),
            "score_breakdown_mismatch_examples": score_breakdown_mismatches[:10],
            "race_simulation_present": prediction.race_simulation is not None,
            "race_simulation_empty_fields": core_simulation_fields_empty,
            "field_sources": {
                "base_total_score": "recent_runs heuristic scoring",
                "pedigree_adjustment": "pedigree analyzer; weighted by scoring_config.pedigree_weight",
                "race_level_adjustment": "race_level analyzer; weighted by scoring_config.race_level_weight",
                "pace_adjustment": "pace analyzer; raw retained, default weight may be 0",
                "borderline_recovery": "rule-based Top5 boundary recovery, only if enabled",
                "marks": "sorted by final total_score",
                "summary_commentary_simulation": "LLM or fallback text; must not alter scores/marks",
                "odds_popularity": "reference fields; not ranking input when market_signal_weight=0",
            },
        },
        issues,
    )


def _audit_result(race_id: str, result_data: ResultData | None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if result_data is None:
        return {"source": "netkeiba result page", "status": "missing"}, [
            _issue("warning", "RESULT_MISSING", "result.json が見つからない。")
        ]

    if result_data.race_id != race_id:
        issues.append(_issue("error", "RESULT_RACE_ID_MISMATCH", f"result.race_id={result_data.race_id} が race_id と一致しない。"))
    top3 = [result_data.result.first, result_data.result.second, result_data.result.third]
    finish_top3 = [item.horse_no for item in sorted(result_data.finish_order, key=lambda item: item.finish)[:3]]
    if result_data.finish_order and finish_top3 != top3:
        issues.append(_issue("error", "RESULT_TOP3_FINISH_ORDER_MISMATCH", f"result top3={top3}, finish_order top3={finish_top3}"))
    if not result_data.finish_order:
        issues.append(_issue("warning", "FINISH_ORDER_MISSING", "finish_order が空。simulation_review の信頼性が下がる。"))
    if not result_data.payouts:
        issues.append(_issue("warning", "PAYOUTS_MISSING", "payouts が空。ROI は不可靠になる可能性。"))
    if result_data.warnings:
        issues.append(_issue("warning", "RESULT_WARNINGS_PRESENT", f"result warnings: {result_data.warnings}"))

    payout_types = Counter(payout.bet_type for payout in result_data.payouts)
    return (
        {
            "source": "netkeiba result page parser",
            "status": "ok",
            "top3": top3,
            "finish_order_count": len(result_data.finish_order),
            "finish_order_top3": finish_top3,
            "payouts_count": len(result_data.payouts),
            "payout_types": dict(sorted(payout_types.items())),
            "result_warnings": result_data.warnings,
            "field_sources": {
                "top3": "result.finish_order first 3 when parser has full order; legacy may only contain result",
                "finish_order": "netkeiba result table",
                "payouts": "netkeiba payout table",
            },
        },
        issues,
    )


def _audit_review(
    race_id: str,
    review: Review | None,
    prediction: Prediction | None,
    result_data: ResultData | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if review is None:
        return {"source": "review-race deterministic metrics + optional LLM review text", "status": "missing"}, [
            _issue("warning", "REVIEW_MISSING", "review.json が見つからない。")
        ]

    if review.race_id != race_id:
        issues.append(_issue("error", "REVIEW_RACE_ID_MISMATCH", f"review.race_id={review.race_id} が race_id と一致しない。"))
    if review.hit_summary.bet_hit and review.hit_summary.total_return == 0 and not review.payout_warning:
        issues.append(_issue("error", "BET_HIT_ZERO_RETURN_WITHOUT_WARNING", "bet_hit=true だが total_return=0 かつ payout_warning=false。"))
    if review.payout_warning:
        issues.append(_issue("warning", "PAYOUT_WARNING", "review.payout_warning=true。ROI は暫定扱い。"))
    if review.review_warnings:
        issues.append(_issue("warning", "REVIEW_WARNINGS_PRESENT", f"review warnings: {review.review_warnings}"))

    simulation_unknown_count = 0
    favorable_count = 0
    risk_count = 0
    if review.simulation_review is not None:
        favorable_count = len(review.simulation_review.favorable_horses_result)
        risk_count = len(review.simulation_review.risk_horses_result)
        simulation_unknown_count = sum(
            1 for item in review.simulation_review.favorable_horses_result if item.status == "unknown" or item.result == "unknown"
        ) + sum(
            1 for item in review.simulation_review.risk_horses_result if item.status == "unknown" or item.result == "unknown"
        )
        if result_data is not None and result_data.finish_order and simulation_unknown_count > 0:
            issues.append(_issue("warning", "SIMULATION_REVIEW_UNKNOWN_WITH_FINISH_ORDER", "finish_order があるのに simulation_review に unknown が残っている。"))
    elif prediction is not None and prediction.race_simulation is not None:
        issues.append(_issue("warning", "SIMULATION_REVIEW_MISSING", "prediction.race_simulation はあるが review.simulation_review が欠損。"))

    expected_marked_top3: int | None = None
    if prediction is not None and result_data is not None:
        result_top3 = {result_data.result.first, result_data.result.second, result_data.result.third}
        marked = {horse_no for horse_no in prediction.marks.values() if horse_no > 0}
        expected_marked_top3 = len(marked & result_top3)
        if expected_marked_top3 != review.hit_summary.marked_horses_top3_count:
            issues.append(_issue("warning", "HIT_SUMMARY_MARKED_COUNT_MISMATCH", f"review={review.hit_summary.marked_horses_top3_count}, expected={expected_marked_top3}"))

    return (
        {
            "source": "review-race; deterministic hit/ROI first, optional LLM for review comments",
            "status": "ok",
            "hit_summary": review.hit_summary.model_dump(),
            "bet_results_count": len(review.bet_results),
            "payout_warning": review.payout_warning,
            "review_warnings": review.review_warnings,
            "expected_marked_top3_count": expected_marked_top3,
            "simulation_review_present": review.simulation_review is not None,
            "simulation_review_favorable_count": favorable_count,
            "simulation_review_risk_count": risk_count,
            "simulation_review_unknown_count": simulation_unknown_count,
            "field_sources": {
                "hit_summary": "prediction.marks/bets matched against result top3/payouts",
                "roi_total_return": "result.payouts; unreliable if payout_warning=true",
                "simulation_review": "prediction.race_simulation checked against result.finish_order",
                "good_bad_lessons": "LLM or fallback review text",
            },
        },
        issues,
    )


def _status_from_issues(issues: list[dict[str, str]]) -> str:
    if any(issue["severity"] == "error" for issue in issues):
        return "error"
    if any(issue["severity"] == "warning" for issue in issues):
        return "warning"
    return "ok"


def audit_single_race_data_flow(
    *,
    race_id: str,
    race_data_dir: Path,
    predictions_dir: Path,
    results_dir: Path,
    reviews_dir: Path,
) -> dict[str, Any]:
    race_data_path = race_data_dir / f"{race_id}.json"
    prediction_path = predictions_dir / f"{race_id}.json"
    result_path = results_dir / f"{race_id}.json"
    review_path = reviews_dir / f"{race_id}.json"

    race_data, race_data_file = _load_model(RaceData, race_data_path)
    prediction, prediction_file = _load_model(Prediction, prediction_path)
    result_data, result_file = _load_model(ResultData, result_path)
    review, review_file = _load_model(Review, review_path)

    sections: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    race_data_audit, race_data_issues = _audit_race_data(race_id, race_data)
    prediction_audit, prediction_issues = _audit_prediction(race_id, prediction, race_data)
    result_audit, result_issues = _audit_result(race_id, result_data)
    review_audit, review_issues = _audit_review(race_id, review, prediction, result_data)
    sections["race_data"] = race_data_audit
    sections["prediction"] = prediction_audit
    sections["result"] = result_audit
    sections["review"] = review_audit
    issues.extend(race_data_issues)
    issues.extend(prediction_issues)
    issues.extend(result_issues)
    issues.extend(review_issues)

    file_issues = []
    file_infos = (
        ("race_data", race_data_file),
        ("prediction", prediction_file),
        ("result", result_file),
        ("review", review_file),
    )
    missing_codes_for_parse_failure = {
        "race_data": "RACE_DATA_MISSING",
        "prediction": "PREDICTION_MISSING",
        "result": "RESULT_MISSING",
        "review": "REVIEW_MISSING",
    }
    parse_failed_missing_codes: set[str] = set()
    for label, info in file_infos:
        if info["exists"] and not info.get("parse_ok", False):
            parse_failed_missing_codes.add(missing_codes_for_parse_failure[label])
            file_issues.append(_issue("error", f"{label.upper()}_PARSE_FAILED", str(info.get("error", ""))))
    if parse_failed_missing_codes:
        issues = [issue for issue in issues if issue["code"] not in parse_failed_missing_codes]
    issues.extend(file_issues)

    prediction_ready = race_data is not None and prediction is not None
    backtest_ready = prediction is not None and result_data is not None and review is not None
    review_ready = prediction is not None and result_data is not None
    return {
        "race_id": race_id,
        "status": _status_from_issues(issues),
        "files": {
            "race_data": race_data_file,
            "prediction": prediction_file,
            "result": result_file,
            "review": review_file,
        },
        "data_flow": sections,
        "readiness": {
            "prediction_ready": prediction_ready,
            "review_ready": review_ready,
            "backtest_ready": backtest_ready,
            "roi_reliable": review is not None and not review.payout_warning and result_data is not None and bool(result_data.payouts),
            "simulation_review_reliable": review is not None and result_data is not None and bool(result_data.finish_order),
        },
        "issues": issues,
        "issue_counts": dict(Counter(issue["severity"] for issue in issues)),
    }


def _prediction_date(path: Path) -> str | None:
    try:
        payload = _read_json(path)
    except json.JSONDecodeError:
        return None
    if not payload:
        return None
    race_info = payload.get("race_info") or {}
    return race_info.get("race_date")


def audit_data_flow_period(
    *,
    from_date: str,
    to_date: str,
    race_data_dir: Path,
    predictions_dir: Path,
    results_dir: Path,
    reviews_dir: Path,
) -> dict[str, Any]:
    race_ids: list[str] = []
    warnings: list[str] = []
    if not predictions_dir.exists():
        warnings.append(f"predictions_dir missing: {predictions_dir}")
    else:
        for path in sorted(predictions_dir.glob("*.json")):
            race_date = _prediction_date(path)
            if race_date is None:
                warnings.append(f"prediction skipped because race_info.race_date is missing: {path.name}")
                continue
            if from_date <= race_date <= to_date:
                race_ids.append(path.stem)

    race_audits = [
        audit_single_race_data_flow(
            race_id=race_id,
            race_data_dir=race_data_dir,
            predictions_dir=predictions_dir,
            results_dir=results_dir,
            reviews_dir=reviews_dir,
        )
        for race_id in race_ids
    ]
    issue_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()
    for audit in race_audits:
        for issue in audit["issues"]:
            issue_counter.update([issue["code"]])
            severity_counter.update([issue["severity"]])

    summary = {
        "race_count": len(race_audits),
        "ok_count": sum(1 for audit in race_audits if audit["status"] == "ok"),
        "warning_count": sum(1 for audit in race_audits if audit["status"] == "warning"),
        "error_count": sum(1 for audit in race_audits if audit["status"] == "error"),
        "prediction_ready_count": sum(1 for audit in race_audits if audit["readiness"]["prediction_ready"]),
        "review_ready_count": sum(1 for audit in race_audits if audit["readiness"]["review_ready"]),
        "backtest_ready_count": sum(1 for audit in race_audits if audit["readiness"]["backtest_ready"]),
        "roi_reliable_count": sum(1 for audit in race_audits if audit["readiness"]["roi_reliable"]),
        "simulation_review_reliable_count": sum(1 for audit in race_audits if audit["readiness"]["simulation_review_reliable"]),
    }
    return {
        "period": {"from": from_date, "to": to_date},
        "summary": summary,
        "issue_counts": dict(sorted(issue_counter.items(), key=lambda item: (-item[1], item[0]))),
        "severity_counts": dict(severity_counter),
        "race_audits": race_audits,
        "problem_races": [
            {
                "race_id": audit["race_id"],
                "status": audit["status"],
                "issues": audit["issues"],
            }
            for audit in race_audits
            if audit["issues"]
        ],
        "warnings": warnings,
    }


def generate_single_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Data Flow Audit",
        "",
        "## Summary",
        f"- race_id: {audit['race_id']}",
        f"- status: {audit['status']}",
        f"- prediction_ready: {str(audit['readiness']['prediction_ready']).lower()}",
        f"- review_ready: {str(audit['readiness']['review_ready']).lower()}",
        f"- backtest_ready: {str(audit['readiness']['backtest_ready']).lower()}",
        f"- roi_reliable: {str(audit['readiness']['roi_reliable']).lower()}",
        f"- simulation_review_reliable: {str(audit['readiness']['simulation_review_reliable']).lower()}",
        "",
        "## Files",
        "| file | exists | parse_ok | path |",
        "| --- | ---: | ---: | --- |",
    ]
    for label, info in audit["files"].items():
        lines.append(f"| {label} | {str(info['exists']).lower()} | {str(info.get('parse_ok', False)).lower()} | {info['path']} |")

    lines.extend(["", "## Data Sources", "| section | source | key status |", "| --- | --- | --- |"])
    for section, payload in audit["data_flow"].items():
        lines.append(f"| {section} | {payload.get('source', '-')} | {payload.get('status', '-')} |")

    prediction = audit["data_flow"]["prediction"]
    result = audit["data_flow"]["result"]
    review = audit["data_flow"]["review"]
    race_data = audit["data_flow"]["race_data"]
    lines.extend(
        [
            "",
            "## Key Checks",
            f"- horse_count: {race_data.get('horse_count', '-')}",
            f"- avg_recent_runs: {race_data.get('avg_recent_runs', '-')}",
            f"- future_leakage_count: {race_data.get('future_leakage_count', '-')}",
            f"- scoring_profile: {prediction.get('scoring_profile', '-')}",
            f"- scoring_mode: {prediction.get('scoring_mode', '-')}",
            f"- market_signal_config: {prediction.get('market_signal_config', '-')}",
            f"- marks_match_current_rule_order: {str(prediction.get('marks_match_current_rule_order', False)).lower()}",
            f"- marks_match_total_score_order: {str(prediction.get('marks_match_total_score_order', False)).lower()}",
            f"- finish_order_count: {result.get('finish_order_count', '-')}",
            f"- payouts_count: {result.get('payouts_count', '-')}",
            f"- payout_warning: {str(review.get('payout_warning', False)).lower()}",
            f"- simulation_review_unknown_count: {review.get('simulation_review_unknown_count', '-')}",
            "",
            "## Issues",
            "| severity | code | message |",
            "| --- | --- | --- |",
        ]
    )
    if audit["issues"]:
        for issue in audit["issues"]:
            lines.append(f"| {issue['severity']} | {issue['code']} | {issue['message']} |")
    else:
        lines.append("| ok | NONE | 問題なし |")
    return "\n".join(lines) + "\n"


def generate_period_audit_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Data Flow Audit Summary",
        "",
        "## Summary",
        f"- 対象期間: {report['period']['from']} - {report['period']['to']}",
        f"- race_count: {summary['race_count']}",
        f"- ok_count: {summary['ok_count']}",
        f"- warning_count: {summary['warning_count']}",
        f"- error_count: {summary['error_count']}",
        f"- prediction_ready_count: {summary['prediction_ready_count']}",
        f"- review_ready_count: {summary['review_ready_count']}",
        f"- backtest_ready_count: {summary['backtest_ready_count']}",
        f"- roi_reliable_count: {summary['roi_reliable_count']}",
        f"- simulation_review_reliable_count: {summary['simulation_review_reliable_count']}",
        "",
        "## Issue Ranking",
        "| issue | count |",
        "| --- | ---: |",
    ]
    if report["issue_counts"]:
        for code, count in report["issue_counts"].items():
            lines.append(f"| {code} | {count} |")
    else:
        lines.append("| NONE | 0 |")

    lines.extend(
        [
            "",
            "## Problem Races",
            "| race_id | status | issues |",
            "| --- | --- | --- |",
        ]
    )
    if report["problem_races"]:
        for row in report["problem_races"]:
            issue_text = ", ".join(issue["code"] for issue in row["issues"])
            lines.append(f"| {row['race_id']} | {row['status']} | {issue_text} |")
    else:
        lines.append("| なし | ok | - |")

    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def save_audit_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_audit_markdown(markdown: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keiba_llm_agent.agents.race_analysis_agent import RaceAnalysisAgent
from keiba_llm_agent.agents.race_review_agent import RaceReviewAgent
from keiba_llm_agent.analysis_reports.score_factor_analyzer import (
    generate_score_factor_analysis_markdown,
    run_score_factor_analysis,
    save_score_factor_analysis_json,
    save_score_factor_analysis_markdown,
)
from keiba_llm_agent.audit.data_flow_auditor import (
    audit_data_flow_period,
    audit_single_race_data_flow,
    generate_period_audit_markdown,
    generate_single_audit_markdown,
    save_audit_json,
    save_audit_markdown,
)
from keiba_llm_agent.backtest.backtest_runner import (
    generate_backtest_markdown,
    generate_weight_tuning_markdown,
    run_backtest,
    run_backtest_weights,
    save_backtest_json,
    save_backtest_markdown,
)
from keiba_llm_agent.fetchers.netkeiba_fetcher import (
    fetch_and_parse_netkeiba_race,
    get_html_cache_path,
    save_race_data,
)
from keiba_llm_agent.fetchers.netkeiba_horse_fetcher import (
    enrich_race_data_with_recent_runs,
    fetch_horse_html,
    get_horse_cache_path,
)
from keiba_llm_agent.fetchers.netkeiba_result_fetcher import (
    fetch_and_parse_netkeiba_result,
    save_result_data,
)
from keiba_llm_agent.config import (
    DEFAULT_SCORING_PROFILE,
    LOCAL_SCORING_PROFILE,
    SCORING_MODES,
    SCORING_PROFILES,
    get_llm_config,
    resolve_scoring_profile_config,
)
from keiba_llm_agent.llm import MockLLMClient, create_llm_client
from keiba_llm_agent.memory.lesson_manager import LessonManager
from keiba_llm_agent.memory.lesson_store import LessonStore
from keiba_llm_agent.parsers.netkeiba_horse_parser import parse_horse_recent_runs
from keiba_llm_agent.parsers.netkeiba_race_parser import parse_netkeiba_shutuba_html
from keiba_llm_agent.parsers.netkeiba_url_parser import extract_race_id
from keiba_llm_agent.pedigree.pedigree_analyzer import (
    fetch_pedigree_html,
    get_pedigree_cache_path,
)
from keiba_llm_agent.pedigree.pedigree_parser import parse_pedigree_info
from keiba_llm_agent.daily.daily_summary_generator import (
    generate_daily_report,
    generate_daily_social_post,
    save_daily_report,
)
from keiba_llm_agent.error_analysis.missed_top3_analyzer import (
    generate_missed_top3_markdown,
    run_missed_top3_analysis,
    save_missed_top3_json,
    save_missed_top3_markdown,
)
from keiba_llm_agent.error_analysis.deep_miss_analyzer import (
    generate_deep_miss_markdown,
    run_deep_miss_analysis,
    save_deep_miss_json,
    save_deep_miss_markdown,
)
from keiba_llm_agent.error_analysis.deep_miss_rule_simulator import (
    BASELINE_MODE_CHOICES,
    generate_deep_miss_rule_simulation_markdown,
    run_deep_miss_rule_simulation,
    save_deep_miss_rule_simulation_json,
    save_deep_miss_rule_simulation_markdown,
)
from keiba_llm_agent.error_analysis.penalty_refinement_simulator import (
    generate_penalty_refinement_simulation_markdown,
    run_penalty_refinement_simulation,
    save_penalty_refinement_simulation_json,
    save_penalty_refinement_simulation_markdown,
)
from keiba_llm_agent.error_analysis.score_recalibration_simulator import (
    generate_score_recalibration_simulation_markdown,
    run_score_recalibration_simulation,
    save_score_recalibration_simulation_json,
    save_score_recalibration_simulation_markdown,
)
from keiba_llm_agent.error_analysis.condition_weight_simulator import (
    generate_condition_weight_simulation_markdown,
    run_condition_weight_simulation,
    save_condition_weight_simulation_json,
    save_condition_weight_simulation_markdown,
)
from keiba_llm_agent.reports.report_generator import (
    generate_prediction_report,
    generate_review_report,
    save_report,
)
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review
from keiba_llm_agent.social.post_generator import (
    build_prediction_post,
    build_review_post,
    save_post,
)
from keiba_llm_agent.validators.race_data_validator import validate_race_data_file


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LESSONS_PATH = BASE_DIR / "memory" / "lessons.json"
DEFAULT_PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
DEFAULT_REVIEWS_DIR = BASE_DIR / "data" / "reviews"
DEFAULT_RESULTS_DIR = BASE_DIR / "data" / "results"
DEFAULT_RACE_DATA_DIR = BASE_DIR / "data" / "race_data"
DEFAULT_REPORTS_DIR = BASE_DIR / "data" / "reports"
DEFAULT_SOCIAL_POSTS_DIR = BASE_DIR / "data" / "social_posts"
DEFAULT_DAILY_REPORTS_DIR = BASE_DIR / "data" / "daily_reports"
DEFAULT_AUTO_SKIP_REPORT = True
DEFAULT_BACKTESTS_DIR = BASE_DIR / "data" / "backtests"
DEFAULT_ERROR_ANALYSIS_DIR = BASE_DIR / "data" / "error_analysis"
DEFAULT_AUDITS_DIR = BASE_DIR / "data" / "audits"
DEFAULT_ANALYSIS_REPORTS_DIR = BASE_DIR / "data" / "analysis_reports"


SCORING_MODE_CHOICES = [*SCORING_MODES.keys(), "custom"]
SCORING_PROFILE_CHOICES = list(SCORING_PROFILES.keys())
SCOPE_KEY_CHOICES = ["central", "central_turf", "central_dirt", "local"]
LOCAL_VENUE_CODES = {"30", "35", "36", "42", "43", "44", "45", "46", "47", "48", "50", "51", "54", "55"}


def _configure_stdio_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scoring-profile",
        choices=SCORING_PROFILE_CHOICES,
        help=f"评分 profile，默认 {DEFAULT_SCORING_PROFILE}",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=SCORING_MODE_CHOICES,
        help="评分权重模式；如显式传入，将覆盖 scoring-profile 内置 mode",
    )
    parser.add_argument("--pedigree-weight", type=float, help="覆盖 pedigree 权重")
    parser.add_argument("--race-level-weight", type=float, help="覆盖 race_level 权重")
    parser.add_argument("--pace-weight", type=float, help="覆盖 pace 权重")
    parser.add_argument(
        "--enable-borderline-recovery",
        action="store_true",
        help="启用 Top5 境界補正",
    )
    parser.add_argument(
        "--disable-borderline-recovery",
        action="store_true",
        help="禁用 Top5 境界補正",
    )


def _infer_scope_key_from_race_id(race_id: str | None) -> str:
    race_id_text = str(race_id or "").strip()
    if len(race_id_text) < 6:
        return ""
    if race_id_text[4:6] in LOCAL_VENUE_CODES:
        return "local"
    return ""


def _scope_output_suffix(scope_key: str | None) -> str:
    normalized_scope = str(scope_key or "").strip().lower()
    return f"_{normalized_scope}" if normalized_scope else ""


def _resolve_scoring_profile_for_race(
    race_data: RaceData,
    scoring_profile: str | None,
) -> str | None:
    if scoring_profile:
        return scoring_profile
    race_info = race_data.race_info
    scope_key = str(race_info.scope_key or "").strip().lower()
    if not scope_key:
        scope_key = _infer_scope_key_from_race_id(race_info.race_id)
    if scope_key == "local":
        return LOCAL_SCORING_PROFILE
    return None


def _resolve_borderline_recovery_enabled(
    scoring_mode: str | None,
    enable_borderline_recovery: bool = False,
    disable_borderline_recovery: bool = False,
) -> bool:
    if enable_borderline_recovery and disable_borderline_recovery:
        raise ValueError("不能同时指定 enable-borderline-recovery 和 disable-borderline-recovery")
    if enable_borderline_recovery:
        return True
    if disable_borderline_recovery:
        return False
    return (scoring_mode or "candidate_default") in {"candidate_default", "custom"}


def _resolve_borderline_recovery_override(
    enable_borderline_recovery: bool = False,
    disable_borderline_recovery: bool = False,
) -> bool | None:
    if enable_borderline_recovery and disable_borderline_recovery:
        raise ValueError("不能同时指定 enable-borderline-recovery 和 disable-borderline-recovery")
    if enable_borderline_recovery:
        return True
    if disable_borderline_recovery:
        return False
    return None


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _load_race_info_for_review(
    race_id: str,
    prediction: Prediction,
    race_data_path: str | Path | None = None,
) -> dict | None:
    final_race_data_path = (
        Path(race_data_path)
        if race_data_path is not None
        else DEFAULT_RACE_DATA_DIR / f"{race_id}.json"
    )
    if final_race_data_path.exists():
        return RaceData.from_json_file(final_race_data_path).race_info.model_dump()
    if prediction.race_info is not None:
        return prediction.race_info.model_dump()
    warnings.warn("race_data not found; lesson condition fallback to unknown", stacklevel=2)
    return None


def run_analysis(
    race_data_path: str | Path,
    output_path: str | Path | None = None,
    lessons_path: str | Path = DEFAULT_LESSONS_PATH,
    llm_provider: str | None = None,
    scoring_profile: str | None = None,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    borderline_recovery_enabled: bool | None = None,
) -> tuple[Prediction, Path]:
    race_data = RaceData.from_json_file(race_data_path)
    lesson_store = LessonStore(lessons_path)
    lessons = lesson_store.load_lessons()
    scoring_profile_config, scoring_warnings = resolve_scoring_profile_config(
        scoring_profile=scoring_profile,
        scoring_mode=scoring_mode,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
        borderline_recovery_enabled=borderline_recovery_enabled,
    )
    for warning_text in scoring_warnings:
        warnings.warn(warning_text, stacklevel=2)
    agent = RaceAnalysisAgent(llm_client=create_llm_client(provider_override=llm_provider))
    prediction = agent.run(
        race_data,
        lessons,
        scoring_profile=scoring_profile_config.scoring_profile,
        scoring_config=scoring_profile_config.scoring_config,
        borderline_recovery_enabled=scoring_profile_config.borderline_recovery_enabled,
    )
    prediction = lesson_store.sync_prediction_used_lessons(prediction)
    final_output_path = Path(output_path) if output_path else DEFAULT_PREDICTIONS_DIR / f"{prediction.race_id}.json"
    saved_path = _write_json(final_output_path, prediction.model_dump())
    return prediction, saved_path


def run_review(
    race_id: str,
    result_path: str | Path,
    prediction_path: str | Path | None = None,
    output_path: str | Path | None = None,
    lessons_path: str | Path = DEFAULT_LESSONS_PATH,
    race_data_path: str | Path | None = None,
    llm_provider: str | None = None,
) -> tuple[dict, Path]:
    final_prediction_path = (
        Path(prediction_path)
        if prediction_path
        else DEFAULT_PREDICTIONS_DIR / f"{race_id}.json"
    )
    prediction = Prediction.model_validate(_read_json(final_prediction_path))
    result = _read_json(result_path)
    result_race_id = result.get("race_id")
    if result_race_id and result_race_id != race_id:
        raise ValueError(f"result race_id 不匹配: 预期 {race_id}, 实际 {result_race_id}")

    lesson_store = LessonStore(lessons_path)
    agent = RaceReviewAgent(llm_client=create_llm_client(provider_override=llm_provider))
    race_info = _load_race_info_for_review(race_id, prediction, race_data_path=race_data_path)
    review = agent.run(race_id, result, prediction, race_info=race_info)
    lesson_store.update_effectiveness(
        prediction.used_lessons,
        review,
        strategy=prediction.strategy,
    )
    final_output_path = Path(output_path) if output_path else DEFAULT_REVIEWS_DIR / f"{race_id}.json"
    saved_path = _write_json(final_output_path, review.model_dump())
    total_lessons = lesson_store.upsert_lessons(review.lessons)
    return {
        "review": review,
        "saved_path": saved_path,
        "total_lessons": total_lessons,
    }, saved_path


def run_review_from_result_data(
    result_data: ResultData,
    prediction_path: str | Path | None = None,
    output_path: str | Path | None = None,
    lessons_path: str | Path = DEFAULT_LESSONS_PATH,
    race_data_path: str | Path | None = None,
    llm_provider: str | None = None,
) -> tuple[dict, Path]:
    race_id = result_data.race_id
    final_prediction_path = (
        Path(prediction_path)
        if prediction_path
        else DEFAULT_PREDICTIONS_DIR / f"{race_id}.json"
    )
    if not final_prediction_path.exists():
        raise ValueError(f"prediction not found for race_id={race_id}. Run analyze-url first.")

    prediction = Prediction.model_validate(_read_json(final_prediction_path))
    lesson_store = LessonStore(lessons_path)
    agent = RaceReviewAgent(llm_client=create_llm_client(provider_override=llm_provider))
    race_info = _load_race_info_for_review(race_id, prediction, race_data_path=race_data_path)
    review = agent.run(
        race_id,
        result_data.model_dump(by_alias=True),
        prediction,
        race_info=race_info,
    )
    lesson_store.update_effectiveness(
        prediction.used_lessons,
        review,
        strategy=prediction.strategy,
    )
    final_output_path = Path(output_path) if output_path else DEFAULT_REVIEWS_DIR / f"{race_id}.json"
    saved_path = _write_json(final_output_path, review.model_dump())
    total_lessons = lesson_store.upsert_lessons(review.lessons)
    return {
        "review": review,
        "saved_path": saved_path,
        "total_lessons": total_lessons,
    }, saved_path


def run_parse_url(url: str) -> str:
    return extract_race_id(url)


def run_parse_html(html_path: str | Path, race_id: str | None = None) -> RaceData:
    html = Path(html_path).read_text(encoding="utf-8")
    return parse_netkeiba_shutuba_html(html, race_id=race_id)


def run_fetch_race(
    url: str,
    force_refresh: bool = False,
    with_recent_runs: bool = False,
    recent_run_limit: int | None = None,
) -> dict[str, str]:
    race_data = fetch_and_parse_netkeiba_race(url, force_refresh=force_refresh)
    if with_recent_runs:
        race_data = enrich_race_data_with_recent_runs(
            race_data,
            limit=recent_run_limit,
            force_refresh=force_refresh,
        )
    saved_path = save_race_data(race_data)
    cache_path = get_html_cache_path(race_data.race_info.race_id)
    return {
        "race_id": race_data.race_info.race_id,
        "saved_to": str(saved_path),
        "cache_path": str(cache_path),
    }


def run_analyze_url(
    url: str,
    force_refresh: bool = False,
    dry_run: bool = False,
    lessons_path: str | Path | None = None,
    with_recent_runs: bool = False,
    recent_run_limit: int | None = None,
    llm_provider: str | None = None,
    scoring_profile: str | None = None,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    borderline_recovery_enabled: bool | None = None,
) -> dict[str, str | None]:
    race_data = fetch_and_parse_netkeiba_race(url, force_refresh=force_refresh)
    resolved_scoring_profile = _resolve_scoring_profile_for_race(race_data, scoring_profile)
    if with_recent_runs:
        race_data = enrich_race_data_with_recent_runs(
            race_data,
            limit=recent_run_limit,
            force_refresh=force_refresh,
        )
    race_data_path = save_race_data(race_data)
    prediction_path: Path | None = None

    if not dry_run:
        final_lessons_path = Path(lessons_path) if lessons_path is not None else DEFAULT_LESSONS_PATH
        _, prediction_path = run_analysis(
            race_data_path=race_data_path,
            output_path=DEFAULT_PREDICTIONS_DIR / f"{race_data.race_info.race_id}.json",
            lessons_path=final_lessons_path,
            llm_provider=llm_provider,
            scoring_profile=resolved_scoring_profile,
            scoring_mode=scoring_mode,
            pedigree_weight=pedigree_weight,
            race_level_weight=race_level_weight,
            pace_weight=pace_weight,
            borderline_recovery_enabled=borderline_recovery_enabled,
        )

    return {
        "race_id": race_data.race_info.race_id,
        "source": race_data.race_info.source,
        "scope_key": race_data.race_info.scope_key,
        "scoring_profile": resolved_scoring_profile or DEFAULT_SCORING_PROFILE,
        "race_data_path": str(race_data_path),
        "prediction_path": str(prediction_path) if prediction_path else None,
    }


def run_fetch_horse(horse_id: str, limit: int | None = None, force_refresh: bool = False) -> dict[str, object]:
    html = fetch_horse_html(horse_id, force_refresh=force_refresh)
    recent_runs = parse_horse_recent_runs(html, limit=limit)
    result = {
        "horse_id": horse_id,
        "cache_path": str(get_horse_cache_path(horse_id)),
        "recent_runs_count": len(recent_runs),
        "recent_runs": [recent_run.model_dump() for recent_run in recent_runs],
        "note": "fetch-horse does not apply target race filtering",
    }
    if not recent_runs:
        result["warning"] = "no recent runs parsed"
    return result


def run_parse_pedigree(horse_id: str, horse_name: str | None = None, force_refresh: bool = False) -> dict[str, object]:
    horse_html = None
    horse_cache_path = get_horse_cache_path(horse_id)
    if horse_cache_path.exists():
        horse_html = horse_cache_path.read_text(encoding="utf-8")
    elif horse_id.isdigit():
        horse_html = fetch_horse_html(horse_id, force_refresh=force_refresh)

    pedigree = parse_pedigree_info(horse_html, horse_id, horse_name) if horse_html else None
    if pedigree is None or pedigree.sire is None:
        pedigree_html = fetch_pedigree_html(
            horse_id,
            force_refresh=force_refresh,
            horse_html=horse_html,
        )
        pedigree = parse_pedigree_info(pedigree_html, horse_id, horse_name)

    return {
        "horse_id": horse_id,
        "horse_name": pedigree.horse_name,
        "sire": pedigree.sire,
        "dam": pedigree.dam,
        "damsire": pedigree.damsire,
        "cache_path": str(get_pedigree_cache_path(horse_id)),
    }


def run_fetch_result(url: str, force_refresh: bool = False) -> dict[str, object]:
    result_data = fetch_and_parse_netkeiba_result(url, force_refresh=force_refresh)
    saved_path = save_result_data(result_data)
    return {
        "race_id": result_data.race_id,
        "result_path": str(saved_path),
        "top3": result_data.result.model_dump(),
    }


def run_review_url(
    url: str,
    force_refresh: bool = False,
    lessons_path: str | Path = DEFAULT_LESSONS_PATH,
    llm_provider: str | None = None,
) -> dict[str, object]:
    result_data = fetch_and_parse_netkeiba_result(url, force_refresh=force_refresh)
    result_path = save_result_data(result_data)
    review_result, review_path = run_review_from_result_data(
        result_data=result_data,
        output_path=DEFAULT_REVIEWS_DIR / f"{result_data.race_id}.json",
        lessons_path=lessons_path,
        llm_provider=llm_provider,
    )
    return {
        "race_id": result_data.race_id,
        "result_path": str(result_path),
        "review_path": str(review_path),
        "lessons_count": review_result["total_lessons"],
        "bet_hit": review_result["review"].hit_summary.bet_hit,
        "total_stake": review_result["review"].hit_summary.total_stake,
        "total_return": review_result["review"].hit_summary.total_return,
        "roi": review_result["review"].hit_summary.roi,
        "payout_warning": review_result["review"].payout_warning,
        "review_warnings": review_result["review"].review_warnings,
    }


def run_report_prediction(
    race_id: str,
    prediction_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, str]:
    final_prediction_path = (
        Path(prediction_path)
        if prediction_path is not None
        else DEFAULT_PREDICTIONS_DIR / f"{race_id}.json"
    )
    prediction = Prediction.model_validate(_read_json(final_prediction_path))
    race_data_path = DEFAULT_RACE_DATA_DIR / f"{race_id}.json"
    race_data = RaceData.from_json_file(race_data_path) if race_data_path.exists() else None
    markdown = generate_prediction_report(prediction, race_data=race_data)
    final_output_path = (
        Path(output_path)
        if output_path is not None
        else DEFAULT_REPORTS_DIR / f"{race_id}_prediction.md"
    )
    saved_path = save_report(markdown, final_output_path)
    return {"race_id": race_id, "report_path": str(saved_path)}


def run_report_review(
    race_id: str,
    prediction_path: str | Path | None = None,
    review_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, str]:
    final_prediction_path = (
        Path(prediction_path)
        if prediction_path is not None
        else DEFAULT_PREDICTIONS_DIR / f"{race_id}.json"
    )
    final_review_path = (
        Path(review_path)
        if review_path is not None
        else DEFAULT_REVIEWS_DIR / f"{race_id}.json"
    )
    prediction = Prediction.model_validate(_read_json(final_prediction_path))
    review = Review.model_validate(_read_json(final_review_path))
    race_data_path = DEFAULT_RACE_DATA_DIR / f"{race_id}.json"
    result_data_path = DEFAULT_RESULTS_DIR / f"{race_id}.json"
    race_data = RaceData.from_json_file(race_data_path) if race_data_path.exists() else None
    result_data = ResultData.model_validate(_read_json(result_data_path)) if result_data_path.exists() else None
    markdown = generate_review_report(
        prediction,
        review,
        result_data=result_data,
        race_data=race_data,
    )
    final_output_path = (
        Path(output_path)
        if output_path is not None
        else DEFAULT_REPORTS_DIR / f"{race_id}_review.md"
    )
    saved_path = save_report(markdown, final_output_path)
    return {"race_id": race_id, "report_path": str(saved_path)}


def run_social_prediction(
    race_id: str,
    prediction_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    final_prediction_path = (
        Path(prediction_path)
        if prediction_path is not None
        else DEFAULT_PREDICTIONS_DIR / f"{race_id}.json"
    )
    prediction = Prediction.model_validate(_read_json(final_prediction_path))
    race_data_path = DEFAULT_RACE_DATA_DIR / f"{race_id}.json"
    race_data = RaceData.from_json_file(race_data_path) if race_data_path.exists() else None
    text = build_prediction_post(prediction, race_data=race_data)
    final_output_path = (
        Path(output_path)
        if output_path is not None
        else DEFAULT_SOCIAL_POSTS_DIR / f"{race_id}_prediction.txt"
    )
    saved_path = save_post(text, final_output_path)
    return {"race_id": race_id, "post_path": str(saved_path), "char_count": len(text)}


def run_social_review(
    race_id: str,
    prediction_path: str | Path | None = None,
    review_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    final_prediction_path = (
        Path(prediction_path)
        if prediction_path is not None
        else DEFAULT_PREDICTIONS_DIR / f"{race_id}.json"
    )
    final_review_path = (
        Path(review_path)
        if review_path is not None
        else DEFAULT_REVIEWS_DIR / f"{race_id}.json"
    )
    prediction = Prediction.model_validate(_read_json(final_prediction_path))
    review = Review.model_validate(_read_json(final_review_path))
    race_data_path = DEFAULT_RACE_DATA_DIR / f"{race_id}.json"
    result_data_path = DEFAULT_RESULTS_DIR / f"{race_id}.json"
    race_data = RaceData.from_json_file(race_data_path) if race_data_path.exists() else None
    result_data = ResultData.model_validate(_read_json(result_data_path)) if result_data_path.exists() else None
    text = build_review_post(prediction, review, result_data=result_data, race_data=race_data)
    final_output_path = (
        Path(output_path)
        if output_path is not None
        else DEFAULT_SOCIAL_POSTS_DIR / f"{race_id}_review.txt"
    )
    saved_path = save_post(text, final_output_path)
    return {"race_id": race_id, "post_path": str(saved_path), "char_count": len(text)}


def _resolve_skip_report(skip_report: bool = False, enable_report: bool = False) -> bool:
    if enable_report:
        return False
    if skip_report:
        return True
    return DEFAULT_AUTO_SKIP_REPORT


def run_predict_race(
    url: str,
    force_refresh: bool = False,
    recent_run_limit: int | None = None,
    skip_report: bool = True,
    skip_social: bool = False,
    lessons_path: str | Path = DEFAULT_LESSONS_PATH,
    llm_provider: str | None = None,
    scoring_profile: str | None = None,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    borderline_recovery_enabled: bool | None = None,
) -> dict[str, object]:
    race_data = fetch_and_parse_netkeiba_race(url, force_refresh=force_refresh)
    resolved_scoring_profile = _resolve_scoring_profile_for_race(race_data, scoring_profile)
    warning_messages: list[str] = []
    if not race_data.horses:
        warning_messages.append("no horses parsed from shutuba HTML")
    if race_data.horses:
        race_data = enrich_race_data_with_recent_runs(
            race_data,
            limit=recent_run_limit,
            force_refresh=force_refresh,
        )
        if all(not horse.recent_runs for horse in race_data.horses):
            warning_messages.append("recent_runs are empty for all horses")
    for warning_message in warning_messages:
        warnings.warn(warning_message, stacklevel=2)

    race_data_path = save_race_data(race_data)
    _, prediction_path = run_analysis(
        race_data_path=race_data_path,
        output_path=DEFAULT_PREDICTIONS_DIR / f"{race_data.race_info.race_id}.json",
        lessons_path=lessons_path,
        llm_provider=llm_provider,
        scoring_profile=resolved_scoring_profile,
        scoring_mode=scoring_mode,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
        borderline_recovery_enabled=borderline_recovery_enabled,
    )

    report_path: str | None = None
    social_post_path: str | None = None
    if not skip_report:
        report_result = run_report_prediction(race_data.race_info.race_id, prediction_path=prediction_path)
        report_path = report_result["report_path"]
    if not skip_social:
        social_result = run_social_prediction(race_data.race_info.race_id, prediction_path=prediction_path)
        social_post_path = social_result["post_path"]

    result: dict[str, object] = {
        "race_id": race_data.race_info.race_id,
        "source": race_data.race_info.source,
        "scope_key": race_data.race_info.scope_key,
        "scoring_profile": resolved_scoring_profile or DEFAULT_SCORING_PROFILE,
        "race_data_path": str(race_data_path),
        "prediction_path": str(prediction_path),
        "prediction_report_path": report_path,
        "prediction_social_post_path": social_post_path,
    }
    if warning_messages:
        result["warnings"] = warning_messages
    return result


def run_review_race(
    url: str,
    force_refresh: bool = False,
    skip_report: bool = True,
    skip_social: bool = False,
    lessons_path: str | Path = DEFAULT_LESSONS_PATH,
    llm_provider: str | None = None,
) -> dict[str, object]:
    result_data = fetch_and_parse_netkeiba_result(url, force_refresh=force_refresh)
    if not (
        result_data.result.first
        and result_data.result.second
        and result_data.result.third
    ):
        raise ValueError(f"result top3 not found for race_id={result_data.race_id}")

    prediction_path = DEFAULT_PREDICTIONS_DIR / f"{result_data.race_id}.json"
    if not prediction_path.exists():
        raise ValueError(f"prediction not found for race_id={result_data.race_id}. Run predict-race first.")

    result_path = save_result_data(result_data)
    review_result, review_path = run_review_from_result_data(
        result_data=result_data,
        prediction_path=prediction_path,
        output_path=DEFAULT_REVIEWS_DIR / f"{result_data.race_id}.json",
        lessons_path=lessons_path,
        llm_provider=llm_provider,
    )

    report_path: str | None = None
    social_post_path: str | None = None
    if not skip_report:
        report_result = run_report_review(
            result_data.race_id,
            prediction_path=prediction_path,
            review_path=review_path,
        )
        report_path = report_result["report_path"]
    if not skip_social:
        social_result = run_social_review(
            result_data.race_id,
            prediction_path=prediction_path,
            review_path=review_path,
        )
        social_post_path = social_result["post_path"]

    return {
        "race_id": result_data.race_id,
        "result_path": str(result_path),
        "review_path": str(review_path),
        "lessons_count": review_result["total_lessons"],
        "review_report_path": report_path,
        "review_social_post_path": social_post_path,
    }


def run_validate_race_data(
    race_id: str,
    race_data_path: str | Path | None = None,
) -> dict[str, object]:
    final_race_data_path = (
        Path(race_data_path)
        if race_data_path is not None
        else DEFAULT_RACE_DATA_DIR / f"{race_id}.json"
    )
    return validate_race_data_file(final_race_data_path)


def run_llm_check(provider_override: str | None = None) -> dict[str, object]:
    config = get_llm_config(provider_override)
    requested_provider = config.provider
    active_provider = requested_provider
    status = "OK"
    detail = "ready"
    if requested_provider == "openai" and not config.openai_api_key:
        if config.enable_fallback:
            status = "WARNING"
            active_provider = "mock"
            detail = "OPENAI_API_KEY missing; fallback to mock"
        else:
            return {
                "provider": requested_provider,
                "active_provider": requested_provider,
                "model": config.openai_model,
                "status": "ERROR",
                "detail": "OPENAI_API_KEY is missing",
            }
    try:
        client = create_llm_client(provider_override=provider_override)
        if requested_provider == "openai" and isinstance(client, MockLLMClient):
            status = "WARNING"
            active_provider = "mock"
            detail = "openai package is not installed or OpenAI client init failed; fallback to mock"
        response = client.generate_json(
            "Return JSON only.",
            '{"ping":"pong"} を {"ok": true} 形式で返してください',
            schema_name="llm_check",
        )
        if requested_provider == "openai" and getattr(client, "last_fallback_used", False):
            status = "WARNING"
            active_provider = "mock"
            detail = "OpenAI request failed during llm-check; fallback to mock"
        if response.get("ok") is not True:
            status = "WARNING"
            detail = "JSON response parsed but ok!=true"
    except Exception as exc:
        status = "ERROR"
        detail = str(exc)
    return {
        "provider": requested_provider,
        "active_provider": active_provider,
        "model": config.openai_model,
        "status": status,
        "detail": detail,
    }


def run_lessons_list(lessons_path: str | Path = DEFAULT_LESSONS_PATH) -> dict[str, object]:
    manager = LessonManager(lessons_path)
    lessons = manager.list_lessons()
    return {
        "count": len(lessons),
        "lessons": [lesson.model_dump() for lesson in lessons],
    }


def run_lessons_disable(lesson_id: str, lessons_path: str | Path = DEFAULT_LESSONS_PATH) -> dict[str, object]:
    manager = LessonManager(lessons_path)
    lesson = manager.disable_lesson(lesson_id)
    return {"lesson_id": lesson.lesson_id, "enabled": lesson.enabled}


def run_lessons_enable(lesson_id: str, lessons_path: str | Path = DEFAULT_LESSONS_PATH) -> dict[str, object]:
    manager = LessonManager(lessons_path)
    lesson = manager.enable_lesson(lesson_id)
    return {"lesson_id": lesson.lesson_id, "enabled": lesson.enabled}


def run_lessons_prune(
    min_score: float,
    lessons_path: str | Path = DEFAULT_LESSONS_PATH,
) -> dict[str, object]:
    manager = LessonManager(lessons_path)
    disabled_count = manager.prune_lessons(min_score)
    return {"disabled_count": disabled_count, "min_score": min_score}


def run_daily_summary(
    target_date: str,
    skip_report: bool = True,
    skip_social: bool = False,
    lessons_path: str | Path = DEFAULT_LESSONS_PATH,
    scope_key: str | None = None,
) -> dict[str, object]:
    markdown, context = generate_daily_report(
        target_date=target_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        lessons_path=Path(lessons_path),
        scope_key=scope_key,
    )
    output_key = f"{target_date}{_scope_output_suffix(scope_key)}"
    report_path: Path | None = None
    social_path: Path | None = None
    if not skip_report:
        report_path = save_daily_report(markdown, DEFAULT_DAILY_REPORTS_DIR / f"{output_key}.md")
    if not skip_social:
        social_text = generate_daily_social_post(context)
        social_path = save_post(social_text, DEFAULT_SOCIAL_POSTS_DIR / f"{output_key}_daily.txt")
    metrics = context["metrics"]
    return {
        "date": target_date,
        "scope_key": context.get("scope_key"),
        "scope_label": context.get("scope_label"),
        "daily_report_path": str(report_path) if report_path is not None else None,
        "daily_social_post_path": str(social_path) if social_path is not None else None,
        "target_race_count": metrics["target_race_count"],
        "reviewed_race_count": metrics["reviewed_race_count"],
        "bet_race_count": metrics["bet_race_count"],
        "hit_count": metrics["hit_bet_count"],
        "total_stake": metrics["total_stake"],
        "total_return": metrics["total_return"],
        "roi": metrics["roi"],
        "roi_reliable": metrics["roi_reliable"],
        "warnings": context["warnings"],
    }


def run_backtest_command(
    from_date: str,
    to_date: str,
    modes: list[str] | None = None,
    min_races: int = 1,
    write_json: bool | None = None,
    write_md: bool | None = None,
    enable_borderline_recovery: bool = False,
    scope_key: str | None = None,
) -> dict[str, object]:
    report = run_backtest(
        from_date=from_date,
        to_date=to_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        modes=modes,
        min_races=min_races,
        enable_borderline_recovery=enable_borderline_recovery,
        scope_key=scope_key,
    )
    period_key = f"{from_date}_{to_date}{_scope_output_suffix(scope_key)}"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_backtest_json(report, DEFAULT_BACKTESTS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_backtest_markdown(report, modes=modes)
        md_path = save_backtest_markdown(markdown, DEFAULT_BACKTESTS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "scope_key": report.get("scope_key"),
        "race_count": report["race_count"],
        "reviewed_race_count": report["reviewed_race_count"],
        "pending_race_count": report["pending_race_count"],
        "total_stake": report["total_stake"],
        "total_return": report["total_return"],
        "roi": report["roi"],
        "roi_reliable": report["roi_reliable"],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def run_backtest_weights_command(
    from_date: str,
    to_date: str,
    modes: list[str] | None = None,
    min_races: int = 1,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    write_json: bool | None = None,
    write_md: bool | None = None,
    enable_borderline_recovery: bool = False,
    scope_key: str | None = None,
) -> dict[str, object]:
    report = run_backtest_weights(
        from_date=from_date,
        to_date=to_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        modes=modes,
        min_races=min_races,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
        enable_borderline_recovery=enable_borderline_recovery,
        scope_key=scope_key,
    )
    period_key = f"{from_date}_{to_date}{_scope_output_suffix(scope_key)}_weight_tuning"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_backtest_json(report, DEFAULT_BACKTESTS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_weight_tuning_markdown(report, modes=modes)
        md_path = save_backtest_markdown(markdown, DEFAULT_BACKTESTS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "scope_key": report.get("scope_key"),
        "race_count": report["race_count"],
        "reviewed_race_count": report["reviewed_race_count"],
        "pending_race_count": report["pending_race_count"],
        "total_stake": report["total_stake"],
        "total_return": report["total_return"],
        "roi": report["roi"],
        "roi_reliable": report["roi_reliable"],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def run_missed_top3_analysis_command(
    from_date: str,
    to_date: str,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    min_popularity: int | None = None,
    finish_filter: int | None = None,
    top_n: int = 5,
    write_json: bool | None = None,
    write_md: bool | None = None,
    simulate_borderline_recovery: bool = False,
    scope_key: str | None = None,
) -> dict[str, object]:
    report = run_missed_top3_analysis(
        from_date=from_date,
        to_date=to_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        race_data_dir=DEFAULT_RACE_DATA_DIR,
        scoring_mode=scoring_mode or "candidate_default",
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
        min_popularity=min_popularity,
        finish_filter=finish_filter,
        top_n=top_n,
        simulate_borderline_recovery=simulate_borderline_recovery,
        scope_key=scope_key,
    )
    period_key = f"{from_date}_{to_date}{_scope_output_suffix(scope_key)}_missed_top3"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_missed_top3_json(report, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_missed_top3_markdown(report)
        md_path = save_missed_top3_markdown(markdown, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "scope_key": report.get("scope_key"),
        "reviewed_race_count": report["summary"]["reviewed_race_count"],
        "captured_top3_horses": report["summary"]["captured_top3_horses"],
        "missed_top3_horses": report["summary"]["missed_top3_horses"],
        "capture_rate": report["summary"]["capture_rate"],
        "recovery_candidate_count": report["summary"].get("recovery_candidate_count", 0),
        "theoretically_recoverable_top3_count": report["summary"].get("theoretically_recoverable_top3_count", 0),
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def run_deep_miss_analysis_command(
    from_date: str,
    to_date: str,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    min_popularity: int | None = None,
    finish_filter: int | None = None,
    write_json: bool | None = None,
    write_md: bool | None = None,
) -> dict[str, object]:
    report = run_deep_miss_analysis(
        from_date=from_date,
        to_date=to_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        race_data_dir=DEFAULT_RACE_DATA_DIR,
        scoring_mode=scoring_mode or "candidate_default",
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
        min_popularity=min_popularity,
        finish_filter=finish_filter,
    )
    period_key = f"{from_date}_{to_date}_deep_miss_top3"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_deep_miss_json(report, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_deep_miss_markdown(report)
        md_path = save_deep_miss_markdown(markdown, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "reviewed_race_count": report["summary"]["reviewed_race_count"],
        "total_top3_horses": report["summary"]["total_top3_horses"],
        "low_rank_top3_horses": report["summary"]["low_rank_top3_horses"],
        "severity_counts": report["severity_counts"],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def run_deep_miss_rule_simulation_command(
    from_date: str,
    to_date: str,
    baseline_mode: str | None = None,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    max_rank: int = 10,
    include_rank13: bool = False,
    min_positive_signals: int | None = None,
    write_json: bool | None = None,
    write_md: bool | None = None,
) -> dict[str, object]:
    report = run_deep_miss_rule_simulation(
        from_date=from_date,
        to_date=to_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        race_data_dir=DEFAULT_RACE_DATA_DIR,
        baseline_mode=baseline_mode,
        scoring_mode=scoring_mode,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
        max_rank=max_rank,
        include_rank13=include_rank13,
        min_positive_signals=min_positive_signals,
    )
    period_key = f"{from_date}_{to_date}_deep_miss_rule_simulation"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_deep_miss_rule_simulation_json(report, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_deep_miss_rule_simulation_markdown(report)
        md_path = save_deep_miss_rule_simulation_markdown(markdown, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "reviewed_race_count": report["summary"]["reviewed_race_count"],
        "baseline_mode": report["analysis_config"]["baseline_mode"],
        "baseline_avg_captured_top3_per_race": report["summary"]["baseline_avg_captured_top3_per_race"],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def run_penalty_refinement_simulation_command(
    from_date: str,
    to_date: str,
    baseline_mode: str | None = None,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    max_rank: int = 12,
    include_rank13: bool = False,
    write_json: bool | None = None,
    write_md: bool | None = None,
) -> dict[str, object]:
    report = run_penalty_refinement_simulation(
        from_date=from_date,
        to_date=to_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        race_data_dir=DEFAULT_RACE_DATA_DIR,
        baseline_mode=baseline_mode,
        scoring_mode=scoring_mode,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
        max_rank=max_rank,
        include_rank13=include_rank13,
    )
    period_key = f"{from_date}_{to_date}_penalty_refinement_simulation"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_penalty_refinement_simulation_json(report, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_penalty_refinement_simulation_markdown(report)
        md_path = save_penalty_refinement_simulation_markdown(markdown, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "reviewed_race_count": report["summary"]["reviewed_race_count"],
        "baseline_mode": report["analysis_config"]["baseline_mode"],
        "baseline_avg_captured_top3_per_race": report["summary"]["baseline_avg_captured_top3_per_race"],
        "best_rule": report["best_rule_candidates"]["best_rule"],
        "recommended_rules": report["best_rule_candidates"]["recommended_rules"],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def run_score_recalibration_simulation_command(
    from_date: str,
    to_date: str,
    baseline_mode: str | None = None,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    write_json: bool | None = None,
    write_md: bool | None = None,
) -> dict[str, object]:
    report = run_score_recalibration_simulation(
        from_date=from_date,
        to_date=to_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        baseline_mode=baseline_mode,
        scoring_mode=scoring_mode,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
    )
    period_key = f"{from_date}_{to_date}_score_recalibration_simulation"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_score_recalibration_simulation_json(report, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_score_recalibration_simulation_markdown(report)
        md_path = save_score_recalibration_simulation_markdown(markdown, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "reviewed_race_count": report["summary"]["reviewed_race_count"],
        "baseline_mode": report["analysis_config"]["baseline_mode"],
        "baseline_avg_captured_top3_per_race": report["summary"]["baseline_avg_captured_top3_per_race"],
        "best_rule": report["best_rule_candidates"]["best_rule"],
        "recommended_rules": report["best_rule_candidates"]["recommended_rules"],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def run_condition_weight_simulation_command(
    from_date: str,
    to_date: str,
    write_json: bool | None = None,
    write_md: bool | None = None,
) -> dict[str, object]:
    report = run_condition_weight_simulation(
        from_date=from_date,
        to_date=to_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        race_data_dir=DEFAULT_RACE_DATA_DIR,
    )
    period_key = f"{from_date}_{to_date}_condition_weight_simulation"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_condition_weight_simulation_json(report, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_condition_weight_simulation_markdown(report)
        md_path = save_condition_weight_simulation_markdown(markdown, DEFAULT_ERROR_ANALYSIS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "reviewed_race_count": report["summary"]["reviewed_race_count"],
        "baseline_avg_marked_top3_count": report["summary"]["baseline_avg_marked_top3_count"],
        "best_scenario": report["best_scenario_summary"]["best_avg_marked_top3_count"],
        "recommended_scenarios": report["best_scenario_summary"]["recommended_scenarios"],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def run_audit_race_data_flow_command(
    race_id: str,
    write_json: bool | None = None,
    write_md: bool | None = None,
) -> dict[str, object]:
    audit = audit_single_race_data_flow(
        race_id=race_id,
        race_data_dir=DEFAULT_RACE_DATA_DIR,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
    )
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_audit_json(audit, DEFAULT_AUDITS_DIR / f"{race_id}_data_flow_audit.json")
    if should_write_md:
        markdown = generate_single_audit_markdown(audit)
        md_path = save_audit_markdown(markdown, DEFAULT_AUDITS_DIR / f"{race_id}_data_flow_audit.md")
    return {
        "race_id": race_id,
        "status": audit["status"],
        "readiness": audit["readiness"],
        "issue_counts": audit["issue_counts"],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
    }


def run_audit_data_flow_command(
    from_date: str,
    to_date: str,
    write_json: bool | None = None,
    write_md: bool | None = None,
) -> dict[str, object]:
    report = audit_data_flow_period(
        from_date=from_date,
        to_date=to_date,
        race_data_dir=DEFAULT_RACE_DATA_DIR,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
    )
    period_key = f"{from_date}_{to_date}_data_flow_audit"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_audit_json(report, DEFAULT_AUDITS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_period_audit_markdown(report)
        md_path = save_audit_markdown(markdown, DEFAULT_AUDITS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "summary": report["summary"],
        "issue_counts": report["issue_counts"],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def run_score_factor_analysis_command(
    from_date: str,
    to_date: str,
    top_n: int = 5,
    write_json: bool | None = None,
    write_md: bool | None = None,
) -> dict[str, object]:
    report = run_score_factor_analysis(
        from_date=from_date,
        to_date=to_date,
        predictions_dir=DEFAULT_PREDICTIONS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        reviews_dir=DEFAULT_REVIEWS_DIR,
        top_n=top_n,
    )
    period_key = f"{from_date}_{to_date}_score_factor_analysis"
    should_write_json = True if write_json is None and write_md is None else bool(write_json)
    should_write_md = True if write_json is None and write_md is None else bool(write_md)
    json_path: Path | None = None
    md_path: Path | None = None
    if should_write_json:
        json_path = save_score_factor_analysis_json(report, DEFAULT_ANALYSIS_REPORTS_DIR / f"{period_key}.json")
    if should_write_md:
        markdown = generate_score_factor_analysis_markdown(report)
        md_path = save_score_factor_analysis_markdown(markdown, DEFAULT_ANALYSIS_REPORTS_DIR / f"{period_key}.md")
    return {
        "period": report["period"],
        "summary": report["summary"],
        "top_factor_diffs": report["factor_comparison"][:5],
        "json_path": str(json_path) if json_path is not None else None,
        "md_path": str(md_path) if md_path is not None else None,
        "warnings": report["warnings"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keiba LLM Agent MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analysis_parser = subparsers.add_parser("analysis", help="生成赛前 prediction.json")
    analysis_parser.add_argument("--race-data", required=True, help="race_data.json 路径")
    analysis_parser.add_argument("--output", help="prediction.json 输出路径，可选")
    analysis_parser.add_argument(
        "--lessons",
        default=str(DEFAULT_LESSONS_PATH),
        help="lessons.json 路径，可选",
    )
    _add_scoring_arguments(analysis_parser)

    review_parser = subparsers.add_parser("review", help="生成赛后 review.json")
    review_parser.add_argument("--race-id", required=True, help="比赛 ID")
    review_parser.add_argument("--result", required=True, help="result.json 路径")
    review_parser.add_argument("--prediction", help="prediction.json 路径，可选")
    review_parser.add_argument("--output", help="review.json 输出路径，可选")
    review_parser.add_argument(
        "--lessons",
        default=str(DEFAULT_LESSONS_PATH),
        help="lessons.json 路径，可选",
    )

    parse_url_parser = subparsers.add_parser("parse-url", help="从 netkeiba URL 提取 race_id")
    parse_url_parser.add_argument("--url", required=True, help="netkeiba 出马表或结果页 URL")

    parse_html_parser = subparsers.add_parser("parse-html", help="解析本地 netkeiba 出马表 HTML")
    parse_html_parser.add_argument("--html", required=True, help="本地 HTML 文件路径")
    parse_html_parser.add_argument("--race-id", help="显式传入 race_id，可选")

    fetch_race_parser = subparsers.add_parser("fetch-race", help="抓取并缓存 netkeiba 出马表 HTML")
    fetch_race_parser.add_argument("--url", required=True, help="netkeiba 出马表 URL")
    fetch_race_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载 HTML，而不是使用缓存",
    )
    fetch_race_parser.add_argument(
        "--with-recent-runs",
        action="store_true",
        help="同时抓取每匹马的近走成绩并写入 race_data.json",
    )
    fetch_race_parser.add_argument(
        "--recent-run-limit",
        type=int,
        default=None,
        help="每匹马抓取的近走条数，默认不限制",
    )

    analyze_url_parser = subparsers.add_parser("analyze-url", help="从 netkeiba URL 直接生成 prediction.json")
    analyze_url_parser.add_argument("--url", required=True, help="netkeiba 出马表 URL")
    analyze_url_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载 HTML，而不是使用缓存",
    )
    analyze_url_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只抓取并解析 URL，保存 race_data.json，不执行 analysis",
    )
    analyze_url_parser.add_argument(
        "--with-recent-runs",
        action="store_true",
        help="同时抓取每匹马的近走成绩并写入 race_data.json",
    )
    analyze_url_parser.add_argument(
        "--recent-run-limit",
        type=int,
        default=None,
        help="每匹马抓取的近走条数，默认不限制",
    )
    _add_scoring_arguments(analyze_url_parser)

    fetch_horse_parser = subparsers.add_parser("fetch-horse", help="抓取单匹马页面并解析 recent_runs")
    fetch_horse_parser.add_argument("--horse-id", required=True, help="netkeiba horse_id")
    fetch_horse_parser.add_argument("--limit", type=int, default=None, help="抓取的近走条数，默认不限制")
    fetch_horse_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载 horse HTML，而不是使用缓存",
    )

    parse_pedigree_parser = subparsers.add_parser("parse-pedigree", help="解析单匹马的父・母・母父")
    parse_pedigree_parser.add_argument("--horse-id", required=True, help="netkeiba horse_id")
    parse_pedigree_parser.add_argument("--horse-name", help="可选 horse_name")
    parse_pedigree_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载 pedigree HTML，而不是使用缓存",
    )

    fetch_result_parser = subparsers.add_parser("fetch-result", help="抓取并解析 netkeiba result HTML")
    fetch_result_parser.add_argument("--url", required=True, help="netkeiba result URL")
    fetch_result_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载 result HTML，而不是使用缓存",
    )

    review_url_parser = subparsers.add_parser("review-url", help="从 netkeiba result URL 直接生成 review.json")
    review_url_parser.add_argument("--url", required=True, help="netkeiba result URL")
    review_url_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载 result HTML，而不是使用缓存",
    )

    report_prediction_parser = subparsers.add_parser("report-prediction", help="生成 prediction Markdown 报告")
    report_prediction_parser.add_argument("--race-id", required=True, help="比赛 ID")
    report_prediction_parser.add_argument("--prediction", help="prediction.json 路径，可选")
    report_prediction_parser.add_argument("--output", help="Markdown 输出路径，可选")

    report_review_parser = subparsers.add_parser("report-review", help="生成 review Markdown 报告")
    report_review_parser.add_argument("--race-id", required=True, help="比赛 ID")
    report_review_parser.add_argument("--prediction", help="prediction.json 路径，可选")
    report_review_parser.add_argument("--review", help="review.json 路径，可选")
    report_review_parser.add_argument("--output", help="Markdown 输出路径，可选")

    social_prediction_parser = subparsers.add_parser("social-prediction", help="生成 prediction 社交短文")
    social_prediction_parser.add_argument("--race-id", required=True, help="比赛 ID")
    social_prediction_parser.add_argument("--prediction", help="prediction.json 路径，可选")
    social_prediction_parser.add_argument("--output", help="txt 输出路径，可选")

    social_review_parser = subparsers.add_parser("social-review", help="生成 review 社交短文")
    social_review_parser.add_argument("--race-id", required=True, help="比赛 ID")
    social_review_parser.add_argument("--prediction", help="prediction.json 路径，可选")
    social_review_parser.add_argument("--review", help="review.json 路径，可选")
    social_review_parser.add_argument("--output", help="txt 输出路径，可选")

    predict_race_parser = subparsers.add_parser("predict-race", help="一键执行赛前 pipeline")
    predict_race_parser.add_argument("--url", required=True, help="netkeiba 出马表 URL")
    predict_race_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载 HTML，而不是使用缓存",
    )
    predict_race_parser.add_argument(
        "--recent-run-limit",
        type=int,
        default=None,
        help="每匹马抓取的近走条数，默认不限制",
    )
    predict_race_parser.add_argument(
        "--skip-report",
        action="store_true",
        help="跳过 prediction markdown report 生成",
    )
    predict_race_parser.add_argument(
        "--enable-report",
        action="store_true",
        help="显式生成 prediction markdown report",
    )
    predict_race_parser.add_argument(
        "--skip-social",
        action="store_true",
        help="跳过 prediction social post 生成",
    )
    predict_race_parser.add_argument(
        "--llm-provider",
        choices=["mock", "openai"],
        help="覆盖环境变量中的 LLM provider",
    )
    _add_scoring_arguments(predict_race_parser)

    review_race_parser = subparsers.add_parser("review-race", help="一键执行赛后 pipeline")
    review_race_parser.add_argument("--url", required=True, help="netkeiba result URL")
    review_race_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载 result HTML，而不是使用缓存",
    )
    review_race_parser.add_argument(
        "--skip-report",
        action="store_true",
        help="跳过 review markdown report 生成",
    )
    review_race_parser.add_argument(
        "--enable-report",
        action="store_true",
        help="显式生成 review markdown report",
    )
    review_race_parser.add_argument(
        "--skip-social",
        action="store_true",
        help="跳过 review social post 生成",
    )
    review_race_parser.add_argument(
        "--llm-provider",
        choices=["mock", "openai"],
        help="覆盖环境变量中的 LLM provider",
    )

    validate_race_data_parser = subparsers.add_parser("validate-race-data", help="校验 race_data.json 数据质量")
    validate_race_data_parser.add_argument("--race-id", required=True, help="比赛 ID")
    validate_race_data_parser.add_argument("--race-data", help="自定义 race_data.json 路径，可选")
    subparsers.add_parser("llm-check", help="检查当前 LLM provider 与最小 JSON 调用")

    lessons_list_parser = subparsers.add_parser("lessons-list", help="列出当前 lessons memory")
    lessons_list_parser.add_argument(
        "--lessons",
        default=str(DEFAULT_LESSONS_PATH),
        help="lessons.json 路径，可选",
    )
    lessons_disable_parser = subparsers.add_parser("lessons-disable", help="禁用指定 lesson")
    lessons_disable_parser.add_argument("--lesson-id", required=True, help="lesson_id")
    lessons_disable_parser.add_argument(
        "--lessons",
        default=str(DEFAULT_LESSONS_PATH),
        help="lessons.json 路径，可选",
    )
    lessons_enable_parser = subparsers.add_parser("lessons-enable", help="启用指定 lesson")
    lessons_enable_parser.add_argument("--lesson-id", required=True, help="lesson_id")
    lessons_enable_parser.add_argument(
        "--lessons",
        default=str(DEFAULT_LESSONS_PATH),
        help="lessons.json 路径，可选",
    )
    lessons_prune_parser = subparsers.add_parser("lessons-prune", help="按 score 禁用低质量 lessons")
    lessons_prune_parser.add_argument("--min-score", type=float, default=0.2, help="低于该分数则禁用")
    lessons_prune_parser.add_argument(
        "--lessons",
        default=str(DEFAULT_LESSONS_PATH),
        help="lessons.json 路径，可选",
    )

    daily_summary_parser = subparsers.add_parser("daily-summary", help="生成指定日期的日次汇总报告与社交短文")
    daily_summary_parser.add_argument("--date", required=True, help="目标日期，格式 YYYY-MM-DD")
    daily_summary_parser.add_argument(
        "--scope-key",
        choices=SCOPE_KEY_CHOICES,
        help="仅汇总指定 scope 的预测样本",
    )
    daily_summary_parser.add_argument(
        "--skip-report",
        action="store_true",
        help="跳过 daily markdown report 生成",
    )
    daily_summary_parser.add_argument(
        "--enable-report",
        action="store_true",
        help="显式生成 daily markdown report",
    )
    daily_summary_parser.add_argument(
        "--skip-social",
        action="store_true",
        help="跳过 daily social post 生成",
    )

    backtest_parser = subparsers.add_parser("backtest", help="比较不同 scoring 模式在指定期间的表现")
    backtest_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    backtest_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    backtest_parser.add_argument(
        "--scope-key",
        choices=SCOPE_KEY_CHOICES,
        help="仅统计指定 scope 的预测样本",
    )
    backtest_parser.add_argument(
        "--mode",
        action="append",
        choices=["base_only", "pedigree_only", "full_adjusted"],
        help="仅输出指定 mode，可重复传入",
    )
    backtest_parser.add_argument("--min-races", type=int, default=1, help="最少样本数要求，默认 1")
    backtest_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 backtest json（若不传 output-json/output-md，则默认两者都输出）",
    )
    backtest_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 backtest markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    backtest_parser.add_argument(
        "--enable-borderline-recovery",
        action="store_true",
        help="在 backtest 中启用 Top5 境界補正",
    )
    backtest_parser.add_argument(
        "--disable-borderline-recovery",
        action="store_true",
        help="在 backtest 中显式禁用 Top5 境界補正",
    )
    backtest_weights_parser = subparsers.add_parser("backtest-weights", help="比较不同 adjustment 权重组合的表现")
    backtest_weights_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    backtest_weights_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    backtest_weights_parser.add_argument(
        "--scope-key",
        choices=SCOPE_KEY_CHOICES,
        help="仅统计指定 scope 的预测样本",
    )
    backtest_weights_parser.add_argument(
        "--mode",
        action="append",
        choices=[
            "base_only",
            "current_full",
            "conservative_full",
            "no_pace",
            "no_race_level",
            "race_level_only",
            "pace_only",
            "pedigree_only",
            "candidate_default",
            "candidate_default_recovered",
            "local_candidate_default",
            "local_candidate_default_recovered",
            "custom",
        ],
        help="仅输出指定 mode，可重复传入",
    )
    backtest_weights_parser.add_argument("--min-races", type=int, default=1, help="最少样本数要求，默认 1")
    backtest_weights_parser.add_argument("--pedigree-weight", type=float, help="custom mode 的 pedigree 权重")
    backtest_weights_parser.add_argument("--race-level-weight", type=float, help="custom mode 的 race_level 权重")
    backtest_weights_parser.add_argument("--pace-weight", type=float, help="custom mode 的 pace 权重")
    backtest_weights_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 weight tuning json（若不传 output-json/output-md，则默认两者都输出）",
    )
    backtest_weights_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 weight tuning markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    backtest_weights_parser.add_argument(
        "--enable-borderline-recovery",
        action="store_true",
        help="在 weight tuning 中启用 Top5 境界補正",
    )
    backtest_weights_parser.add_argument(
        "--disable-borderline-recovery",
        action="store_true",
        help="在 weight tuning 中显式禁用 Top5 境界補正",
    )
    missed_top3_parser = subparsers.add_parser("missed-top3-analysis", help="分析实际Top3中漏出 predicted topN 的马")
    missed_top3_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    missed_top3_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    missed_top3_parser.add_argument(
        "--scope-key",
        choices=SCOPE_KEY_CHOICES,
        help="仅统计指定 scope 的预测样本",
    )
    missed_top3_parser.add_argument(
        "--scoring-mode",
        choices=SCORING_MODE_CHOICES,
        default="candidate_default",
        help="用于重排 topN 的评分模式，默认 candidate_default",
    )
    missed_top3_parser.add_argument("--pedigree-weight", type=float, help="custom / override 的 pedigree 权重")
    missed_top3_parser.add_argument("--race-level-weight", type=float, help="custom / override 的 race_level 权重")
    missed_top3_parser.add_argument("--pace-weight", type=float, help="custom / override 的 pace 权重")
    missed_top3_parser.add_argument("--min-popularity", type=int, help="只分析 popularity >= N 的漏网马")
    missed_top3_parser.add_argument("--finish", type=int, choices=[1, 2, 3], help="只分析指定着顺")
    missed_top3_parser.add_argument("--top-n", type=int, default=5, help="按 topN 判断是否漏网，默认 5")
    missed_top3_parser.add_argument(
        "--simulate-borderline-recovery",
        action="store_true",
        help="模拟 Top5 境界補正的理论回收效果",
    )
    missed_top3_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 missed-top3 json（若不传 output-json/output-md，则默认两者都输出）",
    )
    missed_top3_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 missed-top3 markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    deep_miss_parser = subparsers.add_parser("deep-miss-analysis", help="分析实际Top3但预测 rank>=7 的低排位漏马")
    deep_miss_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    deep_miss_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    deep_miss_parser.add_argument(
        "--scoring-mode",
        choices=SCORING_MODE_CHOICES,
        default="candidate_default",
        help="用于重排名次的评分模式，默认 candidate_default",
    )
    deep_miss_parser.add_argument("--pedigree-weight", type=float, help="custom / override 的 pedigree 权重")
    deep_miss_parser.add_argument("--race-level-weight", type=float, help="custom / override 的 race_level 权重")
    deep_miss_parser.add_argument("--pace-weight", type=float, help="custom / override 的 pace 权重")
    deep_miss_parser.add_argument("--min-popularity", type=int, help="只分析 popularity >= N 的低排位漏马")
    deep_miss_parser.add_argument("--finish", type=int, choices=[1, 2, 3], help="只分析指定着顺")
    deep_miss_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 deep-miss json（若不传 output-json/output-md，则默认两者都输出）",
    )
    deep_miss_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 deep-miss markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    deep_miss_rule_parser = subparsers.add_parser("deep-miss-rule-simulate", help="模拟低排位漏马 safety net 规则的 what-if 效果")
    deep_miss_rule_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    deep_miss_rule_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    deep_miss_rule_parser.add_argument(
        "--baseline-mode",
        choices=BASELINE_MODE_CHOICES,
        help="baseline top5 模式；默认 candidate_default_recovered，若未传且指定 scoring-mode，则跟随 scoring-mode",
    )
    deep_miss_rule_parser.add_argument(
        "--scoring-mode",
        choices=SCORING_MODE_CHOICES,
        help="baseline-mode 未指定时可作为 fallback baseline；custom 时可配合权重",
    )
    deep_miss_rule_parser.add_argument("--pedigree-weight", type=float, help="custom baseline 的 pedigree 权重")
    deep_miss_rule_parser.add_argument("--race-level-weight", type=float, help="custom baseline 的 race_level 权重")
    deep_miss_rule_parser.add_argument("--pace-weight", type=float, help="custom baseline 的 pace 权重")
    deep_miss_rule_parser.add_argument("--max-rank", type=int, choices=[8, 10, 12], default=10, help="限制模拟候选的最大 rank，默认 10")
    deep_miss_rule_parser.add_argument(
        "--include-rank13",
        action="store_true",
        help="允许 rank13+ 进入模拟候选，默认关闭",
    )
    deep_miss_rule_parser.add_argument("--min-positive-signals", type=int, help="覆盖各 rule 的最小 positive signal 要求")
    deep_miss_rule_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 deep-miss-rule-simulation json（若不传 output-json/output-md，则默认两者都输出）",
    )
    deep_miss_rule_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 deep-miss-rule-simulation markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    penalty_refinement_parser = subparsers.add_parser("penalty-refinement-simulate", help="模拟 penalty refinement / positive stack protection 的 what-if 效果")
    penalty_refinement_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    penalty_refinement_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    penalty_refinement_parser.add_argument(
        "--baseline-mode",
        choices=BASELINE_MODE_CHOICES,
        help="baseline top5 模式；默认 candidate_default_recovered，若未传且指定 scoring-mode，则跟随 scoring-mode",
    )
    penalty_refinement_parser.add_argument(
        "--scoring-mode",
        choices=SCORING_MODE_CHOICES,
        help="baseline-mode 未指定时可作为 fallback baseline；custom 时可配合权重",
    )
    penalty_refinement_parser.add_argument("--pedigree-weight", type=float, help="custom baseline 的 pedigree 权重")
    penalty_refinement_parser.add_argument("--race-level-weight", type=float, help="custom baseline 的 race_level 权重")
    penalty_refinement_parser.add_argument("--pace-weight", type=float, help="custom baseline 的 pace 权重")
    penalty_refinement_parser.add_argument("--max-rank", type=int, choices=[8, 10, 12], default=12, help="限制模拟候选的最大 rank，默认 12")
    penalty_refinement_parser.add_argument(
        "--include-rank13",
        action="store_true",
        help="允许 rank13+ 进入模拟候选，默认关闭",
    )
    penalty_refinement_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 penalty-refinement json（若不传 output-json/output-md，则默认两者都输出）",
    )
    penalty_refinement_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 penalty-refinement markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    score_recalibration_parser = subparsers.add_parser("score-recalibration-simulate", help="模拟评分校准规则的 what-if 效果")
    score_recalibration_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    score_recalibration_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    score_recalibration_parser.add_argument(
        "--baseline-mode",
        choices=BASELINE_MODE_CHOICES,
        help="baseline top5 模式；默认 candidate_default_recovered，若未传且指定 scoring-mode，则跟随 scoring-mode",
    )
    score_recalibration_parser.add_argument(
        "--scoring-mode",
        choices=SCORING_MODE_CHOICES,
        help="baseline-mode 未指定时可作为 fallback baseline；custom 时可配合权重",
    )
    score_recalibration_parser.add_argument("--pedigree-weight", type=float, help="custom baseline 的 pedigree 权重")
    score_recalibration_parser.add_argument("--race-level-weight", type=float, help="custom baseline 的 race_level 权重")
    score_recalibration_parser.add_argument("--pace-weight", type=float, help="custom baseline 的 pace 权重")
    score_recalibration_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 score-recalibration json（若不传 output-json/output-md，则默认两者都输出）",
    )
    score_recalibration_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 score-recalibration markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    condition_weight_parser = subparsers.add_parser("condition-weight-simulate", help="模拟条件型权重与rank5/rank6边界优化的 what-if 效果")
    condition_weight_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    condition_weight_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    condition_weight_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 condition-weight json（若不传 output-json/output-md，则默认两者都输出）",
    )
    condition_weight_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 condition-weight markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    audit_race_parser = subparsers.add_parser("audit-race-data-flow", help="审计单场 race_data / prediction / result / review 的数据来源与一致性")
    audit_race_parser.add_argument("--race-id", required=True, help="比赛 ID")
    audit_race_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 audit json（若不传 output-json/output-md，则默认两者都输出）",
    )
    audit_race_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 audit markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    audit_period_parser = subparsers.add_parser("audit-data-flow", help="按期间批量审计数据来源与一致性")
    audit_period_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    audit_period_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    audit_period_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 audit json（若不传 output-json/output-md，则默认两者都输出）",
    )
    audit_period_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 audit markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    score_factor_parser = subparsers.add_parser("score-factor-analysis", help="分析评分分项与实际Top3/漏马之间的关系")
    score_factor_parser.add_argument("--from-date", required=True, help="开始日期，格式 YYYY-MM-DD")
    score_factor_parser.add_argument("--to-date", required=True, help="结束日期，格式 YYYY-MM-DD")
    score_factor_parser.add_argument("--top-n", type=int, default=5, help="按 topN 判断 captured/missed，默认 5")
    score_factor_parser.add_argument(
        "--output-json",
        action="store_true",
        help="仅输出 score-factor json（若不传 output-json/output-md，则默认两者都输出）",
    )
    score_factor_parser.add_argument(
        "--output-md",
        action="store_true",
        help="仅输出 score-factor markdown（若不传 output-json/output-md，则默认两者都输出）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analysis":
        prediction, saved_path = run_analysis(
            race_data_path=args.race_data,
            output_path=args.output,
            lessons_path=args.lessons,
            scoring_profile=args.scoring_profile,
            scoring_mode=args.scoring_mode,
            pedigree_weight=args.pedigree_weight,
            race_level_weight=args.race_level_weight,
            pace_weight=args.pace_weight,
            borderline_recovery_enabled=_resolve_borderline_recovery_override(
                enable_borderline_recovery=args.enable_borderline_recovery,
                disable_borderline_recovery=args.disable_borderline_recovery,
            ),
        )
        print(
            json.dumps(
                {
                    "saved_to": str(saved_path),
                    "prediction": prediction.model_dump(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "review":
        result, saved_path = run_review(
            race_id=args.race_id,
            result_path=args.result,
            prediction_path=args.prediction,
            output_path=args.output,
            lessons_path=args.lessons,
        )
        review = result["review"]
        print(
            json.dumps(
                {
                    "saved_to": str(saved_path),
                    "review": review.model_dump(),
                    "total_lessons": result["total_lessons"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "parse-url":
        race_id = run_parse_url(args.url)
        print(f"race_id: {race_id}")
        return 0

    if args.command == "parse-html":
        race_data = run_parse_html(args.html, race_id=args.race_id)
        print(json.dumps(race_data.model_dump(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "fetch-race":
        result = run_fetch_race(
            args.url,
            force_refresh=args.force_refresh,
            with_recent_runs=args.with_recent_runs,
            recent_run_limit=args.recent_run_limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "analyze-url":
        result = run_analyze_url(
            args.url,
            force_refresh=args.force_refresh,
            dry_run=args.dry_run,
            with_recent_runs=args.with_recent_runs,
            recent_run_limit=args.recent_run_limit,
            scoring_profile=args.scoring_profile,
            scoring_mode=args.scoring_mode,
            pedigree_weight=args.pedigree_weight,
            race_level_weight=args.race_level_weight,
            pace_weight=args.pace_weight,
            borderline_recovery_enabled=_resolve_borderline_recovery_override(
                enable_borderline_recovery=args.enable_borderline_recovery,
                disable_borderline_recovery=args.disable_borderline_recovery,
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "fetch-horse":
        result = run_fetch_horse(
            args.horse_id,
            limit=args.limit,
            force_refresh=args.force_refresh,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "parse-pedigree":
        result = run_parse_pedigree(
            args.horse_id,
            horse_name=args.horse_name,
            force_refresh=args.force_refresh,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "fetch-result":
        result = run_fetch_result(args.url, force_refresh=args.force_refresh)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "review-url":
        result = run_review_url(args.url, force_refresh=args.force_refresh)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "report-prediction":
        result = run_report_prediction(
            race_id=args.race_id,
            prediction_path=args.prediction,
            output_path=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "report-review":
        result = run_report_review(
            race_id=args.race_id,
            prediction_path=args.prediction,
            review_path=args.review,
            output_path=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "social-prediction":
        result = run_social_prediction(
            race_id=args.race_id,
            prediction_path=args.prediction,
            output_path=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "social-review":
        result = run_social_review(
            race_id=args.race_id,
            prediction_path=args.prediction,
            review_path=args.review,
            output_path=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "predict-race":
        result = run_predict_race(
            args.url,
            force_refresh=args.force_refresh,
            recent_run_limit=args.recent_run_limit,
            skip_report=_resolve_skip_report(
                skip_report=args.skip_report,
                enable_report=args.enable_report,
            ),
            skip_social=args.skip_social,
            llm_provider=args.llm_provider,
            scoring_profile=args.scoring_profile,
            scoring_mode=args.scoring_mode,
            pedigree_weight=args.pedigree_weight,
            race_level_weight=args.race_level_weight,
            pace_weight=args.pace_weight,
            borderline_recovery_enabled=_resolve_borderline_recovery_override(
                enable_borderline_recovery=args.enable_borderline_recovery,
                disable_borderline_recovery=args.disable_borderline_recovery,
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "review-race":
        result = run_review_race(
            args.url,
            force_refresh=args.force_refresh,
            skip_report=_resolve_skip_report(
                skip_report=args.skip_report,
                enable_report=args.enable_report,
            ),
            skip_social=args.skip_social,
            llm_provider=args.llm_provider,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate-race-data":
        result = run_validate_race_data(
            race_id=args.race_id,
            race_data_path=args.race_data,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "llm-check":
        result = run_llm_check()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "lessons-list":
        result = run_lessons_list(args.lessons)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "lessons-disable":
        result = run_lessons_disable(args.lesson_id, args.lessons)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "lessons-enable":
        result = run_lessons_enable(args.lesson_id, args.lessons)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "lessons-prune":
        result = run_lessons_prune(args.min_score, args.lessons)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "daily-summary":
        result = run_daily_summary(
            args.date,
            skip_report=_resolve_skip_report(
                skip_report=args.skip_report,
                enable_report=args.enable_report,
            ),
            skip_social=args.skip_social,
            scope_key=args.scope_key,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "backtest":
        selected_outputs = args.output_json or args.output_md
        result = run_backtest_command(
            from_date=args.from_date,
            to_date=args.to_date,
            modes=args.mode,
            min_races=args.min_races,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
            enable_borderline_recovery=_resolve_borderline_recovery_enabled(
                "base_only",
                enable_borderline_recovery=args.enable_borderline_recovery,
                disable_borderline_recovery=args.disable_borderline_recovery,
            ),
            scope_key=args.scope_key,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "backtest-weights":
        selected_outputs = args.output_json or args.output_md
        result = run_backtest_weights_command(
            from_date=args.from_date,
            to_date=args.to_date,
            modes=args.mode,
            min_races=args.min_races,
            pedigree_weight=args.pedigree_weight,
            race_level_weight=args.race_level_weight,
            pace_weight=args.pace_weight,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
            enable_borderline_recovery=_resolve_borderline_recovery_enabled(
                "base_only",
                enable_borderline_recovery=args.enable_borderline_recovery,
                disable_borderline_recovery=args.disable_borderline_recovery,
            ),
            scope_key=args.scope_key,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "missed-top3-analysis":
        selected_outputs = args.output_json or args.output_md
        result = run_missed_top3_analysis_command(
            from_date=args.from_date,
            to_date=args.to_date,
            scoring_mode=args.scoring_mode,
            pedigree_weight=args.pedigree_weight,
            race_level_weight=args.race_level_weight,
            pace_weight=args.pace_weight,
            min_popularity=args.min_popularity,
            finish_filter=args.finish,
            top_n=args.top_n,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
            simulate_borderline_recovery=args.simulate_borderline_recovery,
            scope_key=args.scope_key,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "deep-miss-analysis":
        selected_outputs = args.output_json or args.output_md
        result = run_deep_miss_analysis_command(
            from_date=args.from_date,
            to_date=args.to_date,
            scoring_mode=args.scoring_mode,
            pedigree_weight=args.pedigree_weight,
            race_level_weight=args.race_level_weight,
            pace_weight=args.pace_weight,
            min_popularity=args.min_popularity,
            finish_filter=args.finish,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "deep-miss-rule-simulate":
        selected_outputs = args.output_json or args.output_md
        result = run_deep_miss_rule_simulation_command(
            from_date=args.from_date,
            to_date=args.to_date,
            baseline_mode=args.baseline_mode,
            scoring_mode=args.scoring_mode,
            pedigree_weight=args.pedigree_weight,
            race_level_weight=args.race_level_weight,
            pace_weight=args.pace_weight,
            max_rank=args.max_rank,
            include_rank13=args.include_rank13,
            min_positive_signals=args.min_positive_signals,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "penalty-refinement-simulate":
        selected_outputs = args.output_json or args.output_md
        result = run_penalty_refinement_simulation_command(
            from_date=args.from_date,
            to_date=args.to_date,
            baseline_mode=args.baseline_mode,
            scoring_mode=args.scoring_mode,
            pedigree_weight=args.pedigree_weight,
            race_level_weight=args.race_level_weight,
            pace_weight=args.pace_weight,
            max_rank=args.max_rank,
            include_rank13=args.include_rank13,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "score-recalibration-simulate":
        selected_outputs = args.output_json or args.output_md
        result = run_score_recalibration_simulation_command(
            from_date=args.from_date,
            to_date=args.to_date,
            baseline_mode=args.baseline_mode,
            scoring_mode=args.scoring_mode,
            pedigree_weight=args.pedigree_weight,
            race_level_weight=args.race_level_weight,
            pace_weight=args.pace_weight,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "condition-weight-simulate":
        selected_outputs = args.output_json or args.output_md
        result = run_condition_weight_simulation_command(
            from_date=args.from_date,
            to_date=args.to_date,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "audit-race-data-flow":
        selected_outputs = args.output_json or args.output_md
        result = run_audit_race_data_flow_command(
            race_id=args.race_id,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "audit-data-flow":
        selected_outputs = args.output_json or args.output_md
        result = run_audit_data_flow_command(
            from_date=args.from_date,
            to_date=args.to_date,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "score-factor-analysis":
        selected_outputs = args.output_json or args.output_md
        result = run_score_factor_analysis_command(
            from_date=args.from_date,
            to_date=args.to_date,
            top_n=args.top_n,
            write_json=args.output_json if selected_outputs else None,
            write_md=args.output_md if selected_outputs else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

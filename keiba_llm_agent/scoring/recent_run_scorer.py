from __future__ import annotations

from keiba_llm_agent.analysis.race_deep_analyzer import analyze_race_deeply
from keiba_llm_agent.config.scoring_config import (
    DEFAULT_CONDITIONAL_WEIGHT_PROFILE,
    DEFAULT_SCORING_WEIGHTS,
    effective_scoring_weights,
)
from keiba_llm_agent.analysis.race_level_analyzer import analyze_race_level_for_race
from keiba_llm_agent.analysis.pace_analyzer import analyze_pace_for_race
from keiba_llm_agent.pedigree.pedigree_analyzer import build_pedigree_analyses_for_race
from keiba_llm_agent.policies.bet_policy import build_bets_from_strategy, evaluate_bet_strategy
from keiba_llm_agent.scoring.analysis_adjustment_integrator import (
    build_score_breakdown,
    calculate_pace_adjustment,
    calculate_race_level_adjustment,
)
from keiba_llm_agent.scoring.borderline_recovery import apply_top5_borderline_recovery
from keiba_llm_agent.scoring.pedigree_score_adjuster import calculate_pedigree_adjustment
from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis
from keiba_llm_agent.schemas.prediction import (
    BorderlineRecoveryConfigSnapshot,
    BorderlineRecoveryResult,
    HorseScore,
    MarketSignalConfigSnapshot,
    PedigreeAdjustment,
    Prediction,
    ScoringConfigSnapshot,
    ScoreBreakdown,
    StrategyDecision,
)
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo, RecentRun
from keiba_llm_agent.schemas.review import LessonItem


WEIGHTS = [1.5, 1.2, 1.0, 0.8, 0.6]
RECENT_FORM_RUN_LIMIT = 5
RECENT_QUALITY_RUN_LIMIT = 3
ABILITY_SCORE_WEIGHT = 1.45
RECENT_QUALITY_SCORE_WEIGHT = 0.9
TREND_SCORE_WEIGHT = 0.4
CONDITION_FIT_SCORE_WEIGHT = 1.0
RACE_LEVEL_SCORE_WEIGHT = 0.45
PACE_JOCKEY_SCORE_WEIGHT = 0.45
MARK_LABELS = ("◎", "○", "▲", "△", "☆")
DEBUT_RACE_NAME_TOKENS = ("新馬", "メイクデビュー", "フレッシュチャレンジ")
UNRACED_BASE_SCORE = 8.0
UNRACED_RISK = -2
LOW_SAMPLE_PEDIGREE_WEIGHT_BY_RUN_COUNT = {
    0: 2.0,
    1: 1.7,
    2: 1.4,
    3: 1.15,
    4: 0.95,
}
LOW_SAMPLE_PEDIGREE_PRIOR_BLEND_BY_RUN_COUNT = {
    0: 1.0,
    1: 0.75,
    2: 0.55,
    3: 0.35,
    4: 0.2,
}


def clamp_score(value: float, minimum: int = 0, maximum: int = 10) -> int:
    return max(minimum, min(maximum, round(value)))


def weighted_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def get_weight(index: int) -> float:
    return WEIGHTS[index] if index < len(WEIGHTS) else WEIGHTS[-1]


def scorable_recent_runs(
    recent_runs: list[RecentRun],
    limit: int | None = RECENT_FORM_RUN_LIMIT,
) -> list[RecentRun]:
    source_runs = recent_runs if limit is None else recent_runs[:limit]
    return [run for run in source_runs if run.finish is not None]


def scorable_career_runs(recent_runs: list[RecentRun]) -> list[RecentRun]:
    return scorable_recent_runs(recent_runs, limit=None)


def has_valid_recent_result(recent_runs: list[RecentRun]) -> bool:
    return bool(scorable_recent_runs(recent_runs))


def effective_recent_run_count(horse: HorseEntry) -> int:
    return len(scorable_career_runs(horse.recent_runs))


def is_low_sample_horse(horse: HorseEntry) -> bool:
    return effective_recent_run_count(horse) < 5


def is_debut_race(race_info: RaceInfo) -> bool:
    race_name = str(race_info.race_name or "")
    return any(token in race_name for token in DEBUT_RACE_NAME_TOKENS)


def is_unraced_debut_horse(horse: HorseEntry, race_info: RaceInfo) -> bool:
    return is_debut_race(race_info) and not has_valid_recent_result(horse.recent_runs)


def effective_pedigree_weight_for_horse(
    base_weight: float,
    horse: HorseEntry,
    race_info: RaceInfo,
) -> float:
    run_count = min(effective_recent_run_count(horse), 5)
    low_sample_weight = LOW_SAMPLE_PEDIGREE_WEIGHT_BY_RUN_COUNT.get(run_count)
    if low_sample_weight is None:
        return base_weight
    return max(base_weight, low_sample_weight)


def _low_sample_prior_blend(run_count: int) -> float:
    return LOW_SAMPLE_PEDIGREE_PRIOR_BLEND_BY_RUN_COUNT.get(min(run_count, 5), 0.0)


def _blend_low_sample_prior(current_score: int, prior_score: int | None, run_count: int) -> int:
    if prior_score is None:
        return current_score
    blend = _low_sample_prior_blend(run_count)
    if blend <= 0:
        return current_score
    blended = clamp_score(current_score * (1.0 - blend) + prior_score * blend)
    if prior_score >= current_score:
        return max(current_score, blended)
    return min(current_score, blended) if run_count <= 2 else current_score


def _pedigree_distance_prior_score(pedigree_analysis: PedigreeAnalysis | None) -> int | None:
    if pedigree_analysis is None:
        return None
    positive_flags = set(pedigree_analysis.positive_flags)
    risk_flags = set(pedigree_analysis.risk_flags)
    if "PEDIGREE_DISTANCE_RISK" in risk_flags:
        return 1
    if "PEDIGREE_STAMINA_FIT" in positive_flags and "PEDIGREE_DISTANCE_FIT" in positive_flags:
        return 9
    if "PEDIGREE_STAMINA_FIT" in positive_flags or "PEDIGREE_DISTANCE_FIT" in positive_flags:
        return 8
    return None


def _pedigree_course_prior_score(
    pedigree_analysis: PedigreeAnalysis | None,
    race_info: RaceInfo,
) -> int | None:
    if pedigree_analysis is None:
        return None
    positive_flags = set(pedigree_analysis.positive_flags)
    score = 0
    if "PEDIGREE_SURFACE_FIT" in positive_flags:
        score = max(score, 8)
    if race_info.surface == "ダート" and "PEDIGREE_POWER_FIT" in positive_flags:
        score = max(score, 6)
    return score or None


def _pedigree_track_condition_prior_score(
    pedigree_analysis: PedigreeAnalysis | None,
    race_info: RaceInfo,
) -> int | None:
    if pedigree_analysis is None:
        return None
    positive_flags = set(pedigree_analysis.positive_flags)
    if "PEDIGREE_TRACK_CONDITION_FIT" in positive_flags:
        return 8
    if race_info.surface == "ダート" and "PEDIGREE_POWER_FIT" in positive_flags:
        return 6
    if race_info.track_condition and race_info.track_condition != "良" and "PEDIGREE_POWER_FIT" in positive_flags:
        return 6
    return None


def apply_low_sample_pedigree_priors(
    *,
    recent_run_count: int,
    distance_fit: int,
    course_fit: int,
    track_condition_fit: int,
    pedigree_analysis: PedigreeAnalysis | None,
    race_info: RaceInfo,
) -> tuple[int, int, int]:
    if recent_run_count >= 5 or pedigree_analysis is None:
        return distance_fit, course_fit, track_condition_fit
    return (
        _blend_low_sample_prior(
            distance_fit,
            _pedigree_distance_prior_score(pedigree_analysis),
            recent_run_count,
        ),
        _blend_low_sample_prior(
            course_fit,
            _pedigree_course_prior_score(pedigree_analysis, race_info),
            recent_run_count,
        ),
        _blend_low_sample_prior(
            track_condition_fit,
            _pedigree_track_condition_prior_score(pedigree_analysis, race_info),
            recent_run_count,
        ),
    )


def jockey_matches(current_jockey: str | None, past_jockey: str | None) -> bool:
    if not current_jockey or not past_jockey:
        return False
    current = current_jockey.strip()
    past = past_jockey.strip()
    return current.startswith(past) or past.startswith(current) or current in past or past in current


def safe_finish_bucket(run: RecentRun) -> int:
    if run.finish is None:
        return 0
    if run.finish <= 1:
        return 10
    if run.finish <= 3:
        return 8
    if run.finish <= 5:
        return 6
    if run.field_size and run.finish <= run.field_size / 2:
        return 4
    return 1


def clamp_float(value: float, minimum: float = 0.0, maximum: float = 10.0) -> float:
    return max(minimum, min(maximum, value))


def finish_percentile_score(run: RecentRun) -> float:
    if run.finish is None:
        return 0.0
    if run.field_size and run.field_size > 1:
        return clamp_float(10.0 * (run.field_size - run.finish) / (run.field_size - 1))
    return float(safe_finish_bucket(run))


def parse_margin_value(value: str | None) -> float | None:
    if not value:
        return None
    text = str(value).strip()
    symbolic = {
        "ハナ": 0.05,
        "アタマ": 0.1,
        "クビ": 0.2,
        "大": 3.0,
        "大差": 4.0,
    }
    if text in symbolic:
        return symbolic[text]
    cleaned = ""
    for char in text:
        if char.isdigit() or char in ".-":
            cleaned += char
    if not cleaned:
        return None
    try:
        return abs(float(cleaned))
    except ValueError:
        return None


def run_condition_mismatch(run: RecentRun, race_info: RaceInfo | None) -> bool:
    if race_info is None:
        return False
    if race_info.surface and run.surface and run.surface != race_info.surface:
        return True
    if race_info.distance is not None and run.distance is not None and abs(run.distance - race_info.distance) > 400:
        return True
    return False


def score_run_performance(run: RecentRun, race_info: RaceInfo | None = None) -> float:
    if run.finish is None:
        return 0.0

    score = finish_percentile_score(run) * 0.65 + safe_finish_bucket(run) * 0.35

    if run.field_size is not None and run.field_size >= 14 and run.finish <= 3:
        score += 0.5
    if run.popularity is not None:
        if run.popularity >= 8 and run.finish <= 3:
            score += 0.8
        if run.finish < run.popularity and run.popularity - run.finish >= 5:
            score += 0.4
        if run.popularity <= 2 and run.field_size is not None and run.finish > run.field_size / 2:
            score -= 1.0
    if run.odds is not None and run.odds >= 20 and run.finish <= 3:
        score += 0.4

    margin = parse_margin_value(run.margin)
    if margin is not None:
        if run.finish == 1:
            score += 0.4
        elif margin <= 0.3:
            score += 0.3
        elif margin >= 1.5 and run.finish > 5:
            score -= 0.5

    if run_condition_mismatch(run, race_info) and score < 5.0:
        score += 0.9
    if (
        race_info is not None
        and race_info.distance is not None
        and run.distance == race_info.distance
        and score >= 7.0
    ):
        score += 0.3

    return round(clamp_float(score), 2)


def weighted_score_from_float_runs(scored_runs: list[tuple[float, float]]) -> int:
    if not scored_runs:
        return 0
    weighted_scores = [score * weight for score, weight in scored_runs]
    weight_total = sum(weight for _, weight in scored_runs)
    if not weighted_scores or weight_total == 0:
        return 0
    return clamp_score(sum(weighted_scores) / weight_total)


def score_recent_form(recent_runs: list[RecentRun], race_info: RaceInfo | None = None) -> int:
    scored_runs = [
        (score_run_performance(run, race_info), get_weight(index))
        for index, run in enumerate(recent_runs[:RECENT_QUALITY_RUN_LIMIT])
        if run.finish is not None
    ]
    return weighted_score_from_float_runs(scored_runs)


def score_ability(career_runs: list[RecentRun], race_info: RaceInfo) -> int:
    performances = [score_run_performance(run, race_info) for run in career_runs if run.finish is not None]
    if not performances:
        return 0
    top_count = int(len(performances) * 0.3 + 0.999)
    if len(performances) >= 2:
        top_count = max(2, top_count)
    top_count = min(len(performances), max(1, min(5, top_count)))
    top_scores = sorted(performances, reverse=True)[:top_count]
    top_average = sum(top_scores) / len(top_scores)
    career_average = sum(performances) / len(performances)
    return clamp_score(top_average * 0.75 + career_average * 0.25)


def score_trend(career_runs: list[RecentRun], race_info: RaceInfo) -> int:
    performances = [score_run_performance(run, race_info) for run in career_runs[:RECENT_FORM_RUN_LIMIT] if run.finish is not None]
    if len(performances) < 4:
        return 5
    recent_average = sum(performances[:2]) / 2
    previous = performances[2:5]
    previous_average = sum(previous) / len(previous)
    return clamp_score(5.0 + (recent_average - previous_average) * 0.8)


def weighted_score_from_runs(scored_runs: list[tuple[int, float]]) -> int:
    if not scored_runs:
        return 0
    weighted_scores = [score * weight for score, weight in scored_runs]
    weight_total = sum(weight for _, weight in scored_runs)
    if weight_total == 0:
        return 0
    return clamp_score(sum(weighted_scores) / weight_total)


def _distance_fit_score(run: RecentRun, race_distance: int) -> int | None:
    if run.distance is None:
        return None
    diff = abs(run.distance - race_distance)
    if diff == 0:
        distance_base = 10
    elif diff <= 200:
        distance_base = 7
    elif diff <= 400:
        distance_base = 4
    else:
        return None
    return clamp_score(distance_base * 0.65 + safe_finish_bucket(run) * 0.35)


def _surface_fit_score(run: RecentRun) -> int:
    return min(8, clamp_score(safe_finish_bucket(run) * 0.8))


def _condition_fit_score(run: RecentRun) -> int:
    if run.finish is not None and run.finish <= 3:
        return 10
    if run.finish is not None and run.finish <= 5:
        return 7
    return 4


def score_distance_fit(recent_runs: list[RecentRun], race_distance: int | None) -> int:
    if race_distance is None:
        return 0
    has_distance_data = False
    scored_runs: list[tuple[int, float]] = []
    for index, run in enumerate(recent_runs):
        if run.distance is None:
            continue
        has_distance_data = True
        score = _distance_fit_score(run, race_distance)
        if score is None:
            continue
        scored_runs.append((score, get_weight(index)))
    if not scored_runs:
        return 1 if has_distance_data else 0
    return weighted_score_from_runs(scored_runs)


def score_course_fit(
    recent_runs: list[RecentRun],
    course: str | None,
    surface: str | None = None,
) -> int:
    if not course and not surface:
        return 0
    has_condition_data = False
    same_course_scores: list[tuple[int, float]] = []
    same_surface_scores: list[tuple[int, float]] = []
    for index, run in enumerate(recent_runs):
        if run.course or run.surface:
            has_condition_data = True
        if course and run.course == course:
            same_course_scores.append((_condition_fit_score(run), get_weight(index)))
            continue
        if surface and run.surface == surface:
            same_surface_scores.append((_surface_fit_score(run), get_weight(index)))
    course_score = weighted_score_from_runs(same_course_scores)
    surface_score = weighted_score_from_runs(same_surface_scores)
    if course_score and surface_score:
        return max(course_score, clamp_score(course_score * 0.7 + surface_score * 0.3))
    if course_score:
        return course_score
    if surface_score:
        return surface_score
    return 1 if has_condition_data else 0


def score_track_condition_fit(recent_runs: list[RecentRun], track_condition: str | None) -> int:
    if not track_condition:
        return 0
    has_track_condition_data = False
    scored_runs: list[tuple[int, float]] = []
    for index, run in enumerate(recent_runs):
        if not run.track_condition:
            continue
        has_track_condition_data = True
        if run.track_condition == track_condition:
            scored_runs.append((_condition_fit_score(run), get_weight(index)))
    if not scored_runs:
        return 1 if has_track_condition_data else 0
    return weighted_score_from_runs(scored_runs)


def score_jockey_fit(recent_runs: list[RecentRun], jockey: str | None) -> int:
    if not jockey:
        return 0
    has_jockey_data = False
    scored_runs: list[tuple[int, float]] = []
    for index, run in enumerate(recent_runs):
        if not run.jockey:
            continue
        has_jockey_data = True
        same_jockey = jockey_matches(jockey, run.jockey)
        if same_jockey:
            scored_runs.append((_condition_fit_score(run), get_weight(index)))
    if not scored_runs:
        return 1 if has_jockey_data else 0
    return weighted_score_from_runs(scored_runs)


def score_condition_fit_score(
    distance_fit: int,
    course_fit: int,
    track_condition_fit: int,
    *,
    distance_weight: float,
    course_weight: float,
) -> int:
    track_weight = 0.75
    total_weight = distance_weight + course_weight + track_weight
    if total_weight <= 0:
        return 0
    return clamp_score(
        (
            distance_fit * distance_weight
            + course_fit * course_weight
            + track_condition_fit * track_weight
        )
        / total_weight
    )


def score_race_level_component(race_level_analysis: object | None = None) -> int:
    if race_level_analysis is None:
        return 5
    adjustment_hint = float(getattr(race_level_analysis, "adjustment_hint", 0.0) or 0.0)
    positive_flags = set(getattr(race_level_analysis, "positive_flags", []) or [])
    risk_flags = set(getattr(race_level_analysis, "risk_flags", []) or [])
    score = 5.0 + clamp_float(adjustment_hint, -1.5, 1.5) * 2.0
    if len(positive_flags) >= 2:
        score += 0.4
    if len(risk_flags) >= 2:
        score -= 0.4
    return clamp_score(score)


def score_pace_jockey_component(
    jockey_fit: int,
    pace_analysis: object | None = None,
    race_pace_projection: object | None = None,
) -> int:
    jockey_component = jockey_fit if jockey_fit > 0 else 5
    score = jockey_component * 0.45 + 5.0 * 0.55
    if pace_analysis is None:
        return clamp_score(score)

    positive_flags = set(getattr(pace_analysis, "positive_flags", []) or [])
    risk_flags = set(getattr(pace_analysis, "risk_flags", []) or [])
    projected_pace = getattr(race_pace_projection, "projected_pace", None)
    running_style = getattr(pace_analysis, "running_style", "")
    early_score = float(getattr(pace_analysis, "early_position_score", 0.0) or 0.0)
    late_score = float(getattr(pace_analysis, "late_position_score", 0.0) or 0.0)

    if "PACE_FIT" in positive_flags:
        score += 1.0
    if "POSITION_STABLE" in positive_flags:
        score += 0.5
    if "STALKER_ADVANTAGE" in positive_flags:
        score += 0.5
    if "FRONT_RUNNING_ADVANTAGE" in positive_flags and projected_pace in {"slow", "average", None}:
        score += 0.4
    if "CLOSING_SPEED" in positive_flags and projected_pace in {"fast", "average", None}:
        score += 0.4
    if early_score >= 7.0 and projected_pace in {"slow", "average"}:
        score += 0.3
    if late_score >= 7.0 and projected_pace in {"fast", "average"}:
        score += 0.3

    if "PACE_MISMATCH" in risk_flags:
        score -= 1.0
    if "POSITION_UNSTABLE" in risk_flags:
        score -= 0.5
    if running_style == "追込" and projected_pace == "slow":
        score -= 0.6
    if running_style == "逃げ" and projected_pace == "fast":
        score -= 0.6
    if "PACE_DATA_INCOMPLETE" in risk_flags and jockey_fit <= 1:
        score -= 0.4

    return clamp_score(score)


def score_odds_value(horse: HorseEntry, recent_form: int) -> int:
    if horse.odds is not None:
        if horse.odds >= 20 and recent_form >= 5:
            return 9
        if horse.odds >= 10 and recent_form >= 6:
            return 8
        if horse.odds < 3 and recent_form >= 6:
            return 4
        if horse.odds < 3 and recent_form < 5:
            return 2
        return 5

    has_hole_run = any(
        (run.popularity is not None and run.popularity >= 8 and run.finish is not None and run.finish <= 3)
        or (run.odds is not None and run.odds >= 20 and run.finish is not None and run.finish <= 3)
        for run in horse.recent_runs
    )
    if has_hole_run:
        return 8
    if recent_form >= 6:
        return 5
    return 2


def calculate_base_total_score(
    scores: ScoreBreakdown,
    *,
    distance_weight: float,
    course_weight: float,
    use_market_score_in_ranking: bool = False,
    market_signal_weight: float = 0.0,
) -> float:
    market_component = (
        scores.odds_value * market_signal_weight
        if use_market_score_in_ranking and market_signal_weight > 0
        else 0.0
    )
    has_component_scores = (
        scores.ability_score > 0
        or scores.recent_quality_score > 0
        or scores.condition_fit_score > 0
        or scores.pace_jockey_score > 0
        or scores.trend_score != 5
        or scores.race_level_score != 5
    )
    if has_component_scores:
        return round(
            scores.ability_score * ABILITY_SCORE_WEIGHT
            + scores.recent_quality_score * RECENT_QUALITY_SCORE_WEIGHT
            + scores.trend_score * TREND_SCORE_WEIGHT
            + scores.condition_fit_score * CONDITION_FIT_SCORE_WEIGHT
            + scores.race_level_score * RACE_LEVEL_SCORE_WEIGHT
            + scores.pace_jockey_score * PACE_JOCKEY_SCORE_WEIGHT
            + market_component
            + scores.risk,
            1,
        )
    return round(
        scores.recent_form * 1.5
        + scores.distance_fit * distance_weight
        + scores.course_fit * course_weight
        + scores.track_condition_fit * 0.8
        + scores.jockey_fit * 0.8
        + market_component
        + scores.risk,
        1,
    )


def calculate_unraced_base_total_score(
    scores: ScoreBreakdown,
    *,
    use_market_score_in_ranking: bool = False,
    market_signal_weight: float = 0.0,
) -> float:
    market_component = (
        scores.odds_value * market_signal_weight
        if use_market_score_in_ranking and market_signal_weight > 0
        else 0.0
    )
    return round(UNRACED_BASE_SCORE + scores.risk + market_component, 1)


def score_risk(
    recent_runs: list[RecentRun],
    recent_form: int,
    distance_fit: int,
) -> int:
    if not recent_runs:
        return UNRACED_RISK

    valid_finishes = [run for run in recent_runs[:5] if run.finish is not None and run.field_size is not None]
    if len(valid_finishes) >= 2 and all(run.finish > run.field_size / 2 for run in valid_finishes[:2]):
        return -7
    if distance_fit <= 3 and recent_form <= 3:
        return -6
    null_finish_count = sum(1 for run in recent_runs[:5] if run.finish is None)
    if null_finish_count >= 2:
        return -5
    if recent_form >= 7 and len(valid_finishes) >= 3:
        return -1
    return -3


def get_lesson_weight_adjustment(lessons: list[LessonItem]) -> tuple[float, float, list[str]]:
    distance_weight = 1.2
    course_weight = 1.0
    adjustments: list[str] = []
    lesson_text = " ".join(lesson.lesson for lesson in lessons)
    if any(token in lesson_text for token in ("同距離", "距離")):
        distance_weight = 1.35
        adjustments.append("距離")
    if any(token in lesson_text for token in ("同コース", "コース")):
        course_weight = 1.15
        adjustments.append("コース")
    return distance_weight, course_weight, adjustments


def summarize_run_facts(horse: HorseEntry, race_info: RaceInfo) -> tuple[int, int, int, int, int]:
    recent_runs = scorable_recent_runs(horse.recent_runs)
    career_runs = scorable_career_runs(horse.recent_runs)
    wins = sum(1 for run in recent_runs if run.finish == 1)
    top3 = sum(1 for run in recent_runs if run.finish is not None and run.finish <= 3)
    same_distance_top3 = sum(
        1
        for run in career_runs
        if race_info.distance is not None
        and run.distance == race_info.distance
        and run.finish is not None
        and run.finish <= 3
    )
    same_course_top3 = sum(
        1
        for run in career_runs
        if race_info.course
        and run.course == race_info.course
        and run.finish is not None
        and run.finish <= 3
    )
    same_surface_top3 = sum(
        1
        for run in career_runs
        if race_info.surface
        and run.surface == race_info.surface
        and run.finish is not None
        and run.finish <= 3
    )
    return wins, top3, same_distance_top3, same_course_top3, same_surface_top3


def build_distance_phrase(race_info: RaceInfo, same_distance_runs: list[RecentRun], same_distance_top3: int) -> str:
    distance = race_info.distance if race_info.distance is not None else "unknown"
    if not same_distance_runs:
        return f"同距離{distance}mの経験は少なく、距離適性は未知。"
    if same_distance_top3 > 0:
        return f"同距離{distance}mで好走歴{same_distance_top3}回があり、距離適性を評価。"
    return f"同距離{distance}mでの好走歴はまだなく、距離適性は慎重に判断。"


def build_course_phrase(
    race_info: RaceInfo,
    same_course_runs: list[RecentRun],
    same_course_top3: int,
    same_surface_runs: list[RecentRun],
    same_surface_top3: int,
) -> str:
    course = race_info.course if race_info.course else "unknown"
    surface = race_info.surface if race_info.surface else "同サーフェス"
    if not same_course_runs:
        if same_surface_top3 > 0:
            return f"{course}の経験は少ないが、{surface}では好走歴{same_surface_top3}回があり、適性を評価。"
        if same_surface_runs:
            return f"{course}の経験は少ないが、{surface}での出走歴はあり、適性は慎重に判断。"
        return f"{course}の経験は少なく、コース適性は未知。"
    if same_course_top3 > 0:
        return f"{course}で3着以内{same_course_top3}回があり、コース適性を評価。"
    if same_surface_top3 > 0:
        return f"{course}では出走歴があるが、好走歴はまだない。{surface}では好走歴{same_surface_top3}回。"
    return f"{course}では出走歴があるが、好走歴はまだなくコース適性は慎重に判断。"


def build_jockey_phrase(horse: HorseEntry, same_jockey_top3: int, same_jockey_runs: list[RecentRun]) -> str:
    if not same_jockey_runs:
        return "同騎手での実績はなく、継続効果は未知。"
    if same_jockey_top3 > 0:
        return f"同騎手での好走{same_jockey_top3}回を評価。"
    return "同騎手での騎乗経験はあるが、好走実績はまだない。"


def build_performance_notes(scores: ScoreBreakdown) -> list[str]:
    notes: list[str] = []
    if scores.ability_score >= 8:
        notes.append("通算上位パフォーマンスは高水準。")
    elif scores.ability_score and scores.ability_score <= 3:
        notes.append("通算能力面は強調しづらい。")

    if scores.recent_quality_score >= 8:
        notes.append("直近内容も強い。")
    elif scores.recent_quality_score and scores.recent_quality_score <= 3:
        notes.append("直近内容は割引。")

    if scores.trend_score >= 7:
        notes.append("近走は上昇気配。")
    elif scores.trend_score <= 3:
        notes.append("近走は下降気味。")

    if scores.condition_fit_score >= 8:
        notes.append("今回条件への適性は高い。")
    elif scores.condition_fit_score and scores.condition_fit_score <= 3:
        notes.append("今回条件への適性は慎重に見る。")

    if scores.race_level_score >= 8:
        notes.append("相手関係・レースレベル面も評価。")
    elif scores.race_level_score <= 3:
        notes.append("相手関係・レースレベル面は割引。")

    if scores.pace_jockey_score >= 8:
        notes.append("展開と騎手面の後押しがある。")
    elif scores.pace_jockey_score and scores.pace_jockey_score <= 3:
        notes.append("展開と騎手面は割引。")
    return notes


def build_reason(
    horse: HorseEntry,
    race_info: RaceInfo,
    scores: ScoreBreakdown,
    lesson_adjustments: list[str] | None = None,
    deep_overall_comment: str | None = None,
    pedigree_adjustment_value: float = 0.0,
    race_level_adjustment_value: float = 0.0,
    pace_adjustment_value: float = 0.0,
) -> str:
    recent_runs = scorable_recent_runs(horse.recent_runs)
    career_runs = scorable_career_runs(horse.recent_runs)
    if not career_runs:
        if is_debut_race(race_info):
            reason = "新馬戦で実戦履歴がなく、血統面を主な判断材料として扱う。"
        else:
            reason = "実戦履歴が不足しており、血統面と適性面を重めに扱う。"
        if pedigree_adjustment_value > 0:
            reason += f" 血統面から+{pedigree_adjustment_value:.1f}補正。"
        elif pedigree_adjustment_value < 0:
            reason += f" 血統面で{pedigree_adjustment_value:.1f}補正。"
        if race_level_adjustment_value > 0:
            reason += f" 相手関係面から+{race_level_adjustment_value:.1f}補正。"
        elif race_level_adjustment_value < 0:
            reason += f" 相手関係面で{race_level_adjustment_value:.1f}補正。"
        if pace_adjustment_value > 0:
            reason += f" 展開面から+{pace_adjustment_value:.1f}補正。"
        elif pace_adjustment_value < 0:
            reason += f" 展開面で{pace_adjustment_value:.1f}補正。"
        return reason

    wins, top3, same_distance_top3, same_course_top3, same_surface_top3 = summarize_run_facts(horse, race_info)
    best_finish = min((run.finish for run in recent_runs if run.finish is not None), default=None)
    same_distance_runs = [
        run for run in career_runs
        if race_info.distance is not None and run.distance == race_info.distance
    ]
    same_course_runs = [
        run for run in career_runs
        if race_info.course and run.course == race_info.course
    ]
    same_surface_runs = [
        run for run in career_runs
        if race_info.surface and run.surface == race_info.surface
    ]
    distance_phrase = build_distance_phrase(race_info, same_distance_runs, same_distance_top3)
    course_phrase = build_course_phrase(
        race_info,
        same_course_runs,
        same_course_top3,
        same_surface_runs,
        same_surface_top3,
    )
    same_jockey_top3 = sum(
        1
        for run in career_runs
        if jockey_matches(horse.jockey, run.jockey) and run.finish is not None and run.finish <= 3
    )
    same_jockey_runs = [
        run for run in career_runs if jockey_matches(horse.jockey, run.jockey)
    ]
    jockey_phrase = build_jockey_phrase(horse, same_jockey_top3, same_jockey_runs)
    if scores.recent_form >= 7 and scores.distance_fit >= 7:
        reason = (
            f"近5走で1着{wins}回・3着以内{top3}回。"
            f"{distance_phrase}"
        )
    elif scores.course_fit >= 7 or same_course_top3 > 0:
        reason = (
            f"近5走最高着順は{best_finish if best_finish is not None else 'unknown'}着。"
            f"{course_phrase} {jockey_phrase}"
        )
    else:
        reason = (
            f"近5走最高着順は{best_finish if best_finish is not None else 'unknown'}着。"
            f"{distance_phrase} {course_phrase} 近走の安定感は限定的。"
        )
    if lesson_adjustments:
        reason += f" 過去lessonに基づき{ '/'.join(lesson_adjustments) }評価を微調整。"
    performance_notes = build_performance_notes(scores)
    if performance_notes:
        reason += " " + " ".join(performance_notes)
    if deep_overall_comment:
        reason += f" 深掘りでは{deep_overall_comment}"
    if pedigree_adjustment_value > 0:
        reason += f" 血統面から+{pedigree_adjustment_value:.1f}補正。"
    elif pedigree_adjustment_value < 0:
        reason += f" 血統面で{pedigree_adjustment_value:.1f}補正。"
    if race_level_adjustment_value > 0:
        reason += f" 相手関係面から+{race_level_adjustment_value:.1f}補正。"
    elif race_level_adjustment_value < 0:
        reason += f" 相手関係面で{race_level_adjustment_value:.1f}補正。"
    if pace_adjustment_value > 0:
        reason += f" 展開面から+{pace_adjustment_value:.1f}補正。"
    elif pace_adjustment_value < 0:
        reason += f" 展開面で{pace_adjustment_value:.1f}補正。"
    return reason


def score_horse_by_recent_runs(
    horse: HorseEntry,
    race_info: RaceInfo,
    lessons: list[LessonItem] | None = None,
    *,
    use_market_score_in_ranking: bool = False,
    market_signal_weight: float = 0.0,
    pedigree_analysis: PedigreeAnalysis | None = None,
    race_level_analysis: object | None = None,
    pace_analysis: object | None = None,
    race_pace_projection: object | None = None,
) -> HorseScore:
    recent_runs = scorable_recent_runs(horse.recent_runs)
    career_runs = scorable_career_runs(horse.recent_runs)
    career_run_count = len(career_runs)
    lesson_list = lessons or []
    distance_weight, course_weight, lesson_adjustments = get_lesson_weight_adjustment(lesson_list)
    ability_score = score_ability(career_runs, race_info)
    recent_quality_score = score_recent_form(recent_runs, race_info)
    recent_form = recent_quality_score
    trend_score = score_trend(career_runs, race_info)
    distance_fit = score_distance_fit(career_runs, race_info.distance)
    course_fit = score_course_fit(career_runs, race_info.course, race_info.surface)
    track_condition_fit = score_track_condition_fit(career_runs, race_info.track_condition)
    distance_fit, course_fit, track_condition_fit = apply_low_sample_pedigree_priors(
        recent_run_count=career_run_count,
        distance_fit=distance_fit,
        course_fit=course_fit,
        track_condition_fit=track_condition_fit,
        pedigree_analysis=pedigree_analysis,
        race_info=race_info,
    )
    jockey_fit = score_jockey_fit(career_runs, horse.jockey)
    condition_fit_score = score_condition_fit_score(
        distance_fit,
        course_fit,
        track_condition_fit,
        distance_weight=distance_weight,
        course_weight=course_weight,
    )
    race_level_score = score_race_level_component(race_level_analysis)
    pace_jockey_score = score_pace_jockey_component(jockey_fit, pace_analysis, race_pace_projection)
    odds_value = score_odds_value(horse, recent_quality_score)
    risk = score_risk(recent_runs, recent_quality_score, distance_fit)

    scores = ScoreBreakdown(
        recent_form=recent_form,
        distance_fit=distance_fit,
        course_fit=course_fit,
        track_condition_fit=track_condition_fit,
        jockey_fit=jockey_fit,
        odds_value=odds_value,
        risk=risk,
        ability_score=ability_score,
        recent_quality_score=recent_quality_score,
        trend_score=trend_score,
        condition_fit_score=condition_fit_score,
        race_level_score=race_level_score,
        pace_jockey_score=pace_jockey_score,
    )
    if not career_runs:
        total_score = calculate_unraced_base_total_score(
            scores,
            use_market_score_in_ranking=use_market_score_in_ranking,
            market_signal_weight=market_signal_weight,
        )
    else:
        total_score = calculate_base_total_score(
            scores,
            distance_weight=distance_weight,
            course_weight=course_weight,
            use_market_score_in_ranking=use_market_score_in_ranking,
            market_signal_weight=market_signal_weight,
        )
    return HorseScore(
        horse_no=horse.horse_no,
        horse_name=horse.horse_name,
        scores=scores,
        total_score=total_score,
        reason=build_reason(horse, race_info, scores, lesson_adjustments=lesson_adjustments),
    )


def build_marks(horse_scores: list[HorseScore]) -> dict[str, int]:
    marks = {label: 0 for label in MARK_LABELS}
    for index, score in enumerate(horse_scores[: len(MARK_LABELS)]):
        marks[MARK_LABELS[index]] = score.horse_no
    return marks


def collect_risks(race_data: RaceData, used_lessons: list[LessonItem]) -> list[str]:
    risks: list[str] = []
    if any(is_low_sample_horse(horse) for horse in race_data.horses):
        risks.append("有効な実戦履歴5走未満の馬は血統・距離・馬場適性の先験評価を補強。")
    if any(any(run.course is None for run in horse.recent_runs) for horse in race_data.horses):
        risks.append("recent_runsにcourse欠損が含まれている。")
    if any(any(run.race_id is not None and len(run.race_id) < 8 for run in horse.recent_runs) for horse in race_data.horses):
        risks.append("recent_runs内に不完全なrace_idが含まれている。")
    if used_lessons:
        risks.append("過去lessonは軽量な重み補正としてのみ使用しており、モデル学習ではない。")
    risks.append("heuristic scoringを使用しており、正式なML modelではない。")
    return risks


def build_summary(
    race_data: RaceData,
    horse_scores: list[HorseScore],
    used_lessons: list[LessonItem],
    strategy: StrategyDecision,
    top_deep_comment: str | None = None,
    top_pedigree_comment: str | None = None,
    top_pedigree_flags: list[str] | None = None,
    top_race_level_comment: str | None = None,
    race_pace_projection_text: str | None = None,
    pedigree_rank_changed: bool = False,
    analysis_rank_changed: bool = False,
) -> str:
    if not horse_scores:
        return "有効な近走データが不足しており、summaryはunknownに近い状態。"
    top = horse_scores[0]
    top_horse_runs = next((horse.recent_runs for horse in race_data.horses if horse.horse_no == top.horse_no), [])
    career_run_count = len(scorable_career_runs(top_horse_runs))
    lesson_note = ""
    _, _, lesson_adjustments = get_lesson_weight_adjustment(used_lessons)
    if used_lessons:
        lesson_note = f" 過去lesson参考: {used_lessons[0].lesson}"
    if lesson_adjustments:
        lesson_note += f" lesson補正={ '/'.join(lesson_adjustments) }。"
    pedigree_note = ""
    if top_pedigree_flags and any(flag in top_pedigree_flags for flag in ("PEDIGREE_STAMINA_FIT", "PEDIGREE_DISTANCE_FIT")):
        pedigree_note = " 血統面でも距離適性を補強。"
    race_level_note = f" 相手関係では{top_race_level_comment}" if top_race_level_comment else ""
    pace_note = f" 展開想定: {race_pace_projection_text}" if race_pace_projection_text else ""
    rank_notes: list[str] = []
    if pedigree_rank_changed:
        rank_notes.append("血統補正により一部順位が変動。")
    if analysis_rank_changed:
        rank_notes.append("展開・相手関係補正により一部順位が変動。")
    rank_note = f" {' '.join(rank_notes)}" if rank_notes else ""
    return (
        f"本命は{top.horse_name}。通算能力・近走内容・条件適性を分離して評価した。"
        f"recent_runs使用数={career_run_count}、lessons使用数={len(used_lessons)}。"
        f"{lesson_note}"
        f"{f' 深掘り所見: {top_deep_comment}' if top_deep_comment else ''}"
        f"{f' 血統所見: {top_pedigree_comment}' if top_pedigree_comment else ''}"
        f"{pedigree_note}"
        f"{race_level_note}"
        f"{pace_note}"
        f"{rank_note}"
        f" 買い判断={strategy.bet_decision}、confidence={strategy.confidence}。"
        f" {strategy.reason}"
    )


def build_prediction_from_recent_runs(race_data: RaceData, lessons: list[LessonItem]) -> Prediction:
    return build_prediction_from_recent_runs_with_scoring_config(
        race_data,
        lessons,
        scoring_profile="accuracy_default",
        scoring_config=ScoringConfigSnapshot(
            scoring_mode="candidate_default",
            pedigree_weight=DEFAULT_SCORING_WEIGHTS["pedigree_weight"],
            race_level_weight=DEFAULT_SCORING_WEIGHTS["race_level_weight"],
            pace_weight=DEFAULT_SCORING_WEIGHTS["pace_weight"],
            conditional_weight_profile=DEFAULT_CONDITIONAL_WEIGHT_PROFILE,
            use_market_score_in_ranking=False,
            market_signal_weight=0.0,
        ),
        borderline_recovery_enabled=True,
    )


def build_prediction_from_recent_runs_with_scoring_config(
    race_data: RaceData,
    lessons: list[LessonItem],
    scoring_profile: str,
    scoring_config: ScoringConfigSnapshot,
    borderline_recovery_enabled: bool = False,
) -> Prediction:
    scoring_config_snapshot = ScoringConfigSnapshot.model_validate(scoring_config.model_dump())
    effective_weights = effective_scoring_weights(
        scoring_config_snapshot,
        surface=race_data.race_info.surface,
        field_size=len(race_data.horses),
    )
    scoring_config_snapshot = scoring_config_snapshot.model_copy(update=effective_weights)
    deep_analyses = analyze_race_deeply(race_data.horses, race_data.race_info)
    pedigree_analyses = build_pedigree_analyses_for_race(race_data.horses, race_data.race_info)
    race_level_analyses = analyze_race_level_for_race(race_data.horses, race_data.race_info)
    pace_analyses, race_pace_projection = analyze_pace_for_race(race_data.horses, race_data.race_info)
    deep_analysis_by_horse = {analysis.horse_no: analysis for analysis in deep_analyses}
    pedigree_analysis_by_horse = {analysis.horse_no: analysis for analysis in pedigree_analyses}
    race_level_analysis_by_horse = {analysis.horse_no: analysis for analysis in race_level_analyses}
    pace_analysis_by_horse = {analysis.horse_no: analysis for analysis in pace_analyses}
    base_horse_scores = [
        score_horse_by_recent_runs(
            horse,
            race_data.race_info,
            lessons=lessons,
            use_market_score_in_ranking=scoring_config_snapshot.use_market_score_in_ranking,
            market_signal_weight=scoring_config_snapshot.market_signal_weight,
            pedigree_analysis=pedigree_analysis_by_horse.get(horse.horse_no),
            race_level_analysis=race_level_analysis_by_horse.get(horse.horse_no),
            pace_analysis=pace_analysis_by_horse.get(horse.horse_no),
            race_pace_projection=race_pace_projection,
        )
        for horse in race_data.horses
    ]
    base_ranking = [score.horse_no for score in sorted(base_horse_scores, key=lambda item: (-item.total_score, item.horse_no))]
    lesson_adjustments = get_lesson_weight_adjustment(lessons)[2]
    pedigree_adjusted_scores: list[HorseScore] = []
    enriched_horse_scores: list[HorseScore] = []
    for horse_score in base_horse_scores:
        horse_entry = next(horse for horse in race_data.horses if horse.horse_no == horse_score.horse_no)
        deep_analysis = deep_analysis_by_horse.get(horse_score.horse_no)
        pedigree_analysis = pedigree_analysis_by_horse.get(horse_score.horse_no)
        race_level_analysis = race_level_analysis_by_horse.get(horse_score.horse_no)
        pace_analysis = pace_analysis_by_horse.get(horse_score.horse_no)
        pedigree_adjustment = (
            calculate_pedigree_adjustment(pedigree_analysis, race_data.race_info)
            if pedigree_analysis is not None
            else PedigreeAdjustment(reason="血統補正なし")
        )
        pedigree_weight = effective_pedigree_weight_for_horse(
            scoring_config_snapshot.pedigree_weight,
            horse_entry,
            race_data.race_info,
        )
        race_level_adjustment = calculate_race_level_adjustment(race_level_analysis)
        pace_adjustment = calculate_pace_adjustment(pace_analysis, race_pace_projection)
        score_breakdown = build_score_breakdown(
            base_total_score=horse_score.total_score,
            pedigree_adjustment_raw=pedigree_adjustment.pedigree_adjustment,
            pedigree_weight=pedigree_weight,
            race_level_adjustment_raw=race_level_adjustment.adjustment,
            race_level_weight=scoring_config_snapshot.race_level_weight,
            pace_adjustment_raw=pace_adjustment.adjustment,
            pace_weight=scoring_config_snapshot.pace_weight,
        )
        updated_reason = build_reason(
            horse_entry,
            race_data.race_info,
            horse_score.scores,
            lesson_adjustments=lesson_adjustments,
            deep_overall_comment=deep_analysis.overall_comment if deep_analysis else None,
            pedigree_adjustment_value=score_breakdown.pedigree_adjustment_weighted,
            race_level_adjustment_value=score_breakdown.race_level_adjustment_weighted,
            pace_adjustment_value=score_breakdown.pace_adjustment_weighted,
        )
        pedigree_adjusted_scores.append(
            horse_score.model_copy(
                update={
                    "total_score": round(
                        horse_score.total_score + score_breakdown.pedigree_adjustment_weighted,
                        1,
                    )
                }
            )
        )
        enriched_horse_scores.append(
            horse_score.model_copy(
                update={
                    "base_total_score": horse_score.total_score,
                    "pedigree_adjustment": pedigree_adjustment,
                    "race_level_adjustment": race_level_adjustment,
                    "pace_adjustment": pace_adjustment,
                    "score_breakdown": score_breakdown,
                    "odds": horse_entry.odds,
                    "popularity": horse_entry.popularity,
                    "total_score": score_breakdown.total_score,
                    "reason": updated_reason,
                }
            )
        )
    horse_scores = enriched_horse_scores
    pedigree_ranking = [
        score.horse_no for score in sorted(pedigree_adjusted_scores, key=lambda item: (-item.total_score, item.horse_no))
    ]
    horse_scores.sort(key=lambda item: (-item.total_score, item.horse_no))
    adjusted_ranking = [score.horse_no for score in horse_scores]
    pedigree_rank_changed = base_ranking[:5] != pedigree_ranking[:5]
    analysis_rank_changed = pedigree_ranking[:5] != adjusted_ranking[:5]
    borderline_recovery_config = BorderlineRecoveryConfigSnapshot(enabled=borderline_recovery_enabled)
    borderline_recovery_result_payload = apply_top5_borderline_recovery(
        horse_scores,
        deep_analyses,
        pedigree_analyses,
        race_level_analyses,
        pace_analyses,
        race_data.race_info,
        scoring_config_snapshot,
        enabled=borderline_recovery_enabled,
    )
    if borderline_recovery_result_payload["recovery_applied"]:
        horse_scores = borderline_recovery_result_payload["adjusted_horse_scores"]
        horse_scores.sort(key=lambda item: (-item.total_score, item.horse_no))
    used_lessons = lessons
    marks = build_marks(horse_scores)
    risks = collect_risks(race_data, used_lessons)
    if pedigree_rank_changed:
        risks.append("血統補正により一部順位が変動。")
    if analysis_rank_changed:
        risks.append("展開・相手関係補正により一部順位が変動。")
    strategy = evaluate_bet_strategy(
        race_data=race_data,
        horse_scores=horse_scores,
        marks=marks,
        used_lessons=used_lessons,
        risks=risks,
        pace_analyses=pace_analyses,
        scoring_config=scoring_config_snapshot,
    )
    bets = build_bets_from_strategy(strategy, horse_scores)
    top_deep_comment = None
    top_pedigree_comment = None
    top_pedigree_flags: list[str] | None = None
    top_race_level_comment = None
    race_pace_projection_text = None
    if horse_scores:
        top_deep = deep_analysis_by_horse.get(horse_scores[0].horse_no)
        if top_deep is not None:
            top_deep_comment = top_deep.overall_comment
        top_pedigree = pedigree_analysis_by_horse.get(horse_scores[0].horse_no)
        if top_pedigree is not None:
            top_pedigree_comment = top_pedigree.overall_comment
            top_pedigree_flags = top_pedigree.positive_flags
        top_race_level = race_level_analysis_by_horse.get(horse_scores[0].horse_no)
        if top_race_level is not None:
            top_race_level_comment = top_race_level.overall_comment
        if race_pace_projection is not None:
            if (
                race_data.race_info.course == "京都"
                and race_data.race_info.surface == "芝"
                and race_data.race_info.distance == 2400
                and race_pace_projection.projected_pace in {"average", "slow"}
            ):
                race_pace_projection_text = "平均〜やや落ち着く。先行〜差しの持続力を評価。"
            else:
                styles = "〜".join(race_pace_projection.favorable_styles[:2]) if race_pace_projection.favorable_styles else "不明"
                pace_label_map = {
                    "slow": "スローペース",
                    "average": "平均ペース",
                    "fast": "ハイペース",
                    "unknown": "不明",
                }
                race_pace_projection_text = f"{pace_label_map.get(race_pace_projection.projected_pace, race_pace_projection.projected_pace)}。{styles}勢にやや向く。"
    return Prediction(
        race_id=race_data.race_info.race_id,
        race_info=race_data.race_info,
        scoring_profile=scoring_profile,
        scoring_mode=scoring_config_snapshot.scoring_mode,
        borderline_recovery_enabled=borderline_recovery_enabled,
        scoring_config=scoring_config_snapshot,
        market_signal_config=MarketSignalConfigSnapshot(
            use_market_score_in_ranking=scoring_config_snapshot.use_market_score_in_ranking,
            market_signal_weight=scoring_config_snapshot.market_signal_weight,
        ),
        borderline_recovery_config=borderline_recovery_config,
        marks=marks,
        horse_scores=horse_scores,
        bets=bets,
        summary=build_summary(
            race_data,
            horse_scores,
            used_lessons,
            strategy,
            top_deep_comment=top_deep_comment,
            top_pedigree_comment=top_pedigree_comment,
            top_pedigree_flags=top_pedigree_flags,
            top_race_level_comment=top_race_level_comment,
            race_pace_projection_text=race_pace_projection_text,
            pedigree_rank_changed=pedigree_rank_changed,
            analysis_rank_changed=analysis_rank_changed,
        ),
        risks=risks,
        used_lessons=used_lessons,
        deep_analyses=deep_analyses,
        pedigree_analyses=pedigree_analyses,
        race_level_analyses=race_level_analyses,
        pace_analyses=pace_analyses,
        race_pace_projection=race_pace_projection,
        strategy=strategy,
        borderline_recovery_result=BorderlineRecoveryResult.model_validate(
            {
                "recovery_applied": borderline_recovery_result_payload["recovery_applied"],
                "recovery_cases": borderline_recovery_result_payload["recovery_cases"],
            }
        ),
    )


def has_recent_runs_data(race_data: RaceData) -> bool:
    if not race_data.horses:
        return False
    return (
        is_debut_race(race_data.race_info)
        or any(has_valid_recent_result(horse.recent_runs) for horse in race_data.horses)
        or any(bool(horse.horse_id) for horse in race_data.horses)
    )

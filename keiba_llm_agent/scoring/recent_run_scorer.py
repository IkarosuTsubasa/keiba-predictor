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
MARK_LABELS = ("◎", "○", "▲", "△", "☆")


def clamp_score(value: float, minimum: int = 0, maximum: int = 10) -> int:
    return max(minimum, min(maximum, round(value)))


def weighted_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def get_weight(index: int) -> float:
    return WEIGHTS[index] if index < len(WEIGHTS) else WEIGHTS[-1]


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


def score_recent_form(recent_runs: list[RecentRun]) -> int:
    weighted_scores = [safe_finish_bucket(run) * get_weight(index) for index, run in enumerate(recent_runs[:5])]
    weight_total = sum(get_weight(index) for index, run in enumerate(recent_runs[:5]) if run.finish is not None)
    if not weighted_scores or weight_total == 0:
        return 0
    return clamp_score(sum(weighted_scores) / weight_total)


def score_distance_fit(recent_runs: list[RecentRun], race_distance: int | None) -> int:
    if race_distance is None:
        return 0
    weighted_scores: list[float] = []
    weights: list[float] = []
    for index, run in enumerate(recent_runs[:5]):
        if run.distance is None:
            continue
        diff = abs(run.distance - race_distance)
        if diff == 0:
            score = 10
        elif diff <= 200:
            score = 7
        elif diff <= 400:
            score = 4
        else:
            score = 1
        weight = get_weight(index)
        weighted_scores.append(score * weight)
        weights.append(weight)
    if not weights:
        return 0
    return clamp_score(sum(weighted_scores) / sum(weights))


def score_course_fit(recent_runs: list[RecentRun], course: str | None) -> int:
    if not course:
        return 0
    weighted_scores: list[float] = []
    weights: list[float] = []
    for index, run in enumerate(recent_runs[:5]):
        if not run.course:
            continue
        if run.course == course and run.finish is not None and run.finish <= 3:
            score = 10
        elif run.course == course and run.finish is not None and run.finish <= 5:
            score = 7
        elif run.course == course:
            score = 4
        else:
            score = 1
        weight = get_weight(index)
        weighted_scores.append(score * weight)
        weights.append(weight)
    if not weights:
        return 0
    return clamp_score(sum(weighted_scores) / sum(weights))


def score_track_condition_fit(recent_runs: list[RecentRun], track_condition: str | None) -> int:
    if not track_condition:
        return 0
    weighted_scores: list[float] = []
    weights: list[float] = []
    for index, run in enumerate(recent_runs[:5]):
        if not run.track_condition:
            continue
        if run.track_condition == track_condition and run.finish is not None and run.finish <= 3:
            score = 10
        elif run.track_condition == track_condition and run.finish is not None and run.finish <= 5:
            score = 7
        elif run.track_condition == track_condition:
            score = 4
        else:
            score = 1
        weight = get_weight(index)
        weighted_scores.append(score * weight)
        weights.append(weight)
    if not weights:
        return 0
    return clamp_score(sum(weighted_scores) / sum(weights))


def score_jockey_fit(recent_runs: list[RecentRun], jockey: str | None) -> int:
    if not jockey:
        return 0
    weighted_scores: list[float] = []
    weights: list[float] = []
    for index, run in enumerate(recent_runs[:5]):
        if not run.jockey:
            continue
        same_jockey = jockey_matches(jockey, run.jockey)
        if same_jockey and run.finish is not None and run.finish <= 3:
            score = 10
        elif same_jockey and run.finish is not None and run.finish <= 5:
            score = 7
        elif same_jockey:
            score = 4
        else:
            score = 1
        weight = get_weight(index)
        weighted_scores.append(score * weight)
        weights.append(weight)
    if not weights:
        return 0
    return clamp_score(sum(weighted_scores) / sum(weights))


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
        for run in horse.recent_runs[:5]
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


def score_risk(
    recent_runs: list[RecentRun],
    recent_form: int,
    distance_fit: int,
) -> int:
    if not recent_runs:
        return -10

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


def summarize_run_facts(horse: HorseEntry, race_info: RaceInfo) -> tuple[int, int, int, int]:
    recent_runs = horse.recent_runs[:5]
    wins = sum(1 for run in recent_runs if run.finish == 1)
    top3 = sum(1 for run in recent_runs if run.finish is not None and run.finish <= 3)
    same_distance_top3 = sum(
        1
        for run in recent_runs
        if race_info.distance is not None
        and run.distance == race_info.distance
        and run.finish is not None
        and run.finish <= 3
    )
    same_course_top3 = sum(
        1
        for run in recent_runs
        if race_info.course
        and run.course == race_info.course
        and run.finish is not None
        and run.finish <= 3
    )
    return wins, top3, same_distance_top3, same_course_top3


def build_distance_phrase(race_info: RaceInfo, same_distance_runs: list[RecentRun], same_distance_top3: int) -> str:
    distance = race_info.distance if race_info.distance is not None else "unknown"
    if not same_distance_runs:
        return f"同距離{distance}mの経験は少なく、距離適性は未知。"
    if same_distance_top3 > 0:
        return f"同距離{distance}mで好走歴{same_distance_top3}回があり、距離適性を評価。"
    return f"同距離{distance}mでの好走歴はまだなく、距離適性は慎重に判断。"


def build_course_phrase(race_info: RaceInfo, same_course_runs: list[RecentRun], same_course_top3: int) -> str:
    course = race_info.course if race_info.course else "unknown"
    if not same_course_runs:
        return f"{course}の経験は少なく、コース適性は未知。"
    if same_course_top3 > 0:
        return f"{course}で3着以内{same_course_top3}回があり、コース適性を評価。"
    return f"{course}では出走歴があるが、好走歴はまだなくコース適性は慎重に判断。"


def build_jockey_phrase(horse: HorseEntry, same_jockey_top3: int, same_jockey_runs: list[RecentRun]) -> str:
    if not same_jockey_runs:
        return "同騎手での実績はなく、継続効果は未知。"
    if same_jockey_top3 > 0:
        return f"同騎手での好走{same_jockey_top3}回を評価。"
    return "同騎手での騎乗経験はあるが、好走実績はまだない。"


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
    recent_runs = horse.recent_runs[:5]
    if not recent_runs:
        reason = "近走データが不足しており、距離・コース適性の判断材料が限られる。"
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

    wins, top3, same_distance_top3, same_course_top3 = summarize_run_facts(horse, race_info)
    best_finish = min((run.finish for run in recent_runs if run.finish is not None), default=None)
    same_distance_runs = [
        run for run in recent_runs
        if race_info.distance is not None and run.distance == race_info.distance
    ]
    same_course_runs = [
        run for run in recent_runs
        if race_info.course and run.course == race_info.course
    ]
    distance_phrase = build_distance_phrase(race_info, same_distance_runs, same_distance_top3)
    course_phrase = build_course_phrase(race_info, same_course_runs, same_course_top3)
    same_jockey_top3 = sum(
        1
        for run in recent_runs
        if jockey_matches(horse.jockey, run.jockey) and run.finish is not None and run.finish <= 3
    )
    same_jockey_runs = [
        run for run in recent_runs if jockey_matches(horse.jockey, run.jockey)
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
) -> HorseScore:
    recent_runs = horse.recent_runs[:5]
    lesson_list = lessons or []
    recent_form = score_recent_form(recent_runs)
    distance_fit = score_distance_fit(recent_runs, race_info.distance)
    course_fit = score_course_fit(recent_runs, race_info.course)
    track_condition_fit = score_track_condition_fit(recent_runs, race_info.track_condition)
    jockey_fit = score_jockey_fit(recent_runs, horse.jockey)
    odds_value = score_odds_value(horse, recent_form)
    risk = score_risk(recent_runs, recent_form, distance_fit)
    distance_weight, course_weight, lesson_adjustments = get_lesson_weight_adjustment(lesson_list)

    scores = ScoreBreakdown(
        recent_form=recent_form,
        distance_fit=distance_fit,
        course_fit=course_fit,
        track_condition_fit=track_condition_fit,
        jockey_fit=jockey_fit,
        odds_value=odds_value,
        risk=risk,
    )
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
    if any(horse.odds is None or horse.popularity is None for horse in race_data.horses):
        risks.append("一部の馬でoddsまたは人気が欠損している。")
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
    recent_run_count = len(next((horse.recent_runs for horse in race_data.horses if horse.horse_no == top.horse_no), []))
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
        f"本命は{top.horse_name}。近走安定度と距離・コース適性を重視した。"
        f"recent_runs使用数={recent_run_count}、lessons使用数={len(used_lessons)}。"
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
        race_level_adjustment = calculate_race_level_adjustment(race_level_analysis)
        pace_adjustment = calculate_pace_adjustment(pace_analysis, race_pace_projection)
        score_breakdown = build_score_breakdown(
            base_total_score=horse_score.total_score,
            pedigree_adjustment_raw=pedigree_adjustment.pedigree_adjustment,
            pedigree_weight=scoring_config_snapshot.pedigree_weight,
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
    return any(horse.recent_runs for horse in race_data.horses)

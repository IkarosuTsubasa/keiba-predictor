from __future__ import annotations

from collections import Counter

from keiba_llm_agent.schemas.deep_analysis import HorseDeepAnalysis
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun


def jockey_matches(current_jockey: str | None, past_jockey: str | None) -> bool:
    if not current_jockey or not past_jockey:
        return False
    current = current_jockey.strip()
    past = past_jockey.strip()
    return current.startswith(past) or past.startswith(current) or current in past or past in current


def _valid_runs(horse: HorseEntry) -> list[RecentRun]:
    return horse.recent_runs[:5]


def _top3_count(runs: list[RecentRun]) -> int:
    return sum(1 for run in runs if run.finish is not None and run.finish <= 3)


def _wins_count(runs: list[RecentRun]) -> int:
    return sum(1 for run in runs if run.finish == 1)


def _double_digit_count(runs: list[RecentRun]) -> int:
    return sum(1 for run in runs if run.finish is not None and run.finish >= 10)


def _best_finish(runs: list[RecentRun]) -> int | None:
    finishes = [run.finish for run in runs if run.finish is not None]
    return min(finishes) if finishes else None


def _is_recent_declining(runs: list[RecentRun]) -> bool:
    if len(runs) < 2 or runs[0].finish is None or runs[1].finish is None:
        return False
    return runs[0].finish > runs[1].finish


def _is_recent_improving(runs: list[RecentRun]) -> bool:
    if len(runs) < 2 or runs[0].finish is None or runs[1].finish is None:
        return False
    return runs[0].finish < runs[1].finish


def _is_stable(runs: list[RecentRun]) -> bool:
    valid = [run for run in runs if run.finish is not None]
    if len(valid) < 3:
        return False
    return sum(1 for run in valid if run.finish <= 5) >= 3 and _double_digit_count(valid) <= 1


def _distance_relation(current_distance: int | None, recent_distance: int | None) -> str:
    if current_distance is None or recent_distance is None:
        return "距離比較はunknown。"
    if current_distance == recent_distance:
        return "今回も同距離。"
    if current_distance > recent_distance:
        return "今回は距離延長。"
    return "今回は距離短縮。"


def _main_distance_band(runs: list[RecentRun]) -> str | None:
    distances = [run.distance for run in runs if run.distance is not None]
    if not distances:
        return None
    counts = Counter(distances)
    most_common_distance, _ = counts.most_common(1)[0]
    if most_common_distance <= 1400:
        return "短距離中心"
    if most_common_distance <= 1800:
        return "マイル中心"
    if most_common_distance <= 2200:
        return "中距離中心"
    return "長めの距離中心"


def _classify_odds(odds: float | None) -> str:
    if odds is None:
        return "オッズunknown"
    if odds < 3:
        return "人気馬"
    if odds < 10:
        return "上位人気"
    if odds < 30:
        return "中穴"
    return "大穴"


def _append_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def analyze_horse_deeply(horse: HorseEntry, race_info: RaceInfo) -> HorseDeepAnalysis:
    runs = _valid_runs(horse)
    positive_flags: list[str] = []
    risk_flags: list[str] = []

    if not runs:
        _append_flag(risk_flags, "DATA_INCOMPLETE")
        return HorseDeepAnalysis(
            horse_no=horse.horse_no,
            horse_name=horse.horse_name,
            positive_flags=positive_flags,
            risk_flags=risk_flags,
            recent_form_summary="近走データがなく、状態比較はunknown。",
            distance_analysis="距離適性を判断する材料が不足している。",
            course_analysis="同コース実績はunknown。",
            track_condition_analysis="馬場適性はunknown。",
            jockey_analysis="騎手継続評価はunknown。",
            odds_analysis="現在オッズが不足しており、市場評価はunknown。",
            overall_comment="近走データ不足のため評価は慎重。強い根拠が乏しく、相手候補までの扱いが妥当。",
        )

    wins = _wins_count(runs)
    top3 = _top3_count(runs)
    double_digits = _double_digit_count(runs)
    best_finish = _best_finish(runs)
    declining = _is_recent_declining(runs)
    improving = _is_recent_improving(runs)
    stable = _is_stable(runs)

    if top3 >= 3:
        _append_flag(positive_flags, "RECENT_FORM_STRONG")
    if stable:
        _append_flag(positive_flags, "RECENT_FORM_STABLE")
        _append_flag(positive_flags, "STABLE_PERFORMER")
    if double_digits >= 2:
        _append_flag(risk_flags, "RECENT_FORM_WEAK")
    if declining:
        _append_flag(risk_flags, "RECENT_FORM_DECLINING")

    recent_form_summary = (
        f"近5走で1着{wins}回・3着以内{top3}回。"
        f"二桁着順は{double_digits}回で、最高着順は{best_finish if best_finish is not None else 'unknown'}着。"
    )
    if stable:
        recent_form_summary += "大崩れは少なく、近走安定度は高い。"
    elif declining:
        recent_form_summary += "直近2走はやや下降気味。"
    elif improving:
        recent_form_summary += "直近2走は上向き傾向。"
    else:
        recent_form_summary += "近走の波はやや大きい。"

    same_distance_runs = [
        run for run in runs
        if race_info.distance is not None and run.distance == race_info.distance
    ]
    same_distance_top3 = _top3_count(same_distance_runs)
    if same_distance_top3 >= 1:
        _append_flag(positive_flags, "DISTANCE_FIT")
    if not same_distance_runs:
        _append_flag(risk_flags, "DISTANCE_UNKNOWN")
    main_distance_band = _main_distance_band(runs)
    distance_relation = _distance_relation(race_info.distance, runs[0].distance)
    distance_analysis = (
        f"同距離{race_info.distance if race_info.distance is not None else 'unknown'}mでは"
        f"{len(same_distance_runs)}走して{same_distance_top3}回好走。"
        f"{distance_relation}"
    )
    if main_distance_band:
        distance_analysis += f" 近走は{main_distance_band}で、今回との整合性を確認したい。"
    if not same_distance_runs:
        distance_analysis += " 同距離実績が乏しく未知のリスクが残る。"

    same_course_runs = [run for run in runs if race_info.course and run.course == race_info.course]
    same_course_top3 = _top3_count(same_course_runs)
    if same_course_top3 >= 1:
        _append_flag(positive_flags, "COURSE_FIT")
    if not same_course_runs:
        _append_flag(risk_flags, "COURSE_UNKNOWN")
    course_analysis = (
        f"{race_info.course if race_info.course else 'unknown'}では{len(same_course_runs)}走して"
        f"{same_course_top3}回3着以内。"
    )
    course_analysis += "コース適性は高い。" if same_course_top3 >= 1 else "同コース経験は少なく、適性判断は保留。"

    same_track_runs = [
        run for run in runs
        if race_info.track_condition and run.track_condition == race_info.track_condition
    ]
    same_track_top3 = _top3_count(same_track_runs)
    if same_track_top3 >= 2:
        _append_flag(positive_flags, "TRACK_CONDITION_FIT")
    if not same_track_runs:
        _append_flag(risk_flags, "TRACK_CONDITION_UNKNOWN")
    track_counts = Counter(run.track_condition for run in runs if run.track_condition)
    dominant_track = track_counts.most_common(1)[0][0] if track_counts else None
    track_condition_analysis = (
        f"同馬場{race_info.track_condition if race_info.track_condition else 'unknown'}では"
        f"{len(same_track_runs)}走して{same_track_top3}回3着以内。"
    )
    if dominant_track:
        track_condition_analysis += f" 近走は{dominant_track}での出走が多い。"
    if not same_track_runs:
        track_condition_analysis += " 今回の馬場での裏付けは弱い。"

    same_jockey_runs = [run for run in runs if jockey_matches(horse.jockey, run.jockey)]
    same_jockey_top3 = _top3_count(same_jockey_runs)
    latest_same_jockey = jockey_matches(horse.jockey, runs[0].jockey)
    if same_jockey_top3 >= 2:
        _append_flag(positive_flags, "JOCKEY_CONTINUITY")
    if not latest_same_jockey:
        _append_flag(risk_flags, "JOCKEY_CHANGE")
    jockey_analysis = (
        f"同騎手では{len(same_jockey_runs)}走して{same_jockey_top3}回3着以内。"
    )
    jockey_analysis += "騎手継続はプラス。" if latest_same_jockey else "騎手替わりでリズム変化に注意。"

    odds_class = _classify_odds(horse.odds)
    if horse.odds is not None and horse.odds >= 10 and top3 >= 3:
        _append_flag(positive_flags, "VALUE_CANDIDATE")
    if horse.odds is not None and horse.odds < 3 and len(risk_flags) >= 2:
        _append_flag(risk_flags, "OVERBET_RISK")
    odds_analysis = (
        f"現在オッズは{horse.odds:.1f}" if horse.odds is not None else "現在オッズはunknown"
    )
    if horse.popularity is not None:
        odds_analysis += f"、人気は{horse.popularity}番手。"
    else:
        odds_analysis += "。"
    odds_analysis += f" 市場評価は{odds_class}。"
    if "VALUE_CANDIDATE" in positive_flags:
        odds_analysis += " 実績に対して妙味がある。"
    if "OVERBET_RISK" in risk_flags:
        odds_analysis += " 人気先行で過信は禁物。"

    overall_parts: list[str] = []
    if {"RECENT_FORM_STRONG", "RECENT_FORM_STABLE"} & set(positive_flags):
        overall_parts.append("近走の安定感は評価材料。")
    if {"DISTANCE_FIT", "COURSE_FIT"} & set(positive_flags):
        overall_parts.append("条件適性の裏付けもある。")
    if "JOCKEY_CONTINUITY" in positive_flags:
        overall_parts.append("騎手継続の強みも見逃せない。")
    if "VALUE_CANDIDATE" in positive_flags:
        overall_parts.append("人気とのバランスでは妙味もある。")
    if {"DISTANCE_UNKNOWN", "COURSE_UNKNOWN", "TRACK_CONDITION_UNKNOWN"} & set(risk_flags):
        overall_parts.append("一方で条件面に未知の部分は残る。")
    if {"RECENT_FORM_WEAK", "RECENT_FORM_DECLINING"} & set(risk_flags):
        overall_parts.append("近走内容には警戒も必要。")
    if "OVERBET_RISK" in risk_flags:
        overall_parts.append("人気ほど絶対視はしにくい。")
    overall_comment = " ".join(overall_parts[:4]) or "材料とリスクが混在しており、相手候補としての評価が妥当。"

    return HorseDeepAnalysis(
        horse_no=horse.horse_no,
        horse_name=horse.horse_name,
        positive_flags=positive_flags,
        risk_flags=risk_flags,
        recent_form_summary=recent_form_summary,
        distance_analysis=distance_analysis,
        course_analysis=course_analysis,
        track_condition_analysis=track_condition_analysis,
        jockey_analysis=jockey_analysis,
        odds_analysis=odds_analysis,
        overall_comment=overall_comment,
    )


def analyze_race_deeply(horses: list[HorseEntry], race_info: RaceInfo) -> list[HorseDeepAnalysis]:
    return [analyze_horse_deeply(horse, race_info) for horse in horses]

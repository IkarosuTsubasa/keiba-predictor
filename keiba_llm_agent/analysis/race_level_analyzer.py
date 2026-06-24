from __future__ import annotations

from collections import defaultdict

from keiba_llm_agent.analysis.race_class import infer_run_race_class_level, infer_target_race_class_level
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis


POSITIVE_HINTS = {
    "HEAD_TO_HEAD_POSITIVE": 0.5,
    "LARGE_FIELD_GOOD_RUN": 0.3,
    "GRADE1_SMALL_MARGIN": 0.7,
    "GRADED_SMALL_MARGIN": 0.6,
    "OPEN_CLASS_COMPETITIVE": 0.4,
    "CLASS_DROP_PLUS": 0.4,
}

RISK_HINTS = {
    "HEAD_TO_HEAD_NEGATIVE": 0.5,
    "WEAK_RECENT_LEVEL": 0.5,
    "CLASS_RISE_RISK": 0.4,
    "LOW_CLASS_WIN_ONLY": 0.3,
}


def _append_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _valid_race_id(value: str | None) -> bool:
    return isinstance(value, str) and len(value) >= 8


def _top3_runs(runs: list[RecentRun]) -> int:
    return sum(1 for run in runs if run.finish is not None and run.finish <= 3)


def _double_digit_runs(runs: list[RecentRun]) -> int:
    return sum(1 for run in runs if run.finish is not None and run.finish >= 10)


def _parse_margin_value(value: str | None) -> float | None:
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


def _build_run_index(all_horses: list[HorseEntry]) -> dict[str, list[tuple[HorseEntry, RecentRun]]]:
    index: dict[str, list[tuple[HorseEntry, RecentRun]]] = defaultdict(list)
    for horse in all_horses:
        for run in horse.recent_runs[:5]:
            if _valid_race_id(run.race_id):
                index[run.race_id].append((horse, run))
    return index


def _calculate_adjustment_hint(positive_flags: list[str], risk_flags: list[str]) -> float:
    bonus = sum(POSITIVE_HINTS.get(flag, 0.0) for flag in positive_flags)
    penalty = sum(RISK_HINTS.get(flag, 0.0) for flag in risk_flags)
    return max(-1.5, min(1.5, round(bonus - penalty, 1)))


def analyze_race_level_for_horse(
    horse: HorseEntry,
    all_horses: list[HorseEntry],
    race_info: RaceInfo,
) -> RaceLevelAnalysis:
    positive_flags: list[str] = []
    risk_flags: list[str] = []
    run_index = _build_run_index(all_horses)
    wins_vs_field = 0
    losses_vs_field = 0
    shared_races: list[str] = []

    for run in horse.recent_runs[:5]:
        if not _valid_race_id(run.race_id) or run.finish is None:
            continue
        opponents = [
            (other_horse, other_run)
            for other_horse, other_run in run_index.get(run.race_id, [])
            if other_horse.horse_no != horse.horse_no and other_run.finish is not None
        ]
        if not opponents:
            continue
        shared_races.append(run.race_id)
        for _, opponent_run in opponents:
            if run.finish < opponent_run.finish:
                wins_vs_field += 1
            elif run.finish > opponent_run.finish:
                losses_vs_field += 1

    if wins_vs_field > 0 and wins_vs_field >= losses_vs_field:
        _append_flag(positive_flags, "HEAD_TO_HEAD_POSITIVE")
    if losses_vs_field > wins_vs_field and losses_vs_field > 0:
        _append_flag(risk_flags, "HEAD_TO_HEAD_NEGATIVE")

    head_to_head_summary = "本場につながる再戦関係は目立たない。"
    if shared_races:
        head_to_head_summary = "近走で本レース出走馬と再戦関係あり。"
        if wins_vs_field > losses_vs_field:
            head_to_head_summary += " 同組比較では優位。"
        elif losses_vs_field > wins_vs_field:
            head_to_head_summary += " 同組比較では劣勢。"
        else:
            head_to_head_summary += " 同組比較は五分。"

    large_field_good_runs = 0
    grade1_small_margins = 0
    graded_small_margins = 0
    open_class_competitive_runs = 0
    class_drop_plus_runs = 0
    lower_class_wins = 0
    lower_class_top3 = 0
    target_level = infer_target_race_class_level(race_info)

    for run in horse.recent_runs[:5]:
        if run.finish is None:
            continue
        if run.field_size is not None and run.field_size >= 14 and run.finish <= 3:
            large_field_good_runs += 1
            _append_flag(positive_flags, "LARGE_FIELD_GOOD_RUN")
        run_level = infer_run_race_class_level(run)
        margin = _parse_margin_value(run.margin)
        if run_level is not None:
            if run_level >= 9.0 and margin is not None and margin <= 1.2 and run.finish <= 12:
                grade1_small_margins += 1
                _append_flag(positive_flags, "GRADE1_SMALL_MARGIN")
            if run_level >= 7.0 and margin is not None and margin <= 0.8 and run.finish <= 8:
                graded_small_margins += 1
                _append_flag(positive_flags, "GRADED_SMALL_MARGIN")
            if run_level >= 5.5 and run.finish <= 3:
                open_class_competitive_runs += 1
                _append_flag(positive_flags, "OPEN_CLASS_COMPETITIVE")
            if target_level is not None:
                class_delta = run_level - target_level
                if class_delta >= 1.0 and (run.finish <= 5 or (margin is not None and margin <= 1.0)):
                    class_drop_plus_runs += 1
                    _append_flag(positive_flags, "CLASS_DROP_PLUS")
                if class_delta <= -1.5 and run.finish == 1:
                    lower_class_wins += 1
                if class_delta <= -1.5 and run.finish <= 3:
                    lower_class_top3 += 1

    if _double_digit_runs(horse.recent_runs[:5]) >= 2:
        _append_flag(risk_flags, "WEAK_RECENT_LEVEL")
    if lower_class_wins and not (grade1_small_margins or graded_small_margins or open_class_competitive_runs):
        _append_flag(risk_flags, "LOW_CLASS_WIN_ONLY")
    if lower_class_top3 >= 2 and not (grade1_small_margins or graded_small_margins):
        _append_flag(risk_flags, "CLASS_RISE_RISK")

    race_level_parts: list[str] = []
    if large_field_good_runs:
        race_level_parts.append(f"多頭数戦での好走が{large_field_good_runs}回。")
    if grade1_small_margins:
        race_level_parts.append(f"G1級で小差の内容が{grade1_small_margins}回。")
    if graded_small_margins:
        race_level_parts.append(f"重賞級で着差の小さい内容が{graded_small_margins}回。")
    if open_class_competitive_runs:
        race_level_parts.append(f"オープン以上での好走が{open_class_competitive_runs}回。")
    if class_drop_plus_runs:
        race_level_parts.append(f"今回より高い相手関係で通用した内容が{class_drop_plus_runs}回。")
    if lower_class_wins:
        race_level_parts.append(f"下級条件の勝利が{lower_class_wins}回あり、相手強化は慎重に評価。")
    if not race_level_parts:
        race_level_parts.append("近走レベル評価は中立で、強い補強材料は少ない。")
    race_level_summary = " ".join(race_level_parts)

    opponent_context_summary = "相手関係の強い裏付けは少ない。"
    if shared_races:
        opponent_context_summary = "近走で本場メンバーとの交差あり。"
        if wins_vs_field > losses_vs_field:
            opponent_context_summary += " 先着実績があり、相手比較では優位。"
        elif losses_vs_field > wins_vs_field:
            opponent_context_summary += " 後着実績が目立ち、相手比較では劣勢。"
        else:
            opponent_context_summary += " 明確な優劣はついていない。"

    overall_parts: list[str] = []
    if "HEAD_TO_HEAD_POSITIVE" in positive_flags:
        overall_parts.append("相手関係では前走比較に強みがある。")
    if any(
        flag in positive_flags
        for flag in ("LARGE_FIELD_GOOD_RUN", "GRADE1_SMALL_MARGIN", "GRADED_SMALL_MARGIN", "OPEN_CLASS_COMPETITIVE")
    ):
        overall_parts.append("近走内容の含金量は一定評価できる。")
    if "CLASS_DROP_PLUS" in positive_flags:
        overall_parts.append("今回より強い相手関係でも内容を作れている。")
    if "HEAD_TO_HEAD_NEGATIVE" in risk_flags:
        overall_parts.append("再戦相手比較では見劣る面がある。")
    if any(flag in risk_flags for flag in ("WEAK_RECENT_LEVEL", "CLASS_RISE_RISK", "LOW_CLASS_WIN_ONLY")):
        overall_parts.append("近走内容には割引が必要。")
    if not overall_parts:
        overall_parts.append("相手関係・レースレベル面では大きな加減点はない。")

    return RaceLevelAnalysis(
        horse_no=horse.horse_no,
        horse_name=horse.horse_name,
        positive_flags=positive_flags,
        risk_flags=risk_flags,
        head_to_head_summary=head_to_head_summary,
        race_level_summary=race_level_summary,
        opponent_context_summary=opponent_context_summary,
        overall_comment=" ".join(overall_parts[:3]),
        adjustment_hint=_calculate_adjustment_hint(positive_flags, risk_flags),
    )


def analyze_race_level_for_race(
    horses: list[HorseEntry],
    race_info: RaceInfo,
) -> list[RaceLevelAnalysis]:
    return [analyze_race_level_for_horse(horse, horses, race_info) for horse in horses]

from __future__ import annotations

from collections import defaultdict

from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis


POSITIVE_HINTS = {
    "HEAD_TO_HEAD_POSITIVE": 0.5,
    "LARGE_FIELD_GOOD_RUN": 0.4,
    "UNDERVALUED_GOOD_RUN": 0.4,
    "VALUE_WIN": 0.4,
}

RISK_HINTS = {
    "HEAD_TO_HEAD_NEGATIVE": 0.5,
    "POPULAR_DISAPPOINTMENT": 0.3,
    "WEAK_RECENT_LEVEL": 0.5,
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
    undervalued_good_runs = 0
    popular_disappointments = 0
    expected_wins = 0
    value_wins = 0

    for run in horse.recent_runs[:5]:
        if run.finish is None:
            continue
        if run.field_size is not None and run.field_size >= 14 and run.finish <= 3:
            large_field_good_runs += 1
            _append_flag(positive_flags, "LARGE_FIELD_GOOD_RUN")
        if run.popularity is not None and run.popularity >= 8 and run.finish <= 3:
            undervalued_good_runs += 1
            _append_flag(positive_flags, "UNDERVALUED_GOOD_RUN")
        if (
            run.popularity is not None
            and run.popularity <= 2
            and run.field_size is not None
            and run.finish > run.field_size / 2
        ):
            popular_disappointments += 1
            _append_flag(risk_flags, "POPULAR_DISAPPOINTMENT")
        if run.finish == 1 and run.popularity is not None and run.popularity <= 2:
            expected_wins += 1
            _append_flag(positive_flags, "EXPECTED_WIN")
        if run.finish == 1 and run.popularity is not None and run.popularity >= 5:
            value_wins += 1
            _append_flag(positive_flags, "VALUE_WIN")

    if _double_digit_runs(horse.recent_runs[:5]) >= 2:
        _append_flag(risk_flags, "WEAK_RECENT_LEVEL")

    race_level_parts: list[str] = []
    if large_field_good_runs:
        race_level_parts.append(f"多頭数戦での好走が{large_field_good_runs}回。")
    if undervalued_good_runs:
        race_level_parts.append(f"人気薄での好走が{undervalued_good_runs}回あり、内容に妙味。")
    if value_wins:
        race_level_parts.append(f"人気以上に走った勝利が{value_wins}回。")
    if expected_wins:
        race_level_parts.append(f"支持に応えた勝利が{expected_wins}回。")
    if popular_disappointments:
        race_level_parts.append(f"人気を裏切った敗戦が{popular_disappointments}回。")
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
    if "LARGE_FIELD_GOOD_RUN" in positive_flags or "UNDERVALUED_GOOD_RUN" in positive_flags:
        overall_parts.append("近走内容の含金量は一定評価できる。")
    if "HEAD_TO_HEAD_NEGATIVE" in risk_flags:
        overall_parts.append("再戦相手比較では見劣る面がある。")
    if "POPULAR_DISAPPOINTMENT" in risk_flags or "WEAK_RECENT_LEVEL" in risk_flags:
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


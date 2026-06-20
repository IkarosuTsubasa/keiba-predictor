from __future__ import annotations

import re
from collections import Counter

from keiba_llm_agent.fetchers.netkeiba_horse_fetcher import fetch_horse_html
from keiba_llm_agent.parsers.netkeiba_horse_parser import parse_horse_recent_runs
from keiba_llm_agent.schemas.pedigree import (
    PedigreeInfo,
    PedigreePerformanceProfile,
)
from keiba_llm_agent.schemas.race_data import RaceInfo, RecentRun


RELATION_WEIGHTS = {
    "sire": 0.55,
    "damsire": 0.30,
}


def _append_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _top3_rate(top3: int, starts: int) -> float:
    return top3 / starts if starts > 0 else 0.0


def _win_rate(wins: int, starts: int) -> float:
    return wins / starts if starts > 0 else 0.0


def _distance_band(distance: int | None) -> str | None:
    if distance is None:
        return None
    if distance <= 1400:
        return "短距離"
    if distance <= 1700:
        return "マイル"
    if distance <= 2100:
        return "中距離"
    return "中長距離"


def _distance_fit(target_band: str | None, source_band: str | None) -> bool:
    if target_band is None or source_band is None:
        return False
    if "〜" in source_band:
        return any(_distance_fit(target_band, part) for part in source_band.split("〜"))
    if target_band == source_band:
        return True
    adjacent = {
        ("短距離", "マイル"),
        ("マイル", "短距離"),
        ("マイル", "中距離"),
        ("中距離", "マイル"),
        ("中距離", "中長距離"),
        ("中長距離", "中距離"),
    }
    return (target_band, source_band) in adjacent


def _distance_risk(target_band: str | None, source_band: str | None) -> bool:
    if target_band is None or source_band is None:
        return False
    if "〜" in source_band:
        return all(_distance_risk(target_band, part) for part in source_band.split("〜"))
    return (target_band, source_band) in {
        ("短距離", "中距離"),
        ("短距離", "中長距離"),
        ("中長距離", "短距離"),
    }


def _birth_year_from_horse_id(horse_id: str | None) -> int | None:
    match = re.match(r"^(19|20)\d{2}", str(horse_id or ""))
    if not match:
        return None
    return int(str(horse_id)[:4])


def _is_fetchable_horse_id(horse_id: str | None) -> bool:
    text = str(horse_id or "").strip()
    return len(text) >= 8 and bool(re.fullmatch(r"[0-9A-Za-z]+", text))


def _run_year(run: RecentRun) -> int | None:
    match = re.match(r"^(\d{4})-", str(run.date or ""))
    return int(match.group(1)) if match else None


def _is_grade_race(run: RecentRun) -> bool:
    name = str(run.race_name or "")
    return bool(re.search(r"\((?:G|Jpn)[I1-3]+\)", name) or re.search(r"G[1-3]|Jpn[1-3]", name))


def _is_bad_track(run: RecentRun) -> bool:
    condition = str(run.track_condition or "").strip()
    return bool(condition and condition != "良")


def _dominant_surface(runs: list[RecentRun]) -> str:
    top3_surfaces = [
        run.surface
        for run in runs
        if run.surface and run.finish is not None and run.finish <= 3
    ]
    surfaces = top3_surfaces or [run.surface for run in runs if run.surface]
    if not surfaces:
        return "unknown"
    counts = Counter(surfaces)
    surface, count = counts.most_common(1)[0]
    if len(counts) >= 2 and count / max(1, sum(counts.values())) < 0.67:
        return "芝ダート"
    return surface


def _dominant_distance_band(runs: list[RecentRun]) -> str:
    top3_bands = [
        _distance_band(run.distance)
        for run in runs
        if run.finish is not None and run.finish <= 3
    ]
    bands = [band for band in (top3_bands or [_distance_band(run.distance) for run in runs]) if band]
    if not bands:
        return "unknown"
    counts = Counter(bands)
    most_common = counts.most_common(2)
    if len(most_common) >= 2 and most_common[1][1] > 0:
        first, second = most_common[0][0], most_common[1][0]
        order = ["短距離", "マイル", "中距離", "中長距離"]
        if abs(order.index(first) - order.index(second)) == 1:
            left, right = sorted((first, second), key=order.index)
            return f"{left}〜{right}"
    return most_common[0][0]


def _track_tendency(runs: list[RecentRun]) -> str:
    bad_track_runs = [run for run in runs if _is_bad_track(run)]
    if not bad_track_runs:
        return "unknown"
    bad_track_top3 = sum(1 for run in bad_track_runs if run.finish is not None and run.finish <= 3)
    if _top3_rate(bad_track_top3, len(bad_track_runs)) >= 0.4:
        return "重馬場"
    return "unknown"


def _pace_tendency(runs: list[RecentRun]) -> str:
    first_positions: list[int] = []
    final_3fs: list[float] = []
    for run in runs:
        if run.corner_positions:
            first_positions.append(run.corner_positions[0])
        if run.final_3f is not None:
            final_3fs.append(run.final_3f)
    parts: list[str] = []
    if first_positions and sum(1 for pos in first_positions if pos <= 4) / len(first_positions) >= 0.5:
        parts.append("先行力")
    if final_3fs and min(final_3fs) <= 35.0:
        parts.append("末脚")
    return "・".join(parts) if parts else "unknown"


def _early_maturity(runs: list[RecentRun], horse_id: str) -> str:
    birth_year = _birth_year_from_horse_id(horse_id)
    if birth_year is None:
        return "unknown"
    early_runs = [
        run
        for run in runs
        if (year := _run_year(run)) is not None and year - birth_year <= 2
    ]
    if not early_runs:
        return "unknown"
    early_top3 = sum(1 for run in early_runs if run.finish is not None and run.finish <= 3)
    if _top3_rate(early_top3, len(early_runs)) >= 0.4:
        return "早期実績あり"
    return "unknown"


def _class_power(runs: list[RecentRun]) -> str:
    grade_runs = [run for run in runs if _is_grade_race(run)]
    if not grade_runs:
        return "unknown"
    grade_wins = sum(1 for run in grade_runs if run.finish == 1)
    grade_top3 = sum(1 for run in grade_runs if run.finish is not None and run.finish <= 3)
    if grade_wins:
        return "重賞勝ち"
    if grade_top3:
        return "重賞好走"
    return "重賞経験"


def build_performance_profile(
    *,
    relation: str,
    horse_id: str,
    horse_name: str | None,
    runs: list[RecentRun],
    race_info: RaceInfo,
) -> PedigreePerformanceProfile:
    valid_runs = [run for run in runs if run.finish is not None]
    starts = len(valid_runs)
    if starts == 0:
        return PedigreePerformanceProfile(
            relation=relation,
            horse_id=horse_id,
            horse_name=horse_name,
            overall_comment="戦績データが不足。",
        )

    wins = sum(1 for run in valid_runs if run.finish == 1)
    top3 = sum(1 for run in valid_runs if run.finish <= 3)
    surface = _dominant_surface(valid_runs)
    distance = _dominant_distance_band(valid_runs)
    track = _track_tendency(valid_runs)
    pace = _pace_tendency(valid_runs)
    class_power = _class_power(valid_runs)
    early_maturity = _early_maturity(valid_runs, horse_id)
    target_band = _distance_band(race_info.distance)
    positive_flags: list[str] = []
    risk_flags: list[str] = []

    score_hint = 0.0
    relation_weight = RELATION_WEIGHTS.get(relation, 0.1)
    top3_rate = _top3_rate(top3, starts)
    win_rate = _win_rate(wins, starts)

    if race_info.surface and surface in {race_info.surface, "芝ダート"}:
        _append_flag(positive_flags, "PEDIGREE_SURFACE_FIT")
        score_hint += 0.5 * relation_weight
    elif race_info.surface and surface != "unknown":
        _append_flag(risk_flags, "PEDIGREE_SURFACE_UNKNOWN")

    if _distance_fit(target_band, distance):
        _append_flag(positive_flags, "PEDIGREE_DISTANCE_FIT")
        score_hint += 0.6 * relation_weight
    elif _distance_risk(target_band, distance):
        _append_flag(risk_flags, "PEDIGREE_DISTANCE_RISK")
        score_hint -= 0.6 * relation_weight

    if track == "重馬場" and race_info.track_condition and race_info.track_condition != "良":
        _append_flag(positive_flags, "PEDIGREE_TRACK_CONDITION_FIT")
        score_hint += 0.3 * relation_weight
    if race_info.surface == "ダート" and surface in {"ダート", "芝ダート"}:
        _append_flag(positive_flags, "PEDIGREE_POWER_FIT")
        score_hint += 0.3 * relation_weight
    if class_power in {"重賞勝ち", "重賞好走"}:
        _append_flag(positive_flags, "PEDIGREE_CLASS_POWER")
        score_hint += 0.4 * relation_weight
    if early_maturity == "早期実績あり":
        _append_flag(positive_flags, "PEDIGREE_EARLY_MATURITY")
        score_hint += 0.3 * relation_weight
    if top3_rate >= 0.5:
        score_hint += 0.3 * relation_weight
    if win_rate >= 0.25:
        score_hint += 0.2 * relation_weight

    score_hint = round(max(-0.8, min(1.2, score_hint)), 2)
    comment = (
        f"{horse_name or horse_id}は{surface}・{distance}寄り。"
        f"通算{starts}戦{wins}勝、3着以内{top3}回。"
    )
    if class_power != "unknown":
        comment += f"{class_power}。"
    if early_maturity != "unknown":
        comment += f"{early_maturity}。"

    return PedigreePerformanceProfile(
        relation=relation,
        horse_id=horse_id,
        horse_name=horse_name,
        starts=starts,
        wins=wins,
        top3=top3,
        surface_tendency=surface,
        distance_tendency=distance,
        track_condition_tendency=track,
        pace_tendency=pace,
        class_power=class_power,
        early_maturity=early_maturity,
        positive_flags=positive_flags,
        risk_flags=risk_flags,
        score_hint=score_hint,
        overall_comment=comment,
    )


def fetch_performance_profile(
    *,
    relation: str,
    horse_id: str | None,
    horse_name: str | None,
    race_info: RaceInfo,
) -> PedigreePerformanceProfile | None:
    if not horse_id:
        return None
    if not _is_fetchable_horse_id(horse_id):
        return None
    try:
        html = fetch_horse_html(horse_id)
        runs = parse_horse_recent_runs(html, limit=None)
    except Exception:
        return None
    return build_performance_profile(
        relation=relation,
        horse_id=horse_id,
        horse_name=horse_name,
        runs=runs,
        race_info=race_info,
    )


def build_performance_profiles_for_pedigree(
    pedigree: PedigreeInfo,
    race_info: RaceInfo,
) -> list[PedigreePerformanceProfile]:
    candidates = [
        ("sire", pedigree.sire_id, pedigree.sire),
        ("damsire", pedigree.damsire_id, pedigree.damsire),
    ]
    profiles: list[PedigreePerformanceProfile] = []
    seen_ids: set[str] = set()
    for relation, horse_id, horse_name in candidates:
        if not horse_id or horse_id in seen_ids:
            continue
        seen_ids.add(horse_id)
        profile = fetch_performance_profile(
            relation=relation,
            horse_id=horse_id,
            horse_name=horse_name,
            race_info=race_info,
        )
        if profile is not None:
            profiles.append(profile)
    return profiles

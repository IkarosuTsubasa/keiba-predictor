from __future__ import annotations

import re
import warnings
from pathlib import Path

import requests

from keiba_llm_agent.fetchers.netkeiba_horse_fetcher import (
    DB_NETKEIBA_ENCODING,
    DEFAULT_HEADERS,
    fetch_horse_html,
    get_horse_cache_path,
)
from keiba_llm_agent.pedigree.pedigree_knowledge import PEDIGREE_KNOWLEDGE
from keiba_llm_agent.pedigree.pedigree_performance import build_performance_profiles_for_pedigree
from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis, PedigreeInfo, PedigreePerformanceProfile
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo
from keiba_llm_agent.pedigree.pedigree_parser import extract_pedigree_url, parse_pedigree_info


BASE_DIR = Path(__file__).resolve().parents[1]
PEDIGREE_HTML_CACHE_DIR = BASE_DIR / "data" / "pedigree_html_cache"


def get_pedigree_cache_path(horse_id: str) -> Path:
    return PEDIGREE_HTML_CACHE_DIR / f"{horse_id}.html"


def get_pedigree_url(horse_id: str) -> str:
    return f"https://db.netkeiba.com/horse/ped/{horse_id}/"


def fetch_pedigree_html(
    horse_id: str,
    force_refresh: bool = False,
    horse_html: str | None = None,
) -> str:
    cache_path = get_pedigree_cache_path(horse_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_refresh:
        return cache_path.read_text(encoding="utf-8")

    url = extract_pedigree_url(horse_html, horse_id) if horse_html else None
    if url is None:
        url = get_pedigree_url(horse_id)

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"failed to fetch pedigree page: horse_id={horse_id}") from exc

    response.encoding = DB_NETKEIBA_ENCODING
    html = response.text
    cache_path.write_text(html, encoding="utf-8")
    return html


def _append_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _merge_surface(sire_surface: str | None, damsire_surface: str | None) -> str:
    values = [value for value in (sire_surface, damsire_surface) if value and value != "unknown"]
    if not values:
        return "unknown"
    merged: list[str] = []
    for value in values:
        if value not in merged:
            merged.append(value)
    return " / ".join(merged)


def _merge_distance(sire_distance: str | None, damsire_distance: str | None) -> str:
    values = [value for value in (sire_distance, damsire_distance) if value and value != "unknown"]
    if not values:
        return "unknown"
    merged: list[str] = []
    for value in values:
        if value not in merged:
            merged.append(value)
    return " / ".join(merged)


def _merge_traits(*trait_groups: list[str]) -> list[str]:
    merged: list[str] = []
    for traits in trait_groups:
        for trait in traits:
            if trait not in merged:
                merged.append(trait)
    return merged


def normalize_pedigree_name(value: str | None) -> str:
    text = " ".join(str(value or "").replace("\xa0", " ").split())
    if not text:
        return ""
    text = re.sub(r"\s*[（(][^)）]+[)）]\s*$", "", text).strip()
    match = re.match(r"^(.+?)\s+[A-Za-z][A-Za-z '.-]*$", text)
    if match and re.search(r"[ぁ-んァ-ン一-龥]", match.group(1)):
        text = match.group(1).strip()
    return text


def _pace_tendency_from_traits(traits: list[str]) -> str:
    pace_traits = [trait for trait in traits if trait in {"瞬発力", "持続力", "スピード", "スタミナ", "パワー", "小回り"}]
    return "・".join(pace_traits) if pace_traits else "unknown"


def _track_tendency_from_traits(traits: list[str]) -> str:
    track_traits = [trait for trait in traits if trait in {"パワー", "重馬場", "洋芝"}]
    return "・".join(track_traits) if track_traits else "unknown"


def _distance_has_fit(distance_tendency: str, traits: list[str], distance: int) -> bool:
    if distance >= 2200:
        return "中長距離" in distance_tendency or "スタミナ" in traits
    if distance <= 1400:
        return "短距離" in distance_tendency or "マイル" in distance_tendency or "スピード" in traits
    if 1600 <= distance <= 2000:
        return any(token in distance_tendency for token in ("マイル〜中距離", "中距離")) or ("マイル" in distance_tendency and distance == 1600)
    return False


def _distance_has_risk(distance_tendency: str, traits: list[str], distance: int) -> bool:
    if distance <= 1400:
        return "中長距離" in distance_tendency or ("スタミナ" in traits and "スピード" not in traits)
    return False


def _unknown_analysis(horse_no: int, horse_name: str, pedigree: PedigreeInfo) -> PedigreeAnalysis:
    return PedigreeAnalysis(
        horse_no=horse_no,
        horse_name=horse_name,
        sire=pedigree.sire,
        dam=pedigree.dam,
        damsire=pedigree.damsire,
        surface_tendency="unknown",
        distance_tendency="unknown",
        track_condition_tendency="unknown",
        pace_tendency="unknown",
        positive_flags=[],
        risk_flags=["PEDIGREE_DATA_INCOMPLETE"] if not pedigree.sire else ["PEDIGREE_SURFACE_UNKNOWN", "PEDIGREE_DISTANCE_UNKNOWN"],
        overall_comment="血統情報または血統知識が不足しており、血統面の強調材料はunknown。",
    )


def _merge_performance_tendency(base: str, profiles: list[PedigreePerformanceProfile], attr: str) -> str:
    values = [base] if base and base != "unknown" else []
    for profile in profiles:
        value = getattr(profile, attr, "unknown")
        if value and value != "unknown" and value not in values:
            values.append(value)
    return " / ".join(values) if values else "unknown"


def _merge_performance_flags(
    positive_flags: list[str],
    risk_flags: list[str],
    profiles: list[PedigreePerformanceProfile],
) -> tuple[list[str], list[str]]:
    for profile in profiles:
        for flag in getattr(profile, "positive_flags", []):
            _append_flag(positive_flags, flag)
        for flag in getattr(profile, "risk_flags", []):
            _append_flag(risk_flags, flag)
    if "PEDIGREE_SURFACE_FIT" in positive_flags and "PEDIGREE_SURFACE_UNKNOWN" in risk_flags:
        risk_flags.remove("PEDIGREE_SURFACE_UNKNOWN")
    if (
        {"PEDIGREE_DISTANCE_FIT", "PEDIGREE_STAMINA_FIT"} & set(positive_flags)
        and "PEDIGREE_DISTANCE_UNKNOWN" in risk_flags
    ):
        risk_flags.remove("PEDIGREE_DISTANCE_UNKNOWN")
    return positive_flags, risk_flags


def _effective_recent_result_count(horse: HorseEntry) -> int:
    return sum(1 for run in horse.recent_runs[:5] if run.finish is not None)


def analyze_pedigree(
    pedigree: PedigreeInfo,
    race_info: RaceInfo,
    horse_no: int = 0,
    horse_name: str | None = None,
    performance_profiles: list[PedigreePerformanceProfile] | None = None,
) -> PedigreeAnalysis:
    final_horse_name = horse_name or pedigree.horse_name or "unknown"
    profiles = list(performance_profiles or [])
    performance_score_hint = round(sum(float(getattr(profile, "score_hint", 0.0)) for profile in profiles), 1)
    if not pedigree.sire and not pedigree.damsire and not profiles:
        return _unknown_analysis(horse_no, final_horse_name, pedigree)

    sire_lookup_name = normalize_pedigree_name(pedigree.sire)
    damsire_lookup_name = normalize_pedigree_name(pedigree.damsire)
    sire_knowledge = PEDIGREE_KNOWLEDGE.get(sire_lookup_name) if sire_lookup_name else None
    damsire_knowledge = PEDIGREE_KNOWLEDGE.get(damsire_lookup_name) if damsire_lookup_name else None

    if sire_knowledge is None and damsire_knowledge is None and not profiles:
        return PedigreeAnalysis(
            horse_no=horse_no,
            horse_name=final_horse_name,
            sire=pedigree.sire,
            dam=pedigree.dam,
            damsire=pedigree.damsire,
            surface_tendency="unknown",
            distance_tendency="unknown",
            track_condition_tendency="unknown",
            pace_tendency="unknown",
            positive_flags=[],
            risk_flags=["PEDIGREE_SURFACE_UNKNOWN", "PEDIGREE_DISTANCE_UNKNOWN"],
            overall_comment="父系・母父ともに血統知識が不足しており、血統面はunknownとして扱う。",
        )

    positive_flags: list[str] = []
    risk_flags: list[str] = []
    sire_surface = sire_knowledge["surface"] if sire_knowledge else "unknown"
    sire_distance = sire_knowledge["distance"] if sire_knowledge else "unknown"
    sire_traits = sire_knowledge["traits"] if sire_knowledge else []
    damsire_surface = damsire_knowledge["surface"] if damsire_knowledge else "unknown"
    damsire_distance = damsire_knowledge["distance"] if damsire_knowledge else "unknown"
    damsire_traits = damsire_knowledge["traits"] if damsire_knowledge else []

    surface_tendency = _merge_surface(sire_surface, damsire_surface)
    distance_tendency = _merge_distance(sire_distance, damsire_distance)
    traits = _merge_traits(sire_traits, damsire_traits)
    track_tendency = _track_tendency_from_traits(traits)
    pace_tendency = _pace_tendency_from_traits(traits)

    if race_info.surface and race_info.surface in surface_tendency:
        _append_flag(positive_flags, "PEDIGREE_SURFACE_FIT")
    elif surface_tendency == "unknown":
        _append_flag(risk_flags, "PEDIGREE_SURFACE_UNKNOWN")
    elif race_info.surface and "芝ダート" not in surface_tendency and race_info.surface not in surface_tendency:
        _append_flag(risk_flags, "PEDIGREE_SURFACE_UNKNOWN")

    distance = race_info.distance or 0
    if distance >= 2200 and ("中長距離" in distance_tendency or "スタミナ" in traits):
        _append_flag(positive_flags, "PEDIGREE_STAMINA_FIT")
        _append_flag(positive_flags, "PEDIGREE_DISTANCE_FIT")
    elif _distance_has_fit(distance_tendency, traits, distance):
        _append_flag(positive_flags, "PEDIGREE_DISTANCE_FIT")
    elif _distance_has_risk(distance_tendency, traits, distance):
        _append_flag(risk_flags, "PEDIGREE_DISTANCE_RISK")
    elif distance > 0 and distance_tendency == "unknown":
        _append_flag(risk_flags, "PEDIGREE_DISTANCE_UNKNOWN")

    if race_info.track_condition and race_info.track_condition != "良" and any(token in traits for token in ("パワー", "重馬場")):
        _append_flag(positive_flags, "PEDIGREE_TRACK_CONDITION_FIT")
    if race_info.track_condition and race_info.track_condition != "良" and track_tendency == "unknown":
        _append_flag(risk_flags, "PEDIGREE_DATA_INCOMPLETE")
    if any(token in traits for token in ("パワー", "重馬場")):
        _append_flag(positive_flags, "PEDIGREE_POWER_FIT")

    positive_flags, risk_flags = _merge_performance_flags(positive_flags, risk_flags, profiles)
    surface_tendency = _merge_performance_tendency(surface_tendency, profiles, "surface_tendency")
    distance_tendency = _merge_performance_tendency(distance_tendency, profiles, "distance_tendency")
    track_tendency = _merge_performance_tendency(track_tendency, profiles, "track_condition_tendency")
    pace_tendency = _merge_performance_tendency(pace_tendency, profiles, "pace_tendency")

    comment_parts: list[str] = []
    if pedigree.sire and sire_knowledge:
        comment_parts.append(f"父{pedigree.sire}は{ sire_surface }・{ sire_distance }向き。")
    elif pedigree.sire:
        comment_parts.append("父系の知識は不足している。")
    if pedigree.damsire and damsire_knowledge:
        trait_hint = "スタミナ/パワー" if any(token in damsire_traits for token in ("スタミナ", "パワー")) else "適性"
        comment_parts.append(f"母父{pedigree.damsire}からは{trait_hint}面の補強もある。")
    if "PEDIGREE_DISTANCE_FIT" in positive_flags:
        comment_parts.append("今回の距離設定は血統面で後押しになる。")
    if "PEDIGREE_STAMINA_FIT" in positive_flags:
        comment_parts.append(f"{race_info.course}{race_info.surface}{distance}mではスタミナと持続力を評価できる。")
    if "PEDIGREE_TRACK_CONDITION_FIT" in positive_flags:
        comment_parts.append("馬場悪化でも血統面の下支えがある。")
    if "PEDIGREE_DISTANCE_RISK" in risk_flags:
        comment_parts.append("一方で今回の距離は忙しい可能性があり、対応力は課題。")
    if sire_knowledge is None and damsire_knowledge is not None:
        comment_parts.insert(0, "父系の知識は不足しているが、")
    if profiles:
        profile_comments = [
            str(getattr(profile, "overall_comment", "") or "").strip()
            for profile in profiles[:2]
            if str(getattr(profile, "overall_comment", "") or "").strip()
        ]
        if profile_comments:
            comment_parts.append("祖先実績: " + " ".join(profile_comments))
    if not positive_flags and risk_flags:
        comment_parts.append("強調材料は限られ、血統評価は慎重。")

    return PedigreeAnalysis(
        horse_no=horse_no,
        horse_name=final_horse_name,
        sire=pedigree.sire,
        dam=pedigree.dam,
        damsire=pedigree.damsire,
        surface_tendency=surface_tendency,
        distance_tendency=distance_tendency,
        track_condition_tendency=track_tendency,
        pace_tendency=pace_tendency,
        positive_flags=positive_flags,
        risk_flags=risk_flags,
        performance_profiles=profiles,
        performance_score_hint=performance_score_hint,
        overall_comment=" ".join(comment_parts[:4]),
    )


def build_pedigree_analyses_for_race(horses: list[HorseEntry], race_info: RaceInfo) -> list[PedigreeAnalysis]:
    analyses: list[PedigreeAnalysis] = []
    for horse in horses:
        if not horse.horse_id:
            analyses.append(
                _unknown_analysis(
                    horse.horse_no,
                    horse.horse_name,
                    PedigreeInfo(horse_id="unknown", horse_name=horse.horse_name),
                )
            )
            continue
        html = None
        cache_path = get_horse_cache_path(horse.horse_id)
        if cache_path.exists():
            html = cache_path.read_text(encoding="utf-8")
        elif horse.horse_id.isdigit():
            try:
                html = fetch_horse_html(horse.horse_id)
            except Exception as exc:
                warnings.warn(
                    f"failed to fetch horse pedigree page: horse_id={horse.horse_id}: {exc}",
                    stacklevel=2,
                )
        pedigree = parse_pedigree_info(html, horse.horse_id, horse.horse_name) if html else PedigreeInfo(horse_id=horse.horse_id, horse_name=horse.horse_name)
        if pedigree.sire is None and horse.horse_id.isdigit():
            try:
                pedigree_html = fetch_pedigree_html(horse.horse_id, horse_html=html)
                pedigree = parse_pedigree_info(pedigree_html, horse.horse_id, horse.horse_name)
            except Exception as exc:
                warnings.warn(
                    f"failed to fetch or parse pedigree page: horse_id={horse.horse_id}: {exc}",
                    stacklevel=2,
                )
        performance_profiles = (
            build_performance_profiles_for_pedigree(pedigree, race_info)
            if _effective_recent_result_count(horse) < 5
            else []
        )
        analyses.append(
            analyze_pedigree(
                pedigree,
                race_info,
                horse_no=horse.horse_no,
                horse_name=horse.horse_name,
                performance_profiles=performance_profiles,
            )
        )
    return analyses

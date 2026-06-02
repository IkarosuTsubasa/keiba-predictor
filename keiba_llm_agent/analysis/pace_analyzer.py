from __future__ import annotations

from statistics import mean

from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun


def _append_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _first_positions(runs: list[RecentRun]) -> list[int]:
    positions: list[int] = []
    for run in runs:
        if run.corner_positions:
            positions.append(run.corner_positions[0])
    return positions


def _infer_running_style(runs: list[RecentRun]) -> tuple[str, float]:
    first_positions = _first_positions(runs)
    if not first_positions:
        return "不明", 0.0
    avg_position = mean(first_positions)
    if avg_position <= 2:
        return "逃げ", avg_position
    if avg_position <= 5:
        return "先行", avg_position

    valid_relative = [
        (first_position, run.field_size)
        for first_position, run in zip(first_positions, [run for run in runs if run.corner_positions], strict=False)
        if run.field_size
    ]
    if valid_relative:
        avg_relative = mean(position / field_size for position, field_size in valid_relative)
        if avg_relative <= 0.6:
            return "差し", avg_position
    return "追込", avg_position


def _early_position_score(runs: list[RecentRun]) -> float:
    first_positions = _first_positions(runs)
    if not first_positions:
        return 0.0
    scores: list[float] = []
    for position, run in zip(first_positions, [run for run in runs if run.corner_positions], strict=False):
        field_size = run.field_size or max(position, 10)
        score = max(0.0, min(10.0, 10.0 * (1 - (position - 1) / max(field_size - 1, 1))))
        scores.append(score)
    return round(mean(scores), 1) if scores else 0.0


def _late_position_score(runs: list[RecentRun]) -> float:
    sectionals = [run.final_3f for run in runs if run.final_3f is not None]
    if sectionals:
        best = min(sectionals)
        worst = max(sectionals)
        if best == worst:
            return 6.0
        normalized = [(worst - sec) / (worst - best) * 10 for sec in sectionals]
        return round(mean(normalized), 1)

    improvements: list[float] = []
    for run in runs:
        if run.corner_positions and run.finish is not None:
            improvements.append(max(0.0, min(10.0, run.corner_positions[0] - run.finish + 5)))
    return round(mean(improvements), 1) if improvements else 0.0


def _position_stability(styles: list[str]) -> str:
    usable = [style for style in styles if style != "不明"]
    if len(usable) < 2:
        return "unknown"
    return "安定" if len(set(usable)) == 1 else "不安定"


def analyze_horse_pace(horse: HorseEntry, race_info: RaceInfo) -> HorsePaceAnalysis:
    runs = horse.recent_runs[:5]
    positive_flags: list[str] = []
    risk_flags: list[str] = []
    if not runs or not any(run.corner_positions for run in runs):
        _append_flag(risk_flags, "PACE_DATA_INCOMPLETE")
        return HorsePaceAnalysis(
            horse_no=horse.horse_no,
            horse_name=horse.horse_name,
            running_style="不明",
            early_position_score=0.0,
            late_position_score=0.0,
            position_stability="unknown",
            positive_flags=positive_flags,
            risk_flags=risk_flags,
            overall_comment="通過順データが不足しており、脚質評価はunknown。",
        )

    style, _ = _infer_running_style(runs)
    style_history = [_infer_running_style([run])[0] for run in runs]
    early_score = _early_position_score(runs)
    late_score = _late_position_score(runs)
    stability = _position_stability(style_history)

    if style in {"逃げ", "先行"}:
        _append_flag(positive_flags, "FRONT_RUNNING_ADVANTAGE" if style == "逃げ" else "STALKER_ADVANTAGE")
    if late_score >= 6.5:
        _append_flag(positive_flags, "CLOSING_SPEED")
    if stability == "安定":
        _append_flag(positive_flags, "POSITION_STABLE")
    if stability == "不安定":
        _append_flag(risk_flags, "POSITION_UNSTABLE")

    comment_parts: list[str] = [f"脚質は{style}寄り。"]
    if stability == "安定":
        comment_parts.append("位置取りは安定。")
    elif stability == "不安定":
        comment_parts.append("位置取りはやや不安定。")
    if late_score >= 6.5:
        comment_parts.append("終いの脚も一定水準。")
    return HorsePaceAnalysis(
        horse_no=horse.horse_no,
        horse_name=horse.horse_name,
        running_style=style,
        early_position_score=early_score,
        late_position_score=late_score,
        position_stability=stability,
        positive_flags=positive_flags,
        risk_flags=risk_flags,
        overall_comment=" ".join(comment_parts),
    )


def project_race_pace(horse_analyses: list[HorsePaceAnalysis], race_info: RaceInfo) -> RacePaceProjection:
    front_runner_count = sum(1 for item in horse_analyses if item.running_style == "逃げ")
    stalker_count = sum(1 for item in horse_analyses if item.running_style == "先行")
    closer_count = sum(1 for item in horse_analyses if item.running_style in {"差し", "追込"})

    if front_runner_count >= 3:
        projected_pace = "fast"
    elif front_runner_count == 2 and stalker_count >= 4:
        projected_pace = "fast"
    elif front_runner_count == 1 and stalker_count >= 5:
        projected_pace = "average"
    elif front_runner_count == 0 and stalker_count <= 2:
        projected_pace = "slow"
    elif front_runner_count <= 1 and stalker_count <= 3:
        projected_pace = "average" if stalker_count >= 2 else "slow"
    else:
        projected_pace = "average"

    favorable_styles: list[str] = []
    risk_styles: list[str] = []
    pace_comment = "展開傾向は平均的。"

    if projected_pace == "fast":
        favorable_styles = ["差し", "追込"]
        risk_styles = ["逃げ"]
        pace_comment = "逃げ・先行勢が多く、流れは速くなりやすい。"
    elif projected_pace == "slow":
        favorable_styles = ["逃げ", "先行"]
        risk_styles = ["追込"]
        pace_comment = "前半は落ち着きやすく、前有利の想定。"
    else:
        favorable_styles = ["先行", "差し"]
        risk_styles = []
        pace_comment = "隊列は極端になりにくく、平均的な流れを想定。"

    if race_info.course == "京都" and race_info.surface == "芝" and race_info.distance == 2400:
        if front_runner_count <= 1:
            projected_pace = "average" if stalker_count >= 2 else "slow"
            favorable_styles = ["先行", "差し"] if projected_pace == "average" else ["逃げ", "先行"]
            risk_styles = [] if projected_pace == "average" else ["追込"]
            pace_comment = "逃げ候補は多くなく、京都芝2400mらしくslow〜average寄りを想定。先行〜好位と持続力を重視。"
        else:
            pace_comment += " 京都芝2400mでも前受け勢の多さから流れは締まりやすい。"
    elif race_info.course == "東京" and race_info.surface == "芝" and race_info.distance == 1600:
        pace_comment += " 東京芝1600mはaverage〜fast寄りで、差しも届きやすい。"
        if "差し" not in favorable_styles:
            favorable_styles.append("差し")
    elif race_info.course == "新潟" and race_info.surface == "芝" and race_info.distance == 1000:
        pace_comment += " 新潟芝1000mは先行・スピード型を重視。"
        if "逃げ" not in favorable_styles:
            favorable_styles.append("逃げ")

    return RacePaceProjection(
        projected_pace=projected_pace,
        front_runner_count=front_runner_count,
        stalker_count=stalker_count,
        closer_count=closer_count,
        pace_comment=pace_comment,
        favorable_styles=favorable_styles,
        risk_styles=risk_styles,
    )


def apply_pace_fit_flags(
    horse_analyses: list[HorsePaceAnalysis],
    race_projection: RacePaceProjection,
) -> list[HorsePaceAnalysis]:
    updated: list[HorsePaceAnalysis] = []
    favorable = set(race_projection.favorable_styles)
    risk = set(race_projection.risk_styles)
    for analysis in horse_analyses:
        positive_flags = list(analysis.positive_flags)
        risk_flags = list(analysis.risk_flags)
        if analysis.running_style in favorable:
            _append_flag(positive_flags, "PACE_FIT")
        if analysis.running_style in risk:
            _append_flag(risk_flags, "PACE_MISMATCH")
        comment = analysis.overall_comment
        if "PACE_FIT" in positive_flags:
            comment += " 想定ペースとの噛み合いも見込める。"
        if "PACE_MISMATCH" in risk_flags:
            comment += " 想定ペースとのズレには注意。"
        updated.append(
            analysis.model_copy(
                update={
                    "positive_flags": positive_flags,
                    "risk_flags": risk_flags,
                    "overall_comment": comment.strip(),
                }
            )
        )
    return updated


def analyze_pace_for_race(
    horses: list[HorseEntry],
    race_info: RaceInfo,
) -> tuple[list[HorsePaceAnalysis], RacePaceProjection]:
    horse_analyses = [analyze_horse_pace(horse, race_info) for horse in horses]
    race_projection = project_race_pace(horse_analyses, race_info)
    horse_analyses = apply_pace_fit_flags(horse_analyses, race_projection)
    return horse_analyses, race_projection

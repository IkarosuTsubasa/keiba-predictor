from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun


NAR_COURSE_SOURCE_URL = "https://www.keiba.go.jp/guide/course/"


@dataclass(frozen=True)
class LocalCourseProfile:
    course: str
    surface: str
    layout: str
    direction: str
    lap_length_m: float
    home_straight_m: float
    elevation_m: float
    width_m: float
    full_gate: int
    source_url: str = NAR_COURSE_SOURCE_URL


@dataclass(frozen=True)
class LocalCourseFit:
    residual: float
    shrunken_residual: float
    analogous_performance: float
    baseline_performance: float
    reliability: float
    effective_runs: float
    analog_count: int
    exact_course_count: int
    exact_course_performance: float
    exact_course_shrunken_residual: float
    average_similarity: float
    best_similarity: float
    same_direction_shrunken_residual: float


def _profile(
    course: str,
    surface: str,
    layout: str,
    direction: str,
    lap_length_m: float,
    home_straight_m: float,
    elevation_m: float,
    width_m: float,
    full_gate: int,
) -> LocalCourseProfile:
    return LocalCourseProfile(
        course=course,
        surface=surface,
        layout=layout,
        direction=direction,
        lap_length_m=lap_length_m,
        home_straight_m=home_straight_m,
        elevation_m=elevation_m,
        width_m=width_m,
        full_gate=full_gate,
    )


# Current flat NAR courses. Width uses the midpoint when the official source
# publishes a range. Oi and Monbetsu retain their distinct inner/outer layouts.
_LOCAL_COURSE_PROFILES = {
    ("門別", "ダート", "outer"): _profile(
        "門別", "ダート", "outer", "right", 1600.0, 330.0, 1.54, 25.0, 16
    ),
    ("門別", "ダート", "inner"): _profile(
        "門別", "ダート", "inner", "right", 1376.0, 218.0, 1.54, 25.0, 16
    ),
    ("盛岡", "ダート", "standard"): _profile(
        "盛岡", "ダート", "standard", "left", 1600.0, 300.0, 4.4, 25.0, 16
    ),
    ("盛岡", "芝", "standard"): _profile(
        "盛岡", "芝", "standard", "left", 1400.0, 300.0, 4.6, 25.0, 14
    ),
    ("水沢", "ダート", "standard"): _profile(
        "水沢", "ダート", "standard", "right", 1200.0, 245.0, 0.0, 20.0, 12
    ),
    ("浦和", "ダート", "standard"): _profile(
        "浦和", "ダート", "standard", "left", 1200.0, 220.0, 0.0, 18.75, 14
    ),
    ("船橋", "ダート", "outer"): _profile(
        "船橋", "ダート", "outer", "left", 1400.0, 308.0, 0.0, 25.0, 14
    ),
    ("船橋", "ダート", "inner"): _profile(
        "船橋", "ダート", "inner", "left", 1250.0, 308.0, 0.0, 20.0, 14
    ),
    ("大井", "ダート", "outer_right"): _profile(
        "大井", "ダート", "outer_right", "right", 1600.0, 386.0, 0.0, 25.0, 16
    ),
    ("大井", "ダート", "inner_right"): _profile(
        "大井", "ダート", "inner_right", "right", 1400.0, 286.0, 0.0, 25.0, 16
    ),
    ("大井", "ダート", "outer_left"): _profile(
        "大井", "ダート", "outer_left", "left", 1600.0, 300.0, 0.0, 25.0, 14
    ),
    ("川崎", "ダート", "standard"): _profile(
        "川崎", "ダート", "standard", "left", 1200.0, 300.0, 0.0, 25.0, 14
    ),
    ("金沢", "ダート", "standard"): _profile(
        "金沢", "ダート", "standard", "right", 1200.0, 236.0, 0.0, 20.0, 12
    ),
    ("笠松", "ダート", "standard"): _profile(
        "笠松", "ダート", "standard", "right", 1100.0, 201.0, 1.92, 20.0, 12
    ),
    ("名古屋", "ダート", "standard"): _profile(
        "名古屋", "ダート", "standard", "right", 1180.0, 240.0, 0.0, 30.0, 12
    ),
    ("園田", "ダート", "standard"): _profile(
        "園田", "ダート", "standard", "right", 1051.0, 213.0, 1.23, 22.0, 12
    ),
    ("姫路", "ダート", "standard"): _profile(
        "姫路", "ダート", "standard", "right", 1200.0, 230.0, 0.0, 22.5, 12
    ),
    ("高知", "ダート", "standard"): _profile(
        "高知", "ダート", "standard", "right", 1100.0, 200.0, 1.58, 24.5, 12
    ),
    ("佐賀", "ダート", "standard"): _profile(
        "佐賀", "ダート", "standard", "right", 1100.0, 200.0, 1.0, 21.6, 12
    ),
}


def normalize_local_course(value: str | None) -> str | None:
    normalized = str(value or "").strip().replace("競馬場", "")
    return normalized or None


def normalize_local_surface(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if normalized.startswith("芝"):
        return "芝"
    if normalized in {"ダ", "ダート"} or normalized.startswith("ダート"):
        return "ダート"
    return normalized or None


def _layout_for(course: str, distance: int | None) -> str:
    if course == "門別":
        return "inner" if distance in {1500, 1600} else "outer"
    if course == "船橋":
        return "inner" if distance in {1400, 1500} else "outer"
    if course == "大井":
        if distance == 1650:
            return "outer_left"
        if distance in {1500, 1600}:
            return "inner_right"
        return "outer_right"
    return "standard"


def resolve_local_course_profile(
    course: str | None,
    surface: str | None,
    distance: int | None = None,
) -> LocalCourseProfile | None:
    normalized_course = normalize_local_course(course)
    normalized_surface = normalize_local_surface(surface)
    if not normalized_course or not normalized_surface:
        return None
    return _LOCAL_COURSE_PROFILES.get(
        (
            normalized_course,
            normalized_surface,
            _layout_for(normalized_course, distance),
        )
    )


def _numeric_similarity(left: float, right: float, scale: float) -> float:
    return math.exp(-abs(left - right) / scale)


def local_course_similarity(
    target_course: str | None,
    target_surface: str | None,
    target_distance: int | None,
    previous_course: str | None,
    previous_surface: str | None,
    previous_distance: int | None,
) -> float:
    target = resolve_local_course_profile(
        target_course, target_surface, target_distance
    )
    previous = resolve_local_course_profile(
        previous_course, previous_surface, previous_distance
    )
    if target is None or previous is None or target.surface != previous.surface:
        return 0.0
    components = (
        (0.18, 1.0 if target.direction == previous.direction else 0.0),
        (
            0.22,
            _numeric_similarity(
                target.home_straight_m, previous.home_straight_m, 90.0
            ),
        ),
        (
            0.18,
            _numeric_similarity(
                target.lap_length_m, previous.lap_length_m, 300.0
            ),
        ),
        (
            0.12,
            _numeric_similarity(
                target.elevation_m, previous.elevation_m, 1.8
            ),
        ),
        (
            0.08,
            _numeric_similarity(target.width_m, previous.width_m, 8.0),
        ),
        (
            0.06,
            _numeric_similarity(
                float(target.full_gate), float(previous.full_gate), 4.0
            ),
        ),
        (
            0.16,
            _numeric_similarity(
                float(target_distance or 0),
                float(previous_distance or 0),
                500.0,
            )
            if target_distance and previous_distance
            else 0.5,
        ),
    )
    return sum(weight * value for weight, value in components)


def _finish_percentile(run: RecentRun) -> float | None:
    if run.finish is None or run.field_size is None or run.field_size <= 1:
        return None
    return max(
        0.0,
        min(1.0, (run.field_size - run.finish) / (run.field_size - 1)),
    )


def _weighted_average(
    rows: list[tuple[RecentRun, float, float]],
    weight_for: Callable[[RecentRun, float, float], float],
) -> tuple[float, float]:
    weights = [
        float(weight_for(run, similarity, recency))
        for run, similarity, recency in rows
    ]
    denominator = sum(weights)
    if denominator <= 0.0:
        return 0.5, 0.0
    value = sum(
        (_finish_percentile(run) or 0.0) * weight
        for (run, _, _), weight in zip(rows, weights, strict=True)
    ) / denominator
    squared = sum(weight * weight for weight in weights)
    effective_runs = denominator * denominator / squared if squared else 0.0
    return value, effective_runs


def horse_local_course_fit(
    horse: HorseEntry,
    race_info: RaceInfo,
    *,
    max_runs: int = 12,
    reliability_prior: float = 2.5,
) -> LocalCourseFit:
    target = resolve_local_course_profile(
        race_info.course, race_info.surface, race_info.distance
    )
    if target is None:
        return LocalCourseFit(
            0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0, 0, 0.5, 0.0, 0.0, 0.0, 0.0
        )

    target_course = normalize_local_course(race_info.course)
    rows: list[tuple[RecentRun, float, float]] = []
    for index, run in enumerate(horse.recent_runs[:max_runs]):
        if _finish_percentile(run) is None:
            continue
        similarity = local_course_similarity(
            race_info.course,
            race_info.surface,
            race_info.distance,
            run.course,
            run.surface,
            run.distance,
        )
        if similarity <= 0.0:
            continue
        rows.append((run, similarity, 0.86**index))
    if not rows:
        return LocalCourseFit(
            0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0, 0, 0.5, 0.0, 0.0, 0.0, 0.0
        )

    baseline, _ = _weighted_average(
        rows, lambda _run, _similarity, recency: recency
    )
    analogous, effective_runs = _weighted_average(
        rows,
        lambda _run, similarity, recency: recency * similarity * similarity,
    )
    reliability = effective_runs / (effective_runs + reliability_prior)
    recency_sum = sum(recency for _, _, recency in rows)
    average_similarity = sum(
        similarity * recency for _, similarity, recency in rows
    ) / recency_sum
    residual = analogous - baseline

    exact_rows = [
        row
        for row in rows
        if normalize_local_course(row[0].course) == target_course
    ]
    exact_performance, exact_effective_runs = _weighted_average(
        exact_rows, lambda _run, _similarity, recency: recency
    )
    exact_reliability = exact_effective_runs / (exact_effective_runs + 2.0)

    direction_rows = [
        row
        for row in rows
        if (
            previous := resolve_local_course_profile(
                row[0].course, row[0].surface, row[0].distance
            )
        )
        is not None
        and previous.direction == target.direction
    ]
    direction_performance, direction_effective_runs = _weighted_average(
        direction_rows, lambda _run, _similarity, recency: recency
    )
    direction_reliability = direction_effective_runs / (
        direction_effective_runs + 2.5
    )
    return LocalCourseFit(
        residual=residual,
        shrunken_residual=(
            residual * reliability * average_similarity
        ),
        analogous_performance=analogous,
        baseline_performance=baseline,
        reliability=reliability,
        effective_runs=effective_runs,
        analog_count=sum(
            similarity >= 0.62 for _, similarity, _ in rows
        ),
        exact_course_count=len(exact_rows),
        exact_course_performance=exact_performance,
        exact_course_shrunken_residual=(
            (exact_performance - baseline) * exact_reliability
        ),
        average_similarity=average_similarity,
        best_similarity=max(similarity for _, similarity, _ in rows),
        same_direction_shrunken_residual=(
            (direction_performance - baseline) * direction_reliability
        ),
    )


LOCAL_COURSE_GEOMETRY_FEATURE_NAMES = (
    "local_geometry_fit_residual",
    "local_geometry_fit_shrunken_residual",
    "local_geometry_analogous_performance",
    "local_geometry_baseline_performance",
    "local_geometry_reliability",
    "local_geometry_effective_runs",
    "local_geometry_analog_count",
    "local_geometry_exact_course_count",
    "local_geometry_exact_course_performance",
    "local_geometry_exact_course_shrunken_residual",
    "local_geometry_average_similarity",
    "local_geometry_best_similarity",
    "local_geometry_same_direction_shrunken_residual",
)


def build_local_course_geometry_features(
    horse: HorseEntry,
    race_info: RaceInfo,
) -> dict[str, float]:
    fit = horse_local_course_fit(horse, race_info)
    return {
        "local_geometry_fit_residual": fit.residual,
        "local_geometry_fit_shrunken_residual": fit.shrunken_residual,
        "local_geometry_analogous_performance": fit.analogous_performance,
        "local_geometry_baseline_performance": fit.baseline_performance,
        "local_geometry_reliability": fit.reliability,
        "local_geometry_effective_runs": fit.effective_runs,
        "local_geometry_analog_count": float(fit.analog_count),
        "local_geometry_exact_course_count": float(fit.exact_course_count),
        "local_geometry_exact_course_performance": fit.exact_course_performance,
        "local_geometry_exact_course_shrunken_residual": (
            fit.exact_course_shrunken_residual
        ),
        "local_geometry_average_similarity": fit.average_similarity,
        "local_geometry_best_similarity": fit.best_similarity,
        "local_geometry_same_direction_shrunken_residual": (
            fit.same_direction_shrunken_residual
        ),
    }


LOCAL_COURSE_STYLE_FEATURE_NAMES = (
    "local_style_short_straight_early",
    "local_style_long_straight_closing",
    "local_style_compact_early",
    "local_style_elevation_resilience",
    "local_style_wide_course_closing",
)


def build_local_course_style_features(
    horse: HorseEntry,
    race_info: RaceInfo,
    *,
    max_runs: int = 8,
) -> dict[str, float]:
    profile = resolve_local_course_profile(
        race_info.course, race_info.surface, race_info.distance
    )
    if profile is None:
        return {name: 0.0 for name in LOCAL_COURSE_STYLE_FEATURE_NAMES}

    rows: list[tuple[float, float, float, float]] = []
    for index, run in enumerate(horse.recent_runs[:max_runs]):
        performance = _finish_percentile(run)
        positions = run.corner_positions or []
        if performance is None or not positions or not run.field_size:
            continue
        denominator = max(run.field_size - 1, 1)
        early = max(
            0.0,
            min(1.0, (run.field_size - positions[0]) / denominator),
        )
        closing = max(
            -1.0,
            min(1.0, (positions[-1] - (run.finish or 0)) / denominator),
        )
        rows.append((early, closing, performance, 0.86**index))
    if not rows:
        return {name: 0.0 for name in LOCAL_COURSE_STYLE_FEATURE_NAMES}

    weight_sum = sum(row[3] for row in rows)
    early = sum(row[0] * row[3] for row in rows) / weight_sum
    closing = sum(row[1] * row[3] for row in rows) / weight_sum
    performance = sum(row[2] * row[3] for row in rows) / weight_sum
    shortness = max(
        0.0, min(1.0, (300.0 - profile.home_straight_m) / 100.0)
    )
    longness = max(
        0.0, min(1.0, (profile.home_straight_m - 220.0) / 170.0)
    )
    compactness = max(
        0.0, min(1.0, (1500.0 - profile.lap_length_m) / 450.0)
    )
    elevation = max(
        0.0, min(1.0, profile.elevation_m / 4.5)
    )
    width = max(0.0, min(1.0, (profile.width_m - 20.0) / 10.0))
    resilience = performance - early
    return {
        "local_style_short_straight_early": shortness * early,
        "local_style_long_straight_closing": longness * closing,
        "local_style_compact_early": compactness * early,
        "local_style_elevation_resilience": elevation * resilience,
        "local_style_wide_course_closing": width * closing,
    }

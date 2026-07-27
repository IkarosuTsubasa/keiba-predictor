from __future__ import annotations

import math
from dataclasses import dataclass

from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun


@dataclass(frozen=True)
class CourseProfile:
    course: str
    surface: str
    layout: str
    direction: str
    lap_length_m: float | None
    straight_length_m: float | None
    elevation_m: float | None
    finish_climb_m: float | None
    turf_base: str | None
    source_url: str


@dataclass(frozen=True)
class CourseSimilarityWeights:
    direction: float = 0.14
    layout: float = 0.14
    straight: float = 0.24
    lap_length: float = 0.16
    elevation: float = 0.13
    finish_climb: float = 0.12
    turf_base: float = 0.09
    distance: float = 0.10


@dataclass(frozen=True)
class HorseCourseGeometryFit:
    residual: float
    shrunken_residual: float
    analogous_performance: float
    baseline_performance: float
    reliability: float
    effective_runs: float
    analog_count: int
    exact_course_count: int
    average_similarity: float
    best_similarity: float


_JRA_COURSE_URLS = {
    "札幌": "https://www.jra.go.jp/facilities/race/sapporo/course/index.html",
    "函館": "https://www.jra.go.jp/facilities/race/hakodate/course/index.html",
    "福島": "https://www.jra.go.jp/facilities/race/fukushima/course/index.html",
    "新潟": "https://www.jra.go.jp/facilities/race/niigata/course/index.html",
    "東京": "https://www.jra.go.jp/facilities/race/tokyo/course/index.html",
    "中山": "https://www.jra.go.jp/facilities/race/nakayama/course/index.html",
    "中京": "https://www.jra.go.jp/facilities/race/chukyo/course/index.html",
    "京都": "https://www.jra.go.jp/facilities/race/kyoto/course/index.html",
    "阪神": "https://www.jra.go.jp/facilities/race/hanshin/course/index.html",
    "小倉": "https://www.jra.go.jp/facilities/race/kokura/course/index.html",
}


def _profile(
    course: str,
    surface: str,
    layout: str,
    direction: str,
    lap_length_m: float | None,
    straight_length_m: float | None,
    elevation_m: float | None,
    turf_base: str | None = None,
    finish_climb_m: float | None = 0.0,
) -> CourseProfile:
    return CourseProfile(
        course=course,
        surface=surface,
        layout=layout,
        direction=direction,
        lap_length_m=lap_length_m,
        straight_length_m=straight_length_m,
        elevation_m=elevation_m,
        finish_climb_m=finish_climb_m,
        turf_base=turf_base,
        source_url=_JRA_COURSE_URLS[course],
    )


# A-course measurements from the official JRA course pages. Turf base
# distinguishes the all-cool-season Hokkaido tracks from the warm-season base
# used by the other JRA tracks.
_COURSE_PROFILES = {
    ("札幌", "芝", "standard"): _profile(
        "札幌", "芝", "standard", "right", 1640.9, 266.1, 0.7, "cool"
    ),
    ("札幌", "ダート", "standard"): _profile(
        "札幌", "ダート", "standard", "right", 1487.0, 264.3, 0.9
    ),
    ("函館", "芝", "standard"): _profile(
        "函館", "芝", "standard", "right", 1626.6, 262.1, 3.5, "cool"
    ),
    ("函館", "ダート", "standard"): _profile(
        "函館", "ダート", "standard", "right", 1475.8, 260.3, 3.5
    ),
    ("福島", "芝", "standard"): _profile(
        "福島",
        "芝",
        "standard",
        "right",
        1600.0,
        292.0,
        1.9,
        "warm",
        1.2,
    ),
    ("福島", "ダート", "standard"): _profile(
        "福島",
        "ダート",
        "standard",
        "right",
        1444.6,
        295.7,
        2.1,
        finish_climb_m=1.2,
    ),
    ("新潟", "芝", "straight"): _profile(
        "新潟", "芝", "straight", "straight", None, 1000.0, None, "warm"
    ),
    ("新潟", "芝", "inner"): _profile(
        "新潟", "芝", "inner", "left", 1623.0, 358.7, 0.8, "warm"
    ),
    ("新潟", "芝", "outer"): _profile(
        "新潟", "芝", "outer", "left", 2223.0, 658.7, 2.2, "warm"
    ),
    ("新潟", "ダート", "standard"): _profile(
        "新潟", "ダート", "standard", "left", 1472.5, 353.9, 0.6
    ),
    ("東京", "芝", "standard"): _profile(
        "東京",
        "芝",
        "standard",
        "left",
        2083.1,
        525.9,
        2.7,
        "warm",
        2.0,
    ),
    ("東京", "ダート", "standard"): _profile(
        "東京",
        "ダート",
        "standard",
        "left",
        1899.0,
        501.6,
        2.5,
        finish_climb_m=2.4,
    ),
    ("中山", "芝", "inner"): _profile(
        "中山",
        "芝",
        "inner",
        "right",
        1667.1,
        310.0,
        5.3,
        "warm",
        2.2,
    ),
    ("中山", "芝", "outer"): _profile(
        "中山",
        "芝",
        "outer",
        "right",
        1839.7,
        310.0,
        5.3,
        "warm",
        2.2,
    ),
    ("中山", "ダート", "standard"): _profile(
        "中山",
        "ダート",
        "standard",
        "right",
        1493.0,
        308.0,
        4.5,
        finish_climb_m=2.2,
    ),
    ("中京", "芝", "standard"): _profile(
        "中京",
        "芝",
        "standard",
        "left",
        1705.9,
        412.5,
        3.5,
        "warm",
        2.0,
    ),
    ("中京", "ダート", "standard"): _profile(
        "中京",
        "ダート",
        "standard",
        "left",
        1530.0,
        410.7,
        3.4,
        finish_climb_m=2.0,
    ),
    ("京都", "芝", "inner"): _profile(
        "京都", "芝", "inner", "right", 1782.8, 328.4, 3.1, "warm"
    ),
    ("京都", "芝", "outer"): _profile(
        "京都", "芝", "outer", "right", 1894.3, 403.7, 4.3, "warm"
    ),
    ("京都", "ダート", "standard"): _profile(
        "京都", "ダート", "standard", "right", 1607.6, 329.1, 3.0
    ),
    ("阪神", "芝", "inner"): _profile(
        "阪神",
        "芝",
        "inner",
        "right",
        1689.0,
        356.5,
        1.9,
        "warm",
        1.8,
    ),
    ("阪神", "芝", "outer"): _profile(
        "阪神",
        "芝",
        "outer",
        "right",
        2089.0,
        473.6,
        2.4,
        "warm",
        1.8,
    ),
    ("阪神", "ダート", "standard"): _profile(
        "阪神",
        "ダート",
        "standard",
        "right",
        1517.6,
        352.7,
        1.6,
        finish_climb_m=1.6,
    ),
    ("小倉", "芝", "standard"): _profile(
        "小倉", "芝", "standard", "right", 1615.1, 293.0, 3.0, "warm"
    ),
    ("小倉", "ダート", "standard"): _profile(
        "小倉",
        "ダート",
        "standard",
        "right",
        1445.4,
        291.3,
        2.9,
        finish_climb_m=0.6,
    ),
}


_TURF_LAYOUT_BY_DISTANCE = {
    "新潟": {
        1000: "straight",
        1200: "inner",
        1400: "mixed",
        1600: "outer",
        1800: "outer",
        2000: "mixed",
        2200: "inner",
        2400: "inner",
        3000: "outer",
        3200: "outer",
    },
    "中山": {
        1200: "outer",
        1600: "outer",
        1800: "inner",
        2000: "inner",
        2200: "outer",
        2500: "inner",
        2600: "outer",
        3200: "mixed",
        3600: "inner",
        4000: "outer",
    },
    "京都": {
        1100: "inner",
        1200: "inner",
        1400: "mixed",
        1600: "mixed",
        1800: "outer",
        2000: "mixed",
        2200: "outer",
        2400: "outer",
        3000: "outer",
        3200: "outer",
    },
    "阪神": {
        1200: "inner",
        1400: "mixed",
        1600: "outer",
        1800: "outer",
        2000: "inner",
        2200: "inner",
        2400: "outer",
        2600: "outer",
        3000: "inner",
        3200: "mixed",
    },
}


def normalize_course(value: str | None) -> str | None:
    normalized = str(value or "").strip().replace("競馬場", "")
    return normalized or None


def normalize_surface(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if normalized.startswith("芝"):
        return "芝"
    if normalized in {"ダ", "ダート"} or normalized.startswith("ダート"):
        return "ダート"
    return normalized or None


def _average_optional(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _mixed_profile(course: str, surface: str) -> CourseProfile | None:
    variants = [
        profile
        for (profile_course, profile_surface, layout), profile in _COURSE_PROFILES.items()
        if profile_course == course
        and profile_surface == surface
        and layout in {"inner", "outer"}
    ]
    if not variants:
        return None
    return CourseProfile(
        course=course,
        surface=surface,
        layout="mixed",
        direction=variants[0].direction,
        lap_length_m=_average_optional(
            [profile.lap_length_m for profile in variants]
        ),
        straight_length_m=_average_optional(
            [profile.straight_length_m for profile in variants]
        ),
        elevation_m=_average_optional(
            [profile.elevation_m for profile in variants]
        ),
        finish_climb_m=_average_optional(
            [profile.finish_climb_m for profile in variants]
        ),
        turf_base=variants[0].turf_base,
        source_url=variants[0].source_url,
    )


def resolve_course_profile(
    course: str | None,
    surface: str | None,
    distance: int | None = None,
) -> CourseProfile | None:
    normalized_course = normalize_course(course)
    normalized_surface = normalize_surface(surface)
    if not normalized_course or not normalized_surface:
        return None
    standard = _COURSE_PROFILES.get(
        (normalized_course, normalized_surface, "standard")
    )
    if standard is not None:
        return standard
    if normalized_surface != "芝":
        return None
    layout = _TURF_LAYOUT_BY_DISTANCE.get(
        normalized_course, {}
    ).get(distance)
    if layout == "mixed" or layout is None:
        return _mixed_profile(normalized_course, normalized_surface)
    return _COURSE_PROFILES.get(
        (normalized_course, normalized_surface, layout)
    )


def _numeric_similarity(
    left: float | None,
    right: float | None,
    scale: float,
) -> float | None:
    if left is None or right is None:
        return None
    return math.exp(-abs(left - right) / scale)


def _layout_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if "straight" in {left, right}:
        return 0.0
    if "mixed" in {left, right}:
        return 0.8
    if "standard" in {left, right}:
        return 0.7
    return 0.45


def course_profile_similarity(
    target: CourseProfile,
    previous: CourseProfile,
    *,
    target_distance: int | None = None,
    previous_distance: int | None = None,
    weights: CourseSimilarityWeights | None = None,
) -> float:
    if target.surface != previous.surface:
        return 0.0
    selected = weights or CourseSimilarityWeights()
    components: list[tuple[float, float | None]] = [
        (
            selected.direction,
            1.0 if target.direction == previous.direction else 0.0,
        ),
        (
            selected.layout,
            _layout_similarity(target.layout, previous.layout),
        ),
        (
            selected.straight,
            _numeric_similarity(
                target.straight_length_m,
                previous.straight_length_m,
                180.0,
            ),
        ),
        (
            selected.lap_length,
            _numeric_similarity(
                target.lap_length_m,
                previous.lap_length_m,
                450.0,
            ),
        ),
        (
            selected.elevation,
            _numeric_similarity(
                target.elevation_m,
                previous.elevation_m,
                2.5,
            ),
        ),
        (
            selected.finish_climb,
            _numeric_similarity(
                target.finish_climb_m,
                previous.finish_climb_m,
                1.2,
            ),
        ),
        (
            selected.turf_base,
            (
                1.0
                if target.turf_base == previous.turf_base
                else 0.0
            )
            if target.turf_base is not None
            and previous.turf_base is not None
            else None,
        ),
        (
            selected.distance,
            _numeric_similarity(
                float(target_distance)
                if target_distance is not None
                else None,
                float(previous_distance)
                if previous_distance is not None
                else None,
                600.0,
            ),
        ),
    ]
    available = [
        (weight, value)
        for weight, value in components
        if value is not None and weight > 0.0
    ]
    if not available:
        return 0.0
    denominator = sum(weight for weight, _ in available)
    return sum(weight * value for weight, value in available) / denominator


def course_similarity(
    target_course: str | None,
    target_surface: str | None,
    target_distance: int | None,
    previous_course: str | None,
    previous_surface: str | None,
    previous_distance: int | None,
    *,
    weights: CourseSimilarityWeights | None = None,
) -> float:
    target = resolve_course_profile(
        target_course, target_surface, target_distance
    )
    previous = resolve_course_profile(
        previous_course, previous_surface, previous_distance
    )
    if target is None or previous is None:
        return 0.0
    return course_profile_similarity(
        target,
        previous,
        target_distance=target_distance,
        previous_distance=previous_distance,
        weights=weights,
    )


def _finish_percentile(run: RecentRun) -> float | None:
    if (
        run.finish is None
        or run.field_size is None
        or run.field_size <= 1
    ):
        return None
    return max(
        0.0,
        min(
            1.0,
            (run.field_size - run.finish) / (run.field_size - 1),
        ),
    )


def horse_course_geometry_fit(
    horse: HorseEntry,
    race_info: RaceInfo,
    *,
    max_runs: int = 12,
    reliability_prior: float = 2.5,
    weights: CourseSimilarityWeights | None = None,
) -> HorseCourseGeometryFit:
    target_surface = normalize_surface(race_info.surface)
    if resolve_course_profile(
        race_info.course,
        target_surface,
        race_info.distance,
    ) is None:
        return HorseCourseGeometryFit(
            residual=0.0,
            shrunken_residual=0.0,
            analogous_performance=0.5,
            baseline_performance=0.5,
            reliability=0.0,
            effective_runs=0.0,
            analog_count=0,
            exact_course_count=0,
            average_similarity=0.0,
            best_similarity=0.0,
        )

    rows: list[tuple[RecentRun, float, float, float]] = []
    for index, run in enumerate(horse.recent_runs[:max_runs]):
        performance = _finish_percentile(run)
        if (
            performance is None
            or normalize_surface(run.surface) != target_surface
        ):
            continue
        similarity = course_similarity(
            race_info.course,
            target_surface,
            race_info.distance,
            run.course,
            run.surface,
            run.distance,
            weights=weights,
        )
        if similarity <= 0.0:
            continue
        recency_weight = 0.86**index
        rows.append((run, performance, similarity, recency_weight))

    if not rows:
        return HorseCourseGeometryFit(
            residual=0.0,
            shrunken_residual=0.0,
            analogous_performance=0.5,
            baseline_performance=0.5,
            reliability=0.0,
            effective_runs=0.0,
            analog_count=0,
            exact_course_count=0,
            average_similarity=0.0,
            best_similarity=0.0,
        )

    baseline_weight = sum(row[3] for row in rows)
    baseline = sum(
        performance * recency_weight
        for _, performance, _, recency_weight in rows
    ) / baseline_weight
    analog_weights = [
        recency_weight * similarity * similarity
        for _, _, similarity, recency_weight in rows
    ]
    analog_weight = sum(analog_weights)
    analogous = sum(
        performance * analog_weight_for_run
        for (_, performance, _, _), analog_weight_for_run in zip(
            rows, analog_weights, strict=True
        )
    ) / analog_weight
    squared_weight_sum = sum(weight * weight for weight in analog_weights)
    effective_runs = (
        analog_weight * analog_weight / squared_weight_sum
        if squared_weight_sum > 0.0
        else 0.0
    )
    reliability = effective_runs / (effective_runs + reliability_prior)
    residual = analogous - baseline
    average_similarity = sum(
        similarity * recency_weight
        for _, _, similarity, recency_weight in rows
    ) / baseline_weight
    target_course = normalize_course(race_info.course)
    exact_course_count = sum(
        normalize_course(run.course) == target_course
        for run, _, _, _ in rows
    )
    return HorseCourseGeometryFit(
        residual=residual,
        shrunken_residual=residual * reliability * average_similarity,
        analogous_performance=analogous,
        baseline_performance=baseline,
        reliability=reliability,
        effective_runs=effective_runs,
        analog_count=sum(
            similarity >= 0.55 for _, _, similarity, _ in rows
        ),
        exact_course_count=exact_course_count,
        average_similarity=average_similarity,
        best_similarity=max(row[2] for row in rows),
    )


COURSE_GEOMETRY_FEATURE_NAMES = (
    "geometry_fit_residual",
    "geometry_fit_shrunken_residual",
    "geometry_analogous_performance",
    "geometry_baseline_performance",
    "geometry_reliability",
    "geometry_effective_runs",
    "geometry_analog_count",
    "geometry_exact_course_count",
    "geometry_average_similarity",
    "geometry_best_similarity",
)


def build_course_geometry_features(
    horse: HorseEntry,
    race_info: RaceInfo,
) -> dict[str, float]:
    fit = horse_course_geometry_fit(horse, race_info)
    return {
        "geometry_fit_residual": fit.residual,
        "geometry_fit_shrunken_residual": fit.shrunken_residual,
        "geometry_analogous_performance": fit.analogous_performance,
        "geometry_baseline_performance": fit.baseline_performance,
        "geometry_reliability": fit.reliability,
        "geometry_effective_runs": fit.effective_runs,
        "geometry_analog_count": float(fit.analog_count),
        "geometry_exact_course_count": float(fit.exact_course_count),
        "geometry_average_similarity": fit.average_similarity,
        "geometry_best_similarity": fit.best_similarity,
    }


COURSE_STYLE_FEATURE_NAMES = (
    "course_style_short_straight_early",
    "course_style_long_straight_closing",
    "course_style_compact_early",
    "course_style_finish_hill_resilience",
    "course_style_straight_course_speed",
)


def build_course_style_features(
    horse: HorseEntry,
    race_info: RaceInfo,
    *,
    max_runs: int = 8,
) -> dict[str, float]:
    profile = resolve_course_profile(
        race_info.course,
        race_info.surface,
        race_info.distance,
    )
    if profile is None:
        return {name: 0.0 for name in COURSE_STYLE_FEATURE_NAMES}

    rows: list[tuple[float, float, float, float]] = []
    target_surface = normalize_surface(race_info.surface)
    for index, run in enumerate(horse.recent_runs[:max_runs]):
        if (
            normalize_surface(run.surface) != target_surface
            or run.field_size is None
            or run.field_size <= 1
        ):
            continue
        positions = run.corner_positions or []
        performance = _finish_percentile(run)
        if performance is None or not positions:
            continue
        denominator = run.field_size - 1
        early = max(
            0.0,
            min(1.0, (run.field_size - positions[0]) / denominator),
        )
        closing = max(
            -1.0,
            min(1.0, (positions[-1] - run.finish) / denominator),
        )
        rows.append((early, closing, performance, 0.86**index))
    if not rows:
        return {name: 0.0 for name in COURSE_STYLE_FEATURE_NAMES}

    weight_sum = sum(row[3] for row in rows)
    early_mean = sum(row[0] * row[3] for row in rows) / weight_sum
    closing_mean = sum(row[1] * row[3] for row in rows) / weight_sum
    performance_mean = (
        sum(row[2] * row[3] for row in rows) / weight_sum
    )
    straight = profile.straight_length_m or 400.0
    lap_length = profile.lap_length_m or 1800.0
    shortness = max(0.0, min(1.0, (525.0 - straight) / 265.0))
    longness = max(0.0, min(1.0, (straight - 350.0) / 310.0))
    compactness = max(
        0.0, min(1.0, (1900.0 - lap_length) / 450.0)
    )
    finish_hill = max(
        0.0,
        min(1.0, (profile.finish_climb_m or 0.0) / 2.2),
    )
    late_resilience = performance_mean - early_mean
    return {
        "course_style_short_straight_early": (
            shortness * (early_mean - 0.5)
        ),
        "course_style_long_straight_closing": (
            longness * closing_mean
        ),
        "course_style_compact_early": (
            compactness * (early_mean - 0.5)
        ),
        "course_style_finish_hill_resilience": (
            finish_hill * late_resilience
        ),
        "course_style_straight_course_speed": (
            (early_mean - 0.5)
            if profile.layout == "straight"
            else 0.0
        ),
    }

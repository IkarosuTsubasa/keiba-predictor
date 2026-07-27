from keiba_llm_agent.analysis.local_course_geometry import (
    build_local_course_geometry_features,
    horse_local_course_fit,
    local_course_similarity,
    resolve_local_course_profile,
)
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun


def test_resolves_nar_course_variants() -> None:
    oi_inner = resolve_local_course_profile("大井", "ダート", 1600)
    oi_outer = resolve_local_course_profile("大井", "ダート", 1200)
    oi_left = resolve_local_course_profile("大井", "ダート", 1650)
    monbetsu_inner = resolve_local_course_profile("門別", "ダ", 1500)

    assert oi_inner is not None
    assert oi_inner.layout == "inner_right"
    assert oi_inner.home_straight_m == 286.0
    assert oi_outer is not None
    assert oi_outer.layout == "outer_right"
    assert oi_outer.home_straight_m == 386.0
    assert oi_left is not None
    assert oi_left.direction == "left"
    assert monbetsu_inner is not None
    assert monbetsu_inner.layout == "inner"


def test_similar_local_courses_score_above_dissimilar_courses() -> None:
    sonoda_kochi = local_course_similarity(
        "園田", "ダート", 1400, "高知", "ダート", 1400
    )
    monbetsu_kochi = local_course_similarity(
        "門別", "ダート", 1200, "高知", "ダート", 1400
    )

    assert sonoda_kochi > monbetsu_kochi


def test_fit_rewards_matching_physical_history() -> None:
    horse = HorseEntry(
        horse_no=1,
        horse_name="test",
        recent_runs=[
            RecentRun(
                course="高知",
                surface="ダート",
                distance=1400,
                finish=1,
                field_size=12,
            ),
            RecentRun(
                course="佐賀",
                surface="ダート",
                distance=1400,
                finish=2,
                field_size=12,
            ),
            RecentRun(
                course="門別",
                surface="ダート",
                distance=1200,
                finish=12,
                field_size=12,
            ),
        ],
    )
    sonoda_fit = horse_local_course_fit(
        horse,
        RaceInfo(
            race_id="local-1",
            course="園田",
            surface="ダート",
            distance=1400,
        ),
    )
    monbetsu_fit = horse_local_course_fit(
        horse,
        RaceInfo(
            race_id="local-2",
            course="門別",
            surface="ダート",
            distance=1200,
        ),
    )

    assert sonoda_fit.shrunken_residual > monbetsu_fit.shrunken_residual
    assert sonoda_fit.analog_count >= 2


def test_unknown_course_returns_neutral_features() -> None:
    horse = HorseEntry(horse_no=1, horse_name="test")
    features = build_local_course_geometry_features(
        horse,
        RaceInfo(
            race_id="unknown",
            course="未知",
            surface="ダート",
            distance=1400,
        ),
    )

    assert features["local_geometry_reliability"] == 0.0
    assert features["local_geometry_analogous_performance"] == 0.5

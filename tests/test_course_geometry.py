from __future__ import annotations

import unittest

from keiba_llm_agent.analysis.course_geometry import (
    build_course_style_features,
    course_similarity,
    horse_course_geometry_fit,
    resolve_course_profile,
)
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun


class CourseGeometryTests(unittest.TestCase):
    def test_official_tokyo_turf_profile(self) -> None:
        profile = resolve_course_profile("東京", "芝", 1600)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.direction, "left")
        self.assertEqual(profile.layout, "standard")
        self.assertAlmostEqual(profile.lap_length_m or 0.0, 2083.1)
        self.assertAlmostEqual(profile.straight_length_m or 0.0, 525.9)
        self.assertAlmostEqual(profile.elevation_m or 0.0, 2.7)
        self.assertAlmostEqual(profile.finish_climb_m or 0.0, 2.0)

    def test_niigata_distance_resolves_layout(self) -> None:
        straight = resolve_course_profile("新潟", "芝", 1000)
        inner = resolve_course_profile("新潟", "芝", 1200)
        outer = resolve_course_profile("新潟", "芝", 1600)
        mixed = resolve_course_profile("新潟", "芝", 2000)

        self.assertEqual(straight.layout, "straight")
        self.assertEqual(inner.layout, "inner")
        self.assertEqual(outer.layout, "outer")
        self.assertEqual(mixed.layout, "mixed")
        self.assertGreater(
            outer.straight_length_m or 0.0,
            inner.straight_length_m or 0.0,
        )

    def test_similarity_distinguishes_physical_shape(self) -> None:
        exact = course_similarity(
            "東京", "芝", 1600, "東京", "芝", 1600
        )
        long_left = course_similarity(
            "東京", "芝", 1600, "新潟", "芝", 1600
        )
        short_right = course_similarity(
            "東京", "芝", 1600, "札幌", "芝", 1500
        )

        self.assertAlmostEqual(exact, 1.0)
        self.assertGreater(long_left, short_right)
        self.assertGreater(long_left, 0.5)

    def test_horse_fit_uses_within_horse_performance_residual(self) -> None:
        horse = HorseEntry(
            horse_no=1,
            horse_name="Geometry",
            recent_runs=[
                RecentRun(
                    course="東京",
                    surface="芝",
                    distance=1600,
                    finish=2,
                    field_size=16,
                ),
                RecentRun(
                    course="中京",
                    surface="芝",
                    distance=1600,
                    finish=3,
                    field_size=16,
                ),
                RecentRun(
                    course="札幌",
                    surface="芝",
                    distance=1500,
                    finish=12,
                    field_size=14,
                ),
                RecentRun(
                    course="函館",
                    surface="芝",
                    distance=1800,
                    finish=11,
                    field_size=14,
                ),
            ],
        )
        tokyo = RaceInfo(
            race_id="202605050101",
            course="東京",
            surface="芝",
            distance=1600,
        )
        sapporo = RaceInfo(
            race_id="202601010101",
            course="札幌",
            surface="芝",
            distance=1500,
        )

        tokyo_fit = horse_course_geometry_fit(horse, tokyo)
        sapporo_fit = horse_course_geometry_fit(horse, sapporo)

        self.assertGreater(tokyo_fit.shrunken_residual, 0.0)
        self.assertLess(sapporo_fit.shrunken_residual, 0.0)
        self.assertGreater(
            tokyo_fit.shrunken_residual,
            sapporo_fit.shrunken_residual,
        )

    def test_style_features_interact_with_target_shape(self) -> None:
        horse = HorseEntry(
            horse_no=2,
            horse_name="FrontRunner",
            recent_runs=[
                RecentRun(
                    course="中山",
                    surface="芝",
                    distance=1600,
                    finish=2,
                    field_size=16,
                    corner_positions=[2, 2, 2],
                ),
                RecentRun(
                    course="福島",
                    surface="芝",
                    distance=1800,
                    finish=3,
                    field_size=14,
                    corner_positions=[1, 1, 2, 2],
                ),
            ],
        )
        nakayama = RaceInfo(
            race_id="202606030101",
            course="中山",
            surface="芝",
            distance=1600,
        )
        niigata = RaceInfo(
            race_id="202604020101",
            course="新潟",
            surface="芝",
            distance=1600,
        )

        short_course = build_course_style_features(horse, nakayama)
        long_course = build_course_style_features(horse, niigata)

        self.assertGreater(
            short_course["course_style_short_straight_early"],
            long_course["course_style_short_straight_early"],
        )
        self.assertEqual(
            long_course["course_style_short_straight_early"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

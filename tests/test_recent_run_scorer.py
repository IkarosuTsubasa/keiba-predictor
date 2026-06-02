from __future__ import annotations

import unittest

from keiba_llm_agent.scoring.recent_run_scorer import score_horse_by_recent_runs
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun


def build_run(
    date: str,
    course: str,
    surface: str,
    distance: int,
    finish: int | None,
    field_size: int | None,
    jockey: str,
    track_condition: str = "良",
    odds: float | None = None,
    popularity: int | None = None,
) -> RecentRun:
    return RecentRun(
        date=date,
        course=course,
        surface=surface,
        distance=distance,
        finish=finish,
        field_size=field_size,
        jockey=jockey,
        track_condition=track_condition,
        odds=odds,
        popularity=popularity,
    )


class RecentRunScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.race_info = RaceInfo(
            race_id="sample_001",
            race_date="2026-05-18",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
        )

    def test_empty_recent_runs_sets_risk_to_minus_ten(self) -> None:
        horse = HorseEntry(horse_no=1, horse_name="A", jockey="騎手A", recent_runs=[])
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertEqual(score.scores.risk, -10)

    def test_good_recent_finishes_raise_recent_form(self) -> None:
        horse = HorseEntry(
            horse_no=1,
            horse_name="A",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-04-20", "東京", "芝", 1600, 1, 16, "騎手A"),
                build_run("2026-03-20", "中山", "芝", 1600, 2, 16, "騎手A"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.recent_form, 8)

    def test_same_distance_improves_distance_fit(self) -> None:
        horse = HorseEntry(
            horse_no=2,
            horse_name="B",
            jockey="騎手B",
            recent_runs=[
                build_run("2026-04-10", "阪神", "芝", 1600, 3, 18, "騎手B"),
                build_run("2026-03-15", "阪神", "芝", 1600, 4, 18, "騎手B"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.distance_fit, 8)

    def test_same_course_improves_course_fit(self) -> None:
        horse = HorseEntry(
            horse_no=3,
            horse_name="C",
            jockey="騎手C",
            recent_runs=[
                build_run("2026-04-12", "東京", "芝", 1400, 2, 18, "騎手C"),
                build_run("2026-03-12", "東京", "芝", 1800, 5, 18, "騎手C"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.course_fit, 7)

    def test_same_jockey_improves_jockey_fit(self) -> None:
        horse = HorseEntry(
            horse_no=4,
            horse_name="D",
            jockey="横山典",
            recent_runs=[
                build_run("2026-04-12", "東京", "芝", 1600, 2, 18, "横山典弘"),
                build_run("2026-03-12", "中山", "芝", 1600, 4, 18, "横山典弘"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.jockey_fit, 7)

    def test_total_score_sorting_prefers_stronger_horse(self) -> None:
        strong = HorseEntry(
            horse_no=1,
            horse_name="Strong",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-04-20", "東京", "芝", 1600, 1, 16, "騎手A", odds=12.0, popularity=8),
                build_run("2026-03-20", "東京", "芝", 1600, 2, 16, "騎手A", odds=8.0, popularity=4),
            ],
        )
        weak = HorseEntry(
            horse_no=2,
            horse_name="Weak",
            jockey="騎手B",
            recent_runs=[
                build_run("2026-04-20", "阪神", "芝", 1200, 10, 16, "騎手C", odds=2.2, popularity=1),
                build_run("2026-03-20", "阪神", "芝", 1200, 12, 16, "騎手D", odds=2.5, popularity=1),
            ],
        )
        strong_score = score_horse_by_recent_runs(strong, self.race_info)
        weak_score = score_horse_by_recent_runs(weak, self.race_info)
        self.assertGreater(strong_score.total_score, weak_score.total_score)

    def test_market_signal_disabled_by_default_does_not_change_base_score(self) -> None:
        base_runs = [
            build_run("2026-04-20", "東京", "芝", 1600, 2, 16, "騎手A"),
            build_run("2026-03-20", "東京", "芝", 1600, 3, 16, "騎手A"),
        ]
        favorite = HorseEntry(
            horse_no=11,
            horse_name="Favorite",
            jockey="騎手A",
            odds=2.4,
            popularity=1,
            recent_runs=base_runs,
        )
        outsider = HorseEntry(
            horse_no=12,
            horse_name="Outsider",
            jockey="騎手A",
            odds=25.0,
            popularity=11,
            recent_runs=base_runs,
        )
        favorite_score = score_horse_by_recent_runs(favorite, self.race_info)
        outsider_score = score_horse_by_recent_runs(outsider, self.race_info)
        self.assertNotEqual(favorite_score.scores.odds_value, outsider_score.scores.odds_value)
        self.assertEqual(favorite_score.total_score, outsider_score.total_score)

    def test_market_signal_can_be_explicitly_enabled(self) -> None:
        base_runs = [
            build_run("2026-04-20", "東京", "芝", 1600, 2, 16, "騎手A"),
            build_run("2026-03-20", "東京", "芝", 1600, 3, 16, "騎手A"),
        ]
        outsider = HorseEntry(
            horse_no=13,
            horse_name="Outsider",
            jockey="騎手A",
            odds=25.0,
            popularity=11,
            recent_runs=base_runs,
        )
        baseline_score = score_horse_by_recent_runs(outsider, self.race_info)
        market_enabled_score = score_horse_by_recent_runs(
            outsider,
            self.race_info,
            use_market_score_in_ranking=True,
            market_signal_weight=0.7,
        )
        self.assertGreater(market_enabled_score.total_score, baseline_score.total_score)

    def test_risk_stays_in_required_range(self) -> None:
        horse = HorseEntry(
            horse_no=5,
            horse_name="E",
            jockey="騎手E",
            recent_runs=[build_run("2026-04-20", "東京", "芝", 1600, 8, 16, "騎手E")],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertLessEqual(score.scores.risk, 0)
        self.assertGreaterEqual(score.scores.risk, -10)

    def test_reason_does_not_say_distance_fit_when_same_distance_good_count_is_zero(self) -> None:
        horse = HorseEntry(
            horse_no=6,
            horse_name="F",
            jockey="騎手F",
            recent_runs=[
                build_run("2026-04-20", "阪神", "芝", 1600, 8, 16, "騎手F"),
                build_run("2026-03-20", "中山", "芝", 1600, 7, 16, "騎手F"),
                build_run("2026-02-20", "東京", "芝", 1400, 6, 16, "騎手F"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertNotIn("好走歴0回があり", score.reason)
        self.assertIn("同距離1600mでの好走歴はまだなく", score.reason)

    def test_reason_does_not_say_course_fit_when_same_course_good_count_is_zero(self) -> None:
        horse = HorseEntry(
            horse_no=7,
            horse_name="G",
            jockey="騎手G",
            recent_runs=[
                build_run("2026-04-20", "東京", "芝", 1400, 7, 16, "騎手G"),
                build_run("2026-03-20", "東京", "芝", 1800, 8, 16, "騎手G"),
                build_run("2026-02-20", "中山", "芝", 1600, 9, 16, "騎手G"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertNotIn("好走0回を評価", score.reason)
        self.assertIn("東京では出走歴があるが、好走歴はまだなく", score.reason)

    def test_reason_can_normally_evaluate_when_good_count_positive(self) -> None:
        horse = HorseEntry(
            horse_no=8,
            horse_name="H",
            jockey="横山典",
            recent_runs=[
                build_run("2026-04-20", "東京", "芝", 1600, 2, 16, "横山典弘"),
                build_run("2026-03-20", "東京", "芝", 1600, 3, 16, "横山典弘"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertIn("同距離1600mで好走歴", score.reason)


if __name__ == "__main__":
    unittest.main()

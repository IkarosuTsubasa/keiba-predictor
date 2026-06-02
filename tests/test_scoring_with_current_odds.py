from __future__ import annotations

import unittest

from keiba_llm_agent.scoring.recent_run_scorer import score_horse_by_recent_runs
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun


def build_recent_run(
    *,
    date: str = "2026-04-20",
    course: str = "東京",
    surface: str = "芝",
    distance: int = 1600,
    finish: int | None = 2,
    field_size: int | None = 16,
    jockey: str = "騎手A",
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


class ScoringWithCurrentOddsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.race_info = RaceInfo(
            race_id="sample_001",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
        )

    def test_high_current_odds_and_good_form_raise_odds_value(self) -> None:
        horse = HorseEntry(
            horse_no=1,
            horse_name="A",
            jockey="騎手A",
            odds=25.0,
            recent_runs=[
                build_recent_run(finish=1, field_size=16),
                build_recent_run(date="2026-03-20", finish=3, field_size=16),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertEqual(score.scores.odds_value, 9)

    def test_short_current_odds_do_not_overrate_even_with_good_form(self) -> None:
        horse = HorseEntry(
            horse_no=2,
            horse_name="B",
            jockey="騎手B",
            odds=2.4,
            recent_runs=[
                build_recent_run(finish=1, field_size=16, jockey="騎手B"),
                build_recent_run(date="2026-03-20", finish=2, field_size=16, jockey="騎手B"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertEqual(score.scores.odds_value, 4)

    def test_null_current_odds_uses_history_fallback(self) -> None:
        horse = HorseEntry(
            horse_no=3,
            horse_name="C",
            jockey="騎手C",
            odds=None,
            recent_runs=[
                build_recent_run(finish=2, field_size=16, odds=24.0, popularity=9),
                build_recent_run(date="2026-03-20", finish=6, field_size=16, odds=8.0, popularity=4),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertEqual(score.scores.odds_value, 8)


if __name__ == "__main__":
    unittest.main()

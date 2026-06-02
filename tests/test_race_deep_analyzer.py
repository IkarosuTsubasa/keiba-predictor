from __future__ import annotations

import unittest

from keiba_llm_agent.analysis.race_deep_analyzer import analyze_horse_deeply
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo


def _race_info() -> RaceInfo:
    return RaceInfo(
        race_id="sample",
        race_name="Sample Stakes",
        race_date="2026-05-17",
        course="東京",
        surface="芝",
        distance=1600,
        track_condition="良",
        weather="晴",
    )


def _horse(recent_runs: list[dict], odds: float | None = 12.5, popularity: int | None = 6, jockey: str = "横山典") -> HorseEntry:
    return HorseEntry.model_validate(
        {
            "horse_no": 16,
            "frame_no": 8,
            "horse_id": "horse_016",
            "horse_name": "ニシノティアモ",
            "jockey": jockey,
            "carried_weight": 55.0,
            "odds": odds,
            "popularity": popularity,
            "recent_runs": recent_runs,
        }
    )


class RaceDeepAnalyzerTests(unittest.TestCase):
    def test_recent_runs_empty_adds_data_incomplete(self) -> None:
        analysis = analyze_horse_deeply(_horse([], odds=None, popularity=None), _race_info())
        self.assertIn("DATA_INCOMPLETE", analysis.risk_flags)

    def test_strong_recent_form_and_distance_course_track_jockey_flags(self) -> None:
        analysis = analyze_horse_deeply(
            _horse(
                [
                    {"date": "2026-04-20", "course": "東京", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 1, "field_size": 16, "jockey": "横山典弘", "odds": 15.0, "popularity": 6},
                    {"date": "2026-03-20", "course": "東京", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 2, "field_size": 16, "jockey": "横山典弘", "odds": 18.0, "popularity": 7},
                    {"date": "2026-02-20", "course": "中山", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 3, "field_size": 16, "jockey": "横山典弘", "odds": 22.0, "popularity": 9},
                    {"date": "2026-01-20", "course": "中山", "surface": "芝", "distance": 1600, "track_condition": "稍重", "finish": 6, "field_size": 16, "jockey": "戸崎圭太", "odds": 5.0, "popularity": 3},
                    {"date": "2025-12-20", "course": "東京", "surface": "芝", "distance": 1800, "track_condition": "良", "finish": 1, "field_size": 14, "jockey": "横山典弘", "odds": 9.0, "popularity": 4},
                ],
                odds=14.9,
                popularity=4,
            ),
            _race_info(),
        )
        self.assertIn("RECENT_FORM_STRONG", analysis.positive_flags)
        self.assertIn("DISTANCE_FIT", analysis.positive_flags)
        self.assertIn("COURSE_FIT", analysis.positive_flags)
        self.assertIn("TRACK_CONDITION_FIT", analysis.positive_flags)
        self.assertIn("JOCKEY_CONTINUITY", analysis.positive_flags)
        self.assertIn("VALUE_CANDIDATE", analysis.positive_flags)

    def test_weak_recent_form_marks_recent_form_weak(self) -> None:
        analysis = analyze_horse_deeply(
            _horse(
                [
                    {"date": "2026-04-20", "course": "阪神", "surface": "芝", "distance": 1400, "track_condition": "重", "finish": 12, "field_size": 16, "jockey": "川田将雅"},
                    {"date": "2026-03-20", "course": "京都", "surface": "芝", "distance": 1400, "track_condition": "良", "finish": 11, "field_size": 16, "jockey": "岩田望来"},
                    {"date": "2026-02-20", "course": "中京", "surface": "芝", "distance": 1400, "track_condition": "稍重", "finish": 8, "field_size": 18, "jockey": "岩田望来"},
                ],
                odds=2.4,
                popularity=1,
                jockey="川田",
            ),
            _race_info(),
        )
        self.assertIn("RECENT_FORM_WEAK", analysis.risk_flags)
        self.assertIn("OVERBET_RISK", analysis.risk_flags)


if __name__ == "__main__":
    unittest.main()

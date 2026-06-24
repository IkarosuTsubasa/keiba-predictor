from __future__ import annotations

import unittest

from keiba_llm_agent.analysis.race_level_analyzer import analyze_race_level_for_horse
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo


def _race_info() -> RaceInfo:
    return RaceInfo(
        race_id="target",
        race_name="target",
        race_date="2026-05-17",
        course="京都",
        surface="芝",
        distance=2400,
        track_condition="良",
        weather="晴",
    )


class RaceLevelAnalyzerTests(unittest.TestCase):
    def test_head_to_head_and_race_level_flags(self) -> None:
        horse_a = HorseEntry.model_validate(
            {
                "horse_no": 1,
                "frame_no": 1,
                "horse_id": "a",
                "horse_name": "A",
                "jockey": "騎手A",
                "carried_weight": 56.0,
                "odds": 10.0,
                "popularity": 5,
                "recent_runs": [
                    {
                        "race_id": "202601010101",
                        "date": "2026-04-20",
                        "course": "阪神",
                        "surface": "芝",
                        "distance": 2400,
                        "track_condition": "良",
                        "race_name": "京都新聞杯(G2)",
                        "finish": 2,
                        "field_size": 16,
                        "jockey": "騎手A",
                        "margin": "0.4",
                        "odds": 12.3,
                        "popularity": 8,
                    }
                ],
            }
        )
        horse_b = HorseEntry.model_validate(
            {
                "horse_no": 2,
                "frame_no": 2,
                "horse_id": "b",
                "horse_name": "B",
                "jockey": "騎手B",
                "carried_weight": 56.0,
                "odds": 4.0,
                "popularity": 2,
                "recent_runs": [
                    {
                        "race_id": "202601010101",
                        "date": "2026-04-20",
                        "course": "阪神",
                        "surface": "芝",
                        "distance": 2400,
                        "track_condition": "良",
                        "finish": 5,
                        "field_size": 16,
                        "jockey": "騎手B",
                        "odds": 3.2,
                        "popularity": 2,
                    }
                ],
            }
        )
        analysis = analyze_race_level_for_horse(horse_a, [horse_a, horse_b], _race_info())
        self.assertIn("HEAD_TO_HEAD_POSITIVE", analysis.positive_flags)
        self.assertIn("LARGE_FIELD_GOOD_RUN", analysis.positive_flags)
        self.assertIn("GRADED_SMALL_MARGIN", analysis.positive_flags)
        self.assertNotIn("UNDERVALUED_GOOD_RUN", analysis.positive_flags)
        self.assertLessEqual(analysis.adjustment_hint, 1.5)

    def test_head_to_head_negative_and_recent_level_risk(self) -> None:
        horse_a = HorseEntry.model_validate(
            {
                "horse_no": 1,
                "frame_no": 1,
                "horse_id": "a",
                "horse_name": "A",
                "jockey": "騎手A",
                "carried_weight": 56.0,
                "odds": 2.1,
                "popularity": 1,
                "recent_runs": [
                    {
                        "race_id": "202601010102",
                        "date": "2026-04-20",
                        "course": "阪神",
                        "surface": "芝",
                        "distance": 2000,
                        "track_condition": "良",
                        "finish": 10,
                        "field_size": 16,
                        "jockey": "騎手A",
                        "odds": 2.1,
                        "popularity": 1,
                    },
                    {
                        "race_id": "202601010103",
                        "date": "2026-03-10",
                        "course": "阪神",
                        "surface": "芝",
                        "distance": 1800,
                        "track_condition": "良",
                        "finish": 12,
                        "field_size": 16,
                        "jockey": "騎手A",
                        "odds": 2.5,
                        "popularity": 2,
                    },
                ],
            }
        )
        horse_b = HorseEntry.model_validate(
            {
                "horse_no": 2,
                "frame_no": 2,
                "horse_id": "b",
                "horse_name": "B",
                "jockey": "騎手B",
                "carried_weight": 56.0,
                "odds": 7.0,
                "popularity": 4,
                "recent_runs": [
                    {
                        "race_id": "202601010102",
                        "date": "2026-04-20",
                        "course": "阪神",
                        "surface": "芝",
                        "distance": 2000,
                        "track_condition": "良",
                        "finish": 3,
                        "field_size": 16,
                        "jockey": "騎手B",
                        "odds": 7.0,
                        "popularity": 4,
                    }
                ],
            }
        )
        analysis = analyze_race_level_for_horse(horse_a, [horse_a, horse_b], _race_info())
        self.assertIn("HEAD_TO_HEAD_NEGATIVE", analysis.risk_flags)
        self.assertIn("WEAK_RECENT_LEVEL", analysis.risk_flags)
        self.assertNotIn("POPULAR_DISAPPOINTMENT", analysis.risk_flags)
        self.assertGreaterEqual(analysis.adjustment_hint, -1.5)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from keiba_llm_agent.analysis.pace_analyzer import analyze_horse_pace, analyze_pace_for_race
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo


def _race_info(course: str = "京都", distance: int = 2400) -> RaceInfo:
    return RaceInfo(
        race_id="sample",
        race_name="sample",
        race_date="2026-05-17",
        course=course,
        surface="芝",
        distance=distance,
        track_condition="良",
        weather="晴",
    )


def _horse(horse_no: int, horse_name: str, passing_orders: list[str]) -> HorseEntry:
    recent_runs = []
    for index, order in enumerate(passing_orders, start=1):
        positions = [int(part) for part in order.split("-")]
        recent_runs.append(
            {
                "race_id": f"20260101010{index}",
                "date": "2026-04-20",
                "course": "阪神",
                "surface": "芝",
                "distance": 2400,
                "track_condition": "良",
                "finish": positions[-1],
                "field_size": 16,
                "jockey": "騎手A",
                "odds": 10.0,
                "popularity": 5,
                "passing_order": order,
                "corner_positions": positions,
                "final_3f": 34.5 + index * 0.1,
            }
        )
    return HorseEntry.model_validate(
        {
            "horse_no": horse_no,
            "frame_no": horse_no,
            "horse_id": f"h{horse_no}",
            "horse_name": horse_name,
            "jockey": "騎手A",
            "carried_weight": 56.0,
            "odds": 5.0,
            "popularity": 3,
            "recent_runs": recent_runs,
        }
    )


class PaceAnalyzerTests(unittest.TestCase):
    def test_running_style_front(self) -> None:
        analysis = analyze_horse_pace(_horse(1, "A", ["1-1-1-1"]), _race_info())
        self.assertIn(analysis.running_style, {"逃げ", "先行"})

    def test_running_style_stalker(self) -> None:
        analysis = analyze_horse_pace(_horse(2, "B", ["4-4-5-4"]), _race_info())
        self.assertEqual(analysis.running_style, "先行")

    def test_running_style_closer(self) -> None:
        analysis = analyze_horse_pace(_horse(3, "C", ["10-10-9-8"]), _race_info())
        self.assertIn(analysis.running_style, {"差し", "追込"})

    def test_fast_projection_when_front_runners_many(self) -> None:
        horses = [
            _horse(1, "A", ["1-1-1-1"]),
            _horse(2, "B", ["1-2-1-2"]),
            _horse(3, "C", ["2-1-2-3"]),
            _horse(4, "D", ["10-10-9-8"]),
        ]
        _, projection = analyze_pace_for_race(horses, _race_info())
        self.assertEqual(projection.projected_pace, "fast")
        self.assertEqual(projection.favorable_styles, ["差し", "追込"])

    def test_fast_projection_when_two_front_and_many_stalkers(self) -> None:
        horses = [
            _horse(1, "A", ["1-1-1-1"]),
            _horse(2, "B", ["2-2-2-2"]),
            _horse(3, "C", ["4-4-4-4"]),
            _horse(4, "D", ["4-5-4-5"]),
            _horse(5, "E", ["5-5-5-5"]),
            _horse(6, "F", ["5-4-5-4"]),
        ]
        _, projection = analyze_pace_for_race(horses, _race_info("東京", 1600))
        self.assertEqual(projection.projected_pace, "fast")

    def test_slow_projection_when_front_absent(self) -> None:
        horses = [
            _horse(1, "A", ["10-10-9-8"]),
            _horse(2, "B", ["11-11-10-9"]),
            _horse(3, "C", ["12-12-11-10"]),
        ]
        _, projection = analyze_pace_for_race(horses, _race_info())
        self.assertEqual(projection.projected_pace, "slow")

    def test_kyoto_2400_comment_mentions_stalker_and_sustain(self) -> None:
        horses = [
            _horse(1, "A", ["4-4-5-4"]),
            _horse(2, "B", ["10-10-9-8"]),
        ]
        _, projection = analyze_pace_for_race(horses, _race_info("京都", 2400))
        self.assertIn("先行", projection.pace_comment)
        self.assertIn("持続力", projection.pace_comment)

    def test_kyoto_2400_with_one_front_and_three_stalkers_is_not_fast(self) -> None:
        horses = [
            _horse(1, "A", ["1-1-1-1"]),
            _horse(2, "B", ["4-4-5-4"]),
            _horse(3, "C", ["4-5-4-5"]),
            _horse(4, "D", ["5-5-5-4"]),
            _horse(5, "E", ["10-10-9-8"]),
            _horse(6, "F", ["11-11-10-9"]),
            _horse(7, "G", ["12-12-11-10"]),
            _horse(8, "H", ["13-13-12-11"]),
            _horse(9, "I", ["14-14-13-12"]),
            _horse(10, "J", ["15-15-14-13"]),
        ]
        _, projection = analyze_pace_for_race(horses, _race_info("京都", 2400))
        self.assertNotEqual(projection.projected_pace, "fast")
        self.assertNotIn("前が多く", projection.pace_comment)
        self.assertEqual(projection.favorable_styles, ["先行", "差し"])

    def test_data_missing_sets_incomplete_flag(self) -> None:
        horse = HorseEntry.model_validate(
            {
                "horse_no": 9,
                "frame_no": 9,
                "horse_id": "hx",
                "horse_name": "X",
                "jockey": "騎手X",
                "carried_weight": 56.0,
                "odds": 8.0,
                "popularity": 4,
                "recent_runs": [],
            }
        )
        analysis = analyze_horse_pace(horse, _race_info())
        self.assertIn("PACE_DATA_INCOMPLETE", analysis.risk_flags)


if __name__ == "__main__":
    unittest.main()

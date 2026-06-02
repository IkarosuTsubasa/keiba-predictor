from __future__ import annotations

import unittest

from keiba_llm_agent.policies.bet_policy import build_bets_from_strategy, evaluate_bet_strategy
from keiba_llm_agent.schemas.prediction import HorseScore, ScoreBreakdown
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo
from keiba_llm_agent.schemas.review import LessonItem


def build_horse_score(horse_no: int, total_score: float, odds_value: int = 5, risk: int = -3) -> HorseScore:
    return HorseScore(
        horse_no=horse_no,
        horse_name=f"Horse{horse_no}",
        total_score=total_score,
        reason="reason",
        scores=ScoreBreakdown(
            recent_form=7,
            distance_fit=7,
            course_fit=7,
            track_condition_fit=7,
            jockey_fit=7,
            odds_value=odds_value,
            risk=risk,
        ),
    )


class BetPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.race_info = RaceInfo(
            race_id="sample_001",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
        )
        self.lesson = LessonItem(
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
            lesson="同条件では近走の同距離・同コース実績を優先して評価する。",
            confidence="medium",
            source_race_id="source_001",
        )

    def build_race_data(self, odds_list: list[float | None]) -> RaceData:
        horses = []
        for index, odds in enumerate(odds_list, start=1):
            horses.append(
                HorseEntry(
                    horse_no=index,
                    horse_name=f"Horse{index}",
                    odds=odds,
                    popularity=index if odds is not None else None,
                )
            )
        return RaceData(race_info=self.race_info, horses=horses)

    def test_low_top_score_results_in_skip(self) -> None:
        race_data = self.build_race_data([12.0, 14.0, 16.0])
        strategy = evaluate_bet_strategy(
            race_data,
            [build_horse_score(1, 31.0), build_horse_score(2, 30.0), build_horse_score(3, 29.0)],
            {"◎": 1, "○": 2, "▲": 3, "△": 0, "☆": 0},
            [],
            ["heuristic scoringを使用しており、正式なML modelではない。"],
        )
        self.assertEqual(strategy.bet_decision, "SKIP")

    def test_high_risk_top_results_in_skip(self) -> None:
        race_data = self.build_race_data([12.0, 14.0, 16.0])
        strategy = evaluate_bet_strategy(
            race_data,
            [build_horse_score(1, 40.0, risk=-7), build_horse_score(2, 35.0), build_horse_score(3, 34.0)],
            {"◎": 1, "○": 2, "▲": 3, "△": 0, "☆": 0},
            [],
            [],
        )
        self.assertEqual(strategy.bet_decision, "SKIP")

    def test_all_odds_missing_results_in_skip(self) -> None:
        race_data = self.build_race_data([None, None, None])
        strategy = evaluate_bet_strategy(
            race_data,
            [build_horse_score(1, 40.0), build_horse_score(2, 37.0), build_horse_score(3, 36.0)],
            {"◎": 1, "○": 2, "▲": 3, "△": 0, "☆": 0},
            [],
            ["一部の馬でoddsまたは人気が欠損している。"],
        )
        self.assertEqual(strategy.bet_decision, "SKIP")
        self.assertIn("ODDS_MISSING", strategy.reason_codes)

    def test_high_top_score_and_odds_results_in_bet(self) -> None:
        race_data = self.build_race_data([14.9, 23.3, 49.4])
        strategy = evaluate_bet_strategy(
            race_data,
            [build_horse_score(1, 42.8), build_horse_score(2, 41.2), build_horse_score(3, 41.2)],
            {"◎": 1, "○": 2, "▲": 3, "△": 0, "☆": 0},
            [self.lesson],
            ["heuristic scoringを使用しており、正式なML modelではない。"],
        )
        self.assertEqual(strategy.bet_decision, "BET")
        self.assertIn(strategy.confidence, {"medium", "high"})

    def test_close_top_group_adds_reason_code(self) -> None:
        race_data = self.build_race_data([14.9, 23.3, 49.4])
        strategy = evaluate_bet_strategy(
            race_data,
            [build_horse_score(1, 42.8), build_horse_score(2, 41.2), build_horse_score(3, 41.2)],
            {"◎": 1, "○": 2, "▲": 3, "△": 0, "☆": 0},
            [],
            [],
        )
        self.assertIn("CLOSE_TOP_GROUP", strategy.reason_codes)

    def test_lessons_used_adds_reason_code(self) -> None:
        race_data = self.build_race_data([14.9, 23.3, 49.4])
        strategy = evaluate_bet_strategy(
            race_data,
            [build_horse_score(1, 42.8), build_horse_score(2, 41.2), build_horse_score(3, 41.2)],
            {"◎": 1, "○": 2, "▲": 3, "△": 0, "☆": 0},
            [self.lesson],
            [],
        )
        self.assertIn("LESSON_USED", strategy.reason_codes)

    def test_skip_strategy_results_in_empty_bets(self) -> None:
        strategy = evaluate_bet_strategy(
            self.build_race_data([None, None, None]),
            [build_horse_score(1, 28.0), build_horse_score(2, 27.0), build_horse_score(3, 26.0)],
            {"◎": 1, "○": 2, "▲": 3, "△": 0, "☆": 0},
            [],
            ["一部の馬でoddsまたは人気が欠損している。"],
        )
        bets = build_bets_from_strategy(strategy, [build_horse_score(1, 28.0), build_horse_score(2, 27.0)])
        self.assertEqual(strategy.bet_decision, "SKIP")
        self.assertEqual(bets, [])


if __name__ == "__main__":
    unittest.main()

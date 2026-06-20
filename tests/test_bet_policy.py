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

    def build_race_data(
        self,
        odds_list: list[float | None],
        *,
        surface: str | None = None,
        scope_key: str | None = None,
        source: str | None = None,
    ) -> RaceData:
        race_info = self.race_info.model_copy(
            update={
                "surface": surface or self.race_info.surface,
                "scope_key": scope_key,
                "source": source,
            }
        )
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
        return RaceData(race_info=race_info, horses=horses)

    def test_low_top_score_results_in_skip(self) -> None:
        race_data = self.build_race_data([12.0, 14.0, 16.0, 18.0, 20.0, 22.0])
        strategy = evaluate_bet_strategy(
            race_data,
            [
                build_horse_score(1, 31.0),
                build_horse_score(2, 30.0),
                build_horse_score(3, 29.0),
                build_horse_score(4, 28.0),
                build_horse_score(5, 27.0),
                build_horse_score(6, 26.8),
            ],
            {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
            [],
            ["heuristic scoringを使用しており、正式なML modelではない。"],
        )
        self.assertEqual(strategy.bet_decision, "SKIP")
        self.assertGreaterEqual(strategy.confidence_score, 0.0)

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

    def test_all_odds_missing_can_still_rate_by_model_scores(self) -> None:
        race_data = self.build_race_data([None, None, None])
        strategy = evaluate_bet_strategy(
            race_data,
            [build_horse_score(1, 40.0), build_horse_score(2, 37.0), build_horse_score(3, 36.0)],
            {"◎": 1, "○": 2, "▲": 3, "△": 0, "☆": 0},
            [],
            [],
        )
        self.assertEqual(strategy.bet_decision, "BET")
        self.assertIn("MARKET_DATA_UNAVAILABLE", strategy.reason_codes)
        self.assertIn("市場オッズを使わず", strategy.reason)

    def test_high_top_score_and_odds_results_in_bet(self) -> None:
        race_data = self.build_race_data([14.9, 23.3, 49.4, 16.0, 18.0, 20.0])
        strategy = evaluate_bet_strategy(
            race_data,
            [
                build_horse_score(1, 42.8),
                build_horse_score(2, 41.2),
                build_horse_score(3, 41.2),
                build_horse_score(4, 39.8),
                build_horse_score(5, 38.8),
                build_horse_score(6, 38.7),
            ],
            {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
            [self.lesson],
            ["heuristic scoringを使用しており、正式なML modelではない。"],
        )
        self.assertEqual(strategy.bet_decision, "BET")
        self.assertEqual(strategy.confidence, "low")
        self.assertGreaterEqual(strategy.confidence_score, 0.46)
        self.assertLess(strategy.confidence_score, 0.70)

    def test_central_dirt_clear_marks_can_use_high_pair_coverage_confidence(self) -> None:
        race_data = self.build_race_data([4.0, 8.0, 12.0, 16.0, 18.0, 20.0], surface="ダート", scope_key="central_dirt")
        strategy = evaluate_bet_strategy(
            race_data,
            [
                build_horse_score(1, 50.0),
                build_horse_score(2, 45.0),
                build_horse_score(3, 39.0),
                build_horse_score(4, 35.0),
                build_horse_score(5, 33.0),
                build_horse_score(6, 25.0),
            ],
            {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
            [],
            [],
        )
        self.assertEqual(strategy.bet_decision, "BET")
        self.assertEqual(strategy.confidence, "high")
        self.assertGreaterEqual(strategy.confidence_score, 0.85)
        self.assertLessEqual(strategy.confidence_score, 0.90)

    def test_central_turf_coverage_confidence_uses_fifth_mark_boundary(self) -> None:
        race_data = self.build_race_data([3.5, 8.0, 12.0, 16.0, 18.0, 20.0], surface="芝", scope_key="central_turf")
        weak_boundary = evaluate_bet_strategy(
            race_data,
            [
                build_horse_score(1, 45.0),
                build_horse_score(2, 44.0),
                build_horse_score(3, 43.5),
                build_horse_score(4, 43.0),
                build_horse_score(5, 42.5),
                build_horse_score(6, 42.4),
            ],
            {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
            [],
            [],
        )
        clear_boundary = evaluate_bet_strategy(
            race_data,
            [
                build_horse_score(1, 55.0, risk=-1),
                build_horse_score(2, 48.0),
                build_horse_score(3, 42.0),
                build_horse_score(4, 38.0),
                build_horse_score(5, 34.0),
                build_horse_score(6, 28.0),
            ],
            {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
            [],
            [],
        )
        self.assertEqual(weak_boundary.confidence, "low")
        self.assertEqual(clear_boundary.confidence, "medium")
        self.assertGreaterEqual(clear_boundary.confidence_score, 0.70)
        self.assertLess(clear_boundary.confidence_score, 0.85)

    def test_local_clear_axis_can_still_use_high_confidence(self) -> None:
        race_data = self.build_race_data([4.0, 8.0, 12.0, 16.0, 18.0, 20.0], surface="ダート", scope_key="local", source="local")
        strategy = evaluate_bet_strategy(
            race_data,
            [
                build_horse_score(1, 45.0),
                build_horse_score(2, 40.0),
                build_horse_score(3, 36.0),
                build_horse_score(4, 34.0),
                build_horse_score(5, 32.0),
                build_horse_score(6, 29.0),
            ],
            {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
            [],
            [],
        )
        self.assertEqual(strategy.confidence, "high")
        self.assertGreaterEqual(strategy.confidence_score, 0.85)
        self.assertLessEqual(strategy.confidence_score, 0.90)
        self.assertIn("confidence_score", strategy.model_dump())

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
            [],
        )
        bets = build_bets_from_strategy(strategy, [build_horse_score(1, 28.0), build_horse_score(2, 27.0)])
        self.assertEqual(strategy.bet_decision, "SKIP")
        self.assertEqual(bets, [])


if __name__ == "__main__":
    unittest.main()

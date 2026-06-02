from __future__ import annotations

import unittest
from types import SimpleNamespace

from keiba_llm_agent.config.scoring_config import ScoringConfig
from keiba_llm_agent.scoring.borderline_recovery import apply_top5_borderline_recovery
from keiba_llm_agent.schemas.prediction import HorseScore, ScoreBreakdown


def _horse_score(horse_no: int, score: float, popularity: int, odds: float) -> HorseScore:
    return HorseScore(
        horse_no=horse_no,
        horse_name=f"Horse{horse_no}",
        scores=ScoreBreakdown(
            recent_form=7,
            distance_fit=7,
            course_fit=7,
            track_condition_fit=7,
            jockey_fit=7,
            odds_value=5,
            risk=-2,
        ),
        odds=odds,
        popularity=popularity,
        total_score=score,
        reason="reason",
    )


def _analysis(horse_no: int, positive: list[str] | None = None, risk: list[str] | None = None) -> dict:
    return SimpleNamespace(
        horse_no=horse_no,
        horse_name=f"Horse{horse_no}",
        positive_flags=positive or [],
        risk_flags=risk or [],
        overall_comment="",
        head_to_head_summary="",
        race_level_summary="",
        opponent_context_summary="",
        adjustment_hint=0.0,
        running_style="先行",
        position_stability="安定",
    )


class BorderlineRecoveryTests(unittest.TestCase):
    def test_rank6_gap_with_signals_is_recovered(self) -> None:
        horse_scores = [
            _horse_score(1, 40.0, 1, 2.0),
            _horse_score(2, 39.0, 2, 3.0),
            _horse_score(3, 38.0, 3, 4.0),
            _horse_score(4, 37.0, 4, 5.0),
            _horse_score(5, 36.0, 5, 9.0),
            _horse_score(6, 35.4, 3, 8.0),
        ]
        result = apply_top5_borderline_recovery(
            horse_scores,
            deep_analyses=[_analysis(6, positive=["RECENT_FORM_STRONG"])],
            pedigree_analyses=[],
            race_level_analyses=[_analysis(6, positive=["HEAD_TO_HEAD_POSITIVE"])],
            pace_analyses=[],
            race_info=None,
            scoring_config=ScoringConfig(),
            enabled=True,
        )
        self.assertTrue(result["recovery_applied"])
        self.assertEqual(result["recovery_cases"][0]["horse_no"], 6)
        self.assertIn(6, result["recovered_top5"])
        self.assertLessEqual(result["recovery_cases"][0]["new_rank"], 5)

    def test_rank7_is_not_recovered(self) -> None:
        horse_scores = [
            _horse_score(1, 40.0, 1, 2.0),
            _horse_score(2, 39.0, 2, 3.0),
            _horse_score(3, 38.0, 3, 4.0),
            _horse_score(4, 37.0, 4, 5.0),
            _horse_score(5, 36.0, 5, 9.0),
            _horse_score(6, 35.5, 6, 12.0),
            _horse_score(7, 35.4, 3, 8.0),
        ]
        result = apply_top5_borderline_recovery(
            horse_scores,
            deep_analyses=[_analysis(7, positive=["RECENT_FORM_STRONG"])],
            pedigree_analyses=[],
            race_level_analyses=[_analysis(7, positive=["HEAD_TO_HEAD_POSITIVE"])],
            pace_analyses=[],
            race_info=None,
            scoring_config=ScoringConfig(),
            enabled=True,
        )
        self.assertFalse(result["recovery_applied"])

    def test_gap_above_one_is_not_recovered(self) -> None:
        horse_scores = [
            _horse_score(1, 40.0, 1, 2.0),
            _horse_score(2, 39.0, 2, 3.0),
            _horse_score(3, 38.0, 3, 4.0),
            _horse_score(4, 37.0, 4, 5.0),
            _horse_score(5, 36.5, 5, 9.0),
            _horse_score(6, 35.0, 3, 8.0),
        ]
        result = apply_top5_borderline_recovery(
            horse_scores,
            deep_analyses=[_analysis(6, positive=["RECENT_FORM_STRONG"])],
            pedigree_analyses=[],
            race_level_analyses=[_analysis(6, positive=["HEAD_TO_HEAD_POSITIVE"])],
            pace_analyses=[],
            race_info=None,
            scoring_config=ScoringConfig(),
            enabled=True,
        )
        self.assertFalse(result["recovery_applied"])

    def test_signal_below_two_is_not_recovered(self) -> None:
        horse_scores = [
            _horse_score(1, 40.0, 1, 2.0),
            _horse_score(2, 39.0, 2, 3.0),
            _horse_score(3, 38.0, 3, 4.0),
            _horse_score(4, 37.0, 4, 5.0),
            _horse_score(5, 36.0, 5, 9.0),
            _horse_score(6, 35.4, 8, 20.0),
        ]
        result = apply_top5_borderline_recovery(
            horse_scores,
            deep_analyses=[],
            pedigree_analyses=[],
            race_level_analyses=[],
            pace_analyses=[],
            race_info=None,
            scoring_config=ScoringConfig(),
            enabled=True,
        )
        self.assertFalse(result["recovery_applied"])

    def test_only_one_horse_is_recovered_per_race(self) -> None:
        horse_scores = [
            _horse_score(1, 40.0, 1, 2.0),
            _horse_score(2, 39.0, 2, 3.0),
            _horse_score(3, 38.0, 3, 4.0),
            _horse_score(4, 37.0, 4, 5.0),
            _horse_score(5, 36.0, 5, 9.0),
            _horse_score(6, 35.4, 3, 8.0),
            _horse_score(7, 35.3, 2, 6.0),
        ]
        result = apply_top5_borderline_recovery(
            horse_scores,
            deep_analyses=[_analysis(6, positive=["RECENT_FORM_STRONG"]), _analysis(7, positive=["RECENT_FORM_STRONG"])],
            pedigree_analyses=[],
            race_level_analyses=[_analysis(6, positive=["HEAD_TO_HEAD_POSITIVE"]), _analysis(7, positive=["HEAD_TO_HEAD_POSITIVE"])],
            pace_analyses=[],
            race_info=None,
            scoring_config=ScoringConfig(),
            enabled=True,
        )
        self.assertEqual(len(result["recovery_cases"]), 1)

    def test_popular_and_race_level_signals_are_recorded(self) -> None:
        horse_scores = [
            _horse_score(1, 40.0, 1, 2.0),
            _horse_score(2, 39.0, 2, 3.0),
            _horse_score(3, 38.0, 3, 4.0),
            _horse_score(4, 37.0, 4, 5.0),
            _horse_score(5, 36.0, 5, 9.0),
            _horse_score(6, 35.4, 3, 8.0),
        ]
        result = apply_top5_borderline_recovery(
            horse_scores,
            deep_analyses=[],
            pedigree_analyses=[],
            race_level_analyses=[_analysis(6, positive=["HEAD_TO_HEAD_POSITIVE"])],
            pace_analyses=[],
            race_info=None,
            scoring_config=ScoringConfig(),
            enabled=True,
        )
        reasons = result["recovery_cases"][0]["recovery_reasons"]
        self.assertIn("POPULAR_SAFETY_NET", reasons)
        self.assertIn("RACE_LEVEL_SIGNAL", reasons)

    def test_recovery_changes_top5_membership(self) -> None:
        horse_scores = [
            _horse_score(1, 40.0, 1, 2.0),
            _horse_score(2, 39.0, 2, 3.0),
            _horse_score(3, 38.0, 3, 4.0),
            _horse_score(4, 37.0, 4, 5.0),
            _horse_score(5, 36.0, 5, 9.0),
            _horse_score(6, 35.0, 3, 8.0),
        ]
        result = apply_top5_borderline_recovery(
            horse_scores,
            deep_analyses=[_analysis(6, positive=["RECENT_FORM_STRONG"])],
            pedigree_analyses=[],
            race_level_analyses=[_analysis(6, positive=["HEAD_TO_HEAD_POSITIVE"])],
            pace_analyses=[],
            race_info=None,
            scoring_config=ScoringConfig(),
            enabled=True,
        )
        self.assertTrue(result["recovery_applied"])
        self.assertNotEqual(result["recovered_top5"], [1, 2, 3, 4, 5])

    def test_many_risks_offset_signal(self) -> None:
        horse_scores = [
            _horse_score(1, 40.0, 1, 2.0),
            _horse_score(2, 39.0, 2, 3.0),
            _horse_score(3, 38.0, 3, 4.0),
            _horse_score(4, 37.0, 4, 5.0),
            _horse_score(5, 36.0, 5, 9.0),
            _horse_score(6, 35.4, 8, 20.0),
        ]
        result = apply_top5_borderline_recovery(
            horse_scores,
            deep_analyses=[_analysis(6, positive=["RECENT_FORM_STRONG"], risk=["A", "B", "C", "D"])],
            pedigree_analyses=[],
            race_level_analyses=[],
            pace_analyses=[],
            race_info=None,
            scoring_config=ScoringConfig(),
            enabled=True,
        )
        self.assertFalse(result["recovery_applied"])


if __name__ == "__main__":
    unittest.main()

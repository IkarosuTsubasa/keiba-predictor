from __future__ import annotations

import unittest

from keiba_llm_agent.backtest.scoring_comparator import (
    build_marks_for_mode,
    evaluate_mode_against_result,
    score_for_mode,
)
from keiba_llm_agent.schemas.prediction import HorseScore, PedigreeAdjustment, ScoreAdjustment, ScoreBreakdown
from keiba_llm_agent.schemas.result import ResultData


def _horse_score(
    horse_no: int,
    base: float,
    pedigree: float = 0.0,
    race_level: float = 0.0,
    pace: float = 0.0,
) -> HorseScore:
    total = round(base + pedigree * 0.2 + race_level + pace * 0.0, 1)
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
            risk=-3,
        ),
        base_total_score=base,
        pedigree_adjustment=PedigreeAdjustment(pedigree_adjustment=pedigree),
        race_level_adjustment=ScoreAdjustment(adjustment=race_level),
        pace_adjustment=ScoreAdjustment(adjustment=pace),
        total_score=total,
        reason="reason",
    )


class ScoringComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.horse_scores = [
            _horse_score(1, 35.0, pedigree=0.0, race_level=-0.5, pace=-0.3),
            _horse_score(2, 34.8, pedigree=0.8, race_level=0.4, pace=0.3),
            _horse_score(3, 33.5, pedigree=0.2, race_level=0.0, pace=0.0),
        ]
        self.result = ResultData.model_validate(
            {"race_id": "r1", "result": {"1st": 2, "2nd": 1, "3rd": 3}, "payouts": []}
        )

    def test_base_only_uses_base_total_score(self) -> None:
        self.assertEqual(score_for_mode(self.horse_scores[0], "base_only"), 35.0)

    def test_pedigree_only_uses_base_plus_pedigree(self) -> None:
        self.assertEqual(score_for_mode(self.horse_scores[1], "pedigree_only"), 35.6)

    def test_full_adjusted_uses_raw_adjustments_not_saved_total_score(self) -> None:
        self.assertEqual(score_for_mode(self.horse_scores[1], "full_adjusted"), 36.3)

    def test_marks_reordered_correctly_by_mode(self) -> None:
        base_marks = build_marks_for_mode(self.horse_scores, "base_only")
        full_marks = build_marks_for_mode(self.horse_scores, "full_adjusted")
        self.assertEqual(base_marks["◎"], 1)
        self.assertEqual(full_marks["◎"], 2)

    def test_result_comparison_is_correct(self) -> None:
        evaluated = evaluate_mode_against_result(
            prediction=type("P", (), {"horse_scores": self.horse_scores})(),
            result_data=self.result,
            mode="full_adjusted",
        )
        self.assertTrue(evaluated["main_mark_top3"])
        self.assertTrue(evaluated["main_mark_win"])
        self.assertEqual(evaluated["marked_top3_count"], 3)
        self.assertTrue(evaluated["top5_contains_winner"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from keiba_llm_agent.scoring.analysis_adjustment_integrator import (
    DEFAULT_PACE_SCORE_WEIGHT,
    DEFAULT_PEDIGREE_SCORE_WEIGHT,
    DEFAULT_RACE_LEVEL_SCORE_WEIGHT,
    apply_adjustment_weight,
    build_score_breakdown,
    calculate_pace_adjustment,
    calculate_race_level_adjustment,
)
from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis


class AnalysisAdjustmentIntegratorTests(unittest.TestCase):
    def test_race_level_adjustment_is_capped(self) -> None:
        positive = calculate_race_level_adjustment(
            RaceLevelAnalysis(
                horse_no=1,
                horse_name="A",
                positive_flags=["HEAD_TO_HEAD_POSITIVE"],
                risk_flags=[],
                head_to_head_summary="",
                race_level_summary="",
                opponent_context_summary="",
                overall_comment="",
                adjustment_hint=1.8,
            )
        )
        negative = calculate_race_level_adjustment(
            RaceLevelAnalysis(
                horse_no=1,
                horse_name="A",
                positive_flags=[],
                risk_flags=["HEAD_TO_HEAD_NEGATIVE"],
                head_to_head_summary="",
                race_level_summary="",
                opponent_context_summary="",
                overall_comment="",
                adjustment_hint=-1.6,
            )
        )
        self.assertEqual(positive.adjustment, 1.0)
        self.assertEqual(negative.adjustment, -1.0)

    def test_pace_fit_adds_points(self) -> None:
        adjustment = calculate_pace_adjustment(
            HorsePaceAnalysis(
                horse_no=1,
                horse_name="A",
                running_style="先行",
                position_stability="安定",
                positive_flags=["PACE_FIT", "POSITION_STABLE"],
                risk_flags=[],
                overall_comment="",
            ),
            RacePaceProjection(
                projected_pace="average",
                pace_comment="",
                favorable_styles=["先行", "差し"],
                risk_styles=[],
            ),
        )
        self.assertGreater(adjustment.adjustment, 0)

    def test_pace_mismatch_and_position_unstable_reduce_points(self) -> None:
        adjustment = calculate_pace_adjustment(
            HorsePaceAnalysis(
                horse_no=1,
                horse_name="A",
                running_style="追込",
                position_stability="不安定",
                positive_flags=[],
                risk_flags=["PACE_MISMATCH", "POSITION_UNSTABLE"],
                overall_comment="",
            ),
            RacePaceProjection(
                projected_pace="slow",
                pace_comment="",
                favorable_styles=["逃げ", "先行"],
                risk_styles=["追込"],
            ),
        )
        self.assertLess(adjustment.adjustment, 0)

    def test_pace_adjustment_is_capped(self) -> None:
        positive = calculate_pace_adjustment(
            HorsePaceAnalysis(
                horse_no=1,
                horse_name="A",
                running_style="先行",
                position_stability="安定",
                positive_flags=[
                    "PACE_FIT",
                    "STALKER_ADVANTAGE",
                    "CLOSING_SPEED",
                    "POSITION_STABLE",
                    "FRONT_RUNNING_ADVANTAGE",
                ],
                risk_flags=[],
                overall_comment="",
            ),
            RacePaceProjection(
                projected_pace="average",
                pace_comment="",
                favorable_styles=["先行", "差し"],
                risk_styles=[],
            ),
        )
        negative = calculate_pace_adjustment(
            HorsePaceAnalysis(
                horse_no=1,
                horse_name="A",
                running_style="追込",
                position_stability="不安定",
                positive_flags=[],
                risk_flags=["PACE_MISMATCH", "POSITION_UNSTABLE"],
                overall_comment="",
            ),
            RacePaceProjection(
                projected_pace="slow",
                pace_comment="",
                favorable_styles=["逃げ", "先行"],
                risk_styles=["追込"],
            ),
        )
        self.assertLessEqual(positive.adjustment, 0.8)
        self.assertGreaterEqual(negative.adjustment, -0.8)

    def test_total_score_breakdown_is_sum_of_components(self) -> None:
        breakdown = build_score_breakdown(35.5, 0.8, 0.2, 0.5, 1.0, 0.3, 0.0)
        self.assertEqual(breakdown.base_total_score, 35.5)
        self.assertEqual(breakdown.pedigree_adjustment_raw, 0.8)
        self.assertEqual(breakdown.pedigree_adjustment_weighted, 0.2)
        self.assertEqual(breakdown.race_level_adjustment_weighted, 0.5)
        self.assertEqual(breakdown.pace_adjustment_weighted, 0.0)
        self.assertEqual(breakdown.total_score, 36.2)

    def test_missing_analysis_returns_zero_adjustment(self) -> None:
        self.assertEqual(calculate_race_level_adjustment(None).adjustment, 0.0)
        self.assertEqual(calculate_pace_adjustment(None, None).adjustment, 0.0)

    def test_default_score_weights_match_current_recommendation(self) -> None:
        self.assertEqual(DEFAULT_PEDIGREE_SCORE_WEIGHT, 0.2)
        self.assertEqual(DEFAULT_RACE_LEVEL_SCORE_WEIGHT, 1.0)
        self.assertEqual(DEFAULT_PACE_SCORE_WEIGHT, 0.2)
        self.assertEqual(apply_adjustment_weight(0.8, DEFAULT_PEDIGREE_SCORE_WEIGHT), 0.2)
        self.assertEqual(apply_adjustment_weight(0.6, DEFAULT_PACE_SCORE_WEIGHT), 0.1)


if __name__ == "__main__":
    unittest.main()

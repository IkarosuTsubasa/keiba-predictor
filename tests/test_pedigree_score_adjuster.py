from __future__ import annotations

import unittest

from keiba_llm_agent.scoring.pedigree_score_adjuster import calculate_pedigree_adjustment
from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis
from keiba_llm_agent.schemas.race_data import RaceInfo


def _race_info(distance: int = 2400) -> RaceInfo:
    return RaceInfo(
        race_id="sample",
        race_name="sample",
        race_date="2026-05-17",
        course="京都",
        surface="芝",
        distance=distance,
        track_condition="良",
        weather="晴",
    )


def _analysis(positive: list[str], risks: list[str]) -> PedigreeAnalysis:
    return PedigreeAnalysis(
        horse_no=1,
        horse_name="A",
        sire="キズナ",
        dam="B",
        damsire="キングカメハメハ",
        surface_tendency="芝",
        distance_tendency="中距離",
        track_condition_tendency="パワー",
        pace_tendency="持続力",
        positive_flags=positive,
        risk_flags=risks,
        overall_comment="sample",
    )


class PedigreeScoreAdjusterTests(unittest.TestCase):
    def test_surface_and_distance_fit_is_positive(self) -> None:
        adjustment = calculate_pedigree_adjustment(
            _analysis(["PEDIGREE_SURFACE_FIT", "PEDIGREE_DISTANCE_FIT"], []),
            _race_info(),
        )
        self.assertGreater(adjustment.pedigree_adjustment, 0)

    def test_stamina_and_power_fit_is_positive(self) -> None:
        adjustment = calculate_pedigree_adjustment(
            _analysis(["PEDIGREE_STAMINA_FIT", "PEDIGREE_POWER_FIT"], []),
            _race_info(),
        )
        self.assertGreater(adjustment.pedigree_adjustment, 0)

    def test_distance_risk_is_negative(self) -> None:
        adjustment = calculate_pedigree_adjustment(
            _analysis([], ["PEDIGREE_DISTANCE_RISK"]),
            _race_info(distance=1200),
        )
        self.assertLess(adjustment.pedigree_adjustment, 0)

    def test_data_incomplete_is_zero(self) -> None:
        adjustment = calculate_pedigree_adjustment(
            _analysis([], ["PEDIGREE_DATA_INCOMPLETE"]),
            _race_info(),
        )
        self.assertEqual(adjustment.pedigree_adjustment, 0.0)

    def test_adjustment_upper_bound(self) -> None:
        adjustment = calculate_pedigree_adjustment(
            _analysis(
                [
                    "PEDIGREE_SURFACE_FIT",
                    "PEDIGREE_DISTANCE_FIT",
                    "PEDIGREE_STAMINA_FIT",
                    "PEDIGREE_POWER_FIT",
                    "PEDIGREE_TRACK_CONDITION_FIT",
                ],
                [],
            ),
            _race_info(),
        )
        self.assertLessEqual(adjustment.pedigree_adjustment, 2.0)

    def test_adjustment_lower_bound(self) -> None:
        adjustment = calculate_pedigree_adjustment(
            _analysis([], ["PEDIGREE_DISTANCE_RISK", "PEDIGREE_DISTANCE_RISK", "PEDIGREE_DISTANCE_RISK"]),
            _race_info(distance=1200),
        )
        self.assertGreaterEqual(adjustment.pedigree_adjustment, -1.5)


if __name__ == "__main__":
    unittest.main()


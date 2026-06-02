from __future__ import annotations

import unittest

from keiba_llm_agent.pedigree.pedigree_analyzer import analyze_pedigree
from keiba_llm_agent.schemas.pedigree import PedigreeInfo
from keiba_llm_agent.schemas.race_data import RaceInfo


def _race_info(distance: int) -> RaceInfo:
    return RaceInfo(
        race_id="sample",
        race_name="sample",
        race_date="2026-05-17",
        course="東京",
        surface="芝",
        distance=distance,
        track_condition="良",
        weather="晴",
    )


class PedigreeAnalyzerTests(unittest.TestCase):
    def test_hearts_cry_at_2400_gets_stamina_fit(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h1", horse_name="A", sire="ハーツクライ"),
            _race_info(2400),
            horse_no=1,
            horse_name="A",
        )
        self.assertIn("PEDIGREE_STAMINA_FIT", analysis.positive_flags)

    def test_lord_kanaloa_at_mile_gets_distance_fit(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h2", horse_name="B", sire="ロードカナロア"),
            _race_info(1600),
            horse_no=2,
            horse_name="B",
        )
        self.assertIn("PEDIGREE_DISTANCE_FIT", analysis.positive_flags)

    def test_long_distance_pedigree_at_1200_gets_distance_risk(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h3", horse_name="C", sire="ハーツクライ"),
            _race_info(1200),
            horse_no=3,
            horse_name="C",
        )
        self.assertIn("PEDIGREE_DISTANCE_RISK", analysis.risk_flags)

    def test_unknown_sire_gets_unknown_flags(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h4", horse_name="D", sire="unknown sire"),
            _race_info(1600),
            horse_no=4,
            horse_name="D",
        )
        self.assertTrue(
            "PEDIGREE_DISTANCE_UNKNOWN" in analysis.risk_flags
            or "PEDIGREE_SURFACE_UNKNOWN" in analysis.risk_flags
        )

    def test_missing_pedigree_gets_data_incomplete(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h5", horse_name="E"),
            _race_info(1600),
            horse_no=5,
            horse_name="E",
        )
        self.assertIn("PEDIGREE_DATA_INCOMPLETE", analysis.risk_flags)


if __name__ == "__main__":
    unittest.main()

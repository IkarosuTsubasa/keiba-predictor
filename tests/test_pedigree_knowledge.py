from __future__ import annotations

import unittest

from keiba_llm_agent.pedigree.pedigree_analyzer import analyze_pedigree
from keiba_llm_agent.schemas.pedigree import PedigreeInfo
from keiba_llm_agent.schemas.race_data import RaceInfo


def _race_info(distance: int, track_condition: str = "良") -> RaceInfo:
    return RaceInfo(
        race_id="sample",
        race_name="sample",
        race_date="2026-05-17",
        course="京都",
        surface="芝",
        distance=distance,
        track_condition=track_condition,
        weather="晴",
    )


class PedigreeKnowledgeRemovalTests(unittest.TestCase):
    def test_named_sire_no_longer_gets_static_stamina_fit(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h1", horse_name="A", sire="サトノダイヤモンド"),
            _race_info(2400),
            horse_no=1,
            horse_name="A",
        )
        self.assertEqual(analysis.positive_flags, [])
        self.assertIn("PEDIGREE_DISTANCE_UNKNOWN", analysis.risk_flags)

    def test_named_damsire_no_longer_gets_static_comment_or_fit(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h2", horse_name="B", sire="unknown", damsire="ハーツクライ"),
            _race_info(2400),
            horse_no=2,
            horse_name="B",
        )
        self.assertNotIn("母父ハーツクライ", analysis.overall_comment)
        self.assertNotIn("PEDIGREE_STAMINA_FIT", analysis.positive_flags)

    def test_heavy_track_needs_performance_profile(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h3", horse_name="C", sire="ゴールドシップ"),
            _race_info(2400, track_condition="重"),
            horse_no=3,
            horse_name="C",
        )
        self.assertNotIn("PEDIGREE_TRACK_CONDITION_FIT", analysis.positive_flags)
        self.assertIn("PEDIGREE_SURFACE_UNKNOWN", analysis.risk_flags)


if __name__ == "__main__":
    unittest.main()

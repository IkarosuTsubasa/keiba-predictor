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


class PedigreeKnowledgeTests(unittest.TestCase):
    def test_satono_diamond_at_2400_gets_stamina_fit(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h1", horse_name="A", sire="サトノダイヤモンド"),
            _race_info(2400),
            horse_no=1,
            horse_name="A",
        )
        self.assertIn("PEDIGREE_STAMINA_FIT", analysis.positive_flags)

    def test_kizuna_with_king_kamehameha_damsire_gets_surface_fit_and_comment(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h2", horse_name="B", sire="キズナ", damsire="キングカメハメハ"),
            _race_info(2400),
            horse_no=2,
            horse_name="B",
        )
        self.assertIn("PEDIGREE_SURFACE_FIT", analysis.positive_flags)
        self.assertIn("母父キングカメハメハ", analysis.overall_comment)

    def test_mikki_isle_at_mile_gets_distance_fit(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h3", horse_name="C", sire="ミッキーアイル"),
            _race_info(1600),
            horse_no=3,
            horse_name="C",
        )
        self.assertIn("PEDIGREE_DISTANCE_FIT", analysis.positive_flags)

    def test_mikki_isle_at_2400_does_not_get_stamina_fit(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h4", horse_name="D", sire="ミッキーアイル"),
            _race_info(2400),
            horse_no=4,
            horse_name="D",
        )
        self.assertNotIn("PEDIGREE_STAMINA_FIT", analysis.positive_flags)

    def test_unknown_sire_with_hearts_cry_damsire_gets_damsire_comment(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h5", horse_name="E", sire="unknown", damsire="ハーツクライ"),
            _race_info(2400),
            horse_no=5,
            horse_name="E",
        )
        self.assertIn("母父ハーツクライ", analysis.overall_comment)
        self.assertIn("PEDIGREE_STAMINA_FIT", analysis.positive_flags)

    def test_heavy_track_orfevre_or_gold_ship_gets_track_condition_fit(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h6", horse_name="F", sire="ゴールドシップ"),
            _race_info(2400, track_condition="重"),
            horse_no=6,
            horse_name="F",
        )
        self.assertIn("PEDIGREE_TRACK_CONDITION_FIT", analysis.positive_flags)


if __name__ == "__main__":
    unittest.main()


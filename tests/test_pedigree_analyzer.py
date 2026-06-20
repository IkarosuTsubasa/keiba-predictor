from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from keiba_llm_agent.pedigree.pedigree_analyzer import (
    analyze_pedigree,
    build_pedigree_analyses_for_race,
    normalize_pedigree_name,
)
from keiba_llm_agent.pedigree.pedigree_performance import build_performance_profile
from keiba_llm_agent.scoring.pedigree_score_adjuster import calculate_pedigree_adjustment
from keiba_llm_agent.schemas.pedigree import PedigreeInfo
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceInfo, RecentRun


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


def _local_dirt_race_info(distance: int) -> RaceInfo:
    return RaceInfo(
        race_id="202654060601",
        race_name="sample",
        race_date="2026-06-08",
        course="高知",
        surface="ダート",
        distance=distance,
        track_condition="良",
        weather="晴",
        source="local",
        scope_key="local",
    )


class PedigreeAnalyzerTests(unittest.TestCase):
    def test_named_sire_without_performance_profile_stays_unknown(self) -> None:
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h1", horse_name="A", sire="ハーツクライ"),
            _race_info(2400),
            horse_no=1,
            horse_name="A",
        )
        self.assertEqual(analysis.positive_flags, [])
        self.assertIn("PEDIGREE_DISTANCE_UNKNOWN", analysis.risk_flags)

    def test_sire_performance_profile_at_2400_gets_distance_fit(self) -> None:
        race_info = _race_info(2400)
        profile = build_performance_profile(
            relation="sire",
            horse_id="sire1",
            horse_name="実績父",
            runs=[
                RecentRun(date="2020-01-01", race_name="sample", surface="芝", distance=2400, finish=1, field_size=12),
                RecentRun(date="2020-02-01", race_name="sample", surface="芝", distance=2200, finish=2, field_size=12),
            ],
            race_info=race_info,
        )
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h2", horse_name="B", sire="実績父", sire_id="sire1"),
            race_info,
            horse_no=2,
            horse_name="B",
            performance_profiles=[profile],
        )
        self.assertIn("PEDIGREE_DISTANCE_FIT", analysis.positive_flags)

    def test_long_distance_sire_performance_at_1200_gets_distance_risk(self) -> None:
        race_info = _race_info(1200)
        profile = build_performance_profile(
            relation="sire",
            horse_id="sire2",
            horse_name="長距離父",
            runs=[
                RecentRun(date="2020-01-01", race_name="sample", surface="芝", distance=2400, finish=1, field_size=12),
                RecentRun(date="2020-02-01", race_name="sample", surface="芝", distance=2500, finish=2, field_size=12),
            ],
            race_info=race_info,
        )
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h3", horse_name="C", sire="長距離父", sire_id="sire2"),
            race_info,
            horse_no=3,
            horse_name="C",
            performance_profiles=[profile],
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

    def test_mixed_foreign_suffix_name_is_normalized(self) -> None:
        self.assertEqual(normalize_pedigree_name("パイロ Pyro(米)"), "パイロ")
        self.assertEqual(normalize_pedigree_name("Majestic Warrior(米)"), "Majestic Warrior")

    def test_local_dirt_sire_gets_nonzero_adjustment(self) -> None:
        race_info = _local_dirt_race_info(1300)
        profile = build_performance_profile(
            relation="sire",
            horse_id="sire3",
            horse_name="ダート父",
            runs=[
                RecentRun(date="2020-01-01", race_name="sample", surface="ダート", distance=1400, finish=1, field_size=12),
                RecentRun(date="2020-02-01", race_name="sample", surface="ダート", distance=1600, finish=2, field_size=12),
            ],
            race_info=race_info,
        )
        analysis = analyze_pedigree(
            PedigreeInfo(horse_id="h6", horse_name="F", sire="ダート父", sire_id="sire3"),
            race_info,
            horse_no=6,
            horse_name="F",
            performance_profiles=[profile],
        )
        adjustment = calculate_pedigree_adjustment(analysis, race_info)
        self.assertIn("PEDIGREE_SURFACE_FIT", analysis.positive_flags)
        self.assertGreater(adjustment.pedigree_adjustment, 0.0)

    def test_local_dirt_fresh_sire_gets_surface_and_distance_fit(self) -> None:
        race_info = _local_dirt_race_info(1100)
        sire_profile = build_performance_profile(
            relation="sire",
            horse_id="sire4",
            horse_name="短距離ダート父",
            runs=[
                RecentRun(date="2020-01-01", race_name="sample", surface="ダート", distance=1200, finish=1, field_size=12),
                RecentRun(date="2020-02-01", race_name="sample", surface="ダート", distance=1400, finish=2, field_size=12),
            ],
            race_info=race_info,
        )
        damsire_profile = build_performance_profile(
            relation="damsire",
            horse_id="damsire4",
            horse_name="母父ダート",
            runs=[
                RecentRun(date="2020-01-01", race_name="sample", surface="ダート", distance=1200, finish=1, field_size=12),
            ],
            race_info=race_info,
        )
        analysis = analyze_pedigree(
            PedigreeInfo(
                horse_id="h7",
                horse_name="G",
                sire="短距離ダート父",
                sire_id="sire4",
                damsire="母父ダート",
                damsire_id="damsire4",
            ),
            race_info,
            horse_no=7,
            horse_name="G",
            performance_profiles=[sire_profile, damsire_profile],
        )
        adjustment = calculate_pedigree_adjustment(analysis, race_info)
        self.assertIn("PEDIGREE_SURFACE_FIT", analysis.positive_flags)
        self.assertIn("PEDIGREE_DISTANCE_FIT", analysis.positive_flags)
        self.assertGreater(adjustment.pedigree_adjustment, 0.0)

    def test_performance_profile_fetch_runs_for_full_sample_horse(self) -> None:
        horse = HorseEntry(
            horse_no=1,
            horse_id="2024100058",
            horse_name="A",
            recent_runs=[
                RecentRun(date=f"2026-05-{day:02d}", finish=day, field_size=12)
                for day in range(1, 6)
            ],
        )
        pedigree = PedigreeInfo(
            horse_id="2024100058",
            horse_name="A",
            sire="ルヴァンスレーヴ",
            sire_id="2015104189",
        )
        with (
            patch("keiba_llm_agent.pedigree.pedigree_analyzer.get_horse_cache_path", return_value=Path("__missing_cache__")),
            patch("keiba_llm_agent.pedigree.pedigree_analyzer.fetch_horse_html", return_value="<html></html>"),
            patch("keiba_llm_agent.pedigree.pedigree_analyzer.parse_pedigree_info", return_value=pedigree),
            patch("keiba_llm_agent.pedigree.pedigree_analyzer.build_performance_profiles_for_pedigree") as build_profiles,
        ):
            build_profiles.return_value = []
            analyses = build_pedigree_analyses_for_race([horse], _local_dirt_race_info(1100))

        build_profiles.assert_called_once()
        self.assertEqual(analyses[0].performance_profiles, [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

from keiba_llm_agent.pedigree.pedigree_performance import (
    build_performance_profile,
    build_performance_profiles_for_pedigree,
)
from keiba_llm_agent.scoring.pedigree_score_adjuster import calculate_pedigree_adjustment
from keiba_llm_agent.pedigree.pedigree_analyzer import analyze_pedigree
from keiba_llm_agent.schemas.pedigree import PedigreeInfo
from keiba_llm_agent.schemas.race_data import RaceInfo, RecentRun


def _race_info() -> RaceInfo:
    return RaceInfo(
        race_id="202630061105",
        race_name="JRA認定競走フレッシュチャレンジ競走(2歳)",
        race_date="2026-06-11",
        course="門別",
        surface="ダート",
        distance=1100,
        track_condition="良",
        source="local",
        scope_key="local",
    )


def _run(
    *,
    date: str,
    race_name: str,
    surface: str,
    distance: int,
    finish: int,
    track_condition: str = "良",
) -> RecentRun:
    return RecentRun(
        date=date,
        race_name=race_name,
        course="川崎",
        surface=surface,
        distance=distance,
        track_condition=track_condition,
        finish=finish,
        field_size=14,
        passing_order="2-2-2-1",
        corner_positions=[2, 2, 2, 1],
        final_3f=36.0,
    )


class PedigreePerformanceTests(unittest.TestCase):
    def test_sire_result_profile_adds_surface_distance_and_class_flags(self) -> None:
        runs = [
            _run(date="2017-10-14", race_name="プラタナス賞", surface="ダート", distance=1600, finish=1),
            _run(date="2018-12-02", race_name="チャンピオンズC(GI)", surface="ダート", distance=1800, finish=1),
            _run(date="2018-07-11", race_name="ジャパンダートダービー(JpnI)", surface="ダート", distance=2000, finish=1),
        ]
        profile = build_performance_profile(
            relation="sire",
            horse_id="2015104189",
            horse_name="ルヴァンスレーヴ",
            runs=runs,
            race_info=_race_info(),
        )
        self.assertIn("PEDIGREE_SURFACE_FIT", profile.positive_flags)
        self.assertIn("PEDIGREE_DISTANCE_FIT", profile.positive_flags)
        self.assertIn("PEDIGREE_CLASS_POWER", profile.positive_flags)
        self.assertIn("PEDIGREE_EARLY_MATURITY", profile.positive_flags)
        self.assertGreater(profile.score_hint, 0.0)

    def test_analyze_pedigree_uses_performance_profile_without_knowledge(self) -> None:
        profile = build_performance_profile(
            relation="sire",
            horse_id="2015104189",
            horse_name="ルヴァンスレーヴ",
            runs=[
                _run(date="2017-10-14", race_name="プラタナス賞", surface="ダート", distance=1600, finish=1),
                _run(date="2018-12-02", race_name="チャンピオンズC(GI)", surface="ダート", distance=1800, finish=1),
            ],
            race_info=_race_info(),
        )
        analysis = analyze_pedigree(
            PedigreeInfo(
                horse_id="2024100058",
                horse_name="サンプル",
                sire="unknown sire",
                sire_id="2015104189",
            ),
            _race_info(),
            horse_no=1,
            horse_name="サンプル",
            performance_profiles=[profile],
        )
        adjustment = calculate_pedigree_adjustment(analysis, _race_info())
        self.assertIn("PEDIGREE_SURFACE_FIT", analysis.positive_flags)
        self.assertGreater(analysis.performance_score_hint, 0.0)
        self.assertGreater(adjustment.pedigree_adjustment, 0.0)

    def test_build_performance_profiles_fetches_sire_and_damsire_only(self) -> None:
        html = """
        <html><body>
          <table class="db_h_race_results nk_tb_common">
            <tr><th>日付</th><th>レース名</th><th>着順</th><th>距離</th><th>馬場</th><th>開催</th><th>頭数</th></tr>
            <tr><td>2018/12/02</td><td>チャンピオンズC(GI)</td><td>1</td><td>ダ1800</td><td>良</td><td>中京</td><td>15</td></tr>
          </table>
        </body></html>
        """
        pedigree = PedigreeInfo(
            horse_id="2024100058",
            horse_name="参戦馬",
            sire="ルヴァンスレーヴ",
            sire_id="2015104189",
            damsire="バトルプラン",
            damsire_id="000a0118e8",
            sire_sire="シンボリクリスエス",
            sire_sire_id="1999110099",
        )
        with patch("keiba_llm_agent.pedigree.pedigree_performance.fetch_horse_html", return_value=html):
            profiles = build_performance_profiles_for_pedigree(pedigree, _race_info())

        self.assertEqual([profile.relation for profile in profiles], ["sire", "damsire"])


if __name__ == "__main__":
    unittest.main()

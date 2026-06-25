from __future__ import annotations

import unittest

from keiba_llm_agent.scoring.recent_run_scorer import (
    build_prediction_from_recent_runs,
    has_recent_runs_data,
    score_jockey_fit,
    score_horse_by_recent_runs,
    score_run_performance,
    score_track_condition_fit,
)
from keiba_llm_agent.schemas.deep_analysis import HorseDeepAnalysis
from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo, RecentRun
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis


def build_run(
    date: str,
    course: str,
    surface: str,
    distance: int,
    finish: int | None,
    field_size: int | None,
    jockey: str,
    track_condition: str = "良",
    odds: float | None = None,
    popularity: int | None = None,
    race_name: str | None = None,
    margin: str | None = None,
) -> RecentRun:
    return RecentRun(
        race_name=race_name,
        date=date,
        course=course,
        surface=surface,
        distance=distance,
        finish=finish,
        field_size=field_size,
        jockey=jockey,
        track_condition=track_condition,
        odds=odds,
        popularity=popularity,
        margin=margin,
    )


def build_pedigree_analysis(
    positive_flags: list[str],
    risk_flags: list[str] | None = None,
) -> PedigreeAnalysis:
    return PedigreeAnalysis(
        horse_no=1,
        horse_name="A",
        sire="ルヴァンスレーヴ",
        dam="サンプル母",
        damsire="バトルプラン",
        surface_tendency="ダート",
        distance_tendency="短距離〜マイル",
        track_condition_tendency="パワー",
        pace_tendency="スピード",
        positive_flags=positive_flags,
        risk_flags=risk_flags or [],
        overall_comment="血統面の後押しあり。",
    )


class RecentRunScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.race_info = RaceInfo(
            race_id="sample_001",
            race_date="2026-05-18",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
        )
        self.debut_race_info = RaceInfo(
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

    def test_empty_recent_runs_uses_unraced_baseline(self) -> None:
        horse = HorseEntry(horse_no=1, horse_name="A", jockey="騎手A", recent_runs=[])
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertEqual(score.scores.risk, -2)
        self.assertEqual(score.total_score, 6.0)
        self.assertIn("実戦履歴が不足", score.reason)

    def test_unraced_debut_horse_uses_debut_baseline(self) -> None:
        horse = HorseEntry(horse_no=1, horse_name="A", jockey="騎手A", recent_runs=[])
        score = score_horse_by_recent_runs(horse, self.debut_race_info)
        self.assertEqual(score.scores.risk, -2)
        self.assertEqual(score.total_score, 6.0)
        self.assertIn("新馬戦", score.reason)

    def test_runs_without_finish_do_not_create_distance_or_course_fit(self) -> None:
        horse = HorseEntry(
            horse_no=2,
            horse_name="B",
            jockey="騎手B",
            recent_runs=[
                build_run(
                    "2026-06-11",
                    "門別",
                    "ダート",
                    1100,
                    None,
                    None,
                    "騎手B",
                )
            ],
        )
        score = score_horse_by_recent_runs(horse, self.debut_race_info)
        self.assertEqual(score.scores.distance_fit, 0)
        self.assertEqual(score.scores.course_fit, 0)
        self.assertEqual(score.scores.track_condition_fit, 0)
        self.assertEqual(score.scores.jockey_fit, 0)
        self.assertEqual(score.total_score, 6.0)

    def test_low_sample_pedigree_priors_raise_fit_scores(self) -> None:
        horse = HorseEntry(
            horse_no=3,
            horse_name="C",
            jockey="騎手C",
            recent_runs=[
                build_run(
                    "2026-05-20",
                    "浦和",
                    "ダート",
                    1400,
                    8,
                    12,
                    "騎手D",
                )
            ],
        )
        pedigree = build_pedigree_analysis(
            ["PEDIGREE_SURFACE_FIT", "PEDIGREE_DISTANCE_FIT", "PEDIGREE_POWER_FIT"]
        )
        score = score_horse_by_recent_runs(horse, self.debut_race_info, pedigree_analysis=pedigree)
        self.assertGreaterEqual(score.scores.distance_fit, 7)
        self.assertGreaterEqual(score.scores.course_fit, 6)
        self.assertGreaterEqual(score.scores.track_condition_fit, 6)
        self.assertEqual(score.scores.risk, -3)

    def test_debut_race_without_recent_runs_still_uses_scoring_path(self) -> None:
        race_data = RaceData(
            race_info=self.debut_race_info,
            horses=[HorseEntry(horse_no=1, horse_name="A", recent_runs=[])],
        )
        self.assertTrue(has_recent_runs_data(race_data))

    def test_horse_id_without_recent_runs_still_uses_scoring_path(self) -> None:
        race_data = RaceData(
            race_info=self.race_info,
            horses=[HorseEntry(horse_no=1, horse_id="2024100058", horse_name="A", recent_runs=[])],
        )
        self.assertTrue(has_recent_runs_data(race_data))

    def test_good_recent_finishes_raise_recent_form(self) -> None:
        horse = HorseEntry(
            horse_no=1,
            horse_name="A",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-04-20", "東京", "芝", 1600, 1, 16, "騎手A"),
                build_run("2026-03-20", "中山", "芝", 1600, 2, 16, "騎手A"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.recent_form, 8)

    def test_same_distance_improves_distance_fit(self) -> None:
        horse = HorseEntry(
            horse_no=2,
            horse_name="B",
            jockey="騎手B",
            recent_runs=[
                build_run("2026-04-10", "阪神", "芝", 1600, 3, 18, "騎手B"),
                build_run("2026-03-15", "阪神", "芝", 1600, 4, 18, "騎手B"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.distance_fit, 8)

    def test_same_course_improves_course_fit(self) -> None:
        horse = HorseEntry(
            horse_no=3,
            horse_name="C",
            jockey="騎手C",
            recent_runs=[
                build_run("2026-04-12", "東京", "芝", 1400, 2, 18, "騎手C"),
                build_run("2026-03-12", "東京", "芝", 1800, 5, 18, "騎手C"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.course_fit, 7)

    def test_career_surface_history_beyond_recent_five_improves_course_fit(self) -> None:
        horse = HorseEntry(
            horse_no=30,
            horse_name="SurfaceCareer",
            jockey="騎手S",
            recent_runs=[
                build_run("2026-04-20", "大井", "ダート", 1600, 10, 16, "騎手S"),
                build_run("2026-03-20", "船橋", "ダート", 1600, 12, 16, "騎手S"),
                build_run("2026-02-20", "川崎", "ダート", 1500, 11, 14, "騎手S"),
                build_run("2026-01-20", "浦和", "ダート", 1400, 9, 12, "騎手S"),
                build_run("2025-12-20", "大井", "ダート", 1800, 13, 16, "騎手S"),
                build_run("2025-10-20", "中山", "芝", 1600, 1, 16, "騎手T"),
                build_run("2025-09-20", "京都", "芝", 1600, 2, 16, "騎手T"),
                build_run("2025-08-20", "阪神", "芝", 1800, 3, 16, "騎手T"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertLess(score.scores.recent_quality_score, score.scores.ability_score)
        self.assertGreaterEqual(score.scores.ability_score, 8)
        self.assertGreaterEqual(score.scores.course_fit, 6)
        self.assertIn("芝では好走歴", score.reason)

    def test_career_ability_is_separated_from_recent_quality(self) -> None:
        horse = HorseEntry(
            horse_no=31,
            horse_name="CareerAbility",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-05-20", "大井", "ダート", 1600, 10, 16, "騎手A"),
                build_run("2026-04-20", "船橋", "ダート", 1600, 12, 16, "騎手A"),
                build_run("2026-03-20", "川崎", "ダート", 1500, 11, 14, "騎手A"),
                build_run("2025-12-20", "東京", "芝", 1600, 1, 16, "騎手B"),
                build_run("2025-11-20", "京都", "芝", 1600, 2, 16, "騎手B"),
                build_run("2025-10-20", "阪神", "芝", 1800, 3, 16, "騎手B"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.ability_score, 8)
        self.assertLessEqual(score.scores.recent_quality_score, 4)
        self.assertGreater(score.scores.ability_score, score.scores.recent_quality_score)

    def test_trend_score_identifies_improvement(self) -> None:
        horse = HorseEntry(
            horse_no=32,
            horse_name="Improver",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-05-20", "東京", "芝", 1600, 1, 16, "騎手A"),
                build_run("2026-04-20", "東京", "芝", 1600, 2, 16, "騎手A"),
                build_run("2026-03-20", "東京", "芝", 1600, 10, 16, "騎手A"),
                build_run("2026-02-20", "東京", "芝", 1600, 11, 16, "騎手A"),
                build_run("2026-01-20", "東京", "芝", 1600, 12, 16, "騎手A"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.trend_score, 8)
        self.assertIn("上昇気配", score.reason)

    def test_trend_score_identifies_decline(self) -> None:
        horse = HorseEntry(
            horse_no=33,
            horse_name="Decliner",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-05-20", "東京", "芝", 1600, 11, 16, "騎手A"),
                build_run("2026-04-20", "東京", "芝", 1600, 12, 16, "騎手A"),
                build_run("2026-03-20", "東京", "芝", 1600, 1, 16, "騎手A"),
                build_run("2026-02-20", "東京", "芝", 1600, 2, 16, "騎手A"),
                build_run("2026-01-20", "東京", "芝", 1600, 3, 16, "騎手A"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertLessEqual(score.scores.trend_score, 2)
        self.assertIn("下降気味", score.reason)

    def test_competitive_recent_grade_runs_do_not_over_penalize_trend(self) -> None:
        horse = HorseEntry(
            horse_no=36,
            horse_name="GradeCampaigner",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-05-20", "東京", "芝", 1600, 8, 18, "騎手A", race_name="読売マイラーズC(G2)", margin="0.4"),
                build_run("2026-04-20", "東京", "芝", 1600, 13, 16, "騎手A", race_name="東京新聞杯(G3)", margin="0.8"),
                build_run("2026-03-20", "東京", "芝", 1600, 5, 18, "騎手A", race_name="マイルCS(G1)", margin="0.4"),
                build_run("2026-02-20", "東京", "芝", 1800, 5, 11, "騎手A", race_name="毎日王冠(G2)", margin="0.5"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.trend_score, 4)

    def test_condition_fit_score_summarizes_fit_fields(self) -> None:
        horse = HorseEntry(
            horse_no=34,
            horse_name="ConditionFit",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-05-20", "東京", "芝", 1600, 2, 16, "騎手A", track_condition="良"),
                build_run("2026-04-20", "東京", "芝", 1600, 3, 16, "騎手A", track_condition="良"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.condition_fit_score, 8)

    def test_grade_close_loss_is_not_undervalued_against_lower_class_win(self) -> None:
        target = RaceInfo(
            race_id="202609030611",
            race_name="しらさぎS",
            race_date="2026-06-21",
            course="阪神",
            surface="芝",
            distance=1600,
            track_condition="稍重",
            scope_key="central",
        )
        graded_close_loss = build_run(
            "2026-04-20",
            "京都",
            "芝",
            1600,
            4,
            18,
            "騎手A",
            race_name="読売マイラーズC(G2)",
            margin="0.3",
        )
        lower_class_win = build_run(
            "2026-04-20",
            "阪神",
            "芝",
            1600,
            1,
            18,
            "騎手B",
            race_name="3勝クラス",
            margin="0.0",
        )
        graded_score = score_run_performance(graded_close_loss, target)
        lower_class_score = score_run_performance(lower_class_win, target)
        self.assertGreaterEqual(graded_score, 7.7)
        self.assertLessEqual(lower_class_score, 8.1)
        self.assertGreater(graded_score, lower_class_score)

    def test_graded_small_margin_backmarker_keeps_competitive_floor(self) -> None:
        target = RaceInfo(
            race_id="graded_target",
            race_name="しらさぎS",
            course="阪神",
            surface="芝",
            distance=1600,
            track_condition="良",
            scope_key="central",
        )
        run = build_run(
            "2026-02-10",
            "東京",
            "芝",
            1600,
            13,
            16,
            "騎手A",
            race_name="東京新聞杯(G3)",
            margin="0.8",
        )
        self.assertGreaterEqual(score_run_performance(run, target), 5.8)

    def test_nearby_track_condition_is_neutral_or_better(self) -> None:
        fit = score_track_condition_fit(
            [build_run("2026-05-20", "阪神", "芝", 1600, 2, 18, "騎手A", track_condition="良")],
            "稍重",
        )
        self.assertGreaterEqual(fit, 7)

    def test_no_same_jockey_uses_neutral_fit(self) -> None:
        fit = score_jockey_fit(
            [build_run("2026-05-20", "阪神", "芝", 1600, 2, 18, "騎手B")],
            "騎手A",
        )
        self.assertEqual(fit, 5)

    def test_carried_weight_adjustment_changes_prediction_order(self) -> None:
        race_data = RaceData(
            race_info=RaceInfo(
                race_id="weight_001",
                race_name="しらさぎS",
                race_date="2026-06-21",
                course="阪神",
                surface="芝",
                distance=1600,
                track_condition="良",
                scope_key="central",
            ),
            horses=[
                HorseEntry(
                    horse_no=1,
                    horse_name="Light",
                    jockey="騎手A",
                    carried_weight=53.0,
                    recent_runs=[
                        build_run("2026-05-20", "阪神", "芝", 1600, 2, 18, "騎手A"),
                        build_run("2026-04-20", "京都", "芝", 1600, 3, 18, "騎手A"),
                    ],
                ),
                HorseEntry(
                    horse_no=2,
                    horse_name="Heavy",
                    jockey="騎手A",
                    carried_weight=57.0,
                    recent_runs=[
                        build_run("2026-05-20", "阪神", "芝", 1600, 2, 18, "騎手A"),
                        build_run("2026-04-20", "京都", "芝", 1600, 3, 18, "騎手A"),
                    ],
                ),
            ],
        )
        prediction = build_prediction_from_recent_runs(race_data, [])
        scores = {score.horse_no: score for score in prediction.horse_scores}
        self.assertEqual(prediction.marks["◎"], 1)
        self.assertGreater(scores[1].total_score, scores[2].total_score)
        self.assertIn("斤量面から", scores[1].reason)

    def test_race_level_and_pace_components_are_scored(self) -> None:
        horse = HorseEntry(
            horse_no=35,
            horse_name="ContextHorse",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-05-20", "東京", "芝", 1600, 2, 16, "騎手A"),
                build_run("2026-04-20", "東京", "芝", 1600, 3, 16, "騎手A"),
            ],
        )
        race_level = RaceLevelAnalysis(
            horse_no=35,
            horse_name="ContextHorse",
            positive_flags=["HEAD_TO_HEAD_POSITIVE", "GRADED_SMALL_MARGIN"],
            risk_flags=[],
            head_to_head_summary="近走で本場メンバーとの交差あり。",
            race_level_summary="重賞級で着差の小さい内容がある。",
            opponent_context_summary="相手比較では優位。",
            overall_comment="相手関係では前走比較に強みがある。",
            adjustment_hint=1.0,
        )
        pace = HorsePaceAnalysis(
            horse_no=35,
            horse_name="ContextHorse",
            running_style="先行",
            early_position_score=8.0,
            late_position_score=6.0,
            position_stability="安定",
            positive_flags=["PACE_FIT", "STALKER_ADVANTAGE", "POSITION_STABLE"],
            risk_flags=[],
            overall_comment="展開利が見込める。",
        )
        projection = RacePaceProjection(
            projected_pace="average",
            front_runner_count=1,
            stalker_count=4,
            closer_count=3,
            pace_comment="平均的な流れを想定。",
        )
        score = score_horse_by_recent_runs(
            horse,
            self.race_info,
            race_level_analysis=race_level,
            pace_analysis=pace,
            race_pace_projection=projection,
        )
        self.assertGreater(score.scores.race_level_score, 5)
        self.assertGreater(score.scores.pace_jockey_score, 5)

    def test_central_competitive_field_boosts_race_level_anchor(self) -> None:
        race_info = RaceInfo(
            race_id="202605030811",
            race_name="3歳未勝利",
            race_date="2026-06-21",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
            source="central",
            scope_key="central",
        )
        horse = HorseEntry(
            horse_no=37,
            horse_name="MidFieldAnchor",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-05-20", "東京", "芝", 1600, 5, 15, "騎手A"),
                build_run("2026-04-20", "東京", "芝", 1600, 4, 14, "騎手A"),
            ],
        )
        race_level = RaceLevelAnalysis(
            horse_no=37,
            horse_name="MidFieldAnchor",
            positive_flags=["HEAD_TO_HEAD_POSITIVE", "LARGE_FIELD_GOOD_RUN"],
            risk_flags=[],
            head_to_head_summary="同組比較では優位。",
            race_level_summary="多頭数戦で内容を作れている。",
            opponent_context_summary="相手比較では優位。",
            overall_comment="相手関係では前走比較に強みがある。",
            adjustment_hint=0.5,
        )
        normal_field_score = score_horse_by_recent_runs(
            horse,
            race_info,
            field_size=12,
            race_level_analysis=race_level,
        )
        competitive_field_score = score_horse_by_recent_runs(
            horse,
            race_info,
            field_size=14,
            race_level_analysis=race_level,
        )
        self.assertGreater(
            competitive_field_score.scores.race_level_score,
            normal_field_score.scores.race_level_score,
        )
        self.assertGreaterEqual(competitive_field_score.scores.race_level_score, 8)

    def test_central_competitive_field_penalizes_weak_horse_without_anchor(self) -> None:
        race_info = RaceInfo(
            race_id="202605030811",
            race_name="3歳未勝利",
            race_date="2026-06-21",
            course="東京",
            surface="芝",
            distance=1600,
            track_condition="良",
            source="central",
            scope_key="central",
        )
        horse = HorseEntry(
            horse_no=38,
            horse_name="WeakNoAnchor",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-05-20", "東京", "芝", 1600, 12, 15, "騎手A"),
                build_run("2026-04-20", "東京", "芝", 1600, 13, 14, "騎手A"),
            ],
        )
        deep_analysis = HorseDeepAnalysis(
            horse_no=38,
            horse_name="WeakNoAnchor",
            positive_flags=[],
            risk_flags=["RECENT_FORM_WEAK"],
            recent_form_summary="近走内容は弱い。",
            distance_analysis="距離適性は強調しにくい。",
            course_analysis="コース適性は強調しにくい。",
            track_condition_analysis="馬場適性は強調しにくい。",
            jockey_analysis="騎手面の強調材料は少ない。",
            odds_analysis="市場情報は使わない。",
            overall_comment="強い根拠が乏しい。",
        )
        normal_field_score = score_horse_by_recent_runs(
            horse,
            race_info,
            field_size=12,
            deep_analysis=deep_analysis,
        )
        competitive_field_score = score_horse_by_recent_runs(
            horse,
            race_info,
            field_size=14,
            deep_analysis=deep_analysis,
        )
        self.assertEqual(competitive_field_score.scores.risk, normal_field_score.scores.risk - 1)

    def test_local_midfield_context_boosts_non_market_anchor(self) -> None:
        race_info = RaceInfo(
            race_id="202646030811",
            race_name="地方一般戦",
            race_date="2026-06-21",
            course="大井",
            surface="ダート",
            distance=1600,
            track_condition="良",
            source="local",
            scope_key="local",
        )
        horse = HorseEntry(
            horse_no=39,
            horse_name="LocalMidfieldAnchor",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-05-20", "大井", "ダート", 1600, 3, 14, "騎手A"),
                build_run("2026-04-20", "大井", "ダート", 1600, 4, 13, "騎手A"),
            ],
        )
        deep_analysis = HorseDeepAnalysis(
            horse_no=39,
            horse_name="LocalMidfieldAnchor",
            positive_flags=["RECENT_FORM_STABLE", "DISTANCE_FIT"],
            risk_flags=[],
            recent_form_summary="近走は安定。",
            distance_analysis="距離適性あり。",
            course_analysis="コース適性は標準。",
            track_condition_analysis="馬場適性は標準。",
            jockey_analysis="騎手面は標準。",
            odds_analysis="市場情報は使わない。",
            overall_comment="近走と距離面に根拠あり。",
        )
        pace_analysis = HorsePaceAnalysis(
            horse_no=39,
            horse_name="LocalMidfieldAnchor",
            running_style="先行",
            early_position_score=7.0,
            late_position_score=5.0,
            position_stability="安定",
            positive_flags=["PACE_FIT", "POSITION_STABLE"],
            risk_flags=[],
            overall_comment="展開利が見込める。",
        )
        normal_field_score = score_horse_by_recent_runs(
            horse,
            race_info,
            field_size=12,
            deep_analysis=deep_analysis,
            pace_analysis=pace_analysis,
        )
        midfield_score = score_horse_by_recent_runs(
            horse,
            race_info,
            field_size=14,
            deep_analysis=deep_analysis,
            pace_analysis=pace_analysis,
        )
        self.assertGreater(midfield_score.scores.pace_jockey_score, normal_field_score.scores.pace_jockey_score)

    def test_same_jockey_improves_jockey_fit(self) -> None:
        horse = HorseEntry(
            horse_no=4,
            horse_name="D",
            jockey="横山典",
            recent_runs=[
                build_run("2026-04-12", "東京", "芝", 1600, 2, 18, "横山典弘"),
                build_run("2026-03-12", "中山", "芝", 1600, 4, 18, "横山典弘"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertGreaterEqual(score.scores.jockey_fit, 7)

    def test_total_score_sorting_prefers_stronger_horse(self) -> None:
        strong = HorseEntry(
            horse_no=1,
            horse_name="Strong",
            jockey="騎手A",
            recent_runs=[
                build_run("2026-04-20", "東京", "芝", 1600, 1, 16, "騎手A", odds=12.0, popularity=8),
                build_run("2026-03-20", "東京", "芝", 1600, 2, 16, "騎手A", odds=8.0, popularity=4),
            ],
        )
        weak = HorseEntry(
            horse_no=2,
            horse_name="Weak",
            jockey="騎手B",
            recent_runs=[
                build_run("2026-04-20", "阪神", "芝", 1200, 10, 16, "騎手C", odds=2.2, popularity=1),
                build_run("2026-03-20", "阪神", "芝", 1200, 12, 16, "騎手D", odds=2.5, popularity=1),
            ],
        )
        strong_score = score_horse_by_recent_runs(strong, self.race_info)
        weak_score = score_horse_by_recent_runs(weak, self.race_info)
        self.assertGreater(strong_score.total_score, weak_score.total_score)

    def test_market_signal_disabled_by_default_does_not_change_base_score(self) -> None:
        base_runs = [
            build_run("2026-04-20", "東京", "芝", 1600, 2, 16, "騎手A"),
            build_run("2026-03-20", "東京", "芝", 1600, 3, 16, "騎手A"),
        ]
        favorite = HorseEntry(
            horse_no=11,
            horse_name="Favorite",
            jockey="騎手A",
            odds=2.4,
            popularity=1,
            recent_runs=base_runs,
        )
        outsider = HorseEntry(
            horse_no=12,
            horse_name="Outsider",
            jockey="騎手A",
            odds=25.0,
            popularity=11,
            recent_runs=base_runs,
        )
        favorite_score = score_horse_by_recent_runs(favorite, self.race_info)
        outsider_score = score_horse_by_recent_runs(outsider, self.race_info)
        self.assertNotEqual(favorite_score.scores.odds_value, outsider_score.scores.odds_value)
        self.assertEqual(favorite_score.total_score, outsider_score.total_score)

    def test_market_signal_can_be_explicitly_enabled(self) -> None:
        base_runs = [
            build_run("2026-04-20", "東京", "芝", 1600, 2, 16, "騎手A"),
            build_run("2026-03-20", "東京", "芝", 1600, 3, 16, "騎手A"),
        ]
        outsider = HorseEntry(
            horse_no=13,
            horse_name="Outsider",
            jockey="騎手A",
            odds=25.0,
            popularity=11,
            recent_runs=base_runs,
        )
        baseline_score = score_horse_by_recent_runs(outsider, self.race_info)
        market_enabled_score = score_horse_by_recent_runs(
            outsider,
            self.race_info,
            use_market_score_in_ranking=True,
            market_signal_weight=0.7,
        )
        self.assertGreater(market_enabled_score.total_score, baseline_score.total_score)

    def test_risk_stays_in_required_range(self) -> None:
        horse = HorseEntry(
            horse_no=5,
            horse_name="E",
            jockey="騎手E",
            recent_runs=[build_run("2026-04-20", "東京", "芝", 1600, 8, 16, "騎手E")],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertLessEqual(score.scores.risk, 0)
        self.assertGreaterEqual(score.scores.risk, -10)

    def test_reason_does_not_say_distance_fit_when_same_distance_good_count_is_zero(self) -> None:
        horse = HorseEntry(
            horse_no=6,
            horse_name="F",
            jockey="騎手F",
            recent_runs=[
                build_run("2026-04-20", "阪神", "芝", 1600, 8, 16, "騎手F"),
                build_run("2026-03-20", "中山", "芝", 1600, 7, 16, "騎手F"),
                build_run("2026-02-20", "東京", "芝", 1400, 6, 16, "騎手F"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertNotIn("好走歴0回があり", score.reason)
        self.assertIn("同距離1600mでの好走歴はまだなく", score.reason)

    def test_reason_does_not_say_course_fit_when_same_course_good_count_is_zero(self) -> None:
        horse = HorseEntry(
            horse_no=7,
            horse_name="G",
            jockey="騎手G",
            recent_runs=[
                build_run("2026-04-20", "東京", "芝", 1400, 7, 16, "騎手G"),
                build_run("2026-03-20", "東京", "芝", 1800, 8, 16, "騎手G"),
                build_run("2026-02-20", "中山", "芝", 1600, 9, 16, "騎手G"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertNotIn("好走0回を評価", score.reason)
        self.assertIn("東京では出走歴があるが、好走歴はまだなく", score.reason)

    def test_reason_can_normally_evaluate_when_good_count_positive(self) -> None:
        horse = HorseEntry(
            horse_no=8,
            horse_name="H",
            jockey="横山典",
            recent_runs=[
                build_run("2026-04-20", "東京", "芝", 1600, 2, 16, "横山典弘"),
                build_run("2026-03-20", "東京", "芝", 1600, 3, 16, "横山典弘"),
            ],
        )
        score = score_horse_by_recent_runs(horse, self.race_info)
        self.assertIn("同距離1600mで好走歴", score.reason)


if __name__ == "__main__":
    unittest.main()

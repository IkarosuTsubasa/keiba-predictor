from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.main import run_analysis
from keiba_llm_agent.schemas.deep_analysis import HorseDeepAnalysis
from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis
from keiba_llm_agent.schemas.prediction import HorseScore, ScoreBreakdown
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis


def _mock_score(horse_no: int, horse_name: str, total_score: float) -> HorseScore:
    return HorseScore(
        horse_no=horse_no,
        horse_name=horse_name,
        scores=ScoreBreakdown(
            recent_form=7,
            distance_fit=7,
            course_fit=7,
            track_condition_fit=7,
            jockey_fit=7,
            odds_value=5,
            risk=-3,
        ),
        total_score=total_score,
        reason="近走内容を評価。",
    )


class PredictionScoringModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.race_data_path = self.temp_path / "race_data.json"
        self.prediction_path = self.temp_path / "prediction.json"
        self.lessons_path = self.temp_path / "lessons.json"
        self.lessons_path.write_text("[]\n", encoding="utf-8")
        race_data = {
            "race_info": {
                "race_id": "sample_scoring_mode",
                "race_name": "補正モード確認特別",
                "race_date": "2026-05-17",
                "course": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "weather": "晴",
            },
            "horses": [
                {
                    "horse_no": 1,
                    "frame_no": 1,
                    "horse_id": "h1",
                    "horse_name": "ホースA",
                    "jockey": "騎手A",
                    "carried_weight": 56.0,
                    "odds": 6.2,
                    "popularity": 3,
                    "recent_runs": [{"race_id": "202601010101", "date": "2026-04-01", "course": "東京", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 2, "field_size": 16, "jockey": "騎手A"}],
                },
                {
                    "horse_no": 2,
                    "frame_no": 2,
                    "horse_id": "h2",
                    "horse_name": "ホースB",
                    "jockey": "騎手B",
                    "carried_weight": 56.0,
                    "odds": 9.8,
                    "popularity": 5,
                    "recent_runs": [{"race_id": "202601010102", "date": "2026-04-01", "course": "東京", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 4, "field_size": 16, "jockey": "騎手B"}],
                },
                {
                    "horse_no": 3,
                    "frame_no": 3,
                    "horse_id": "h3",
                    "horse_name": "ホースC",
                    "jockey": "騎手C",
                    "carried_weight": 56.0,
                    "odds": 12.5,
                    "popularity": 7,
                    "recent_runs": [{"race_id": "202601010103", "date": "2026-04-01", "course": "東京", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 5, "field_size": 16, "jockey": "騎手C"}],
                },
            ],
        }
        self.race_data_path.write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_candidate_default_saves_scoring_config_and_weighted_score(self) -> None:
        mock_scores = {
            1: _mock_score(1, "ホースA", 34.0),
            2: _mock_score(2, "ホースB", 33.8),
            3: _mock_score(3, "ホースC", 33.0),
        }
        mock_race_levels = [
            RaceLevelAnalysis(horse_no=1, horse_name="ホースA", positive_flags=[], risk_flags=["HEAD_TO_HEAD_NEGATIVE"], head_to_head_summary="", race_level_summary="", opponent_context_summary="", overall_comment="同組比較ではやや劣勢。", adjustment_hint=-0.6),
            RaceLevelAnalysis(horse_no=2, horse_name="ホースB", positive_flags=["HEAD_TO_HEAD_POSITIVE"], risk_flags=[], head_to_head_summary="", race_level_summary="", opponent_context_summary="", overall_comment="相手関係では優位。", adjustment_hint=0.8),
        ]
        mock_pace_analyses = [
            HorsePaceAnalysis(horse_no=1, horse_name="ホースA", running_style="逃げ", position_stability="不安定", positive_flags=[], risk_flags=["PACE_MISMATCH", "POSITION_UNSTABLE"], overall_comment="展開リスクあり。"),
            HorsePaceAnalysis(horse_no=2, horse_name="ホースB", running_style="先行", position_stability="安定", positive_flags=["PACE_FIT", "STALKER_ADVANTAGE", "POSITION_STABLE"], risk_flags=[], overall_comment="展開利が見込める。"),
            HorsePaceAnalysis(horse_no=3, horse_name="ホースC", running_style="差し", position_stability="安定", positive_flags=[], risk_flags=[], overall_comment="標準評価。"),
        ]
        mock_projection = RacePaceProjection(projected_pace="average", front_runner_count=1, stalker_count=3, closer_count=2, pace_comment="平均ペース想定。", favorable_styles=["先行", "差し"], risk_styles=[])
        mock_pedigree = [
            PedigreeAnalysis(
                horse_no=2,
                horse_name="ホースB",
                sire="ハーツクライ",
                dam="サンプル母",
                damsire="キングカメハメハ",
                surface_tendency="芝",
                distance_tendency="中長距離",
                track_condition_tendency="パワー",
                pace_tendency="スタミナ",
                positive_flags=["PEDIGREE_DISTANCE_FIT"],
                risk_flags=[],
                overall_comment="血統面の後押しあり。",
            )
        ]

        with (
            patch("keiba_llm_agent.scoring.recent_run_scorer.score_horse_by_recent_runs", side_effect=lambda horse, race_info, lessons=None, **kwargs: mock_scores[horse.horse_no]),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_level_for_race", return_value=mock_race_levels),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_pace_for_race", return_value=(mock_pace_analyses, mock_projection)),
            patch("keiba_llm_agent.scoring.recent_run_scorer.build_pedigree_analyses_for_race", return_value=mock_pedigree),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_deeply", return_value=[]),
        ):
            _, saved_path = run_analysis(
                race_data_path=self.race_data_path,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
            )

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["scoring_profile"], "accuracy_default")
        self.assertEqual(payload["scoring_mode"], "candidate_default")
        self.assertFalse(payload["borderline_recovery_enabled"])
        self.assertEqual(payload["scoring_config"]["scoring_mode"], "candidate_default")
        self.assertEqual(payload["scoring_config"]["pedigree_weight"], 0.2)
        self.assertEqual(payload["scoring_config"]["race_level_weight"], 1.2)
        self.assertEqual(payload["scoring_config"]["pace_weight"], 0.2)
        self.assertEqual(payload["scoring_config"]["conditional_weight_profile"], "candidate_default_v2")
        self.assertFalse(payload["scoring_config"]["use_market_score_in_ranking"])
        self.assertEqual(payload["scoring_config"]["market_signal_weight"], 0.0)
        self.assertFalse(payload["market_signal_config"]["use_market_score_in_ranking"])
        self.assertEqual(payload["market_signal_config"]["market_signal_weight"], 0.0)

        second_score = next(item for item in payload["horse_scores"] if item["horse_no"] == 2)
        breakdown = second_score["score_breakdown"]
        expected_total = round(
            second_score["base_total_score"]
            + breakdown["pedigree_adjustment_weighted"]
            + breakdown["race_level_adjustment_weighted"]
            + breakdown["pace_adjustment_weighted"],
            1,
        )
        self.assertEqual(second_score["total_score"], expected_total)
        self.assertEqual(breakdown["pedigree_adjustment_raw"], second_score["pedigree_adjustment"]["pedigree_adjustment"])
        self.assertEqual(breakdown["race_level_adjustment_raw"], second_score["race_level_adjustment"]["adjustment"])
        self.assertEqual(breakdown["pace_adjustment_raw"], second_score["pace_adjustment"]["adjustment"])
        self.assertEqual(breakdown["pace_adjustment_weighted"], 0.2)
        self.assertIn("PEDIGREE_USED", payload["strategy"]["reason_codes"])
        self.assertIn("RACE_LEVEL_USED", payload["strategy"]["reason_codes"])
        self.assertIn("PACE_USED", payload["strategy"]["reason_codes"])
        self.assertIn("血統面から+1.4補正。", second_score["reason"])
        self.assertIn("相手関係面から+1.0補正。", second_score["reason"])
        self.assertIn("展開面から+0.2補正。", second_score["reason"])
        self.assertNotIn("展開面から+0.0補正", second_score["reason"])

    def test_safe_baseline_profile_uses_base_only_without_recovery(self) -> None:
        with patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_deeply", return_value=[]):
            _, saved_path = run_analysis(
                race_data_path=self.race_data_path,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
                scoring_profile="safe_baseline",
            )

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["scoring_profile"], "safe_baseline")
        self.assertEqual(payload["scoring_mode"], "base_only")
        self.assertFalse(payload["borderline_recovery_enabled"])
        self.assertEqual(payload["scoring_config"]["scoring_mode"], "base_only")
        self.assertEqual(payload["scoring_config"]["pedigree_weight"], 0.0)
        self.assertEqual(payload["scoring_config"]["race_level_weight"], 0.0)
        self.assertEqual(payload["scoring_config"]["pace_weight"], 0.0)
        self.assertEqual(payload["scoring_config"]["conditional_weight_profile"], "none")
        self.assertFalse(payload["market_signal_config"]["use_market_score_in_ranking"])
        self.assertEqual(payload["market_signal_config"]["market_signal_weight"], 0.0)

    def test_central_top_choice_refinement_promotes_triple_anchor_candidate(self) -> None:
        payload = json.loads(self.race_data_path.read_text(encoding="utf-8"))
        payload["race_info"]["source"] = "central"
        payload["race_info"]["scope_key"] = "central"
        self.race_data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        mock_scores = {
            1: _mock_score(1, "ホースA", 40.0),
            2: _mock_score(2, "ホースB", 38.5),
            3: _mock_score(3, "ホースC", 37.8),
        }
        mock_deep = [
            HorseDeepAnalysis(
                horse_no=1,
                horse_name="ホースA",
                positive_flags=[],
                risk_flags=[],
                recent_form_summary="標準。",
                distance_analysis="標準。",
                course_analysis="標準。",
                track_condition_analysis="標準。",
                jockey_analysis="標準。",
                odds_analysis="市場情報は使わない。",
                overall_comment="標準評価。",
            ),
            HorseDeepAnalysis(
                horse_no=2,
                horse_name="ホースB",
                positive_flags=["RECENT_FORM_STABLE"],
                risk_flags=[],
                recent_form_summary="近走は安定。",
                distance_analysis="標準。",
                course_analysis="標準。",
                track_condition_analysis="標準。",
                jockey_analysis="標準。",
                odds_analysis="市場情報は使わない。",
                overall_comment="近走の安定感あり。",
            ),
        ]
        mock_race_levels = [
            RaceLevelAnalysis(horse_no=1, horse_name="ホースA", positive_flags=[], risk_flags=[], head_to_head_summary="", race_level_summary="", opponent_context_summary="", overall_comment="標準評価。", adjustment_hint=0.0),
            RaceLevelAnalysis(horse_no=2, horse_name="ホースB", positive_flags=["HEAD_TO_HEAD_POSITIVE"], risk_flags=[], head_to_head_summary="", race_level_summary="", opponent_context_summary="", overall_comment="相手関係では優位。", adjustment_hint=0.3),
        ]
        mock_pace_analyses = [
            HorsePaceAnalysis(horse_no=1, horse_name="ホースA", running_style="差し", position_stability="標準", positive_flags=[], risk_flags=[], overall_comment="標準評価。"),
            HorsePaceAnalysis(horse_no=2, horse_name="ホースB", running_style="先行", position_stability="安定", positive_flags=["PACE_FIT", "STALKER_ADVANTAGE"], risk_flags=[], overall_comment="展開利が見込める。"),
        ]
        mock_projection = RacePaceProjection(projected_pace="average", front_runner_count=1, stalker_count=3, closer_count=2, pace_comment="平均ペース想定。", favorable_styles=["先行", "差し"], risk_styles=[])

        with (
            patch("keiba_llm_agent.scoring.recent_run_scorer.score_horse_by_recent_runs", side_effect=lambda horse, race_info, lessons=None, **kwargs: mock_scores[horse.horse_no]),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_level_for_race", return_value=mock_race_levels),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_pace_for_race", return_value=(mock_pace_analyses, mock_projection)),
            patch("keiba_llm_agent.scoring.recent_run_scorer.build_pedigree_analyses_for_race", return_value=[]),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_deeply", return_value=mock_deep),
        ):
            _, saved_path = run_analysis(
                race_data_path=self.race_data_path,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
            )

        result = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(result["marks"]["◎"], 2)
        refined_score = next(item for item in result["horse_scores"] if item["horse_no"] == 2)
        self.assertGreater(refined_score["score_breakdown"]["top_choice_refinement_bonus"], 0)
        self.assertIn("中央場の本命補正", refined_score["reason"])


if __name__ == "__main__":
    unittest.main()

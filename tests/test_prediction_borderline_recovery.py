from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.main import run_analysis
from keiba_llm_agent.schemas.deep_analysis import HorseDeepAnalysis
from keiba_llm_agent.reports.report_generator import generate_prediction_report
from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.prediction import HorseScore, ScoreBreakdown
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis


def _mock_score(horse_no: int, total_score: float) -> HorseScore:
    return HorseScore(
        horse_no=horse_no,
        horse_name=f"ホース{horse_no}",
        scores=ScoreBreakdown(
            recent_form=7,
            distance_fit=7,
            course_fit=7,
            track_condition_fit=7,
            jockey_fit=7,
            odds_value=5,
            risk=-2,
        ),
        total_score=total_score,
        reason="近走内容を評価。",
    )


class PredictionBorderlineRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.race_data_path = self.temp_path / "race_data.json"
        self.prediction_path = self.temp_path / "prediction.json"
        self.lessons_path = self.temp_path / "lessons.json"
        self.lessons_path.write_text("[]\n", encoding="utf-8")
        race_data = {
            "race_info": {
                "race_id": "sample_borderline",
                "race_name": "境界補正確認特別",
                "race_date": "2026-05-17",
                "course": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "weather": "晴",
            },
            "horses": [
                {
                    "horse_no": idx,
                    "frame_no": idx,
                    "horse_id": f"h{idx}",
                    "horse_name": f"ホース{idx}",
                    "jockey": f"騎手{idx}",
                    "carried_weight": 56.0,
                    "odds": 8.0 if idx == 6 else 5.0 + idx,
                    "popularity": 3 if idx == 6 else idx,
                    "recent_runs": [{"race_id": f"20260101010{idx}", "date": "2026-04-01", "course": "東京", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 2, "field_size": 16, "jockey": f"騎手{idx}"}],
                }
                for idx in range(1, 7)
            ],
        }
        self.race_data_path.write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_candidate_default_enables_recovery_and_updates_marks(self) -> None:
        mock_scores = {
            1: _mock_score(1, 40.0),
            2: _mock_score(2, 39.0),
            3: _mock_score(3, 38.0),
            4: _mock_score(4, 37.0),
            5: _mock_score(5, 36.0),
            6: _mock_score(6, 35.4),
        }
        race_level = [
            RaceLevelAnalysis(
                horse_no=6,
                horse_name="ホース6",
                positive_flags=["HEAD_TO_HEAD_POSITIVE"],
                risk_flags=[],
                head_to_head_summary="",
                race_level_summary="",
                opponent_context_summary="",
                overall_comment="相手関係は優位。",
                adjustment_hint=0.0,
            )
        ]
        pace_analyses = [
            HorsePaceAnalysis(
                horse_no=6,
                horse_name="ホース6",
                running_style="差し",
                position_stability="安定",
                positive_flags=["PACE_FIT"],
                risk_flags=[],
                overall_comment="展開は向く。",
            )
        ]
        projection = RacePaceProjection(
            projected_pace="average",
            front_runner_count=1,
            stalker_count=3,
            closer_count=2,
            pace_comment="平均ペース想定。",
            favorable_styles=["先行", "差し"],
            risk_styles=[],
        )
        deep_analyses = [
            HorseDeepAnalysis(
                horse_no=6,
                horse_name="ホース6",
                positive_flags=["RECENT_FORM_STRONG"],
                risk_flags=[],
                recent_form_summary="",
                distance_analysis="",
                course_analysis="",
                track_condition_analysis="",
                jockey_analysis="",
                odds_analysis="",
                overall_comment="",
            )
        ]

        with (
            patch("keiba_llm_agent.scoring.recent_run_scorer.score_horse_by_recent_runs", side_effect=lambda horse, race_info, lessons=None, **kwargs: mock_scores[horse.horse_no]),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_level_for_race", return_value=race_level),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_pace_for_race", return_value=(pace_analyses, projection)),
            patch("keiba_llm_agent.scoring.recent_run_scorer.build_pedigree_analyses_for_race", return_value=[]),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_deeply", return_value=deep_analyses),
        ):
            prediction, saved_path = run_analysis(
                race_data_path=self.race_data_path,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
                borderline_recovery_enabled=True,
            )

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["borderline_recovery_config"]["enabled"])
        self.assertTrue(payload["borderline_recovery_result"]["recovery_applied"])
        recovered = next(item for item in payload["horse_scores"] if item["horse_no"] == 6)
        self.assertTrue(recovered["borderline_recovery"]["applied"])
        self.assertEqual(recovered["borderline_recovery"]["original_rank"], 6)
        self.assertEqual(recovered["borderline_recovery"]["new_rank"], 5)
        self.assertGreater(recovered["score_breakdown"]["borderline_recovery_bonus"], 0.0)
        self.assertGreater(recovered["score_breakdown"]["total_score_after_recovery"], recovered["score_breakdown"]["total_score"])
        self.assertIn("Top5境界補正", recovered["reason"])
        top5 = [horse["horse_no"] for horse in payload["horse_scores"][:5]]
        self.assertIn(6, top5)

        race_data = RaceData.from_json_file(self.race_data_path)
        markdown = generate_prediction_report(prediction, race_data=race_data)
        self.assertIn("## Top5境界補正", markdown)
        self.assertIn("ホース6", markdown)

    def test_base_only_defaults_to_disabled_recovery(self) -> None:
        with patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_deeply", return_value=[]):
            _, saved_path = run_analysis(
                race_data_path=self.race_data_path,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
                scoring_mode="base_only",
            )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertFalse(payload["borderline_recovery_config"]["enabled"])
        self.assertFalse(payload["borderline_recovery_result"]["recovery_applied"])


if __name__ == "__main__":
    unittest.main()

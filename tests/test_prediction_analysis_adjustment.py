from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.main import run_analysis
from keiba_llm_agent.reports.report_generator import generate_prediction_report
from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.prediction import HorseScore, ScoreBreakdown
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis


def _build_mock_score(horse_no: int, horse_name: str, total_score: float) -> HorseScore:
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


class PredictionAnalysisAdjustmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.race_data_path = self.temp_path / "race_data.json"
        self.prediction_path = self.temp_path / "prediction.json"
        self.lessons_path = self.temp_path / "lessons.json"
        self.lessons_path.write_text("[]\n", encoding="utf-8")
        race_data = {
            "race_info": {
                "race_id": "sample_adjustment",
                "race_name": "補正確認特別",
                "race_date": "2026-05-17",
                "course": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "weather": "晴",
            },
            "horses": [
                {
                    "horse_no": 1, "frame_no": 1, "horse_id": "h1", "horse_name": "ホースA", "jockey": "騎手A", "carried_weight": 56.0, "odds": 6.2, "popularity": 3,
                    "recent_runs": [{"race_id": "202601010101", "date": "2026-04-01", "course": "東京", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 2, "field_size": 16, "jockey": "騎手A"}],
                },
                {
                    "horse_no": 2, "frame_no": 2, "horse_id": "h2", "horse_name": "ホースB", "jockey": "騎手B", "carried_weight": 56.0, "odds": 9.8, "popularity": 5,
                    "recent_runs": [{"race_id": "202601010102", "date": "2026-04-01", "course": "東京", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 4, "field_size": 16, "jockey": "騎手B"}],
                },
                {
                    "horse_no": 3, "frame_no": 3, "horse_id": "h3", "horse_name": "ホースC", "jockey": "騎手C", "carried_weight": 56.0, "odds": 12.5, "popularity": 7,
                    "recent_runs": [{"race_id": "202601010103", "date": "2026-04-01", "course": "東京", "surface": "芝", "distance": 1600, "track_condition": "良", "finish": 5, "field_size": 16, "jockey": "騎手C"}],
                },
            ],
        }
        self.race_data_path.write_text(json.dumps(race_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prediction_includes_adjustments_breakdown_and_new_order(self) -> None:
        mock_scores = {
            1: _build_mock_score(1, "ホースA", 34.0),
            2: _build_mock_score(2, "ホースB", 33.8),
            3: _build_mock_score(3, "ホースC", 33.0),
        }
        mock_race_levels = [
            RaceLevelAnalysis(
                horse_no=1,
                horse_name="ホースA",
                positive_flags=[],
                risk_flags=["HEAD_TO_HEAD_NEGATIVE"],
                head_to_head_summary="",
                race_level_summary="",
                opponent_context_summary="",
                overall_comment="同組比較ではやや劣勢。",
                adjustment_hint=-0.6,
            ),
            RaceLevelAnalysis(
                horse_no=2,
                horse_name="ホースB",
                positive_flags=["HEAD_TO_HEAD_POSITIVE"],
                risk_flags=[],
                head_to_head_summary="",
                race_level_summary="",
                opponent_context_summary="",
                overall_comment="相手関係では優位。",
                adjustment_hint=0.8,
            ),
        ]
        mock_pace_analyses = [
            HorsePaceAnalysis(
                horse_no=1,
                horse_name="ホースA",
                running_style="逃げ",
                position_stability="不安定",
                positive_flags=[],
                risk_flags=["PACE_MISMATCH", "POSITION_UNSTABLE"],
                overall_comment="展開リスクあり。",
            ),
            HorsePaceAnalysis(
                horse_no=2,
                horse_name="ホースB",
                running_style="先行",
                position_stability="安定",
                positive_flags=["PACE_FIT", "STALKER_ADVANTAGE", "POSITION_STABLE"],
                risk_flags=[],
                overall_comment="展開利が見込める。",
            ),
            HorsePaceAnalysis(
                horse_no=3,
                horse_name="ホースC",
                running_style="差し",
                position_stability="安定",
                positive_flags=[],
                risk_flags=[],
                overall_comment="標準評価。",
            ),
        ]
        mock_projection = RacePaceProjection(
            projected_pace="average",
            front_runner_count=1,
            stalker_count=3,
            closer_count=2,
            pace_comment="平均ペース想定。",
            favorable_styles=["先行", "差し"],
            risk_styles=[],
        )

        with (
            patch("keiba_llm_agent.scoring.recent_run_scorer.score_horse_by_recent_runs", side_effect=lambda horse, race_info, lessons=None, **kwargs: mock_scores[horse.horse_no]),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_level_for_race", return_value=mock_race_levels),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_pace_for_race", return_value=(mock_pace_analyses, mock_projection)),
            patch("keiba_llm_agent.scoring.recent_run_scorer.build_pedigree_analyses_for_race", return_value=[]),
            patch("keiba_llm_agent.scoring.recent_run_scorer.analyze_race_deeply", return_value=[]),
        ):
            prediction, saved_path = run_analysis(
                race_data_path=self.race_data_path,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
            )

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["scoring_config"]["scoring_mode"], "candidate_default")
        first_score = payload["horse_scores"][0]
        self.assertIn("race_level_adjustment", first_score)
        self.assertIn("pace_adjustment", first_score)
        self.assertIn("score_breakdown", first_score)
        expected_total = round(first_score["score_breakdown"]["total_score"], 1)
        self.assertEqual(first_score["total_score"], expected_total)
        self.assertEqual(first_score["score_breakdown"]["total_score"], expected_total)
        self.assertEqual(first_score["score_breakdown"]["pace_adjustment_weighted"], 0.2)
        self.assertEqual(payload["marks"]["◎"], 2)
        self.assertIn("PEDIGREE_USED", payload["strategy"]["reason_codes"])
        self.assertIn("RACE_LEVEL_USED", payload["strategy"]["reason_codes"])
        self.assertIn("PACE_USED", payload["strategy"]["reason_codes"])

        race_data = RaceData.from_json_file(self.race_data_path)
        markdown = generate_prediction_report(prediction, race_data=race_data)
        self.assertIn("| 印 | 馬番 | 馬名 | base | pedigree | raceLv | pace | border_adj | total | odds(参考) | 人気(参考) | reason |", markdown)


if __name__ == "__main__":
    unittest.main()

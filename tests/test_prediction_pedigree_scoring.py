from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.main import run_analysis
from keiba_llm_agent.reports.report_generator import generate_prediction_report
from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis
from keiba_llm_agent.schemas.race_data import RaceData


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
SAMPLE_LESSONS = ROOT_DIR / "keiba_llm_agent" / "memory" / "lessons.json"
class PredictionPedigreeScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.lessons_path = self.temp_path / "lessons.json"
        self.prediction_path = self.temp_path / "prediction.json"
        shutil.copyfile(SAMPLE_LESSONS, self.lessons_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prediction_outputs_pedigree_adjusted_scores(self) -> None:
        mock_pedigree = [
            PedigreeAnalysis(
                horse_no=1,
                horse_name="サンプルホースA",
                sire="ハーツクライ",
                dam="サンプル母",
                damsire="キングカメハメハ",
                surface_tendency="芝",
                distance_tendency="中長距離",
                track_condition_tendency="パワー",
                pace_tendency="スタミナ",
                positive_flags=["PEDIGREE_SURFACE_FIT", "PEDIGREE_DISTANCE_FIT", "PEDIGREE_STAMINA_FIT"],
                risk_flags=[],
                overall_comment="父ハーツクライは芝・中長距離向き。",
            )
        ]
        with patch("keiba_llm_agent.scoring.recent_run_scorer.build_pedigree_analyses_for_race", return_value=mock_pedigree):
            prediction, saved_path = run_analysis(
                race_data_path=SAMPLE_RACE_DATA,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
            )

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        first_score = payload["horse_scores"][0]
        self.assertIn("base_total_score", first_score)
        self.assertIn("pedigree_adjustment", first_score)
        self.assertIn("race_level_adjustment", first_score)
        self.assertIn("pace_adjustment", first_score)
        self.assertIn("score_breakdown", first_score)
        expected_total = round(
            first_score["score_breakdown"]["total_score"],
            1,
        )
        self.assertEqual(first_score["total_score"], expected_total)
        self.assertIn("PEDIGREE_USED", payload["strategy"]["reason_codes"])

        race_data = RaceData.from_json_file(SAMPLE_RACE_DATA)
        markdown = generate_prediction_report(prediction, race_data=race_data)
        self.assertIn(
            "| 印 | 馬番 | 馬名 | base | pedigree | raceLv | pace | border_adj | total | odds(参考) | 人気(参考) | reason |",
            markdown,
        )


if __name__ == "__main__":
    unittest.main()

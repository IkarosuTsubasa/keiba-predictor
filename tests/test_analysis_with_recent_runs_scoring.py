from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.main import run_analysis


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
SAMPLE_LESSONS = ROOT_DIR / "keiba_llm_agent" / "memory" / "lessons.json"


class AnalysisWithRecentRunsScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.lessons_path = self.temp_path / "lessons.json"
        shutil.copyfile(SAMPLE_LESSONS, self.lessons_path)
        self.prediction_path = self.temp_path / "prediction.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_analysis_uses_recent_run_heuristic_scoring(self) -> None:
        prediction, saved_path = run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertTrue(saved_path.exists())
        self.assertTrue(any(score["total_score"] != 0 for score in payload["horse_scores"]))
        self.assertGreater(len({score["total_score"] for score in payload["horse_scores"]}), 1)
        self.assertTrue(any(score["scores"]["ability_score"] > 0 for score in payload["horse_scores"]))
        self.assertTrue(any(score["scores"]["recent_quality_score"] > 0 for score in payload["horse_scores"]))
        self.assertTrue(any(score["scores"]["condition_fit_score"] > 0 for score in payload["horse_scores"]))
        self.assertTrue(all("unknown" not in score["reason"] for score in payload["horse_scores"][:3]))
        self.assertIn("strategy", payload)
        self.assertIn(payload["strategy"]["bet_decision"], {"BET", "SKIP"})
        self.assertEqual(prediction.race_id, "sample_001")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.main import run_analysis


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
SAMPLE_LESSONS = ROOT_DIR / "keiba_llm_agent" / "memory" / "lessons.json"


class AnalysisStrategyOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.prediction_path = self.temp_path / "prediction.json"
        self.lessons_path = self.temp_path / "lessons.json"
        self.lessons_path.write_text(SAMPLE_LESSONS.read_text(encoding="utf-8"), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_analysis_outputs_strategy_and_consistent_bets(self) -> None:
        prediction, saved_path = run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertIn("strategy", payload)
        self.assertIn(payload["strategy"]["bet_decision"], {"BET", "SKIP"})
        self.assertIn(payload["strategy"]["confidence"], {"low", "medium", "high"})
        self.assertIn(payload["strategy"]["participation_level"], {"none", "light", "normal", "strong"})
        self.assertIn("買い判断=", payload["summary"])
        if payload["strategy"]["bet_decision"] == "SKIP":
            self.assertEqual(payload["bets"], [])
        else:
            self.assertGreaterEqual(len(payload["bets"]), 1)
        self.assertIsNotNone(prediction.strategy)


if __name__ == "__main__":
    unittest.main()

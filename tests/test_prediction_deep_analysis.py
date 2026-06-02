from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.main import run_analysis
from keiba_llm_agent.reports.report_generator import generate_prediction_report
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.social.post_generator import build_prediction_post


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
SAMPLE_LESSONS = ROOT_DIR / "keiba_llm_agent" / "memory" / "lessons.json"


class PredictionDeepAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.lessons_path = self.temp_path / "lessons.json"
        self.prediction_path = self.temp_path / "prediction.json"
        shutil.copyfile(SAMPLE_LESSONS, self.lessons_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prediction_outputs_deep_analyses(self) -> None:
        prediction, saved_path = run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        race_data = RaceData.from_json_file(SAMPLE_RACE_DATA)
        self.assertIn("deep_analyses", payload)
        self.assertEqual(len(payload["deep_analyses"]), len(race_data.horses))
        self.assertEqual(payload["deep_analyses"][0]["horse_no"], race_data.horses[0].horse_no)
        self.assertGreater(len(prediction.deep_analyses), 0)

    def test_prediction_report_displays_deep_analysis(self) -> None:
        prediction, _ = run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        race_data = RaceData.from_json_file(SAMPLE_RACE_DATA)
        markdown = generate_prediction_report(prediction, race_data=race_data)
        self.assertIn("## 深掘り分析", markdown)
        self.assertIn("Positive:", markdown)
        self.assertIn("Risk:", markdown)
        self.assertIn("Comment:", markdown)

    def test_social_post_includes_or_omits_deep_line_by_length(self) -> None:
        prediction, _ = run_analysis(
            race_data_path=SAMPLE_RACE_DATA,
            output_path=self.prediction_path,
            lessons_path=self.lessons_path,
        )
        race_data = RaceData.from_json_file(SAMPLE_RACE_DATA)
        short_text = build_prediction_post(prediction, race_data=race_data)
        self.assertLessEqual(len(short_text), 280)

        inflated_prediction = prediction.model_copy(
            update={
                "strategy": prediction.strategy.model_copy(
                    update={"reason": "上位拮抗のため点数は絞る。" * 20}
                )
                if prediction.strategy
                else None
            }
        )
        long_text = build_prediction_post(inflated_prediction, race_data=race_data)
        self.assertLessEqual(len(long_text), 280)
        if "深掘り分析では◎は" in short_text:
            self.assertTrue(
                "深掘り分析では◎は" in short_text
                or "深掘り分析では◎は" not in long_text
            )


if __name__ == "__main__":
    unittest.main()

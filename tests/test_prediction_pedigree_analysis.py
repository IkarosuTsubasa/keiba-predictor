from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.main import run_analysis
from keiba_llm_agent.reports.report_generator import generate_prediction_report
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.social.post_generator import build_prediction_post


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
SAMPLE_LESSONS = ROOT_DIR / "keiba_llm_agent" / "memory" / "lessons.json"
HORSE_FIXTURE = ROOT_DIR / "tests" / "fixtures" / "netkeiba_horse_sample.html"


class PredictionPedigreeAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.lessons_path = self.temp_path / "lessons.json"
        self.prediction_path = self.temp_path / "prediction.json"
        shutil.copyfile(SAMPLE_LESSONS, self.lessons_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prediction_outputs_pedigree_analyses_and_report_social(self) -> None:
        fixture_html = HORSE_FIXTURE.read_text(encoding="utf-8")
        with patch("keiba_llm_agent.pedigree.pedigree_analyzer.fetch_horse_html", return_value=fixture_html):
            prediction, saved_path = run_analysis(
                race_data_path=SAMPLE_RACE_DATA,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
            )
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        race_data = RaceData.from_json_file(SAMPLE_RACE_DATA)
        self.assertIn("pedigree_analyses", payload)
        self.assertEqual(len(payload["pedigree_analyses"]), len(race_data.horses))
        self.assertEqual(payload["pedigree_analyses"][0]["horse_no"], race_data.horses[0].horse_no)

        markdown = generate_prediction_report(prediction, race_data=race_data)
        self.assertIn("## 血統分析", markdown)
        self.assertIn("父:", markdown)
        self.assertIn("母父:", markdown)
        self.assertIn("血統所見:", prediction.summary)

        text = build_prediction_post(prediction, race_data=race_data)
        self.assertLessEqual(len(text), 280)


if __name__ == "__main__":
    unittest.main()

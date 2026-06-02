from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.fetchers import netkeiba_result_fetcher
from keiba_llm_agent.schemas.result import ResultData


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_LESSONS = ROOT_DIR / "keiba_llm_agent" / "memory" / "lessons.json"
SAMPLE_RESULT_DATA = ResultData.model_validate(
    {
        "race_id": "sample_001",
        "result": {"1st": 1, "2nd": 2, "3rd": 3},
        "payouts": [{"type": "wide", "combination": "1-2", "payout": 420}],
    }
)


class ReviewUrlCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.predictions_dir = self.temp_path / "predictions"
        self.results_dir = self.temp_path / "results"
        self.reviews_dir = self.temp_path / "reviews"
        self.race_data_dir = self.temp_path / "race_data"
        self.lessons_path = self.temp_path / "lessons.json"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.race_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SAMPLE_LESSONS, self.lessons_path)

        prediction_payload = {
            "race_id": "sample_001",
            "race_info": {
                "race_id": "sample_001",
                "race_name": "Sample Stakes",
                "race_date": "2026-05-18",
                "course": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "weather": "晴",
            },
            "marks": {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
            "horse_scores": [],
            "bets": [
                {
                    "bet_type": "ワイド",
                    "horse_numbers": [1, 2],
                    "amount": 100,
                    "reason": "reason",
                }
            ],
            "summary": "summary",
            "risks": [],
            "used_lessons": [],
            "strategy": {
                "bet_decision": "BET",
                "confidence": "medium",
                "participation_level": "light",
                "reason_codes": [],
                "reason": "reason",
            },
        }
        (self.predictions_dir / "sample_001.json").write_text(
            json.dumps(prediction_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.race_data_dir / "sample_001.json").write_text(
            json.dumps(
                {
                    "race_info": {
                        "race_id": "sample_001",
                        "race_name": "Sample Stakes",
                        "race_date": "2026-05-18",
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "weather": "晴",
                    },
                    "horses": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.predictions_patch = patch.object(main_module, "DEFAULT_PREDICTIONS_DIR", self.predictions_dir)
        self.reviews_patch = patch.object(main_module, "DEFAULT_REVIEWS_DIR", self.reviews_dir)
        self.race_data_patch = patch.object(main_module, "DEFAULT_RACE_DATA_DIR", self.race_data_dir)
        self.results_patch = patch.object(netkeiba_result_fetcher, "RESULTS_DIR", self.results_dir)
        self.predictions_patch.start()
        self.reviews_patch.start()
        self.race_data_patch.start()
        self.results_patch.start()

    def tearDown(self) -> None:
        self.predictions_patch.stop()
        self.reviews_patch.stop()
        self.race_data_patch.stop()
        self.results_patch.stop()
        self.temp_dir.cleanup()

    def test_review_url_generates_review_and_appends_lessons(self) -> None:
        with patch.object(main_module, "fetch_and_parse_netkeiba_result", return_value=SAMPLE_RESULT_DATA):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                result = main_module.run_review_url(
                    "https://race.netkeiba.com/race/result.html?race_id=sample_001",
                    lessons_path=self.lessons_path,
                )

        self.assertEqual(result["race_id"], "sample_001")
        self.assertTrue(Path(result["result_path"]).exists())
        self.assertTrue(Path(result["review_path"]).exists())
        review_payload = json.loads(Path(result["review_path"]).read_text(encoding="utf-8"))
        self.assertIn("bet_results", review_payload)
        self.assertEqual(review_payload["hit_summary"]["total_stake"], 100)
        self.assertEqual(review_payload["hit_summary"]["total_return"], 420)
        self.assertEqual(review_payload["hit_summary"]["roi"], 4.2)
        lesson = review_payload["lessons"][0]
        self.assertEqual(lesson["course"], "東京")
        self.assertEqual(lesson["surface"], "芝")
        self.assertEqual(lesson["distance"], 1600)
        self.assertEqual(lesson["track_condition"], "良")
        self.assertEqual(lesson["source_race_id"], "sample_001")
        self.assertTrue(review_payload["bet_results"][0]["hit"])
        lessons = json.loads(self.lessons_path.read_text(encoding="utf-8"))
        self.assertGreater(len(lessons), 0)
        matched = [
            lesson
            for lesson in lessons
            if lesson["course"] == "東京"
            and lesson["surface"] == "芝"
            and lesson["distance"] == 1600
            and lesson["track_condition"] == "良"
        ]
        self.assertTrue(matched)
        self.assertEqual(matched[0]["course"], "東京")
        self.assertIn("lesson_id", matched[0])
        self.assertIn("score", matched[0])
        self.assertEqual(result["bet_hit"], True)
        self.assertEqual(result["total_stake"], 100)
        self.assertEqual(result["total_return"], 420)
        self.assertEqual(result["roi"], 4.2)

    def test_review_url_raises_when_prediction_missing(self) -> None:
        (self.predictions_dir / "sample_001.json").unlink()
        with patch.object(main_module, "fetch_and_parse_netkeiba_result", return_value=SAMPLE_RESULT_DATA):
            with self.assertRaisesRegex(
                ValueError,
                "prediction not found for race_id=sample_001. Run analyze-url first.",
            ):
                main_module.run_review_url(
                    "https://race.netkeiba.com/race/result.html?race_id=sample_001",
                    lessons_path=self.lessons_path,
                )

    def test_review_url_falls_back_to_prediction_race_info_when_race_data_missing(self) -> None:
        (self.race_data_dir / "sample_001.json").unlink()
        with patch.object(main_module, "fetch_and_parse_netkeiba_result", return_value=SAMPLE_RESULT_DATA):
            result = main_module.run_review_url(
                "https://race.netkeiba.com/race/result.html?race_id=sample_001",
                lessons_path=self.lessons_path,
            )

        review_payload = json.loads(Path(result["review_path"]).read_text(encoding="utf-8"))
        lesson = review_payload["lessons"][0]
        self.assertEqual(lesson["course"], "東京")
        self.assertEqual(lesson["surface"], "芝")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.fetchers import netkeiba_fetcher, netkeiba_result_fetcher
from keiba_llm_agent.schemas.race_data import RaceData, RecentRun
from keiba_llm_agent.schemas.result import ResultData


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA_PATH = ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
SAMPLE_RACE_DATA = RaceData.from_json_file(SAMPLE_RACE_DATA_PATH)
PREDICT_URL = "https://race.netkeiba.com/race/shutuba.html?race_id=sample_001"
REVIEW_URL = "https://race.netkeiba.com/race/result.html?race_id=sample_001"


def _enriched_race_data() -> RaceData:
    return SAMPLE_RACE_DATA.model_copy(
        update={
            "horses": [
                horse.model_copy(
                    update={
                        "odds": 5.8 + index,
                        "popularity": index + 1,
                        "recent_runs": [
                            RecentRun(
                                race_id=f"2026000000{index}",
                                date="2026-05-01",
                                course="東京",
                                surface="芝",
                                distance=1600,
                                track_condition="良",
                                finish=2,
                                field_size=16,
                                jockey=horse.jockey,
                                odds=6.0,
                                popularity=2,
                            )
                        ],
                    }
                )
                for index, horse in enumerate(SAMPLE_RACE_DATA.horses)
            ]
        }
    )


SAMPLE_RESULT_DATA = ResultData.model_validate(
    {
        "race_id": "sample_001",
        "result": {"1st": 1, "2nd": 2, "3rd": 3},
        "payouts": [{"type": "wide", "combination": "1-2", "payout": 420}],
    }
)


class PipelineCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.race_data_dir = self.temp_path / "race_data"
        self.predictions_dir = self.temp_path / "predictions"
        self.results_dir = self.temp_path / "results"
        self.reviews_dir = self.temp_path / "reviews"
        self.reports_dir = self.temp_path / "reports"
        self.social_posts_dir = self.temp_path / "social_posts"
        self.lessons_path = self.temp_path / "lessons.json"
        for directory in (
            self.race_data_dir,
            self.predictions_dir,
            self.results_dir,
            self.reviews_dir,
            self.reports_dir,
            self.social_posts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.lessons_path.write_text("[]\n", encoding="utf-8")

        self.patches = [
            patch.object(netkeiba_fetcher, "RACE_DATA_DIR", self.race_data_dir),
            patch.object(netkeiba_result_fetcher, "RESULTS_DIR", self.results_dir),
            patch.object(main_module, "DEFAULT_PREDICTIONS_DIR", self.predictions_dir),
            patch.object(main_module, "DEFAULT_RESULTS_DIR", self.results_dir),
            patch.object(main_module, "DEFAULT_REVIEWS_DIR", self.reviews_dir),
            patch.object(main_module, "DEFAULT_RACE_DATA_DIR", self.race_data_dir),
            patch.object(main_module, "DEFAULT_REPORTS_DIR", self.reports_dir),
            patch.object(main_module, "DEFAULT_SOCIAL_POSTS_DIR", self.social_posts_dir),
            patch.object(main_module, "DEFAULT_LESSONS_PATH", self.lessons_path),
        ]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    def test_predict_race_default_skips_report_and_keeps_social(self) -> None:
        enriched = _enriched_race_data()
        with (
            patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=SAMPLE_RACE_DATA),
            patch.object(main_module, "enrich_race_data_with_recent_runs", return_value=enriched),
        ):
            result = main_module.run_predict_race(PREDICT_URL, lessons_path=self.lessons_path)

        self.assertTrue(Path(result["race_data_path"]).exists())
        self.assertTrue(Path(result["prediction_path"]).exists())
        self.assertIsNone(result["prediction_report_path"])
        self.assertFalse((self.reports_dir / "sample_001_prediction.md").exists())
        self.assertTrue(Path(result["prediction_social_post_path"]).exists())

    def test_predict_race_explicitly_enables_report(self) -> None:
        with (
            patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=SAMPLE_RACE_DATA),
            patch.object(main_module, "enrich_race_data_with_recent_runs", return_value=_enriched_race_data()),
        ):
            result = main_module.run_predict_race(
                PREDICT_URL,
                skip_report=False,
                lessons_path=self.lessons_path,
            )

        self.assertTrue(Path(result["prediction_report_path"]).exists())
        self.assertTrue(Path(result["prediction_social_post_path"]).exists())

    def test_predict_race_skip_social_does_not_generate_social_post(self) -> None:
        with (
            patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=SAMPLE_RACE_DATA),
            patch.object(main_module, "enrich_race_data_with_recent_runs", return_value=_enriched_race_data()),
        ):
            result = main_module.run_predict_race(
                PREDICT_URL,
                skip_report=False,
                skip_social=True,
                lessons_path=self.lessons_path,
            )

        self.assertIsNone(result["prediction_social_post_path"])
        self.assertFalse((self.social_posts_dir / "sample_001_prediction.txt").exists())
        self.assertTrue(Path(result["prediction_report_path"]).exists())

    def test_review_race_explicitly_generates_result_review_report_and_social(self) -> None:
        enriched = _enriched_race_data()
        with (
            patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=SAMPLE_RACE_DATA),
            patch.object(main_module, "enrich_race_data_with_recent_runs", return_value=enriched),
        ):
            main_module.run_predict_race(PREDICT_URL, lessons_path=self.lessons_path)

        with patch.object(main_module, "fetch_and_parse_netkeiba_result", return_value=SAMPLE_RESULT_DATA):
            result = main_module.run_review_race(REVIEW_URL, skip_report=False, lessons_path=self.lessons_path)

        self.assertTrue(Path(result["result_path"]).exists())
        self.assertTrue(Path(result["review_path"]).exists())
        self.assertTrue(Path(result["review_report_path"]).exists())
        self.assertTrue(Path(result["review_social_post_path"]).exists())

    def test_review_race_raises_when_prediction_missing(self) -> None:
        with patch.object(main_module, "fetch_and_parse_netkeiba_result", return_value=SAMPLE_RESULT_DATA):
            with self.assertRaisesRegex(
                ValueError,
                "prediction not found for race_id=sample_001. Run predict-race first.",
            ):
                main_module.run_review_race(REVIEW_URL, lessons_path=self.lessons_path)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent.backtest.audit_snapshot import read_backtest_audit_payload
from keiba_llm_agent import main as main_module
from keiba_llm_agent.fetchers import netkeiba_fetcher, netkeiba_result_fetcher
from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RecentRun
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
        self.backtest_audit_dir = self.temp_path / "backtest_audit"
        self.lessons_path = self.temp_path / "lessons.json"
        for directory in (
            self.race_data_dir,
            self.predictions_dir,
            self.results_dir,
            self.reviews_dir,
            self.reports_dir,
            self.social_posts_dir,
            self.backtest_audit_dir,
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
            patch.object(main_module, "DEFAULT_BACKTEST_AUDIT_DIR", self.backtest_audit_dir),
            patch.object(main_module, "DEFAULT_LESSONS_PATH", self.lessons_path),
        ]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    def test_predict_race_blocks_non_debut_races_with_no_valid_history(self) -> None:
        old_prediction = self.predictions_dir / "sample_001.json"
        old_prediction.write_text('{"existing": true}', encoding="utf-8")
        for scope, surface in (("central_turf", "\u829d"), ("central_dirt", "\u30c0\u30fc\u30c8"),
                               ("local", "\u30c0\u30fc\u30c8")):
            for runs in ([], [RecentRun(date="2026-05-01")], [RecentRun(date="2026-05-01", finish=0)]):
                race = RaceData(
                    race_info=SAMPLE_RACE_DATA.race_info.model_copy(update={
                        "scope_key": scope, "surface": surface, "race_name": "1\u52dd\u30af\u30e9\u30b9",
                    }),
                    horses=[HorseEntry(horse_no=1, horse_name="sample", recent_runs=runs)],
                )
                with (self.subTest(scope=scope, runs=runs),
                      patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=race),
                      patch.object(main_module, "enrich_race_data_with_recent_runs", return_value=race),
                      patch.object(main_module, "run_analysis") as analyze,
                      patch.object(main_module, "save_race_data") as save,
                      patch.object(main_module, "_capture_prediction_supplements") as capture):
                    with self.assertRaisesRegex(RuntimeError, "no_valid_recent_runs: race_id=sample_001"):
                        main_module.run_predict_race(PREDICT_URL, lessons_path=self.lessons_path)
                    analyze.assert_not_called()
                    save.assert_not_called()
                    capture.assert_not_called()
        self.assertEqual(old_prediction.read_text(encoding="utf-8"), '{"existing": true}')
        self.assertEqual(list(self.backtest_audit_dir.iterdir()), [])

    def test_predict_race_blocks_empty_race_card(self) -> None:
        race = SAMPLE_RACE_DATA.model_copy(update={"horses": []})
        with (patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=race),
              patch.object(main_module, "run_analysis") as analyze):
            with self.assertRaisesRegex(RuntimeError, "empty_race_card"):
                main_module.run_predict_race(PREDICT_URL, lessons_path=self.lessons_path)
        analyze.assert_not_called()

    def test_predict_race_allows_debut_and_partially_available_history(self) -> None:
        for name, race_class, finish in (("2\u6b73\u65b0\u99ac", None, None), ("sample", "\u65b0\u99ac", None),
                                         ("1\u52dd\u30af\u30e9\u30b9", None, 2)):
            race = RaceData(
                race_info=SAMPLE_RACE_DATA.race_info.model_copy(update={"race_name": name, "race_class": race_class}),
                horses=[HorseEntry(horse_no=1, horse_name="sample1",
                                   recent_runs=[RecentRun(date="2026-05-01", finish=finish)] if finish else []),
                        HorseEntry(horse_no=2, horse_name="sample2")],
            )
            with (self.subTest(name=name, race_class=race_class),
                  patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=race),
                  patch.object(main_module, "enrich_race_data_with_recent_runs", return_value=race),
                  patch.object(main_module, "_capture_prediction_supplements"),
                  patch.object(main_module, "run_analysis", return_value=(None, self.predictions_dir / "sample_001.json")) as analyze):
                main_module.run_predict_race(PREDICT_URL, lessons_path=self.lessons_path, skip_social=True)
                analyze.assert_called_once()

    def test_predict_race_default_skips_report_and_keeps_social(self) -> None:
        enriched = _enriched_race_data()
        with (
            patch.object(main_module, "fetch_and_parse_netkeiba_race", return_value=SAMPLE_RACE_DATA),
            patch.object(main_module, "enrich_race_data_with_recent_runs", return_value=enriched),
        ):
            result = main_module.run_predict_race(PREDICT_URL, lessons_path=self.lessons_path)

        self.assertTrue(Path(result["race_data_path"]).exists())
        self.assertTrue(Path(result["prediction_path"]).exists())
        self.assertTrue(Path(result["backtest_audit_path"]).exists())
        audit_payload = read_backtest_audit_payload(result["backtest_audit_path"])
        self.assertEqual(audit_payload["audit_kind"], "full")
        self.assertTrue(audit_payload["backtest_capabilities"]["can_recompute_from_recent_runs"])
        self.assertEqual(audit_payload["inputs"]["race_data"]["race_info"]["race_id"], "sample_001")
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

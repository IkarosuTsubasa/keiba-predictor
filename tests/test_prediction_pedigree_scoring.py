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

    def test_unraced_debut_horse_uses_emphasized_pedigree_weight(self) -> None:
        race_data_path = self.temp_path / "debut_race_data.json"
        race_data_path.write_text(
            json.dumps(
                {
                    "race_info": {
                        "race_id": "202630061105",
                        "race_name": "JRA認定競走フレッシュチャレンジ競走(2歳)",
                        "race_date": "2026-06-11",
                        "course": "門別",
                        "surface": "ダート",
                        "distance": 1100,
                        "track_condition": "良",
                        "weather": "晴",
                        "source": "local",
                        "scope_key": "local",
                    },
                    "horses": [
                        {
                            "horse_no": 1,
                            "horse_id": "h1",
                            "horse_name": "血統評価馬",
                            "jockey": "騎手A",
                            "recent_runs": [],
                        },
                        {
                            "horse_no": 2,
                            "horse_id": "h2",
                            "horse_name": "中立評価馬",
                            "jockey": "騎手B",
                            "recent_runs": [],
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        mock_pedigree = [
            PedigreeAnalysis(
                horse_no=1,
                horse_name="血統評価馬",
                sire="ルヴァンスレーヴ",
                dam="サンプル母",
                damsire="バトルプラン",
                surface_tendency="ダート",
                distance_tendency="短距離〜マイル",
                track_condition_tendency="パワー",
                pace_tendency="スピード",
                positive_flags=[
                    "PEDIGREE_SURFACE_FIT",
                    "PEDIGREE_DISTANCE_FIT",
                    "PEDIGREE_POWER_FIT",
                ],
                risk_flags=[],
                overall_comment="ダート短距離向き。",
            ),
            PedigreeAnalysis(
                horse_no=2,
                horse_name="中立評価馬",
                sire="unknown",
                dam="サンプル母",
                damsire=None,
                surface_tendency="unknown",
                distance_tendency="unknown",
                track_condition_tendency="unknown",
                pace_tendency="unknown",
                positive_flags=[],
                risk_flags=["PEDIGREE_SURFACE_UNKNOWN", "PEDIGREE_DISTANCE_UNKNOWN"],
                overall_comment="血統面はunknown。",
            ),
        ]
        with patch(
            "keiba_llm_agent.scoring.recent_run_scorer.build_pedigree_analyses_for_race",
            return_value=mock_pedigree,
        ):
            _, saved_path = run_analysis(
                race_data_path=race_data_path,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
            )

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        top_score = payload["horse_scores"][0]
        self.assertEqual(top_score["horse_no"], 1)
        self.assertEqual(top_score["base_total_score"], 6.0)
        self.assertEqual(top_score["score_breakdown"]["pedigree_weight"], 2.0)
        self.assertGreater(top_score["score_breakdown"]["pedigree_adjustment_weighted"], 0.0)
        self.assertGreater(top_score["total_score"], top_score["base_total_score"])

    def test_one_run_horse_uses_low_sample_pedigree_weight(self) -> None:
        race_data_path = self.temp_path / "one_run_race_data.json"
        race_data_path.write_text(
            json.dumps(
                {
                    "race_info": {
                        "race_id": "sample_one_run",
                        "race_name": "低サンプル確認戦",
                        "race_date": "2026-06-11",
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "weather": "晴",
                    },
                    "horses": [
                        {
                            "horse_no": 1,
                            "horse_id": "h1",
                            "horse_name": "低サンプル馬",
                            "jockey": "騎手A",
                            "recent_runs": [
                                {
                                    "race_id": "202605010101",
                                    "date": "2026-05-01",
                                    "course": "中山",
                                    "surface": "芝",
                                    "distance": 1400,
                                    "track_condition": "良",
                                    "finish": 6,
                                    "field_size": 16,
                                    "jockey": "騎手B",
                                }
                            ],
                        },
                        {
                            "horse_no": 2,
                            "horse_id": "h2",
                            "horse_name": "通常馬",
                            "jockey": "騎手C",
                            "recent_runs": [
                                {
                                    "race_id": f"20260501010{index}",
                                    "date": "2026-05-01",
                                    "course": "東京",
                                    "surface": "芝",
                                    "distance": 1600,
                                    "track_condition": "良",
                                    "finish": 4,
                                    "field_size": 16,
                                    "jockey": "騎手C",
                                }
                                for index in range(1, 6)
                            ],
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        mock_pedigree = [
            PedigreeAnalysis(
                horse_no=1,
                horse_name="低サンプル馬",
                sire="キズナ",
                dam="サンプル母",
                damsire="キングカメハメハ",
                surface_tendency="芝",
                distance_tendency="マイル〜中距離",
                track_condition_tendency="パワー",
                pace_tendency="持続力",
                positive_flags=["PEDIGREE_SURFACE_FIT", "PEDIGREE_DISTANCE_FIT"],
                risk_flags=[],
                overall_comment="血統面の後押しあり。",
            ),
            PedigreeAnalysis(
                horse_no=2,
                horse_name="通常馬",
                sire="unknown",
                dam="サンプル母",
                damsire=None,
                surface_tendency="unknown",
                distance_tendency="unknown",
                track_condition_tendency="unknown",
                pace_tendency="unknown",
                positive_flags=[],
                risk_flags=[],
                overall_comment="血統面は中立。",
            ),
        ]
        with patch(
            "keiba_llm_agent.scoring.recent_run_scorer.build_pedigree_analyses_for_race",
            return_value=mock_pedigree,
        ):
            _, saved_path = run_analysis(
                race_data_path=race_data_path,
                output_path=self.prediction_path,
                lessons_path=self.lessons_path,
            )

        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        low_sample_score = next(item for item in payload["horse_scores"] if item["horse_no"] == 1)
        self.assertEqual(low_sample_score["score_breakdown"]["pedigree_weight"], 1.7)
        self.assertGreaterEqual(low_sample_score["scores"]["distance_fit"], 7)
        self.assertGreaterEqual(low_sample_score["scores"]["course_fit"], 6)


if __name__ == "__main__":
    unittest.main()

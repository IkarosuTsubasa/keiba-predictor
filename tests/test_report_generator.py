from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.reports.report_generator import (
    generate_prediction_report,
    generate_review_report,
)
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


def _build_prediction() -> Prediction:
    return Prediction.model_validate(
        {
            "race_id": "202605020811",
            "race_info": {
                "race_id": "202605020811",
                "race_name": "ヴィクトリアマイル",
                "race_date": "2026-05-17",
                "course": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "weather": "晴",
            },
            "marks": {"◎": 16, "○": 6, "▲": 15, "△": 8, "☆": 12},
            "horse_scores": [
                {
                    "horse_no": 16,
                    "horse_name": "ニシノティアモ",
                    "scores": {
                        "recent_form": 9,
                        "distance_fit": 6,
                        "course_fit": 4,
                        "track_condition_fit": 6,
                        "jockey_fit": 9,
                        "odds_value": 8,
                        "risk": -1,
                    },
                    "total_score": 42.8,
                    "reason": "近5走の安定度と同条件適性を評価。",
                },
                {
                    "horse_no": 6,
                    "horse_name": "ラヴァンダ",
                    "scores": {
                        "recent_form": 5,
                        "distance_fit": 10,
                        "course_fit": 5,
                        "track_condition_fit": 7,
                        "jockey_fit": 7,
                        "odds_value": 9,
                        "risk": -3,
                    },
                    "total_score": 41.2,
                    "reason": "同距離1600mの実績を評価。",
                },
                {
                    "horse_no": 15,
                    "horse_name": "アイサンサン",
                    "scores": {
                        "recent_form": 7,
                        "distance_fit": 8,
                        "course_fit": 3,
                        "track_condition_fit": 8,
                        "jockey_fit": 6,
                        "odds_value": 9,
                        "risk": -1,
                    },
                    "total_score": 41.2,
                    "reason": "近5走で1着3回。",
                },
                {
                    "horse_no": 8,
                    "horse_name": "カムニャック",
                    "scores": {
                        "recent_form": 7,
                        "distance_fit": 6,
                        "course_fit": 3,
                        "track_condition_fit": 9,
                        "jockey_fit": 6,
                        "odds_value": 5,
                        "risk": -1,
                    },
                    "total_score": 36.5,
                    "reason": "東京実績あり。",
                },
                {
                    "horse_no": 12,
                    "horse_name": "エンブロイダリー",
                    "scores": {
                        "recent_form": 7,
                        "distance_fit": 7,
                        "course_fit": 2,
                        "track_condition_fit": 7,
                        "jockey_fit": 7,
                        "odds_value": 4,
                        "risk": -1,
                    },
                    "total_score": 35.2,
                    "reason": "同距離で好走歴あり。",
                },
            ],
            "bets": [
                {
                    "bet_type": "ワイド",
                    "horse_numbers": [16, 6],
                    "amount": 100,
                    "reason": "上位拮抗のため、軽めにワイド1点。",
                }
            ],
            "summary": "買い判断=BET、confidence=medium。",
            "risks": ["heuristic scoringを使用しており、正式なML modelではない。"],
            "used_lessons": [
                {
                    "course": "東京",
                    "surface": "芝",
                    "distance": 1600,
                    "track_condition": "良",
                    "lesson": "同条件では近走の同距離・同コース実績を優先して評価する。",
                    "confidence": "medium",
                    "source_race_id": "202605020811",
                }
            ],
            "strategy": {
                "bet_decision": "BET",
                "confidence": "medium",
                "participation_level": "light",
                "reason_codes": ["LESSON_USED", "HEURISTIC_MODEL_ONLY"],
                "reason": "上位拮抗のためワイド中心。",
            },
        }
    )


def _build_review() -> Review:
    return Review.model_validate(
        {
            "race_id": "202605020811",
            "hit_summary": {
                "main_mark_top3": False,
                "marked_horses_top3_count": 2,
                "bet_hit": False,
                "roi": 0.0,
                "total_stake": 100,
                "total_return": 0,
            },
            "bet_results": [
                {
                    "bet_type": "ワイド",
                    "horse_numbers": [16, 6],
                    "amount": 100,
                    "hit": False,
                    "payout": 0,
                    "return_amount": 0,
                }
            ],
            "good_points": ["印上位と実着順の整合性がありました。"],
            "bad_points": ["本命印が3着以内を外しました。"],
            "lessons": [
                {
                    "course": "東京",
                    "surface": "芝",
                    "distance": 1600,
                    "track_condition": "良",
                    "lesson": "同条件では近走の同距離・同コース実績を優先して評価する。",
                    "confidence": "medium",
                    "source_race_id": "202605020811",
                }
            ],
        }
    )


def _build_race_data() -> RaceData:
    return RaceData.model_validate(
        {
            "race_info": {
                "race_id": "202605020811",
                "race_name": "ヴィクトリアマイル",
                "race_date": "2026-05-17",
                "course": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "weather": "晴",
            },
            "horses": [
                {"horse_no": 16, "frame_no": 8, "horse_id": "h16", "horse_name": "ニシノティアモ", "jockey": "騎手A", "carried_weight": 55.0, "odds": 14.9, "popularity": 4, "recent_runs": []},
                {"horse_no": 6, "frame_no": 3, "horse_id": "h6", "horse_name": "ラヴァンダ", "jockey": "騎手B", "carried_weight": 55.0, "odds": 23.3, "popularity": 8, "recent_runs": []},
                {"horse_no": 15, "frame_no": 8, "horse_id": "h15", "horse_name": "アイサンサン", "jockey": "騎手C", "carried_weight": 55.0, "odds": 49.4, "popularity": 11, "recent_runs": []},
                {"horse_no": 8, "frame_no": 4, "horse_id": "h8", "horse_name": "カムニャック", "jockey": "騎手D", "carried_weight": 55.0, "odds": 8.8, "popularity": 3, "recent_runs": []},
                {"horse_no": 12, "frame_no": 6, "horse_id": "h12", "horse_name": "エンブロイダリー", "jockey": "騎手E", "carried_weight": 55.0, "odds": 5.4, "popularity": 2, "recent_runs": []},
                {"horse_no": 7, "frame_no": 4, "horse_id": "h7", "horse_name": "クイーンズウォーク", "jockey": "騎手F", "carried_weight": 55.0, "odds": 12.1, "popularity": 5, "recent_runs": []},
            ],
        }
    )


def _build_result_data() -> ResultData:
    return ResultData.model_validate(
        {
            "race_id": "202605020811",
            "result": {"1st": 12, "2nd": 8, "3rd": 7},
            "payouts": [],
        }
    )


class ReportGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = _build_prediction()
        self.review = _build_review()
        self.race_data = _build_race_data()
        self.result_data = _build_result_data()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.predictions_dir = self.temp_path / "predictions"
        self.reviews_dir = self.temp_path / "reviews"
        self.reports_dir = self.temp_path / "reports"
        self.race_data_dir = self.temp_path / "race_data"
        self.results_dir = self.temp_path / "results"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.reviews_dir.mkdir(parents=True, exist_ok=True)
        self.race_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.predictions_dir / "202605020811.json").write_text(
            json.dumps(self.prediction.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.reviews_dir / "202605020811.json").write_text(
            json.dumps(self.review.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.race_data_dir / "202605020811.json").write_text(
            json.dumps(self.race_data.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.results_dir / "202605020811.json").write_text(
            json.dumps(self.result_data.model_dump(by_alias=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prediction_report_contains_race_name_mark_strategy_and_bets(self) -> None:
        markdown = generate_prediction_report(self.prediction, race_data=self.race_data)
        self.assertIn("ヴィクトリアマイル", markdown)
        self.assertIn("◎ 16 ニシノティアモ", markdown)
        self.assertIn("BET", markdown)
        self.assertIn("ワイド 16-6 100円", markdown)
        self.assertIn("## Scoring Profile", markdown)
        self.assertIn("odds / 人気: 参考情報", markdown)
        self.assertIn("odds(参考)", markdown)
        self.assertIn("scoring_profile: accuracy_default", markdown)
        self.assertIn("scoring_mode: candidate_default", markdown)
        self.assertIn("borderline_recovery_enabled: true", markdown)
        self.assertIn("| 印 | 馬番 | 馬名 | base | pedigree | raceLv | pace | border_adj | total | odds(参考) | 人気(参考) | reason |", markdown)
        self.assertIn("14.9", markdown)
        self.assertIn("| 4 |", markdown)
        self.assertIn("## Top5境界補正", markdown)

    def test_review_report_contains_roi_and_bet_results(self) -> None:
        markdown = generate_review_report(
            self.prediction,
            self.review,
            result_data=self.result_data,
            race_data=self.race_data,
        )
        self.assertIn("ROI", markdown)
        self.assertIn("買い目結果", markdown)
        self.assertIn("ワイド", markdown)
        self.assertIn("本命印が3着以内を外しました。", markdown)
        self.assertIn("1着: 12 エンブロイダリー", markdown)
        self.assertIn("2着: 8 カムニャック", markdown)

    def test_review_report_does_not_crash_without_result_json(self) -> None:
        markdown = generate_review_report(self.prediction, self.review, race_data=self.race_data)
        self.assertIn("review.jsonに実馬番は未保存", markdown)

    def test_report_commands_save_markdown_files(self) -> None:
        with (
            patch.object(main_module, "DEFAULT_PREDICTIONS_DIR", self.predictions_dir),
            patch.object(main_module, "DEFAULT_REVIEWS_DIR", self.reviews_dir),
            patch.object(main_module, "DEFAULT_REPORTS_DIR", self.reports_dir),
            patch.object(main_module, "DEFAULT_RACE_DATA_DIR", self.race_data_dir),
            patch.object(main_module, "DEFAULT_RESULTS_DIR", self.results_dir),
        ):
            prediction_result = main_module.run_report_prediction("202605020811")
            review_result = main_module.run_report_review("202605020811")

        prediction_path = Path(prediction_result["report_path"])
        review_path = Path(review_result["report_path"])
        self.assertTrue(prediction_path.exists())
        self.assertTrue(review_path.exists())
        prediction_markdown = prediction_path.read_text(encoding="utf-8")
        review_markdown = review_path.read_text(encoding="utf-8")
        self.assertIn("予想レポート", prediction_markdown)
        self.assertIn("14.9", prediction_markdown)
        self.assertIn("回顧レポート", review_markdown)
        self.assertIn("1着: 12 エンブロイダリー", review_markdown)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review
from keiba_llm_agent.social.post_generator import build_prediction_post, build_review_post


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
                }
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
            "used_lessons": [],
            "race_simulation": {
                "race_id": "202605020811",
                "projected_pace": "average",
                "race_flow": "平均的な流れを想定。",
                "key_positions": "好位勢が主導。",
                "favorable_horses": [
                    {"horse_no": 16, "horse_name": "ニシノティアモ", "reason": "好位で運べる。"}
                ],
                "risk_horses": [
                    {"horse_no": 6, "horse_name": "ラヴァンダ", "reason": "距離・コース適性に未知要素が残る。"}
                ],
                "win_scenario": "先行抜け出し。",
                "top3_scenario": "上位は拮抗。",
                "betting_scenario": "ワイド中心。",
                "confidence_comment": "中位の信頼度。",
                "reasoning_summary": "シミュレーションでは平均寄り。先行〜差し勢の持続力勝負を想定。",
                "warnings": [],
            },
            "strategy": {
                "bet_decision": "BET",
                "confidence": "medium",
                "participation_level": "light",
                "reason_codes": ["HEURISTIC_MODEL_ONLY"],
                "reason": "上位拮抗のため点数は絞る。",
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
                    "lesson": "同条件では近走の同距離・同コース実績を優先",
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
                {"horse_no": 15, "frame_no": 8, "horse_id": "h15", "horse_name": "アイサンサン", "jockey": "騎手C", "carried_weight": 55.0, "odds": 49.4, "popularity": 13, "recent_runs": []},
                {"horse_no": 8, "frame_no": 4, "horse_id": "h8", "horse_name": "カムニャック", "jockey": "騎手D", "carried_weight": 55.0, "odds": 5.3, "popularity": 2, "recent_runs": []},
                {"horse_no": 12, "frame_no": 6, "horse_id": "h12", "horse_name": "エンブロイダリー", "jockey": "騎手E", "carried_weight": 55.0, "odds": 1.9, "popularity": 1, "recent_runs": []},
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


class SocialPostGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = _build_prediction()
        self.review = _build_review()
        self.race_data = _build_race_data()
        self.result_data = _build_result_data()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.predictions_dir = self.temp_path / "predictions"
        self.reviews_dir = self.temp_path / "reviews"
        self.race_data_dir = self.temp_path / "race_data"
        self.results_dir = self.temp_path / "results"
        self.social_posts_dir = self.temp_path / "social_posts"
        for directory in (
            self.predictions_dir,
            self.reviews_dir,
            self.race_data_dir,
            self.results_dir,
            self.social_posts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

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

    def test_prediction_post_contains_core_fields_and_fits_limit(self) -> None:
        text = build_prediction_post(self.prediction, race_data=self.race_data)
        self.assertIn("ヴィクトリアマイル", text)
        self.assertIn("◎16", text)
        self.assertIn("買い判断", text)
        self.assertIn("#いかいもAI競馬 #競馬", text)
        self.assertLessEqual(len(text), 280)
        self.assertNotIn("シミュレーションではシミュレーションでは", text)
        self.assertNotIn("※heurist", text)

    def test_prediction_post_drops_optional_sentences_without_hard_truncation(self) -> None:
        self.prediction.summary = (
            "買い判断=BET、confidence=medium。"
            "上位拮抗のため点数は絞る。"
            "本命は近走安定度と距離適性を評価。"
            "さらに補足文を長めに追加して投稿の長さを意図的に伸ばす。"
        )
        text = build_prediction_post(self.prediction, race_data=self.race_data)
        self.assertLessEqual(len(text), 280)
        self.assertNotIn("シミュレーションではシミュレーションでは", text)
        self.assertNotIn("※heurist", text)
        self.assertFalse(text.rstrip().endswith("…"))

    def test_review_post_contains_roi_and_top3_and_fits_limit(self) -> None:
        text = build_review_post(
            self.prediction,
            self.review,
            result_data=self.result_data,
            race_data=self.race_data,
        )
        self.assertIn("回収率：0%", text)
        self.assertIn("印内Top3：2頭", text)
        self.assertIn("結果：12→8→7", text)
        self.assertIn("#いかいもAI競馬 #競馬", text)
        self.assertLessEqual(len(text), 280)

    def test_social_commands_save_text_files(self) -> None:
        with (
            patch.object(main_module, "DEFAULT_PREDICTIONS_DIR", self.predictions_dir),
            patch.object(main_module, "DEFAULT_REVIEWS_DIR", self.reviews_dir),
            patch.object(main_module, "DEFAULT_RACE_DATA_DIR", self.race_data_dir),
            patch.object(main_module, "DEFAULT_RESULTS_DIR", self.results_dir),
            patch.object(main_module, "DEFAULT_SOCIAL_POSTS_DIR", self.social_posts_dir),
        ):
            prediction_result = main_module.run_social_prediction("202605020811")
            review_result = main_module.run_social_review("202605020811")

        prediction_path = Path(prediction_result["post_path"])
        review_path = Path(review_result["post_path"])
        self.assertTrue(prediction_path.exists())
        self.assertTrue(review_path.exists())
        self.assertLessEqual(prediction_result["char_count"], 280)
        self.assertLessEqual(review_result["char_count"], 280)
        self.assertIn("買い判断", prediction_path.read_text(encoding="utf-8"))
        self.assertIn("回収率", review_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

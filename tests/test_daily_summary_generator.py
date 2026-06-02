from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from keiba_llm_agent import main as main_module
from keiba_llm_agent.daily.daily_summary_generator import (
    generate_daily_report,
    generate_daily_social_post,
)


TARGET_DATE = "2026-05-17"


def _prediction_payload(
    race_id: str,
    race_name: str,
    bets: list[dict],
    *,
    with_race_info: bool = True,
) -> dict:
    return {
        "race_id": race_id,
        "race_info": {
            "race_id": race_id,
            "race_name": race_name,
            "race_date": TARGET_DATE,
            "course": "東京",
            "surface": "芝",
            "distance": 1600,
            "track_condition": "良",
            "weather": "晴",
        }
        if with_race_info
        else None,
        "marks": {"◎": 16, "○": 6, "▲": 15, "△": 8, "☆": 12},
        "horse_scores": [
            {
                "horse_no": 16,
                "horse_name": "ニシノティアモ",
                "scores": {
                    "recent_form": 8,
                    "distance_fit": 8,
                    "course_fit": 8,
                    "track_condition_fit": 8,
                    "jockey_fit": 8,
                    "odds_value": 5,
                    "risk": -1,
                },
                "total_score": 40.0,
                "reason": "reason",
            }
        ],
        "bets": bets,
        "summary": "summary",
        "commentary": None,
        "risks": [],
        "used_lessons": [],
        "strategy": {
            "bet_decision": "BET" if bets else "SKIP",
            "confidence": "medium",
            "participation_level": "light" if bets else "none",
            "reason_codes": [],
            "reason": "reason",
        },
    }


def _review_payload(
    race_id: str,
    *,
    bet_hit: bool,
    roi: float,
    total_stake: int,
    total_return: int,
    marked_count: int,
    main_top3: bool,
    lesson_text: str,
    good_points: list[str],
    bad_points: list[str],
) -> dict:
    return {
        "race_id": race_id,
        "hit_summary": {
            "main_mark_top3": main_top3,
            "marked_horses_top3_count": marked_count,
            "bet_hit": bet_hit,
            "roi": roi,
            "total_stake": total_stake,
            "total_return": total_return,
        },
        "bet_results": [
            {
                "bet_type": "ワイド",
                "horse_numbers": [16, 6],
                "amount": total_stake,
                "hit": bet_hit,
                "payout": total_return,
                "return_amount": total_return,
            }
        ]
        if total_stake > 0
        else [],
        "good_points": good_points,
        "bad_points": bad_points,
        "lessons": [
            {
                "course": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "lesson": lesson_text,
                "confidence": "medium",
                "source_race_id": race_id,
            }
        ],
    }


def _result_payload(race_id: str, first: int, second: int, third: int) -> dict:
    return {
        "race_id": race_id,
        "result": {"1st": first, "2nd": second, "3rd": third},
        "payouts": [],
    }


class DailySummaryGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.predictions_dir = self.temp_path / "predictions"
        self.reviews_dir = self.temp_path / "reviews"
        self.results_dir = self.temp_path / "results"
        self.daily_reports_dir = self.temp_path / "daily_reports"
        self.social_posts_dir = self.temp_path / "social_posts"
        self.memory_dir = self.temp_path / "memory"
        self.lessons_path = self.memory_dir / "lessons.json"
        for directory in (
            self.predictions_dir,
            self.reviews_dir,
            self.results_dir,
            self.daily_reports_dir,
            self.social_posts_dir,
            self.memory_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        (self.predictions_dir / "race_a.json").write_text(
            json.dumps(
                _prediction_payload(
                    "race_a",
                    "A Stakes",
                    [{"bet_type": "ワイド", "horse_numbers": [16, 6], "amount": 100, "reason": "reason"}],
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.predictions_dir / "race_b.json").write_text(
            json.dumps(
                _prediction_payload(
                    "race_b",
                    "B Stakes",
                    [{"bet_type": "ワイド", "horse_numbers": [16, 6], "amount": 100, "reason": "reason"}],
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.predictions_dir / "race_c.json").write_text(
            json.dumps(
                _prediction_payload("race_c", "C Stakes", []),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.predictions_dir / "race_skip.json").write_text(
            json.dumps(
                _prediction_payload("race_skip", "Skip Stakes", [], with_race_info=False),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (self.reviews_dir / "race_a.json").write_text(
            json.dumps(
                _review_payload(
                    "race_a",
                    bet_hit=True,
                    roi=2.0,
                    total_stake=100,
                    total_return=200,
                    marked_count=2,
                    main_top3=True,
                    lesson_text="同距離重視",
                    good_points=["軸馬の選定は悪くなかった。"],
                    bad_points=["点数管理に改善余地あり。"],
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.reviews_dir / "race_c.json").write_text(
            json.dumps(
                _review_payload(
                    "race_c",
                    bet_hit=False,
                    roi=0.0,
                    total_stake=0,
                    total_return=0,
                    marked_count=1,
                    main_top3=False,
                    lesson_text="近走精査不足",
                    good_points=["印内に一頭は拾えた。"],
                    bad_points=["本命精度に課題。"],
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (self.results_dir / "race_a.json").write_text(
            json.dumps(_result_payload("race_a", 12, 8, 7), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.results_dir / "race_c.json").write_text(
            json.dumps(_result_payload("race_c", 9, 1, 3), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.lessons_path.write_text(
            json.dumps(
                [
                    {
                        "lesson_id": "lesson_a",
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "lesson": "同距離重視",
                        "confidence": "medium",
                        "source_race_id": "race_a",
                        "created_at": "2026-05-17T00:00:00Z",
                        "updated_at": "2026-05-17T00:00:00Z",
                        "enabled": True,
                        "used_count": 1,
                        "success_count": 1,
                        "failure_count": 0,
                        "score": 0.55,
                    },
                    {
                        "lesson_id": "lesson_c",
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "lesson": "近走精査不足",
                        "confidence": "medium",
                        "source_race_id": "race_c",
                        "created_at": "2026-05-17T00:00:00Z",
                        "updated_at": "2026-05-17T00:00:00Z",
                        "enabled": False,
                        "used_count": 0,
                        "success_count": 0,
                        "failure_count": 1,
                        "score": 0.12,
                    },
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_daily_report_aggregates_metrics_and_pending_races(self) -> None:
        markdown, context = generate_daily_report(
            target_date=TARGET_DATE,
            predictions_dir=self.predictions_dir,
            reviews_dir=self.reviews_dir,
            results_dir=self.results_dir,
            lessons_path=self.lessons_path,
        )
        self.assertIn("# 2026-05-17 AI競馬 Daily Summary", markdown)
        self.assertIn("対象レース数: 3", markdown)
        self.assertIn("予想済みレース数: 3", markdown)
        self.assertIn("回顧済みレース数: 2", markdown)
        self.assertIn("BETレース数: 2", markdown)
        self.assertIn("的中BET数: 1", markdown)
        self.assertIn("ROI: 200.0%", markdown)
        self.assertIn("◎ Top3率: 50.0%", markdown)
        self.assertIn("印内Top3率: 50.0%", markdown)
        self.assertIn("| race_b | B Stakes | BET | medium | 16 ニシノティアモ | - | - | - | 0 | 0 | - |", markdown)
        self.assertIn("| race_c | C Stakes | SKIP | medium | 16 ニシノティアモ | 9 / 1 / 3 | 1 | 見送り | 0 | 0 | 0.0% |", markdown)
        self.assertIn("## 未回顧レース", markdown)
        self.assertIn("- race_b B Stakes", markdown)
        self.assertIn("## BET一覧", markdown)
        self.assertIn("| race_a | A Stakes | ワイド | 16-6 | 100 | 的中 | 200 |", markdown)
        self.assertIn("[medium / score=0.55] 同距離重視", markdown)
        self.assertNotIn("近走精査不足", markdown)
        self.assertEqual(context["metrics"]["bet_race_count"], 2)
        self.assertEqual(context["metrics"]["reviewed_race_count"], 2)
        self.assertEqual(context["metrics"]["total_stake"], 100)
        self.assertEqual(context["metrics"]["total_return"], 200)
        self.assertEqual(context["metrics"]["roi"], 2.0)
        self.assertEqual(context["metrics"]["main_mark_top3_rate"], 0.5)
        self.assertEqual(context["metrics"]["marked_top3_rate"], 0.5)
        self.assertEqual(context["metrics"]["bet_hit_rate"], 0.5)
        self.assertTrue(context["warnings"])

    def test_daily_social_post_is_within_limit(self) -> None:
        _, context = generate_daily_report(
            target_date=TARGET_DATE,
            predictions_dir=self.predictions_dir,
            reviews_dir=self.reviews_dir,
            results_dir=self.results_dir,
            lessons_path=self.lessons_path,
        )
        post = generate_daily_social_post(context)
        self.assertIn("2026-05-17 AI競馬まとめ", post)
        self.assertIn("BET: 2レース / 的中: 1", post)
        self.assertIn("回収率: 200%", post)
        self.assertIn("◎Top3率: 50%", post)
        self.assertIn("#いかいもAI競馬 #競馬", post)
        self.assertLessEqual(len(post), 280)

    def test_daily_report_marks_roi_unreliable_when_payout_warning_exists(self) -> None:
        payload = json.loads((self.reviews_dir / "race_a.json").read_text(encoding="utf-8"))
        payload["payout_warning"] = True
        payload["review_warnings"] = ["Bet hit but payout data missing. ROI is unreliable."]
        (self.reviews_dir / "race_a.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        markdown, context = generate_daily_report(
            target_date=TARGET_DATE,
            predictions_dir=self.predictions_dir,
            reviews_dir=self.reviews_dir,
            results_dir=self.results_dir,
            lessons_path=self.lessons_path,
        )
        self.assertIn("ROI: 200.0% (暫定)", markdown)
        self.assertIn("Some payout data missing. ROI may be unreliable.", markdown)
        self.assertFalse(context["metrics"]["roi_reliable"])

    def test_daily_summary_command_saves_files(self) -> None:
        with (
            patch.object(main_module, "DEFAULT_PREDICTIONS_DIR", self.predictions_dir),
            patch.object(main_module, "DEFAULT_REVIEWS_DIR", self.reviews_dir),
            patch.object(main_module, "DEFAULT_RESULTS_DIR", self.results_dir),
            patch.object(main_module, "DEFAULT_DAILY_REPORTS_DIR", self.daily_reports_dir),
            patch.object(main_module, "DEFAULT_SOCIAL_POSTS_DIR", self.social_posts_dir),
            patch.object(main_module, "DEFAULT_LESSONS_PATH", self.lessons_path),
        ):
            result = main_module.run_daily_summary(TARGET_DATE, skip_report=False)

        self.assertTrue(Path(result["daily_report_path"]).exists())
        self.assertTrue(Path(result["daily_social_post_path"]).exists())
        self.assertEqual(result["target_race_count"], 3)
        self.assertEqual(result["reviewed_race_count"], 2)
        self.assertEqual(result["bet_race_count"], 2)
        self.assertEqual(result["hit_count"], 1)

    def test_daily_summary_default_skips_report(self) -> None:
        with (
            patch.object(main_module, "DEFAULT_PREDICTIONS_DIR", self.predictions_dir),
            patch.object(main_module, "DEFAULT_REVIEWS_DIR", self.reviews_dir),
            patch.object(main_module, "DEFAULT_RESULTS_DIR", self.results_dir),
            patch.object(main_module, "DEFAULT_DAILY_REPORTS_DIR", self.daily_reports_dir),
            patch.object(main_module, "DEFAULT_SOCIAL_POSTS_DIR", self.social_posts_dir),
            patch.object(main_module, "DEFAULT_LESSONS_PATH", self.lessons_path),
        ):
            result = main_module.run_daily_summary(TARGET_DATE)

        self.assertIsNone(result["daily_report_path"])
        self.assertTrue(Path(result["daily_social_post_path"]).exists())

    def test_daily_summary_command_supports_skip_flags(self) -> None:
        with (
            patch.object(main_module, "DEFAULT_PREDICTIONS_DIR", self.predictions_dir),
            patch.object(main_module, "DEFAULT_REVIEWS_DIR", self.reviews_dir),
            patch.object(main_module, "DEFAULT_RESULTS_DIR", self.results_dir),
            patch.object(main_module, "DEFAULT_DAILY_REPORTS_DIR", self.daily_reports_dir),
            patch.object(main_module, "DEFAULT_SOCIAL_POSTS_DIR", self.social_posts_dir),
            patch.object(main_module, "DEFAULT_LESSONS_PATH", self.lessons_path),
        ):
            result = main_module.run_daily_summary(
                TARGET_DATE,
                skip_report=True,
                skip_social=True,
            )

        self.assertIsNone(result["daily_report_path"])
        self.assertIsNone(result["daily_social_post_path"])


if __name__ == "__main__":
    unittest.main()

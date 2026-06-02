from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keiba_llm_agent.agents.race_analysis_agent import RaceAnalysisAgent
from keiba_llm_agent.agents.race_review_agent import RaceReviewAgent
from keiba_llm_agent.llm.mock_llm_client import MockLLMClient
from keiba_llm_agent.memory.lesson_store import LessonStore
from keiba_llm_agent.schemas.race_data import RaceData


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = RaceData.from_json_file(
    ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
)
SAMPLE_RESULT = json.loads(
    (ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_result.json").read_text(encoding="utf-8")
)


class _ReviewEnhancingLLM(MockLLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        if schema_name == "review_enhancement":
            return {
                "good_points": ["LLM good point"],
                "bad_points": ["LLM bad point"],
                "lessons": [
                    {
                        "course": "東京",
                        "surface": "芝",
                        "distance": 1600,
                        "track_condition": "良",
                        "lesson": "LLM lesson",
                        "confidence": "medium",
                        "source_race_id": "sample_001",
                    }
                ],
            }
        return super().generate_json(system_prompt, user_prompt, schema_name=schema_name)


class _FailingReviewEnhancementLLM(MockLLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        if schema_name == "review_enhancement":
            raise RuntimeError("review enhancement failed")
        return super().generate_json(system_prompt, user_prompt, schema_name=schema_name)


class _InvalidLessonEnhancingLLM(MockLLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        if schema_name == "review_enhancement":
            return {
                "good_points": ["Invalid lesson test"],
                "bad_points": ["Invalid lesson test"],
                "lessons": [
                    {
                        "course": "unknown",
                        "surface": "unknown",
                        "distance": 0,
                        "track_condition": "unknown",
                        "lesson": "invalid",
                        "confidence": "low",
                        "source_race_id": "sample_001",
                    }
                ],
            }
        return super().generate_json(system_prompt, user_prompt, schema_name=schema_name)


class ReviewLLMEnhancementTests(unittest.TestCase):
    def setUp(self) -> None:
        prediction_agent = RaceAnalysisAgent(llm_client=MockLLMClient())
        self.prediction = prediction_agent.run(SAMPLE_RACE_DATA, [])
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lessons_path = Path(self.temp_dir.name) / "lessons.json"
        self.lessons_path.write_text("[]\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_llm_can_generate_good_bad_points_and_lessons(self) -> None:
        agent = RaceReviewAgent(llm_client=_ReviewEnhancingLLM())
        review = agent.run("sample_001", SAMPLE_RESULT, self.prediction, race_info=self.prediction.race_info)

        self.assertEqual(review.good_points, ["LLM good point"])
        self.assertIn("LLM bad point", review.bad_points)
        self.assertTrue(review.payout_warning)
        self.assertIn("Bet hit but payout data missing. ROI is unreliable.", review.review_warnings)
        self.assertEqual(review.lessons[0].lesson, "LLM lesson")

    def test_llm_failure_does_not_block_review_generation(self) -> None:
        agent = RaceReviewAgent(llm_client=_FailingReviewEnhancementLLM())
        review = agent.run("sample_001", SAMPLE_RESULT, self.prediction, race_info=self.prediction.race_info)

        self.assertEqual(review.race_id, "sample_001")
        self.assertGreaterEqual(len(review.good_points) + len(review.bad_points), 1)

    def test_invalid_lesson_does_not_enter_lessons_json(self) -> None:
        agent = RaceReviewAgent(llm_client=_InvalidLessonEnhancingLLM())
        review = agent.run("sample_001", SAMPLE_RESULT, self.prediction, race_info=self.prediction.race_info)
        store = LessonStore(self.lessons_path)
        store.append_lessons(review.lessons)

        payload = json.loads(self.lessons_path.read_text(encoding="utf-8"))
        self.assertTrue(all(item["course"] != "unknown" for item in payload))
        self.assertTrue(all(item["distance"] != 0 for item in payload))


if __name__ == "__main__":
    unittest.main()

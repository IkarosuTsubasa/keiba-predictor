from __future__ import annotations

import unittest
from pathlib import Path

from keiba_llm_agent.agents.race_analysis_agent import RaceAnalysisAgent
from keiba_llm_agent.llm.mock_llm_client import MockLLMClient
from keiba_llm_agent.scoring.recent_run_scorer import build_prediction_from_recent_runs
from keiba_llm_agent.schemas.race_data import RaceData


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RACE_DATA = RaceData.from_json_file(
    ROOT_DIR / "keiba_llm_agent" / "data" / "samples" / "sample_race_data.json"
)


class _SummaryEnhancingLLM(MockLLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        if schema_name == "prediction_enhancement":
            return {
                "strategy_reason": "◎サンプルホースは道中で脚を使わされにくく、相手筆頭との比較でも最後の踏ん張りを評価する。",
                "summary": "LLM強化summary",
                "risks": ["LLM強化risk"],
                "commentary": "LLM commentary",
                "top_horse_memos": [
                    {"horse_no": 1, "memo": "内で脚をためられれば最後まで勝負になる。"},
                    {"horse_no": 2, "memo": "相手強化でも持続力はここで通用する。"},
                ],
            }
        return super().generate_json(system_prompt, user_prompt, schema_name=schema_name)


class _FailingPredictionEnhancementLLM(MockLLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        if schema_name == "prediction_enhancement":
            raise RuntimeError("enhancement failed")
        return super().generate_json(system_prompt, user_prompt, schema_name=schema_name)


class _ForbiddenCopyEnhancingLLM(MockLLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        if schema_name == "prediction_enhancement":
            return {
                "strategy_reason": "本命想定はサンプルホース 市場オッズを使わず、能力・条件適性を中心に評価。",
                "summary": "ルールベース評価を使用。",
                "risks": [
                    "正式なML modelではない。",
                    "軸馬の決め手はやや薄い。",
                ],
                "commentary": "fallback used",
            }
        return super().generate_json(system_prompt, user_prompt, schema_name=schema_name)


class PredictionLLMEnhancementTests(unittest.TestCase):
    def test_mock_llm_enhancement_can_modify_summary(self) -> None:
        agent = RaceAnalysisAgent(llm_client=_SummaryEnhancingLLM())
        baseline = build_prediction_from_recent_runs(SAMPLE_RACE_DATA, [])
        prediction = agent.run(SAMPLE_RACE_DATA, [])

        self.assertIn("LLM強化summary", prediction.summary)
        self.assertIsNotNone(prediction.race_simulation)
        self.assertIn(prediction.race_simulation.reasoning_summary, prediction.summary)
        self.assertEqual(prediction.commentary, "LLM commentary")
        self.assertEqual(
            prediction.strategy.reason,
            "◎サンプルホースは道中で脚を使わされにくく、相手筆頭との比較でも最後の踏ん張りを評価する。",
        )
        self.assertIn("LLM強化risk", prediction.risks)
        self.assertNotIn("LLM simulation fallback used.", prediction.risks)
        self.assertGreaterEqual(len(prediction.top_horse_memos), 1)
        self.assertEqual(prediction.top_horse_memos[0].memo, "内で脚をためられれば最後まで勝負になる。")
        self.assertEqual(
            [horse.total_score for horse in prediction.horse_scores],
            [horse.total_score for horse in baseline.horse_scores],
        )

    def test_internal_copy_is_filtered_from_public_prediction(self) -> None:
        agent = RaceAnalysisAgent(llm_client=_ForbiddenCopyEnhancingLLM())
        prediction = agent.run(SAMPLE_RACE_DATA, [])

        self.assertNotIn("ルールベース評価を使用。", prediction.summary)
        self.assertNotIn("市場オッズを使わず", prediction.strategy.reason)
        self.assertEqual(prediction.risks, ["軸馬の決め手はやや薄い。"])
        self.assertIsNone(prediction.commentary)

    def test_llm_failure_does_not_block_prediction_generation(self) -> None:
        agent = RaceAnalysisAgent(llm_client=_FailingPredictionEnhancementLLM())
        prediction = agent.run(SAMPLE_RACE_DATA, [])

        self.assertEqual(prediction.race_id, "sample_001")
        self.assertGreater(len(prediction.horse_scores), 0)
        self.assertIsNone(prediction.commentary)
        self.assertIn("買い判断=", prediction.summary)


if __name__ == "__main__":
    unittest.main()

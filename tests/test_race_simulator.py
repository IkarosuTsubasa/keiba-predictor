from __future__ import annotations

import json
import unittest

from keiba_llm_agent.llm.llm_client import BaseLLMClient
from keiba_llm_agent.llm.mock_llm_client import MockLLMClient
from keiba_llm_agent.schemas.deep_analysis import HorseDeepAnalysis
from keiba_llm_agent.schemas.pace_analysis import HorsePaceAnalysis, RacePaceProjection
from keiba_llm_agent.schemas.pedigree import PedigreeAnalysis
from keiba_llm_agent.schemas.prediction import BetSuggestion, HorseScore, ScoreBreakdown, StrategyDecision
from keiba_llm_agent.schemas.race_data import RaceInfo
from keiba_llm_agent.schemas.race_level_analysis import RaceLevelAnalysis
from keiba_llm_agent.simulation.race_simulator import simulate_race


def _horse_score(index: int) -> HorseScore:
    return HorseScore(
        horse_no=index,
        horse_name=f"ホース{index}",
        scores=ScoreBreakdown(
            recent_form=8,
            distance_fit=7,
            course_fit=6,
            track_condition_fit=6,
            jockey_fit=6,
            odds_value=5,
            risk=-2,
        ),
        base_total_score=35.0 - index,
        total_score=35.0 - index,
        reason=f"理由{index}",
    )


def _deep(index: int) -> HorseDeepAnalysis:
    return HorseDeepAnalysis(
        horse_no=index,
        horse_name=f"ホース{index}",
        positive_flags=["RECENT_FORM_STRONG"],
        risk_flags=[],
        recent_form_summary="近走良好",
        distance_analysis="距離合う",
        course_analysis="コース合う",
        track_condition_analysis="馬場合う",
        jockey_analysis="騎手合う",
        odds_analysis="妙味あり",
        overall_comment="総合的に有力。",
    )


def _pedigree(index: int) -> PedigreeAnalysis:
    return PedigreeAnalysis(
        horse_no=index,
        horse_name=f"ホース{index}",
        sire="キズナ",
        dam="母",
        damsire="キングカメハメハ",
        surface_tendency="芝",
        distance_tendency="中距離",
        track_condition_tendency="パワー",
        pace_tendency="持続力",
        positive_flags=["PEDIGREE_DISTANCE_FIT"],
        risk_flags=[],
        overall_comment="血統面もプラス。",
    )


def _race_level(index: int) -> RaceLevelAnalysis:
    return RaceLevelAnalysis(
        horse_no=index,
        horse_name=f"ホース{index}",
        positive_flags=["HEAD_TO_HEAD_POSITIVE"] if index == 1 else [],
        risk_flags=["HEAD_TO_HEAD_NEGATIVE"] if index == 7 else [],
        head_to_head_summary="再戦関係あり。",
        race_level_summary="近走水準は一定。",
        opponent_context_summary="比較可能。",
        overall_comment="相手比較は悪くない。",
        adjustment_hint=0.5 if index == 1 else 0.0,
    )


def _pace(index: int) -> HorsePaceAnalysis:
    return HorsePaceAnalysis(
        horse_no=index,
        horse_name=f"ホース{index}",
        running_style="先行" if index <= 3 else "差し",
        early_position_score=7.0,
        late_position_score=6.5,
        position_stability="安定",
        positive_flags=["PACE_FIT"] if index <= 3 else [],
        risk_flags=[],
        overall_comment="展開待ちだが悪くない。",
    )


class _RecordingLLM(BaseLLMClient):
    def __init__(self) -> None:
        super().__init__(fallback_client=None)
        self.calls: list[dict] = []

    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        self.calls.append(json.loads(user_prompt))
        top_horses = self.calls[-1]["top_horses"]
        return {
            "race_flow": "平均的な流れを想定。",
            "key_positions": "先行勢が隊列を作る。",
            "favorable_horses": [
                {
                    "horse_no": top_horses[0]["horse_no"],
                    "horse_name": top_horses[0]["horse_name"],
                    "reason": "先行力を評価。",
                }
            ],
            "risk_horses": [],
            "win_scenario": "好位抜け出し。",
            "top3_scenario": "上位勢中心。",
            "betting_scenario": "ワイド向き。",
            "confidence_comment": "中位の信頼度。",
            "reasoning_summary": "シミュレーションでは先行〜差しの持続力勝負を想定。",
            "warnings": [],
        }


class _FailingLLM(BaseLLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        raise RuntimeError("simulation failed")


class _OpenAILikeLLM(BaseLLMClient):
    def __init__(self) -> None:
        super().__init__(fallback_client=None)
        self.called = False

    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        self.called = True
        payload = json.loads(user_prompt)
        top = payload["top_horses"][0]
        return {
            "race_flow": "先行勢が自然に流れを作る。",
            "key_positions": "好位勢が主導。",
            "favorable_horses": [
                {
                    "horse_no": top["horse_no"],
                    "horse_name": top["horse_name"],
                    "reason": "先行力を評価。",
                }
            ],
            "risk_horses": [],
            "win_scenario": "好位抜け出し。",
            "top3_scenario": "上位候補は絞れる。",
            "betting_scenario": "ワイド中心。",
            "confidence_comment": "中位の信頼度。",
            "reasoning_summary": "平均寄り。先行〜差し勢の持続力勝負を想定。",
            "warnings": [],
        }


class _IncompleteLLM(BaseLLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str | None = None) -> dict:
        return {
            "race_flow": "",
            "key_positions": "",
            "favorable_horses": [],
            "risk_horses": [],
            "win_scenario": "",
            "top3_scenario": "",
            "betting_scenario": "",
            "confidence_comment": "",
            "reasoning_summary": "",
            "warnings": [],
        }


class RaceSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.race_info = RaceInfo(
            race_id="sim_001",
            race_name="シミュレーション特別",
            race_date="2026-05-17",
            course="京都",
            surface="芝",
            distance=2400,
            track_condition="良",
            weather="晴",
        )
        self.horse_scores = [_horse_score(index) for index in range(1, 9)]
        self.deep_analyses = [_deep(index) for index in range(1, 9)]
        self.pedigree_analyses = [_pedigree(index) for index in range(1, 9)]
        self.race_level_analyses = [_race_level(index) for index in range(1, 9)]
        self.pace_analyses = [_pace(index) for index in range(1, 9)]
        self.race_pace_projection = RacePaceProjection(
            projected_pace="average",
            front_runner_count=1,
            stalker_count=3,
            closer_count=4,
            pace_comment="平均〜やや落ち着く想定。",
            favorable_styles=["先行", "差し"],
            risk_styles=[],
        )
        self.strategy = StrategyDecision(
            bet_decision="BET",
            confidence="medium",
            participation_level="light",
            reason_codes=["HEURISTIC_MODEL_ONLY"],
            reason="上位拮抗のため点数は絞る。",
        )
        self.marks = {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5}
        self.bets = [
            BetSuggestion(
                bet_type="ワイド",
                horse_numbers=[5, 6],
                amount=100,
                reason="上位拮抗のため軽めに絞る。",
            )
        ]

    def test_mock_llm_uses_fallback_and_generates_simulation(self) -> None:
        before_scores = [score.total_score for score in self.horse_scores]
        simulation = simulate_race(
            self.race_info,
            self.marks,
            self.horse_scores,
            self.bets,
            self.deep_analyses,
            self.pedigree_analyses,
            self.race_level_analyses,
            self.pace_analyses,
            self.race_pace_projection,
            self.strategy,
            llm_client=MockLLMClient(),
        )
        self.assertEqual(simulation.race_id, "sim_001")
        self.assertGreater(len(simulation.favorable_horses), 0)
        self.assertIn("LLM simulation fallback used.", simulation.warnings)
        self.assertEqual(before_scores, [score.total_score for score in self.horse_scores])

    def test_llm_failure_uses_fallback(self) -> None:
        simulation = simulate_race(
            self.race_info,
            self.marks,
            self.horse_scores,
            self.bets,
            self.deep_analyses,
            self.pedigree_analyses,
            self.race_level_analyses,
            self.pace_analyses,
            self.race_pace_projection,
            self.strategy,
            llm_client=_FailingLLM(),
        )
        self.assertIn("LLM simulation fallback used.", simulation.warnings)
        self.assertNotIn("…", simulation.risk_horses[0].reason)
        self.assertTrue(simulation.risk_horses[0].reason.endswith("。"))
        self.assertTrue(simulation.race_flow)
        self.assertGreaterEqual(len(simulation.favorable_horses), 3)
        self.assertIn("ワイド5-6", simulation.betting_scenario)
        self.assertTrue(simulation.reasoning_summary)
        self.assertNotIn("買い目の具体情報は入力に含まれていない", "".join(simulation.warnings))

    def test_only_top7_are_sent_to_llm(self) -> None:
        llm = _RecordingLLM()
        simulation = simulate_race(
            self.race_info,
            self.marks,
            self.horse_scores,
            self.bets,
            self.deep_analyses,
            self.pedigree_analyses,
            self.race_level_analyses,
            self.pace_analyses,
            self.race_pace_projection,
            self.strategy,
            llm_client=llm,
        )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(llm.calls[0]["top_horses"]), 7)
        self.assertEqual(simulation.reasoning_summary, "シミュレーションでは先行〜差しの持続力勝負を想定。")

    def test_openai_like_llm_is_not_forced_to_mock_fallback(self) -> None:
        llm = _OpenAILikeLLM()
        simulation = simulate_race(
            self.race_info,
            self.marks,
            self.horse_scores,
            self.bets,
            self.deep_analyses,
            self.pedigree_analyses,
            self.race_level_analyses,
            self.pace_analyses,
            self.race_pace_projection,
            self.strategy,
            llm_client=llm,
        )
        self.assertTrue(llm.called)
        self.assertEqual(simulation.warnings, [])

    def test_incomplete_llm_output_uses_fallback(self) -> None:
        simulation = simulate_race(
            self.race_info,
            self.marks,
            self.horse_scores,
            self.bets,
            self.deep_analyses,
            self.pedigree_analyses,
            self.race_level_analyses,
            self.pace_analyses,
            self.race_pace_projection,
            self.strategy,
            llm_client=_IncompleteLLM(),
        )
        self.assertIn("LLM simulation output was incomplete; fallback used.", simulation.warnings)
        self.assertTrue(simulation.race_flow)
        self.assertTrue(simulation.key_positions)
        self.assertGreaterEqual(len(simulation.favorable_horses), 3)
        self.assertTrue(simulation.top3_scenario)
        self.assertTrue(simulation.reasoning_summary)
        self.assertIn("ワイド5-6", simulation.betting_scenario)


if __name__ == "__main__":
    unittest.main()

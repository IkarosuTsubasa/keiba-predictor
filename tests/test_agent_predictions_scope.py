import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from web_data import agent_predictions


class AgentPredictionsScopeTests(unittest.TestCase):
    def test_scope_matches_local_from_race_info_scope_key(self) -> None:
        payload = {
            "race_id": "202644031102",
            "race_info": {
                "scope_key": "local",
                "source": "local",
                "course": "",
                "surface": "ダート",
            },
        }

        self.assertTrue(agent_predictions._scope_matches(payload, "local"))
        self.assertFalse(agent_predictions._scope_matches(payload, "central_dirt"))
        self.assertFalse(agent_predictions._scope_matches(payload, "central"))

    def test_scope_matches_central_group_from_scope_key(self) -> None:
        dirt_payload = {
            "race_id": "202601010101",
            "race_info": {
                "scope_key": "central_dirt",
                "course": "東京",
                "surface": "ダート",
            },
        }
        turf_payload = {
            "race_id": "202601010102",
            "race_info": {
                "scope_key": "central_turf",
                "course": "中山",
                "surface": "芝",
            },
        }
        local_payload = {
            "race_id": "202644031102",
            "race_info": {
                "scope_key": "local",
                "course": "大井",
                "surface": "ダート",
            },
        }

        self.assertTrue(agent_predictions._scope_matches(dirt_payload, "central"))
        self.assertTrue(agent_predictions._scope_matches(turf_payload, "central"))
        self.assertFalse(agent_predictions._scope_matches(local_payload, "central"))

    def test_strategy_confidence_uses_score_shape_with_same_label(self) -> None:
        close_payload = {
            "strategy": {"confidence": "high"},
            "horse_scores": [
                {"horse_no": 1, "total_score": 31.0},
                {"horse_no": 2, "total_score": 30.8},
                {"horse_no": 3, "total_score": 30.5},
                {"horse_no": 4, "total_score": 28.0},
                {"horse_no": 5, "total_score": 27.0},
            ],
        }
        clear_payload = {
            "strategy": {"confidence": "high"},
            "horse_scores": [
                {"horse_no": 1, "total_score": 39.0},
                {"horse_no": 2, "total_score": 29.0},
                {"horse_no": 3, "total_score": 27.0},
                {"horse_no": 4, "total_score": 24.0},
                {"horse_no": 5, "total_score": 21.0},
            ],
        }

        close_score = agent_predictions.strategy_confidence_score(close_payload)
        clear_score = agent_predictions.strategy_confidence_score(clear_payload)

        self.assertGreater(clear_score, close_score)
        self.assertNotEqual(close_score, 0.82)
        self.assertNotEqual(clear_score, 0.82)

    def test_top_horse_items_prefers_public_llm_memo(self) -> None:
        payload = {
            "marks": {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
            "horse_scores": [
                {"horse_no": 1, "horse_name": "ホースA", "total_score": 40.0, "reason": "raw reason"},
            ],
            "top_horse_memos": [
                {"horse_no": 1, "memo": "勝負所で前を射程に入れられれば崩れにくい。"},
            ],
        }

        rows = agent_predictions._top_horse_items(payload, limit=1)

        self.assertEqual(rows[0]["reason"], "勝負所で前を射程に入れられれば崩れにくい。")

    def test_history_public_evaluation_counts_use_confidence_not_raw_bet(self) -> None:
        def payload(race_id, bet_decision, confidence_score):
            return {
                "race_id": race_id,
                "race_info": {
                    "race_id": race_id,
                    "race_date": "2026-06-10",
                    "course": "東京",
                },
                "marks": {"◎": 1, "○": 2, "▲": 3, "△": 4, "☆": 5},
                "strategy": {
                    "bet_decision": bet_decision,
                    "confidence_score": confidence_score,
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            records = [
                agent_predictions._history_record(base_dir, payload("202606100101", "BET", 0.75)),
                agent_predictions._history_record(base_dir, payload("202606100102", "BET", 0.86)),
                agent_predictions._history_record(base_dir, payload("202606100103", "", 0.65)),
            ]

        summary = agent_predictions._summarize_history_records(records)

        self.assertEqual(summary["bet_races"], 2)
        self.assertEqual(summary["high_evaluation_races"], 1)
        self.assertEqual(summary["watch_evaluation_races"], 1)
        self.assertEqual(summary["skip_evaluation_races"], 1)


if __name__ == "__main__":
    unittest.main()

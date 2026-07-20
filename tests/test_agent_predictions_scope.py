import json
import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from web_data import agent_predictions


class AgentPredictionsScopeTests(unittest.TestCase):
    def test_final_version_is_publicly_preferred_without_overwriting_early_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            prediction_dir = base_dir / "data" / "agent_predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            race_id = "202601010101"
            early_payload = {
                "race_id": race_id,
                "race_info": {
                    "race_id": race_id,
                    "race_date": "2026-07-19",
                    "course": "東京",
                    "surface": "芝",
                    "distance": 1600,
                },
                "marks": {"◎": 1},
                "horse_scores": [{"horse_no": 1, "horse_name": "早期馬", "total_score": 10}],
            }
            legacy_path = prediction_dir / f"{race_id}.json"
            legacy_text = json.dumps(early_payload, ensure_ascii=False, indent=2)
            legacy_path.write_text(legacy_text, encoding="utf-8")
            agent_predictions.write_agent_prediction_version_payload(
                base_dir,
                race_id,
                "early",
                "task-early",
                {**early_payload, "prediction_stage": "early"},
            )
            final_payload = {
                **early_payload,
                "prediction_stage": "final",
                "horse_scores": [{"horse_no": 2, "horse_name": "最終馬", "total_score": 20}],
                "marks": {"◎": 2},
            }
            final_path = agent_predictions.write_agent_prediction_version_payload(
                base_dir,
                race_id,
                "final",
                "task-final",
                final_payload,
            )

            resolved = agent_predictions.agent_prediction_path_for_race(base_dir, race_id)
            rows = agent_predictions.build_public_races(base_dir, "2026-07-19")
            preserved_legacy_text = legacy_path.read_text(encoding="utf-8")

        self.assertEqual(resolved, final_path)
        self.assertEqual(rows[0]["prediction_stage"], "final")
        self.assertEqual(rows[0]["prediction_stage_label"], "最終予測")
        self.assertEqual(rows[0]["top5"][0]["horse_name"], "最終馬")
        self.assertEqual(preserved_legacy_text, legacy_text)

    def test_result_is_attached_to_early_and_final_version_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            race_id = "202601010101"
            for stage in ("early", "final"):
                prediction = {"race_id": race_id, "prediction_stage": stage}
                audit = agent_predictions.build_prediction_only_backtest_audit(prediction)
                agent_predictions.write_agent_backtest_audit_version_payload(
                    base_dir,
                    race_id,
                    stage,
                    f"task-{stage}",
                    audit,
                )
            result = {"race_id": race_id, "finish_order": [{"horse_no": 1}]}

            agent_predictions.attach_result_to_agent_backtest_audit(base_dir, race_id, result)
            early_path = agent_predictions.agent_backtest_audit_version_path_for_race(
                base_dir, race_id, "early", "task-early"
            )
            final_path = agent_predictions.agent_backtest_audit_version_path_for_race(
                base_dir, race_id, "final", "task-final"
            )
            early = agent_predictions.read_agent_backtest_audit_payload(early_path)
            final = agent_predictions.read_agent_backtest_audit_payload(final_path)

        self.assertEqual(early["outcomes"]["result"], result)
        self.assertEqual(final["outcomes"]["result"], result)

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

    def test_strategy_confidence_fallback_uses_scope_calibration(self) -> None:
        horse_scores = [
            {"horse_no": 1, "total_score": 42.0},
            {"horse_no": 2, "total_score": 39.0},
            {"horse_no": 3, "total_score": 35.0},
            {"horse_no": 4, "total_score": 32.0},
            {"horse_no": 5, "total_score": 30.0},
            {"horse_no": 6, "total_score": 27.0},
        ]
        central_payload = {
            "race_info": {"scope_key": "central_turf", "course": "東京", "surface": "芝"},
            "strategy": {"confidence": "high"},
            "horse_scores": horse_scores,
        }
        local_payload = {
            "race_info": {"scope_key": "local", "source": "local", "course": "大井", "surface": "ダート"},
            "strategy": {"confidence": "high"},
            "horse_scores": horse_scores,
        }

        central_score = agent_predictions.strategy_confidence_score(central_payload)
        local_score = agent_predictions.strategy_confidence_score(local_payload)

        self.assertLess(central_score, local_score)
        self.assertGreaterEqual(local_score, 0.46)
        self.assertLessEqual(central_score, 0.86)

    def test_strategy_confidence_keeps_explicit_score(self) -> None:
        payload = {
            "race_info": {"scope_key": "central_turf", "course": "東京", "surface": "芝"},
            "strategy": {"confidence": "low", "confidence_score": 0.86},
            "horse_scores": [
                {"horse_no": 1, "total_score": 31.0},
                {"horse_no": 2, "total_score": 30.8},
            ],
        }

        self.assertEqual(agent_predictions.strategy_confidence_score(payload), 0.86)

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

    def test_custom_prediction_payload_exposes_factors_without_final_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "pipeline"
            prediction_dir = base_dir / "data" / "agent_predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            race_id = "202601010101"
            payload = {
                "race_id": race_id,
                "prediction_stage": "final",
                "race_info": {
                    "race_id": race_id,
                    "race_date": "2026-07-19",
                    "race_name": "テスト競走",
                    "course": "東京",
                    "surface": "芝",
                    "distance": 1600,
                    "scope_key": "central_turf",
                },
                "strategy": {"bet_decision": "BET"},
                "marks": {"◎": 7},
                "horse_scores": [
                    {
                        "horse_no": 7,
                        "horse_name": "テストホース",
                        "base_total_score": 21.5,
                        "total_score": 99.9,
                        "reason": "非公開テキスト",
                        "scores": {
                            "ability_score": 8,
                            "recent_quality_score": 7,
                            "trend_score": 6,
                            "condition_fit_score": 5,
                            "race_level_score": 4,
                            "pace_jockey_score": 3,
                            "risk": -2,
                        },
                        "score_breakdown": {
                            "base_total_score": 21.5,
                            "pedigree_adjustment_raw": 1.4,
                            "pedigree_weight": 0.6,
                            "race_level_adjustment_raw": -0.3,
                            "race_level_weight": 1.1,
                            "pace_adjustment_raw": 0.8,
                            "pace_weight": 0.2,
                            "total_score": 99.9,
                        },
                    }
                ],
            }
            (prediction_dir / f"{race_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            rows = agent_predictions.build_custom_prediction_races(base_dir, "2026-07-19")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["race_title"], "東京1R")
        self.assertEqual(rows[0]["scope_key"], "central_turf")
        horse = rows[0]["horses"][0]
        self.assertEqual(horse["horse_no"], 7)
        self.assertEqual(horse["factors"]["base"], 21.5)
        self.assertEqual(horse["factors"]["pedigree"], 1.4)
        self.assertEqual(horse["factors"]["ability"], 8.0)
        self.assertEqual(horse["factors"]["recent_quality"], 7.0)
        self.assertEqual(horse["factors"]["risk"], -2.0)
        self.assertEqual(horse["default_weights"]["race_level"], 1.1)
        self.assertNotIn("total_score", horse)
        self.assertNotIn("marks", rows[0])
        self.assertNotIn("strategy", rows[0])
        self.assertNotIn("reason", horse)

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

from __future__ import annotations

import unittest

from pipeline.train_scope_reranker import (
    TRAINING_CONFIG,
    top3_from_audit,
)


class TrainScopeRerankerTests(unittest.TestCase):
    def test_place_training_uses_all_full_races(self) -> None:
        turf = TRAINING_CONFIG["central_turf"]
        local = TRAINING_CONFIG["local"]

        self.assertIsNone(turf["training_end"])
        self.assertEqual(turf["training_target"], "top3")
        self.assertEqual(turf["winner_weight_multiplier"], 1.0)
        self.assertEqual(turf["blend_weight"], 0.5)
        self.assertEqual(turf["preserve_top_n"], 1)

        self.assertIsNone(local["training_end"])
        self.assertEqual(local["training_target"], "top3")
        self.assertEqual(local["winner_weight_multiplier"], 2.0)
        self.assertEqual(local["blend_weight"], 1.0)
        self.assertEqual(local["preserve_top_n"], 2)

    def test_top3_falls_back_to_finish_order(self) -> None:
        payload = {
            "outcomes": {
                "result": {
                    "finish_order": [
                        {"finish": 1, "horse_no": 7},
                        {"finish": 2, "horse_no": 3},
                        {"finish": 3, "horse_no": 9},
                        {"finish": 4, "horse_no": 1},
                    ]
                }
            }
        }

        self.assertEqual(top3_from_audit(payload), {3, 7, 9})


if __name__ == "__main__":
    unittest.main()

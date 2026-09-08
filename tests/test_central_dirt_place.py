from __future__ import annotations

import unittest
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from keiba_llm_agent.scoring.place_context import build_place_context_matrix
from keiba_llm_agent.scoring import scope_reranker as runtime
from pipeline import experiment_agent_place_models as exp
from pipeline import validate_central_dirt_place as validation
from tests.test_scope_reranker import _horse_scores, _race_data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from web_data.agent_predictions import _custom_prediction_race_item


def dirt_fixture(size=13):
    race = _race_data(source="central", surface="\u30c0\u30fc\u30c8")
    horse, score = race.horses[0], _horse_scores()[0]
    race.horses = [horse.model_copy(deep=True, update={"horse_no": n}) for n in range(1, size + 1)]
    scores = [score.model_copy(deep=True, update={"horse_no": n, "total_score": 90 - n})
              for n in range(1, size + 1)]
    return race, scores


class AscendingModel:
    n_features_in_ = 65

    def predict_proba(self, matrix):
        p = np.linspace(0.1, 0.9, len(matrix))
        return np.column_stack((1 - p, p))


class CentralDirtPlaceTests(unittest.TestCase):
    def setUp(self):
        runtime.clear_scope_reranker_caches()

    def tearDown(self):
        runtime.clear_scope_reranker_caches()

    def test_context_matches_review_feature_definition(self):
        race = _race_data(source="central", surface="\u30c0\u30fc\u30c8")
        scores = _horse_scores()
        for missing in (False, True):
            if missing:
                race.horses[0].recent_runs = []
                race.horses[1].recent_runs[0].date = None
                race.horses[1].recent_runs[0].margin = None
                race.horses[2].recent_runs[0].corner_positions = None
            native, numbers = exp.reranker.build_scope_reranker_feature_matrix(race, scores, "course_geometry")
            actual = build_place_context_matrix(race, scores, native)
            np.testing.assert_allclose(actual, exp.context_matrix(race, scores, native), equal_nan=True)
            self.assertEqual(actual.shape, (len(numbers), 65))
            for horse in race.horses:
                horse.odds, horse.popularity = 999, 18
                for run in horse.recent_runs:
                    run.odds, run.popularity = 999, 18
            np.testing.assert_allclose(actual, build_place_context_matrix(race, scores, native), equal_nan=True)

    @patch.object(validation, "HistGradientBoostingClassifier")
    def test_equal_top3_target_without_winner_or_class_reweighting(self, classifier):
        rows = [{"numbers": list(range(1, n + 1)), "top3": [1, 2, 3], "winner": winner,
                 "fresh_features": {"context": np.zeros((n, 65))}}
                for n, winner in ((8, 1), (15, 3))]
        validation.fit_place_model(rows)
        call = classifier.return_value.fit.call_args
        np.testing.assert_array_equal(call.args[1], [1, 1, 1, 0, 0, 0, 0, 0] + [1, 1, 1] + [0] * 12)
        weights = call.kwargs["sample_weight"]
        np.testing.assert_allclose(weights, np.full(23, 34 / 23))

    def test_gate_uses_entire_race_and_keeps_small_fields(self):
        class Model:
            def predict_proba(self, matrix):
                p = matrix[:, 0]
                return np.column_stack((1 - p, p))

        rows = [{"race_id": str(n), "date": "2026-09-01", "numbers": list(range(1, n + 1)),
                 "top3": [1, 2, 3], "winner": 1, "fresh_raw": list(range(1, n + 1)),
                 "fresh_features": {"context": np.linspace(0, 1, n).reshape(n, 1)}}
                for n in (12, 13)]
        before, after, records = validation.evaluate(Model(), rows)
        np.testing.assert_array_equal(before[0], after[0])
        self.assertEqual(records[0]["old_ranking"], records[0]["candidate_ranking"])
        self.assertEqual(records[1]["candidate_ranking"], list(range(13, 0, -1)))

    @patch.dict(os.environ, {"KEIBA_CENTRAL_DIRT_PLACE_MODE": "active"})
    @patch.object(runtime, "_load_dirt_model", return_value=AscendingModel())
    def test_active_orders_by_top3_signal_and_preserves_my_payload(self, loader):
        race, scores = dirt_fixture()
        original = [s.model_dump() for s in scores]
        changed, config, result = runtime.apply_scope_reranker(race, scores, enabled=True)
        self.assertEqual(result.reranked_ranking, list(range(13, 0, -1)))
        self.assertEqual(result.original_ranking, list(range(1, 14)))
        self.assertEqual(result.candidate_ranking, result.reranked_ranking)
        self.assertEqual(config.preserve_top_n, 0)
        self.assertEqual(config.model_version, validation.MODEL_VERSION)
        self.assertEqual(len(result.candidate_model_scores), 13)
        self.assertEqual(original, [s.model_dump() for s in scores])
        payload = {"race_info": race.race_info.model_dump(), "horse_scores": original}
        candidate_payload = {**payload, "horse_scores": [s.model_dump() for s in changed]}
        self.assertEqual(_custom_prediction_race_item(payload), _custom_prediction_race_item(candidate_payload))
        for before, after in zip(sorted(scores, key=lambda s: s.horse_no),
                                 sorted(changed, key=lambda s: s.horse_no), strict=True):
            self.assertEqual(before.scores, after.scores)
            self.assertEqual(before.score_breakdown.total_score, after.score_breakdown.total_score)

    @patch.dict(os.environ, {"KEIBA_CENTRAL_DIRT_PLACE_MODE": "shadow"})
    @patch.object(runtime, "_load_dirt_model", return_value=AscendingModel())
    def test_shadow_records_candidate_without_changing_scores(self, loader):
        race, scores = dirt_fixture()
        changed, config, result = runtime.apply_scope_reranker(race, scores, enabled=True)
        self.assertEqual(changed, scores)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason_code, "shadow")
        self.assertEqual(result.reranked_ranking, result.original_ranking)
        self.assertEqual(result.candidate_ranking, list(range(13, 0, -1)))
        self.assertEqual(config.mode, "shadow")

    @patch.object(runtime, "_load_dirt_model")
    def test_disabled_small_and_invalid_rosters_never_load_candidate(self, loader):
        for mode in ("off", "misspelled"):
            with patch.dict(os.environ, {"KEIBA_CENTRAL_DIRT_PLACE_MODE": mode}):
                race, scores = dirt_fixture()
                changed, _, result = runtime.apply_scope_reranker(race, scores, enabled=True)
                self.assertEqual(changed, scores)
                self.assertEqual(result.reason_code, "scope_disabled")
        with patch.dict(os.environ, {"KEIBA_CENTRAL_DIRT_PLACE_MODE": "active"}):
            race, scores = dirt_fixture(12)
            self.assertEqual(runtime.apply_scope_reranker(race, scores, enabled=True)[0], scores)
            race, scores = dirt_fixture()
            self.assertEqual(runtime.apply_scope_reranker(race, scores, enabled=False)[0], scores)
            self.assertEqual(runtime.apply_scope_reranker(race, scores[:-1], enabled=True)[2].reason_code,
                             "invalid_runner_roster")
            race.horses[1].horse_no = 1
            self.assertEqual(runtime.apply_scope_reranker(race, scores, enabled=True)[2].reason_code,
                             "invalid_runner_roster")
        loader.assert_not_called()

    @patch.dict(os.environ, {"KEIBA_CENTRAL_DIRT_PLACE_MODE": "active"})
    def test_missing_corrupt_or_incompatible_models_fall_back(self):
        race, scores = dirt_fixture()
        for error in (FileNotFoundError(), ValueError("checksum")):
            with patch.object(runtime, "_load_dirt_model", side_effect=error):
                changed, _, result = runtime.apply_scope_reranker(race, scores, enabled=True)
                self.assertEqual(changed, scores)
                self.assertEqual(result.reason_code, "model_load_failed")
        bad = AscendingModel()
        bad.n_features_in_ = 28
        with patch.object(runtime, "_load_dirt_model", return_value=bad):
            self.assertEqual(runtime.apply_scope_reranker(race, scores, enabled=True)[2].reason_code,
                             "prediction_failed")
        bad = AscendingModel()
        for invalid in (np.nan, 1.1):
            with patch.object(runtime, "_load_dirt_model", return_value=bad), patch.object(
                bad, "predict_proba", return_value=np.full((13, 2), invalid),
            ):
                changed, _, result = runtime.apply_scope_reranker(race, scores, enabled=True)
                self.assertEqual(changed, scores)
                self.assertEqual(result.reason_code, "prediction_failed")

    @patch.object(runtime, "_load_model", return_value=AscendingModel())
    def test_loader_checks_metadata_and_checksum_before_deserializing(self, loader):
        spec = runtime.MODEL_SPECS["central_dirt"]
        original = json.loads((runtime.MODEL_DIRECTORY / spec["metadata_filename"]).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / spec["filename"]
            model_path.write_bytes(b"test-fixture-not-a-pickle")
            metadata_path = Path(directory) / spec["metadata_filename"]
            valid = {**original, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
            for field, value in (("model_version", "wrong"), ("feature_set", "wrong"),
                                 ("training_target", "winner"), ("min_field_size", 12),
                                 ("preserve_top_n", 1), ("blend_weight", 0.5),
                                 ("market_features_used", True), ("n_features", 28),
                                 ("model_sha256", "wrong")):
                with self.subTest(field=field):
                    metadata_path.write_text(json.dumps({**valid, field: value}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        runtime._load_dirt_model(str(model_path))
            loader.assert_not_called()
            metadata_path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertIs(runtime._load_dirt_model(str(model_path)), loader.return_value)
            loader.assert_called_once_with(str(model_path))


if __name__ == "__main__":
    unittest.main()

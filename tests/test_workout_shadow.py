import base64
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from tests.test_turf_shadow import sample
from keiba_llm_agent import main
from keiba_llm_agent.backtest.audit_snapshot import read_backtest_audit_payload
from keiba_llm_agent.fetchers.netkeiba_workout_fetcher import parse_workout_page
from keiba_llm_agent.scoring import workout_shadow as shadow, turf_place_model as active
from pipeline.web_data import agent_predictions
from pipeline.workout_shadow_metrics import compare_workout_shadow
from pipeline.report_public_workout_shadow import report


def fixture():
    race, prediction = sample()
    for horse in race.horses:
        horse.horse_id = str(2023100000 + horse.horse_no)
    prediction.scope_reranker_result.applied = True
    prediction.scope_reranker_result.model_version = active.MODEL_VERSION
    prediction.scope_reranker_result.scope = "central_turf"
    prediction.scope_reranker_result.reranked_ranking = list(range(1, 7))
    html = '<link rel="canonical" href="https://race.netkeiba.com/race/oikiri.html?race_id=' + race.race_info.race_id + '"><table class="OikiriTable">'
    for horse in race.horses:
        html += f'<tr class="HorseList"><td class="Umaban">{horse.horse_no}</td><td><a href="/horse/{horse.horse_id}/">馬</a></td><td class="Rank_B">B</td></tr>'
    html += '</table>'
    raw = html.encode()
    observation = {**parse_workout_page(html, race), "race_id": race.race_info.race_id,
                   "received_at": "2026-11-01T05:00:00+00:00", "raw_response": {
                       "data": base64.b64encode(gzip.compress(raw)).decode(), "sha256": hashlib.sha256(raw).hexdigest()}}
    return race, prediction, observation


class WorkoutShadowTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(shadow, "datetime", wraps=datetime)
        clock = self.clock.start()
        clock.now.return_value = datetime(2026, 11, 1, 5, 5, tzinfo=timezone.utc)
        self.addCleanup(self.clock.stop)

    def test_released_model_uses_324_features_and_preserves_official_prediction(self):
        race, prediction, observation = fixture()
        old = deepcopy((race.model_dump(), prediction.model_dump(), observation))
        result = shadow.evaluate_workout_shadow(race, prediction, observation)
        self.assertEqual(result["status"], "predicted", result)
        self.assertTrue(result["timing"]["eligible_at_generation"])
        self.assertFalse(result["authoritative"])
        self.assertEqual(result["baseline_top5"], [1, 2, 3, 4, 5])
        self.assertEqual(sorted(result["candidate_ranking"]), list(range(1, 7)))
        self.assertEqual(len(result["features"][0]), 324)
        self.assertEqual(old, (race.model_dump(), prediction.model_dump(), observation))
        json.dumps(result, allow_nan=False)

    def test_workout_feature_values_and_strictly_past_history(self):
        grades = {"1": {"horse_id": "a", "grade": "A"}, "2": {"horse_id": "b", "grade": "C"}}
        matrix = shadow.workout_matrix([1, 2], ["a", "b"], "2026-11-01", grades,
                                       {"a": {"date": "2026-10-25", "grade": "B"}})
        np.testing.assert_allclose(matrix, [[3, 2, 1, 7, 1, 0, 0, 0],
                                            [1, np.nan, np.nan, np.nan, -1, np.nan, np.nan, np.nan]], equal_nan=True)
        with self.assertRaises(ValueError):
            shadow.workout_matrix([1], ["a"], "2026-11-01", grades, {"a": {"date": "2026-11-01", "grade": "B"}})

    def test_missing_data_other_scopes_and_corrupt_data_do_not_activate(self):
        race, prediction, observation = fixture()
        with patch.object(shadow, "load_bundle") as load:
            self.assertEqual(shadow.evaluate_workout_shadow(race, prediction, {})["reason_code"], "workout_unavailable")
            for source, surface in (("local", "ダート"), ("central", "ダート")):
                race.race_info.source = source
                race.race_info.surface = surface
                self.assertIsNone(shadow.evaluate_workout_shadow(race, prediction, observation))
            load.assert_not_called()
        race, prediction, observation = fixture()
        observation["horses"]["1"]["grade"] = "A"
        self.assertEqual(shadow.evaluate_workout_shadow(race, prediction, observation)["status"], "failed")

    def test_market_and_critic_text_cannot_change_predictions(self):
        race, prediction, observation = fixture()
        first = shadow.evaluate_workout_shadow(race, prediction, observation)
        for horse in race.horses:
            horse.odds = 1.1
            horse.popularity = 1
            for run in horse.recent_runs:
                run.odds = 999
                run.popularity = 18
        raw = gzip.decompress(base64.b64decode(observation["raw_response"]["data"]))
        raw = raw.replace(b'</tr>', '<td class="Training_Critic">評価用の短評</td></tr>'.encode())
        observation.update(parse_workout_page(raw.decode(), race))
        observation["raw_response"] = {"data": base64.b64encode(gzip.compress(raw)).decode(),
                                       "sha256": hashlib.sha256(raw).hexdigest()}
        second = shadow.evaluate_workout_shadow(race, prediction, observation)
        self.assertEqual(first["features"], second["features"])
        self.assertEqual(first["candidate_ranking"], second["candidate_ranking"])

    def test_historical_or_late_samples_are_never_forward_accuracy_evidence(self):
        race, prediction, observation = fixture()
        race.race_info.race_date = "2026-10-25"
        self.assertEqual(shadow.evaluate_workout_shadow(race, prediction, observation)["reason_code"], "not_prospective")
        race, prediction, observation = fixture()
        observation["received_at"] = "2026-11-02T05:00:00+00:00"
        self.assertEqual(shadow.evaluate_workout_shadow(race, prediction, observation)["status"], "failed")

    def test_pipeline_saves_both_and_render_settlement_compares_them(self):
        race, prediction, observation = fixture()
        before = prediction.model_dump()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(main, "DEFAULT_BACKTEST_AUDIT_DIR", root / "local"):
                local_path = main._write_prediction_backtest_audit(race_data=race, prediction=prediction,
                    supplementary_observations={"public_workout": observation})
            audit = read_backtest_audit_payload(local_path)
            self.assertEqual(audit["shadow_predictions"]["public_workout"]["status"], "predicted")
            audit["outputs"]["prediction"]["prediction_received_at"] = "2026-11-01T05:06:00+00:00"
            rid = race.race_info.race_id
            result = {"race_id": rid, "result": {"1st": 1, "2nd": 2, "3rd": 6}}
            persisted = agent_predictions.write_agent_backtest_audit_payload(root / "render", rid, audit)
            agent_predictions.write_agent_backtest_audit_version_payload(root / "render", rid, "final", "test", audit)
            agent_predictions.attach_result_to_agent_backtest_audit(root / "render", rid, result)
            compared = read_backtest_audit_payload(persisted)["outcomes"]["public_workout_comparison"]
            self.assertEqual(compared["status"], "settled", compared)
            self.assertEqual(compared["baseline"]["covered_slots"], 2)
            expected = len(set(audit["shadow_predictions"]["public_workout"]["candidate_top5"]) & {1, 2, 6})
            self.assertEqual(compared["candidate"]["covered_slots"], expected)
            stats = report(root / "render/data")
            self.assertEqual(stats["overall"]["races"], 1)
            self.assertEqual(stats["overall"]["delta_covered_slots"], expected - 2)
            late = deepcopy(audit)
            late["outputs"]["prediction"]["prediction_received_at"] = "2026-11-01T06:01:00+00:00"
            self.assertEqual(compare_workout_shadow(late, result)["status"], "excluded")
            self.assertEqual(compare_workout_shadow(audit, None)["status"], "pending")
        self.assertEqual(prediction.model_dump(), before)

    def test_bad_model_does_not_break_official_prediction(self):
        race, prediction, observation = fixture()
        before = prediction.model_dump()
        with patch.object(shadow, "load_bundle", side_effect=ValueError("checksum")):
            self.assertEqual(shadow.evaluate_workout_shadow(race, prediction, observation)["status"], "failed")
        self.assertEqual(prediction.model_dump(), before)


if __name__ == "__main__":
    unittest.main()

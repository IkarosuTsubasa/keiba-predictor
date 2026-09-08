from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from keiba_llm_agent.schemas.prediction import (
    HorseScore,
    ScoreBreakdown,
    TotalScoreBreakdown,
)
from keiba_llm_agent.schemas.race_data import (
    HorseEntry,
    RaceData,
    RaceInfo,
    RecentRun,
)
from keiba_llm_agent.scoring import scope_reranker


class _FixedLocalBaselineModel:
    n_features_in_ = 35

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probabilities = np.asarray([0.1, 0.4, 0.9, 0.8])
        if matrix.shape[0] != len(probabilities):
            raise AssertionError("unexpected horse count")
        return np.column_stack((1.0 - probabilities, probabilities))


class _FixedLocalGeometryModel:
    n_features_in_ = 48

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probabilities = np.asarray([0.2, 0.1, 0.3, 0.9])
        if matrix.shape[0] != len(probabilities):
            raise AssertionError("unexpected horse count")
        return np.column_stack((1.0 - probabilities, probabilities))


class _FixedCentralTurfModel:
    n_features_in_ = 28

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probabilities = np.asarray([0.1, 0.4, 0.9, 0.8])
        if matrix.shape[0] != len(probabilities):
            raise AssertionError("unexpected horse count")
        return np.column_stack((1.0 - probabilities, probabilities))


class _FixedCentralBaselineModel:
    n_features_in_ = 13

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probabilities = np.asarray([0.9, 0.4, 0.1, 0.2])
        if matrix.shape[0] != len(probabilities):
            raise AssertionError("unexpected horse count")
        return np.column_stack((1.0 - probabilities, probabilities))


def _central_model_for_path(path: str) -> object:
    if path.endswith("central_turf_v1.joblib"):
        return _FixedCentralBaselineModel()
    return _FixedCentralTurfModel()


def _local_model_for_path(path: str) -> object:
    if path.endswith("local_v1.joblib"):
        return _FixedLocalBaselineModel()
    return _FixedLocalGeometryModel()


def _race_data(*, source: str, surface: str) -> RaceData:
    horses = []
    for horse_no in range(1, 5):
        horses.append(
            HorseEntry(
                horse_no=horse_no,
                horse_name=f"ホース{horse_no}",
                jockey=f"騎手{horse_no}",
                carried_weight=55.0 + horse_no * 0.2,
                odds=2.0 + horse_no,
                popularity=horse_no,
                recent_runs=[
                    RecentRun(
                        race_id=f"past-{horse_no}",
                        date="2026-07-01",
                        course="大井" if source == "local" else "東京",
                        surface=surface,
                        distance=1600,
                        track_condition="良",
                        finish=horse_no,
                        field_size=12,
                        jockey=f"騎手{horse_no}",
                        corner_positions=[horse_no, horse_no],
                        final_3f=36.0 + horse_no * 0.1,
                        margin=str(horse_no * 0.2),
                    )
                ],
            )
        )
    return RaceData(
        race_info=RaceInfo(
            race_id="scope-reranker-test",
            race_name="テスト競走",
            race_date="2026-07-20",
            course="大井" if source == "local" else "東京",
            surface=surface,
            distance=1600,
            track_condition="良",
            source=source,
            scope_key=source,
        ),
        horses=horses,
    )


def _horse_scores() -> list[HorseScore]:
    scores = []
    for horse_no, total in enumerate((80.0, 79.0, 78.0, 77.0), start=1):
        scores.append(
            HorseScore(
                horse_no=horse_no,
                horse_name=f"ホース{horse_no}",
                scores=ScoreBreakdown(
                    recent_form=7,
                    distance_fit=7,
                    course_fit=7,
                    track_condition_fit=7,
                    jockey_fit=7,
                    odds_value=5,
                    risk=-2,
                    ability_score=8 - horse_no,
                    recent_quality_score=7,
                    trend_score=5,
                    condition_fit_score=6,
                    race_level_score=5,
                    pace_jockey_score=6,
                ),
                score_breakdown=TotalScoreBreakdown(
                    base_total_score=total,
                    total_score=total,
                ),
                odds=2.0 + horse_no,
                popularity=horse_no,
                total_score=total,
                reason="テスト",
            )
        )
    return scores


class ScopeRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        scope_reranker.clear_scope_reranker_caches()

    def tearDown(self) -> None:
        scope_reranker.clear_scope_reranker_caches()

    @patch.object(
        scope_reranker,
        "_load_model",
        side_effect=_local_model_for_path,
    )
    def test_local_reranker_applies_without_market_inputs(
        self,
        _load_model: object,
    ) -> None:
        race_data = _race_data(source="local", surface="ダート")
        adjusted, config, result = scope_reranker.apply_scope_reranker(
            race_data,
            _horse_scores(),
            enabled=True,
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.blend_weight, 1.0)
        self.assertEqual(config.preserve_top_n, 2)
        self.assertTrue(result.applied)
        self.assertEqual(result.original_ranking, [1, 2, 3, 4])
        self.assertEqual(result.baseline_ranking, [3, 1, 2, 4])
        self.assertEqual(result.reranked_ranking, [3, 1, 4, 2])
        self.assertEqual(
            [score.horse_no for score in adjusted],
            [3, 1, 4, 2],
        )

        market_changed = race_data.model_copy(deep=True)
        for horse in market_changed.horses:
            horse.odds = 999.0 - horse.horse_no
            horse.popularity = 5 - horse.horse_no
        changed_scores = _horse_scores()
        for score in changed_scores:
            score.odds = 999.0 - score.horse_no
            score.popularity = 5 - score.horse_no
        changed, _, changed_result = scope_reranker.apply_scope_reranker(
            market_changed,
            changed_scores,
            enabled=True,
        )

        self.assertEqual(
            changed_result.reranked_ranking,
            result.reranked_ranking,
        )
        self.assertEqual(
            [score.scope_reranker.model_score for score in changed],
            [score.scope_reranker.model_score for score in adjusted],
        )
        for score in adjusted:
            self.assertTrue(score.scope_reranker.applied)
            self.assertAlmostEqual(
                score.score_breakdown.total_score_after_reranker,
                score.total_score,
            )
            self.assertAlmostEqual(
                score.scope_reranker.score_adjustment,
                score.score_breakdown.scope_reranker_adjustment,
            )

    def test_small_central_dirt_field_is_unchanged(self) -> None:
        adjusted, config, result = scope_reranker.apply_scope_reranker(
            _race_data(source="central", surface="ダート"),
            _horse_scores(),
            enabled=True,
        )

        self.assertTrue(config.enabled)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason_code, "field_size_below_minimum")
        self.assertEqual([score.horse_no for score in adjusted], [1, 2, 3, 4])

    @patch.object(
        scope_reranker,
        "_load_model",
        side_effect=_central_model_for_path,
    )
    def test_central_turf_preserves_top1_and_reranks_the_rest(
        self,
        _load_model: object,
    ) -> None:
        adjusted, config, result = scope_reranker.apply_scope_reranker(
            _race_data(source="central", surface="芝"),
            _horse_scores(),
            enabled=True,
        )

        self.assertEqual(config.blend_weight, 0.5)
        self.assertEqual(config.preserve_top_n, 1)
        self.assertTrue(result.applied)
        self.assertEqual(result.original_ranking, [1, 2, 3, 4])
        self.assertEqual(result.baseline_ranking, [1, 2, 3, 4])
        self.assertEqual(result.reranked_ranking, [1, 3, 2, 4])
        self.assertEqual(
            [score.horse_no for score in adjusted],
            [1, 3, 2, 4],
        )

    def test_disabled_reranker_is_unchanged(self) -> None:
        adjusted, config, result = scope_reranker.apply_scope_reranker(
            _race_data(source="local", surface="ダート"),
            _horse_scores(),
            enabled=False,
        )

        self.assertFalse(config.enabled)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason_code, "disabled")
        self.assertEqual([score.horse_no for score in adjusted], [1, 2, 3, 4])

    @patch.object(
        scope_reranker,
        "_load_model",
        side_effect=FileNotFoundError,
    )
    def test_missing_model_falls_back_safely(
        self,
        _load_model: object,
    ) -> None:
        adjusted, _, result = scope_reranker.apply_scope_reranker(
            _race_data(source="local", surface="ダート"),
            _horse_scores(),
            enabled=True,
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.reason_code, "model_load_failed")
        self.assertEqual([score.horse_no for score in adjusted], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()

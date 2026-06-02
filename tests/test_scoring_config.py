from __future__ import annotations

import unittest

from keiba_llm_agent.config.scoring_config import (
    DEFAULT_SCORING_PROFILE,
    DEFAULT_SCORING_MODE,
    DEFAULT_SCORING_WEIGHTS,
    effective_scoring_weights,
    resolve_scoring_config,
    resolve_scoring_profile_config,
)


class ScoringConfigTests(unittest.TestCase):
    def test_accuracy_default_profile_returns_candidate_default_with_recovery(self) -> None:
        profile_config, warnings = resolve_scoring_profile_config("accuracy_default")
        self.assertEqual(profile_config.scoring_profile, "accuracy_default")
        self.assertEqual(profile_config.scoring_config.scoring_mode, "candidate_default")
        self.assertEqual(profile_config.scoring_config.pedigree_weight, 0.2)
        self.assertEqual(profile_config.scoring_config.race_level_weight, 1.0)
        self.assertEqual(profile_config.scoring_config.pace_weight, 0.0)
        self.assertEqual(profile_config.scoring_config.conditional_weight_profile, "candidate_default_v2")
        self.assertFalse(profile_config.scoring_config.use_market_score_in_ranking)
        self.assertEqual(profile_config.scoring_config.market_signal_weight, 0.0)
        self.assertTrue(profile_config.borderline_recovery_enabled)
        self.assertEqual(warnings, [])

    def test_safe_baseline_profile_returns_base_only_without_recovery(self) -> None:
        profile_config, _ = resolve_scoring_profile_config("safe_baseline")
        self.assertEqual(profile_config.scoring_profile, "safe_baseline")
        self.assertEqual(profile_config.scoring_config.scoring_mode, "base_only")
        self.assertEqual(profile_config.scoring_config.pedigree_weight, 0.0)
        self.assertEqual(profile_config.scoring_config.race_level_weight, 0.0)
        self.assertEqual(profile_config.scoring_config.pace_weight, 0.0)
        self.assertEqual(profile_config.scoring_config.conditional_weight_profile, "none")
        self.assertFalse(profile_config.scoring_config.use_market_score_in_ranking)
        self.assertEqual(profile_config.scoring_config.market_signal_weight, 0.0)
        self.assertFalse(profile_config.borderline_recovery_enabled)

    def test_candidate_default_returns_expected_weights(self) -> None:
        config, warnings = resolve_scoring_config("candidate_default")
        self.assertEqual(config.scoring_mode, "candidate_default")
        self.assertEqual(config.pedigree_weight, 0.2)
        self.assertEqual(config.race_level_weight, 1.0)
        self.assertEqual(config.pace_weight, 0.0)
        self.assertEqual(config.conditional_weight_profile, "candidate_default_v2")
        self.assertFalse(config.use_market_score_in_ranking)
        self.assertEqual(config.market_signal_weight, 0.0)
        self.assertEqual(warnings, [])

    def test_base_only_returns_zero_weights(self) -> None:
        config, _ = resolve_scoring_config("base_only")
        self.assertEqual(config.pedigree_weight, 0.0)
        self.assertEqual(config.race_level_weight, 0.0)
        self.assertEqual(config.pace_weight, 0.0)
        self.assertEqual(config.conditional_weight_profile, "none")

    def test_current_full_returns_all_one_weights(self) -> None:
        config, _ = resolve_scoring_config("current_full")
        self.assertEqual(config.pedigree_weight, 1.0)
        self.assertEqual(config.race_level_weight, 1.0)
        self.assertEqual(config.pace_weight, 1.0)
        self.assertEqual(config.conditional_weight_profile, "none")

    def test_custom_can_override_weights(self) -> None:
        config, warnings = resolve_scoring_config(
            "custom",
            pedigree_weight=0.2,
            race_level_weight=1.0,
            pace_weight=0.0,
        )
        self.assertEqual(config.scoring_mode, "custom")
        self.assertEqual(config.pedigree_weight, 0.2)
        self.assertEqual(config.race_level_weight, 1.0)
        self.assertEqual(config.pace_weight, 0.0)
        self.assertEqual(config.conditional_weight_profile, "none")
        self.assertEqual(warnings, [])

    def test_explicit_candidate_default_weight_override_disables_conditional_weights(self) -> None:
        config, warnings = resolve_scoring_config(
            "candidate_default",
            race_level_weight=1.0,
        )
        self.assertEqual(config.scoring_mode, "candidate_default")
        self.assertEqual(config.race_level_weight, 1.0)
        self.assertEqual(config.conditional_weight_profile, "none")
        self.assertEqual(warnings, [])

    def test_candidate_default_effective_weights_by_race_condition(self) -> None:
        config, _ = resolve_scoring_config("candidate_default")
        turf_weights = effective_scoring_weights(config, surface="芝", field_size=12)
        dirt_weights = effective_scoring_weights(config, surface="ダート", field_size=12)
        large_field_weights = effective_scoring_weights(config, surface="ダート", field_size=14)
        self.assertEqual(turf_weights["pedigree_weight"], 0.2)
        self.assertEqual(turf_weights["race_level_weight"], 1.2)
        self.assertEqual(dirt_weights["pedigree_weight"], 0.3)
        self.assertEqual(dirt_weights["race_level_weight"], 1.0)
        self.assertEqual(large_field_weights["pedigree_weight"], 0.3)
        self.assertEqual(large_field_weights["race_level_weight"], 1.2)

    def test_effective_weights_do_not_apply_when_profile_is_none(self) -> None:
        config, _ = resolve_scoring_config("custom", pedigree_weight=0.2, race_level_weight=1.0, pace_weight=0.0)
        weights = effective_scoring_weights(config, surface="芝", field_size=18)
        self.assertEqual(weights["pedigree_weight"], 0.2)
        self.assertEqual(weights["race_level_weight"], 1.0)
        self.assertEqual(weights["pace_weight"], 0.0)

    def test_custom_without_weights_falls_back_to_candidate_default_with_warning(self) -> None:
        config, warnings = resolve_scoring_config("custom")
        self.assertEqual(config.scoring_mode, DEFAULT_SCORING_MODE)
        self.assertEqual(config.pedigree_weight, DEFAULT_SCORING_WEIGHTS["pedigree_weight"])
        self.assertEqual(config.race_level_weight, DEFAULT_SCORING_WEIGHTS["race_level_weight"])
        self.assertEqual(config.pace_weight, DEFAULT_SCORING_WEIGHTS["pace_weight"])
        self.assertTrue(warnings)

    def test_custom_profile_without_weights_falls_back_to_accuracy_default_with_warning(self) -> None:
        profile_config, warnings = resolve_scoring_profile_config(
            "accuracy_default",
            scoring_mode="custom",
        )
        self.assertEqual(profile_config.scoring_profile, DEFAULT_SCORING_PROFILE)
        self.assertEqual(profile_config.scoring_config.scoring_mode, DEFAULT_SCORING_MODE)
        self.assertTrue(profile_config.borderline_recovery_enabled)
        self.assertTrue(warnings)

    def test_explicit_scoring_mode_overrides_profile_default_mode(self) -> None:
        profile_config, _ = resolve_scoring_profile_config(
            "accuracy_default",
            scoring_mode="base_only",
        )
        self.assertEqual(profile_config.scoring_profile, "accuracy_default")
        self.assertEqual(profile_config.scoring_config.scoring_mode, "base_only")
        self.assertFalse(profile_config.borderline_recovery_enabled)

    def test_explicit_recovery_override_wins_over_profile(self) -> None:
        profile_config, _ = resolve_scoring_profile_config(
            "safe_baseline",
            borderline_recovery_enabled=True,
        )
        self.assertEqual(profile_config.scoring_profile, "safe_baseline")
        self.assertTrue(profile_config.borderline_recovery_enabled)

    def test_invalid_scoring_mode_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid scoring_mode"):
            resolve_scoring_config("bad_mode")

    def test_invalid_scoring_profile_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid scoring_profile"):
            resolve_scoring_profile_config("bad_profile")


if __name__ == "__main__":
    unittest.main()

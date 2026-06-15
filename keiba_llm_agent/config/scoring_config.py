from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SCORING_PROFILE = "accuracy_default"
DEFAULT_SCORING_MODE = "candidate_default"
DEFAULT_SCORING_WEIGHTS = {
    "pedigree_weight": 0.2,
    "race_level_weight": 1.0,
    "pace_weight": 0.0,
}
LOCAL_SCORING_PROFILE = "local_accuracy_default"
LOCAL_SCORING_MODE = "local_candidate_default"
LOCAL_SCORING_WEIGHTS = {
    "pedigree_weight": 0.75,
    "race_level_weight": 0.5,
    "pace_weight": 1.0,
}
NO_CONDITIONAL_WEIGHT_PROFILE = "none"
DEFAULT_CONDITIONAL_WEIGHT_PROFILE = "candidate_default_v2"
DEFAULT_CONDITIONAL_WEIGHT_RULES = {
    "dirt_pedigree_weight": 0.3,
    "turf_race_level_weight": 1.2,
    "large_field_min_size": 14,
    "large_field_race_level_weight": 1.2,
}
DEFAULT_MARKET_SIGNAL_CONFIG = {
    "use_market_score_in_ranking": False,
    "market_signal_weight": 0.0,
}

SCORING_MODES: dict[str, dict[str, float]] = {
    "base_only": {
        "pedigree_weight": 0.0,
        "race_level_weight": 0.0,
        "pace_weight": 0.0,
    },
    "pedigree_only": {
        "pedigree_weight": 1.0,
        "race_level_weight": 0.0,
        "pace_weight": 0.0,
    },
    "race_level_only": {
        "pedigree_weight": 0.0,
        "race_level_weight": 1.0,
        "pace_weight": 0.0,
    },
    "pace_only": {
        "pedigree_weight": 0.0,
        "race_level_weight": 0.0,
        "pace_weight": 1.0,
    },
    "current_full": {
        "pedigree_weight": 1.0,
        "race_level_weight": 1.0,
        "pace_weight": 1.0,
    },
    "candidate_default": dict(DEFAULT_SCORING_WEIGHTS),
    LOCAL_SCORING_MODE: dict(LOCAL_SCORING_WEIGHTS),
}

SCORING_PROFILES: dict[str, dict[str, object]] = {
    "accuracy_default": {
        "scoring_mode": "candidate_default",
        "borderline_recovery_enabled": True,
    },
    LOCAL_SCORING_PROFILE: {
        "scoring_mode": LOCAL_SCORING_MODE,
        "borderline_recovery_enabled": False,
    },
    "safe_baseline": {
        "scoring_mode": "base_only",
        "borderline_recovery_enabled": False,
    },
}


@dataclass(frozen=True)
class ScoringConfig:
    scoring_mode: str = DEFAULT_SCORING_MODE
    pedigree_weight: float = DEFAULT_SCORING_WEIGHTS["pedigree_weight"]
    race_level_weight: float = DEFAULT_SCORING_WEIGHTS["race_level_weight"]
    pace_weight: float = DEFAULT_SCORING_WEIGHTS["pace_weight"]
    conditional_weight_profile: str = DEFAULT_CONDITIONAL_WEIGHT_PROFILE
    use_market_score_in_ranking: bool = DEFAULT_MARKET_SIGNAL_CONFIG["use_market_score_in_ranking"]
    market_signal_weight: float = DEFAULT_MARKET_SIGNAL_CONFIG["market_signal_weight"]

    def model_dump(self) -> dict[str, float | str | bool]:
        return {
            "scoring_mode": self.scoring_mode,
            "pedigree_weight": self.pedigree_weight,
            "race_level_weight": self.race_level_weight,
            "pace_weight": self.pace_weight,
            "conditional_weight_profile": self.conditional_weight_profile,
            "use_market_score_in_ranking": self.use_market_score_in_ranking,
            "market_signal_weight": self.market_signal_weight,
        }


@dataclass(frozen=True)
class ScoringProfileConfig:
    scoring_profile: str = DEFAULT_SCORING_PROFILE
    scoring_config: ScoringConfig = ScoringConfig()
    borderline_recovery_enabled: bool = True

    def model_dump(self) -> dict[str, object]:
        payload = self.scoring_config.model_dump()
        payload.update(
            {
                "scoring_profile": self.scoring_profile,
                "borderline_recovery_enabled": self.borderline_recovery_enabled,
            }
        )
        return payload


def resolve_scoring_config(
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
) -> tuple[ScoringConfig, list[str]]:
    mode = (scoring_mode or DEFAULT_SCORING_MODE).strip().lower()
    warnings: list[str] = []

    if mode == "custom":
        if pedigree_weight is None and race_level_weight is None and pace_weight is None:
            warnings.append("custom scoring-mode 未指定任何权重，已回退到 candidate_default。")
            mode = DEFAULT_SCORING_MODE
        else:
            return (
                ScoringConfig(
                    scoring_mode="custom",
                    pedigree_weight=DEFAULT_SCORING_WEIGHTS["pedigree_weight"] if pedigree_weight is None else pedigree_weight,
                    race_level_weight=DEFAULT_SCORING_WEIGHTS["race_level_weight"] if race_level_weight is None else race_level_weight,
                    pace_weight=DEFAULT_SCORING_WEIGHTS["pace_weight"] if pace_weight is None else pace_weight,
                    conditional_weight_profile=NO_CONDITIONAL_WEIGHT_PROFILE,
                    use_market_score_in_ranking=DEFAULT_MARKET_SIGNAL_CONFIG["use_market_score_in_ranking"],
                    market_signal_weight=DEFAULT_MARKET_SIGNAL_CONFIG["market_signal_weight"],
                ),
                warnings,
            )

    if mode not in SCORING_MODES:
        raise ValueError(f"invalid scoring_mode: {mode}")

    resolved = dict(SCORING_MODES[mode])
    if pedigree_weight is not None:
        resolved["pedigree_weight"] = pedigree_weight
    if race_level_weight is not None:
        resolved["race_level_weight"] = race_level_weight
    if pace_weight is not None:
        resolved["pace_weight"] = pace_weight
    has_explicit_weight_override = pedigree_weight is not None or race_level_weight is not None or pace_weight is not None
    conditional_weight_profile = (
        DEFAULT_CONDITIONAL_WEIGHT_PROFILE
        if mode == DEFAULT_SCORING_MODE and not has_explicit_weight_override
        else NO_CONDITIONAL_WEIGHT_PROFILE
    )
    return (
        ScoringConfig(
            scoring_mode=mode,
            pedigree_weight=resolved["pedigree_weight"],
            race_level_weight=resolved["race_level_weight"],
            pace_weight=resolved["pace_weight"],
            conditional_weight_profile=conditional_weight_profile,
            use_market_score_in_ranking=DEFAULT_MARKET_SIGNAL_CONFIG["use_market_score_in_ranking"],
            market_signal_weight=DEFAULT_MARKET_SIGNAL_CONFIG["market_signal_weight"],
        ),
        warnings,
    )


def effective_scoring_weights(
    scoring_config: object,
    *,
    surface: str | None,
    field_size: int,
) -> dict[str, float]:
    weights = {
        "pedigree_weight": float(getattr(scoring_config, "pedigree_weight", DEFAULT_SCORING_WEIGHTS["pedigree_weight"])),
        "race_level_weight": float(getattr(scoring_config, "race_level_weight", DEFAULT_SCORING_WEIGHTS["race_level_weight"])),
        "pace_weight": float(getattr(scoring_config, "pace_weight", DEFAULT_SCORING_WEIGHTS["pace_weight"])),
    }
    if getattr(scoring_config, "conditional_weight_profile", NO_CONDITIONAL_WEIGHT_PROFILE) != DEFAULT_CONDITIONAL_WEIGHT_PROFILE:
        return weights
    if getattr(scoring_config, "scoring_mode", DEFAULT_SCORING_MODE) != DEFAULT_SCORING_MODE:
        return weights

    if surface == "ダート":
        weights["pedigree_weight"] = float(DEFAULT_CONDITIONAL_WEIGHT_RULES["dirt_pedigree_weight"])
    if surface == "芝" or field_size >= int(DEFAULT_CONDITIONAL_WEIGHT_RULES["large_field_min_size"]):
        weights["race_level_weight"] = float(
            DEFAULT_CONDITIONAL_WEIGHT_RULES[
                "turf_race_level_weight" if surface == "芝" else "large_field_race_level_weight"
            ]
        )
    return weights


def resolve_scoring_profile_config(
    scoring_profile: str | None = None,
    scoring_mode: str | None = None,
    pedigree_weight: float | None = None,
    race_level_weight: float | None = None,
    pace_weight: float | None = None,
    borderline_recovery_enabled: bool | None = None,
) -> tuple[ScoringProfileConfig, list[str]]:
    profile_name = (scoring_profile or DEFAULT_SCORING_PROFILE).strip().lower()
    if profile_name not in SCORING_PROFILES:
        raise ValueError(f"invalid scoring_profile: {profile_name}")

    profile_defaults = SCORING_PROFILES[profile_name]
    explicit_scoring_mode = scoring_mode is not None
    resolved_mode = (
        scoring_mode.strip().lower()
        if explicit_scoring_mode and scoring_mode is not None
        else str(profile_defaults["scoring_mode"])
    )
    resolved_config, warnings = resolve_scoring_config(
        scoring_mode=resolved_mode,
        pedigree_weight=pedigree_weight,
        race_level_weight=race_level_weight,
        pace_weight=pace_weight,
    )

    if borderline_recovery_enabled is not None:
        resolved_recovery_enabled = borderline_recovery_enabled
    elif explicit_scoring_mode:
        resolved_recovery_enabled = resolved_config.scoring_mode in {"candidate_default", "custom"}
    else:
        resolved_recovery_enabled = bool(profile_defaults["borderline_recovery_enabled"])

    return (
        ScoringProfileConfig(
            scoring_profile=profile_name,
            scoring_config=resolved_config,
            borderline_recovery_enabled=resolved_recovery_enabled,
        ),
        warnings,
    )

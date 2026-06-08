from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .scoring_config import (
    DEFAULT_CONDITIONAL_WEIGHT_PROFILE,
    DEFAULT_CONDITIONAL_WEIGHT_RULES,
    DEFAULT_SCORING_PROFILE,
    DEFAULT_SCORING_MODE,
    DEFAULT_SCORING_WEIGHTS,
    LOCAL_SCORING_PROFILE,
    NO_CONDITIONAL_WEIGHT_PROFILE,
    SCORING_MODES,
    SCORING_PROFILES,
    ScoringConfig,
    ScoringProfileConfig,
    effective_scoring_weights,
    resolve_scoring_config,
    resolve_scoring_profile_config,
)


DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _parse_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _get_setting(name: str, default: str | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value
    dotenv_values = _parse_dotenv(DOTENV_PATH)
    return dotenv_values.get(name, default)


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = _get_setting(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    enable_fallback: bool = True


def get_llm_config(provider_override: str | None = None) -> LLMConfig:
    provider = (provider_override or _get_setting("KEIBA_LLM_PROVIDER", "openai") or "openai").strip().lower()
    if provider not in {"mock", "openai"}:
        provider = "openai"
    return LLMConfig(
        provider=provider,
        openai_api_key=_get_setting("OPENAI_API_KEY"),
        openai_model=(_get_setting("OPENAI_MODEL", "gpt-5.4-mini") or "gpt-5.4-mini").strip() or "gpt-5.4-mini",
        enable_fallback=_parse_bool_env("KEIBA_LLM_ENABLE_FALLBACK", True),
    )


__all__ = [
    "DEFAULT_SCORING_PROFILE",
    "DEFAULT_CONDITIONAL_WEIGHT_PROFILE",
    "DEFAULT_CONDITIONAL_WEIGHT_RULES",
    "DEFAULT_SCORING_MODE",
    "DEFAULT_SCORING_WEIGHTS",
    "LOCAL_SCORING_PROFILE",
    "DOTENV_PATH",
    "LLMConfig",
    "NO_CONDITIONAL_WEIGHT_PROFILE",
    "SCORING_MODES",
    "SCORING_PROFILES",
    "ScoringConfig",
    "ScoringProfileConfig",
    "effective_scoring_weights",
    "get_llm_config",
    "resolve_scoring_config",
    "resolve_scoring_profile_config",
]

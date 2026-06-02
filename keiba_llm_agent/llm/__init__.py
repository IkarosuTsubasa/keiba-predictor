from __future__ import annotations

import warnings

from keiba_llm_agent.config import get_llm_config
from keiba_llm_agent.llm.llm_client import BaseLLMClient, extract_json_object
from keiba_llm_agent.llm.mock_llm_client import MockLLMClient
from keiba_llm_agent.llm.openai_llm_client import OpenAILLMClient


def create_llm_client(provider_override: str | None = None) -> BaseLLMClient:
    config = get_llm_config(provider_override)
    fallback_client = MockLLMClient()
    if config.provider == "mock":
        return fallback_client
    if not config.openai_api_key:
        if config.enable_fallback:
            warnings.warn("OPENAI_API_KEY is missing; fallback to MockLLMClient", stacklevel=2)
            return fallback_client
        raise RuntimeError("OPENAI_API_KEY is required when KEIBA_LLM_PROVIDER=openai")
    try:
        return OpenAILLMClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            fallback_client=fallback_client,
            enable_fallback=config.enable_fallback,
        )
    except RuntimeError as exc:
        if config.enable_fallback:
            warnings.warn(f"{exc}; fallback to MockLLMClient", stacklevel=2)
            return fallback_client
        raise


__all__ = [
    "BaseLLMClient",
    "MockLLMClient",
    "OpenAILLMClient",
    "create_llm_client",
    "extract_json_object",
]

from __future__ import annotations

import warnings

from keiba_llm_agent.llm.llm_client import BaseLLMClient, extract_json_object


class OpenAILLMClient(BaseLLMClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        fallback_client: BaseLLMClient | None = None,
        enable_fallback: bool = True,
    ) -> None:
        super().__init__(fallback_client=fallback_client)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.enable_fallback = enable_fallback
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    def _extract_response_text(self, response: object) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        text_parts: list[str] = []
        for output_item in getattr(response, "output", []) or []:
            for content_item in getattr(output_item, "content", []) or []:
                content_type = getattr(content_item, "type", "")
                if content_type in {"output_text", "text"}:
                    text_value = getattr(content_item, "text", "")
                    if text_value:
                        text_parts.append(text_value)
        if text_parts:
            return "".join(text_parts)
        raise RuntimeError("OpenAI response did not contain text output")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str | None = None,
    ) -> dict:
        self.last_fallback_used = False
        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = self._extract_response_text(response)
            return extract_json_object(text)
        except Exception as exc:
            if self.enable_fallback and self.fallback_client is not None:
                self.last_fallback_used = True
                warnings.warn(f"OpenAI JSON generation failed; fallback to mock: {exc}", stacklevel=2)
                return self.fallback_client.generate_json(system_prompt, user_prompt, schema_name=schema_name)
            raise RuntimeError(f"OpenAI JSON generation failed: {exc}") from exc

from __future__ import annotations

import json
import warnings

from keiba_llm_agent.llm.llm_client import BaseLLMClient, extract_json_object


class GeminiLLMClient(BaseLLMClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        fallback_client: BaseLLMClient | None = None,
        enable_fallback: bool = True,
    ) -> None:
        super().__init__(fallback_client=fallback_client)
        self.api_key = api_key
        self.model = model
        self.enable_fallback = enable_fallback
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai package is not installed") from exc
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _extract_response_text(response: object) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", []) or []
        parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                value = getattr(part, "text", "")
                if value:
                    parts.append(value)
        if parts:
            return "".join(parts)
        raise RuntimeError("Gemini response did not contain text output")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str | None = None,
    ) -> dict:
        self.last_fallback_used = False
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config={
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                },
            )
            text = self._extract_response_text(response)
            return extract_json_object(text)
        except Exception as exc:
            if self.enable_fallback and self.fallback_client is not None:
                self.last_fallback_used = True
                warnings.warn(f"Gemini JSON generation failed; fallback to mock: {exc}", stacklevel=2)
                if schema_name == "analysis":
                    return self.fallback_client.generate_analysis(system_prompt, json.loads(user_prompt))
                if schema_name == "review":
                    return self.fallback_client.generate_review(system_prompt, json.loads(user_prompt))
                return self.fallback_client.generate_json(system_prompt, user_prompt, schema_name=schema_name)
            raise RuntimeError(f"Gemini JSON generation failed: {exc}") from exc

    def generate_analysis(self, prompt: str, payload: dict) -> dict:
        return self.generate_json(
            prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            schema_name="analysis",
        )

    def generate_review(self, prompt: str, payload: dict) -> dict:
        return self.generate_json(
            prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            schema_name="review",
        )

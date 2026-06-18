from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from keiba_llm_agent import config as config_module
from keiba_llm_agent.llm import GeminiLLMClient, MockLLMClient, create_llm_client, extract_json_object
from keiba_llm_agent.llm.openai_llm_client import OpenAILLMClient
from keiba_llm_agent.main import run_llm_check


class _FailingResponses:
    def create(self, *args, **kwargs):
        raise RuntimeError("boom")


class _FailingOpenAI:
    def __init__(self, *args, **kwargs) -> None:
        self.responses = _FailingResponses()


class _FallbackLikeClient:
    def __init__(self) -> None:
        self.last_fallback_used = True

    def generate_json(self, *args, **kwargs):
        return {"ok": True}


class _FakeGeminiResponse:
    text = '{"ok": true}'


class _FakeGeminiModels:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.last_kwargs = None

    def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        if self.fail:
            raise RuntimeError("gemini boom")
        return _FakeGeminiResponse()


class _FakeGenAIClient:
    last_instance = None

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.models = _FakeGeminiModels()
        _FakeGenAIClient.last_instance = self


class _FailingGenAIClient:
    def __init__(self, *args, **kwargs) -> None:
        self.models = _FakeGeminiModels(fail=True)


def _google_genai_modules(client_cls):
    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    genai_module.Client = client_cls
    google_module.genai = genai_module
    return {"google": google_module, "google.genai": genai_module}


class LLMClientTests(unittest.TestCase):
    def test_mock_llm_client_returns_dict(self) -> None:
        client = MockLLMClient()
        response = client.generate_json("Return JSON only.", "{}", schema_name="llm_check")
        self.assertIsInstance(response, dict)
        self.assertTrue(response["ok"])

    def test_provider_mock_does_not_require_openai_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = create_llm_client("mock")
        self.assertIsInstance(client, MockLLMClient)

    def test_provider_openai_without_api_key_errors_when_fallback_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"KEIBA_LLM_ENABLE_FALLBACK": "false"},
            clear=True,
        ), patch.object(config_module, "DOTENV_PATH", Path(temp_dir) / ".env"):
            with self.assertRaisesRegex(
                RuntimeError,
                "OPENAI_API_KEY is required when KEIBA_LLM_PROVIDER=openai",
            ):
                create_llm_client("openai")

    def test_provider_gemini_without_api_key_errors_when_fallback_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"KEIBA_LLM_ENABLE_FALLBACK": "false"},
            clear=True,
        ), patch.object(config_module, "DOTENV_PATH", Path(temp_dir) / ".env"):
            with self.assertRaisesRegex(
                RuntimeError,
                "GEMINI_API_KEY is required when KEIBA_LLM_PROVIDER=gemini",
            ):
                create_llm_client("gemini")

    def test_config_can_read_keiba_llm_agent_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / ".env"
            dotenv_path.write_text(
                "\n".join(
                    [
                        "KEIBA_LLM_PROVIDER=openai",
                        "OPENAI_API_KEY=test-from-dotenv",
                        'OPENAI_MODEL="gpt-5.4-mini"',
                        "KEIBA_LLM_ENABLE_FALLBACK=true",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True), patch.object(config_module, "DOTENV_PATH", dotenv_path):
                config = config_module.get_llm_config()
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.openai_api_key, "test-from-dotenv")
        self.assertEqual(config.openai_model, "gpt-5.4-mini")
        self.assertTrue(config.enable_fallback)

    def test_config_can_read_gemini_settings_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / ".env"
            dotenv_path.write_text(
                "\n".join(
                    [
                        "KEIBA_LLM_PROVIDER=gemini",
                        "GEMINI_API_KEY=test-gemini-key",
                        'GEMINI_MODEL="gemini-3.1-flash-lite"',
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True), patch.object(config_module, "DOTENV_PATH", dotenv_path):
                config = config_module.get_llm_config()
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.gemini_api_key, "test-gemini-key")
        self.assertEqual(config.gemini_model, "gemini-3.1-flash-lite")

    def test_json_extraction_handles_fenced_json(self) -> None:
        payload = extract_json_object('```json\n{"ok": true}\n```')
        self.assertEqual(payload, {"ok": True})

    def test_openai_failure_falls_back_to_mock_when_enabled(self) -> None:
        with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=_FailingOpenAI)}):
            client = OpenAILLMClient(
                api_key="test-key",
                model="gpt-5.4-mini",
                fallback_client=MockLLMClient(),
                enable_fallback=True,
            )
            response = client.generate_json("Return JSON only.", "{}", schema_name="llm_check")
        self.assertEqual(response, {"ok": True})

    def test_gemini_client_generates_json(self) -> None:
        with patch.dict("sys.modules", _google_genai_modules(_FakeGenAIClient)):
            client = GeminiLLMClient(
                api_key="test-key",
                model="gemini-3.1-flash-lite",
                fallback_client=MockLLMClient(),
                enable_fallback=True,
            )
            response = client.generate_json("Return JSON only.", "{}", schema_name="llm_check")
        self.assertEqual(response, {"ok": True})
        self.assertEqual(_FakeGenAIClient.last_instance.kwargs["api_key"], "test-key")
        self.assertEqual(
            _FakeGenAIClient.last_instance.models.last_kwargs["model"],
            "gemini-3.1-flash-lite",
        )
        self.assertEqual(
            _FakeGenAIClient.last_instance.models.last_kwargs["config"]["response_mime_type"],
            "application/json",
        )

    def test_gemini_failure_falls_back_to_mock_when_enabled(self) -> None:
        with patch.dict("sys.modules", _google_genai_modules(_FailingGenAIClient)):
            client = GeminiLLMClient(
                api_key="test-key",
                model="gemini-3.1-flash-lite",
                fallback_client=MockLLMClient(),
                enable_fallback=True,
            )
            response = client.generate_json("Return JSON only.", "{}", schema_name="llm_check")
        self.assertEqual(response, {"ok": True})

    def test_llm_check_marks_openai_runtime_fallback_as_warning(self) -> None:
        fake_config = SimpleNamespace(
            provider="openai",
            openai_api_key="test-key",
            openai_model="gpt-5.4-mini",
            gemini_api_key=None,
            gemini_model="gemini-3.1-flash-lite",
            enable_fallback=True,
        )
        with (
            patch("keiba_llm_agent.main.get_llm_config", return_value=fake_config),
            patch("keiba_llm_agent.main.create_llm_client", return_value=_FallbackLikeClient()),
        ):
            result = run_llm_check()
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["active_provider"], "mock")
        self.assertIn("fallback to mock", result["detail"])

    def test_llm_check_uses_gemini_model(self) -> None:
        fake_config = SimpleNamespace(
            provider="gemini",
            openai_api_key=None,
            openai_model="gpt-5.4-mini",
            gemini_api_key="test-key",
            gemini_model="gemini-3.1-flash-lite",
            enable_fallback=True,
        )
        with (
            patch("keiba_llm_agent.main.get_llm_config", return_value=fake_config),
            patch("keiba_llm_agent.main.create_llm_client", return_value=_FallbackLikeClient()),
        ):
            result = run_llm_check()
        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["model"], "gemini-3.1-flash-lite")


if __name__ == "__main__":
    unittest.main()

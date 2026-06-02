from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from keiba_llm_agent import config as config_module
from keiba_llm_agent.llm import MockLLMClient, create_llm_client, extract_json_object
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

    def test_llm_check_marks_openai_runtime_fallback_as_warning(self) -> None:
        fake_config = SimpleNamespace(
            provider="openai",
            openai_api_key="test-key",
            openai_model="gpt-5.4-mini",
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


if __name__ == "__main__":
    unittest.main()

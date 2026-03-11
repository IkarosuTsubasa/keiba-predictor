import concurrent.futures
import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

from local_env import load_local_env
from . import gemini_policy as _gemini
from .gemini_policy import RacePolicyInput, RacePolicyOutput, deterministic_policy

DEFAULT_GEMINI_MODEL = _gemini.DEFAULT_GEMINI_MODEL
DEFAULT_SILICONFLOW_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"
DEFAULT_OPENAI_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_GROK_MODEL = "grok-4-1-fast-reasoning"
POLICY_ENGINE_DEFAULTS = {
    "gemini": DEFAULT_GEMINI_MODEL,
    "siliconflow": DEFAULT_SILICONFLOW_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
    "grok": DEFAULT_GROK_MODEL,
}
POLICY_CACHE_VERSION_MAP = {
    "gemini": _gemini.POLICY_CACHE_VERSION,
    "siliconflow": "siliconflow_policy_v1",
    "openai": "openai_policy_v3",
    "grok": "grok_policy_v1",
}
_MODULE_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _MODULE_DIR.parent
load_local_env(_PIPELINE_DIR, override=True)
_CACHE_DIRS = {
    "gemini": _gemini.DEFAULT_CACHE_DIR,
    "siliconflow": _PIPELINE_DIR / "data" / "policy_cache_siliconflow",
    "openai": _PIPELINE_DIR / "data" / "policy_cache_openai",
    "grok": _PIPELINE_DIR / "data" / "policy_cache_grok",
}
_LAST_CALL_META = {
    "cache_hit": False,
    "llm_latency_ms": 0,
    "fallback_reason": "",
    "error_detail": "",
    "picked_count": 0,
    "buy_style": "",
    "requested_budget_yen": 0,
    "requested_race_budget_yen": 0,
    "reused": False,
    "source_budget_yen": 0,
    "policy_version": "",
    "policy_engine": "none",
    "policy_model": "",
}
_SILICONFLOW_BUCKET = _gemini._TokenBucket(rpm=int(os.environ.get("SILICONFLOW_POLICY_RPM", "10") or "10"))
_OPENAI_BUCKET = _gemini._TokenBucket(rpm=int(os.environ.get("OPENAI_POLICY_RPM", "10") or "10"))
_GROK_BUCKET = _gemini._TokenBucket(rpm=int(os.environ.get("GROK_POLICY_RPM", "10") or "10"))


def normalize_policy_engine(value: str) -> str:
    engine = str(value or "").strip().lower()
    if engine in ("gemini", "siliconflow", "openai", "grok"):
        return engine
    return "none"


def resolve_policy_model(policy_engine: str, model: str = "", gemini_model_compat: str = "") -> str:
    engine = normalize_policy_engine(policy_engine)
    explicit = str(model or "").strip()
    if explicit:
        return explicit
    compat = str(gemini_model_compat or "").strip()
    if engine == "gemini" and compat:
        return compat
    env_model_map = {
        "gemini": ["GEMINI_MODEL", "GEMINI_POLICY_MODEL"],
        "siliconflow": ["SILICONFLOW_POLICY_MODEL"],
        "openai": ["OPENAI_POLICY_MODEL", "OPENAI_MODEL"],
        "grok": ["GROK_POLICY_MODEL", "XAI_MODEL"],
    }
    for key in env_model_map.get(engine, []):
        env_model = str(os.environ.get(key, "") or "").strip()
        if env_model:
            return env_model
    return POLICY_ENGINE_DEFAULTS.get(engine, "")


def _build_cache_key(input_obj: RacePolicyInput, policy_engine: str, model: str) -> str:
    engine = normalize_policy_engine(policy_engine)
    payload = {
        "v": POLICY_CACHE_VERSION_MAP.get(engine, "policy_v1"),
        "prompt_version": _gemini.POLICY_PROMPT_VERSION,
        "policy_engine": engine,
        "race_id": str(input_obj.race_id),
        "scope_key": str(input_obj.scope_key),
        "model": str(model or ""),
        "context_digest": _gemini._input_context_digest(input_obj),
    }
    return hashlib.sha256(_gemini._stable_json_dumps(payload).encode("utf-8")).hexdigest()


def get_policy_cache_key(
    input: RacePolicyInput,
    policy_engine: str = "gemini",
    model: str = "",
) -> str:
    input_obj = _gemini._model_validate(RacePolicyInput, input)
    engine = normalize_policy_engine(policy_engine)
    resolved_model = resolve_policy_model(engine, model)
    return _build_cache_key(input_obj, engine, resolved_model)


def _update_last_meta(meta: Dict[str, Any], output: RacePolicyOutput, policy_engine: str, policy_model: str) -> None:
    global _LAST_CALL_META
    _LAST_CALL_META = {
        "cache_hit": bool(meta.get("cache_hit", False)),
        "llm_latency_ms": int(meta.get("llm_latency_ms", 0) or 0),
        "fallback_reason": str(meta.get("fallback_reason", "") or ""),
        "error_detail": str(meta.get("error_detail", "") or ""),
        "picked_count": int(max(len(output.pick_ids or []), int(output.max_ticket_count or 0))),
        "buy_style": str(output.buy_style or ""),
        "requested_budget_yen": int(meta.get("requested_budget_yen", 0) or 0),
        "requested_race_budget_yen": int(meta.get("requested_race_budget_yen", 0) or 0),
        "reused": bool(meta.get("reused", False)),
        "source_budget_yen": int(meta.get("source_budget_yen", 0) or 0),
        "policy_version": str(meta.get("policy_version", "") or ""),
        "policy_engine": normalize_policy_engine(policy_engine),
        "policy_model": str(policy_model or ""),
    }
    print(
        "[policy_runtime] engine={policy_engine} model={policy_model} cache_hit={cache_hit} "
        "llm_latency_ms={llm_latency_ms} fallback_reason={fallback_reason} error_detail={error_detail} "
        "picked_count={picked_count} "
        "buy_style={buy_style} requested_budget_yen={requested_budget_yen} "
        "requested_race_budget_yen={requested_race_budget_yen} reused={reused} "
        "source_budget_yen={source_budget_yen} policy_version={policy_version}".format(**_LAST_CALL_META)
    )


def get_last_call_meta() -> Dict[str, Any]:
    return dict(_LAST_CALL_META)


def _extract_siliconflow_text(response_payload: Dict[str, Any]) -> str:
    if not isinstance(response_payload, dict):
        return ""
    choices = list(response_payload.get("choices", []) or [])
    for item in choices:
        if not isinstance(item, dict):
            continue
        message = dict(item.get("message", {}) or {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _extract_openai_text(response_payload: Dict[str, Any]) -> str:
    if not isinstance(response_payload, dict):
        return ""
    direct = response_payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    output_items = list(response_payload.get("output", []) or [])
    chunks = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        content_items = list(item.get("content", []) or [])
        for content in content_items:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks).strip()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in ("1", "true", "yes", "on")


def _normalize_base_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "https://api.openai.com/v1"
    return text.rstrip("/")


def _get_api_style(prefix: str) -> str:
    style = str(os.environ.get(f"{prefix}_API_STYLE", "responses") or "responses").strip().lower()
    if style in ("responses", "chat_completions"):
        return style
    return "responses"


def _normalize_openai_reasoning_effort(value: str) -> str:
    effort = str(value or "").strip().lower()
    if effort in ("low", "medium", "high"):
        return effort
    return "low"


def _normalize_grok_reasoning_effort(value: str) -> str:
    effort = str(value or "").strip().lower()
    if effort in ("low", "high"):
        return effort
    return "low"


def _build_responses_endpoint(prefix: str, default_base_url: str) -> str:
    explicit = str(os.environ.get(f"{prefix}_RESPONSES_URL", "") or "").strip()
    if explicit:
        return explicit
    base_url = _normalize_base_url(
        str(
            os.environ.get(f"{prefix}_BASE_URL", "")
            or os.environ.get(f"{prefix}_API_BASE", "")
            or os.environ.get(f"{prefix}_BASE", "")
            or default_base_url
        )
    )
    style = _get_api_style(prefix)
    if re.search(r"/(responses|chat/completions)$", base_url, flags=re.IGNORECASE):
        return base_url
    if style == "chat_completions":
        return urllib.parse.urljoin(f"{base_url}/", "chat/completions")
    return urllib.parse.urljoin(f"{base_url}/", "responses")


def _build_openai_endpoint() -> str:
    return _build_responses_endpoint("OPENAI", "https://api.openai.com/v1")


def _build_grok_endpoint() -> str:
    return _build_responses_endpoint("XAI", "https://api.x.ai/v1")


def _grok_supports_reasoning_effort(model_name: str) -> bool:
    text = str(model_name or "").strip().lower()
    return text.startswith("grok-3-mini")


def _extract_http_error_detail(exc: urllib.error.HTTPError) -> str:
    detail = ""
    try:
        raw = exc.read()
        if raw:
            detail = raw.decode("utf-8", errors="replace").strip()
    except Exception:
        detail = ""
    if not detail:
        return ""
    try:
        payload = json.loads(detail)
    except Exception:
        return detail[:500]
    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            message = str(error_obj.get("message", "") or "").strip()
            code = str(error_obj.get("code", "") or "").strip()
            param = str(error_obj.get("param", "") or "").strip()
            parts = [part for part in (message, f"code={code}" if code else "", f"param={param}" if param else "") if part]
            if parts:
                return " | ".join(parts)[:500]
    return detail[:500]


def _call_siliconflow_once(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    body = {
        "model": str(model or DEFAULT_SILICONFLOW_MODEL),
        "messages": [
            {"role": "system", "content": "你是一个严格输出JSON的赛马策略助手。"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": int(os.environ.get("SILICONFLOW_MAX_TOKENS", "2048") or "2048"),
        "enable_thinking": False,
    }
    thinking_raw = str(os.environ.get("SILICONFLOW_ENABLE_THINKING", "") or "").strip().lower()
    if thinking_raw in ("0", "1", "true", "false", "yes", "no", "on", "off"):
        body["enable_thinking"] = thinking_raw in ("1", "true", "yes", "on")
    thinking_budget = str(os.environ.get("SILICONFLOW_THINKING_BUDGET", "") or "").strip()
    if thinking_budget:
        try:
            body["thinking_budget"] = int(thinking_budget)
        except ValueError:
            pass

    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.siliconflow.cn/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(1, int(timeout_s or 1))) as response:
        raw = response.read().decode("utf-8")
    text = _extract_siliconflow_text(json.loads(raw))
    if not text:
        raise ValueError("empty_response_content")
    return text


def _call_openai_once(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    style = _get_api_style("OPENAI")
    model_name = str(model or DEFAULT_OPENAI_MODEL)
    body: Dict[str, Any]
    if style == "chat_completions":
        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "你是一个严格输出JSON的赛马策略助手。"},
                {"role": "user", "content": prompt},
            ],
        }
        max_tokens_raw = str(os.environ.get("OPENAI_POLICY_MAX_OUTPUT_TOKENS", "") or "").strip()
        if max_tokens_raw:
            try:
                body["max_tokens"] = int(max_tokens_raw)
            except ValueError:
                pass
        temperature_raw = str(os.environ.get("OPENAI_POLICY_TEMPERATURE", "0.2") or "0.2").strip()
        if temperature_raw:
            try:
                body["temperature"] = float(temperature_raw)
            except ValueError:
                pass
    else:
        body = {
            "model": model_name,
            "input": prompt,
        }
        if _bool_env("OPENAI_POLICY_INCLUDE_REASONING", False):
            body["reasoning"] = {
                "effort": _normalize_openai_reasoning_effort(
                    str(os.environ.get("OPENAI_POLICY_REASONING_EFFORT", "medium") or "medium")
                ),
            }
        if _bool_env("OPENAI_POLICY_INCLUDE_TEXT", True):
            body["text"] = {
                "verbosity": str(os.environ.get("OPENAI_POLICY_VERBOSITY", "medium") or "medium"),
            }
        max_output_tokens_raw = str(os.environ.get("OPENAI_POLICY_MAX_OUTPUT_TOKENS", "") or "").strip()
        if max_output_tokens_raw:
            try:
                body["max_output_tokens"] = int(max_output_tokens_raw)
            except ValueError:
                pass
    request = urllib.request.Request(
        _build_openai_endpoint(),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(1, int(timeout_s or 1))) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if style == "chat_completions":
        text = _extract_siliconflow_text(payload)
    else:
        text = _extract_openai_text(payload)
    if not text:
        raise ValueError("empty_response_content")
    return text


def _call_grok_once(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    from xai_sdk import Client
    from xai_sdk.chat import system, user

    model_name = str(model or DEFAULT_GROK_MODEL)
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "temperature": 0.2,
        "response_format": "json_object",
    }
    temperature_raw = str(os.environ.get("GROK_POLICY_TEMPERATURE", "0.2") or "0.2").strip()
    if temperature_raw:
        try:
            kwargs["temperature"] = float(temperature_raw)
        except ValueError:
            pass
    max_tokens_raw = str(os.environ.get("GROK_POLICY_MAX_OUTPUT_TOKENS", "") or "").strip()
    if max_tokens_raw:
        try:
            kwargs["max_tokens"] = int(max_tokens_raw)
        except ValueError:
            pass
    if _bool_env("GROK_POLICY_INCLUDE_REASONING", False) and _grok_supports_reasoning_effort(model_name):
        kwargs["reasoning_effort"] = _normalize_grok_reasoning_effort(
            str(os.environ.get("GROK_POLICY_REASONING_EFFORT", "low") or "low")
        )

    client = Client(api_key=api_key, timeout=float(max(1, int(timeout_s or 1))))
    chat = client.chat.create(**kwargs)
    chat.append(system("你是一个严格输出JSON的赛马策略助手。"))
    chat.append(user(prompt))
    response = chat.sample()
    text = str(getattr(response, "content", "") or "").strip()
    if not text:
        raise ValueError("empty_response_content")
    return text


def call_siliconflow_policy(
    input: RacePolicyInput,
    model: str = DEFAULT_SILICONFLOW_MODEL,
    timeout_s: int = 75,
    cache_enable: bool = True,
) -> RacePolicyOutput:
    input_obj = _gemini._model_validate(RacePolicyInput, input)
    resolved_model = resolve_policy_model("siliconflow", model)
    request_meta = _gemini._build_request_meta(input_obj)
    request_meta["policy_version"] = POLICY_CACHE_VERSION_MAP["siliconflow"]
    cache_dir = _CACHE_DIRS["siliconflow"]
    _gemini._ensure_cache_dir(cache_dir)
    cache_path = cache_dir / f"{_build_cache_key(input_obj, 'siliconflow', resolved_model)}.json"

    if bool(cache_enable):
        cached = _gemini._read_cache(cache_path)
        if cached is not None:
            _update_last_meta(
                {
                    **request_meta,
                    "cache_hit": True,
                    "llm_latency_ms": 0,
                    "fallback_reason": "cache",
                },
                cached,
                "siliconflow",
                resolved_model,
            )
            return cached

    fallback_reason = ""
    error_detail = ""
    llm_latency_ms = 0
    output: Optional[RacePolicyOutput] = None
    mock_enabled = str(os.environ.get("SILICONFLOW_POLICY_MOCK", "") or "").strip() == "1"
    if mock_enabled:
        fallback_reason = "mock_mode"
        output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
    else:
        api_key = str(os.environ.get("SILICONFLOW_API_KEY", "") or "").strip()
        if not api_key:
            fallback_reason = "missing_api_key"
            output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
        elif not _SILICONFLOW_BUCKET.consume(1.0):
            fallback_reason = "rate_limited_local"
            output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
        else:
            prompt = _gemini._make_prompt(input_obj)
            retry_count_raw = str(os.environ.get("SILICONFLOW_POLICY_RETRIES", "2") or "2").strip()
            try:
                max_attempts = max(1, int(retry_count_raw))
            except ValueError:
                max_attempts = 2
            for attempt_idx in range(max_attempts):
                start = time.perf_counter()
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_call_siliconflow_once, prompt, resolved_model, api_key, timeout_s)
                        raw_text = future.result(timeout=max(1, int(timeout_s or 1)))
                    llm_latency_ms = int((time.perf_counter() - start) * 1000)
                    payload = _gemini._parse_json_payload(raw_text)
                    parsed = _gemini._model_validate(RacePolicyOutput, payload)
                    output = _gemini._sanitize_output(parsed, input_obj)
                    fallback_reason = ""
                    error_detail = ""
                    break
                except (concurrent.futures.TimeoutError, TimeoutError, socket.timeout):
                    fallback_reason = "timeout"
                    error_detail = ""
                except json.JSONDecodeError:
                    fallback_reason = "json_parse_failed"
                    error_detail = ""
                except ValueError as exc:
                    text = str(exc).lower()
                    error_detail = str(exc).strip()
                    if "unknown id" in text:
                        fallback_reason = "unknown_pick_id"
                    elif "http error 401" in text or "http error 403" in text:
                        fallback_reason = "auth_error"
                    elif "empty_response_content" in text:
                        fallback_reason = "empty_response_content"
                    else:
                        fallback_reason = "value_error"
                except urllib.error.HTTPError as exc:
                    code = int(getattr(exc, "code", 0) or 0)
                    error_detail = _extract_http_error_detail(exc)
                    if code in (401, 403):
                        fallback_reason = "auth_error"
                    elif code == 429:
                        fallback_reason = "quota_or_429"
                    else:
                        fallback_reason = f"http_{code or 'error'}"
                except urllib.error.URLError:
                    fallback_reason = "network_error"
                    error_detail = ""
                except Exception as exc:
                    text = str(exc).lower()
                    error_detail = str(exc).strip()
                    if "429" in text or "quota" in text or "rate" in text:
                        fallback_reason = "quota_or_429"
                    elif "auth" in text or "permission" in text or "api key" in text:
                        fallback_reason = "auth_error"
                    else:
                        fallback_reason = "network_or_sdk_error"
                if output is not None or attempt_idx + 1 >= max_attempts:
                    break

            if output is None:
                output = deterministic_policy(input_obj, fallback_reason=fallback_reason)

    final_output = _gemini._sanitize_output(output, input_obj)
    meta = {
        **request_meta,
        "cache_hit": False,
        "llm_latency_ms": int(llm_latency_ms),
        "fallback_reason": str(fallback_reason or ""),
        "error_detail": str(error_detail or ""),
    }
    if bool(cache_enable):
        _gemini._write_cache(cache_path, final_output, meta)
    _update_last_meta(meta, final_output, "siliconflow", resolved_model)
    return final_output


def call_openai_policy(
    input: RacePolicyInput,
    model: str = DEFAULT_OPENAI_MODEL,
    timeout_s: int = 90,
    cache_enable: bool = True,
) -> RacePolicyOutput:
    input_obj = _gemini._model_validate(RacePolicyInput, input)
    resolved_model = resolve_policy_model("openai", model)
    request_meta = _gemini._build_request_meta(input_obj)
    request_meta["policy_version"] = POLICY_CACHE_VERSION_MAP["openai"]
    cache_dir = _CACHE_DIRS["openai"]
    _gemini._ensure_cache_dir(cache_dir)
    cache_path = cache_dir / f"{_build_cache_key(input_obj, 'openai', resolved_model)}.json"

    if bool(cache_enable):
        cached = _gemini._read_cache(cache_path)
        if cached is not None:
            _update_last_meta(
                {
                    **request_meta,
                    "cache_hit": True,
                    "llm_latency_ms": 0,
                    "fallback_reason": "cache",
                },
                cached,
                "openai",
                resolved_model,
            )
            return cached

    fallback_reason = ""
    llm_latency_ms = 0
    output: Optional[RacePolicyOutput] = None
    mock_enabled = str(os.environ.get("OPENAI_POLICY_MOCK", "") or "").strip() == "1"
    if mock_enabled:
        fallback_reason = "mock_mode"
        output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
    else:
        api_key = str(os.environ.get("OPENAI_API_KEY", "") or "").strip()
        if not api_key:
            fallback_reason = "missing_api_key"
            output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
        elif not _OPENAI_BUCKET.consume(1.0):
            fallback_reason = "rate_limited_local"
            output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
        else:
            prompt = _gemini._make_prompt(input_obj)
            retry_count_raw = str(os.environ.get("OPENAI_POLICY_RETRIES", "3") or "3").strip()
            try:
                max_attempts = max(1, int(retry_count_raw))
            except ValueError:
                max_attempts = 3
            for attempt_idx in range(max_attempts):
                start = time.perf_counter()
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_call_openai_once, prompt, resolved_model, api_key, timeout_s)
                        raw_text = future.result(timeout=max(1, int(timeout_s or 1)))
                    llm_latency_ms = int((time.perf_counter() - start) * 1000)
                    payload = _gemini._parse_json_payload(raw_text)
                    parsed = _gemini._model_validate(RacePolicyOutput, payload)
                    output = _gemini._sanitize_output(parsed, input_obj)
                    fallback_reason = ""
                    break
                except (concurrent.futures.TimeoutError, TimeoutError, socket.timeout):
                    fallback_reason = "timeout"
                except json.JSONDecodeError:
                    fallback_reason = "json_parse_failed"
                except ValueError as exc:
                    text = str(exc).lower()
                    if "unknown id" in text:
                        fallback_reason = "unknown_pick_id"
                    elif "http error 401" in text or "http error 403" in text:
                        fallback_reason = "auth_error"
                    elif "empty_response_content" in text:
                        fallback_reason = "empty_response_content"
                    else:
                        fallback_reason = "value_error"
                except urllib.error.HTTPError as exc:
                    code = int(getattr(exc, "code", 0) or 0)
                    if code in (401, 403):
                        fallback_reason = "auth_error"
                    elif code == 429:
                        fallback_reason = "quota_or_429"
                    else:
                        fallback_reason = f"http_{code or 'error'}"
                except urllib.error.URLError:
                    fallback_reason = "network_error"
                except Exception as exc:
                    text = str(exc).lower()
                    if "429" in text or "quota" in text or "rate" in text:
                        fallback_reason = "quota_or_429"
                    elif "auth" in text or "permission" in text or "api key" in text:
                        fallback_reason = "auth_error"
                    else:
                        fallback_reason = "network_or_sdk_error"
                if output is not None or attempt_idx + 1 >= max_attempts:
                    break

            if output is None:
                output = deterministic_policy(input_obj, fallback_reason=fallback_reason)

    final_output = _gemini._sanitize_output(output, input_obj)
    meta = {
        **request_meta,
        "cache_hit": False,
        "llm_latency_ms": int(llm_latency_ms),
        "fallback_reason": str(fallback_reason or ""),
    }
    if bool(cache_enable):
        _gemini._write_cache(cache_path, final_output, meta)
    _update_last_meta(meta, final_output, "openai", resolved_model)
    return final_output


def call_grok_policy(
    input: RacePolicyInput,
    model: str = DEFAULT_GROK_MODEL,
    timeout_s: int = 120,
    cache_enable: bool = True,
) -> RacePolicyOutput:
    input_obj = _gemini._model_validate(RacePolicyInput, input)
    resolved_model = resolve_policy_model("grok", model)
    request_meta = _gemini._build_request_meta(input_obj)
    request_meta["policy_version"] = POLICY_CACHE_VERSION_MAP["grok"]
    cache_dir = _CACHE_DIRS["grok"]
    _gemini._ensure_cache_dir(cache_dir)
    cache_path = cache_dir / f"{_build_cache_key(input_obj, 'grok', resolved_model)}.json"

    if bool(cache_enable):
        cached = _gemini._read_cache(cache_path)
        if cached is not None:
            _update_last_meta(
                {
                    **request_meta,
                    "cache_hit": True,
                    "llm_latency_ms": 0,
                    "fallback_reason": "cache",
                },
                cached,
                "grok",
                resolved_model,
            )
            return cached

    fallback_reason = ""
    error_detail = ""
    llm_latency_ms = 0
    output: Optional[RacePolicyOutput] = None
    mock_enabled = str(os.environ.get("GROK_POLICY_MOCK", "") or "").strip() == "1"
    if mock_enabled:
        fallback_reason = "mock_mode"
        output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
    else:
        api_key = str(os.environ.get("XAI_API_KEY", "") or "").strip()
        if not api_key:
            fallback_reason = "missing_api_key"
            output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
        elif not _GROK_BUCKET.consume(1.0):
            fallback_reason = "rate_limited_local"
            output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
        else:
            prompt = _gemini._make_prompt(input_obj)
            retry_count_raw = str(os.environ.get("GROK_POLICY_RETRIES", "3") or "3").strip()
            try:
                max_attempts = max(1, int(retry_count_raw))
            except ValueError:
                max_attempts = 3
            for attempt_idx in range(max_attempts):
                start = time.perf_counter()
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_call_grok_once, prompt, resolved_model, api_key, timeout_s)
                        raw_text = future.result(timeout=max(1, int(timeout_s or 1)))
                    llm_latency_ms = int((time.perf_counter() - start) * 1000)
                    payload = _gemini._parse_json_payload(raw_text)
                    parsed = _gemini._model_validate(RacePolicyOutput, payload)
                    output = _gemini._sanitize_output(parsed, input_obj)
                    fallback_reason = ""
                    error_detail = ""
                    break
                except (concurrent.futures.TimeoutError, TimeoutError, socket.timeout):
                    fallback_reason = "timeout"
                    error_detail = ""
                except json.JSONDecodeError:
                    fallback_reason = "json_parse_failed"
                    error_detail = ""
                except ValueError as exc:
                    text = str(exc).lower()
                    error_detail = str(exc).strip()
                    if "unknown id" in text:
                        fallback_reason = "unknown_pick_id"
                    elif "http error 401" in text or "http error 403" in text:
                        fallback_reason = "auth_error"
                    elif "empty_response_content" in text:
                        fallback_reason = "empty_response_content"
                    else:
                        fallback_reason = "value_error"
                except urllib.error.HTTPError as exc:
                    code = int(getattr(exc, "code", 0) or 0)
                    error_detail = _extract_http_error_detail(exc)
                    if code in (401, 403):
                        fallback_reason = "auth_error"
                    elif code == 429:
                        fallback_reason = "quota_or_429"
                    else:
                        fallback_reason = f"http_{code or 'error'}"
                except urllib.error.URLError:
                    fallback_reason = "network_error"
                    error_detail = ""
                except Exception as exc:
                    text = str(exc).lower()
                    error_detail = str(exc).strip()
                    if "429" in text or "quota" in text or "rate" in text:
                        fallback_reason = "quota_or_429"
                    elif "auth" in text or "permission" in text or "api key" in text:
                        fallback_reason = "auth_error"
                    else:
                        fallback_reason = "network_or_sdk_error"
                if output is not None or attempt_idx + 1 >= max_attempts:
                    break

            if output is None:
                output = deterministic_policy(input_obj, fallback_reason=fallback_reason)

    final_output = _gemini._sanitize_output(output, input_obj)
    meta = {
        **request_meta,
        "cache_hit": False,
        "llm_latency_ms": int(llm_latency_ms),
        "fallback_reason": str(fallback_reason or ""),
        "error_detail": str(error_detail or ""),
    }
    if bool(cache_enable):
        _gemini._write_cache(cache_path, final_output, meta)
    _update_last_meta(meta, final_output, "grok", resolved_model)
    return final_output


def call_policy(
    input: RacePolicyInput,
    policy_engine: str = "gemini",
    model: str = "",
    timeout_s: int = 20,
    cache_enable: bool = True,
) -> RacePolicyOutput:
    input_obj = _gemini._model_validate(RacePolicyInput, input)
    engine = normalize_policy_engine(policy_engine)
    resolved_model = resolve_policy_model(engine, model)
    if engine == "gemini":
        output = _gemini.call_gemini_policy(
            input=input_obj,
            model=resolved_model or DEFAULT_GEMINI_MODEL,
            timeout_s=timeout_s,
            cache_enable=cache_enable,
        )
        meta = _gemini.get_last_call_meta()
        _update_last_meta(meta, output, "gemini", resolved_model or DEFAULT_GEMINI_MODEL)
        return output
    if engine == "siliconflow":
        return call_siliconflow_policy(
            input=input_obj,
            model=resolved_model or DEFAULT_SILICONFLOW_MODEL,
            timeout_s=timeout_s,
            cache_enable=cache_enable,
        )
    if engine == "openai":
        return call_openai_policy(
            input=input_obj,
            model=resolved_model or DEFAULT_OPENAI_MODEL,
            timeout_s=timeout_s,
            cache_enable=cache_enable,
        )
    if engine == "grok":
        return call_grok_policy(
            input=input_obj,
            model=resolved_model or DEFAULT_GROK_MODEL,
            timeout_s=timeout_s,
            cache_enable=cache_enable,
        )

    output = deterministic_policy(input_obj, fallback_reason="policy_engine_none")
    meta = {
        **_gemini._build_request_meta(input_obj),
        "cache_hit": False,
        "llm_latency_ms": 0,
        "fallback_reason": "policy_engine_none",
        "policy_version": "deterministic_only",
    }
    _update_last_meta(meta, output, "none", "")
    return output


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GROK_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_SILICONFLOW_MODEL",
    "POLICY_ENGINE_DEFAULTS",
    "call_grok_policy",
    "RacePolicyInput",
    "RacePolicyOutput",
    "call_openai_policy",
    "call_policy",
    "call_siliconflow_policy",
    "deterministic_policy",
    "get_last_call_meta",
    "get_policy_cache_key",
    "normalize_policy_engine",
    "resolve_policy_model",
]

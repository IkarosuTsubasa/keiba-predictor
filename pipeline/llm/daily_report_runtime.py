import json
import os
import socket
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List

from . import gemini_policy as _gemini
from . import policy_runtime as _policy_runtime


REPORT_ENGINE_LABELS = {
    "openai": "ChatGPT",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
}


def normalize_report_engine(value: str = "") -> str:
    engine = _policy_runtime.normalize_policy_engine(value)
    if engine in REPORT_ENGINE_LABELS:
        return engine
    return "openai"


def resolve_report_model(engine: str, model: str = "") -> str:
    return _policy_runtime.resolve_policy_model(normalize_report_engine(engine), model)


def _report_system_prompt() -> str:
    return (
        "あなたは競馬分析サイトの編集者です。"
        "入力された当日データをもとに、日本語で読みやすい日報記事を作成してください。"
        "煽らず、数値に基づいて、落ち着いたトーンで要約してください。"
        "出力は必ず JSON オブジェクトのみを返してください。"
    )


def _report_user_prompt(source_payload: Dict[str, Any]) -> str:
    schema = {
        "title": "記事タイトル",
        "lead": "導入文",
        "summary": "全体要約",
        "tags": ["短いタグ"],
        "markdown": "# 見出し\n本文\n\n## 小見出し\n- 箇条書き",
    }
    return (
        f"{_report_system_prompt()}\n\n"
        "次のデータから公開ページ用の日報記事を作成してください。\n"
        "条件:\n"
        "- 文章は日本語\n"
        "- Markdown 本文は 3 〜 6 セクション程度\n"
        "- h1 は使わず、本文は h2 / h3 / 箇条書きを中心に構成\n"
        "- 数字がある部分は可能な限り反映\n"
        "- 不明なことは断定しない\n"
        "- 出力は JSON のみ\n"
        f"- 出力形式: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"入力データ:\n{json.dumps(source_payload, ensure_ascii=False)}"
    )


def _coerce_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _coerce_list_text(values: Any, limit: int = 8) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in list(values or []):
        text = _coerce_text(item)
        if (not text) or (text in seen):
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


def _markdown_from_sections(sections: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for item in list(sections or []):
        if not isinstance(item, dict):
            continue
        heading = _coerce_text(item.get("heading"))
        if heading:
            blocks.append(f"## {heading}")
        for paragraph in _coerce_list_text(item.get("paragraphs"), limit=6):
            blocks.append(paragraph)
        for bullet in _coerce_list_text(item.get("bullets"), limit=8):
            blocks.append(f"- {bullet}")
        if blocks and blocks[-1] != "":
            blocks.append("")
    return "\n".join(blocks).strip()


def _build_fallback_document(source_payload: Dict[str, Any], *, fallback_reason: str = "") -> Dict[str, Any]:
    target_date_label = _coerce_text(source_payload.get("target_date_label"), "対象日")
    totals = dict(source_payload.get("totals") or {})
    best_ticket = dict(source_payload.get("best_ticket") or {})
    llm_cards = list(source_payload.get("llm_cards") or [])
    predictor_leader = dict(source_payload.get("predictor_leader") or {})
    top_engine = dict(llm_cards[0] or {}) if llm_cards else {}

    title = f"{target_date_label}の私の日報"
    lead = f"{target_date_label}の公開レースをもとに、AIモデルと定量モデルの当日結果をまとめました。"
    summary = (
        f"公開 {int(totals.get('race_count', 0) or 0)} レースのうち、"
        f"結果確定は {int(totals.get('settled_count', 0) or 0)} レースでした。"
    )

    sections: List[Dict[str, Any]] = [
        {
            "heading": "全体総括",
            "paragraphs": [
                summary,
                (
                    f"回収面では {_coerce_text(top_engine.get('label'), '集計なし')} が目立ち、"
                    f"ROI は {_coerce_text(top_engine.get('roi_text'), '-')}"
                    f"、損益は {_coerce_text(top_engine.get('profit_yen_text'), '-')} でした。"
                    if top_engine
                    else "当日のモデル別回収データはまだ十分にそろっていません。"
                ),
            ],
            "bullets": [],
        },
        {
            "heading": "AIモデルの振り返り",
            "paragraphs": [
                "モデルごとの回収率、的中レース数、損益を見比べることで、当日の買い目構成の差が見えます。",
            ],
            "bullets": [
                (
                    f"{_coerce_text(item.get('label'), '-')}: "
                    f"ROI {_coerce_text(item.get('roi_text'), '-')}, "
                    f"的中 {_coerce_text(item.get('hit_races_text'), '-')}, "
                    f"損益 {_coerce_text(item.get('profit_yen_text'), '-')}"
                )
                for item in llm_cards[:4]
            ],
        },
        {
            "heading": "定量モデルの見どころ",
            "paragraphs": [
                (
                    f"定量モデルでは {_coerce_text(predictor_leader.get('label'), '集計なし')} が"
                    f" 上位5頭カバー率 {_coerce_text(predictor_leader.get('top5_to_top3_hit_rate_text'), '-')}"
                    " で先頭でした。"
                    if predictor_leader
                    else "当日の定量モデル比較データはまだ十分にありません。"
                ),
            ],
            "bullets": [],
        },
    ]
    if best_ticket:
        sections.append(
            {
                "heading": "最大ヒット",
                "paragraphs": [
                    (
                        f"{_coerce_text(best_ticket.get('race_title'), '-')} で "
                        f"{_coerce_text(best_ticket.get('bet_type_label'), '的中')} が入り、"
                        f"回収倍率は {_coerce_text(best_ticket.get('multiplier_text'), '-')} でした。"
                    )
                ],
                "bullets": [],
            }
        )
    if fallback_reason:
        sections.append(
            {
                "heading": "生成メモ",
                "paragraphs": [
                    f"今回は LLM 応答を取得できなかったため、ローカル要約で日報を生成しました。理由: {fallback_reason}",
                ],
                "bullets": [],
            }
        )

    return {
        "title": title,
        "lead": lead,
        "summary": summary,
        "tags": _coerce_list_text(["日報", target_date_label, _coerce_text(top_engine.get("label"))], limit=5),
        "markdown": _markdown_from_sections(sections),
        "sections": sections,
    }


def _sanitize_document(payload: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    fallback_sections = list(fallback.get("sections") or [])
    fallback_markdown = _coerce_text(fallback.get("markdown"))
    sections = list(payload.get("sections") or [])
    markdown = _coerce_text(payload.get("markdown"))
    if not markdown and sections:
        markdown = _markdown_from_sections(sections)
    if not markdown:
        markdown = fallback_markdown
    return {
        "title": _coerce_text(payload.get("title"), fallback.get("title", "私の日報")),
        "lead": _coerce_text(payload.get("lead"), fallback.get("lead", "")),
        "summary": _coerce_text(payload.get("summary"), fallback.get("summary", "")),
        "tags": _coerce_list_text(payload.get("tags"), limit=8) or list(fallback.get("tags") or []),
        "markdown": markdown,
        "sections": sections or fallback_sections,
    }


def _openai_report_call(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    style = _policy_runtime._get_api_style("OPENAI")
    model_name = str(model or _policy_runtime.DEFAULT_OPENAI_MODEL)
    if style == "chat_completions":
        body: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": _report_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }
    else:
        body = {
            "model": model_name,
            "input": prompt,
            "text": {"verbosity": "medium"},
        }
    request = urllib.request.Request(
        _policy_runtime._build_openai_endpoint(),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(1, int(timeout_s or 1))) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if style == "chat_completions":
        return _policy_runtime._extract_chat_completions_text(payload)
    return _policy_runtime._extract_openai_text(payload)


def _deepseek_report_call(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    body = {
        "model": str(model or _policy_runtime.DEFAULT_DEEPSEEK_MODEL),
        "messages": [
            {"role": "system", "content": _report_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
    }
    request = urllib.request.Request(
        _policy_runtime._build_deepseek_endpoint(),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(1, int(timeout_s or 1))) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _policy_runtime._extract_chat_completions_text(payload)


def _grok_report_call(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    from xai_sdk import Client
    from xai_sdk.chat import system, user

    client = Client(api_key=api_key, timeout=float(max(1, int(timeout_s or 1))))
    chat = client.chat.create(
        model=str(model or _policy_runtime.DEFAULT_GROK_MODEL),
        temperature=0.3,
        response_format="json_object",
    )
    chat.append(system(_report_system_prompt()))
    chat.append(user(prompt))
    response = chat.sample()
    return str(getattr(response, "content", "") or "").strip()


def _gemini_report_call(prompt: str, model: str, api_key: str, timeout_s: int) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=str(model or _policy_runtime.DEFAULT_GEMINI_MODEL),
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.3,
            "max_output_tokens": 4096,
        },
    )
    return _gemini._extract_response_text(response)


def generate_daily_report_document(
    source_payload: Dict[str, Any],
    *,
    policy_engine: str = "openai",
    model: str = "",
    timeout_s: int = 90,
) -> Dict[str, Any]:
    engine = normalize_report_engine(policy_engine)
    resolved_model = resolve_report_model(engine, model)
    fallback = _build_fallback_document(source_payload)
    prompt = _report_user_prompt(source_payload)
    raw_text = ""
    fallback_reason = ""
    llm_latency_ms = 0
    started_at = datetime.now()

    if str(os.environ.get("DAILY_REPORT_MOCK", "") or "").strip() == "1":
        fallback_reason = "mock_mode"
    else:
        api_key = ""
        call_fn = None
        if engine == "openai":
            api_key = str(os.environ.get("OPENAI_API_KEY", "") or "").strip()
            call_fn = _openai_report_call
        elif engine == "deepseek":
            api_key = str(os.environ.get("DEEPSEEK_API_KEY", "") or "").strip()
            call_fn = _deepseek_report_call
        elif engine == "grok":
            api_key = str(os.environ.get("XAI_API_KEY", "") or "").strip()
            call_fn = _grok_report_call
        elif engine == "gemini":
            api_key = str(os.environ.get("GEMINI_API_KEY", "") or "").strip()
            call_fn = _gemini_report_call
        if not api_key:
            fallback_reason = "missing_api_key"
        elif call_fn is None:
            fallback_reason = "unsupported_engine"
        else:
            try:
                call_started = datetime.now()
                raw_text = str(call_fn(prompt, resolved_model, api_key, timeout_s) or "").strip()
                llm_latency_ms = int((datetime.now() - call_started).total_seconds() * 1000)
                payload = _gemini._parse_json_payload(raw_text)
                document = _sanitize_document(payload, fallback)
                return {
                    "document": document,
                    "engine": engine,
                    "engine_label": REPORT_ENGINE_LABELS.get(engine, engine),
                    "model": resolved_model,
                    "fallback_reason": "",
                    "llm_latency_ms": llm_latency_ms,
                    "raw_text": raw_text,
                    "generated_at": started_at.isoformat(timespec="seconds"),
                    "mode": "llm",
                }
            except (json.JSONDecodeError, ValueError):
                fallback_reason = "json_parse_failed"
            except urllib.error.HTTPError as exc:
                fallback_reason = f"http_{int(getattr(exc, 'code', 0) or 0)}"
            except urllib.error.URLError:
                fallback_reason = "network_error"
            except socket.timeout:
                fallback_reason = "timeout"
            except Exception:
                fallback_reason = "runtime_error"

    return {
        "document": fallback,
        "engine": engine,
        "engine_label": REPORT_ENGINE_LABELS.get(engine, engine),
        "model": resolved_model,
        "fallback_reason": fallback_reason,
        "llm_latency_ms": llm_latency_ms,
        "raw_text": raw_text,
        "generated_at": started_at.isoformat(timespec="seconds"),
        "mode": "fallback",
    }


__all__ = [
    "REPORT_ENGINE_LABELS",
    "generate_daily_report_document",
    "normalize_report_engine",
    "resolve_report_model",
]

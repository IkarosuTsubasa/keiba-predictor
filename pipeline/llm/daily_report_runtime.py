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
    return "gemini"


def resolve_report_model(engine: str, model: str = "") -> str:
    return _policy_runtime.resolve_policy_model(normalize_report_engine(engine), model)


def _report_system_prompt() -> str:
    return (
        "あなたは競馬分析サイトの編集者です。"
        "公開された当日データを受け取り、数字の意味を読み解いた日報記事を書いてください。"
        "文章は人が書いたコラムのように自然で、観察、比較、留保があることを重視します。"
        "単なるデータの再掲や、項目ごとの機械的な列挙は避けてください。"
        "不明なことは断定せず、サンプルが少ない場合は慎重な表現を使ってください。"
        "出力は必ず JSON オブジェクトのみを返してください。"
    )


def _report_user_prompt(source_payload: Dict[str, Any]) -> str:
    schema = {
        "title": "記事タイトル",
        "lead": "導入文",
        "summary": "全体要約",
        "tags": ["短いタグ"],
        "markdown": "## 全体総括\n本文\n\n## 観察\n本文\n- 補足",
    }
    return (
        f"{_report_system_prompt()}\n\n"
        "次のデータから公開ページ用の日報記事を作成してください。\n\n"
        "条件:\n"
        "- 文章は日本語\n"
        "- Markdown 本文は 4 〜 5 セクション程度\n"
        "- h1 は使わず、本文は h2 / h3 を中心に構成する\n"
        "- 先に結論や観察を書き、その後に数字を根拠として添える\n"
        "- 重要な結論、モデル名、主要指標だけを **太字** で強調する\n"
        "- 全モデルの全指標を漏れなく並べる必要はない。差が出た箇所だけを選んで述べる\n"
        "- 今日の結果だけで終わらせず、全期間傾向やサンプル数とのズレにも触れる\n"
        "- 『比較します』『概観します』『以下の通りです』のような説明的な前置きは避ける\n"
        "- 文章どうしを自然につなぎ、短い分析記事として読める流れにする\n"
        "- 箇条書きは補助的に使い、1 セクション最大 4 項目までに抑える\n"
        "- 不明なことは断定しない\n"
        "- 出力は JSON のみ\n"
        f"- 出力形式: {json.dumps(schema, ensure_ascii=False)}\n\n"
        "本文で必ず触れる観点:\n"
        "- 当日の総括\n"
        "- Agent AI予測の本命、上位印、見送り判断が結果とどう噛み合ったか\n"
        "- 高評価、注目、見送りの配分から見える運用上の注意点\n"
        "- 未確定レースがある場合は、断定せず暫定評価として扱うこと\n"
        "- 直近30日と全期間傾向に照らした位置づけ\n\n"
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
        if not text or text in seen:
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


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _fallback_llm_bullets(llm_cards: List[Dict[str, Any]]) -> List[str]:
    bullets: List[str] = []
    for item in llm_cards[:3]:
        bullets.append(
            (
                f"**{_coerce_text(item.get('label'), '-')}**: "
                f"損益 {_coerce_text(item.get('profit_yen_text'), '-')}, "
                f"ROI {_coerce_text(item.get('roi_text'), '-')}, "
                f"的中 {_coerce_text(item.get('hit_races_text'), '-')}"
            )
        )
    return bullets


def _fallback_predictor_bullets(predictor_cards: List[Dict[str, Any]]) -> List[str]:
    bullets: List[str] = []
    for item in predictor_cards[:3]:
        bullets.append(
            (
                f"**{_coerce_text(item.get('label'), '-')}**: "
                f"トップ1ヒット率 {_coerce_text(item.get('top1_hit_rate_text'), '-')}, "
                f"トップ3ヒット率 {_coerce_text(item.get('top3_hit_rate_text'), '-')}, "
                f"トップ5からトップ3ヒット率 {_coerce_text(item.get('top5_to_top3_hit_rate_text'), '-')}, "
                f"サンプル {_int_value(item.get('samples'))}"
            )
        )
    return bullets


def _format_rate_note(count_value: Any, denominator_value: Any, label: str) -> str:
    count = _int_value(count_value)
    denominator = _int_value(denominator_value)
    if denominator <= 0:
        return f"{label}はまだ評価できる確定レースがありません。"
    return f"{label}は {count}/{denominator} レースでした。"


def _race_name_text(item: Dict[str, Any]) -> str:
    title = _coerce_text(item.get("race_title"), "-")
    name = _coerce_text(item.get("race_name"))
    if name and name != title and name not in title:
        return f"{title} {name}"
    return title


def _build_agent_fallback_document(source_payload: Dict[str, Any], *, fallback_reason: str = "") -> Dict[str, Any]:
    target_date_label = _coerce_text(source_payload.get("target_date_label"), "対象日")
    summary = dict(source_payload.get("agent_summary") or {})
    recent = dict(source_payload.get("recent_30_agent") or {})
    all_time = dict(source_payload.get("all_time_agent") or {})
    highlights = dict(source_payload.get("race_highlights") or {})
    settled_races = [dict(item or {}) for item in list(highlights.get("settled_races") or [])]
    missed_main = [dict(item or {}) for item in list(highlights.get("missed_main_races") or [])]
    high_confidence = [dict(item or {}) for item in list(highlights.get("high_confidence_races") or [])]

    predicted = _int_value(summary.get("predicted_races"))
    settled = _int_value(summary.get("settled_races"))
    pending = _int_value(summary.get("pending_races"))
    high_eval = _int_value(summary.get("high_evaluation_races"))
    watch_eval = _int_value(summary.get("watch_evaluation_races"))
    skip_eval = _int_value(summary.get("skip_evaluation_races"))

    main_top3_text = _coerce_text(summary.get("main_top3_rate_text"), "-")
    top5_cover_text = _coerce_text(summary.get("top5_cover_rate_text"), "-")
    exact_text = _coerce_text(summary.get("top3_exact_rate_text"), "-")
    recent_main_top3 = _coerce_text(recent.get("main_top3_rate_text"), "-")
    all_time_main_top3 = _coerce_text(all_time.get("main_top3_rate_text"), "-")

    title = f"{target_date_label} AI予測日報"
    lead = (
        f"{target_date_label}のAgent AI予測を、結果確定分と未確定分に分けて振り返ります。"
    )
    if settled > 0:
        article_summary = (
            f"この日は **本命複勝圏率 {main_top3_text}**、"
            f"上位5頭カバー {top5_cover_text} という形で、"
            "軸の置き方と相手候補の広がりを分けて見る一日でした。"
        )
    else:
        article_summary = (
            "この日は結果確定前のレースが中心で、"
            "**評価配分と印の置き方** を先に確認する段階です。"
        )

    overview_paragraphs = [
        (
            f"公開したAgent AI予測は {predicted} レース、結果確定は {settled} レース、"
            f"未確定は {pending} レースでした。"
        ),
        (
            f"判断配分は高評価 {high_eval} レース、注目 {watch_eval} レース、"
            f"見送り {skip_eval} レースです。高評価の数だけでなく、見送りまで含めた濃淡が当日の運用感を作っています。"
        ),
    ]
    if settled > 0:
        overview_paragraphs.append(
            f"確定分では **本命複勝圏率 {main_top3_text}**、"
            f"上位3頭完全的中率 {exact_text}。単に当たり外れを見るより、"
            "本命と相手候補のどちらが崩れたかを分けて読む必要があります。"
        )

    decision_bullets = [
        f"高評価: {high_eval}レース",
        f"注目: {watch_eval}レース",
        f"見送り: {skip_eval}レース",
    ]
    if high_confidence:
        decision_bullets.append(
            f"高評価上位: {', '.join(_race_name_text(item) for item in high_confidence[:3])}"
        )

    result_paragraphs = [
        _format_rate_note(summary.get("main_top3_count"), settled, "本命の複勝圏"),
        (
            f"上位5頭カバーは {top5_cover_text} で、"
            "本命が届かなかったレースでも候補群に結果馬を拾えていたかを見る指標になります。"
        ),
    ]
    if missed_main:
        result_paragraphs.append(
            "本命が馬券圏外だったレースは、印の中心を置いた理由と実際の決着のズレを次回確認したいところです。"
        )
    result_bullets = []
    for item in settled_races[:4]:
        result_bullets.append(
            f"{_race_name_text(item)}: {_coerce_text(item.get('outcome_text'), '-')}"
        )

    trend_paragraphs = []
    if recent:
        trend_paragraphs.append(
            f"直近30日では本命複勝圏率 {recent_main_top3}、"
            f"上位5頭カバー {_coerce_text(recent.get('top5_cover_rate_text'), '-')}。"
            "単日評価はこの範囲から大きく外れていないかを確認する材料になります。"
        )
    if all_time:
        trend_paragraphs.append(
            f"全期間では本命複勝圏率 {all_time_main_top3}、"
            f"上位3頭完全的中率 {_coerce_text(all_time.get('top3_exact_rate_text'), '-')}。"
            "今日の成績は、長期の癖に対して一時的な振れなのかを見極める必要があります。"
        )
    if not trend_paragraphs:
        trend_paragraphs.append("長期比較に使える履歴はまだ少なく、数日単位での追加検証が必要です。")

    closing_paragraphs = [
        "今日は一つの数字で良し悪しを決めるより、"
        "本命、上位候補、見送り判断を別々に点検する方が次につながります。"
    ]
    if fallback_reason:
        closing_paragraphs.append(f"この版はローカル要約で作成しました。理由: {fallback_reason}")

    sections: List[Dict[str, Any]] = [
        {"heading": "全体総括", "paragraphs": overview_paragraphs, "bullets": []},
        {"heading": "判断配分", "paragraphs": [], "bullets": decision_bullets},
        {"heading": "結果確定分の振り返り", "paragraphs": result_paragraphs, "bullets": result_bullets},
        {"heading": "直近と全期間の位置づけ", "paragraphs": trend_paragraphs, "bullets": []},
        {"heading": "まとめ", "paragraphs": closing_paragraphs, "bullets": []},
    ]

    return {
        "title": title,
        "lead": lead,
        "summary": article_summary,
        "tags": _coerce_list_text(["日報", "AI予測", target_date_label], limit=4),
        "markdown": _markdown_from_sections(sections),
        "sections": sections,
    }


def _build_fallback_document(source_payload: Dict[str, Any], *, fallback_reason: str = "") -> Dict[str, Any]:
    if source_payload.get("agent_summary") or source_payload.get("agent_races"):
        return _build_agent_fallback_document(source_payload, fallback_reason=fallback_reason)

    target_date_label = _coerce_text(source_payload.get("target_date_label"), "対象日")
    totals = dict(source_payload.get("totals") or {})
    best_ticket = dict(source_payload.get("best_ticket") or {})
    llm_cards = [dict(item or {}) for item in list(source_payload.get("llm_cards") or [])]
    predictor_cards = [dict(item or {}) for item in list(source_payload.get("predictor_cards") or [])]
    predictor_leader = dict(source_payload.get("predictor_leader") or {})
    all_time_llm = [dict(item or {}) for item in list(source_payload.get("all_time_llm") or [])]
    all_time_predictor = [dict(item or {}) for item in list(source_payload.get("all_time_predictor") or [])]

    top_engine = dict(llm_cards[0] or {}) if llm_cards else {}
    worst_engine = min(llm_cards, key=lambda item: _int_value(item.get("profit_yen"))) if llm_cards else {}
    predictor_runner_up = dict(predictor_cards[1] or {}) if len(predictor_cards) > 1 else {}
    all_time_top_engine = dict(all_time_llm[0] or {}) if all_time_llm else {}
    all_time_top_predictor = dict(all_time_predictor[0] or {}) if all_time_predictor else {}
    llm_all_negative = bool(llm_cards) and all(_int_value(item.get("profit_yen")) <= 0 for item in llm_cards)

    title = f"{target_date_label} 競馬分析日報"
    lead = (
        f"{target_date_label}の公開レースを材料に、"
        "LLM馬券と予測モデルの動きを一つの流れとして振り返ります。"
    )
    if llm_all_negative:
        summary = (
            "この日は **LLM馬券が揃って苦戦** し、"
            "勝ち筋が見えたというよりも、どこで読みが重なって外れたかを意識したい一日でした。"
        )
    elif top_engine:
        summary = (
            f"この日は **{_coerce_text(top_engine.get('label'), '上位モデル')}** が最も粘りましたが、"
            "圧倒的に抜けたというよりは、傷が浅かったという見方が妥当です。"
        )
    else:
        summary = "この日の公開レースについて、当日の傾向を短く整理しました。"

    sections: List[Dict[str, Any]] = []

    overview_paragraphs = [
        (
            f"公開 { _int_value(totals.get('race_count')) } レースのうち、"
            f"結果が確定したのは { _int_value(totals.get('settled_count')) } レースでした。"
            if _int_value(totals.get("race_count")) or _int_value(totals.get("settled_count"))
            else "当日の公開レースをもとに、主要な動きだけを拾って整理します。"
        ),
    ]
    if llm_all_negative:
        overview_paragraphs.append(
            "全体としては **強気に評価できる買い筋が乏しく**、"
            "当日結果だけを見る限りは守りの判断が必要な日でした。"
        )
    elif top_engine:
        overview_paragraphs.append(
            f"LLM側では **{_coerce_text(top_engine.get('label'), '-')}** が最上位でしたが、"
            "数値差だけで優位と決めつけるにはまだ慎重さが要ります。"
        )
    if predictor_leader:
        overview_paragraphs.append(
            f"予測モデルでは **{_coerce_text(predictor_leader.get('label'), '-')}** が目立ち、"
            f"トップ5からトップ3ヒット率 {_coerce_text(predictor_leader.get('top5_to_top3_hit_rate_text'), '-')} を残しました。"
        )
    sections.append({"heading": "全体総括", "paragraphs": overview_paragraphs, "bullets": []})

    llm_paragraphs: List[str] = []
    if top_engine and worst_engine:
        llm_paragraphs.append(
            f"LLM馬券は **{_coerce_text(top_engine.get('label'), '-')}** と "
            f"**{_coerce_text(worst_engine.get('label'), '-')}** の差を見ると、"
            "レース選択や参加本数の違いがそのまま損益差に表れています。"
        )
    if llm_all_negative:
        llm_paragraphs.append(
            "全モデルがマイナス圏に沈んだ日は、順位そのものよりも "
            "**同じような外し方をしていないか** を読む方が次に効きます。"
        )
    elif top_engine:
        llm_paragraphs.append(
            f"最上位でも ROI {_coerce_text(top_engine.get('roi_text'), '-')} にとどまるなら、"
            "今日は当て切ったというより相対的に崩れにくかった日と解釈できます。"
        )
    sections.append(
        {
            "heading": "LLM馬券の振り返り",
            "paragraphs": llm_paragraphs,
            "bullets": _fallback_llm_bullets(llm_cards),
        }
    )

    predictor_paragraphs: List[str] = []
    if predictor_leader:
        predictor_paragraphs.append(
            f"予測モデルでは **{_coerce_text(predictor_leader.get('label'), '-')}** が先頭に立ちましたが、"
            f"サンプル {_int_value(predictor_leader.get('samples'))} の数字なら、"
            "好結果をそのまま実力値とみなすのは早い段階です。"
        )
    if predictor_runner_up:
        predictor_paragraphs.append(
            f"一方で **{_coerce_text(predictor_runner_up.get('label'), '-')}** も近い位置におり、"
            "単日で序列を固定するより数日単位で見た方が輪郭がはっきりします。"
        )
    sections.append(
        {
            "heading": "予測モデルの見どころ",
            "paragraphs": predictor_paragraphs,
            "bullets": _fallback_predictor_bullets(predictor_cards),
        }
    )

    long_term_paragraphs: List[str] = []
    if all_time_top_engine:
        long_term_paragraphs.append(
            f"長期では **{_coerce_text(all_time_top_engine.get('label'), '-')}** が "
            f"ROI {_coerce_text(all_time_top_engine.get('roi_text'), '-')} で先行しており、"
            "今日の並びと一致するかどうかが流れを判断する手掛かりになります。"
        )
    if all_time_top_predictor:
        long_term_paragraphs.append(
            f"予測モデルの全期間軸では **{_coerce_text(all_time_top_predictor.get('label'), '-')}** が"
            f"トップ1ヒット率 {_coerce_text(all_time_top_predictor.get('top1_hit_rate_text'), '-')} を維持しており、"
            "単日好調組とのズレも見逃せません。"
        )
    sections.append({"heading": "全期間との対比", "paragraphs": long_term_paragraphs, "bullets": []})

    closing_paragraphs: List[str] = []
    if best_ticket:
        closing_paragraphs.append(
            f"単発では **{_coerce_text(best_ticket.get('race_title'), '-')}** の"
            f" {_coerce_text(best_ticket.get('bet_type_label'), '的中')} が"
            f" {_coerce_text(best_ticket.get('multiplier_text'), '-')} を付けており、"
            "当日の中では数少ない見せ場になりました。"
        )
    closing_paragraphs.append(
        "総じて今日は『勝ったモデルを称える日』というより、"
        "どの判断が浅く、どの指標が次に繋がりそうかを切り分ける日だったと言えます。"
    )
    if fallback_reason:
        closing_paragraphs.append(f"この版はローカル要約で作成しました。理由: {fallback_reason}")
    sections.append({"heading": "まとめと考察", "paragraphs": closing_paragraphs, "bullets": []})

    return {
        "title": title,
        "lead": lead,
        "summary": summary,
        "tags": _coerce_list_text(["日報", target_date_label], limit=4),
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
            "temperature": 0.45,
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
        "temperature": 0.45,
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
        temperature=0.45,
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
            "temperature": 0.45,
            "max_output_tokens": 4096,
        },
    )
    return _gemini._extract_response_text(response)


def generate_daily_report_document(
    source_payload: Dict[str, Any],
    *,
    policy_engine: str = "gemini",
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

    fallback_document = _build_fallback_document(source_payload, fallback_reason=fallback_reason)
    return {
        "document": fallback_document,
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

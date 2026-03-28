import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from local_env import load_local_env


BASE_DIR = Path(__file__).resolve().parent
load_local_env(BASE_DIR, override=False)


def ntfy_notify_enabled():
    raw = str(os.environ.get("PIPELINE_NTFY_NOTIFY_ENABLED", "") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def ntfy_server_url():
    value = str(os.environ.get("PIPELINE_NTFY_SERVER_URL", "https://ntfy.sh") or "").strip()
    return (value or "https://ntfy.sh").rstrip("/")


def ntfy_topic():
    return str(os.environ.get("PIPELINE_NTFY_TOPIC", "") or "").strip().strip("/")


def preferred_ntfy_engine():
    return str(os.environ.get("PIPELINE_NTFY_ENGINE", "") or "").strip().lower()


def _public_site_url():
    value = str(os.environ.get("PIPELINE_PUBLIC_SITE_URL", "https://www.ikaimo-ai.com") or "").strip()
    return (value or "https://www.ikaimo-ai.com").rstrip("/")


def _public_base_path():
    value = str(os.environ.get("PIPELINE_PUBLIC_BASE_PATH", "/keiba") or "").strip()
    if not value:
        return "/keiba"
    if not value.startswith("/"):
        value = "/" + value
    return value.rstrip("/") or "/keiba"


def build_workspace_url(scope_key, run_id):
    scope_text = quote(str(scope_key or "").strip())
    run_text = quote(str(run_id or "").strip())
    return f"{_public_site_url()}{_public_base_path()}/console/workspace?scope_key={scope_text}&run_id={run_text}"


def build_x_intent_url(share_text):
    return f"https://twitter.com/intent/tweet?text={quote(str(share_text or ''), safe='')}"


def _basic_auth_header(username, password):
    raw = f"{username}:{password}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _build_auth_header():
    token = str(os.environ.get("PIPELINE_NTFY_TOKEN", "") or "").strip()
    if token:
        return "Bearer " + token
    username = str(os.environ.get("PIPELINE_NTFY_USERNAME", "") or "").strip()
    password = str(os.environ.get("PIPELINE_NTFY_PASSWORD", "") or "").strip()
    if username and password:
        return _basic_auth_header(username, password)
    return ""


def _select_share_candidate(scope_key, run_id):
    import web_app  # local import to avoid circular imports

    def _normalize_horse_no_text(value):
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return str(int(float(text)))
        except (TypeError, ValueError):
            return text

    def _build_v6_marks_text(resolved_scope_key, resolved_run_id, row):
        odds_path = web_app.resolve_run_asset_path(resolved_scope_key, resolved_run_id, row, "odds_path", "odds")
        fuku_odds_path = web_app.resolve_run_asset_path(resolved_scope_key, resolved_run_id, row, "fuku_odds_path", "fuku_odds")
        name_to_no_map = web_app.load_name_to_no(odds_path) if odds_path and Path(odds_path).exists() else {}
        win_odds_map = web_app.load_win_odds_map(odds_path) if odds_path and Path(odds_path).exists() else {}
        place_odds_map = web_app.load_place_odds_map(fuku_odds_path) if fuku_odds_path and Path(fuku_odds_path).exists() else {}
        predictor_context = web_app.build_multi_predictor_context(
            resolved_scope_key,
            resolved_run_id,
            row,
            name_to_no_map,
            win_odds_map,
            place_odds_map,
        )
        for item in list((predictor_context or {}).get("predictor_rankings", []) or []):
            predictor_id = str(item.get("predictor_id", "") or "").strip()
            if predictor_id != "v6_kiwami":
                continue
            marks_map = {}
            ranking = list(item.get("ranking", []) or [])
            for symbol, rank_item in zip(("◎", "○", "▲", "△", "☆"), ranking[:5]):
                horse_no = _normalize_horse_no_text((rank_item or {}).get("horse_no", ""))
                if horse_no:
                    marks_map[horse_no] = symbol
            if marks_map:
                return web_app.report_format_marks_text(marks_map)
        return "印なし"

    run_row = web_app.resolve_run(run_id, scope_key)
    if run_row is None:
        raise LookupError(f"run row not found for run_id={run_id}")
    resolved_scope_key = web_app.normalize_scope_key(scope_key) or str(scope_key or "").strip()
    resolved_run_id = str(run_id or run_row.get("run_id") or "").strip()
    location = str(run_row.get("location", "") or "").strip()
    race_no = web_app.report_race_no_text(run_row.get("race_id")) if hasattr(web_app, "report_race_no_text") else ""
    header = " ".join(part for part in (location, race_no) if str(part or "").strip())
    marks_text = _build_v6_marks_text(resolved_scope_key, resolved_run_id, run_row)
    share_lines = [line for line in (header, "極 KIWAMI", marks_text) if str(line or "").strip()]
    share_text = "\n".join(share_lines).strip()
    return {
        "engine": "v6_kiwami",
        "run_row": dict(run_row or {}),
        "share_text": str(share_text or "").strip(),
    }


def build_ntfy_share_notification(scope_key, run_id):
    candidate = _select_share_candidate(scope_key, run_id)
    run_row = dict(candidate.get("run_row", {}) or {})
    race_id = str(run_row.get("race_id", "") or "").strip()
    location = str(run_row.get("location", "") or "").strip()
    race_date = str(run_row.get("date", "") or run_row.get("race_date", "") or "").strip()
    title_parts = [part for part in (location, race_id) if part]
    title = " ".join(title_parts) if title_parts else f"预测完成 {run_id}"
    if race_date:
        title = f"{title} {race_date}"
    share_text = str(candidate.get("share_text", "") or "").strip()
    if not share_text:
        raise ValueError(f"share text not available for run_id={run_id}")
    return {
        "engine": str(candidate.get("engine", "") or "").strip(),
        "share_text": share_text,
        "intent_url": build_x_intent_url(share_text),
        "workspace_url": build_workspace_url(scope_key, run_id),
        "title": title,
    }


def publish_ntfy_share_notification(scope_key, run_id):
    if not ntfy_notify_enabled():
        return {
            "ok": False,
            "skipped": True,
            "reason": "disabled",
        }

    topic = ntfy_topic()
    if not topic:
        return {
            "ok": False,
            "skipped": True,
            "reason": "missing_topic",
        }

    notification = build_ntfy_share_notification(scope_key, run_id)
    request_payload = {
        "topic": topic,
        "message": notification["share_text"],
        "title": notification["title"],
        "click": notification["workspace_url"],
        "tags": ["horse_racing", "signal_strength"],
        "actions": [
            {
                "action": "view",
                "label": "发布到X",
                "url": notification["intent_url"],
                "clear": False,
            },
            {
                "action": "view",
                "label": "Workspace",
                "url": notification["workspace_url"],
                "clear": False,
            },
        ],
    }

    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=f"{ntfy_server_url()}/",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
        },
    )

    auth_header = _build_auth_header()
    if auth_header:
        request.add_header("Authorization", auth_header)

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ntfy http {exc.code}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ntfy request failed: {exc}") from exc

    payload = {}
    if raw.strip():
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}

    return {
        "ok": True,
        "engine": notification["engine"],
        "share_text": notification["share_text"],
        "intent_url": notification["intent_url"],
        "workspace_url": notification["workspace_url"],
        "topic": topic,
        "message_id": str(payload.get("id", "") or "").strip(),
    }

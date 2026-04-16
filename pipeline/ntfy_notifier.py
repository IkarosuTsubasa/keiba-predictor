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
_FCM_APP = None


def _flag_env(name):
    raw = str(os.environ.get(name, "") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def ntfy_notify_enabled():
    return _flag_env("PIPELINE_NTFY_NOTIFY_ENABLED")


def fcm_notify_enabled():
    return _flag_env("PIPELINE_FCM_NOTIFY_ENABLED")


def ntfy_server_url():
    value = str(os.environ.get("PIPELINE_NTFY_SERVER_URL", "https://ntfy.sh") or "").strip()
    return (value or "https://ntfy.sh").rstrip("/")


def ntfy_topic():
    return str(os.environ.get("PIPELINE_NTFY_TOPIC", "") or "").strip().strip("/")


def fcm_topic():
    return str(os.environ.get("PIPELINE_FCM_TOPIC", "keiba-public-updates") or "").strip().strip("/")


def preferred_ntfy_engine():
    return str(os.environ.get("PIPELINE_NTFY_ENGINE", "") or "").strip().lower()


def _fcm_credentials_path():
    explicit = str(os.environ.get("PIPELINE_FCM_SERVICE_ACCOUNT_FILE", "") or "").strip()
    fallback = str(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    value = explicit or fallback
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return str(path)


def _fcm_credentials_info():
    raw = str(os.environ.get("PIPELINE_FCM_SERVICE_ACCOUNT_JSON", "") or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("PIPELINE_FCM_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("PIPELINE_FCM_SERVICE_ACCOUNT_JSON must decode to a JSON object")
    return payload


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


def _build_public_race_url(run_row):
    row = dict(run_row or {})
    run_id = str(row.get("run_id", "") or "").strip()
    if not run_id:
        return f"{_public_site_url()}{_public_base_path()}"
    race_url = f"{_public_site_url()}{_public_base_path()}/race/{quote(run_id, safe='')}"
    race_date = str(row.get("date", "") or row.get("race_date", "") or "").strip()
    if race_date:
        return f"{race_url}?date={quote(race_date, safe='')}"
    return race_url


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


def _get_firebase_app():
    global _FCM_APP

    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError as exc:
        raise RuntimeError("firebase-admin is not installed") from exc

    if _FCM_APP is not None:
        return _FCM_APP

    try:
        _FCM_APP = firebase_admin.get_app()
        return _FCM_APP
    except ValueError:
        pass

    credentials_info = _fcm_credentials_info()
    if credentials_info:
        _FCM_APP = firebase_admin.initialize_app(credentials.Certificate(credentials_info))
        return _FCM_APP

    credentials_path = _fcm_credentials_path()
    if credentials_path:
        cred_file = Path(credentials_path)
        if not cred_file.exists():
            raise RuntimeError(f"fcm service account file not found: {cred_file}")
        _FCM_APP = firebase_admin.initialize_app(credentials.Certificate(str(cred_file)))
        return _FCM_APP

    _FCM_APP = firebase_admin.initialize_app()
    return _FCM_APP


def _select_share_candidate(scope_key, run_id):
    import web_app  # local import to avoid circular imports

    def _confidence_rank_text(value):
        score = float(value or 0.0)
        if score >= 0.60:
            return "SSS"
        if score >= 0.50:
            return "SS"
        if score >= 0.40:
            return "S"
        if score >= 0.32:
            return "A"
        if score >= 0.24:
            return "B"
        if score >= 0.15:
            return "C"
        return "D"

    def _normalize_horse_no_text(value):
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return str(int(float(text)))
        except (TypeError, ValueError):
            return text

    def _horse_sort_key(value):
        try:
            return int(float(str(value or "").strip()))
        except (TypeError, ValueError):
            return 999

    def _build_consensus_marks_text(resolved_scope_key, resolved_run_id, row):
        mark_weight = {"◎": 5, "○": 4, "▲": 3, "△": 2, "☆": 1}
        mark_order = ("◎", "○", "▲", "△", "☆")
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
        predictor_rankings = [
            item for item in list((predictor_context or {}).get("predictor_rankings", []) or [])
            if list((item or {}).get("ranking", []) or [])
        ]
        model_count = len(predictor_rankings)
        if model_count <= 0:
            return {
                "marks_text": "印なし",
                "confidence_text": "",
            }

        tally = {}
        for predictor_item in predictor_rankings:
            ranking = list((predictor_item or {}).get("ranking", []) or [])[:5]
            seen = set()
            marks_by_horse = {}
            for symbol, rank_item in zip(mark_order, ranking):
                horse_no = _normalize_horse_no_text((rank_item or {}).get("horse_no", ""))
                if horse_no:
                    marks_by_horse[horse_no] = symbol
            for rank_item in ranking:
                horse_no = _normalize_horse_no_text((rank_item or {}).get("horse_no", ""))
                if not horse_no:
                    continue
                symbol = str(marks_by_horse.get(horse_no, "") or "").strip()
                entry = tally.setdefault(
                    horse_no,
                    {
                        "horse_no": horse_no,
                        "score": 0.0,
                        "support_count": 0,
                        "main_count": 0,
                        "top3_prob_total": 0.0,
                        "rank_score_total": 0.0,
                        "entry_count": 0,
                    },
                )
                entry["score"] += float(mark_weight.get(symbol, 0) or 0)
                entry["top3_prob_total"] += max(0.0, float((rank_item or {}).get("top3_prob_model", 0.0) or 0.0))
                entry["rank_score_total"] += max(0.0, float((rank_item or {}).get("rank_score_norm", 0.0) or 0.0))
                entry["entry_count"] += 1
                if symbol == "◎":
                    entry["main_count"] += 1
                if horse_no not in seen:
                    entry["support_count"] += 1
                    seen.add(horse_no)

        rows = []
        for horse_no, item in tally.items():
            support_count = int(item.get("support_count", 0) or 0)
            entry_count = int(item.get("entry_count", 0) or 0)
            score = float(item.get("score", 0.0) or 0.0)
            avg_mark_strength = score / float(support_count * 5) if support_count > 0 else 0.0
            avg_top3_prob = float(item.get("top3_prob_total", 0.0) or 0.0) / float(entry_count) if entry_count > 0 else 0.0
            avg_rank_score = float(item.get("rank_score_total", 0.0) or 0.0) / float(entry_count) if entry_count > 0 else 0.0
            ai_index = max(
                1,
                min(
                    99,
                    int(
                        round(
                            42
                            + 20 * max(0.0, min(1.0, avg_mark_strength))
                            + 16 * max(0.0, min(1.0, avg_top3_prob))
                            + 12 * max(0.0, min(1.0, avg_rank_score))
                            + 10 * (float(support_count) / float(model_count)),
                        )
                    ),
                ),
            )
            rows.append(
                {
                    "horse_no": horse_no,
                    "score": score,
                    "main_count": int(item.get("main_count", 0) or 0),
                    "support_count": support_count,
                    "ai_index": ai_index,
                }
            )

        rows.sort(
            key=lambda item: (
                -int(item.get("ai_index", 0) or 0),
                -float(item.get("score", 0.0) or 0.0),
                -int(item.get("main_count", 0) or 0),
                _horse_sort_key(item.get("horse_no", "")),
            )
        )
        top5 = rows[:5]
        marks_map = {}
        for symbol, item in zip(mark_order, top5):
            horse_no = _normalize_horse_no_text(item.get("horse_no", ""))
            if horse_no:
                marks_map[horse_no] = symbol
        if marks_map:
            marks_text = web_app.report_format_marks_text(marks_map)
            top = top5[0] if top5 else {}
            second = top5[1] if len(top5) > 1 else {}
            support_ratio = (
                float(top.get("main_count", 0) or 0) / float(model_count)
                if model_count > 0 else 0.0
            )
            top_index = float(top.get("ai_index", 0) or 0.0)
            second_index = float(second.get("ai_index", 0) or 0.0)
            margin_ratio = (
                max(0.0, (top_index - second_index) / top_index)
                if top_index > 0 else 0.0
            )
            confidence_score = max(0.0, min(1.0, 0.55 * support_ratio + 0.45 * margin_ratio))
            return {
                "marks_text": marks_text,
                "confidence_text": _confidence_rank_text(confidence_score),
            }
        return {
            "marks_text": "印なし",
            "confidence_text": "",
        }

    run_row = web_app.resolve_run(run_id, scope_key)
    if run_row is None:
        raise LookupError(f"run row not found for run_id={run_id}")

    resolved_scope_key = web_app.normalize_scope_key(scope_key) or str(scope_key or "").strip()
    resolved_run_id = str(run_id or run_row.get("run_id") or "").strip()
    location = str(run_row.get("location", "") or "").strip()
    race_no = web_app.report_race_no_text(run_row.get("race_id")) if hasattr(web_app, "report_race_no_text") else ""
    venue = "".join(location.split())
    if venue and not venue.endswith("競馬"):
        venue = f"{venue}競馬"
    race_name = _resolve_notification_race_name(scope_key, resolved_run_id, run_row)
    if race_name and race_name in f"{venue} {race_no}".strip():
        race_name = ""
    header_body = " ".join(part for part in (venue, race_no, race_name) if str(part or "").strip())
    header = f"#{header_body}" if header_body else ""
    marks_meta = _build_consensus_marks_text(resolved_scope_key, resolved_run_id, run_row)
    marks_text = str(marks_meta.get("marks_text", "") or "").strip() or "印なし"
    confidence_text = str(marks_meta.get("confidence_text", "") or "").strip()
    marks_line = f"{marks_text} ｜自信度 {confidence_text}" if confidence_text else marks_text
    public_url = _build_public_race_url(run_row)
    share_text = "\n".join(
        [
            header,
            "",
            marks_line,
            "",
            "このレースの",
            "AI最終評価・期待値はこちら👇",
            "",
            "📱AI予想はアプリで最速公開",
            "今すぐダウンロード👇",
            "https://x.gd/BDVgd",
            "",
            "🌐 全モデル予想はこちら",
            public_url,
            "",
            "#いかいもAI競馬 #競馬予想",
        ]
    ).strip()
    return {
        "engine": "predictor_consensus",
        "run_row": dict(run_row or {}),
        "share_text": share_text,
        "public_url": public_url,
    }


def _prediction_complete_text(run_row, run_id):
    row = dict(run_row or {})
    location = str(row.get("location", "") or "").strip()
    race_no = ""
    try:
        import web_app  # local import to avoid circular imports

        race_no = web_app.report_race_no_text(row.get("race_id")) if hasattr(web_app, "report_race_no_text") else ""
    except Exception:
        race_no = str(row.get("race_id", "") or "").strip()
    venue = "".join(location.split())
    if venue and not venue.endswith("競馬"):
        venue = f"{venue}競馬"
    race_name = _resolve_notification_race_name("", run_id, row)
    core = " ".join(part for part in (venue, race_no, race_name) if str(part or "").strip())
    if core:
        return f"#{core} の予測が完了しました"
    resolved_run_id = str(run_id or row.get("run_id") or "").strip()
    return f"#{resolved_run_id} の予測が完了しました"


def _resolve_notification_race_name(scope_key, run_id, run_row):
    row = dict(run_row or {})
    race_name = str(row.get("race_name", "") or row.get("trigger_race", "") or "").strip()
    if race_name:
        return race_name
    try:
        import web_app  # local import to avoid circular imports

        load_jobs = getattr(web_app, "load_race_jobs", None)
        base_dir = getattr(web_app, "BASE_DIR", None)
        normalize_scope_key = getattr(web_app, "normalize_scope_key", None)
        if not callable(load_jobs) or base_dir is None:
            return ""
        scope_norm = (
            normalize_scope_key(scope_key) if callable(normalize_scope_key) else str(scope_key or "").strip()
        )
        for job in reversed(list(load_jobs(base_dir) or [])):
            job_run_id = str((job or {}).get("current_run_id", "") or "").strip()
            if job_run_id != str(run_id or "").strip():
                continue
            job_scope = (
                normalize_scope_key((job or {}).get("scope_key", ""))
                if callable(normalize_scope_key)
                else str((job or {}).get("scope_key", "") or "").strip()
            )
            if scope_norm and job_scope and job_scope != scope_norm:
                continue
            return str((job or {}).get("race_name", "") or "").strip()
    except Exception:
        return ""
    return ""


def build_ntfy_share_notification(scope_key, run_id):
    candidate = _select_share_candidate(scope_key, run_id)
    run_row = dict(candidate.get("run_row", {}) or {})
    race_id = str(run_row.get("race_id", "") or "").strip()
    location = str(run_row.get("location", "") or "").strip()
    race_name = _resolve_notification_race_name(scope_key, run_id, run_row)
    title_parts = [part for part in (location, race_id, race_name) if part]
    title = " ".join(title_parts) if title_parts else f"预测完成 {run_id}"
    share_text = str(candidate.get("share_text", "") or "").strip()
    if not share_text:
        raise ValueError(f"share text not available for run_id={run_id}")
    return {
        "engine": str(candidate.get("engine", "") or "").strip(),
        "share_text": share_text,
        "intent_url": build_x_intent_url(share_text),
        "public_url": str(candidate.get("public_url", "") or "").strip(),
        "workspace_url": build_workspace_url(scope_key, run_id),
        "title": title,
    }


def build_fcm_prediction_notification(scope_key, run_id):
    candidate = _select_share_candidate(scope_key, run_id)
    run_row = dict(candidate.get("run_row", {}) or {})
    body = _prediction_complete_text(run_row, run_id)
    return {
        "engine": str(candidate.get("engine", "") or "").strip(),
        "title": "🐴予測が完了しました",
        "body": body,
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
        "click": notification["public_url"] or notification["workspace_url"],
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
        "public_url": notification["public_url"],
        "workspace_url": notification["workspace_url"],
        "topic": topic,
        "message_id": str(payload.get("id", "") or "").strip(),
    }


def publish_fcm_prediction_notification(scope_key, run_id):
    if not fcm_notify_enabled():
        return {
            "ok": False,
            "skipped": True,
            "reason": "disabled",
        }

    topic = fcm_topic()
    if not topic:
        return {
            "ok": False,
            "skipped": True,
            "reason": "missing_topic",
        }

    notification = build_fcm_prediction_notification(scope_key, run_id)

    try:
        from firebase_admin import messaging
    except ImportError as exc:
        raise RuntimeError("firebase-admin is not installed") from exc

    app = _get_firebase_app()
    message = messaging.Message(
        topic=topic,
        notification=messaging.Notification(
            title=notification["title"],
            body=notification["body"],
        ),
        android=messaging.AndroidConfig(
            priority="high",
        ),
    )
    message_id = str(messaging.send(message, app=app) or "").strip()
    return {
        "ok": True,
        "engine": notification["engine"],
        "title": notification["title"],
        "body": notification["body"],
        "topic": topic,
        "message_id": message_id,
    }


def publish_share_notifications(scope_key, run_id):
    channel_results = {}
    errors = []
    engine = ""
    sent_any = False

    for channel_name, sender in (
        ("ntfy", publish_ntfy_share_notification),
        ("fcm", publish_fcm_prediction_notification),
    ):
        try:
            result = sender(scope_key, run_id)
        except Exception as exc:
            result = {
                "ok": False,
                "skipped": False,
                "reason": "error",
                "error": str(exc or "").strip(),
            }
            errors.append(f"{channel_name}: {result['error']}")
        channel_results[channel_name] = result
        if result.get("ok") and not result.get("skipped"):
            sent_any = True
            if not engine:
                engine = str(result.get("engine", "") or "").strip()

    if sent_any:
        ntfy_result = channel_results.get("ntfy") or {}
        return {
            "ok": True,
            "engine": engine,
            "topic": str(ntfy_result.get("topic", "") or "").strip(),
            "message_id": str(ntfy_result.get("message_id", "") or "").strip(),
            "channels": channel_results,
        }

    if errors:
        raise RuntimeError("; ".join(error for error in errors if error))

    reasons = [
        str((channel_results.get(name) or {}).get("reason", "") or "").strip()
        for name in ("ntfy", "fcm")
    ]
    reason = ",".join(part for part in reasons if part) or "skipped"
    return {
        "ok": False,
        "skipped": True,
        "reason": reason,
        "channels": channel_results,
    }

import asyncio
import os
from pathlib import Path


def auto_post_x_enabled():
    raw = str(os.environ.get("PIPELINE_AUTO_POST_X_ENABLED", "") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _shared_dir(base_dir):
    path = Path(base_dir) / "data" / "_shared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cookies_file(base_dir):
    raw = str(os.environ.get("TWIKIT_COOKIES_FILE", "") or "").strip()
    if raw:
        return Path(raw)
    return _shared_dir(base_dir) / "twikit_cookies.json"


def _preferred_engine():
    return str(os.environ.get("PIPELINE_AUTO_POST_X_ENGINE", "") or "").strip().lower()


def _twikit_language():
    return str(os.environ.get("TWIKIT_LANGUAGE", "ja-JP") or "ja-JP").strip() or "ja-JP"


def _has_saved_cookies(path):
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _select_payload(payloads):
    rows = [dict(item) for item in list(payloads or []) if isinstance(item, dict)]
    if not rows:
        return None
    preferred = _preferred_engine()
    if preferred:
        for item in rows:
            if str(item.get("policy_engine", "") or "").strip().lower() == preferred:
                return item
    for item in rows:
        budgets = list(item.get("budgets", []) or [])
        for budget in budgets:
            tickets = list((budget or {}).get("tickets", []) or [])
            if tickets:
                return item
    return rows[0]


def build_ready_post_payload(base_dir, scope_key, run_id):
    import web_app

    run_row = web_app.resolve_run(run_id, scope_key)
    if not run_row:
        raise RuntimeError(f"run row not found for x auto post: {run_id}")
    payloads = web_app.load_policy_payloads(scope_key, run_id, run_row)
    payload = _select_payload(payloads)
    if not payload:
        raise RuntimeError(f"policy payload not found for x auto post: {run_id}")
    engine = web_app.normalize_policy_engine(str(payload.get("policy_engine", "") or ""))
    marks_map = web_app.report_policy_marks_map(payload)
    primary_budget = web_app.report_policy_primary_budget(payload)
    ticket_rows = web_app.load_policy_run_ticket_rows(run_id, policy_engine=engine) or list(
        primary_budget.get("tickets", []) or []
    )
    share_text = web_app.build_public_share_text(run_row, engine, marks_map, ticket_rows)
    share_text = str(share_text or "").strip()
    if not share_text:
        raise RuntimeError(f"share text not built for x auto post: {run_id}")
    return {
        "scope_key": str(scope_key or "").strip(),
        "run_id": str(run_id or "").strip(),
        "race_id": str(run_row.get("race_id", "") or "").strip(),
        "engine": engine,
        "text": share_text,
    }


async def _post_with_twikit_async(base_dir, text):
    try:
        from twikit import Client
    except Exception as exc:
        raise RuntimeError("twikit is not installed") from exc

    username = str(os.environ.get("TWIKIT_USERNAME", "") or "").strip()
    email = str(os.environ.get("TWIKIT_EMAIL", "") or "").strip()
    password = str(os.environ.get("TWIKIT_PASSWORD", "") or "").strip()
    totp_secret = str(os.environ.get("TWIKIT_TOTP_SECRET", "") or "").strip()
    if not username or not password:
        raise RuntimeError("TWIKIT_USERNAME / TWIKIT_PASSWORD missing")

    cookies_file = _cookies_file(base_dir)
    cookies_file.parent.mkdir(parents=True, exist_ok=True)
    client = Client(language=_twikit_language())

    cookie_error = None
    if _has_saved_cookies(cookies_file):
        try:
            client.load_cookies(str(cookies_file))
            tweet = await client.create_tweet(text=str(text or ""))
            tweet_id = str(getattr(tweet, "id", "") or "").strip()
            return {"tweet_id": tweet_id, "auth_mode": "cookies"}
        except Exception as exc:
            cookie_error = exc

    login_error = None
    try:
        await client.login(
            auth_info_1=username,
            auth_info_2=email or None,
            password=password,
            totp_secret=totp_secret or None,
            cookies_file=str(cookies_file),
            enable_ui_metrics=False,
        )
        try:
            client.save_cookies(str(cookies_file))
        except Exception:
            pass
    except Exception as exc:
        login_error = exc

    if login_error is not None:
        if cookie_error is not None:
            raise RuntimeError(f"twikit cookie login failed: {cookie_error}; fresh login failed: {login_error}") from login_error
        raise RuntimeError(str(login_error)) from login_error

    tweet = await client.create_tweet(text=str(text or ""))
    tweet_id = str(getattr(tweet, "id", "") or "").strip()
    return {"tweet_id": tweet_id, "auth_mode": "login"}


def post_ready_run(base_dir, scope_key, run_id):
    if not auto_post_x_enabled():
        return {"status": "disabled"}
    payload = build_ready_post_payload(base_dir, scope_key, run_id)
    result = asyncio.run(_post_with_twikit_async(base_dir, payload["text"]))
    return {
        "status": "succeeded",
        "scope_key": payload["scope_key"],
        "run_id": payload["run_id"],
        "race_id": payload["race_id"],
        "engine": payload["engine"],
        "tweet_id": str((result or {}).get("tweet_id", "") or "").strip(),
        "auth_mode": str((result or {}).get("auth_mode", "") or "").strip(),
        "text": payload["text"],
    }

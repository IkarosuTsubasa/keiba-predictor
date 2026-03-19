import os
import os
import sys
from pathlib import Path

import web_app


def assert_true(cond, message):
    if not cond:
        raise AssertionError(message)


def pick_latest_run_id():
    for scope in ("central_turf", "central_dirt", "local"):
        row = web_app.resolve_run("", scope)
        if row and row.get("run_id"):
            return row["run_id"]
    return ""


def main():
    body = web_app.index()
    assert_true(getattr(body, "status_code", 0) in (302, 307), "index should redirect to /keiba")
    assert_true(getattr(body, "headers", {}).get("location") == web_app.PUBLIC_BASE_PATH, "index should redirect to /keiba")

    keiba_home = web_app.llm_today()
    keiba_html = getattr(keiba_home, "body", b"").decode("utf-8", errors="ignore")
    assert_true(getattr(keiba_home, "status_code", 0) == 200, "keiba home should return 200")
    assert_true("<!doctype html" in keiba_html.lower(), "keiba home should render public frontend html")
    assert_true("twitter:card" in keiba_html, "keiba home should inject public meta tags")
    assert_true("Xlogo-white.png" in keiba_html, "keiba home should inject share runtime")

    llm_today_html = web_app.llm_today()
    llm_today_body = getattr(llm_today_html, "body", b"").decode("utf-8", errors="ignore")
    assert_true(getattr(llm_today_html, "status_code", 0) == 200, "llm_today should return 200")
    assert_true("</html>" in llm_today_body.lower(), "llm_today should render public frontend file")
    board_payload = web_app.public_board_api()
    assert_true(getattr(board_payload, "status_code", 200) == 200, "public board api should return JSON")

    console_html = web_app.console_index()
    assert_true("Run Pipeline" in console_html, "console missing Run Pipeline block")
    assert_true('action="/view_run"' in console_html, "console missing view_run form")
    assert_true('id="admin-zone"' in console_html, "console missing admin workspace")
    assert_true(f'action="{web_app.CONSOLE_BASE_PATH}/tasks/create"' in console_html, "console missing merged task create form")
    assert_true(f'action="{web_app.CONSOLE_BASE_PATH}/tasks/import_archive"' in console_html, "console missing import archive form")
    assert_true(f"{web_app.CONSOLE_BASE_PATH}/note" in console_html, "console missing note page link")

    prev_admin_token = os.environ.get("ADMIN_TOKEN")
    os.environ["ADMIN_TOKEN"] = "smoke-token"
    try:
        locked_console = web_app.console_index()
        assert_true("Protected" in locked_console and "ADMIN_TOKEN" in locked_console, "console should show gate without token")
        unlocked_console = web_app.console_index(token="smoke-token")
        assert_true("后台访问" in unlocked_console or "任务后台" in unlocked_console, "console missing admin access panel")
        assert_true("上传输入文件" in unlocked_console and "预测与结算任务" in unlocked_console, "console missing merged admin workspace")
        note_gate = web_app.console_note(token="bad-token")
        assert_true("Protected" in note_gate and "ADMIN_TOKEN" in note_gate, "console note should show gate without valid token")
        note_html = web_app.console_note(token="smoke-token")
        assert_true("Note Workspace" in note_html and "返回控制台" in note_html, "console note page did not render")
        denied_buy = web_app.run_llm_buy(token="bad-token", scope_key="central_dirt", run_id="missing")
        assert_true("LLM buy" in denied_buy and "Error" in denied_buy, "run_llm_buy should be denied by admin token")
        denied_run_due = web_app.internal_run_due(token="bad-token")
        assert_true(getattr(denied_run_due, "status_code", 0) == 403, "internal_run_due should reject wrong token")
        ok_run_due = web_app.internal_run_due(token="smoke-token")
        assert_true(getattr(ok_run_due, "status_code", 0) in (200, 500), "internal_run_due should return JSON response")
    finally:
        if prev_admin_token is None:
            os.environ.pop("ADMIN_TOKEN", None)
        else:
            os.environ["ADMIN_TOKEN"] = prev_admin_token

    run_id = pick_latest_run_id()
    assert_true(bool(run_id), "no run_id found in any scope")
    view_html = web_app.view_run(run_id=run_id, scope_key="")
    assert_true("Top5 Predictions" in view_html or "Gemini Policy" in view_html, "view_run did not render run blocks")
    predictor_env = web_app.build_predictor_env("central_turf", run_id, web_app.resolve_run(run_id, "central_turf") or {})
    assert_true(isinstance(predictor_env, dict), "build_predictor_env did not return dict")

    print("smoke_web_app: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke_web_app: FAIL: {exc}")
        sys.exit(1)

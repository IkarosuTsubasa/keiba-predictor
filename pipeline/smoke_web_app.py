import asyncio
import json
import os
import sys

import web_app


def assert_true(cond, message):
    if not cond:
        raise AssertionError(message)


def make_json_request(payload, headers=None):
    body = json.dumps(payload).encode("utf-8")
    header_pairs = list(headers or [])
    header_pairs.append((b"content-type", b"application/json"))
    scope = {
        "type": "http",
        "method": "POST",
        "headers": header_pairs,
    }
    sent = {"done": False}

    async def receive():
        if sent["done"]:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return web_app.Request(scope=scope, receive=receive)


def pick_latest_run():
    for scope in ("central_turf", "central_dirt", "local"):
        row = web_app.resolve_run("", scope)
        if row and row.get("run_id"):
            return scope, row["run_id"]
    return "", ""


def main():
    body = web_app.index()
    assert_true(getattr(body, "status_code", 0) in (302, 307), "index should redirect")
    assert_true(getattr(body, "headers", {}).get("location") == web_app.PUBLIC_BASE_PATH, "index should redirect to /keiba")

    public_home = web_app.llm_today()
    public_html = getattr(public_home, "body", b"").decode("utf-8", errors="ignore")
    assert_true(getattr(public_home, "status_code", 0) == 200, "public home should return 200")
    assert_true("<!doctype html" in public_html.lower(), "public home should render frontend html")
    assert_true("twitter:card" in public_html, "public home should inject meta tags")

    public_race_detail = web_app.public_race_detail_spa("smoke-run-id")
    public_race_detail_html = getattr(public_race_detail, "body", b"").decode("utf-8", errors="ignore")
    assert_true(getattr(public_race_detail, "status_code", 0) == 200, "public race detail should return 200")
    assert_true("<!doctype html" in public_race_detail_html.lower(), "public race detail should render frontend html")

    public_history = web_app.public_history_spa()
    public_history_html = getattr(public_history, "body", b"").decode("utf-8", errors="ignore")
    assert_true(getattr(public_history, "status_code", 0) == 200, "public history should return 200")
    assert_true("<!doctype html" in public_history_html.lower(), "public history should render frontend html")

    console_spa = web_app.console_spa()
    assert_true(getattr(console_spa, "status_code", 0) == 200, "console spa should return 200")
    workspace_spa = web_app.console_workspace_spa()
    assert_true(getattr(workspace_spa, "status_code", 0) == 200, "workspace spa should return 200")

    board_payload = web_app.public_board_api()
    assert_true(getattr(board_payload, "status_code", 200) == 200, "board api should return json")
    board_payload_body = json.loads(getattr(board_payload, "body", b"{}").decode("utf-8", errors="ignore"))
    assert_true("summary_cards" not in board_payload_body, "board api should not expose summary_cards")
    assert_true("all_time_roi" not in board_payload_body, "board api should not expose all_time_roi")
    assert_true("trend" not in board_payload_body, "board api should not expose trend")
    assert_true("llm" not in dict((board_payload_body.get("history") or {})), "board api should not expose history.llm")
    first_board_race = next((dict(item or {}) for item in list(board_payload_body.get("races", []) or []) if isinstance(item, dict)), {})
    if first_board_race:
        assert_true("cards" not in first_board_race, "board api race should not expose llm cards")
        assert_true("actual_result" not in first_board_race, "board api race should not expose actual_result")
        assert_true("condition_predictor_ranking" not in first_board_race, "board api race should not expose condition_predictor_ranking")
    missing_race_detail = web_app.public_race_detail_api("missing-run-id")
    assert_true(getattr(missing_race_detail, "status_code", 0) == 404, "race detail api should return 404 for missing run")
    detail_board = web_app.build_public_board_payload()
    detail_target_date = str(detail_board.get("target_date", "") or "").strip()
    detail_race_row = next(
        (
            dict(item or {})
            for item in web_app._public_consolidate_board_races(
                detail_board.get("races", []),
                detail_board.get("morning_preview"),
            )
            if str((item or {}).get("run_id", "") or "").strip()
        ),
        {},
    )
    detail_run_id = str(detail_race_row.get("run_id", "") or "").strip()
    if detail_run_id:
        race_detail = web_app.public_race_detail_api(detail_run_id, date=detail_target_date)
        race_detail_body = json.loads(getattr(race_detail, "body", b"{}").decode("utf-8", errors="ignore"))
        assert_true(getattr(race_detail, "status_code", 0) == 200, "race detail api should return 200 for known run")
        assert_true(bool((((race_detail_body.get("data") or {}).get("race") or {}).get("run_id"))), "race detail api should include race")
        assert_true(
            isinstance((((race_detail_body.get("data") or {}).get("race") or {}).get("condition_predictor_ranking")), dict),
            "race detail api should include condition_predictor_ranking",
        )

    prev_admin_token = os.environ.get("ADMIN_TOKEN")
    os.environ["ADMIN_TOKEN"] = "smoke-token"
    try:
        auth_denied = web_app.admin_auth_check(web_app.Request(scope={"type": "http", "headers": []}), token="bad-token")
        auth_denied_body = json.loads(getattr(auth_denied, "body", b"{}").decode("utf-8", errors="ignore"))
        assert_true(auth_denied_body.get("valid") is False, "auth-check should reject bad token")

        auth_ok = web_app.admin_auth_check(
            web_app.Request(scope={"type": "http", "headers": [(b"authorization", b"Bearer smoke-token")]}),
        )
        auth_ok_body = json.loads(getattr(auth_ok, "body", b"{}").decode("utf-8", errors="ignore"))
        assert_true(auth_ok_body.get("valid") is True, "auth-check should accept bearer token")
        assert_true(
            str(auth_ok_body.get("console_url") or "").startswith(web_app.CONSOLE_BASE_PATH),
            "auth-check should point to react console",
        )

        jobs_denied = web_app.admin_jobs_api(web_app.Request(scope={"type": "http", "headers": []}), token="bad-token")
        assert_true(getattr(jobs_denied, "status_code", 0) == 403, "admin jobs api should reject wrong token")

        jobs_ok = web_app.admin_jobs_api(
            web_app.Request(scope={"type": "http", "headers": [(b"authorization", b"Bearer smoke-token")]}),
        )
        jobs_ok_body = json.loads(getattr(jobs_ok, "body", b"{}").decode("utf-8", errors="ignore"))
        assert_true(isinstance(jobs_ok_body.get("jobs"), list), "admin jobs api should return jobs")
        assert_true(isinstance(jobs_ok_body.get("summary"), dict), "admin jobs api should return summary")

        admin_runs_denied = web_app.admin_runs_api(web_app.Request(scope={"type": "http", "headers": []}), token="bad-token")
        assert_true(getattr(admin_runs_denied, "status_code", 0) == 403, "admin runs api should reject wrong token")

        admin_runs_ok = web_app.admin_runs_api(
            web_app.Request(scope={"type": "http", "headers": [(b"authorization", b"Bearer smoke-token")]}),
            scope_key="local",
        )
        admin_runs_ok_body = json.loads(getattr(admin_runs_ok, "body", b"{}").decode("utf-8", errors="ignore"))
        assert_true(isinstance(admin_runs_ok_body.get("runs"), list), "admin runs api should return runs list")

        admin_reset_denied = asyncio.run(
            web_app.admin_reset_llm_state_api(
                make_json_request({}, [(b"authorization", b"Bearer bad-token")]),
            ),
        )
        assert_true(getattr(admin_reset_denied, "status_code", 0) == 403, "admin reset api should reject wrong token")

        latest_scope, latest_run_id = pick_latest_run()
        assert_true(bool(latest_scope and latest_run_id), "workspace smoke requires latest run")

        workspace_ok = web_app.admin_workspace_api(
            web_app.Request(scope={"type": "http", "headers": [(b"authorization", b"Bearer smoke-token")]}),
            scope_key=latest_scope,
            run_id=latest_run_id,
        )
        workspace_ok_body = json.loads(getattr(workspace_ok, "body", b"{}").decode("utf-8", errors="ignore"))
        assert_true(getattr(workspace_ok, "status_code", 0) == 200, "workspace api should return 200")
        assert_true(isinstance(workspace_ok_body.get("predictors"), list), "workspace api should return predictors")
        assert_true(isinstance(workspace_ok_body.get("policies"), list), "workspace api should return policies")
        assert_true(isinstance(workspace_ok_body.get("predictor_overview"), dict), "workspace api should return predictor overview")
        assert_true(isinstance(workspace_ok_body.get("portfolio_summaries"), list), "workspace api should return portfolio summaries")
        assert_true(isinstance(workspace_ok_body.get("run_result_summary"), list), "workspace api should return run result summary")
        assert_true(isinstance(workspace_ok_body.get("odds_snapshots"), dict), "workspace api should return odds snapshots")

        workspace_run_llm_denied = asyncio.run(
            web_app.admin_workspace_run_llm_buy_api(
                make_json_request({"scope_key": latest_scope, "run_id": latest_run_id}, [(b"authorization", b"Bearer bad-token")]),
            ),
        )
        assert_true(getattr(workspace_run_llm_denied, "status_code", 0) == 403, "workspace run_llm should reject wrong token")

        workspace_run_all_denied = asyncio.run(
            web_app.admin_workspace_run_all_llm_buy_api(
                make_json_request({"scope_key": latest_scope, "run_id": latest_run_id}, [(b"authorization", b"Bearer bad-token")]),
            ),
        )
        assert_true(getattr(workspace_run_all_denied, "status_code", 0) == 403, "workspace run_all_llm should reject wrong token")

        workspace_topup_denied = asyncio.run(
            web_app.admin_workspace_topup_all_llm_budget_api(
                make_json_request({"scope_key": latest_scope, "run_id": latest_run_id}, [(b"authorization", b"Bearer bad-token")]),
            ),
        )
        assert_true(getattr(workspace_topup_denied, "status_code", 0) == 403, "workspace topup should reject wrong token")

        workspace_record_denied = asyncio.run(
            web_app.admin_workspace_record_predictor_api(
                make_json_request(
                    {"scope_key": latest_scope, "run_id": latest_run_id, "top1": "1", "top2": "2", "top3": "3"},
                    [(b"authorization", b"Bearer bad-token")],
                ),
            ),
        )
        assert_true(getattr(workspace_record_denied, "status_code", 0) == 403, "workspace record_predictor should reject wrong token")

        workspace_record_missing = asyncio.run(
            web_app.admin_workspace_record_predictor_api(
                make_json_request({"scope_key": latest_scope, "run_id": latest_run_id}, [(b"authorization", b"Bearer smoke-token")]),
            ),
        )
        assert_true(getattr(workspace_record_missing, "status_code", 0) == 400, "workspace record_predictor should require top1-3")

        workspace_missing_run = asyncio.run(
            web_app.admin_workspace_run_llm_buy_api(
                make_json_request({"scope_key": latest_scope, "run_id": "missing-run"}, [(b"authorization", b"Bearer smoke-token")]),
            ),
        )
        assert_true(getattr(workspace_missing_run, "status_code", 0) == 404, "workspace run_llm should return 404 for missing run")

        denied_run_due = web_app.internal_run_due(
            web_app.Request(scope={"type": "http", "headers": [(b"authorization", b"Bearer bad-token")]}),
        )
        assert_true(getattr(denied_run_due, "status_code", 0) == 403, "internal_run_due should reject wrong token")
        ok_run_due = web_app.internal_run_due(
            web_app.Request(scope={"type": "http", "headers": [(b"authorization", b"Bearer smoke-token")]}),
        )
        assert_true(getattr(ok_run_due, "status_code", 0) in (200, 500), "internal_run_due should return json response")
    finally:
        if prev_admin_token is None:
            os.environ.pop("ADMIN_TOKEN", None)
        else:
            os.environ["ADMIN_TOKEN"] = prev_admin_token

    scope_key, run_id = pick_latest_run()
    assert_true(bool(run_id), "no run_id found in any scope")
    predictor_env = web_app.build_predictor_env(scope_key, run_id, web_app.resolve_run(run_id, scope_key) or {})
    assert_true(isinstance(predictor_env, dict), "build_predictor_env should return dict")

    print("smoke_web_app: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke_web_app: FAIL: {exc}")
        sys.exit(1)

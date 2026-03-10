import sys

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
    # 1) index page
    body = web_app.index()
    assert_true("Run Pipeline" in body, "index missing Run Pipeline block")
    assert_true('action="/view_run"' in body, "index missing view_run form")

    # 2) view_run route with real run_id
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

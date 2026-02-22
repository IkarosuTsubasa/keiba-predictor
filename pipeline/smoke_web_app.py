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
    assert_true("Top5 Predictions" in view_html or "Bet Plan" in view_html, "view_run did not render run blocks")

    # 3) update_bet_plan route (validation path, no external odds fetch)
    update_html = web_app.update_bet_plan(race_id="", scope_key="", budget="", style="")
    assert_true("Enter Run ID or Race ID to update." in update_html, "update_bet_plan validation text mismatch")

    # 4) record_pipeline route (validation path)
    record_html = web_app.record_pipeline(
        run_id="",
        scope_key="",
        profit="",
        note="",
        top1="",
        top2="",
        top3="",
    )
    assert_true("Actual 1st/2nd/3rd are required." in record_html, "record_pipeline validation text mismatch")

    print("smoke_web_app: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke_web_app: FAIL: {exc}")
        sys.exit(1)

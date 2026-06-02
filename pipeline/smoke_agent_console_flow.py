import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import web_app
from race_job_store import (
    compute_initial_status,
    create_job,
    get_job,
    initialize_job_step_fields,
    load_jobs,
    scan_due_diagnostics,
    scan_due_jobs,
    set_job_step_state,
    update_job,
)
from v5_remote_tasks import find_latest_task_for_job, update_task
from web_admin import task_routes

JST_OFFSET = timedelta(hours=9)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def jst_now():
    return datetime.utcnow() + JST_OFFSET


def run_due_once(base_dir):
    return task_routes.run_due_jobs_once(
        base_dir=base_dir,
        scan_due_race_jobs=scan_due_jobs,
        scan_due_race_job_diagnostics=scan_due_diagnostics,
        load_race_jobs=load_jobs,
        update_race_job=update_job,
        compute_race_job_initial_status=compute_initial_status,
    )


def fake_dispatch_agent_prediction(base_dir, task, job):
    task_id = str((task or {}).get("task_id", "") or "").strip()
    race_id = str((job or {}).get("race_id", "") or "").strip()
    assert_true(bool(task_id), "remote task should have task_id")

    def mark_dispatched(row, now_text):
        row["status"] = "dispatched"
        row["attempt"] = int(row.get("attempt", 0) or 0) + 1
        row["started_at"] = str(row.get("started_at", "") or now_text)
        row["workflow_dispatch_ref"] = "smoke"
        row["error_message"] = ""
        row["result_summary"] = {"dispatch_http_status": 204}

    update_task(base_dir, task_id, mark_dispatched)
    return {
        "task_id": task_id,
        "workflow": "agent-prediction-remote.yml",
        "ref": "smoke",
        "race_id": race_id,
        "race_url": f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
        "callback_url": f"https://example.test/keiba/internal/v5_tasks/{task_id}/callback",
        "dispatch_http_status": 204,
    }


def create_agent_job(base_dir):
    off_time = jst_now() - timedelta(minutes=45)
    return create_job(
        base_dir,
        race_id="202605021211",
        scope_key="central_turf",
        race_name="スモークテスト",
        location="東京",
        race_date=off_time.date().isoformat(),
        scheduled_off_time=off_time.strftime("%Y-%m-%dT%H:%M:%S"),
        target_surface="芝",
        target_distance="芝1600m",
        target_track_condition="良",
        lead_minutes=60,
        run_kind="final_prediction",
        job_source="agent_prediction",
        notes="smoke",
    )


def save_agent_prediction_via_callback(base_dir, job, task):
    race_id = str(job.get("race_id", "") or "").strip()
    prediction_payload = {
        "race_id": race_id,
        "race_name": str(job.get("race_name", "") or "").strip(),
        "race_date": str(job.get("race_date", "") or "").strip(),
        "venue": str(job.get("location", "") or "").strip(),
        "generated_at": jst_now().isoformat(timespec="seconds"),
        "summary": "スモークテスト用のAI予測です。",
        "top_candidates": [
            {"rank": 1, "horse_no": "1", "horse_name": "テストホースA", "memo": "軸候補"},
            {"rank": 2, "horse_no": "2", "horse_name": "テストホースB", "memo": "相手候補"},
            {"rank": 3, "horse_no": "3", "horse_name": "テストホースC", "memo": "押さえ"},
        ],
        "bets": [
            {"type": "馬連", "selection": "1-2", "stake": 100},
            {"type": "ワイド", "selection": "1-3", "stake": 100},
        ],
    }
    original_base_dir = web_app.BASE_DIR
    web_app.BASE_DIR = Path(base_dir)
    try:
        saved = web_app._apply_remote_agent_prediction_result(
            task,
            {"status": "succeeded", "result": {"prediction": prediction_payload}, "summary": {"source": "smoke"}},
        )
        update_task(
            base_dir,
            str(task.get("task_id", "") or "").strip(),
            lambda row, now_text: row.update(
                {
                    "status": "succeeded",
                    "finished_at": now_text,
                    "error_message": "",
                    "result_path": str(saved.get("result_path", "") or ""),
                    "result_summary": {"source": "smoke"},
                }
            ),
        )
        web_app._promote_job_after_remote_agent_prediction(
            job_id=str(job.get("job_id", "") or "").strip(),
            task_id=str(task.get("task_id", "") or "").strip(),
            race_id=str(saved.get("race_id", "") or "").strip(),
            log_output="smoke callback",
        )
        return saved
    finally:
        web_app.BASE_DIR = original_base_dir


def save_existing_agent_result(base_dir, race_id):
    result_dir = Path(base_dir).resolve().parent / "data" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{race_id}.json"
    result_payload = {
        "race_id": race_id,
        "finish_order": [
            {"rank": 1, "horse_no": "1", "horse_name": "テストホースA"},
            {"rank": 2, "horse_no": "2", "horse_name": "テストホースB"},
            {"rank": 3, "horse_no": "3", "horse_name": "テストホースC"},
        ],
        "payouts": [],
    }
    result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result_path


def main():
    with TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir) / "pipeline"
        base_dir.mkdir(parents=True, exist_ok=True)

        job = create_agent_job(base_dir)
        job_id = str(job.get("job_id", "") or "").strip()
        race_id = str(job.get("race_id", "") or "").strip()
        assert_true(job.get("status") == "scheduled", "agent job should start scheduled")

        original_dispatch = task_routes._dispatch_agent_prediction_task
        task_routes._dispatch_agent_prediction_task = fake_dispatch_agent_prediction
        try:
            dispatch_summary = run_due_once(base_dir)
        finally:
            task_routes._dispatch_agent_prediction_task = original_dispatch

        assert_true(dispatch_summary.get("queued_count") == 1, "run_due should queue due agent job")
        assert_true(dispatch_summary.get("agent_dispatched_count") == 1, "run_due should dispatch agent prediction")
        dispatched_job = get_job(base_dir, job_id) or {}
        assert_true(dispatched_job.get("status") == "processing_agent_prediction", "job should wait for agent callback")
        assert_true(dispatched_job.get("predictor_status") == "running", "predictor step should be running")

        task = find_latest_task_for_job(base_dir, job_id)
        assert_true(bool(task), "remote task should be created")
        assert_true(task.get("status") == "dispatched", "remote task should be dispatched")

        saved_prediction = save_agent_prediction_via_callback(base_dir, dispatched_job, task)
        prediction_path = Path(str(saved_prediction.get("prediction_path", "") or ""))
        assert_true(prediction_path.exists(), "callback should save prediction json")
        ready_job = get_job(base_dir, job_id) or {}
        assert_true(ready_job.get("status") == "agent_prediction_ready", "callback should mark prediction ready")
        assert_true(ready_job.get("predictor_status") == "succeeded", "predictor step should succeed")

        result_path = save_existing_agent_result(base_dir, race_id)
        assert_true(result_path.exists(), "smoke result should be saved before result run")

        result_summary = run_due_once(base_dir)
        assert_true(result_summary.get("agent_result_count") == 1, "run_due should settle saved agent result")
        settled_job = get_job(base_dir, job_id) or {}
        assert_true(settled_job.get("status") == "settled", "agent job should be settled")
        assert_true(settled_job.get("settlement_status") == "succeeded", "settlement step should succeed")
        assert_true(settled_job.get("actual_top1") == "", "pre-saved smoke result should not require fetched top3")
        cleanup = dict(result_summary.get("cleanup") or {})
        assert_true(cleanup.get("reason") == "cleanup_base_mismatch", "temp smoke should skip real cleanup")

    print("smoke_agent_console_flow: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke_agent_console_flow: FAIL: {exc}")
        sys.exit(1)

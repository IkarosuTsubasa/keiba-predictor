import json
import json
import os
import threading
import traceback
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path


def remote_v5_enabled():
    raw = os.environ.get("PIPELINE_REMOTE_V5_ENABLED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def remote_predictor_auto_continue_enabled():
    raw = os.environ.get("PIPELINE_REMOTE_PREDICTORS_AUTO_CONTINUE", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def llm_buy_enabled():
    raw = os.environ.get("PIPELINE_ENABLE_LLM_BUY", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def append_job_process_log_entry(row, step, code, output):
    payload = {}
    raw = str((row or {}).get("last_process_output", "") or "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    process_log = list(payload.get("process_log", []) or [])
    process_log.append(
        {
            "step": str(step or "").strip(),
            "code": int(code) if str(code).strip("-").isdigit() else code,
            "output": str(output or "").strip(),
        }
    )
    payload["process_log"] = process_log
    return json.dumps(payload, ensure_ascii=False, indent=2)


def remote_v5_bundle_zip_bytes(task):
    task_row = dict(task or {})
    bundle = BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for archive_name, src_path in dict(task_row.get("bundle_files", {}) or {}).items():
            path = Path(str(src_path or "").strip())
            if not archive_name or not path.exists():
                continue
            zf.write(path, arcname=str(archive_name))
        meta_bytes = json.dumps(dict(task_row.get("bundle_meta", {}) or {}), ensure_ascii=False, indent=2).encode("utf-8")
        zf.writestr("task_meta.json", meta_bytes)
    bundle.seek(0)
    return bundle.getvalue()


def promote_job_after_remote_v5(
    *,
    base_dir,
    job_id,
    run_id,
    task_id,
    log_output,
    update_race_job,
    initialize_race_job_step_fields,
    set_race_job_step_state,
):
    def mutate(row, now_text):
        row.update(initialize_race_job_step_fields(row))
        current_status = str(row.get("status", "") or "").strip().lower()
        current_task_id = str(row.get("current_v5_task_id", "") or "").strip()
        if current_status not in ("waiting_v5", "queued_policy", "processing_policy"):
            return
        if current_task_id and current_task_id != str(task_id or "").strip():
            return
        row["status"] = "queued_policy" if llm_buy_enabled() else "ready"
        row["current_run_id"] = str(run_id or "").strip()
        row["current_v5_task_id"] = str(task_id or "").strip()
        row["error_message"] = ""
        row["last_process_output"] = append_job_process_log_entry(
            row,
            "predictors_remote_callback",
            0,
            log_output,
        )
        set_race_job_step_state(row, "predictor", "succeeded", now_text)
        set_race_job_step_state(row, "policy", "queued" if llm_buy_enabled() else "idle")
        if not llm_buy_enabled():
            row["ready_at"] = now_text

    return update_race_job(base_dir, job_id, mutate)


def promote_job_after_remote_morning(
    *,
    base_dir,
    job_id,
    run_id,
    task_id,
    log_output,
    update_race_job,
    initialize_race_job_step_fields,
    set_race_job_step_state,
):
    def mutate(row, now_text):
        row.update(initialize_race_job_step_fields(row))
        current_status = str(row.get("status", "") or "").strip().lower()
        current_task_id = str(row.get("current_morning_task_id", "") or "").strip()
        if current_status not in ("queued_morning", "processing_morning"):
            return
        if current_task_id and current_task_id != str(task_id or "").strip():
            return
        row["status"] = "scheduled"
        row["morning_ready_at"] = now_text
        row["morning_run_id"] = str(run_id or "").strip()
        row["current_morning_task_id"] = ""
        row["error_message"] = ""
        row["last_process_output"] = append_job_process_log_entry(
            row,
            "morning_remote_callback",
            0,
            log_output,
        )
        set_race_job_step_state(row, "morning", "succeeded", now_text)

    return update_race_job(base_dir, job_id, mutate)


def auto_continue_remote_policy(*, base_dir, job_id):
    target_job_id = str(job_id or "").strip()
    if not target_job_id or not llm_buy_enabled():
        return

    def _runner():
        try:
            from race_job_runner import process_race_job

            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "remote_predictors_auto_continue_start",
                        "job_id": target_job_id,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            summary = process_race_job(base_dir, target_job_id)
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "remote_predictors_auto_continue_done",
                        "job_id": target_job_id,
                        "run_id": str((summary or {}).get("run_id", "") or "").strip(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            try:
                from race_job_runner import fail_race_job

                fail_race_job(base_dir, target_job_id, str(exc))
            except Exception:
                pass
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "remote_predictors_auto_continue_error",
                        "job_id": target_job_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    threading.Thread(target=_runner, name=f"remote-policy-{target_job_id}", daemon=True).start()

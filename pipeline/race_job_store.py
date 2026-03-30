import json
import re
from datetime import datetime, timedelta
from pathlib import Path


JST_OFFSET = timedelta(hours=9)
JOB_STORE_NAME = "race_jobs.json"
ARTIFACT_DIR_NAME = "race_job_artifacts"
REQUIRED_ARTIFACT_TYPES = ("kachiuma", "shutuba")
STATUS_FLOW = (
    "uploaded",
    "waiting_input_info",
    "scheduled",
    "queued_process",
    "processing",
    "waiting_v5",
    "queued_policy",
    "processing_policy",
    "ready",
    "queued_settle",
    "settling",
    "settled",
    "failed",
)
JOB_STEP_NAMES = ("odds", "predictor", "policy", "settlement")
JOB_STEP_STATE_FLOW = ("idle", "queued", "running", "succeeded", "failed")


def _job_step_field(step_name, suffix):
    return f"{step_name}_{suffix}"


def initialize_job_step_fields(job):
    row = job if isinstance(job, dict) else dict(job or {})
    row.setdefault("race_name", "")
    row.setdefault("race_number", "")
    row.setdefault("meta_source_url", "")
    row.setdefault("meta_fetched_at", "")
    row.setdefault("meta_error", "")
    row.setdefault("meta_retry_count", "0")
    row.setdefault("ntfy_notify_status", "")
    row.setdefault("ntfy_notify_run_id", "")
    row.setdefault("ntfy_notify_engine", "")
    row.setdefault("ntfy_notified_at", "")
    row.setdefault("ntfy_notify_error", "")
    for step_name in JOB_STEP_NAMES:
        row.setdefault(_job_step_field(step_name, "status"), "idle")
        row.setdefault(_job_step_field(step_name, "started_at"), "")
        row.setdefault(_job_step_field(step_name, "finished_at"), "")
        row.setdefault(_job_step_field(step_name, "error"), "")
    return row


def set_job_step_state(job, step_name, state, now_text="", error_text=""):
    row = initialize_job_step_fields(job)
    if step_name not in JOB_STEP_NAMES:
        return row
    state_text = str(state or "").strip().lower() or "idle"
    if state_text not in JOB_STEP_STATE_FLOW:
        state_text = "idle"
    row[_job_step_field(step_name, "status")] = state_text
    if state_text == "idle":
        row[_job_step_field(step_name, "started_at")] = ""
        row[_job_step_field(step_name, "finished_at")] = ""
        row[_job_step_field(step_name, "error")] = ""
        return row
    if state_text == "queued":
        row[_job_step_field(step_name, "started_at")] = ""
        row[_job_step_field(step_name, "finished_at")] = ""
        row[_job_step_field(step_name, "error")] = ""
        return row
    if state_text == "running":
        if now_text:
            row[_job_step_field(step_name, "started_at")] = (
                row.get(_job_step_field(step_name, "started_at"), "") or now_text
            )
        row[_job_step_field(step_name, "finished_at")] = ""
        row[_job_step_field(step_name, "error")] = ""
        return row
    if now_text:
        row[_job_step_field(step_name, "started_at")] = (
            row.get(_job_step_field(step_name, "started_at"), "") or now_text
        )
        row[_job_step_field(step_name, "finished_at")] = now_text
    if state_text == "failed":
        row[_job_step_field(step_name, "error")] = str(error_text or "").strip()
    else:
        row[_job_step_field(step_name, "error")] = ""
    return row


def hydrate_job_step_states(job):
    raw_row = dict(job or {})
    explicit = any(
        str(raw_row.get(_job_step_field(step_name, "status"), "") or "").strip()
        for step_name in JOB_STEP_NAMES
    )
    row = initialize_job_step_fields(raw_row)
    legacy_status = str(row.get("status", "") or "").strip().lower()
    current_run_id = str(row.get("current_run_id", "") or "").strip()
    actual_names_ready = all(str(row.get(name, "") or "").strip() for name in ("actual_top1", "actual_top2", "actual_top3"))
    if not explicit:
        if legacy_status == "queued_process":
            row = set_job_step_state(row, "odds", "queued")
        elif legacy_status == "processing":
            row = set_job_step_state(row, "odds", "running", row.get("processing_started_at", ""))
        elif legacy_status in ("ready", "queued_settle", "settling", "settled"):
            for step_name in ("odds", "predictor", "policy"):
                row = set_job_step_state(row, step_name, "succeeded", row.get("ready_at", ""))
        elif legacy_status == "waiting_v5":
            row = set_job_step_state(row, "odds", "succeeded", row.get("processing_started_at", "") or row.get("updated_at", ""))
            row = set_job_step_state(row, "predictor", "running", row.get("updated_at", "") or row.get("processing_started_at", ""))
            row = set_job_step_state(row, "policy", "idle")
        elif legacy_status == "queued_policy":
            row = set_job_step_state(row, "odds", "succeeded", row.get("processing_started_at", "") or row.get("updated_at", ""))
            row = set_job_step_state(row, "predictor", "succeeded", row.get("updated_at", "") or row.get("processing_started_at", ""))
            row = set_job_step_state(row, "policy", "queued")
        elif legacy_status == "processing_policy":
            row = set_job_step_state(row, "odds", "succeeded", row.get("processing_started_at", "") or row.get("updated_at", ""))
            row = set_job_step_state(row, "predictor", "succeeded", row.get("updated_at", "") or row.get("processing_started_at", ""))
            row = set_job_step_state(row, "policy", "running", row.get("updated_at", "") or row.get("processing_started_at", ""))
        elif legacy_status == "failed":
            if str(row.get("settling_started_at", "") or "").strip():
                for step_name in ("odds", "predictor", "policy"):
                    if current_run_id:
                        row = set_job_step_state(step_name=step_name, job=row, state="succeeded", now_text=row.get("ready_at", ""))
                row = set_job_step_state(
                    row,
                    "settlement",
                    "failed",
                    row.get("updated_at", "") or row.get("settling_started_at", ""),
                    row.get("error_message", ""),
                )
            elif current_run_id:
                row = set_job_step_state(row, "odds", "succeeded", row.get("ready_at", ""))
                row = set_job_step_state(row, "predictor", "succeeded", row.get("ready_at", ""))
                row = set_job_step_state(
                    row,
                    "policy",
                    "failed",
                    row.get("updated_at", "") or row.get("ready_at", ""),
                    row.get("error_message", ""),
                )
            elif str(row.get("processing_started_at", "") or "").strip():
                row = set_job_step_state(
                    row,
                    "odds",
                    "failed",
                    row.get("updated_at", "") or row.get("processing_started_at", ""),
                    row.get("error_message", ""),
                )
        if legacy_status == "queued_settle":
            row = set_job_step_state(row, "settlement", "queued")
        elif legacy_status == "settling":
            row = set_job_step_state(row, "settlement", "running", row.get("settling_started_at", ""))
        elif legacy_status == "settled":
            row = set_job_step_state(row, "settlement", "succeeded", row.get("settled_at", ""))
        elif legacy_status == "failed" and actual_names_ready and str(row.get("settling_started_at", "") or "").strip():
            row = set_job_step_state(
                row,
                "settlement",
                "failed",
                row.get("updated_at", "") or row.get("settling_started_at", ""),
                row.get("error_message", ""),
            )
    return row


def derive_job_display_state(job):
    row = hydrate_job_step_states(job)
    legacy_status = str(row.get("status", "") or "").strip().lower()
    if legacy_status == "waiting_v5":
        return {"code": "waiting_v5", "label": "等待远程预测", "tone": "active"}
    if legacy_status == "queued_policy":
        return {"code": "queued_policy", "label": "等待 LLM", "tone": "active"}
    if legacy_status == "processing_policy":
        return {"code": "processing_policy", "label": "LLM 处理中", "tone": "active"}
    if row.get("settlement_status") == "succeeded":
        return {"code": "settled", "label": "已结算", "tone": "good"}
    if row.get("settlement_status") == "running":
        return {"code": "settling", "label": "结算中", "tone": "active"}
    if row.get("settlement_status") == "queued":
        return {"code": "queued_settle", "label": "待结算", "tone": "active"}
    if legacy_status == "failed":
        for step_name, label in (
            ("settlement", "结算失败"),
            ("policy", "LLM失败"),
            ("predictor", "预测失败"),
            ("odds", "赔率失败"),
        ):
            if row.get(_job_step_field(step_name, "status")) == "failed":
                return {"code": f"failed_{step_name}", "label": label, "tone": "danger"}
        return {"code": "failed", "label": "失败", "tone": "danger"}
    if row.get("policy_status") == "running":
        return {"code": "policy_running", "label": "LLM处理中", "tone": "active"}
    if row.get("predictor_status") == "running":
        return {"code": "predictor_running", "label": "预测中", "tone": "active"}
    if row.get("odds_status") in ("queued", "running"):
        return {"code": "odds_running", "label": "赔率处理中", "tone": "active"}
    if row.get("policy_status") == "succeeded":
        return {"code": "ready", "label": "处理完成", "tone": "good"}
    if row.get("predictor_status") == "succeeded":
        return {"code": "predictor_ready", "label": "预测已生成", "tone": "good"}
    if legacy_status == "waiting_input_info":
        return {"code": "waiting_input_info", "label": "情報補完待ち", "tone": "muted"}
    if legacy_status == "scheduled":
        return {"code": "scheduled", "label": "已排程", "tone": "muted"}
    if legacy_status == "uploaded":
        return {"code": "uploaded", "label": "已上传", "tone": "muted"}
    return {"code": legacy_status or "unknown", "label": legacy_status or "-", "tone": "muted"}


def _shared_dir(base_dir):
    path = Path(base_dir) / "data" / "_shared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jobs_path(base_dir):
    return _shared_dir(base_dir) / JOB_STORE_NAME


def _artifact_dir(base_dir, job_id):
    path = _shared_dir(base_dir) / ARTIFACT_DIR_NAME / str(job_id or "").strip()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jst_now():
    return datetime.utcnow() + JST_OFFSET


def _parse_dt(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _dt_text(value):
    if not value:
        return ""
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def _safe_filename(name):
    base = Path(str(name or "").strip()).name or "upload.csv"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "upload.csv"
    return base


def load_jobs(base_dir):
    path = _jobs_path(base_dir)
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    rows = [dict(item) for item in rows if isinstance(item, dict)]
    rows.sort(
        key=lambda item: (
            str(item.get("scheduled_off_time", "") or ""),
            str(item.get("race_date", "") or ""),
            str(item.get("race_id", "") or ""),
            str(item.get("created_at", "") or ""),
        )
    )
    return [hydrate_job_step_states(item) for item in rows]


def save_jobs(base_dir, jobs):
    rows = [dict(item) for item in list(jobs or []) if isinstance(item, dict)]
    path = _jobs_path(base_dir)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _build_job_id(race_id):
    prefix = _jst_now().strftime("%Y%m%d_%H%M%S")
    suffix = re.sub(r"\D", "", str(race_id or "")) or "race"
    return f"{prefix}_{suffix}"


def _artifact_index(artifacts):
    out = {}
    for item in list(artifacts or []):
        if not isinstance(item, dict):
            continue
        art_type = str(item.get("artifact_type", "") or "").strip().lower()
        if art_type:
            out[art_type] = dict(item)
    return out


def compute_initial_status(job):
    row = dict(job or {})
    artifact_map = _artifact_index(row.get("artifacts", []))
    has_required = all(artifact_map.get(name) for name in REQUIRED_ARTIFACT_TYPES)
    scope_key = str(row.get("scope_key", "") or "").strip()
    off_dt = _parse_dt(row.get("scheduled_off_time", ""))
    target_distance = parse_int_text(row.get("target_distance", ""))
    track_condition = str(row.get("target_track_condition", "") or "").strip()
    if has_required and scope_key and off_dt and target_distance and track_condition:
        return "scheduled"
    if has_required:
        return "waiting_input_info"
    return "uploaded"


def parse_int_text(value):
    text = str(value or "").strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _derive_process_after_dt(job):
    row = dict(job or {})
    off_dt = _parse_dt(row.get("scheduled_off_time", ""))
    if off_dt is None:
        return None
    try:
        lead = max(0, int(row.get("lead_minutes", 0) or 0))
    except (TypeError, ValueError):
        lead = 30
    return off_dt - timedelta(minutes=lead)


def save_artifact(base_dir, job_id, artifact_type, original_name, content):
    now = _jst_now()
    safe_name = _safe_filename(original_name)
    filename = f"{artifact_type}_{now.strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    path = _artifact_dir(base_dir, job_id) / filename
    data = content if isinstance(content, bytes) else bytes(content or b"")
    path.write_bytes(data)
    return {
        "artifact_type": str(artifact_type or "").strip().lower(),
        "original_name": str(original_name or "").strip(),
        "stored_name": path.name,
        "stored_path": str(path),
        "size_bytes": len(data),
        "uploaded_at": _dt_text(now),
    }


def create_job(
    base_dir,
    *,
    race_id="",
    scope_key="",
    race_name="",
    location="",
    race_date="",
    scheduled_off_time="",
    target_surface="",
    target_distance="",
    target_track_condition="",
    lead_minutes=30,
    notes="",
    artifacts=None,
):
    off_dt = _parse_dt(scheduled_off_time)
    try:
        lead = max(0, int(lead_minutes or 0))
    except (TypeError, ValueError):
        lead = 30
    process_after_dt = off_dt - timedelta(minutes=lead) if off_dt else None
    created_at = _jst_now()
    job = {
        "job_id": _build_job_id(race_id),
        "race_id": str(race_id or "").strip(),
        "scope_key": str(scope_key or "").strip(),
        "race_name": str(race_name or "").strip(),
        "race_number": "",
        "location": str(location or "").strip(),
        "race_date": str(race_date or "").strip(),
        "scheduled_off_time": _dt_text(off_dt),
        "process_after_time": _dt_text(process_after_dt),
        "target_surface": str(target_surface or "").strip(),
        "target_distance": str(target_distance or "").strip(),
        "target_track_condition": str(target_track_condition or "").strip(),
        "lead_minutes": lead,
        "status": "uploaded",
        "notes": str(notes or "").strip(),
        "artifacts": [dict(item) for item in list(artifacts or []) if isinstance(item, dict)],
        "created_at": _dt_text(created_at),
        "updated_at": _dt_text(created_at),
        "queued_process_at": "",
        "processing_started_at": "",
        "ready_at": "",
        "queued_settle_at": "",
        "settling_started_at": "",
        "settled_at": "",
        "current_run_id": "",
        "current_v5_task_id": "",
        "actual_top1": "",
        "actual_top2": "",
        "actual_top3": "",
        "error_message": "",
        "last_process_output": "",
        "last_settlement_output": "",
        "meta_source_url": "",
        "meta_fetched_at": "",
        "meta_error": "",
        "meta_retry_count": "0",
        "ntfy_notify_status": "",
        "ntfy_notify_run_id": "",
        "ntfy_notify_engine": "",
        "ntfy_notified_at": "",
        "ntfy_notify_error": "",
    }
    job = initialize_job_step_fields(job)
    job["status"] = compute_initial_status(job)
    jobs = load_jobs(base_dir)
    jobs.append(job)
    save_jobs(base_dir, jobs)
    return job


def get_job(base_dir, job_id):
    for job in load_jobs(base_dir):
        if str(job.get("job_id", "")).strip() == str(job_id or "").strip():
            return job
    return None


def update_job(base_dir, job_id, mutate_fn):
    jobs = load_jobs(base_dir)
    updated = None
    now_text = _dt_text(_jst_now())
    for idx, job in enumerate(jobs):
        if str(job.get("job_id", "")).strip() != str(job_id or "").strip():
            continue
        current = initialize_job_step_fields(job)
        mutate_fn(current, now_text)
        current = hydrate_job_step_states(current)
        current["updated_at"] = now_text
        jobs[idx] = current
        updated = current
        break
    if updated is not None:
        save_jobs(base_dir, jobs)
    return updated


def scan_due_jobs(base_dir, now_text=""):
    now_dt = _parse_dt(now_text) or _jst_now()
    jobs = load_jobs(base_dir)
    changed = []
    dirty = False
    for idx, job in enumerate(jobs):
        job = initialize_job_step_fields(job)
        status = str(job.get("status", "")).strip().lower()
        expected_status = compute_initial_status(job)
        if status == "uploaded" and expected_status in ("waiting_input_info", "scheduled"):
            job["status"] = expected_status
            status = expected_status
            dirty = True
        process_dt = _parse_dt(job.get("process_after_time", ""))
        if process_dt is None:
            derived_process_dt = _derive_process_after_dt(job)
            if derived_process_dt is not None:
                process_dt = derived_process_dt
                job["process_after_time"] = _dt_text(derived_process_dt)
                dirty = True
        if status != "scheduled" or process_dt is None or process_dt > now_dt:
            if dirty:
                jobs[idx] = job
            continue
        job["status"] = "queued_process"
        job["queued_process_at"] = _dt_text(now_dt)
        job = set_job_step_state(job, "odds", "queued")
        job = set_job_step_state(job, "predictor", "idle")
        job = set_job_step_state(job, "policy", "idle")
        job["updated_at"] = _dt_text(now_dt)
        jobs[idx] = job
        dirty = True
        changed.append(dict(job))
    if dirty:
        save_jobs(base_dir, jobs)
    return changed


def scan_due_diagnostics(base_dir, now_text=""):
    now_dt = _parse_dt(now_text) or _jst_now()
    rows = []
    for job in load_jobs(base_dir):
        row = initialize_job_step_fields(job)
        status = str(row.get("status", "") or "").strip().lower()
        artifact_map = _artifact_index(row.get("artifacts", []))
        has_required = all(artifact_map.get(name) for name in REQUIRED_ARTIFACT_TYPES)
        expected_status = compute_initial_status(row)
        effective_status = expected_status if status == "uploaded" and expected_status in ("waiting_input_info", "scheduled") else status
        process_dt = _parse_dt(row.get("process_after_time", ""))
        derived_process_dt = _derive_process_after_dt(row) if process_dt is None else process_dt
        reason = "eligible"
        if effective_status != "scheduled":
            if not has_required:
                reason = "missing_artifacts"
            elif effective_status == "waiting_input_info":
                reason = "wait_input_info"
            elif not _parse_dt(row.get("scheduled_off_time", "")):
                reason = "missing_off_time"
            else:
                reason = f"status_{status or 'unknown'}"
        elif derived_process_dt is None:
            reason = "missing_process_after"
        elif derived_process_dt > now_dt:
            reason = "wait_process_after"
        rows.append(
            {
                "job_id": str(row.get("job_id", "") or "").strip(),
                "status": status,
                "effective_status": effective_status,
                "race_id": str(row.get("race_id", "") or "").strip(),
                "scheduled_off_time": str(row.get("scheduled_off_time", "") or "").strip(),
                "process_after_time": _dt_text(derived_process_dt) if derived_process_dt else "",
                "has_required_artifacts": bool(has_required),
                "reason": reason,
            }
        )
    return rows


def apply_job_action(base_dir, job_id, action):
    action_key = str(action or "").strip().lower()

    def mutate(job, now_text):
        job.update(initialize_job_step_fields(job))
        current = str(job.get("status", "")).strip().lower()
        if action_key == "start_processing":
            if current in ("queued_process", "scheduled"):
                job["status"] = "processing"
                job["processing_started_at"] = now_text
                job["error_message"] = ""
                set_job_step_state(job, "odds", "running", now_text)
                set_job_step_state(job, "predictor", "idle")
                set_job_step_state(job, "policy", "idle")
            elif current == "queued_policy":
                job["status"] = "processing_policy"
                job["error_message"] = ""
                set_job_step_state(job, "policy", "running", now_text)
        elif action_key == "mark_ready":
            if current in ("processing", "queued_process", "processing_policy", "queued_policy"):
                job["status"] = "ready"
                job["ready_at"] = now_text
                for step_name in ("odds", "predictor", "policy"):
                    if str(job.get(_job_step_field(step_name, "status"), "") or "").strip().lower() in (
                        "queued",
                        "running",
                        "idle",
                    ):
                        set_job_step_state(job, step_name, "succeeded", now_text)
        elif action_key == "queue_settle":
            if current in ("ready", "settled"):
                job["status"] = "queued_settle"
                job["queued_settle_at"] = now_text
                set_job_step_state(job, "settlement", "queued")
        elif action_key == "start_settling":
            if current in ("queued_settle", "ready"):
                job["status"] = "settling"
                job["settling_started_at"] = now_text
                set_job_step_state(job, "settlement", "running", now_text)
        elif action_key == "mark_settled":
            if current in ("settling", "queued_settle", "ready"):
                job["status"] = "settled"
                job["settled_at"] = now_text
                set_job_step_state(job, "settlement", "succeeded", now_text)
        elif action_key in ("reset_schedule", "force_reset"):
            job["status"] = compute_initial_status(job)
            job["queued_process_at"] = ""
            job["processing_started_at"] = ""
            job["ready_at"] = ""
            job["queued_settle_at"] = ""
            job["settling_started_at"] = ""
            job["settled_at"] = ""
            job["current_v5_task_id"] = ""
            if action_key == "force_reset":
                job["current_run_id"] = ""
                job["actual_top1"] = ""
                job["actual_top2"] = ""
                job["actual_top3"] = ""
                job["last_process_output"] = ""
                job["last_settlement_output"] = ""
            job["error_message"] = ""
            for step_name in JOB_STEP_NAMES:
                set_job_step_state(job, step_name, "idle")
        elif action_key == "mark_failed":
            job["status"] = "failed"
            job["error_message"] = "manually marked as failed"
            for step_name in ("settlement", "policy", "predictor", "odds"):
                current_step = str(job.get(_job_step_field(step_name, "status"), "") or "").strip().lower()
                if current_step in ("queued", "running"):
                    set_job_step_state(job, step_name, "failed", now_text, job["error_message"])
                    break

    return update_job(base_dir, job_id, mutate)


def delete_job(base_dir, job_id):
    target = str(job_id or "").strip()
    if not target:
        return None
    jobs = load_jobs(base_dir)
    kept = []
    deleted = None
    for job in jobs:
        if str(job.get("job_id", "") or "").strip() == target and deleted is None:
            deleted = dict(job)
            continue
        kept.append(dict(job))
    if deleted is None:
        return None
    save_jobs(base_dir, kept)
    return deleted


__all__ = [
    "STATUS_FLOW",
    "JOB_STEP_NAMES",
    "apply_job_action",
    "compute_initial_status",
    "create_job",
    "derive_job_display_state",
    "delete_job",
    "get_job",
    "hydrate_job_step_states",
    "initialize_job_step_fields",
    "load_jobs",
    "save_artifact",
    "scan_due_jobs",
    "scan_due_diagnostics",
    "set_job_step_state",
    "update_job",
]

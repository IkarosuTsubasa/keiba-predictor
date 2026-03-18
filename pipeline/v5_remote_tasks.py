import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path


JST_OFFSET = timedelta(hours=9)
TASK_STORE_NAME = "v5_remote_tasks.json"
STATUS_FLOW = (
    "queued",
    "dispatching",
    "dispatched",
    "running",
    "succeeded",
    "failed",
    "expired",
)


def _shared_dir(base_dir):
    path = Path(base_dir) / "data" / "_shared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tasks_path(base_dir):
    return _shared_dir(base_dir) / TASK_STORE_NAME


def _jst_now():
    return datetime.utcnow() + JST_OFFSET


def _dt_text(value):
    if not value:
        return ""
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def _build_task_id(job_id="", run_id=""):
    prefix = _jst_now().strftime("%Y%m%d_%H%M%S")
    token = secrets.token_hex(4)
    core = "_".join(part for part in (str(job_id or "").strip(), str(run_id or "").strip()) if part)
    return f"v5_{prefix}_{core or 'task'}_{token}"


def load_tasks(base_dir):
    path = _tasks_path(base_dir)
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    out = [dict(item) for item in rows if isinstance(item, dict)]
    out.sort(
        key=lambda item: (
            str(item.get("created_at", "") or ""),
            str(item.get("job_id", "") or ""),
            str(item.get("task_id", "") or ""),
        )
    )
    return out


def save_tasks(base_dir, tasks):
    rows = [dict(item) for item in list(tasks or []) if isinstance(item, dict)]
    path = _tasks_path(base_dir)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def create_task(
    base_dir,
    *,
    job_id="",
    run_id="",
    race_id="",
    scope_key="",
    task_type="predictor_v5",
    bundle_files=None,
    bundle_meta=None,
):
    now = _jst_now()
    task = {
        "task_id": _build_task_id(job_id, run_id),
        "task_type": str(task_type or "predictor_v5").strip(),
        "job_id": str(job_id or "").strip(),
        "run_id": str(run_id or "").strip(),
        "race_id": str(race_id or "").strip(),
        "scope_key": str(scope_key or "").strip(),
        "status": "queued",
        "attempt": 0,
        "workflow_run_id": "",
        "workflow_dispatch_ref": "",
        "bundle_token": secrets.token_urlsafe(24),
        "bundle_files": dict(bundle_files or {}),
        "bundle_meta": dict(bundle_meta or {}),
        "result_path": "",
        "result_summary": {},
        "error_message": "",
        "created_at": _dt_text(now),
        "started_at": "",
        "finished_at": "",
        "updated_at": _dt_text(now),
    }
    tasks = load_tasks(base_dir)
    tasks.append(task)
    save_tasks(base_dir, tasks)
    return task


def get_task(base_dir, task_id):
    target = str(task_id or "").strip()
    if not target:
        return None
    for task in load_tasks(base_dir):
        if str(task.get("task_id", "") or "").strip() == target:
            return task
    return None


def find_latest_task_for_job(base_dir, job_id, statuses=None):
    target = str(job_id or "").strip()
    if not target:
        return None
    allowed = {str(item or "").strip().lower() for item in list(statuses or []) if str(item or "").strip()}
    matched = []
    for task in load_tasks(base_dir):
        if str(task.get("job_id", "") or "").strip() != target:
            continue
        status = str(task.get("status", "") or "").strip().lower()
        if allowed and status not in allowed:
            continue
        matched.append(task)
    if not matched:
        return None
    matched.sort(key=lambda item: (str(item.get("created_at", "") or ""), str(item.get("task_id", "") or "")))
    return matched[-1]


def update_task(base_dir, task_id, mutate_fn):
    tasks = load_tasks(base_dir)
    updated = None
    now_text = _dt_text(_jst_now())
    for idx, task in enumerate(tasks):
        if str(task.get("task_id", "") or "").strip() != str(task_id or "").strip():
            continue
        current = dict(task)
        mutate_fn(current, now_text)
        status = str(current.get("status", "") or "").strip().lower()
        if status not in STATUS_FLOW:
            current["status"] = "failed"
        current["updated_at"] = now_text
        tasks[idx] = current
        updated = current
        break
    if updated is not None:
        save_tasks(base_dir, tasks)
    return updated


__all__ = [
    "STATUS_FLOW",
    "create_task",
    "find_latest_task_for_job",
    "get_task",
    "load_tasks",
    "save_tasks",
    "update_task",
]

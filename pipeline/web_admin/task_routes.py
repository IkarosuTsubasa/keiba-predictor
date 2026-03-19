import json
import traceback
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi.responses import JSONResponse


def pick_next_process_job_id(*, load_race_jobs):
    jobs = load_race_jobs()
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        if status in ("queued_process", "queued_policy"):
            return str(job.get("job_id", "") or "").strip()
    return ""


def pick_next_settle_job_id(*, load_race_jobs):
    jobs = load_race_jobs()
    for job in jobs:
        status = str(job.get("status", "") or "").strip().lower()
        actual_top1 = str(job.get("actual_top1", "") or "").strip()
        actual_top2 = str(job.get("actual_top2", "") or "").strip()
        actual_top3 = str(job.get("actual_top3", "") or "").strip()
        if status == "queued_settle" and actual_top1 and actual_top2 and actual_top3:
            return str(job.get("job_id", "") or "").strip()
    return ""


def run_due_jobs_once(*, base_dir, scan_due_race_jobs, load_race_jobs):
    changed = scan_due_race_jobs(base_dir)
    process_results = []
    settle_results = []
    errors = []

    while True:
        job_id = pick_next_process_job_id(load_race_jobs=lambda: load_race_jobs(base_dir))
        if not job_id:
            break
        print(
            "[web_app] "
            + json.dumps(
                {"ts": datetime.now().isoformat(timespec="seconds"), "event": "run_due_process_start", "job_id": job_id},
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            from race_job_runner import process_race_job

            summary = process_race_job(base_dir, job_id)
            process_results.append(summary)
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "run_due_process_done",
                        "job_id": job_id,
                        "run_id": str((summary or {}).get("run_id", "") or "").strip(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            try:
                from race_job_runner import fail_race_job

                fail_race_job(base_dir, job_id, str(exc))
            except Exception:
                pass
            print(
                "[web_app] "
                + json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "event": "run_due_process_error",
                        "job_id": job_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            errors.append({"kind": "process", "job_id": job_id, "error": str(exc)})

    while True:
        job_id = pick_next_settle_job_id(load_race_jobs=lambda: load_race_jobs(base_dir))
        if not job_id:
            break
        job = next((item for item in load_race_jobs(base_dir) if str(item.get("job_id", "")).strip() == job_id), {})
        actual_top3 = [
            str(job.get("actual_top1", "") or "").strip(),
            str(job.get("actual_top2", "") or "").strip(),
            str(job.get("actual_top3", "") or "").strip(),
        ]
        try:
            from race_job_runner import settle_race_job

            settle_results.append(settle_race_job(base_dir, job_id, actual_top3))
        except Exception as exc:
            try:
                from race_job_runner import fail_race_job

                fail_race_job(base_dir, job_id, str(exc))
            except Exception:
                pass
            errors.append({"kind": "settle", "job_id": job_id, "error": str(exc)})

    return {
        "queued_count": len(changed),
        "queued_job_ids": [str(item.get("job_id", "") or "").strip() for item in changed],
        "processed_count": len(process_results),
        "processed_job_ids": [str(item.get("job_id", "") or "").strip() for item in process_results],
        "settled_count": len(settle_results),
        "settled_job_ids": [str(item.get("job_id", "") or "").strip() for item in settle_results],
        "errors": errors,
    }


def import_history_zip(*, base_dir, archive_bytes, overwrite=False):
    data_root = Path(base_dir) / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    imported_paths = []
    with zipfile.ZipFile(BytesIO(archive_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            raw_name = str(info.filename or "").replace("\\", "/").strip("/")
            if not raw_name:
                continue
            parts = [part for part in raw_name.split("/") if part and part not in (".", "..")]
            if not parts:
                continue
            rel_parts = None
            if len(parts) >= 2 and parts[0] == "pipeline" and parts[1] == "data":
                rel_parts = parts[2:]
            elif parts[0] == "data":
                rel_parts = parts[1:]
            elif parts[0] in ("_shared", "central_dirt", "central_turf", "local"):
                rel_parts = parts
            if not rel_parts:
                skipped += 1
                continue
            dest = data_root.joinpath(*rel_parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            file_bytes = zf.read(info)
            dest.write_bytes(file_bytes)
            written += 1
            imported_paths.append(str(dest.relative_to(data_root)).replace("\\", "/"))
    return {"written": written, "skipped": skipped, "sample_paths": imported_paths[:8]}


def create_race_job_response(
    *,
    base_dir,
    token,
    scope_key,
    race_id,
    location,
    race_date,
    scheduled_off_time,
    target_distance,
    target_track_condition,
    lead_minutes,
    notes,
    kachiuma_file,
    shutuba_file,
    admin_token_valid,
    build_race_jobs_page,
    normalize_race_id,
    normalize_scope_key,
    default_job_race_date_text,
    target_surface_from_scope,
    create_race_job,
    save_race_job_artifact,
    update_race_job,
):
    if not admin_token_valid(token):
        return build_race_jobs_page(admin_token=token, authorized=False, error_text="管理员口令无效，无法创建任务。")
    race_id = normalize_race_id(race_id)
    scope_norm = normalize_scope_key(scope_key)
    race_date = str(race_date or "").strip() or default_job_race_date_text()
    if not scope_norm:
        return build_race_jobs_page(admin_token=token, error_text="范围无效。")
    if not race_id:
        return build_race_jobs_page(admin_token=token, error_text="Race ID 不能为空。")
    if not str(scheduled_off_time or "").strip():
        return build_race_jobs_page(admin_token=token, error_text="请填写开赛时间。")
    target_surface = target_surface_from_scope(scope_norm)
    try:
        target_distance_value = int(str(target_distance or "").strip())
    except ValueError:
        return build_race_jobs_page(admin_token=token, error_text="请填写比赛距离，例如 1200 或 1800。")
    if target_distance_value <= 0:
        return build_race_jobs_page(admin_token=token, error_text="比赛距离必须大于 0。")
    target_track_condition = str(target_track_condition or "").strip()
    if target_track_condition not in ("良", "稍重", "重", "不良"):
        return build_race_jobs_page(admin_token=token, error_text="场地状态只能是 良 / 稍重 / 重 / 不良。")
    if kachiuma_file is None or not str(getattr(kachiuma_file, "filename", "") or "").strip():
        return build_race_jobs_page(admin_token=token, error_text="请上传 kachiuma.csv。")
    if shutuba_file is None or not str(getattr(shutuba_file, "filename", "") or "").strip():
        return build_race_jobs_page(admin_token=token, error_text="请上传 shutuba.csv。")
    try:
        lead_value = int(str(lead_minutes or "30").strip() or "30")
    except ValueError:
        lead_value = 30

    artifact_payloads = []
    job = create_race_job(
        base_dir,
        race_id=race_id,
        scope_key=scope_norm,
        location=location,
        race_date=race_date,
        scheduled_off_time=scheduled_off_time,
        target_surface=target_surface,
        target_distance=str(target_distance_value),
        target_track_condition=target_track_condition,
        lead_minutes=lead_value,
        notes=notes,
        artifacts=[],
    )
    for artifact_type, upload in (("kachiuma", kachiuma_file), ("shutuba", shutuba_file)):
        if upload is None:
            continue
        payload = upload.file.read()
        artifact_payloads.append(
            save_race_job_artifact(
                base_dir,
                job["job_id"],
                artifact_type,
                upload.filename or f"{artifact_type}.csv",
                payload,
            )
        )

    def _attach_artifacts(row, now_text):
        row["artifacts"] = artifact_payloads
        row["status"] = "scheduled"
        row["updated_at"] = now_text

    update_race_job(base_dir, job["job_id"], _attach_artifacts)
    return build_race_jobs_page(admin_token=token, message_text=f"已创建任务 {job['job_id']}。")


async def import_history_archive_response(
    *,
    base_dir,
    token,
    overwrite,
    archive_file,
    admin_token_valid,
    build_race_jobs_page,
):
    if not admin_token_valid(token):
        return build_race_jobs_page(admin_token=token, authorized=False, error_text="管理员口令无效，无法导入历史数据。")
    if archive_file is None or not str(getattr(archive_file, "filename", "") or "").strip():
        return build_race_jobs_page(admin_token=token, error_text="请上传 ZIP 文件。")
    filename = str(archive_file.filename or "").strip()
    if not filename.lower().endswith(".zip"):
        return build_race_jobs_page(admin_token=token, error_text="只支持 ZIP 文件。")
    try:
        archive_bytes = await archive_file.read()
        summary = import_history_zip(
            base_dir=base_dir,
            archive_bytes=archive_bytes,
            overwrite=str(overwrite or "").strip() == "1",
        )
    except zipfile.BadZipFile:
        return build_race_jobs_page(admin_token=token, error_text="ZIP 文件损坏或格式无效。")
    except Exception as exc:
        return build_race_jobs_page(admin_token=token, error_text=f"导入失败：{exc}")
    message = f"已导入 {summary['written']} 个文件，跳过 {summary['skipped']} 个。"
    sample_paths = list(summary.get("sample_paths", []) or [])
    if sample_paths:
        message += " 示例: " + ", ".join(sample_paths)
    return build_race_jobs_page(admin_token=token, message_text=message)


def scan_due_race_jobs_response(*, base_dir, token, admin_token_valid, build_race_jobs_page, scan_due_race_jobs):
    if not admin_token_valid(token):
        return build_race_jobs_page(admin_token=token, authorized=False, error_text="管理员口令无效，无法扫描任务。")
    changed = scan_due_race_jobs(base_dir)
    if changed:
        return build_race_jobs_page(admin_token=token, message_text=f"已将 {len(changed)} 场比赛加入处理队列。")
    return build_race_jobs_page(admin_token=token, message_text="当前没有到点任务。")


def run_due_race_jobs_now_response(*, base_dir, token, admin_token_valid, build_race_jobs_page, scan_due_race_jobs, load_race_jobs):
    if not admin_token_valid(token):
        return build_race_jobs_page(admin_token=token, authorized=False, error_text="管理员口令无效，无法立即执行任务。")
    summary = run_due_jobs_once(
        base_dir=base_dir,
        scan_due_race_jobs=scan_due_race_jobs,
        load_race_jobs=load_race_jobs,
    )
    message_parts = [
        f"queued={int(summary.get('queued_count', 0) or 0)}",
        f"processed={int(summary.get('processed_count', 0) or 0)}",
        f"settled={int(summary.get('settled_count', 0) or 0)}",
        f"errors={len(list(summary.get('errors', []) or []))}",
    ]
    error_items = list(summary.get("errors", []) or [])
    if error_items:
        error_text = "\n".join(
            f"[{item.get('kind', 'job')}] {item.get('job_id', '-')}: {item.get('error', '')}"
            for item in error_items
        )
        return build_race_jobs_page(
            admin_token=token,
            message_text="已执行到点任务。 " + ", ".join(message_parts),
            error_text=error_text,
        )
    return build_race_jobs_page(
        admin_token=token,
        message_text="已执行到点任务。 " + ", ".join(message_parts),
    )


def internal_run_due_response(*, base_dir, token, admin_token_valid, scan_due_race_jobs, load_race_jobs):
    if not admin_token_valid(token):
        return JSONResponse({"ok": False, "error": "invalid_admin_token"}, status_code=403)
    summary = run_due_jobs_once(
        base_dir=base_dir,
        scan_due_race_jobs=scan_due_race_jobs,
        load_race_jobs=load_race_jobs,
    )
    ok = not bool(list(summary.get("errors", []) or []))
    return JSONResponse({"ok": ok, **summary}, status_code=200 if ok else 500)


def topup_today_all_llm_budget_response(
    *,
    base_dir,
    token,
    admin_token_valid,
    render_console_page,
    default_job_race_date_text,
    resolve_daily_bankroll_yen,
    add_bankroll_topup,
):
    if not admin_token_valid(token):
        return render_console_page(admin_token=token, error_text="管理员口令无效，无法追加预算。")
    ledger_date = default_job_race_date_text().replace("-", "")
    amount_yen = resolve_daily_bankroll_yen(ledger_date)
    for engine in ("gemini", "deepseek", "openai", "grok"):
        add_bankroll_topup(base_dir, ledger_date, amount_yen, policy_engine=engine)
    return render_console_page(
        admin_token=token,
        message_text=f"已为 {ledger_date} 的全部 LLM 追加预算 {amount_yen}。",
    )

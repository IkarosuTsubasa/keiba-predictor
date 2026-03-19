import json


def view_run_response(
    *,
    run_id,
    scope_key,
    token,
    normalize_scope_key,
    infer_scope_and_run,
    render_page,
    resolve_run,
    normalize_race_id,
    resolve_latest_run_by_race_id,
    infer_run_id_from_row,
    update_run_row_fields,
):
    run_id = str(run_id or "").strip()
    scope_key = normalize_scope_key(scope_key)
    run_row = None
    if not scope_key:
        scope_key, run_row = infer_scope_and_run(run_id)
    if not scope_key:
        return render_page("", error_text="无法识别 Run ID 或 Race ID。", admin_token=token)
    if run_row is None:
        run_row = resolve_run(run_id, scope_key)
    if run_row is None:
        race_id = normalize_race_id(run_id)
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_key)
    if run_row is None:
        return render_page(
            scope_key,
            error_text="未找到对应的 Run ID / Race ID。",
            selected_run_id=run_id,
            admin_token=token,
        )
    resolved_run_id = str(run_row.get("run_id", "") or "").strip()
    if not resolved_run_id:
        resolved_run_id = infer_run_id_from_row(run_row)
        if resolved_run_id:
            update_run_row_fields(scope_key, run_row, {"run_id": resolved_run_id})
            run_row["run_id"] = resolved_run_id
    if not resolved_run_id:
        return render_page(
            scope_key,
            error_text="Run 记录存在，但缺少 run_id。",
            selected_run_id=run_id,
            admin_token=token,
        )
    return render_page(
        scope_key,
        selected_run_id=resolved_run_id,
        summary_run_id=resolved_run_id,
        admin_token=token,
    )


def api_runs_response(*, scope_key, limit, query, normalize_scope_key, default_limit, max_limit, get_recent_runs):
    scope_key = normalize_scope_key(scope_key) or "central_dirt"
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = default_limit
    limit = max(1, min(max_limit, limit))
    return get_recent_runs(scope_key, limit=limit, query=query)


def run_pipeline_response(
    *,
    token,
    race_id,
    race_url,
    history_url,
    scope_key,
    location,
    race_date,
    surface,
    distance,
    track_cond,
    admin_token_valid,
    admin_execution_denied,
    normalize_scope_key,
    render_page,
    normalize_race_id,
    run_script,
    run_pipeline_path,
    extract_top5,
    parse_run_id,
):
    if not admin_token_valid(token):
        return admin_execution_denied("管理员令牌无效，不能执行 Pipeline。", scope_key=scope_key, token=token)
    scope_key = normalize_scope_key(scope_key)
    if not scope_key:
        return render_page("", error_text="缺少数据范围。", admin_token=token)
    race_id = normalize_race_id(race_id or race_url)
    history_url = str(history_url or "").strip()
    location = str(location or "").strip()
    race_date = str(race_date or "").strip()
    if not race_id or not history_url or not location:
        return render_page(
            scope_key,
            error_text="Race ID、History URL、Location 为必填项。",
            admin_token=token,
        )
    if scope_key == "local":
        race_url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    else:
        race_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    if not str(surface or "").strip():
        surface = "1" if scope_key == "central_turf" else "2"
    track_cond_norm = str(track_cond or "").strip().lower()
    track_cond_map = {
        "good": "good",
        "slightly_heavy": "slightly_heavy",
        "heavy": "heavy",
        "bad": "bad",
    }
    if track_cond_norm in track_cond_map:
        track_cond = track_cond_map[track_cond_norm]
    inputs = [race_url, history_url, location, race_date, surface, distance, track_cond]
    code, output = run_script(
        run_pipeline_path,
        inputs=inputs,
        extra_env={"SCOPE_KEY": scope_key},
    )
    label = f"Exit code: {code}"
    top5_text = extract_top5(output)
    return render_page(
        scope_key,
        output_text=f"{label}\n{output}",
        top5_text=top5_text,
        summary_run_id=parse_run_id(output),
        admin_token=token,
    )


def record_predictor_response(
    *,
    token,
    scope_key,
    run_id,
    top1,
    top2,
    top3,
    admin_token_valid,
    admin_execution_denied,
    normalize_scope_key,
    infer_scope_and_run,
    resolve_run,
    normalize_race_id,
    resolve_latest_run_by_race_id,
    render_page,
    resolve_run_asset_path,
    refresh_odds_for_run,
    run_script,
    record_predictor_path,
):
    if not admin_token_valid(token):
        return admin_execution_denied(
            "管理员令牌无效，不能执行 Predictor 结果记录。",
            scope_key=scope_key,
            token=token,
            selected_run_id=str(run_id or "").strip(),
        )
    scope_norm = normalize_scope_key(scope_key)
    run_id = str(run_id or "").strip()
    run_row = None
    if not scope_norm:
        scope_norm, run_row = infer_scope_and_run(run_id)
    if run_row is None and scope_norm:
        run_row = resolve_run(run_id, scope_norm)
    if run_row is None and scope_norm:
        race_id = normalize_race_id(run_id)
        if race_id:
            run_row = resolve_latest_run_by_race_id(race_id, scope_norm)
    resolved_run_id = str((run_row or {}).get("run_id", "") or "").strip() or run_id
    if not top1 or not top2 or not top3:
        return render_page(
            scope_norm or scope_key,
            error_text="Top1/Top2/Top3 为必填项。",
            selected_run_id=resolved_run_id,
            admin_token=token,
        )
    refresh_ok = False
    refresh_message = "缺少 run row，未刷新赔率。"
    refresh_warnings = []
    if run_row is not None and scope_norm:
        odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "odds_path", "odds")
        wide_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "wide_odds_path", "wide_odds")
        fuku_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "fuku_odds_path", "fuku_odds")
        quinella_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "quinella_odds_path", "quinella_odds")
        exacta_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "exacta_odds_path", "exacta_odds")
        trio_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "trio_odds_path", "trio_odds")
        trifecta_odds_path = resolve_run_asset_path(scope_norm, resolved_run_id, run_row, "trifecta_odds_path", "trifecta_odds")
        refresh_ok, refresh_message, refresh_warnings = refresh_odds_for_run(
            run_row,
            scope_norm,
            odds_path,
            wide_odds_path=wide_odds_path,
            fuku_odds_path=fuku_odds_path,
            quinella_odds_path=quinella_odds_path,
            exacta_odds_path=exacta_odds_path,
            trio_odds_path=trio_odds_path,
            trifecta_odds_path=trifecta_odds_path,
        )
    code, output = run_script(
        record_predictor_path,
        inputs=[resolved_run_id, top1, top2, top3],
        extra_blanks=2,
        extra_env={"SCOPE_KEY": scope_norm or scope_key or "central_dirt"},
    )
    label = f"Exit code: {code}"
    output_parts = [f"[odds_update] status={'ok' if refresh_ok else 'fail'} message={refresh_message or ''}".strip()]
    if refresh_warnings:
        output_parts.append("[odds_update][warnings] " + "; ".join(str(x) for x in refresh_warnings))
    output_parts.append(label)
    output_parts.append(output)
    return render_page(
        scope_norm or scope_key,
        output_text="\n".join(part for part in output_parts if str(part).strip()),
        selected_run_id=resolved_run_id,
        admin_token=token,
    )


def run_llm_buy_response(
    *,
    token,
    scope_key,
    run_id,
    policy_engine,
    policy_model,
    refresh_odds,
    admin_token_valid,
    admin_execution_denied,
    resolve_run_selection,
    render_page,
    maybe_refresh_run_odds,
    build_llm_buy_output,
    load_policy_bankroll_summary,
    normalize_policy_engine,
    execute_policy_buy,
):
    if not admin_token_valid(token):
        return admin_execution_denied(
            "管理员令牌无效，不能执行单引擎 LLM 买入。",
            scope_key=scope_key,
            token=token,
            selected_run_id=str(run_id or "").strip(),
        )
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return render_page(
            scope_norm or scope_key,
            error_text="未找到可用于 LLM 买入的 Run ID / Race ID。",
            selected_run_id=resolved_run_id or str(run_id or "").strip(),
            admin_token=token,
        )
    refresh_enabled = str(refresh_odds or "").strip() not in ("", "0", "false", "False", "off")
    refresh_ok, refresh_message, refresh_warnings = maybe_refresh_run_odds(scope_norm, run_row, resolved_run_id, refresh_enabled)
    if not refresh_ok:
        return render_page(
            scope_norm,
            error_text=build_llm_buy_output(
                load_policy_bankroll_summary(resolved_run_id, run_row.get("timestamp", ""), policy_engine=policy_engine),
                refresh_ok,
                refresh_message,
                refresh_warnings,
                "[llm_buy][blocked] odds refresh failed; skip policy execution.",
                policy_engine=normalize_policy_engine(policy_engine),
            ),
            selected_run_id=resolved_run_id,
            admin_token=token,
        )
    try:
        result = execute_policy_buy(
            scope_norm,
            run_row,
            resolved_run_id,
            policy_engine=policy_engine,
            policy_model=policy_model,
        )
    except Exception as exc:
        return render_page(
            scope_norm,
            error_text=build_llm_buy_output(
                load_policy_bankroll_summary(resolved_run_id, run_row.get("timestamp", ""), policy_engine=policy_engine),
                refresh_ok,
                refresh_message,
                refresh_warnings,
                f"[llm_buy][error] {exc}",
                policy_engine=normalize_policy_engine(policy_engine),
            ),
            selected_run_id=resolved_run_id,
            admin_token=token,
        )
    return render_page(
        scope_norm,
        output_text=build_llm_buy_output(
            result["summary_before"],
            refresh_ok,
            refresh_message,
            refresh_warnings,
            result["output_text"],
            result["engine"],
        ),
        selected_run_id=resolved_run_id,
        admin_token=token,
    )


def run_all_llm_buy_response(
    *,
    token,
    scope_key,
    run_id,
    refresh_odds,
    admin_token_valid,
    admin_execution_denied,
    resolve_run_selection,
    render_page,
    maybe_refresh_run_odds,
    build_llm_buy_output,
    load_policy_bankroll_summary,
    execute_policy_buy,
):
    if not admin_token_valid(token):
        return admin_execution_denied(
            "管理员令牌无效，不能执行全引擎 LLM 买入。",
            scope_key=scope_key,
            token=token,
            selected_run_id=str(run_id or "").strip(),
        )
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return render_page(
            scope_norm or scope_key,
            error_text="未找到可用于 LLM 买入的 Run ID / Race ID。",
            selected_run_id=resolved_run_id or str(run_id or "").strip(),
            admin_token=token,
        )
    refresh_enabled = str(refresh_odds or "").strip() not in ("", "0", "false", "False", "off")
    refresh_ok, refresh_message, refresh_warnings = maybe_refresh_run_odds(scope_norm, run_row, resolved_run_id, refresh_enabled)
    if not refresh_ok:
        return render_page(
            scope_norm,
            error_text=build_llm_buy_output(
                load_policy_bankroll_summary(resolved_run_id, run_row.get("timestamp", ""), policy_engine="gemini"),
                refresh_ok,
                refresh_message,
                refresh_warnings,
                "[llm_buy][blocked] odds refresh failed; skip all policy engines.",
                policy_engine="all",
            ),
            selected_run_id=resolved_run_id,
            admin_token=token,
        )
    result_blocks = []
    error_blocks = []
    for engine in ("gemini", "deepseek", "openai", "grok"):
        try:
            result = execute_policy_buy(scope_norm, run_row, resolved_run_id, policy_engine=engine, policy_model="")
            result_blocks.append(
                build_llm_buy_output(
                    result["summary_before"],
                    refresh_ok,
                    refresh_message,
                    refresh_warnings,
                    result["output_text"],
                    result["engine"],
                )
            )
        except Exception as exc:
            error_blocks.append(f"[llm_buy][{engine}] {exc}")
    if error_blocks and not result_blocks:
        return render_page(
            scope_norm,
            error_text="\n\n".join(error_blocks),
            selected_run_id=resolved_run_id,
            admin_token=token,
        )
    output_text = "\n\n".join(block for block in result_blocks if str(block).strip())
    if error_blocks:
        output_text = "\n\n".join([output_text] + error_blocks if output_text else error_blocks)
    return render_page(
        scope_norm,
        output_text=output_text,
        selected_run_id=resolved_run_id,
        admin_token=token,
    )


def topup_all_llm_budget_response(
    *,
    base_dir,
    token,
    scope_key,
    run_id,
    admin_token_valid,
    admin_execution_denied,
    resolve_run_selection,
    render_page,
    extract_ledger_date,
    resolve_daily_bankroll_yen,
    add_bankroll_topup,
):
    if not admin_token_valid(token):
        return admin_execution_denied(
            "管理员令牌无效，不能为全引擎 LLM 追加预算。",
            scope_key=scope_key,
            token=token,
            selected_run_id=str(run_id or "").strip(),
        )
    scope_norm, run_row, resolved_run_id = resolve_run_selection(scope_key, run_id)
    if not scope_norm or run_row is None or not resolved_run_id:
        return render_page(
            scope_norm or scope_key,
            error_text="未找到可用于 LLM 预算追加的 Run ID / Race ID。",
            selected_run_id=resolved_run_id or str(run_id or "").strip(),
            admin_token=token,
        )
    ledger_date = extract_ledger_date(resolved_run_id, run_row.get("timestamp", ""))
    amount_yen = resolve_daily_bankroll_yen(ledger_date)
    lines = [f"[llm_budget_topup] ledger_date={ledger_date} amount_yen={amount_yen} engines=4"]
    for engine in ("gemini", "deepseek", "openai", "grok"):
        summary = add_bankroll_topup(base_dir, ledger_date, amount_yen, policy_engine=engine)
        lines.append(
            "[topup][{engine}] available_bankroll_yen={available} topup_yen={topup}".format(
                engine=engine,
                available=int(summary.get("available_bankroll_yen", 0) or 0),
                topup=int(summary.get("topup_yen", 0) or 0),
            )
        )
    return render_page(
        scope_norm,
        output_text="\n".join(lines),
        selected_run_id=resolved_run_id,
        admin_token=token,
    )


def reset_llm_state_response(*, base_dir, token, admin_token_valid, admin_execution_denied, render_page, reset_llm_state_files):
    if not admin_token_valid(token):
        return admin_execution_denied("管理员令牌无效，不能重置 LLM 状态。", token=token)
    summary = reset_llm_state_files(base_dir)
    return render_page("", output_text=json.dumps(summary, ensure_ascii=False, indent=2), admin_token=token)


def edit_race_job_details_response(
    *,
    base_dir,
    token,
    job_id,
    race_id,
    location,
    race_date,
    scheduled_off_time,
    target_distance,
    target_track_condition,
    lead_minutes,
    notes,
    admin_token_valid,
    build_race_jobs_page,
    normalize_race_id,
    default_job_race_date_text,
    load_race_jobs,
    render_console_page,
    target_surface_from_scope,
    parse_job_dt_text,
    format_job_dt_text,
    update_race_job,
    compute_race_job_initial_status,
):
    if not admin_token_valid(token):
        return build_race_jobs_page(admin_token=token, authorized=False, error_text="管理员令牌无效，不能编辑任务。")
    race_id = normalize_race_id(race_id)
    if not race_id:
        return build_race_jobs_page(admin_token=token, error_text="Race ID 不能为空。")
    race_date = str(race_date or "").strip() or default_job_race_date_text()
    scheduled_off_time = str(scheduled_off_time or "").strip()
    if not scheduled_off_time:
        return build_race_jobs_page(admin_token=token, error_text="开赛时间不能为空。")
    try:
        target_distance_value = int(str(target_distance or "").strip())
    except ValueError:
        return build_race_jobs_page(admin_token=token, error_text="距离必须是数字，例如 1200 或 1800。")
    if target_distance_value <= 0:
        return build_race_jobs_page(admin_token=token, error_text="距离必须大于 0。")
    target_track_condition = str(target_track_condition or "").strip()
    if target_track_condition not in ("良", "稍重", "重", "不良"):
        return build_race_jobs_page(admin_token=token, error_text="马场状态必须是 良 / 稍重 / 重 / 不良。")
    try:
        lead_value = max(0, int(str(lead_minutes or "30").strip() or "30"))
    except ValueError:
        lead_value = 30

    current = next(
        (item for item in load_race_jobs(base_dir) if str(item.get("job_id", "") or "").strip() == str(job_id or "").strip()),
        None,
    )
    if current is None:
        return build_race_jobs_page(admin_token=token, error_text="未找到对应任务。")
    current_run_id = str(current.get("current_run_id", "") or "").strip()
    current_race_date = str(current.get("race_date", "") or "").strip()
    if current_run_id and race_date != current_race_date:
        locked_date = current_race_date
        if (not locked_date) and len(current_run_id) >= 8 and current_run_id[:8].isdigit():
            locked_date = f"{current_run_id[:4]}-{current_run_id[4:6]}-{current_run_id[6:8]}"
        return render_console_page(
            admin_token=token,
            error_text=f"该任务已有运行记录，不能修改比赛日期。当前锁定日期：{locked_date or '未知'}",
        )

    target_surface = target_surface_from_scope(str(current.get("scope_key", "") or "").strip())
    off_dt = parse_job_dt_text(scheduled_off_time)
    if off_dt is None:
        return build_race_jobs_page(admin_token=token, error_text="开赛时间格式不正确。")
    from datetime import timedelta

    process_after_dt = off_dt - timedelta(minutes=lead_value) if off_dt else None

    def _edit_job(row, now_text):
        del now_text
        row["race_id"] = race_id
        row["location"] = str(location or "").strip()
        row["race_date"] = race_date
        row["scheduled_off_time"] = format_job_dt_text(off_dt) or scheduled_off_time
        row["process_after_time"] = format_job_dt_text(process_after_dt)
        row["target_surface"] = target_surface
        row["target_distance"] = str(target_distance_value)
        row["target_track_condition"] = target_track_condition
        row["lead_minutes"] = lead_value
        row["notes"] = str(notes or "").strip()
        if str(row.get("status", "") or "").strip() in ("uploaded", "scheduled"):
            row["status"] = compute_race_job_initial_status(row)

    job = update_race_job(base_dir, job_id, _edit_job)
    if job is None:
        return build_race_jobs_page(admin_token=token, error_text="未找到对应任务。")
    return build_race_jobs_page(admin_token=token, message_text=f"{job_id} 已更新。")

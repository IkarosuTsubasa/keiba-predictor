import os
import re

from surface_scope import normalize_scope_key


def render_workspace_page(
    *,
    scope_key="",
    output_text="",
    error_text="",
    top5_text="",
    summary_run_id="",
    selected_run_id="",
    admin_token="",
    admin_workspace_html="",
    build_run_options,
    build_llm_compare_block,
    build_stats_block,
    build_table_html,
    load_policy_bankroll_summary,
    load_policy_daily_profit_summary,
    build_daily_profit_chart_html,
    resolve_run,
    parse_run_id,
    normalize_race_id,
    resolve_predictor_paths,
    load_top5_table,
    load_ability_marks_table,
    load_mark_recommendation_table,
    load_text_file,
    load_prediction_summary,
    load_bet_engine_v3_cfg_summary,
    load_policy_payloads,
    load_policy_run_ticket_rows,
    build_policy_workspace_html,
    build_llm_battle_bundle,
    load_actual_result_map,
    normalize_policy_engine,
    resolve_policy_model,
    default_gemini_model,
    prefix_public_html_routes,
    page_template,
    admin_token_enabled,
    load_predictor_summary,
):
    scope_norm = normalize_scope_key(scope_key)
    default_scope = scope_norm or "central_dirt"
    run_options = build_run_options(scope_norm or scope_key)
    selected_run_id = str(selected_run_id or "").strip()
    llm_compare_html = build_llm_compare_block(
        build_table_html=build_table_html,
        load_policy_bankroll_summary=load_policy_bankroll_summary,
        load_policy_daily_profit_summary=load_policy_daily_profit_summary,
    )
    stats_block = build_stats_block(
        scope_norm,
        load_predictor_summary=load_predictor_summary,
        build_table_html=build_table_html,
        load_policy_bankroll_summary=load_policy_bankroll_summary,
        load_policy_daily_profit_summary=load_policy_daily_profit_summary,
        build_daily_profit_chart_html=build_daily_profit_chart_html,
    )
    run_id = ""
    run_row = None
    if scope_norm:
        if selected_run_id:
            run_row = resolve_run(selected_run_id, scope_norm)
            if run_row:
                run_id = run_row.get("run_id", selected_run_id)
            else:
                run_id = selected_run_id
        else:
            run_id = parse_run_id(output_text)
            if run_id:
                run_row = resolve_run(run_id, scope_norm)
            else:
                run_row = resolve_run("", scope_norm)
                if run_row:
                    run_id = run_row.get("run_id", "")
    view_selected_run_id = selected_run_id or run_id or summary_run_id
    current_race_id = ""
    if run_row:
        current_race_id = normalize_race_id(run_row.get("race_id", ""))
    if not current_race_id:
        race_candidate = re.sub(r"\D", "", selected_run_id or summary_run_id or "")
        if re.fullmatch(r"\d{12}", race_candidate):
            current_race_id = race_candidate
    view_run_options = build_run_options(scope_norm or scope_key, view_selected_run_id)
    top5_table_html = ""
    mark_table_html = ""
    llm_battle_html = ""
    gemini_policy_html = ""
    summary_table_html = ""
    if run_id:
        predictor_top_sections = []
        predictor_mark_sections = []
        predictor_summary_sections = []
        bet_engine_v3_summary = load_bet_engine_v3_cfg_summary(scope_norm or scope_key, run_id)
        policy_payloads = []
        actual_result_map = load_actual_result_map(scope_norm or scope_key)
        for payload in load_policy_payloads(scope_norm or scope_key, run_id, run_row):
            payload = dict(payload)
            budget_items = [dict(item) for item in list(payload.get("budgets", []) or [])]
            live_policy_engine = normalize_policy_engine(payload.get("policy_engine", "gemini"))
            live_summary = load_policy_bankroll_summary(
                run_id,
                (run_row or {}).get("timestamp", ""),
                policy_engine=live_policy_engine,
            )
            live_tickets = load_policy_run_ticket_rows(run_id, policy_engine=live_policy_engine)
            if budget_items:
                first = dict(budget_items[0])
                portfolio = dict(first.get("portfolio", {}) or {})
                portfolio.setdefault("before", dict(live_summary))
                portfolio["after"] = dict(live_summary)
                first["portfolio"] = portfolio
                if live_tickets:
                    first["tickets"] = live_tickets
                budget_items[0] = first
            payload["budgets"] = budget_items
            policy_payloads.append(payload)
        primary_policy_payload = dict(policy_payloads[0]) if policy_payloads else {}
        gemini_policy_html = build_policy_workspace_html(policy_payloads)
        battle_bundle = build_llm_battle_bundle(
            scope_norm or scope_key,
            run_id,
            run_row,
            policy_payloads,
            actual_result_map,
        )
        llm_battle_html = battle_bundle.get("html", "")
        for spec, pred_path in resolve_predictor_paths(scope_norm or scope_key, run_id, run_row):
            if not pred_path or not pred_path.exists():
                continue
            predictor_run_row = dict(run_row or {})
            predictor_run_row["predictions_path"] = str(pred_path)
            top_rows, top_cols = load_top5_table(scope_norm or scope_key, run_id, predictor_run_row)
            if top_rows:
                predictor_top_sections.append(
                    build_table_html(top_rows, top_cols, f"Top5 Predictions - {spec['label']}")
                )
            ability_rows, ability_cols = load_ability_marks_table(scope_norm or scope_key, run_id, predictor_run_row)
            if ability_rows:
                predictor_mark_sections.append(
                    build_table_html(ability_rows, ability_cols, f"Ability Marks - {spec['label']}")
                )
            else:
                mark_rows, mark_cols = load_mark_recommendation_table(scope_norm or scope_key, run_id, predictor_run_row)
                ability_rows = mark_rows
                if mark_rows:
                    predictor_mark_sections.append(
                        build_table_html(mark_rows, mark_cols, f"Integrated Marks - {spec['label']}")
                    )
            summary_rows = load_prediction_summary(scope_norm or scope_key, run_id, predictor_run_row)
            if summary_rows:
                predictor_summary_sections.append(
                    build_table_html(summary_rows, ["metric", "value"], f"Model Status - {spec['label']}")
                )
        if predictor_top_sections:
            top5_table_html = "".join(predictor_top_sections)
        if predictor_mark_sections:
            mark_table_html = "".join(predictor_mark_sections)
        if predictor_summary_sections:
            summary_table_html = "".join(predictor_summary_sections)
    return prefix_public_html_routes(
        page_template(
            output_text=output_text,
            error_text=error_text,
            run_options=run_options,
            view_run_options=view_run_options,
            view_selected_run_id=view_selected_run_id,
            current_race_id=current_race_id,
            top5_text=top5_text if not top5_table_html else "",
            top5_table_html=top5_table_html,
            mark_table_html=mark_table_html,
            llm_battle_html=llm_battle_html,
            llm_compare_html=llm_compare_html,
            gemini_policy_html=gemini_policy_html,
            summary_table_html=summary_table_html,
            stats_block=stats_block,
            default_scope=default_scope,
            default_policy_engine=normalize_policy_engine(os.environ.get("POLICY_ENGINE", "gemini") or "gemini"),
            default_policy_model=resolve_policy_model(
                normalize_policy_engine(os.environ.get("POLICY_ENGINE", "gemini") or "gemini"),
                os.environ.get("POLICY_MODEL", ""),
                os.environ.get("GEMINI_MODEL", default_gemini_model),
            ),
            admin_token=admin_token,
            admin_enabled=admin_token_enabled(),
            admin_workspace_html=admin_workspace_html,
        )
    )

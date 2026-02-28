def build_stats_block(
    scope_norm,
    *,
    load_profit_summary,
    load_daily_profit_summary,
    load_daily_profit_summary_all_scopes,
    load_wide_box_daily_profit_summary,
    load_bet_type_profit_summary,
    load_bet_type_summary,
    load_predictor_summary,
    build_metric_table,
    build_table_html,
    build_daily_profit_chart_html,
):
    if not scope_norm:
        return ""

    profit_rows = load_profit_summary(scope_norm)
    daily_profit_rows = load_daily_profit_summary(scope_norm)
    all_scope_daily_rows = load_daily_profit_summary_all_scopes(days=30)
    wide_box_rows = load_wide_box_daily_profit_summary(scope_norm)
    bet_type_profit_rows = load_bet_type_profit_summary(scope_norm)
    bet_type_rows = load_bet_type_summary(scope_norm)
    predictor_rows = load_predictor_summary(scope_norm)

    parts = []
    if profit_rows:
        parts.append(
            build_table_html(
                profit_rows,
                ["budget_yen", "runs", "total_stake_yen", "total_profit_yen", "overall_roi"],
                "Overall Profit Summary",
            )
        )
    if daily_profit_rows:
        parts.append(
            build_table_html(
                daily_profit_rows,
                ["date", "budget_yen", "runs", "profit_yen", "base_amount", "roi"],
                "Daily Profit",
            )
        )
    if all_scope_daily_rows:
        parts.append(
            build_daily_profit_chart_html(
                all_scope_daily_rows,
                "All Scopes Daily Profit Trend",
            )
        )
        parts.append(
            build_table_html(
                all_scope_daily_rows,
                ["date", "runs", "profit_yen", "base_amount", "roi"],
                "All Scopes Daily Totals",
            )
        )
    if wide_box_rows:
        parts.append(
            build_table_html(
                wide_box_rows,
                ["date", "runs", "profit_yen", "base_amount", "roi"],
                "Top5 Wide Box Profit (1000 JPY)",
            )
        )
    if bet_type_profit_rows:
        parts.append(
            build_table_html(
                bet_type_profit_rows,
                ["budget_yen", "bet_type", "amount_yen", "est_profit_yen", "roi"],
                "Win/Place/Wide Profit (2026+)",
            )
        )
    if bet_type_rows:
        parts.append(
            build_table_html(
                bet_type_rows,
                ["budget_yen", "bet_type", "bets", "hits", "hit_rate", "amount_yen", "est_profit_yen"],
                "Bet Type Hit Rate",
            )
        )
    if predictor_rows:
        parts.append(build_metric_table(predictor_rows, "Predictor Hit Rate"))
    return "\n".join(parts)


def build_run_summary_block(
    scope_norm,
    summary_id,
    *,
    load_run_result_summary,
    load_run_bet_ticket_summary,
    load_run_predictor_summary,
    load_run_bet_type_summary,
    build_metric_table,
    build_table_html,
):
    if not (scope_norm and summary_id):
        return ""

    parts = []
    result_rows = load_run_result_summary(scope_norm, summary_id)
    if result_rows:
        parts.append(
            build_table_html(
                result_rows,
                ["budget_yen", "run_profit_yen", "run_stake_yen", "run_roi"],
                "Run Profit",
            )
        )

    bet_ticket_rows = load_run_bet_ticket_summary(scope_norm, summary_id)
    if bet_ticket_rows:
        parts.append(
            build_table_html(
                bet_ticket_rows,
                [
                    "budget_yen",
                    "bet_type",
                    "horse_no",
                    "horse_name",
                    "amount_yen",
                    "hit",
                    "est_payout_yen",
                    "profit_yen",
                ],
                "Run Bet Plan PnL Details",
            )
        )

    predictor_rows = load_run_predictor_summary(scope_norm, summary_id)
    if predictor_rows:
        parts.append(build_metric_table(predictor_rows, "Run Predictor Hit Rate"))

    bet_type_rows = load_run_bet_type_summary(scope_norm, summary_id)
    if bet_type_rows:
        parts.append(
            build_table_html(
                bet_type_rows,
                ["budget_yen", "bet_type", "bets", "hits", "hit_rate", "amount_yen", "est_profit_yen"],
                "Run Bet Type Hit Rate",
            )
        )
    return "\n".join(parts) if parts else ""

def build_stats_block(
    scope_norm,
    *,
    load_predictor_summary,
    build_table_html,
    load_gemini_bankroll_summary,
    load_gemini_daily_profit_summary,
    build_daily_profit_chart_html,
):
    if not scope_norm:
        return ""

    predictor_rows = load_predictor_summary(scope_norm)
    gemini_bankroll = load_gemini_bankroll_summary()
    gemini_daily_rows = load_gemini_daily_profit_summary(days=30)

    parts = []
    if gemini_bankroll:
        parts.append(
            build_table_html(
                [
                    {"metric": "date", "value": gemini_bankroll.get("ledger_date", "")},
                    {"metric": "start_bankroll_yen", "value": gemini_bankroll.get("start_bankroll_yen", "")},
                    {"metric": "available_bankroll_yen", "value": gemini_bankroll.get("available_bankroll_yen", "")},
                    {"metric": "open_stake_yen", "value": gemini_bankroll.get("open_stake_yen", "")},
                    {"metric": "realized_profit_yen", "value": gemini_bankroll.get("realized_profit_yen", "")},
                    {"metric": "pending_tickets", "value": gemini_bankroll.get("pending_tickets", "")},
                ],
                ["metric", "value"],
                "Gemini Bankroll",
            )
        )
    if gemini_daily_rows:
        parts.append(build_daily_profit_chart_html(gemini_daily_rows, "Gemini Daily Profit"))
        parts.append(
            build_table_html(
                gemini_daily_rows,
                ["date", "runs", "profit_yen", "base_amount", "roi"],
                "Gemini Daily Summary",
            )
        )
    if predictor_rows:
        parts.append(
            build_table_html(
                predictor_rows,
                [
                    "predictor",
                    "samples",
                    "top3_hit_rate",
                    "top5_to_top3_hit_rate",
                    "top1_hit_rate",
                    "top1_in_top3_rate",
                    "top3_exact_rate",
                ],
                "Predictor Hit Rate",
            )
        )
    return "\n".join(parts)

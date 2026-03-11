def _to_int(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def build_llm_compare_block(
    *,
    build_table_html,
    load_policy_bankroll_summary,
    load_policy_daily_profit_summary,
):
    policy_labels = {
        "gemini": "Gemini",
        "siliconflow": "DeepSeek",
        "openai": "OpenAI GPT-5",
    }
    compare_rows = []
    for policy_engine in ("gemini", "siliconflow", "openai"):
        title = policy_labels.get(policy_engine, policy_engine)
        bankroll = load_policy_bankroll_summary(policy_engine=policy_engine) or {}
        daily_rows = load_policy_daily_profit_summary(days=30, policy_engine=policy_engine) or []
        profit_30d = sum(_to_int(row.get("profit_yen")) for row in daily_rows)
        base_30d = sum(_to_int(row.get("base_amount")) for row in daily_rows)
        runs_30d = sum(_to_int(row.get("runs")) for row in daily_rows)
        roi_30d = round((base_30d + profit_30d) / base_30d, 4) if base_30d > 0 else ""
        compare_rows.append(
            {
                "model": title,
                "profit_30d_yen": profit_30d,
                "roi_30d": roi_30d,
                "profit_today_yen": _to_int(bankroll.get("realized_profit_yen")),
                "available_bankroll_yen": _to_int(bankroll.get("available_bankroll_yen", 10000)),
                "open_stake_yen": _to_int(bankroll.get("open_stake_yen")),
                "pending_tickets": _to_int(bankroll.get("pending_tickets")),
                "runs_30d": runs_30d,
            }
        )
    if not compare_rows:
        return ""

    compare_rows = sorted(
        compare_rows,
        key=lambda row: (
            _to_int(row.get("profit_30d_yen")),
            _to_int(row.get("profit_today_yen")),
            _to_int(row.get("available_bankroll_yen")),
        ),
        reverse=True,
    )
    leader_30d = compare_rows[0]
    leader_today = sorted(compare_rows, key=lambda row: _to_int(row.get("profit_today_yen")), reverse=True)[0]
    headline_rows = [
        {
            "metric": "30d leader",
            "value": f"{leader_30d.get('model', '')} ({leader_30d.get('profit_30d_yen', 0)} yen)",
        },
        {
            "metric": "today leader",
            "value": f"{leader_today.get('model', '')} ({leader_today.get('profit_today_yen', 0)} yen)",
        },
    ]
    compare_table = build_table_html(
        compare_rows,
        [
            "model",
            "profit_30d_yen",
            "roi_30d",
            "profit_today_yen",
            "available_bankroll_yen",
            "open_stake_yen",
            "pending_tickets",
            "runs_30d",
        ],
        "LLM Profit Compare",
    )
    headline_table = build_table_html(headline_rows, ["metric", "value"], "LLM Leaders")
    return f"{headline_table}{compare_table}"


def build_stats_block(
    scope_norm,
    *,
    load_predictor_summary,
    build_table_html,
    load_policy_bankroll_summary,
    load_policy_daily_profit_summary,
    build_daily_profit_chart_html,
):
    if not scope_norm:
        return ""

    predictor_rows = load_predictor_summary(scope_norm)
    policy_labels = {
        "gemini": "Gemini",
        "siliconflow": "SiliconFlow",
        "openai": "OpenAI GPT-5",
    }

    parts = []
    for policy_engine in ("gemini", "siliconflow", "openai"):
        title = policy_labels.get(policy_engine, policy_engine)
        bankroll = load_policy_bankroll_summary(policy_engine=policy_engine)
        daily_rows = load_policy_daily_profit_summary(days=30, policy_engine=policy_engine)
        if bankroll:
            parts.append(
                build_table_html(
                    [
                        {"metric": "date", "value": bankroll.get("ledger_date", "")},
                        {"metric": "start_bankroll_yen", "value": bankroll.get("start_bankroll_yen", "")},
                        {"metric": "available_bankroll_yen", "value": bankroll.get("available_bankroll_yen", "")},
                        {"metric": "open_stake_yen", "value": bankroll.get("open_stake_yen", "")},
                        {"metric": "realized_profit_yen", "value": bankroll.get("realized_profit_yen", "")},
                        {"metric": "pending_tickets", "value": bankroll.get("pending_tickets", "")},
                    ],
                    ["metric", "value"],
                    f"{title} Bankroll",
                )
            )
        if daily_rows:
            parts.append(build_daily_profit_chart_html(daily_rows, f"{title} Daily Profit"))
            parts.append(
                build_table_html(
                    daily_rows,
                    ["date", "runs", "profit_yen", "base_amount", "roi"],
                    f"{title} Daily Summary",
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

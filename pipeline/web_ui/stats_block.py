def build_stats_block(
    scope_norm,
    *,
    load_predictor_summary,
    build_table_html,
):
    if not scope_norm:
        return ""

    predictor_rows = load_predictor_summary(scope_norm)

    parts = []
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

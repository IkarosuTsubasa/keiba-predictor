def api_runs_response(*, scope_key, limit, query, normalize_scope_key, default_limit, max_limit, get_recent_runs):
    scope_key = normalize_scope_key(scope_key) or "central_dirt"
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = default_limit
    limit = max(1, min(max_limit, limit))
    return get_recent_runs(scope_key, limit=limit, query=query)

import json
from datetime import datetime


def apply_local_ticket_plan_fallback(output_dict, candidate_lookup, race_budget_yen):
    return dict(output_dict or {})


def build_policy_ticket_rows(policy_output, candidate_lookup, horse_map, policy_engine):
    output_dict = dict(policy_output or {})
    tickets = []
    for item in list(output_dict.get("ticket_plan", []) or []):
        bet_type = str(item.get("bet_type", "") or "").strip().lower()
        legs = [str(x or "").strip() for x in list(item.get("legs", []) or []) if str(x or "").strip()]
        stake_yen = max(0, int(item.get("stake_yen", 0) or 0))
        if not bet_type or not legs or stake_yen <= 0:
            continue
        lookup_legs = sorted(legs, key=lambda x: int(x) if x.isdigit() else x) if bet_type in ("wide", "quinella", "trio") else legs
        ticket_key = f"{bet_type}:{'-'.join(lookup_legs)}"
        candidate = dict(candidate_lookup.get(ticket_key, {}) or {})
        if not candidate:
            continue
        odds_used = round(float(candidate.get("odds_used", 0.0) or 0.0), 6)
        p_hit = round(float(candidate.get("p_hit", 0.0) or 0.0), 6)
        ev = round(float(candidate.get("ev", 0.0) or 0.0), 6)
        horse_names = []
        for leg in legs:
            horse = dict(horse_map.get(leg, {}) or {})
            horse_names.append(str(horse.get("horse_name", "") or leg))
        expected_return_yen = int(round(stake_yen * max(0.0, odds_used))) if odds_used > 0 else 0
        tickets.append(
            {
                "ticket_id": ticket_key,
                "budget_yen": 0,
                "bet_type": bet_type,
                "horse_no": "-".join(legs),
                "horse_name": " / ".join(horse_names),
                "units": max(1, stake_yen // 100),
                "amount_yen": stake_yen,
                "hit_prob_est": p_hit,
                "hit_prob_se": "",
                "hit_prob_ci95_low": "",
                "hit_prob_ci95_high": "",
                "payout_mult": odds_used,
                "ev_ratio_est": round(p_hit * odds_used, 6) if p_hit > 0 else 0.0,
                "expected_return_yen": expected_return_yen,
                "odds_used": odds_used,
                "p_hit": p_hit,
                "edge": ev,
                "kelly_f": 0.0,
                "score": round(float(candidate.get("score", 0.0) or 0.0), 6),
                "stake_yen": stake_yen,
                "notes": "policy_pool=shared;policy={buy_style};decision={decision};construction={construction};reasons={reasons}".format(
                    buy_style=str(output_dict.get("buy_style", "") or ""),
                    decision=str(output_dict.get("bet_decision", "") or ""),
                    construction=str(output_dict.get("strategy_mode", "") or ""),
                    reasons=",".join(str(x) for x in list(output_dict.get("reason_codes", []) or [])),
                ),
                "strategy_text_ja": str(output_dict.get("strategy_text_ja", "") or ""),
                "bet_tendency_ja": str(output_dict.get("bet_tendency_ja", "") or ""),
                "policy_engine": policy_engine,
                "policy_buy_style": str(output_dict.get("buy_style", "") or ""),
                "policy_bet_decision": str(output_dict.get("bet_decision", "") or ""),
                "policy_construction_style": str(output_dict.get("strategy_mode", "") or ""),
            }
        )
    return tickets


def normalize_output_ticket_plan_from_rows(ticket_rows):
    out = []
    for ticket in list(ticket_rows or []):
        bet_type = str(ticket.get("bet_type", "") or "").strip().lower()
        legs = [str(x).strip() for x in str(ticket.get("horse_no", "") or "").split("-") if str(x).strip()]
        stake_yen = int(ticket.get("stake_yen", ticket.get("amount_yen", 0)) or 0)
        if not bet_type or not legs or stake_yen <= 0:
            continue
        out.append({"bet_type": bet_type, "legs": legs, "stake_yen": stake_yen})
    return out


def append_output_warning(output_dict, code):
    warnings = [str(x).strip() for x in list((output_dict or {}).get("warnings", []) or []) if str(x).strip()]
    text = str(code or "").strip()
    if text and text not in warnings:
        warnings.append(text)
    output_dict["warnings"] = warnings
    return output_dict


def set_output_no_bet(output_dict):
    output_dict["bet_decision"] = "no_bet"
    output_dict["participation_level"] = "no_bet"
    output_dict["buy_style"] = "no_bet"
    output_dict["strategy_mode"] = "no_bet"
    output_dict["enabled_bet_types"] = []
    output_dict["key_horses"] = []
    output_dict["secondary_horses"] = []
    output_dict["longshot_horses"] = []
    output_dict["max_ticket_count"] = 0
    output_dict["ticket_plan"] = []
    return output_dict


def save_policy_payload(*, base_dir, scope_key, run_id, race_id, payload, policy_engine, normalize_scope_key, get_data_dir, normalize_policy_engine):
    scope_norm = normalize_scope_key(scope_key)
    if not scope_norm:
        return None
    race_dir = get_data_dir(base_dir, scope_norm) / str(race_id or "")
    race_dir.mkdir(parents=True, exist_ok=True)
    engine = normalize_policy_engine(policy_engine)
    path = race_dir / f"{engine}_policy_{run_id}_{race_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def execute_policy_buy(
    *,
    base_dir,
    scope_key,
    run_row,
    run_id,
    policy_engine="gemini",
    policy_model="",
    normalize_scope_key,
    normalize_policy_engine,
    resolve_policy_model,
    default_gemini_model,
    resolve_pred_path,
    resolve_run_asset_path,
    build_policy_input_payload,
    call_policy,
    resolve_policy_timeout,
    get_last_call_meta,
    reserve_run_tickets,
    summarize_bankroll,
    update_run_row_fields,
    get_data_dir,
):
    scope_norm = normalize_scope_key(scope_key)
    engine = normalize_policy_engine(policy_engine)
    resolved_model = resolve_policy_model(engine, policy_model, default_gemini_model)
    pred_path = resolve_pred_path(scope_norm, run_id, run_row)
    odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "odds_path", "odds")
    fuku_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "fuku_odds_path", "fuku_odds")
    wide_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "wide_odds_path", "wide_odds")
    quinella_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "quinella_odds_path", "quinella_odds")
    exacta_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "exacta_odds_path", "exacta_odds")
    trio_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "trio_odds_path", "trio_odds")
    trifecta_odds_path = resolve_run_asset_path(scope_norm, run_id, run_row, "trifecta_odds_path", "trifecta_odds")
    context, error = build_policy_input_payload(
        scope_norm,
        run_id,
        run_row,
        pred_path,
        odds_path,
        fuku_odds_path,
        wide_odds_path,
        quinella_odds_path,
        exacta_odds_path,
        trio_odds_path,
        trifecta_odds_path,
        engine,
    )
    if error:
        raise ValueError(error)
    policy_output = call_policy(
        input=context["input"],
        policy_engine=engine,
        model=resolved_model,
        timeout_s=resolve_policy_timeout(engine),
        cache_enable=True,
    )
    meta = get_last_call_meta()
    output_dict = policy_output.model_dump() if hasattr(policy_output, "model_dump") else policy_output.dict()
    output_dict = apply_local_ticket_plan_fallback(
        output_dict,
        context["candidate_lookup"],
        context["input"]["constraints"]["race_budget_yen"],
    )
    requested_ticket_count = len(list(output_dict.get("ticket_plan", []) or []))
    tickets = build_policy_ticket_rows(output_dict, context["candidate_lookup"], context["horse_map"], engine)
    output_dict["ticket_plan"] = normalize_output_ticket_plan_from_rows(tickets)
    if len(tickets) < requested_ticket_count:
        output_dict = append_output_warning(output_dict, "INVALID_TICKET_DROPPED")
    if str(output_dict.get("bet_decision", "") or "").strip().lower() == "bet" and not tickets:
        output_dict = append_output_warning(output_dict, "NO_EXECUTABLE_TICKETS")
        output_dict = set_output_no_bet(output_dict)
    reserve_run_tickets(
        base_dir,
        run_id=run_id,
        scope_key=scope_norm,
        race_id=str((run_row or {}).get("race_id", "") or ""),
        ledger_date=context["ledger_date"],
        tickets=tickets,
        policy_engine=engine,
    )
    summary_after = summarize_bankroll(base_dir, context["ledger_date"], policy_engine=engine)
    payload = {
        "scope": scope_norm,
        "race_id": str((run_row or {}).get("race_id", "") or ""),
        "run_id": str(run_id or ""),
        "policy_engine": engine,
        "policy_model": resolved_model,
        "gemini_model": resolved_model if engine == "gemini" else "",
        "policy_budget_reuse": False,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "budgets": [
            {
                "budget_yen": 0,
                "shared_policy": True,
                "output": output_dict,
                "meta": meta,
                "portfolio": {
                    "ledger_date": context["ledger_date"],
                    "before": context["summary_before"],
                    "after": summary_after,
                },
                "tickets": tickets,
            }
        ],
    }
    path = save_policy_payload(
        base_dir=base_dir,
        scope_key=scope_norm,
        run_id=run_id,
        race_id=(run_row or {}).get("race_id", ""),
        payload=payload,
        policy_engine=engine,
        normalize_scope_key=normalize_scope_key,
        get_data_dir=get_data_dir,
        normalize_policy_engine=normalize_policy_engine,
    )
    updates = {f"{engine}_policy_path": str(path or "")}
    if engine == "gemini":
        updates["gemini_policy_path"] = str(path or "")
        updates["tickets"] = str(len(tickets))
        updates["amount_yen"] = str(sum(int(ticket.get("amount_yen", 0) or 0) for ticket in tickets))
    elif engine == "openai":
        updates["openai_policy_path"] = str(path or "")
    elif engine == "grok":
        updates["grok_policy_path"] = str(path or "")
    update_run_row_fields(scope_norm, run_row, updates)
    output_lines = [
        f"[llm_buy] run_id={run_id} engine={engine} model={resolved_model}",
        f"[policy_save] path={path}" if path else "[policy_save] path=",
        (
            "[tickets] count={count} amount_yen={amount} decision={decision} buy_style={buy_style}".format(
                count=len(tickets),
                amount=sum(int(ticket.get("amount_yen", 0) or 0) for ticket in tickets),
                decision=str(output_dict.get("bet_decision", "") or ""),
                buy_style=str(output_dict.get("buy_style", "") or ""),
            )
        ),
        (
            "[policy_meta] cache_hit={cache_hit} llm_latency_ms={latency} fallback_reason={fallback} error_detail={detail}".format(
                cache_hit=int(bool(meta.get("cache_hit", False))),
                latency=int(meta.get("llm_latency_ms", 0) or 0),
                fallback=str(meta.get("fallback_reason", "") or ""),
                detail=str(meta.get("error_detail", "") or ""),
            )
        ),
    ]
    for ticket in tickets:
        output_lines.append(
            "[ticket] {bet_type} {horse_no} {horse_name} stake={stake_yen} odds={odds_used} p_hit={p_hit}".format(
                bet_type=str(ticket.get("bet_type", "") or ""),
                horse_no=str(ticket.get("horse_no", "") or ""),
                horse_name=str(ticket.get("horse_name", "") or ""),
                stake_yen=int(ticket.get("stake_yen", 0) or 0),
                odds_used=str(ticket.get("odds_used", "") or ""),
                p_hit=str(ticket.get("p_hit", "") or ""),
            )
        )
    return {
        "engine": engine,
        "model": resolved_model,
        "summary_before": context["summary_before"],
        "summary_after": summary_after,
        "payload_path": str(path or ""),
        "tickets": tickets,
        "meta": meta,
        "output_text": "\n".join(line for line in output_lines if str(line).strip()),
    }

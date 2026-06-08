import json
import os
import re
from datetime import date, timedelta
from pathlib import Path


MARK_ORDER = ("◎", "○", "▲", "△", "☆")
COURSE_ORDER = (
    "札幌",
    "函館",
    "福島",
    "新潟",
    "東京",
    "中山",
    "中京",
    "京都",
    "阪神",
    "小倉",
)
TRACK_CONDITION_LABELS = {
    "稍": "稍重",
    "稍重": "稍重",
    "良": "良",
    "重": "重",
    "不": "不良",
    "不良": "不良",
}
CONFIDENCE_SCORE = {
    "high": 0.82,
    "medium": 0.62,
    "low": 0.38,
}


def _repo_root(base_dir):
    return Path(base_dir).resolve().parent


def _configured_dir(env_key):
    text = str(os.environ.get(env_key, "") or "").strip()
    return Path(text) if text else None


def agent_prediction_write_dir(base_dir):
    configured = _configured_dir("KEIBA_AGENT_PREDICTIONS_DIR")
    if configured is not None:
        return configured.resolve()
    return Path(base_dir).resolve() / "data" / "agent_predictions"


def agent_result_write_dir(base_dir):
    configured = _configured_dir("KEIBA_AGENT_RESULTS_DIR")
    if configured is not None:
        return configured.resolve()
    return Path(base_dir).resolve() / "data" / "agent_results"


def _existing_unique_dirs(candidates):
    out = []
    seen = set()
    for path in candidates:
        resolved = Path(path).resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            out.append(resolved)
    return out


def _prediction_dir_candidates(base_dir):
    root = _repo_root(base_dir)
    candidates = [
        agent_prediction_write_dir(base_dir),
        Path(base_dir).resolve() / "data" / "predictions",
        root / "data" / "predictions",
        root / "keiba_llm_agent" / "data" / "predictions",
    ]
    return _existing_unique_dirs(candidates)


def _results_dir_candidates(base_dir):
    root = _repo_root(base_dir)
    candidates = [
        agent_result_write_dir(base_dir),
        Path(base_dir).resolve() / "data" / "results",
        root / "data" / "results",
        root / "keiba_llm_agent" / "data" / "results",
    ]
    return _existing_unique_dirs(candidates)


def agent_prediction_path_for_race(base_dir, race_id):
    race_id_text = _safe_text(race_id)
    if not race_id_text:
        return None
    for directory in _prediction_dir_candidates(base_dir):
        path = directory / f"{race_id_text}.json"
        if path.exists():
            return path
    return agent_prediction_write_dir(base_dir) / f"{race_id_text}.json"


def agent_result_path_for_race(base_dir, race_id):
    race_id_text = _safe_text(race_id)
    if not race_id_text:
        return None
    for directory in _results_dir_candidates(base_dir):
        path = directory / f"{race_id_text}.json"
        if path.exists():
            return path
    return agent_result_write_dir(base_dir) / f"{race_id_text}.json"


def _safe_text(value):
    return str(value or "").strip()


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_prediction_payloads(base_dir):
    seen_race_ids = set()
    for directory in _prediction_dir_candidates(base_dir):
        for path in sorted(directory.glob("*.json")):
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            race_id = _safe_text(payload.get("race_id")) or path.stem
            if not race_id or race_id in seen_race_ids:
                continue
            seen_race_ids.add(race_id)
            payload["_source_path"] = str(path)
            yield payload


def available_dates(base_dir, scope_key=""):
    dates = set()
    for payload in _iter_prediction_payloads(base_dir):
        if not _scope_matches(payload, scope_key):
            continue
        race_info = dict(payload.get("race_info") or {})
        race_date = _safe_text(race_info.get("race_date"))
        if race_date:
            dates.add(race_date)
    return sorted(dates)


def _normalize_track_condition(value):
    text = _safe_text(value)
    return TRACK_CONDITION_LABELS.get(text, text)


def _race_no(race_id):
    match = re.search(r"(\d{2})$", _safe_text(race_id))
    if not match:
        return 0
    return _safe_int(match.group(1), 0)


def _race_no_text(race_id):
    race_no = _race_no(race_id)
    return f"{race_no}R" if race_no > 0 else ""


def _course_sort_value(course):
    text = _safe_text(course)
    try:
        return COURSE_ORDER.index(text)
    except ValueError:
        return len(COURSE_ORDER)


def _scope_matches(payload, scope_key):
    scope = _safe_text(scope_key)
    if not scope:
        return True
    race_info = dict(payload.get("race_info") or {})
    payload_scope = _safe_text(payload.get("scope_key")) or _safe_text(race_info.get("scope_key"))
    payload_source = _safe_text(payload.get("source")) or _safe_text(race_info.get("source"))
    if payload_scope:
        if scope == "local":
            return payload_scope == "local"
        if scope in ("central_turf", "central_dirt"):
            return payload_scope == scope
    if payload_source == "local":
        return scope == "local"
    surface = _safe_text(race_info.get("surface"))
    course = _safe_text(race_info.get("course"))
    if scope == "central_turf":
        return "芝" in surface
    if scope == "central_dirt":
        return "ダ" in surface or "砂" in surface
    if scope == "local":
        return course and course not in COURSE_ORDER
    return True


def _horse_map(payload):
    out = {}
    for item in list(payload.get("horse_scores") or []):
        if not isinstance(item, dict):
            continue
        horse_no = _safe_int(item.get("horse_no"), 0)
        if horse_no <= 0:
            continue
        out[horse_no] = item
    return out


def _ranked_horses(payload):
    rows = [dict(item or {}) for item in list(payload.get("horse_scores") or []) if isinstance(item, dict)]
    return sorted(
        rows,
        key=lambda item: (
            -_safe_float(item.get("total_score"), 0.0),
            _safe_int(item.get("horse_no"), 999),
        ),
    )


def _top_horse_items(payload, limit=5):
    horses = _horse_map(payload)
    rows = []
    seen = set()
    marks = dict(payload.get("marks") or {})
    for rank, symbol in enumerate(MARK_ORDER, start=1):
        horse_no = _safe_int(marks.get(symbol), 0)
        if horse_no <= 0 or horse_no in seen:
            continue
        item = dict(horses.get(horse_no) or {})
        if not item:
            continue
        rows.append(_serialize_top_horse(item, rank=rank, symbol=symbol))
        seen.add(horse_no)

    if len(rows) < limit:
        for item in _ranked_horses(payload):
            horse_no = _safe_int(item.get("horse_no"), 0)
            if horse_no <= 0 or horse_no in seen:
                continue
            rows.append(_serialize_top_horse(item, rank=len(rows) + 1, symbol=""))
            seen.add(horse_no)
            if len(rows) >= limit:
                break
    return rows[:limit]


def _support_score(item):
    total_score = _safe_float(item.get("total_score"), 0.0)
    return max(1, min(99, int(round(total_score * 3.0))))


def _serialize_top_horse(item, rank=0, symbol=""):
    return {
        "horse_no": str(_safe_int(item.get("horse_no"), 0) or ""),
        "horse_name": _safe_text(item.get("horse_name")),
        "pred_rank": int(rank or 0),
        "mark": _safe_text(symbol),
        "total_score": round(_safe_float(item.get("total_score"), 0.0), 3),
        "support_score": _support_score(item),
        "top3_prob_model": 0.0,
        "rank_score_norm": round(max(0.05, 1.0 - max(0, int(rank or 1) - 1) * 0.18), 6),
        "win_odds": round(_safe_float(item.get("odds"), 0.0), 3),
        "popularity": _safe_int(item.get("popularity"), 0),
        "reason": _safe_text(item.get("reason")),
    }


def _marks_text(payload):
    marks = dict(payload.get("marks") or {})
    parts = []
    for symbol in MARK_ORDER:
        horse_no = _safe_int(marks.get(symbol), 0)
        if horse_no > 0:
            parts.append(f"{symbol}{horse_no}")
    return " ".join(parts)


def _strategy_confidence(payload):
    strategy = dict(payload.get("strategy") or {})
    confidence = _safe_text(strategy.get("confidence")).lower()
    return CONFIDENCE_SCORE.get(confidence, 0.5)


def _distance_label(race_info):
    surface = _safe_text(race_info.get("surface"))
    distance = _safe_int(race_info.get("distance"), 0)
    if surface and distance > 0:
        return f"{surface}{distance}m"
    if distance > 0:
        return f"{distance}m"
    return surface


def _actual_result_for_prediction(base_dir, payload):
    race_id = _safe_text(payload.get("race_id"))
    if not race_id:
        return "結果未確定", {"is_settled": False, "top3": []}

    result_payload = None
    for directory in _results_dir_candidates(base_dir):
        path = directory / f"{race_id}.json"
        if not path.exists():
            continue
        try:
            result_payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            result_payload = None
        if result_payload:
            break
    if not result_payload:
        return "結果未確定", {"is_settled": False, "top3": []}

    result = dict(result_payload.get("result") or {})
    top_nos = [
        _safe_int(result.get("1st"), 0),
        _safe_int(result.get("2nd"), 0),
        _safe_int(result.get("3rd"), 0),
    ]
    if not all(top_nos):
        return "結果未確定", {"is_settled": False, "top3": []}

    horses = _horse_map(payload)
    top3 = []
    parts = []
    for index, horse_no in enumerate(top_nos, start=1):
        horse = dict(horses.get(horse_no) or {})
        horse_name = _safe_text(horse.get("horse_name"))
        label = f"{horse_no} {horse_name}".strip()
        parts.append(f"{index}着 {label}")
        top3.append(
            {
                "rank": index,
                "horse_no": str(horse_no),
                "horse_name": horse_name,
            }
        )
    return " / ".join(parts), {"is_settled": True, "top3": top3}


def _actual_top_nos_for_race(base_dir, race_id):
    race_id = _safe_text(race_id)
    if not race_id:
        return []
    for directory in _results_dir_candidates(base_dir):
        path = directory / f"{race_id}.json"
        if not path.exists():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        result = dict(payload.get("result") or {})
        top_nos = [
            _safe_int(result.get("1st"), 0),
            _safe_int(result.get("2nd"), 0),
            _safe_int(result.get("3rd"), 0),
        ]
        return [horse_no for horse_no in top_nos if horse_no > 0]
    return []


def _parse_date(value):
    text = _safe_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _percent_text(value, denominator):
    if denominator <= 0:
        return "-"
    return f"{round(float(value) / float(denominator) * 100.0, 1)}%"


def _mark_numbers(payload, limit=5):
    marks = dict(payload.get("marks") or {})
    out = []
    for symbol in MARK_ORDER[:limit]:
        horse_no = _safe_int(marks.get(symbol), 0)
        if horse_no > 0:
            out.append(horse_no)
    return out


def _history_record(base_dir, payload):
    race_info = dict(payload.get("race_info") or {})
    race_id = _safe_text(payload.get("race_id")) or _safe_text(race_info.get("race_id"))
    race_date = _safe_text(race_info.get("race_date"))
    actual_top3 = _actual_top_nos_for_race(base_dir, race_id)
    top3_marks = _mark_numbers(payload, limit=3)
    top5_marks = _mark_numbers(payload, limit=5)
    main_horse = top5_marks[0] if top5_marks else 0
    strategy = dict(payload.get("strategy") or {})
    return {
        "race_id": race_id,
        "race_date": race_date,
        "course": _safe_text(race_info.get("course")),
        "race_title": f"{_safe_text(race_info.get('course'))}{_race_no_text(race_id)}".strip() or race_id,
        "race_name": _safe_text(race_info.get("race_name")),
        "main_horse": main_horse,
        "top3_marks": top3_marks,
        "top5_marks": top5_marks,
        "actual_top3": actual_top3,
        "settled": len(actual_top3) >= 3,
        "bet_decision": _safe_text(strategy.get("bet_decision")),
    }


def _summarize_history_records(records):
    rows = [dict(item or {}) for item in list(records or [])]
    settled = [item for item in rows if item.get("settled")]
    settled_count = len(settled)
    main_win = 0
    main_top3 = 0
    top3_cover_hits = 0
    top5_cover_hits = 0
    top3_exact = 0
    bet_count = 0
    skip_count = 0
    for item in rows:
        decision = _safe_text(item.get("bet_decision")).upper()
        if decision == "BET":
            bet_count += 1
        if decision == "SKIP":
            skip_count += 1
    for item in settled:
        actual_top3 = list(item.get("actual_top3") or [])[:3]
        actual_set = set(actual_top3)
        main_horse = _safe_int(item.get("main_horse"), 0)
        top3_marks = set(list(item.get("top3_marks") or [])[:3])
        top5_marks = set(list(item.get("top5_marks") or [])[:5])
        if main_horse and actual_top3 and main_horse == actual_top3[0]:
            main_win += 1
        if main_horse and main_horse in actual_set:
            main_top3 += 1
        top3_cover_hits += len(actual_set.intersection(top3_marks))
        top5_cover_hits += len(actual_set.intersection(top5_marks))
        if len(actual_set) == 3 and actual_set == top3_marks:
            top3_exact += 1

    top3_denominator = settled_count * 3
    return {
        "predicted_races": len(rows),
        "settled_races": settled_count,
        "pending_races": max(0, len(rows) - settled_count),
        "bet_races": bet_count,
        "skip_races": skip_count,
        "main_win_count": main_win,
        "main_top3_count": main_top3,
        "top3_cover_hits": top3_cover_hits,
        "top5_cover_hits": top5_cover_hits,
        "top3_exact_count": top3_exact,
        "main_win_rate_text": _percent_text(main_win, settled_count),
        "main_top3_rate_text": _percent_text(main_top3, settled_count),
        "top3_cover_rate_text": _percent_text(top3_cover_hits, top3_denominator),
        "top5_cover_rate_text": _percent_text(top5_cover_hits, top3_denominator),
        "top3_exact_rate_text": _percent_text(top3_exact, settled_count),
    }


def _history_group_rows(records, key):
    grouped = {}
    for item in list(records or []):
        group_key = _safe_text(item.get(key)) or "-"
        grouped.setdefault(group_key, []).append(item)
    rows = []
    for group_key, group_records in grouped.items():
        summary = _summarize_history_records(group_records)
        summary[key] = group_key
        rows.append(summary)
    return rows


def build_history_payload(base_dir, target_date="", scope_key=""):
    all_records = []
    target = _parse_date(target_date)
    if target is None:
        dates = available_dates(base_dir)
        target = _parse_date(dates[-1] if dates else "")

    for payload in _iter_prediction_payloads(base_dir):
        if not _scope_matches(payload, scope_key):
            continue
        race_info = dict(payload.get("race_info") or {})
        record_date = _parse_date(race_info.get("race_date"))
        if record_date is None:
            continue
        if target is not None and record_date > target:
            continue
        all_records.append(_history_record(base_dir, payload))

    def period_records(days):
        if target is None or days <= 0:
            return list(all_records)
        start = target - timedelta(days=days - 1)
        return [
            item
            for item in all_records
            if (parsed := _parse_date(item.get("race_date"))) is not None and start <= parsed <= target
        ]

    periods = {
        "days_30": period_records(30),
        "days_365": period_records(365),
        "all_time": list(all_records),
    }
    out_periods = {}
    for period_key, records in periods.items():
        daily_rows = _history_group_rows(records, "race_date")
        daily_rows.sort(key=lambda item: _safe_text(item.get("race_date")), reverse=True)
        course_rows = _history_group_rows(records, "course")
        course_rows.sort(
            key=lambda item: (
                -int(item.get("settled_races", 0) or 0),
                _course_sort_value(item.get("course")),
                _safe_text(item.get("course")),
            )
        )
        out_periods[period_key] = {
            **_summarize_history_records(records),
            "daily_rows": daily_rows[:12],
            "course_rows": course_rows,
        }

    return {
        "available": bool(all_records),
        "target_date": target.isoformat() if target is not None else "",
        "periods": out_periods,
    }


def _agent_prediction_detail(payload):
    strategy = dict(payload.get("strategy") or {})
    pace = dict(payload.get("race_pace_projection") or {})
    simulation = dict(payload.get("race_simulation") or {})
    return {
        "source": "keiba_llm_agent",
        "scoring_profile": _safe_text(payload.get("scoring_profile")),
        "scoring_mode": _safe_text(payload.get("scoring_mode")),
        "marks": dict(payload.get("marks") or {}),
        "marks_text": _marks_text(payload),
        "summary": _safe_text(payload.get("summary")),
        "commentary": _safe_text(payload.get("commentary")),
        "risks": [_safe_text(item) for item in list(payload.get("risks") or []) if _safe_text(item)],
        "strategy": {
            "bet_decision": _safe_text(strategy.get("bet_decision")),
            "confidence": _safe_text(strategy.get("confidence")),
            "participation_level": _safe_text(strategy.get("participation_level")),
            "reason_codes": list(strategy.get("reason_codes") or []),
            "reason": _safe_text(strategy.get("reason")),
        },
        "bets": [
            {
                "bet_type": _safe_text(item.get("bet_type")),
                "horse_numbers": [_safe_int(value, 0) for value in list(item.get("horse_numbers") or []) if _safe_int(value, 0) > 0],
                "amount": _safe_int(item.get("amount"), 0),
                "reason": _safe_text(item.get("reason")),
            }
            for item in list(payload.get("bets") or [])
            if isinstance(item, dict)
        ],
        "top_horses": _top_horse_items(payload, limit=10),
        "pace_projection": {
            "projected_pace": _safe_text(pace.get("projected_pace")),
            "reason": _safe_text(pace.get("reason")),
            "favorable_styles": list(pace.get("favorable_styles") or []),
        },
        "simulation_summary": _safe_text(simulation.get("reasoning_summary")),
    }


def _public_card(payload):
    return {
        "predictor_id": "agent",
        "label": "AI予測",
        "marks_text": _marks_text(payload),
        "top_horses": _top_horse_items(payload, limit=len(MARK_ORDER)),
        "metaLabel": "判定",
        "metaValue": _safe_text((payload.get("strategy") or {}).get("confidence")) or "-",
    }


def _public_race_row(base_dir, payload, include_detail_fields=False):
    race_info = dict(payload.get("race_info") or {})
    race_id = _safe_text(payload.get("race_id")) or _safe_text(race_info.get("race_id"))
    course = _safe_text(race_info.get("course"))
    race_no_text = _race_no_text(race_id)
    race_title = f"{course}{race_no_text}".strip() if course else (race_no_text or race_id)
    actual_text, actual_result = _actual_result_for_prediction(base_dir, payload)
    row = {
        "source_type": "agent_prediction",
        "scope_key": "",
        "scope_label": "AI予測",
        "race_id": race_id,
        "race_title": race_title,
        "race_name": _safe_text(race_info.get("race_name")),
        "date_label": _safe_text(race_info.get("race_date")),
        "actual_text": actual_text,
        "actual_result": actual_result,
        "location": course,
        "scheduled_off_time": "",
        "distance_label": _distance_label(race_info),
        "track_condition": _normalize_track_condition(race_info.get("track_condition")),
        "run_id": race_id,
        "alias_ids": [value for value in [race_id, race_no_text, race_title] if value],
        "cards": [],
        "predictor_compare_cards": [_public_card(payload)],
        "top5": _top_horse_items(payload, limit=5),
        "predictor_top5": {"agent": _top_horse_items(payload, limit=5)},
        "confidence_score": _strategy_confidence(payload),
        "agreement_score": _strategy_confidence(payload),
        "condition_predictor_ranking": {},
        "_agent_sort_key": (_course_sort_value(course), _race_no(race_id), race_id),
    }
    if include_detail_fields:
        row["agent_prediction"] = _agent_prediction_detail(payload)
    return row


def build_public_races(base_dir, target_date="", scope_key="", include_detail_fields=False):
    target = _safe_text(target_date)
    if not target:
        return []
    rows = []
    for payload in _iter_prediction_payloads(base_dir):
        race_info = dict(payload.get("race_info") or {})
        if _safe_text(race_info.get("race_date")) != target:
            continue
        if not _scope_matches(payload, scope_key):
            continue
        row = _public_race_row(base_dir, payload, include_detail_fields=include_detail_fields)
        rows.append(row)
    rows.sort(key=lambda item: item.get("_agent_sort_key") or (999, 999, ""))
    for row in rows:
        row.pop("_agent_sort_key", None)
    return rows

import json
import math
import os
import re
import threading
import time
import gzip
from datetime import date, timedelta
from pathlib import Path

from confidence_policy import AGENT_CONFIDENCE_FALLBACK_SCORE, public_decision_tone


MARK_ORDER = ("◎", "○", "▲", "△", "☆")
PREDICTION_STAGE_EARLY = "early"
PREDICTION_STAGE_FINAL = "final"
PREDICTION_STAGE_LABELS = {
    PREDICTION_STAGE_EARLY: "早期予測",
    PREDICTION_STAGE_FINAL: "最終予測",
}
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

CONFIDENCE_FALLBACK_SCORE_BY_SCOPE = {
    "local": {
        "high": 0.88,
        "medium": 0.76,
        "low": 0.62,
    },
    "central_dirt": {
        "high": 0.84,
        "medium": 0.72,
        "low": 0.56,
    },
    "central_turf": {
        "high": 0.82,
        "medium": 0.70,
        "low": 0.54,
    },
    "central": {
        "high": 0.83,
        "medium": 0.71,
        "low": 0.55,
    },
}

CONFIDENCE_SHAPE_WEIGHT_BY_SCOPE = {
    "local": 0.55,
    "central_dirt": 0.65,
    "central_turf": 0.68,
    "central": 0.66,
}

CONFIDENCE_FALLBACK_LIMIT_BY_SCOPE = {
    "local": (0.46, 0.90),
    "central_dirt": (0.42, 0.88),
    "central_turf": (0.40, 0.86),
    "central": (0.40, 0.87),
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


def agent_backtest_audit_write_dir(base_dir):
    configured = _configured_dir("KEIBA_AGENT_BACKTEST_AUDIT_DIR")
    if configured is not None:
        return configured.resolve()
    return Path(base_dir).resolve() / "data" / "agent_backtest_audit"


def agent_backtest_audit_path_for_race(base_dir, race_id):
    race_id_text = _safe_text(race_id)
    if not race_id_text:
        return None
    return agent_backtest_audit_write_dir(base_dir) / f"{race_id_text}.json.gz"


def normalize_prediction_stage(value):
    text = _safe_text(value).lower()
    if text in {PREDICTION_STAGE_FINAL, "final_prediction", "finalized"}:
        return PREDICTION_STAGE_FINAL
    return PREDICTION_STAGE_EARLY


def prediction_stage_label(value):
    return PREDICTION_STAGE_LABELS[normalize_prediction_stage(value)]


def _safe_version_id(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", _safe_text(value)).strip("._")
    return text or str(time.time_ns())


def agent_prediction_version_path_for_race(base_dir, race_id, stage, version_id):
    race_id_text = _safe_text(race_id)
    if not race_id_text:
        return None
    return (
        agent_prediction_write_dir(base_dir)
        / "_versions"
        / race_id_text
        / normalize_prediction_stage(stage)
        / f"{_safe_version_id(version_id)}.json"
    )


def agent_backtest_audit_version_path_for_race(base_dir, race_id, stage, version_id):
    race_id_text = _safe_text(race_id)
    if not race_id_text:
        return None
    return (
        agent_backtest_audit_write_dir(base_dir)
        / "_versions"
        / race_id_text
        / normalize_prediction_stage(stage)
        / f"{_safe_version_id(version_id)}.json.gz"
    )


def _utc_now_text():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_bytes(path, content):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temp_path.write_bytes(bytes(content or b""))
        temp_path.replace(target)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path, payload, *, compressed=False):
    raw = (json.dumps(dict(payload or {}), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(path, gzip.compress(raw) if compressed else raw)


def _audit_capabilities(race_data_payload=None, prediction_payload=None, result_payload=None):
    has_recent_runs = False
    if isinstance(race_data_payload, dict):
        for horse in list(race_data_payload.get("horses") or []):
            if isinstance(horse, dict) and horse.get("recent_runs"):
                has_recent_runs = True
                break
    return {
        "can_reweight_saved_scores": isinstance(prediction_payload, dict),
        "can_recompute_scoring_logic": isinstance(race_data_payload, dict),
        "can_recompute_from_recent_runs": has_recent_runs,
        "has_result_for_evaluation": isinstance(result_payload, dict),
    }


def _audit_missing_inputs(race_data_payload=None, prediction_payload=None, result_payload=None):
    missing = []
    if not isinstance(prediction_payload, dict):
        missing.append("prediction")
    if not isinstance(race_data_payload, dict):
        missing.append("race_data")
    elif not _audit_capabilities(race_data_payload=race_data_payload).get("can_recompute_from_recent_runs"):
        missing.append("recent_runs")
    if not isinstance(result_payload, dict):
        missing.append("result")
    return missing


def build_prediction_only_backtest_audit(prediction_payload, *, task=None, source_paths=None):
    prediction_payload = dict(prediction_payload or {})
    race_info = dict(prediction_payload.get("race_info") or {})
    race_id = _safe_text(prediction_payload.get("race_id")) or _safe_text(race_info.get("race_id"))
    task_row = dict(task or {})
    task_meta = dict(task_row.get("bundle_meta") or {})
    stage = normalize_prediction_stage(
        prediction_payload.get("prediction_stage") or task_meta.get("prediction_stage")
    )
    return {
        "schema_version": "agent_backtest_audit_v1",
        "audit_kind": "prediction_only",
        "created_at": _utc_now_text(),
        "source": "render_callback",
        "race_id": race_id,
        "prediction_stage": stage,
        "prediction_stage_label": prediction_stage_label(stage),
        "race_info": race_info,
        "source_paths": dict(source_paths or {}),
        "remote_task": {
            "task_id": _safe_text(task_row.get("task_id")),
            "job_id": _safe_text(task_row.get("job_id")),
            "scope_key": _safe_text(task_row.get("scope_key")),
            "prediction_stage": stage,
        },
        "backtest_capabilities": _audit_capabilities(prediction_payload=prediction_payload),
        "missing_inputs": _audit_missing_inputs(prediction_payload=prediction_payload),
        "inputs": {
            "race_data": None,
            "lessons": list(prediction_payload.get("used_lessons") or []),
        },
        "scoring": {
            "scoring_profile": prediction_payload.get("scoring_profile"),
            "scoring_mode": prediction_payload.get("scoring_mode"),
            "scoring_config": prediction_payload.get("scoring_config") or {},
            "market_signal_config": prediction_payload.get("market_signal_config") or {},
            "borderline_recovery_config": prediction_payload.get("borderline_recovery_config") or {},
            "horse_scores": list(prediction_payload.get("horse_scores") or []),
            "marks": dict(prediction_payload.get("marks") or {}),
        },
        "analyses": {
            "deep_analyses": list(prediction_payload.get("deep_analyses") or []),
            "pedigree_analyses": list(prediction_payload.get("pedigree_analyses") or []),
            "race_level_analyses": list(prediction_payload.get("race_level_analyses") or []),
            "pace_analyses": list(prediction_payload.get("pace_analyses") or []),
            "race_pace_projection": prediction_payload.get("race_pace_projection"),
            "race_simulation": prediction_payload.get("race_simulation"),
            "borderline_recovery_result": prediction_payload.get("borderline_recovery_result") or {},
        },
        "outcomes": {
            "result": None,
        },
        "outputs": {
            "prediction": prediction_payload,
        },
    }


def write_agent_backtest_audit_payload(base_dir, race_id, payload):
    path = agent_backtest_audit_path_for_race(base_dir, race_id)
    if path is None:
        raise ValueError("race_id missing for agent backtest audit")
    _atomic_write_json(path, payload, compressed=True)
    return path


def write_agent_backtest_audit_bytes(base_dir, race_id, content_bytes):
    path = agent_backtest_audit_path_for_race(base_dir, race_id)
    if path is None:
        raise ValueError("race_id missing for agent backtest audit")
    _atomic_write_bytes(path, content_bytes)
    return path


def write_agent_prediction_payload(base_dir, race_id, payload):
    race_id_text = _safe_text(race_id)
    if not race_id_text:
        raise ValueError("race_id missing for agent prediction")
    path = agent_prediction_write_dir(base_dir) / f"{race_id_text}.json"
    _atomic_write_json(path, payload)
    clear_agent_prediction_caches()
    return path


def write_agent_prediction_version_payload(base_dir, race_id, stage, version_id, payload):
    path = agent_prediction_version_path_for_race(base_dir, race_id, stage, version_id)
    if path is None:
        raise ValueError("race_id missing for agent prediction version")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _atomic_write_json(path, payload)
    clear_agent_prediction_caches()
    return path


def write_agent_backtest_audit_version_payload(base_dir, race_id, stage, version_id, payload):
    path = agent_backtest_audit_version_path_for_race(base_dir, race_id, stage, version_id)
    if path is None:
        raise ValueError("race_id missing for agent backtest audit version")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _atomic_write_json(path, payload, compressed=True)
    return path


def write_agent_backtest_audit_version_bytes(base_dir, race_id, stage, version_id, content_bytes):
    path = agent_backtest_audit_version_path_for_race(base_dir, race_id, stage, version_id)
    if path is None:
        raise ValueError("race_id missing for agent backtest audit version")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _atomic_write_bytes(path, content_bytes)
    return path


def read_agent_backtest_audit_payload(path):
    audit_path = Path(path)
    if audit_path.suffix == ".gz":
        with gzip.open(audit_path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(audit_path.read_text(encoding="utf-8"))


def attach_result_to_agent_backtest_audit(base_dir, race_id, result_payload):
    race_id_text = _safe_text(race_id)
    if not race_id_text:
        return None
    root = agent_backtest_audit_write_dir(base_dir)
    paths = []
    legacy_path = agent_backtest_audit_path_for_race(base_dir, race_id_text)
    if legacy_path is not None and legacy_path.exists():
        paths.append(legacy_path)
    version_root = root / "_versions" / race_id_text
    if version_root.exists():
        paths.extend(sorted(version_root.glob("*/*.json.gz")))

    updated_paths = []
    for path in paths:
        try:
            payload = read_agent_backtest_audit_payload(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload:
            continue
        payload.setdefault("outcomes", {})["result"] = dict(result_payload or {})
        if (payload.get("shadow_predictions") or {}).get("public_workout") is not None:
            try:
                from workout_shadow_metrics import attach_workout_comparison

                attach_workout_comparison(payload, result_payload)
            except Exception as exc:
                payload["outcomes"]["public_workout_comparison"] = {
                    "status": "failed", "reason_code": "comparison_failed", "error_type": type(exc).__name__,
                }
        race_data_payload = (payload.get("inputs") or {}).get("race_data")
        prediction_payload = (payload.get("outputs") or {}).get("prediction")
        payload["backtest_capabilities"] = _audit_capabilities(
            race_data_payload=race_data_payload,
            prediction_payload=prediction_payload,
            result_payload=result_payload,
        )
        payload["missing_inputs"] = _audit_missing_inputs(
            race_data_payload=race_data_payload,
            prediction_payload=prediction_payload,
            result_payload=result_payload,
        )
        _atomic_write_json(path, payload, compressed=True)
        updated_paths.append(path)
    return updated_paths[-1] if updated_paths else None


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


def _latest_version_path(directory, race_id, stage):
    stage_dir = Path(directory) / "_versions" / _safe_text(race_id) / normalize_prediction_stage(stage)
    if not stage_dir.exists():
        return None
    paths = sorted(stage_dir.glob("*.json"), key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    expected_race_id = _safe_text(race_id)
    expected_stage = normalize_prediction_stage(stage)
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload:
            continue
        race_info = payload.get("race_info")
        payload_race_id = _safe_text(payload.get("race_id"))
        if not payload_race_id and isinstance(race_info, dict):
            payload_race_id = _safe_text(race_info.get("race_id"))
        if payload_race_id and payload_race_id != expected_race_id:
            continue
        if normalize_prediction_stage(payload.get("prediction_stage")) != expected_stage:
            continue
        return path
    return None


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
    directories = _prediction_dir_candidates(base_dir)
    for stage in (PREDICTION_STAGE_FINAL, PREDICTION_STAGE_EARLY):
        for directory in directories:
            path = _latest_version_path(directory, race_id_text, stage)
            if path is not None:
                return path
    for directory in directories:
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


def _clamp01(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


_PREDICTION_INDEX_CACHE = {}
_PREDICTION_HISTORY_CACHE = {}
_PREDICTION_CACHE_LOCK = threading.RLock()


def clear_agent_prediction_caches():
    with _PREDICTION_CACHE_LOCK:
        _PREDICTION_INDEX_CACHE.clear()
        _PREDICTION_HISTORY_CACHE.clear()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prediction_cache_ttl_seconds():
    try:
        return max(0, int(os.environ.get("KEIBA_AGENT_PREDICTION_CACHE_TTL_SECONDS", "120") or "120"))
    except ValueError:
        return 120


def _prediction_cache_key(base_dir):
    resolved_base = str(Path(base_dir).resolve())
    dirs = tuple(str(path) for path in _prediction_dir_candidates(base_dir))
    return resolved_base, dirs


def _prediction_metadata(path, payload):
    race_info = dict(payload.get("race_info") or {})
    race_id = _safe_text(payload.get("race_id")) or _safe_text(race_info.get("race_id")) or Path(path).stem
    stage = normalize_prediction_stage(payload.get("prediction_stage"))
    return {
        "_source_path": str(path),
        "race_id": race_id,
        "prediction_stage": stage,
        "prediction_stage_label": prediction_stage_label(stage),
        "scope_key": _safe_text(payload.get("scope_key")),
        "source": _safe_text(payload.get("source")),
        "race_info": {
            "race_id": race_id,
            "race_date": _safe_text(race_info.get("race_date")),
            "scope_key": _safe_text(race_info.get("scope_key")),
            "source": _safe_text(race_info.get("source")),
            "surface": _safe_text(race_info.get("surface")),
            "course": _safe_text(race_info.get("course")),
        },
    }


def _build_prediction_index(base_dir):
    items = []
    by_date = {}
    selected = {}
    for directory_index, directory in enumerate(_prediction_dir_candidates(base_dir)):
        candidates = []
        candidates.extend((path, 1) for path in sorted(directory.glob("*.json")))
        candidates.extend((path, 2) for path in sorted((directory / "_versions").glob("*/early/*.json")))
        candidates.extend((path, 3) for path in sorted((directory / "_versions").glob("*/final/*.json")))
        for path, stage_priority in candidates:
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            item = _prediction_metadata(path, payload)
            race_id = _safe_text(item.get("race_id"))
            if not race_id:
                continue
            try:
                modified_ns = path.stat().st_mtime_ns
            except OSError:
                modified_ns = 0
            priority = (stage_priority, -directory_index, modified_ns, path.name)
            current = selected.get(race_id)
            if current is None or priority > current[0]:
                selected[race_id] = (priority, item)
    for race_id in sorted(selected):
        item = selected[race_id][1]
        items.append(item)
        race_info = dict(item.get("race_info") or {})
        race_date = _safe_text(race_info.get("race_date"))
        if race_date:
            by_date.setdefault(race_date, []).append(item)
    return {"items": items, "by_date": by_date}


def _prediction_index(base_dir):
    ttl_seconds = _prediction_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return _build_prediction_index(base_dir)

    key = _prediction_cache_key(base_dir)
    now = time.time()
    cached = _PREDICTION_INDEX_CACHE.get(key)
    if cached and now - float(cached.get("created_at", 0.0) or 0.0) <= ttl_seconds:
        return cached.get("index") or {"items": [], "by_date": {}}

    with _PREDICTION_CACHE_LOCK:
        cached = _PREDICTION_INDEX_CACHE.get(key)
        if cached and now - float(cached.get("created_at", 0.0) or 0.0) <= ttl_seconds:
            return cached.get("index") or {"items": [], "by_date": {}}
        index = _build_prediction_index(base_dir)
        _PREDICTION_INDEX_CACHE.clear()
        _PREDICTION_HISTORY_CACHE.clear()
        _PREDICTION_INDEX_CACHE[key] = {"created_at": time.time(), "index": index}
        return index


def _iter_prediction_payloads(base_dir):
    for item in list((_prediction_index(base_dir).get("items") or [])):
        path = str((item or {}).get("_source_path", "") or "").strip()
        if not path:
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        payload["_source_path"] = path
        yield payload


def _prediction_payloads_for_date(base_dir, target_date):
    target = _safe_text(target_date)
    if not target:
        return []
    payloads = []
    index = _prediction_index(base_dir)
    for item in list((index.get("by_date") or {}).get(target, []) or []):
        path = str((item or {}).get("_source_path", "") or "").strip()
        if not path:
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        payload["_source_path"] = path
        payloads.append(payload)
    return payloads


def available_dates(base_dir, scope_key=""):
    scope = _normalize_history_scope_key(scope_key)
    index = _prediction_index(base_dir)
    if not scope:
        return sorted((index.get("by_date") or {}).keys())

    dates = set()
    for item in list(index.get("items") or []):
        if not _scope_matches(item, scope_key):
            continue
        race_info = dict(item.get("race_info") or {})
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


def _normalize_history_scope_key(value):
    raw = _safe_text(value).lower()
    raw = raw.replace(" ", "_").replace("-", "_").replace("/", "_")
    if raw in ("central", "jra", "chuo", "中央"):
        return "central"
    if raw in ("central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"):
        return "central_turf"
    if raw in ("central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"):
        return "central_dirt"
    if raw in ("local", "l", "3", "地方"):
        return "local"
    return ""


def _scope_matches(payload, scope_key):
    scope = _normalize_history_scope_key(scope_key)
    if not scope:
        return True
    race_info = dict(payload.get("race_info") or {})
    payload_scope = _normalize_history_scope_key(payload.get("scope_key")) or _normalize_history_scope_key(
        race_info.get("scope_key")
    )
    payload_source = _safe_text(payload.get("source")) or _safe_text(race_info.get("source"))
    if payload_scope:
        if scope == "central":
            return payload_scope in ("central_turf", "central_dirt")
        if scope == "local":
            return payload_scope == "local"
        if scope in ("central_turf", "central_dirt"):
            return payload_scope == scope
    if payload_source == "local":
        return scope == "local"
    surface = _safe_text(race_info.get("surface"))
    course = _safe_text(race_info.get("course"))
    if scope == "central":
        if course:
            return course in COURSE_ORDER
        return "芝" in surface or "ダ" in surface or "砂" in surface
    if scope == "central_turf":
        return "芝" in surface
    if scope == "central_dirt":
        return "ダ" in surface or "砂" in surface
    if scope == "local":
        return course and course not in COURSE_ORDER
    return True


def _payload_scope_key(payload):
    race_info = dict(payload.get("race_info") or {})
    payload_scope = _normalize_history_scope_key(payload.get("scope_key")) or _normalize_history_scope_key(
        race_info.get("scope_key")
    )
    if payload_scope in ("local", "central_turf", "central_dirt"):
        return payload_scope

    payload_source = (_safe_text(payload.get("source")) or _safe_text(race_info.get("source"))).lower()
    if payload_source == "local":
        return "local"

    surface = _safe_text(race_info.get("surface"))
    if "芝" in surface:
        return "central_turf"
    if "ダ" in surface or "砂" in surface:
        return "central_dirt"

    if payload_scope == "central":
        return "central"

    course = _safe_text(race_info.get("course"))
    if course in COURSE_ORDER:
        return "central"
    if course:
        return "local"
    return ""


def _confidence_fallback_prior(confidence, scope_key):
    scoped_scores = CONFIDENCE_FALLBACK_SCORE_BY_SCOPE.get(scope_key)
    if scoped_scores and confidence in scoped_scores:
        return scoped_scores[confidence]
    return AGENT_CONFIDENCE_FALLBACK_SCORE.get(confidence, 0.5)


def _clamp_confidence_for_scope(value, scope_key):
    lower, upper = CONFIDENCE_FALLBACK_LIMIT_BY_SCOPE.get(scope_key, (0.0, 1.0))
    return max(lower, min(upper, value))


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
    memo_map = _top_horse_memo_map(payload)
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
        rows.append(_serialize_top_horse(item, rank=rank, symbol=symbol, memo_map=memo_map))
        seen.add(horse_no)

    if len(rows) < limit:
        for item in _ranked_horses(payload):
            horse_no = _safe_int(item.get("horse_no"), 0)
            if horse_no <= 0 or horse_no in seen:
                continue
            rows.append(_serialize_top_horse(item, rank=len(rows) + 1, symbol="", memo_map=memo_map))
            seen.add(horse_no)
            if len(rows) >= limit:
                break
    return rows[:limit]


def _top_horse_memo_map(payload):
    out = {}
    for item in list(payload.get("top_horse_memos") or []):
        if not isinstance(item, dict):
            continue
        horse_no = _safe_int(item.get("horse_no"), 0)
        memo = _safe_text(item.get("memo") or item.get("comment") or item.get("reason"))
        if horse_no > 0 and memo:
            out[horse_no] = memo
    return out


def _support_score(item):
    total_score = _safe_float(item.get("total_score"), 0.0)
    return max(1, min(99, int(round(total_score * 3.0))))


def _serialize_top_horse(item, rank=0, symbol="", memo_map=None):
    horse_no = _safe_int(item.get("horse_no"), 0)
    memos = dict(memo_map or {})
    return {
        "horse_no": str(horse_no or ""),
        "horse_name": _safe_text(item.get("horse_name")),
        "pred_rank": int(rank or 0),
        "mark": _safe_text(symbol),
        "total_score": round(_safe_float(item.get("total_score"), 0.0), 3),
        "support_score": _support_score(item),
        "top3_prob_model": 0.0,
        "rank_score_norm": round(max(0.05, 1.0 - max(0, int(rank or 1) - 1) * 0.18), 6),
        "win_odds": round(_safe_float(item.get("odds"), 0.0), 3),
        "popularity": _safe_int(item.get("popularity"), 0),
        "reason": memos.get(horse_no) or _safe_text(item.get("reason")),
    }


def _marks_text(payload):
    marks = dict(payload.get("marks") or {})
    parts = []
    for symbol in MARK_ORDER:
        horse_no = _safe_int(marks.get(symbol), 0)
        if horse_no > 0:
            parts.append(f"{symbol}{horse_no}")
    return " ".join(parts)


def _explicit_confidence_score(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    if 0.0 <= number <= 1.0:
        return number
    return None


def _score_shape_metrics(payload):
    scores = [
        _safe_float(item.get("total_score"), float("nan"))
        for item in _ranked_horses(payload)
        if isinstance(item, dict)
    ]
    scores = [score for score in scores if math.isfinite(score)]
    if len(scores) < 2:
        return None

    top = scores[0]
    second = scores[1]
    third = scores[2] if len(scores) >= 3 else second
    top5 = scores[:5]
    top5_range = max(top - top5[-1], 1.0)
    top_gap_score = _clamp01((top - second) / max(3.0, top5_range * 0.55))
    top3_gap_score = _clamp01((top - third) / max(5.0, top5_range * 0.85))

    positive_top5 = [max(0.0, score) for score in top5]
    positive_total = sum(positive_top5)
    if positive_total > 0:
        share = positive_top5[0] / positive_total
        uniform_share = 1.0 / float(len(positive_top5))
        concentration_score = _clamp01((share - uniform_share) / max(0.01, 1.0 - uniform_share))
    else:
        concentration_score = 0.0

    quality_score = _clamp01((top - 18.0) / 22.0)
    field_score = _clamp01(len(scores) / 12.0)
    shape_score = _clamp01(
        0.34 * top_gap_score
        + 0.28 * top3_gap_score
        + 0.20 * concentration_score
        + 0.12 * quality_score
        + 0.06 * field_score
    )
    return {
        "shape_score": shape_score,
        "top_gap_score": top_gap_score,
        "top3_gap_score": top3_gap_score,
        "concentration_score": concentration_score,
        "quality_score": quality_score,
        "field_score": field_score,
    }


def _confidence_shape_score_for_scope(metrics, scope_key):
    if scope_key in ("central_turf", "central_dirt", "central"):
        return _clamp01(
            0.40 * float(metrics.get("top_gap_score", 0.0) or 0.0)
            + 0.34 * float(metrics.get("top3_gap_score", 0.0) or 0.0)
            + 0.12 * float(metrics.get("concentration_score", 0.0) or 0.0)
            + 0.10 * float(metrics.get("quality_score", 0.0) or 0.0)
            + 0.04 * float(metrics.get("field_score", 0.0) or 0.0)
        )
    return float(metrics.get("shape_score", 0.0) or 0.0)


def strategy_confidence_score(payload):
    strategy = dict(payload.get("strategy") or {})
    explicit = _explicit_confidence_score(strategy.get("confidence_score"))
    if explicit is None:
        explicit = _explicit_confidence_score(payload.get("confidence_score"))
    if explicit is not None:
        return round(_clamp01(explicit), 6)

    scope_key = _payload_scope_key(payload)
    confidence = _safe_text(strategy.get("confidence")).lower()
    prior = _confidence_fallback_prior(confidence, scope_key)
    metrics = _score_shape_metrics(payload)
    if not metrics:
        label_fallback = AGENT_CONFIDENCE_FALLBACK_SCORE.get(confidence, prior)
        return round(_clamp01(_clamp_confidence_for_scope(label_fallback, scope_key)), 6)

    shape_weight = CONFIDENCE_SHAPE_WEIGHT_BY_SCOPE.get(scope_key, 0.55)
    shape_score = _confidence_shape_score_for_scope(metrics, scope_key)
    confidence_score = (1.0 - shape_weight) * prior + shape_weight * shape_score
    return round(_clamp01(_clamp_confidence_for_scope(confidence_score, scope_key)), 6)


def strategy_agreement_score(payload):
    metrics = _score_shape_metrics(payload)
    if metrics:
        score = (
            0.55 * float(metrics.get("top_gap_score", 0.0) or 0.0)
            + 0.30 * float(metrics.get("top3_gap_score", 0.0) or 0.0)
            + 0.15 * float(metrics.get("concentration_score", 0.0) or 0.0)
        )
        return round(_clamp01(score), 6)
    return strategy_confidence_score(payload)


def _strategy_confidence(payload):
    return strategy_confidence_score(payload)


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
    bet_decision = _safe_text(strategy.get("bet_decision"))
    confidence_score = strategy_confidence_score(payload)
    public_tone = public_decision_tone(bet_decision, confidence_score)
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
        "bet_decision": bet_decision,
        "confidence_score": confidence_score,
        "public_decision_tone": public_tone,
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
    high_evaluation_count = 0
    watch_evaluation_count = 0
    skip_evaluation_count = 0
    for item in rows:
        decision = _safe_text(item.get("bet_decision")).upper()
        if decision == "BET":
            bet_count += 1
        if decision == "SKIP":
            skip_count += 1
        public_tone = _safe_text(item.get("public_decision_tone"))
        if public_tone == "bet":
            high_evaluation_count += 1
        elif public_tone == "skip":
            skip_evaluation_count += 1
        else:
            watch_evaluation_count += 1
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
        "high_evaluation_races": high_evaluation_count,
        "watch_evaluation_races": watch_evaluation_count,
        "skip_evaluation_races": skip_evaluation_count,
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


def _history_cache_key(base_dir, target, scope_key):
    target_text = target.isoformat() if target is not None else ""
    return str(Path(base_dir).resolve()), target_text, _normalize_history_scope_key(scope_key)


def _build_history_payload_uncached(base_dir, target=None, scope_key=""):
    all_records = []
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
        "scope_key": _normalize_history_scope_key(scope_key),
        "periods": out_periods,
    }


def build_history_payload(base_dir, target_date="", scope_key=""):
    target = _parse_date(target_date)
    if target is None:
        dates = available_dates(base_dir)
        target = _parse_date(dates[-1] if dates else "")

    ttl_seconds = _prediction_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return _build_history_payload_uncached(base_dir, target=target, scope_key=scope_key)

    key = _history_cache_key(base_dir, target, scope_key)
    now = time.time()
    cached = _PREDICTION_HISTORY_CACHE.get(key)
    if cached and now - float(cached.get("created_at", 0.0) or 0.0) <= ttl_seconds:
        return cached.get("payload") or {}

    with _PREDICTION_CACHE_LOCK:
        cached = _PREDICTION_HISTORY_CACHE.get(key)
        if cached and now - float(cached.get("created_at", 0.0) or 0.0) <= ttl_seconds:
            return cached.get("payload") or {}
        payload = _build_history_payload_uncached(base_dir, target=target, scope_key=scope_key)
        _PREDICTION_HISTORY_CACHE.clear()
        _PREDICTION_HISTORY_CACHE[key] = {"created_at": time.time(), "payload": payload}
        return payload


def _agent_prediction_detail(payload):
    strategy = dict(payload.get("strategy") or {})
    pace = dict(payload.get("race_pace_projection") or {})
    simulation = dict(payload.get("race_simulation") or {})
    return {
        "source": "keiba_llm_agent",
        "prediction_stage": normalize_prediction_stage(payload.get("prediction_stage")),
        "prediction_stage_label": prediction_stage_label(payload.get("prediction_stage")),
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
        "label": "予測",
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
        "prediction_stage": normalize_prediction_stage(payload.get("prediction_stage")),
        "prediction_stage_label": prediction_stage_label(payload.get("prediction_stage")),
        "scope_key": "",
        "scope_label": "予測",
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
        "confidence_score": strategy_confidence_score(payload),
        "agreement_score": strategy_agreement_score(payload),
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
    for payload in _prediction_payloads_for_date(base_dir, target):
        race_info = dict(payload.get("race_info") or {})
        if not _scope_matches(payload, scope_key):
            continue
        row = _public_race_row(base_dir, payload, include_detail_fields=include_detail_fields)
        rows.append(row)
    rows.sort(key=lambda item: item.get("_agent_sort_key") or (999, 999, ""))
    for row in rows:
        row.pop("_agent_sort_key", None)
    return rows


def _custom_prediction_horse_item(horse_score, scoring_config):
    if not isinstance(horse_score, dict):
        return None
    horse_no = _safe_int(horse_score.get("horse_no"), 0)
    horse_name = _safe_text(horse_score.get("horse_name"))
    if horse_no <= 0 or not horse_name:
        return None

    score_breakdown = dict(horse_score.get("score_breakdown") or {})
    scores = dict(horse_score.get("scores") or {})
    pedigree_adjustment = dict(horse_score.get("pedigree_adjustment") or {})
    race_level_adjustment = dict(horse_score.get("race_level_adjustment") or {})
    pace_adjustment = dict(horse_score.get("pace_adjustment") or {})
    legacy_condition_fit = (
        _safe_float(scores.get("distance_fit"), 0.0)
        + _safe_float(scores.get("course_fit"), 0.0)
        + _safe_float(scores.get("track_condition_fit"), 0.0)
    ) / 3.0
    has_component_scores = (
        _safe_float(scores.get("ability_score"), 0.0) > 0
        or _safe_float(scores.get("recent_quality_score"), 0.0) > 0
        or _safe_float(scores.get("condition_fit_score"), 0.0) > 0
        or _safe_float(scores.get("pace_jockey_score"), 0.0) > 0
        or _safe_float(scores.get("trend_score"), 5.0) != 5.0
        or _safe_float(scores.get("race_level_score"), 5.0) != 5.0
    )

    return {
        "horse_no": horse_no,
        "horse_name": horse_name,
        "factors": {
            "base": _safe_float(
                score_breakdown.get("base_total_score"),
                _safe_float(horse_score.get("base_total_score"), 0.0),
            ),
            "pedigree": _safe_float(
                score_breakdown.get("pedigree_adjustment_raw"),
                _safe_float(pedigree_adjustment.get("pedigree_adjustment"), 0.0),
            ),
            "race_level_adjustment": _safe_float(
                score_breakdown.get("race_level_adjustment_raw"),
                _safe_float(race_level_adjustment.get("adjustment"), 0.0),
            ),
            "pace_adjustment": _safe_float(
                score_breakdown.get("pace_adjustment_raw"),
                _safe_float(pace_adjustment.get("adjustment"), 0.0),
            ),
            "ability": _safe_float(scores.get("ability_score"), 0.0)
            if has_component_scores
            else _safe_float(scores.get("recent_form"), 0.0),
            "recent_quality": _safe_float(scores.get("recent_quality_score"), 0.0)
            if has_component_scores
            else _safe_float(scores.get("recent_form"), 0.0),
            "trend": _safe_float(scores.get("trend_score"), 5.0) if has_component_scores else 5.0,
            "condition": _safe_float(scores.get("condition_fit_score"), 0.0)
            if has_component_scores
            else legacy_condition_fit,
            "race_level": _safe_float(scores.get("race_level_score"), 5.0),
            "pace_jockey": _safe_float(scores.get("pace_jockey_score"), 0.0)
            if has_component_scores
            else _safe_float(scores.get("jockey_fit"), 0.0),
            "risk": _safe_float(scores.get("risk"), 0.0),
        },
        "default_weights": {
            "pedigree": _safe_float(
                score_breakdown.get("pedigree_weight"),
                _safe_float(scoring_config.get("pedigree_weight"), 0.2),
            ),
            "race_level": _safe_float(
                score_breakdown.get("race_level_weight"),
                _safe_float(scoring_config.get("race_level_weight"), 1.0),
            ),
            "pace": _safe_float(
                score_breakdown.get("pace_weight"),
                _safe_float(scoring_config.get("pace_weight"), 0.2),
            ),
        },
    }


def _custom_prediction_race_item(payload):
    race_info = dict(payload.get("race_info") or {})
    race_id = _safe_text(payload.get("race_id")) or _safe_text(race_info.get("race_id"))
    if not race_id:
        return None
    course = _safe_text(race_info.get("course"))
    race_no_text = _race_no_text(race_id)
    race_title = f"{course}{race_no_text}".strip() if course else (race_no_text or race_id)
    scoring_config = dict(payload.get("scoring_config") or {})
    horses = []
    for horse_score in list(payload.get("horse_scores") or []):
        item = _custom_prediction_horse_item(horse_score, scoring_config)
        if item is not None:
            horses.append(item)
    horses.sort(key=lambda item: item.get("horse_no", 0))
    if not horses:
        return None
    return {
        "race_id": race_id,
        "race_title": race_title,
        "race_name": _safe_text(race_info.get("race_name")),
        "course": course,
        "surface": _safe_text(race_info.get("surface")),
        "distance": _safe_int(race_info.get("distance"), 0),
        "distance_label": _distance_label(race_info),
        "track_condition": _normalize_track_condition(race_info.get("track_condition")),
        "scope_key": _payload_scope_key(payload),
        "prediction_stage": normalize_prediction_stage(payload.get("prediction_stage")),
        "horses": horses,
        "_sort_key": (_course_sort_value(course), _race_no(race_id), race_id),
    }


def build_custom_prediction_races(base_dir, target_date="", scope_key=""):
    """Build the score-factor payload used by the Flutter local calculator.

    Final scores, marks, betting decisions, and explanatory text are intentionally
    excluded. The mobile client receives only the factors required to recalculate
    a ranking on-device.
    """
    target = _safe_text(target_date)
    if not target:
        return []
    rows = []
    for payload in _prediction_payloads_for_date(base_dir, target):
        if not _scope_matches(payload, scope_key):
            continue
        row = _custom_prediction_race_item(payload)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda item: item.get("_sort_key") or (999, 999, ""))
    for row in rows:
        row.pop("_sort_key", None)
    return rows

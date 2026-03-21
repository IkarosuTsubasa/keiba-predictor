import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
POLICY_CACHE_VERSION = "gemini_policy_v12"
POLICY_PROMPT_VERSION = "gemini_policy_prompt_v17"
_MODULE_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _MODULE_DIR.parent
DEFAULT_CACHE_DIR = _PIPELINE_DIR / "data" / "policy_cache_gemini"


def _model_validate(model_cls, payload):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _model_dump(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


def _model_json_schema(model_cls):
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def _normalize_horse_no_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        pass
    digits = re.findall(r"\d+", text)
    if len(digits) == 1:
        return str(int(digits[0]))
    return text


class MarkTop5(BaseModel):
    horse_no: str
    horse_name: str
    pred_rank: int
    top3_prob_model: float
    rank_score_norm: float


class PolicyPrediction(BaseModel):
    horse_no: str
    horse_name: str
    pred_rank: int
    top3_prob_model: float
    rank_score_norm: float
    win_odds: float = 0.0
    place_odds: float = 0.0


class PairOddsSnapshot(BaseModel):
    bet_type: Literal["wide", "quinella", "exacta", "trio", "trifecta"]
    pair: str
    odds: float


class PolicyCandidate(BaseModel):
    id: str
    bet_type: Literal["win", "place", "wide", "quinella", "exacta", "trio", "trifecta"]
    legs: List[str]
    odds_used: float
    p_hit: float
    ev: float
    score: float


class PolicyAIStats(BaseModel):
    gap: float
    confidence_score: float
    stability_score: float
    risk_score: float


class PolicyConstraints(BaseModel):
    bankroll_yen: int = 0
    race_budget_yen: int = 0
    max_tickets_per_race: int
    high_odds_threshold: float
    allowed_types: List[Literal["win", "place", "wide", "quinella", "exacta", "trio", "trifecta"]] = Field(default_factory=list)


class RacePolicyInput(BaseModel):
    race_id: str
    scope_key: str
    field_size: int
    race_context: Dict[str, Any] = Field(default_factory=dict)
    ai: PolicyAIStats
    multi_model_ai: Dict[str, Any] = Field(default_factory=dict)
    marks_top5: List[MarkTop5] = Field(default_factory=list)
    predictions: List[PolicyPrediction] = Field(default_factory=list)
    predictions_full: List[Dict[str, Any]] = Field(default_factory=list)
    pair_odds_top: List[PairOddsSnapshot] = Field(default_factory=list)
    odds_full: Dict[str, Any] = Field(default_factory=dict)
    prediction_field_guide: Dict[str, str] = Field(default_factory=dict)
    multi_predictor: Dict[str, Any] = Field(default_factory=dict)
    horse_facts: List[Dict[str, Any]] = Field(default_factory=list)
    portfolio_history: Dict[str, Any] = Field(default_factory=dict)
    candidates: List[PolicyCandidate] = Field(default_factory=list)
    candidates_meta: Dict[str, Any] = Field(default_factory=dict)
    constraints: PolicyConstraints


class FocusPoint(BaseModel):
    type: Literal["horse", "pair", "bet_type", "concept"]
    value: str


class PolicyMark(BaseModel):
    symbol: Literal["◎", "○", "▲", "△", "☆"]
    horse_no: str


class PolicyTicketPlan(BaseModel):
    bet_type: Literal["win", "place", "wide", "quinella", "exacta", "trio", "trifecta"]
    legs: List[str]
    stake_yen: int


class RacePolicyOutput(BaseModel):
    bet_decision: Literal["bet", "no_bet"]
    participation_level: Literal["no_bet", "small_bet", "normal_bet"]
    enabled_bet_types: List[Literal["win", "place", "wide", "quinella", "exacta", "trio", "trifecta"]] = Field(default_factory=list)
    construction_style: Optional[Literal["single_axis", "pair_spread", "value_hunt", "conservative_single"]] = None
    key_horses: List[str] = Field(default_factory=list)
    secondary_horses: List[str] = Field(default_factory=list)
    longshot_horses: List[str] = Field(default_factory=list)
    max_ticket_count: int
    risk_tilt: Literal["low", "medium", "high"]
    reason_codes: List[str]
    warnings: Optional[List[str]] = None
    marks: List[PolicyMark] = Field(default_factory=list)
    pick_ids: List[str] = Field(default_factory=list)
    ticket_plan: List[PolicyTicketPlan] = Field(default_factory=list)
    focus_points: List[FocusPoint] = Field(default_factory=list)


class _TokenBucket:
    def __init__(self, rpm: int = 10):
        safe_rpm = max(1, int(rpm or 1))
        self.capacity = float(safe_rpm)
        self.refill_per_sec = float(safe_rpm) / 60.0
        self.tokens = float(safe_rpm)
        self.updated_at = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, count: float = 1.0) -> bool:
        need = max(0.0, float(count))
        with self._lock:
            now = time.monotonic()
            elapsed = max(0.0, now - self.updated_at)
            self.updated_at = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            if self.tokens + 1e-9 < need:
                return False
            self.tokens -= need
            return True


_TOKEN_BUCKET = _TokenBucket(rpm=int(os.environ.get("GEMINI_POLICY_RPM", "10") or "10"))
_LAST_CALL_META = {
    "cache_hit": False,
    "llm_latency_ms": 0,
    "fallback_reason": "",
    "picked_count": 0,
    "requested_budget_yen": 0,
    "requested_race_budget_yen": 0,
    "reused": False,
    "source_budget_yen": 0,
    "policy_version": POLICY_CACHE_VERSION,
}


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _parse_json_payload(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty", text, 0)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no json object found", text, 0)


def _candidate_digest(candidates: List[PolicyCandidate]) -> str:
    slim = []
    for item in candidates:
        slim.append(
            {
                "id": str(item.id),
                "bet_type": str(item.bet_type),
                "legs": [str(x) for x in item.legs],
                "odds_used": round(float(item.odds_used), 6),
                "p_hit": round(float(item.p_hit), 6),
                "ev": round(float(item.ev), 6),
                "score": round(float(item.score), 6),
            }
        )
    slim = sorted(slim, key=lambda x: str(x.get("id", "")))
    return hashlib.sha256(_stable_json_dumps(slim).encode("utf-8")).hexdigest()


def _input_context_digest(input_obj: RacePolicyInput) -> str:
    payload = {
        "field_size": int(input_obj.field_size or 0),
        "race_context": dict(input_obj.race_context or {}),
        "ai": _model_dump(input_obj.ai),
        "multi_model_ai": dict(input_obj.multi_model_ai or {}),
        "marks_top5": [_model_dump(x) for x in list(input_obj.marks_top5 or [])],
        "predictions": [_model_dump(x) for x in list(input_obj.predictions or [])],
        "predictions_full": list(input_obj.predictions_full or []),
        "pair_odds_top": [_model_dump(x) for x in list(input_obj.pair_odds_top or [])],
        "odds_full": dict(input_obj.odds_full or {}),
        "prediction_field_guide": dict(input_obj.prediction_field_guide or {}),
        "multi_predictor": dict(input_obj.multi_predictor or {}),
        "horse_facts": list(input_obj.horse_facts or []),
        "portfolio_history": dict(input_obj.portfolio_history or {}),
        "candidates_meta": dict(input_obj.candidates_meta or {}),
        "constraints": {
            "max_tickets_per_race": int(input_obj.constraints.max_tickets_per_race or 0),
            "high_odds_threshold": float(input_obj.constraints.high_odds_threshold or 0.0),
            "allowed_types": list(input_obj.constraints.allowed_types or []),
        },
        "candidates_digest": _candidate_digest(input_obj.candidates),
    }
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _cache_key(input_obj: RacePolicyInput, model: str) -> str:
    payload = {
        "v": POLICY_CACHE_VERSION,
        "prompt_version": POLICY_PROMPT_VERSION,
        "race_id": str(input_obj.race_id),
        "scope_key": str(input_obj.scope_key),
        "model": str(model or ""),
        "context_digest": _input_context_digest(input_obj),
    }
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def get_policy_cache_key(input: RacePolicyInput, model: str = DEFAULT_GEMINI_MODEL) -> str:
    input_obj = _model_validate(RacePolicyInput, input)
    return _cache_key(input_obj, model)


def _ensure_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_cache(path: Path) -> Optional[RacePolicyOutput]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        output_payload = raw.get("output", raw)
        return _model_validate(RacePolicyOutput, output_payload)
    except Exception:
        return None


def _write_cache(path: Path, output: RacePolicyOutput, meta: Dict[str, Any]) -> None:
    try:
        payload = {
            "output": _model_dump(output),
            "meta": {
                "fallback_reason": str(meta.get("fallback_reason", "") or ""),
                "llm_latency_ms": int(meta.get("llm_latency_ms", 0) or 0),
                "requested_budget_yen": int(meta.get("requested_budget_yen", 0) or 0),
                "requested_race_budget_yen": int(meta.get("requested_race_budget_yen", 0) or 0),
                "reused": bool(meta.get("reused", False)),
                "source_budget_yen": int(meta.get("source_budget_yen", 0) or 0),
                "cached_at": int(time.time()),
                "version": POLICY_CACHE_VERSION,
                "policy_version": str(meta.get("policy_version", POLICY_CACHE_VERSION) or POLICY_CACHE_VERSION),
            },
        }
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(_stable_json_dumps(payload), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        return

def _horse_pool(input_obj: RacePolicyInput) -> List[str]:
    seen = set()
    horses = []
    for row in list(input_obj.predictions or []) + list(input_obj.marks_top5 or []):
        horse_no = _normalize_horse_no_text(getattr(row, "horse_no", ""))
        if horse_no and horse_no not in seen:
            seen.add(horse_no)
            horses.append(horse_no)
    for cand in input_obj.candidates:
        for leg in list(cand.legs or []):
            text = _normalize_horse_no_text(leg)
            if text and text not in seen:
                seen.add(text)
                horses.append(text)
    return horses


def _reason_codes_for(
    ai: PolicyAIStats,
    field_size: int,
    has_value: bool,
    participation_level: str,
    buy_style: str,
    bet_decision: str,
    has_longshot: bool,
) -> List[str]:
    out: List[str] = []
    if field_size >= 14:
        out.append("MIXED_FIELD")
    else:
        out.append("NORMAL_FIELD")
    if float(ai.gap) >= 0.06 and float(ai.confidence_score) >= 0.62:
        out.append("STRONG_FAVORITE")
    if float(ai.confidence_score) < 0.5:
        out.append("LOW_CONFIDENCE")
    if float(ai.stability_score) < 0.45:
        out.append("LOW_STABILITY")
    if has_value:
        out.append("VALUE_PRESENT")
    else:
        out.append("NO_VALUE")
    if has_longshot:
        out.append("HIGH_ODDS_ONE_SHOT")
    if participation_level == "small_bet":
        out.append("SMALL_BET")
    if buy_style == "place_focus":
        out.append("PLACE_FOCUS")
    if buy_style in ("place_only", "conservative"):
        out.append("CONSERVATIVE")
    if buy_style == "pair_focus":
        out.append("PAIR_FOCUS")
    if buy_style == "win_focus":
        out.append("WIN_TILT")
    if bet_decision == "no_bet":
        out.append("NO_BET")
    return out


def _derive_construction_style(strategy_mode: str, buy_style: str, participation_level: str) -> str:
    mode = str(strategy_mode or "").strip().lower()
    style = str(buy_style or "").strip().lower()
    level = str(participation_level or "").strip().lower()
    if mode in ("pair_focus", "spread"):
        return "pair_spread"
    if mode in ("small_probe", "conservative_single", "place_only"):
        return "conservative_single"
    if style == "pair_focus":
        return "pair_spread"
    if style in ("place_only", "conservative") or level == "small_bet":
        return "conservative_single"
    return "single_axis"


def _infer_internal_policy_style(
    *,
    bet_decision: str,
    participation_level: str,
    enabled_bet_types: Optional[List[str]] = None,
    construction_style: str = "",
) -> Dict[str, str]:
    decision = str(bet_decision or "").strip().lower() or "no_bet"
    level = str(participation_level or "").strip().lower() or "no_bet"
    if decision == "no_bet" or level == "no_bet":
        return {"buy_style": "no_bet", "strategy_mode": "no_bet"}

    enabled = []
    seen = set()
    for item in list(enabled_bet_types or []):
        text = str(item or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        enabled.append(text)

    multi_leg_types = {"wide", "quinella", "exacta", "trio", "trifecta"}
    top_type = enabled[0] if enabled else ""
    only_place = enabled == ["place"]
    only_win = enabled == ["win"]
    combo_only = bool(enabled) and all(item in multi_leg_types for item in enabled)

    if only_place and level == "small_bet":
        buy_style = "place_only"
        strategy_mode = "place_only"
    elif only_win:
        buy_style = "win_focus"
        strategy_mode = "win_focus"
    elif combo_only or top_type in multi_leg_types:
        buy_style = "pair_focus"
        strategy_mode = "pair_focus" if level == "normal_bet" else "small_probe"
    elif level == "small_bet":
        buy_style = "conservative"
        strategy_mode = "small_probe"
    elif "place" in enabled and len(enabled) <= 2 and "win" not in enabled and not combo_only:
        buy_style = "place_focus"
        strategy_mode = "place_focus"
    else:
        buy_style = "balanced"
        strategy_mode = "balanced"

    construction = str(construction_style or "").strip().lower()
    if construction == "pair_spread":
        buy_style = "pair_focus"
        strategy_mode = "pair_focus" if level == "normal_bet" else "small_probe"
    elif construction == "conservative_single":
        if only_place:
            buy_style = "place_only" if level == "small_bet" else "place_focus"
            strategy_mode = "place_only" if level == "small_bet" else "place_focus"
        else:
            buy_style = "conservative"
            strategy_mode = "small_probe" if level != "normal_bet" else "balanced"

    return {"buy_style": buy_style, "strategy_mode": strategy_mode}


def fallback_no_bet_policy(input_obj: RacePolicyInput, fallback_reason: str = "") -> RacePolicyOutput:
    ai = input_obj.ai
    warnings: List[str] = []
    if fallback_reason:
        warnings.append(f"FALLBACK_{str(fallback_reason).upper()}")
    return _model_validate(
        RacePolicyOutput,
        {
            "bet_decision": "no_bet",
            "participation_level": "no_bet",
            "enabled_bet_types": [],
            "construction_style": "conservative_single",
            "key_horses": [],
            "secondary_horses": [],
            "longshot_horses": [],
            "marks": [],
            "focus_points": [{"type": "concept", "value": "fallback_no_bet"}],
            "max_ticket_count": 0,
            "risk_tilt": "low",
            "reason_codes": _reason_codes_for(
                ai,
                int(input_obj.field_size or 0),
                False,
                "no_bet",
                "no_bet",
                "no_bet",
                False,
            ),
            "pick_ids": [],
            "ticket_plan": [],
            "warnings": warnings or None,
        },
    )


def deterministic_policy(input_obj: RacePolicyInput, fallback_reason: str = "") -> RacePolicyOutput:
    ai = input_obj.ai
    constraints = input_obj.constraints
    allowed_types = {str(x).strip().lower() for x in list(constraints.allowed_types or []) if str(x).strip()}
    candidates = [
        c for c in input_obj.candidates if (not allowed_types) or (str(c.bet_type) in allowed_types)
    ]
    ranked_candidates = sorted(
        candidates,
        key=lambda c: (
            -float(c.score or 0.0),
            -float(c.ev or 0.0),
            -float(c.p_hit or 0.0),
            str(c.id or ""),
        ),
    )
    horses = _horse_pool(input_obj)
    predictions = list(input_obj.predictions or [])
    high_odds_threshold = float(constraints.high_odds_threshold or 10.0)
    has_value = any(float(c.ev) > 0.0 for c in candidates)
    has_combo_value = any(str(c.bet_type) in ("wide", "quinella", "exacta", "trio", "trifecta") and float(c.ev) > 0.0 for c in candidates)
    longshot_candidates = [c for c in candidates if float(c.odds_used) >= high_odds_threshold and float(c.ev) > 0.0]
    top_key = str(predictions[0].horse_no) if predictions else (horses[0] if horses else "")
    second_key = str(predictions[1].horse_no) if len(predictions) >= 2 else ""
    third_key = str(predictions[2].horse_no) if len(predictions) >= 3 else ""

    if not candidates:
        warnings = ["NO_POSITIVE_EV"] if not has_value else []
        if float(ai.stability_score) < 0.45:
            warnings.append("HIGH_UNCERTAINTY")
        if fallback_reason:
            warnings.append(f"FALLBACK_{str(fallback_reason).upper()}")
        return _model_validate(
            RacePolicyOutput,
            {
                "bet_decision": "no_bet",
                "participation_level": "no_bet",
                "enabled_bet_types": [],
                "construction_style": "conservative_single",
                "key_horses": [],
                "secondary_horses": [],
                "longshot_horses": [],
                "max_ticket_count": 0,
                "risk_tilt": "low",
                "reason_codes": _reason_codes_for(
                    ai, int(input_obj.field_size or 0), has_value, "no_bet", "no_bet", "no_bet", False
                ),
                "warnings": warnings or None,
                "pick_ids": [],
                "focus_points": [{"type": "concept", "value": "見送り"}],
            },
        )

    no_bet_case = (not has_value) and float(ai.confidence_score) < 0.34 and float(ai.stability_score) < 0.32
    if no_bet_case:
        warnings = ["NO_POSITIVE_EV"]
        if fallback_reason:
            warnings.append(f"FALLBACK_{str(fallback_reason).upper()}")
        return _model_validate(
            RacePolicyOutput,
            {
                "bet_decision": "no_bet",
                "participation_level": "no_bet",
                "enabled_bet_types": [],
                "construction_style": "conservative_single",
                "key_horses": [],
                "secondary_horses": [],
                "longshot_horses": [],
                "max_ticket_count": 0,
                "risk_tilt": "low",
                "reason_codes": _reason_codes_for(
                    ai, int(input_obj.field_size or 0), has_value, "no_bet", "no_bet", "no_bet", False
                ),
                "warnings": warnings,
                "pick_ids": [],
                "focus_points": [{"type": "concept", "value": "軽く入る形も作りにくい"}],
            },
        )

    prioritized = [c for c in ranked_candidates if float(c.ev or 0.0) > 0.0]
    if not prioritized:
        prioritized = ranked_candidates[:]
    enabled = []
    for cand in prioritized:
        bet_type = str(cand.bet_type or "").strip().lower()
        if (not bet_type) or (bet_type in enabled):
            continue
        enabled.append(bet_type)
        if len(enabled) >= 3:
            break
    if not enabled and ranked_candidates:
        enabled = [str(ranked_candidates[0].bet_type or "").strip().lower()]

    top_type = enabled[0] if enabled else ""
    multi_leg_types = {"wide", "quinella", "exacta", "trio", "trifecta"}
    top_is_multi_leg = top_type in multi_leg_types
    only_place = enabled == ["place"]
    only_win = enabled == ["win"]
    combo_only = bool(enabled) and all(item in multi_leg_types for item in enabled)

    strong_edge = bool(prioritized and float(prioritized[0].ev or 0.0) > 0.12)
    weak_conf = float(ai.gap) < 0.03 or float(ai.confidence_score) < 0.56 or float(ai.stability_score) < 0.48
    participation_level = "small_bet" if weak_conf and not strong_edge else "normal_bet"
    max_ticket_count = min(
        max(1, int(constraints.max_tickets_per_race or 1)),
        1 if only_place and participation_level == "small_bet" else (2 if participation_level == "small_bet" else max(2, min(4, len(prioritized) or len(enabled) or 1))),
    )

    if only_place and participation_level == "small_bet":
        buy_style = "place_only"
        strategy_mode = "place_only"
    elif only_win:
        buy_style = "win_focus"
        strategy_mode = "win_focus"
    elif combo_only or top_is_multi_leg:
        buy_style = "pair_focus"
        strategy_mode = "pair_focus" if participation_level == "normal_bet" else "small_probe"
    elif participation_level == "small_bet":
        buy_style = "conservative"
        strategy_mode = "small_probe"
    elif "place" in enabled and len(enabled) <= 2 and "win" not in enabled and not combo_only:
        buy_style = "place_focus"
        strategy_mode = "place_focus"
    else:
        buy_style = "balanced"
        strategy_mode = "balanced"

    enabled = [x for x in enabled if ((not allowed_types) or (x in allowed_types))]
    if not enabled:
        enabled = [str(candidates[0].bet_type)] if candidates else []
    risk_tilt = "low"
    if participation_level == "normal_bet":
        risk_tilt = "medium"
    if top_type in {"trio", "trifecta"} and participation_level != "no_bet":
        risk_tilt = "high" if strong_edge else "medium"

    key_horses = [top_key] if top_key else []
    secondary_horses = [x for x in [second_key, third_key] if x and x != top_key]
    longshot_horses: List[str] = []
    if longshot_candidates and participation_level != "no_bet":
        strategy_mode = "small_probe" if participation_level == "small_bet" else strategy_mode
        longshot_horses = [str(longshot_candidates[0].legs[0])] if list(longshot_candidates[0].legs or []) else []
        if risk_tilt == "low":
            risk_tilt = "medium"

    construction_style = _derive_construction_style(strategy_mode, buy_style, participation_level)
    warnings: List[str] = []
    if float(ai.stability_score) < 0.45:
        warnings.append("HIGH_UNCERTAINTY")
    if fallback_reason:
        warnings.append(f"FALLBACK_{str(fallback_reason).upper()}")
    focus_points = []
    if key_horses:
        focus_points.append({"type": "horse", "value": key_horses[0]})
    if enabled:
        focus_points.append({"type": "bet_type", "value": enabled[0]})
    if longshot_horses:
        focus_points.append({"type": "concept", "value": "高オッズは補助1点まで"})

    return _model_validate(
        RacePolicyOutput,
        {
            "bet_decision": "bet",
            "participation_level": participation_level,
            "enabled_bet_types": enabled,
            "construction_style": construction_style,
            "key_horses": key_horses,
            "secondary_horses": secondary_horses,
            "longshot_horses": longshot_horses,
            "max_ticket_count": min(
                max(1, int(max_ticket_count)),
                max(1, int(constraints.max_tickets_per_race or max_ticket_count)),
            ),
            "risk_tilt": risk_tilt,
            "reason_codes": _reason_codes_for(
                ai,
                int(input_obj.field_size or 0),
                has_value,
                participation_level,
                buy_style,
                "bet",
                bool(longshot_horses),
            ),
            "warnings": warnings or None,
            "pick_ids": [],
            "focus_points": focus_points,
        },
    )

def _make_prompt(input_obj: RacePolicyInput) -> str:
    schema = _model_json_schema(RacePolicyOutput)
    payload = _model_dump(input_obj)
    payload.pop("candidates", None)
    input_json = _stable_json_dumps(payload)
    schema_json = _stable_json_dumps(schema)
    constraints = input_obj.constraints
    return (
        "あなたは馬券購入AIコンペティションの参加者です。\n"
        "複数のAIが同条件で競い合い、週末終了時の資金残高で順位が決まります。\n"
        "あなたの出力がそのまま購入指示になります。ローカル側は検証・記録のみで、買い目や配分には一切介入しません。\n\n"

        "== コンペティション条件 ==\n"
        f"- 現在の残り本金: {int(constraints.bankroll_yen)}円（残高を増やすことが目的）\n"
        f"- このレースの上限: {int(constraints.race_budget_yen)}円\n"
        f"- 最大購入点数: {int(constraints.max_tickets_per_race)}\n"
        "- 購入単位: 100円刻み\n"
        "- 実運用では同じ時間帯や近い時間帯に 4-5 レース同時に買う必要が生じることがある\n"
        "- 1レースで資金を使いすぎると、期待値がある後続レースに参加できず機会損失になる\n"
        "- 単レース最適ではなく、当日全体のレース配分を意識して資金を残すこと\n"
        "- 資金が尽きたら残りのレースに参加不可。一方、全て少額で薄く買うだけでは他AIに勝てない。\n\n"

        "== 提供データの概要 ==\n"
        "- race_context: レースの場所・馬場・距離・日付などの基礎条件\n"
        "- predictions / predictions_full: 互換用の主表示予測。利用可能なら v5 を優先した単一モデル表示であり、これだけを特別扱いしてはいけない\n"
        "- multi_model_ai: モデル間分歧の集約指標。consensus_gap / top1_vote_margin / disagreement_score などを含む\n"
        "- prediction_field_guide: predictions_full の各列の説明\n"
        "- multi_predictor: 5つの予測モデル（v1-v5）の平等な入力本体。個別結果・全馬順位・共識表・モデル別命中率履歴を含む\n"
        "  - predictor_rankings: 各モデルの全馬順位。まずここを見て、モデル間の一致と不一致を把握すること\n"
        "  - profiles: 各モデルの設計思想（v1=総合バランス, v2=能力比較, v3=市場融合, v4=文脈適性, v5=スタッキング統合）\n"
        "  - consensus: 馬番ごとの top1_votes / top3_votes / avg_pred_rank / rank_std / top3_prob_range\n"
        "  - performance.current_scope_history: 現在条件（芝/ダート/地方）での各モデルの実績命中率\n"
        "- horse_facts: 各馬の共通ファクト。TI、経験、騎手、オッズ、休み明け日数などの軽量サマリ\n"
        "- odds_full: 全券種の全量オッズ（win/place/wide/quinella/exacta/trio/trifecta）\n"
        "- portfolio_history: あなた自身の直近購入履歴・損益推移・券種別成績\n"
        "  - today: 本日の開始本金・確定損益・未決済拘束額・利用可能残高\n"
        "  - recent_days / lookback_summary / bet_type_breakdown / recent_tickets\n"
        "- ai: レースの予測信頼度（gap, confidence_score, stability_score, risk_score）\n\n"
        "- candidates / candidates_meta: 実際に使ってよい候補買い目の shortlist。p_hit / ev / score は下見用の近似値であり、盲信せず他情報と突き合わせて使う\n\n"

        "== あなたの仕事 ==\n"
        "上記データを全て分析し、このレースで「何を」「いくら」買うか（または買わないか）を自分で判断してください。\n"
        "券種の選択、馬の選定、金額の配分、参加/見送りの判断、全てあなたに委ねます。\n\n"

        "== 分析の進め方（推奨、強制ではない） ==\n"
        "1. multi_predictor.predictor_rankings で v1-v5 全モデルの全馬順位を平等に確認する\n"
        "2. multi_predictor の consensus と performance.current_scope_history、multi_model_ai を見て、モデル間の一致・不一致と実績差を確認する\n"
        "3. horse_facts と race_context を使って、順位の背景を確認する\n"
        "4. predictions_full は互換用の主表示に過ぎないため、補助情報として扱う\n"
        "5. odds_full と candidates を突き合わせ、期待値の高い馬・組み合わせを探す\n"
        "6. portfolio_history で自分の最近の調子・癖を確認し、資金配分に反映する\n"
        "7. 以上を踏まえ、券種・買い目・金額を自由に決定する\n\n"

        "== 勝つための視点 ==\n"
        "- 「予測で優位性があり、かつオッズが過小評価」な組み合わせが本当の value\n"
        "- 特定の1モデルをデフォルトで優先しないこと。v1-v5 は平等な判断材料である\n"
        "- 予測モデル間で意見が割れている場合、各モデルの命中率実績を参考に判断する\n"
        "- consensus の投票数は参考情報の一つに過ぎない。投票数が多い＝買うべき、ではない\n"
        "- 買い方は完全に自由。特定の馬を中心に組む必要はなく、データから自分なりの根拠で組み立てること\n"
        "- 命中率の高い堅実な馬券と、たまに当たる高配当のバランスが週間収支の鍵\n"
        "- 今この1レースだけでなく、このあと 4-5 レース続けて買う可能性を前提に資金配分する\n"
        "- 明確な優位がないのに単レースへ過大投入するのは、後続の高期待値レースを逃すので避ける\n"
        "- 「このレースはデータ的に自信が持てる」時に厚く張り、曖昧な時は薄く張るか見送る\n"
        "- 連敗中の取り返し買い、連勝中の過信買いは典型的な失敗パターン\n"
        "- 見送り（no_bet）も立派な戦略。全レース参加する義務はない\n\n"

        "== 出力の制約（厳守） ==\n"
        "- 入力データに存在する馬番・組み合わせ・券種のみ使用可能（創作禁止）\n"
        "- ticket_plan の合計金額は race_budget_yen 以下\n"
        "- stake_yen は 100円単位\n"
        "- bet_decision が bet なら ticket_plan を必ず記入\n"
        "- JSON のみ出力\n\n"

        "== 出力フィールド説明 ==\n"
        "- bet_decision: bet（購入）/ no_bet（見送り）\n"
        "- participation_level: no_bet / small_bet / normal_bet\n"
        "- enabled_bet_types: 今回使う券種リスト（odds_full に存在するもののみ）\n"
        "- key_horses: 注目馬 / secondary_horses: 次点馬 / longshot_horses: 穴馬（全て任意、無理に埋めなくてよい）\n"
        "- marks: ◎○▲△☆（必要な分だけ）\n"
        "- focus_points: type=horse/pair/bet_type/concept, value=内容\n"
        "- max_ticket_count: 購入点数\n"
        "- risk_tilt: low / medium / high\n"
        "- reason_codes: MIXED_FIELD / NORMAL_FIELD / STRONG_FAVORITE / LOW_CONFIDENCE / LOW_STABILITY / VALUE_PRESENT / NO_VALUE / HIGH_ODDS_ONE_SHOT / PLACE_FOCUS / PAIR_FOCUS / WIN_TILT / CONSERVATIVE / SMALL_BET / NO_BET から該当するものを選択\n"
        "- ticket_plan: [{\"bet_type\": \"券種\", \"legs\": [\"馬番\",...], \"stake_yen\": 金額}] ← これが実際の購入指示\n"
        "- pick_ids: 補助情報（空でも可）\n"
        "- warnings: 注意事項があれば\n\n"

        "--------------------------------\n"
        "【入力JSON】\n"
        "--------------------------------\n"
        "<INPUT_JSON>\n"
        f"{input_json}\n"
        "</INPUT_JSON>\n\n"

        "strict JSON only。response_json_schema:\n"
        f"{schema_json}\n"
    )


def _build_request_meta(input_obj: RacePolicyInput) -> Dict[str, Any]:
    requested_budget_yen = int(input_obj.constraints.bankroll_yen or 0)
    requested_race_budget_yen = int(input_obj.constraints.race_budget_yen or 0)
    return {
        "requested_budget_yen": requested_budget_yen,
        "requested_race_budget_yen": requested_race_budget_yen,
        "reused": False,
        "source_budget_yen": requested_budget_yen,
        "policy_version": POLICY_CACHE_VERSION,
    }


def _extract_response_text(response: Any) -> str:
    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = str(getattr(part, "text", "") or "").strip()
            if part_text:
                return part_text
    return ""


def _call_gemini_once(prompt: str, model: str, api_key: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    response_schema = _model_json_schema(RacePolicyOutput)
    config = {
        "response_mime_type": "application/json",
        "response_json_schema": response_schema,
        "temperature": 0.2,
    }
    response = client.models.generate_content(
        model=str(model or DEFAULT_GEMINI_MODEL),
        contents=prompt,
        config=config,
    )
    return _extract_response_text(response)

def _sanitize_horse_list(values: List[str], allowed_horses: List[str]) -> List[str]:
    allowed = {_normalize_horse_no_text(x) for x in allowed_horses if _normalize_horse_no_text(x)}
    out = []
    seen = set()
    for value in list(values or []):
        text = _normalize_horse_no_text(value)
        if (not text) or (text in seen) or (text not in allowed):
            continue
        seen.add(text)
        out.append(text)
    return out


def _sanitize_marks(marks: List[PolicyMark], allowed_horses: List[str]) -> List[Dict[str, str]]:
    allowed = {_normalize_horse_no_text(x) for x in allowed_horses if _normalize_horse_no_text(x)}
    out = []
    seen_symbols = set()
    seen_horses = set()
    for mark in list(marks or []):
        symbol = str(getattr(mark, "symbol", "") or "").strip()
        horse_no = _normalize_horse_no_text(getattr(mark, "horse_no", ""))
        if (not symbol) or (not horse_no):
            continue
        if symbol in seen_symbols or horse_no in seen_horses:
            continue
        if horse_no not in allowed:
            continue
        seen_symbols.add(symbol)
        seen_horses.add(horse_no)
        out.append({"symbol": symbol, "horse_no": horse_no})
    return out


def _sanitize_ticket_plan(
    ticket_plan: List[PolicyTicketPlan],
    allowed_types: set,
    allowed_horses: List[str],
    max_budget: int,
) -> List[Dict[str, Any]]:
    budget_cap = max(0, int(max_budget or 0))
    allowed_set = {_normalize_horse_no_text(h) for h in allowed_horses if _normalize_horse_no_text(h)}
    out: List[Dict[str, Any]] = []
    seen_keys: set = set()
    used = 0
    for item in list(ticket_plan or []):
        bet_type = str(getattr(item, "bet_type", "") or "").strip().lower()
        raw_legs = list(getattr(item, "legs", []) or [])
        legs = [_normalize_horse_no_text(x) for x in raw_legs if _normalize_horse_no_text(x)]
        stake_yen = int(getattr(item, "stake_yen", 0) or 0)
        if not bet_type or not legs or stake_yen <= 0:
            continue
        if allowed_types and bet_type not in allowed_types:
            continue
        if not all(h in allowed_set for h in legs):
            continue
        expected_leg_count = {"win": 1, "place": 1, "wide": 2, "quinella": 2, "exacta": 2, "trio": 3, "trifecta": 3}
        if len(legs) != expected_leg_count.get(bet_type, 0):
            continue
        # For unordered bet types, sort legs to canonicalize the key
        canon_legs = sorted(legs, key=lambda x: int(x) if x.isdigit() else x) if bet_type in ("wide", "quinella", "trio") else legs
        ticket_key = f"{bet_type}:{'-'.join(canon_legs)}"
        if ticket_key in seen_keys:
            continue
        stake_yen = int(stake_yen // 100) * 100
        if stake_yen <= 0:
            continue
        if budget_cap > 0 and used + stake_yen > budget_cap:
            continue
        seen_keys.add(ticket_key)
        used += stake_yen
        out.append({"bet_type": bet_type, "legs": legs, "stake_yen": stake_yen})
    return out


def _sanitize_output(output: RacePolicyOutput, input_obj: RacePolicyInput) -> RacePolicyOutput:
    allowed_types = {str(x).strip().lower() for x in list(input_obj.constraints.allowed_types or []) if str(x).strip()}
    allowed_horses = _horse_pool(input_obj)

    pick_ids = [str(x).strip() for x in list(output.pick_ids or []) if str(x).strip()]

    enabled_bet_types = []
    seen_types = set()
    for bet_type in list(output.enabled_bet_types or []):
        text = str(bet_type or "").strip().lower()
        if (not text) or (text in seen_types):
            continue
        if allowed_types and text not in allowed_types:
            continue
        seen_types.add(text)
        enabled_bet_types.append(text)

    key_horses = _sanitize_horse_list(list(output.key_horses or []), allowed_horses)
    secondary_horses = [
        x for x in _sanitize_horse_list(list(output.secondary_horses or []), allowed_horses) if x not in set(key_horses)
    ]
    longshot_horses = [
        x
        for x in _sanitize_horse_list(list(output.longshot_horses or []), allowed_horses)
        if x not in set(key_horses + secondary_horses)
    ]
    max_ticket_count = int(output.max_ticket_count or 0)
    if max_ticket_count < 0:
        max_ticket_count = 0
    hard_cap = int(input_obj.constraints.max_tickets_per_race or 0)
    if hard_cap > 0:
        max_ticket_count = min(max_ticket_count, hard_cap)
    budget_cap = int(input_obj.constraints.race_budget_yen or 0) or int(input_obj.constraints.bankroll_yen or 0)
    marks = _sanitize_marks(list(output.marks or []), allowed_horses)
    ticket_plan = _sanitize_ticket_plan(list(output.ticket_plan or []), allowed_types, allowed_horses, budget_cap)
    warnings = [str(x) for x in list(output.warnings or []) if str(x).strip()] if output.warnings else []

    participation_level = str(output.participation_level or "no_bet").strip().lower()
    bet_decision = str(output.bet_decision or "no_bet").strip().lower()
    derived_style = _infer_internal_policy_style(
        bet_decision=bet_decision,
        participation_level=participation_level,
        enabled_bet_types=enabled_bet_types,
        construction_style=str(output.construction_style or "").strip(),
    )
    buy_style = str(derived_style.get("buy_style") or "no_bet").strip().lower()
    strategy_mode = str(derived_style.get("strategy_mode") or "no_bet").strip().lower()
    if bet_decision == "bet" and not ticket_plan:
        warnings.append("MISSING_TICKET_PLAN")
    if bet_decision == "no_bet" or buy_style == "no_bet":
        bet_decision = "no_bet"
        participation_level = "no_bet"
        buy_style = "no_bet"
        strategy_mode = "no_bet"
        enabled_bet_types = []
        key_horses = []
        secondary_horses = []
        longshot_horses = []
        max_ticket_count = 0
        marks = []
        pick_ids = []
        ticket_plan = []

    focus_points = []
    for point in list(output.focus_points or []):
        point_type = str(getattr(point, "type", "") or "").strip()
        point_value = str(getattr(point, "value", "") or "").strip()
        if not point_type or not point_value:
            continue
        focus_points.append({"type": point_type, "value": point_value})

    return _model_validate(
        RacePolicyOutput,
        {
            "bet_decision": bet_decision,
            "participation_level": participation_level,
            "buy_style": buy_style,
            "strategy_mode": strategy_mode,
            "enabled_bet_types": enabled_bet_types,
            "construction_style": _derive_construction_style(
                strategy_mode,
                buy_style,
                participation_level,
            )
            if not output.construction_style
            else str(output.construction_style).strip(),
            "key_horses": key_horses,
            "secondary_horses": secondary_horses,
            "longshot_horses": longshot_horses,
            "max_ticket_count": max_ticket_count,
            "risk_tilt": str(output.risk_tilt or "low").strip().lower(),
            "reason_codes": [str(x) for x in list(output.reason_codes or []) if str(x).strip()],
            "warnings": warnings or None,
            "marks": marks,
            "pick_ids": pick_ids,
            "ticket_plan": ticket_plan,
            "focus_points": focus_points,
        },
    )


def _update_last_meta(meta: Dict[str, Any], output: RacePolicyOutput) -> None:
    global _LAST_CALL_META
    _LAST_CALL_META = {
        "cache_hit": bool(meta.get("cache_hit", False)),
        "llm_latency_ms": int(meta.get("llm_latency_ms", 0) or 0),
        "fallback_reason": str(meta.get("fallback_reason", "") or ""),
        "picked_count": int(max(len(output.pick_ids or []), int(output.max_ticket_count or 0))),
        "requested_budget_yen": int(meta.get("requested_budget_yen", 0) or 0),
        "requested_race_budget_yen": int(meta.get("requested_race_budget_yen", 0) or 0),
        "reused": bool(meta.get("reused", False)),
        "source_budget_yen": int(meta.get("source_budget_yen", 0) or 0),
        "policy_version": str(meta.get("policy_version", POLICY_CACHE_VERSION) or POLICY_CACHE_VERSION),
    }
    print(
        "[gemini_policy] cache_hit={cache_hit} llm_latency_ms={llm_latency_ms} "
        "fallback_reason={fallback_reason} picked_count={picked_count} "
        "requested_budget_yen={requested_budget_yen} requested_race_budget_yen={requested_race_budget_yen} "
        "reused={reused} source_budget_yen={source_budget_yen} policy_version={policy_version}".format(
            cache_hit=int(_LAST_CALL_META["cache_hit"]),
            llm_latency_ms=_LAST_CALL_META["llm_latency_ms"],
            fallback_reason=_LAST_CALL_META["fallback_reason"],
            picked_count=_LAST_CALL_META["picked_count"],
            requested_budget_yen=_LAST_CALL_META["requested_budget_yen"],
            requested_race_budget_yen=_LAST_CALL_META["requested_race_budget_yen"],
            reused=int(bool(_LAST_CALL_META["reused"])),
            source_budget_yen=_LAST_CALL_META["source_budget_yen"],
            policy_version=_LAST_CALL_META["policy_version"],
        )
    )


def get_last_call_meta() -> Dict[str, Any]:
    return dict(_LAST_CALL_META)


def call_gemini_policy(
    input: RacePolicyInput,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_s: int = 60,
    cache_enable: bool = True,
) -> RacePolicyOutput:
    input_obj = _model_validate(RacePolicyInput, input)
    request_meta = _build_request_meta(input_obj)
    cache_dir = DEFAULT_CACHE_DIR
    _ensure_cache_dir(cache_dir)
    cache_path = cache_dir / f"{_cache_key(input_obj, model)}.json"

    if bool(cache_enable):
        cached = _read_cache(cache_path)
        if cached is not None:
            _update_last_meta(
                {
                    **request_meta,
                    "cache_hit": True,
                    "llm_latency_ms": 0,
                    "fallback_reason": "cache",
                },
                cached,
            )
            return cached

    fallback_reason = ""
    llm_latency_ms = 0
    output: Optional[RacePolicyOutput] = None

    mock_enabled = str(os.environ.get("GEMINI_POLICY_MOCK", "")).strip() == "1"
    if mock_enabled:
        fallback_reason = "mock_mode"
        output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
    else:
        api_key = str(os.environ.get("GEMINI_API_KEY", "") or "").strip()
        if not api_key:
            fallback_reason = "missing_api_key"
            output = fallback_no_bet_policy(input_obj, fallback_reason=fallback_reason)
        elif not _TOKEN_BUCKET.consume(1.0):
            fallback_reason = "rate_limited_local"
            output = fallback_no_bet_policy(input_obj, fallback_reason=fallback_reason)
        else:
            prompt = _make_prompt(input_obj)
            retry_count_raw = str(os.environ.get("GEMINI_POLICY_RETRIES", "3") or "3").strip()
            try:
                max_attempts = max(1, int(retry_count_raw))
            except ValueError:
                max_attempts = 3
            for attempt_idx in range(max_attempts):
                start = time.perf_counter()
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        fut = executor.submit(_call_gemini_once, prompt, model, api_key)
                        raw_text = fut.result(timeout=max(1, int(timeout_s or 1)))
                    llm_latency_ms = int((time.perf_counter() - start) * 1000)
                    payload = _parse_json_payload(raw_text)
                    parsed = _model_validate(RacePolicyOutput, payload)
                    output = _sanitize_output(parsed, input_obj)
                    fallback_reason = ""
                    break
                except concurrent.futures.TimeoutError:
                    fallback_reason = "timeout"
                except json.JSONDecodeError:
                    fallback_reason = "json_parse_failed"
                except ValueError as exc:
                    if "unknown id" in str(exc).lower():
                        fallback_reason = "unknown_pick_id"
                    else:
                        fallback_reason = "value_error"
                except Exception as exc:
                    text = str(exc).lower()
                    if "503" in text or "unavailable" in text or "high demand" in text:
                        fallback_reason = "service_unavailable"
                    elif "429" in text or "quota" in text or "rate" in text:
                        fallback_reason = "quota_or_429"
                    elif "api key" in text or "permission" in text or "auth" in text:
                        fallback_reason = "auth_error"
                    else:
                        fallback_reason = "network_or_sdk_error"
                if output is not None or attempt_idx + 1 >= max_attempts:
                    break
                if fallback_reason in ("timeout", "service_unavailable", "network_or_sdk_error"):
                    time.sleep(min(3.0, 1.0 + attempt_idx))

            if output is None:
                output = fallback_no_bet_policy(input_obj, fallback_reason=fallback_reason)

    final_output = _sanitize_output(output, input_obj)
    meta = {
        **request_meta,
        "cache_hit": False,
        "llm_latency_ms": int(llm_latency_ms),
        "fallback_reason": str(fallback_reason or ""),
    }
    if bool(cache_enable):
        _write_cache(cache_path, final_output, meta)
    _update_last_meta(meta, final_output)
    return final_output


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "RacePolicyInput",
    "RacePolicyOutput",
    "call_gemini_policy",
    "deterministic_policy",
    "fallback_no_bet_policy",
    "get_last_call_meta",
    "get_policy_cache_key",
]

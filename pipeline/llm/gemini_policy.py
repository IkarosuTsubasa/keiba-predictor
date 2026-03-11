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
POLICY_CACHE_VERSION = "gemini_policy_v11"
POLICY_PROMPT_VERSION = "gemini_policy_prompt_v11"
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
    ai: PolicyAIStats
    marks_top5: List[MarkTop5] = Field(default_factory=list)
    predictions: List[PolicyPrediction] = Field(default_factory=list)
    predictions_full: List[Dict[str, Any]] = Field(default_factory=list)
    pair_odds_top: List[PairOddsSnapshot] = Field(default_factory=list)
    odds_full: Dict[str, Any] = Field(default_factory=dict)
    prediction_field_guide: Dict[str, str] = Field(default_factory=dict)
    multi_predictor: Dict[str, Any] = Field(default_factory=dict)
    portfolio_history: Dict[str, Any] = Field(default_factory=dict)
    candidates: List[PolicyCandidate] = Field(default_factory=list)
    constraints: PolicyConstraints


class FocusPoint(BaseModel):
    type: Literal["horse", "pair", "bet_type", "concept"]
    value: str


class PolicyMark(BaseModel):
    symbol: Literal["◎", "○", "▲", "△", "☆"]
    horse_no: str


class PolicyTicketPlan(BaseModel):
    id: str
    stake_yen: int


class RacePolicyOutput(BaseModel):
    bet_decision: Literal["bet", "no_bet"]
    participation_level: Literal["no_bet", "small_bet", "normal_bet"]
    buy_style: Literal["no_bet", "place_only", "place_focus", "balanced", "win_focus", "pair_focus", "conservative"]
    strategy_mode: Literal[
        "no_bet",
        "place_only",
        "place_focus",
        "balanced",
        "win_focus",
        "pair_focus",
        "spread",
        "conservative_single",
        "small_probe",
    ]
    enabled_bet_types: List[Literal["win", "place", "wide", "quinella", "exacta", "trio", "trifecta"]] = Field(default_factory=list)
    construction_style: Optional[Literal["single_axis", "pair_spread", "value_hunt", "conservative_single"]] = None
    key_horses: List[str] = Field(default_factory=list)
    secondary_horses: List[str] = Field(default_factory=list)
    longshot_horses: List[str] = Field(default_factory=list)
    max_ticket_count: int
    risk_tilt: Literal["low", "medium", "high"]
    strategy_text_ja: str
    bet_tendency_ja: str
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
    "buy_style": "",
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
        "ai": _model_dump(input_obj.ai),
        "marks_top5": [_model_dump(x) for x in list(input_obj.marks_top5 or [])],
        "predictions": [_model_dump(x) for x in list(input_obj.predictions or [])],
        "predictions_full": list(input_obj.predictions_full or []),
        "pair_odds_top": [_model_dump(x) for x in list(input_obj.pair_odds_top or [])],
        "odds_full": dict(input_obj.odds_full or {}),
        "prediction_field_guide": dict(input_obj.prediction_field_guide or {}),
        "multi_predictor": dict(input_obj.multi_predictor or {}),
        "portfolio_history": dict(input_obj.portfolio_history or {}),
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
        horse_no = str(getattr(row, "horse_no", "") or "").strip()
        if horse_no and horse_no not in seen:
            seen.add(horse_no)
            horses.append(horse_no)
    for cand in input_obj.candidates:
        for leg in list(cand.legs or []):
            text = str(leg or "").strip()
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


def _render_strategy_text(
    bet_decision: str,
    participation_level: str,
    buy_style: str,
    strategy_mode: str,
    has_longshot: bool,
    enabled_bet_types: Optional[List[str]] = None,
) -> Dict[str, str]:
    type_labels = {
        "win": "単勝",
        "place": "複勝",
        "wide": "ワイド",
        "quinella": "馬連",
        "exacta": "馬単",
        "trio": "三連複",
        "trifecta": "三連単",
    }
    ordered_types = []
    seen_types = set()
    for value in list(enabled_bet_types or []):
        text = str(value or "").strip().lower()
        if (not text) or (text in seen_types):
            continue
        seen_types.add(text)
        ordered_types.append(text)
    labels = [type_labels.get(item, item) for item in ordered_types]
    if not labels:
        labels = ["券種未設定"]
    if len(labels) == 1:
        type_text = labels[0]
    elif len(labels) == 2:
        type_text = f"{labels[0]}・{labels[1]}"
    else:
        type_text = "・".join(labels[:-1]) + f" を中心に、{labels[-1]}も候補に入れる形"

    if bet_decision == "no_bet":
        return {
            "strategy_text_ja": "優位性が薄く、軽く入る形も作りにくいため、今回は見送りとします。\n券種を絞っても買う根拠が弱く、無理に参加しない判断を優先します。",
            "bet_tendency_ja": "買い目傾向：見送り",
        }
    if participation_level == "small_bet":
        strategy_text = (
            "混戦寄りで断定しにくい面はありますが、完全に見送るほどではないと判断しました。\n"
            f"今回は点数を絞り、{type_text}を中心に小さく参加する方針です。"
        )
    elif buy_style == "win_focus":
        strategy_text = (
            "上位の軸は比較的見えており、通常参加できるレースと見ています。\n"
            f"今回は{type_text}を主軸に、期待値と配分のバランスを取りながら組み立てます。"
        )
    elif strategy_mode in ("pair_focus", "spread"):
        strategy_text = (
            "単独の軸だけでなく、組み合わせや並びまで含めて妙味を取りにいけるレースと見ています。\n"
            f"今回は{type_text}を中心に、候補の中から合理的な組み合わせを選ぶ方針です。"
        )
    else:
        strategy_text = (
            "上位の信頼は極端ではありませんが、参加の根拠は十分にあると判断しました。\n"
            f"今回は{type_text}を中心に、無理のない範囲で期待値を取りにいく方針です。"
        )
    if has_longshot:
        strategy_text += "\n高配当側に妙味がある候補もあるため、本線を崩さない範囲で補助的に評価します。"
    return {
        "strategy_text_ja": strategy_text,
        "bet_tendency_ja": f"買い目傾向：{type_text}",
    }


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
        text = _render_strategy_text("no_bet", "no_bet", "no_bet", "no_bet", False, [])
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
                "buy_style": "no_bet",
                "strategy_mode": "no_bet",
                "enabled_bet_types": [],
                "construction_style": "conservative_single",
                "key_horses": [],
                "secondary_horses": [],
                "longshot_horses": [],
                "max_ticket_count": 0,
                "risk_tilt": "low",
                "strategy_text_ja": text["strategy_text_ja"],
                "bet_tendency_ja": text["bet_tendency_ja"],
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
        text = _render_strategy_text("no_bet", "no_bet", "no_bet", "no_bet", False, [])
        warnings = ["NO_POSITIVE_EV"]
        if fallback_reason:
            warnings.append(f"FALLBACK_{str(fallback_reason).upper()}")
        return _model_validate(
            RacePolicyOutput,
            {
                "bet_decision": "no_bet",
                "participation_level": "no_bet",
                "buy_style": "no_bet",
                "strategy_mode": "no_bet",
                "enabled_bet_types": [],
                "construction_style": "conservative_single",
                "key_horses": [],
                "secondary_horses": [],
                "longshot_horses": [],
                "max_ticket_count": 0,
                "risk_tilt": "low",
                "strategy_text_ja": text["strategy_text_ja"],
                "bet_tendency_ja": text["bet_tendency_ja"],
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
    text = _render_strategy_text(
        "bet",
        participation_level,
        buy_style,
        strategy_mode,
        bool(longshot_horses),
        enabled,
    )
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
            "buy_style": buy_style,
            "strategy_mode": strategy_mode,
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
            "strategy_text_ja": text["strategy_text_ja"],
            "bet_tendency_ja": text["bet_tendency_ja"],
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
    input_json = _stable_json_dumps(payload)
    schema_json = _stable_json_dumps(schema)
    constraints = input_obj.constraints
    ticket_cap_text = (
        "【ローカル制約】\n"
        f"- max_tickets_per_race: {int(constraints.max_tickets_per_race)}\n"
        "- あなたが購入方針だけでなく、実際の買い目と金額まで決めてください。\n"
        "- ローカル側はデータ整形・検証・記録だけを行い、買い目の選定や配分には介入しません。\n"
        "- bet_decision が bet の場合は、必ず ticket_plan に実際の買い目と stake_yen を入れてください。\n"
        "- ticket_plan が空なら、ローカル側は購入しません。\n\n"
    )
    full_data_text = (
        "【入力データの読み方】\n"
        "- predictions は要約版、predictions_full は全馬・全列の予測テーブルです。\n"
        "- multi_predictor には v1-v4 全 predictor の要約・設計上の特徴・共識表が入っています。\n"
        "- multi_predictor.profiles は各 predictor の設計上の強みです。絶対評価ではなく、視点の違いとして扱ってください。\n"
        "- multi_predictor.summaries は predictor ごとの上位馬一覧です。\n"
        "- multi_predictor.consensus は馬番単位で揃えた共識表です。top1_votes / top3_votes / avg_pred_rank を優先的に見てください。\n"
        "- multi_predictor.performance には current_context と、現在の分類範囲（central_turf / central_dirt / local）で集計した predictor 別 hit rate が入っています。\n"
        "- multi_predictor.performance.current_scope_history の samples が少ない場合は弱い参考情報です。samples が十分ある predictor ほど、その predictor の見解信頼度判断に使ってください。\n"
        "- odds_full には win/place/wide/quinella/exacta/trio/trifecta の全量オッズが入っています。\n"
        "- prediction_field_guide には predictions_full の各列が何を意味するかの説明があります。\n"
        "- 要約だけで判断せず、multi_predictor・predictions_full・odds_full・prediction_field_guide を必ず参照して考えてください。\n"
        "- candidates や要約情報は補助材料です。最終判断では multi_predictor・predictions_full・odds_full を優先して参照してください。\n"
        "- 特に軸馬・相手・券種構成を決める際は、全馬の予測順位、確率、スコア、オッズのバランスを見てください。\n"
        "- 入力に含まれる horse_no / pair / candidates の範囲から逸脱してはいけません。\n\n"
    )
    multi_predictor_text = (
        "【4 predictor の使い方】\n"
        "- v1 は総合バランス型の主軸視点です。まず基準線として参照してください。\n"
        "- v2 は上位抽出・能力比較寄りの視点です。強い上位候補の濃淡確認に向いています。\n"
        "- v3 は市場融合・説明性寄りの視点です。値頃感や市場整合性の確認に向いています。\n"
        "- v4 は文脈適性ハイブリッド型です。Top3確率の分類と順位付けを混合し、コース・距離・馬場条件への適合を強く見ています。\n"
        "- multi_predictor.performance.current_scope_history を使って、現在が中央芝・中央ダート・地方のどれかに応じた predictor の履歴命中率を参照してください。\n"
        "- ある predictor が現在の分類範囲で長期的に弱いなら、その predictor の単独主張は割り引いてください。逆に同分類で samples が十分あり hit rate が安定して高い predictor はやや重く見て構いません。\n"
        "- ただし samples が少ない場合は過信せず、共識・個別予測・オッズとの整合を優先してください。\n"
        "- 4 路の top1/top3 の共識が強い馬は軸候補です。ただし、オッズとのバランスが悪い場合は券種や配分を柔軟に調整してください。\n"
        "- 4 路の見解差が大きい場合は、単勝・複勝、馬連・ワイド・馬単などの組み合わせ系、三連系、見送りのいずれが最も合理的かを、そのレースの予測とオッズに基づいて中立的に判断してください。\n"
        "- 1 路だけが強く推す穴馬は、そのまま採用しないでください。オッズの裏付け、他 predictor の否定度、candidates の EV を合わせて判断してください。\n"
        "- 最終判断は『4 predictor の共識/見解差』と『現在のオッズ』の両方が必要です。どちらか片方だけで決めてはいけません。\n\n"
    )
    portfolio_history_text = (
        "【あなた自身の購入履歴と資金推移】\n"
        "- portfolio_history.today には今日ここまでの開始本金、確定損益、未決済で拘束中の金額、利用可能な本金が入っています。\n"
        "- portfolio_history.recent_days には直近の日別損益と投資額が入っています。\n"
        "- portfolio_history.lookback_summary には直近期間の総 stake / payout / profit / hit_rate / roi が入っています。\n"
        "- portfolio_history.bet_type_breakdown には券種ごとの損益傾向が入っています。\n"
        "- portfolio_history.recent_tickets には最近の購入・決済の軌跡が入っています。\n"
        "- 今回の判断では、現在レースの予測とオッズを最優先にしつつ、自分の最近の損益推移と買い方の癖も確認してください。\n"
        "- 直近で連敗・回撤が大きい、または今日すでに open_stake が重い場合は、無理に取り返しに行かず、より保守的な戦略に寄せてください。\n"
        "- 逆に直近で勝っていても、過信して点数やリスクを膨らませないでください。履歴は反省材料であり、強気化の免罪符ではありません。\n\n"
    )
    return (
        "あなたは中央競馬・地方競馬のデータ分析を行う「馬券戦略アナリスト」です。\n"
        "入力されたレース情報・予測・オッズ・市場情報をもとに、"
        "このレースで最も合理的な購入方針を決定してください。\n\n"
        "あなたはこのレースでどう買うかだけでなく、何をいくら買うかまで決めてください。\n"
        "ローカル実行側はあなたの出力を検証して記録するだけで、買い目や金額の再構成は行いません。\n\n"
        "【判断対象】\n"
        "- レースの難易度\n"
        "- 予測上位の優位性\n"
        "- オッズとのバランス（value）\n"
        "- フィールドの混戦度\n"
        "- 信頼度 / 安定性\n\n"
        "これらを総合して、以下を決定してください。\n"
        "- bet_decision\n"
        "- participation_level\n"
        "- buy_style\n"
        "- strategy_mode\n"
        "- enabled_bet_types\n"
        "- key_horses\n"
        "- secondary_horses\n"
        "- longshot_horses\n"
        "- max_ticket_count\n"
        "- risk_tilt\n"
        "- strategy_text_ja\n"
        "- bet_tendency_ja\n"
        "- reason_codes\n\n"
        "【重要ルール】\n"
        "1. 見送り・少額参加・通常参加は中立に選ぶ\n"
        "no_bet / small_bet / normal_bet のどれも選択可能です。\n"
        "無理に参加する必要も、無理に見送る必要もありません。予測優位性、オッズ妙味、不確実性、券種構成を総合して判断してください。\n\n"
        "2. 特定の券種を先入観で優先しない\n"
        "win / place / wide / quinella / exacta / trio / trifecta のうち、入力に存在する candidates の中から最も合理的なものを選んでください。\n"
        "単勝系・複勝系・組み合わせ系・三連系のどれにも初期バイアスを持たず、期待値とリスクのバランスで判断してください。\n\n"
        "3. 高オッズも低オッズも中立に扱う\n"
        "高オッズだから買う、低オッズだから避ける、のような固定観念は持たず、予測確率と払戻期待のバランスを見てください。\n\n"
        "4. 混戦レースの扱いも固定化しない\n"
        "混戦だから必ず保守、という前提は置かず、分散、見送り、組み合わせ重視、単独重視のどれが妥当かをデータから決めてください。\n\n"
        "5. 券種構成は候補集合から逆算する\n"
        "enabled_bet_types は、今回の race で優位性があると判断した candidates の bet_type を中心に選んでください。\n"
        "1種類に絞ってもよく、複数種類を併用しても構いません。\n\n"
        "6. 点数は必要最小限\n"
        "max_ticket_count は合理的な範囲で設定し、過剰な多点買いは避けてください。\n\n"
        "7. 馬の選び方\n"
        "- key_horses は戦略の中心になる馬\n"
        "- secondary_horses は相手候補\n"
        "- longshot_horses は穴として考慮する馬\n"
        "ただし無理に全カテゴリを埋める必要はありません。\n\n"
        "8. 資金管理\n"
        "- constraints.bankroll_yen は今日この時点で残っている共有本金です。\n"
        "- constraints.race_budget_yen はこのレースで使ってよい上限です。\n"
        "- 1 日に 5〜6 レース程度買う可能性がある前提で、今回 1 レースに資金を寄せすぎないでください。\n"
        "- 今日の残り本金を見ながら、後続レースに回す余力を意識して配分してください。\n"
        "- ticket_plan を使って、残り予算の範囲で今回の買い目と金額を決めてください。\n"
        "- stake_yen は 100 円単位で、ticket_plan 全体の合計は race_budget_yen を超えないでください。\n\n"
        "9. 出力形式\n"
        "必ず指定された JSON schema に従って出力してください。\n"
        "説明文は strategy_text_ja と bet_tendency_ja に記述してください。\n"
        "JSON 以外のテキストは出力しないでください。\n\n"
        "【追加の絶対ルール】\n"
        "- 入力に存在しない馬・組み合わせ・券種を創作しない\n"
        "- enabled_bet_types は candidates に存在する bet_type のみ使用可能\n"
        "- key_horses / secondary_horses / longshot_horses / marks / focus_points に入れる horse や pair も入力に存在するもののみ使用可能\n"
        "- pick_ids は補助情報です。実際の購入は ticket_plan を基準にします。\n"
        "- ticket_plan の id は candidates[].id のみ使用可能です。\n"
        "- 特定の券種や買い方を機械的に優遇しない\n\n"
        "【判断の実務ルール】\n"
        "- participation_level は no_bet / small_bet / normal_bet から選んでください。\n"
        "- buy_style は no_bet / place_only / place_focus / balanced / win_focus / pair_focus / conservative から選んでください。\n"
        "- strategy_mode は no_bet / place_only / place_focus / balanced / win_focus / pair_focus / spread / conservative_single / small_probe から選んでください。\n"
        "- focus_points は horse / pair / bet_type / concept のみ使用可能です。\n"
        "- marks は ◎ / ○ / ▲ / △ / ☆ を高評価順に必要なものだけ使ってください。\n"
        "- buy_style / strategy_mode は名前に引きずられず、今回の買い目構成を最も近く表すものを選んでください。\n"
        "- max_ticket_count は期待値の集中度と分散の必要性に応じて設定してください。\n"
        "- risk_tilt は券種の種類ではなく、今回の配分と不確実性に応じて low / medium / high を選んでください。\n"
        "- strategy_text_ja は 2〜4 文の自然な日本語で、レースの見立て、参加レベル、主軸券種、見送りなら理由、small_bet なら軽く参加する理由を含めてください。\n"
        "- bet_tendency_ja は 1 行のみで書いてください。\n"
        "- 参加判断が bet のときは、必ず ticket_plan に実際の買い目を入れてください。\n\n"
        "reason_codes は次から必要なものだけを選んでください。\n"
        "- MIXED_FIELD\n"
        "- NORMAL_FIELD\n"
        "- STRONG_FAVORITE\n"
        "- LOW_CONFIDENCE\n"
        "- LOW_STABILITY\n"
        "- VALUE_PRESENT\n"
        "- NO_VALUE\n"
        "- HIGH_ODDS_ONE_SHOT\n"
        "- PLACE_FOCUS\n"
        "- PAIR_FOCUS\n"
        "- WIN_TILT\n"
        "- CONSERVATIVE\n"
        "- SMALL_BET\n"
        "- NO_BET\n\n"
        f"{ticket_cap_text}"
        f"{full_data_text}"
        f"{multi_predictor_text}"
        f"{portfolio_history_text}"
        "--------------------------------\n"
        "【入力JSON】\n"
        "--------------------------------\n"
        "<INPUT_JSON>\n"
        f"{input_json}\n"
        "</INPUT_JSON>\n\n"
        "--------------------------------\n"
        "【出力JSON】\n"
        "--------------------------------\n"
        "{\n"
        '  "bet_decision": "bet | no_bet",\n'
        '  "participation_level": "no_bet | small_bet | normal_bet",\n'
        '  "buy_style": "no_bet | place_only | place_focus | balanced | win_focus | pair_focus | conservative",\n'
        '  "strategy_mode": "no_bet | place_only | place_focus | balanced | win_focus | pair_focus | spread | conservative_single | small_probe",\n'
        '  "enabled_bet_types": [],\n'
        '  "key_horses": [],\n'
        '  "secondary_horses": [],\n'
        '  "longshot_horses": [],\n'
        '  "marks": [{"symbol": "◎ | ○ | ▲ | △ | ☆", "horse_no": ""}],\n'
        '  "focus_points": [{"type": "horse | pair | bet_type | concept", "value": ""}],\n'
        '  "max_ticket_count": 0,\n'
        '  "risk_tilt": "low | medium | high",\n'
        '  "strategy_text_ja": "",\n'
        '  "bet_tendency_ja": "",\n'
        '  "reason_codes": [],\n'
        '  "pick_ids": [],\n'
        '  "ticket_plan": [{"id": "", "stake_yen": 100}],\n'
        '  "warnings": []\n'
        "}\n\n"
        "strict JSON only。response_json_schema は以下です。\n"
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
    allowed = {str(x) for x in allowed_horses}
    out = []
    seen = set()
    for value in list(values or []):
        text = str(value or "").strip()
        if (not text) or (text in seen) or (text not in allowed):
            continue
        seen.add(text)
        out.append(text)
    return out


def _sanitize_marks(marks: List[PolicyMark], allowed_horses: List[str]) -> List[Dict[str, str]]:
    allowed = {str(x) for x in allowed_horses}
    out = []
    seen_symbols = set()
    seen_horses = set()
    for mark in list(marks or []):
        symbol = str(getattr(mark, "symbol", "") or "").strip()
        horse_no = str(getattr(mark, "horse_no", "") or "").strip()
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


def _sanitize_ticket_plan(ticket_plan: List[PolicyTicketPlan], allowed_ids: set, max_budget: int) -> List[Dict[str, int]]:
    budget_cap = max(0, int(max_budget or 0))
    out = []
    seen_ids = set()
    used = 0
    for item in list(ticket_plan or []):
        ticket_id = str(getattr(item, "id", "") or "").strip()
        stake_yen = int(getattr(item, "stake_yen", 0) or 0)
        if (not ticket_id) or (ticket_id in seen_ids) or (ticket_id not in allowed_ids):
            continue
        if stake_yen <= 0:
            continue
        stake_yen = int(stake_yen // 100) * 100
        if stake_yen <= 0:
            continue
        if budget_cap > 0 and used + stake_yen > budget_cap:
            continue
        seen_ids.add(ticket_id)
        used += stake_yen
        out.append({"id": ticket_id, "stake_yen": stake_yen})
    return out


def _sanitize_output(output: RacePolicyOutput, input_obj: RacePolicyInput) -> RacePolicyOutput:
    allowed_ids = {str(c.id) for c in input_obj.candidates}
    allowed_types = {str(x).strip().lower() for x in list(input_obj.constraints.allowed_types or []) if str(x).strip()}
    allowed_horses = _horse_pool(input_obj)

    pick_ids = []
    seen_ids = set()
    for pid in list(output.pick_ids or []):
        sid = str(pid).strip()
        if (not sid) or (sid in seen_ids):
            continue
        if sid not in allowed_ids:
            continue
        seen_ids.add(sid)
        pick_ids.append(sid)

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
    ticket_plan = _sanitize_ticket_plan(list(output.ticket_plan or []), allowed_ids, budget_cap)
    warnings = [str(x) for x in list(output.warnings or []) if str(x).strip()] if output.warnings else []

    participation_level = str(output.participation_level or "no_bet").strip().lower()
    bet_decision = str(output.bet_decision or "no_bet").strip().lower()
    buy_style = str(output.buy_style or "no_bet").strip().lower()
    strategy_mode = str(output.strategy_mode or "no_bet").strip().lower()
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
            "strategy_text_ja": str(output.strategy_text_ja or "").strip(),
            "bet_tendency_ja": str(output.bet_tendency_ja or "").strip(),
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
        "buy_style": str(output.buy_style or ""),
        "requested_budget_yen": int(meta.get("requested_budget_yen", 0) or 0),
        "requested_race_budget_yen": int(meta.get("requested_race_budget_yen", 0) or 0),
        "reused": bool(meta.get("reused", False)),
        "source_budget_yen": int(meta.get("source_budget_yen", 0) or 0),
        "policy_version": str(meta.get("policy_version", POLICY_CACHE_VERSION) or POLICY_CACHE_VERSION),
    }
    print(
        "[gemini_policy] cache_hit={cache_hit} llm_latency_ms={llm_latency_ms} "
        "fallback_reason={fallback_reason} picked_count={picked_count} buy_style={buy_style} "
        "requested_budget_yen={requested_budget_yen} requested_race_budget_yen={requested_race_budget_yen} "
        "reused={reused} source_budget_yen={source_budget_yen} policy_version={policy_version}".format(
            cache_hit=int(_LAST_CALL_META["cache_hit"]),
            llm_latency_ms=_LAST_CALL_META["llm_latency_ms"],
            fallback_reason=_LAST_CALL_META["fallback_reason"],
            picked_count=_LAST_CALL_META["picked_count"],
            buy_style=_LAST_CALL_META["buy_style"],
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
            output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
        elif not _TOKEN_BUCKET.consume(1.0):
            fallback_reason = "rate_limited_local"
            output = deterministic_policy(input_obj, fallback_reason=fallback_reason)
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
                output = deterministic_policy(input_obj, fallback_reason=fallback_reason)

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
    "get_last_call_meta",
    "get_policy_cache_key",
]

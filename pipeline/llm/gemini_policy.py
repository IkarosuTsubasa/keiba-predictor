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
POLICY_CACHE_VERSION = "gemini_policy_v15"
POLICY_PROMPT_VERSION = "gemini_policy_prompt_v20"
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
    bet_type: Literal["wide", "quinella", "exacta", "trio"]
    pair: str
    odds: float


class PolicyCandidate(BaseModel):
    id: str
    bet_type: Literal["win", "place", "wide", "quinella", "exacta", "trio"]
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
    allowed_types: List[Literal["win", "place", "wide", "quinella", "exacta", "trio"]] = Field(default_factory=list)


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
    bet_type: Literal["win", "place", "wide", "quinella", "exacta", "trio"]
    legs: List[str]
    stake_yen: int


class PolicySelectedTicket(BaseModel):
    id: str
    stake_yen: int


class RacePolicyOutput(BaseModel):
    bet_decision: Literal["bet", "no_bet"]
    participation_level: Literal["no_bet", "small_bet", "normal_bet"]
    enabled_bet_types: List[Literal["win", "place", "wide", "quinella", "exacta", "trio"]] = Field(default_factory=list)
    construction_style: Optional[Literal["single_axis", "pair_spread", "value_hunt", "conservative_single"]] = None
    key_horses: List[str] = Field(default_factory=list)
    secondary_horses: List[str] = Field(default_factory=list)
    longshot_horses: List[str] = Field(default_factory=list)
    max_ticket_count: int = 0
    risk_tilt: Literal["low", "medium", "high"] = "low"
    # 策略解释标签。由本地规则统一补齐，用来说明这次买法/见送背后的判断依据。
    reason_codes: List[str] = Field(default_factory=list)
    comment: Optional[str] = None
    warnings: Optional[List[str]] = None
    marks: List[PolicyMark] = Field(default_factory=list)
    selected_tickets: List[PolicySelectedTicket] = Field(default_factory=list)
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
    "execution_status": "unknown",
    "picked_count": 0,
    "requested_budget_yen": 0,
    "requested_race_budget_yen": 0,
    "reused": False,
    "source_budget_yen": 0,
    "policy_version": POLICY_CACHE_VERSION,
}


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _dedupe_text_items(values: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in list(values or []):
        text = str(item or "").strip()
        if (not text) or (text in seen):
            continue
        seen.add(text)
        out.append(text)
    return out


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


def _canonical_legs_for_bet_type(bet_type: str, legs: List[str]) -> List[str]:
    normalized = [_normalize_horse_no_text(x) for x in list(legs or []) if _normalize_horse_no_text(x)]
    if str(bet_type or "").strip().lower() in ("wide", "quinella", "trio"):
        return sorted(normalized, key=lambda x: int(x) if x.isdigit() else x)
    return normalized


def _ticket_key_from_parts(bet_type: str, legs: List[str]) -> str:
    bet_type_text = str(bet_type or "").strip().lower()
    canon_legs = _canonical_legs_for_bet_type(bet_type_text, legs)
    return f"{bet_type_text}:{'-'.join(canon_legs)}"


def _build_candidate_maps(input_obj: RacePolicyInput) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_ticket_key: Dict[str, Dict[str, Any]] = {}
    for cand in list(input_obj.candidates or []):
        candidate_id = str(cand.id or "").strip()
        bet_type = str(cand.bet_type or "").strip().lower()
        legs = _canonical_legs_for_bet_type(bet_type, list(cand.legs or []))
        if (not candidate_id) or (not bet_type) or (not legs):
            continue
        row = {
            "id": candidate_id,
            "bet_type": bet_type,
            "legs": legs,
            "odds_used": round(float(cand.odds_used or 0.0), 6),
            "p_hit": round(float(cand.p_hit or 0.0), 6),
            "ev": round(float(cand.ev or 0.0), 6),
            "score": round(float(cand.score or 0.0), 6),
        }
        by_id[candidate_id] = row
        by_ticket_key[_ticket_key_from_parts(bet_type, legs)] = row
    return {"by_id": by_id, "by_ticket_key": by_ticket_key}


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
            "bankroll_yen": int(input_obj.constraints.bankroll_yen or 0),
            "race_budget_yen": int(input_obj.constraints.race_budget_yen or 0),
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
                "execution_status": str(meta.get("execution_status", "") or ""),
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
    # reason_codes 语义说明：
    # - MIXED_FIELD: 参赛头数多（14头以上），视为混战盘面
    # - NORMAL_FIELD: 参赛头数较少（13头以下），不按混战处理
    # - STRONG_FAVORITE: 顶部优势明显，前列马较强势
    # - LOW_CONFIDENCE: 模型总体把握偏弱
    # - LOW_STABILITY: 结果波动较大，稳定性不足
    # - VALUE_PRESENT: 当前候选里存在正 EV 的买法
    # - NO_VALUE: 当前候选里没有明确正 EV
    # - HIGH_ODDS_ONE_SHOT: 带一点高赔率冷门尝试
    # - PLACE_FOCUS: 策略偏保守，重心在複勝
    # - PAIR_FOCUS: 策略重心在马连/ワイド/连系组合
    # - WIN_TILT: 策略偏向単勝直取
    # - CONSERVATIVE: 以保守、小范围参与为主
    # - SMALL_BET: 即使参与也只是轻注
    # - NO_BET: 最终判断为见送り
    out: List[str] = []
    if field_size >= 14:
        # 大头数默认按“混战”处理。
        out.append("MIXED_FIELD")
    else:
        # 非大头数，视为常规盘面。
        out.append("NORMAL_FIELD")
    if float(ai.gap) >= 0.06 and float(ai.confidence_score) >= 0.62:
        # 头部领先差和信心同时够高，说明强马形态较明确。
        out.append("STRONG_FAVORITE")
    if float(ai.confidence_score) < 0.5:
        # 模型自己都不够有把握。
        out.append("LOW_CONFIDENCE")
    if float(ai.stability_score) < 0.45:
        # 波动偏大，复现性较弱。
        out.append("LOW_STABILITY")
    if has_value:
        # 至少存在一个正 EV 候选。
        out.append("VALUE_PRESENT")
    else:
        # 没找到明确值得下注的价值点。
        out.append("NO_VALUE")
    if has_longshot:
        # 这次方案里带了高赔率一击型选择。
        out.append("HIGH_ODDS_ONE_SHOT")
    if participation_level == "small_bet":
        # 参与级别被压到轻注。
        out.append("SMALL_BET")
    if buy_style == "place_focus":
        # 以複勝为主的稳健打法。
        out.append("PLACE_FOCUS")
    if buy_style in ("place_only", "conservative"):
        # 玩法收缩，优先控制回撤。
        out.append("CONSERVATIVE")
    if buy_style == "pair_focus":
        # 重心放在双马组合类票种。
        out.append("PAIR_FOCUS")
    if buy_style == "win_focus":
        # 重心放在単勝。
        out.append("WIN_TILT")
    if bet_decision == "no_bet":
        # 本场直接见送り。
        out.append("NO_BET")
    return out


def _trim_text(value: Any, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _looks_like_japanese(text: Any) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", str(text or "")))


def _looks_like_human_bet_comment(text: Any) -> bool:
    value = str(text or "").strip()
    if (not value) or (not _looks_like_japanese(value)):
        return False
    forbidden_patterns = (
        r"candidate",
        r"selected_tickets",
        r"ticket_plan",
        r"focus_points",
        r"\bev\b",
        r"p_hit",
        r"score",
        r"top3",
        r"confidence",
        r"stability",
        r"risk",
        r"model_summary",
        r"horse_summary",
        r"候補プール",
        r"優位性",
        r"正\s*ev",
        r"期待値",
        r"実行可能",
        r"参加レベル",
        r"bet_decision",
        r"no_bet",
    )
    return not any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in forbidden_patterns)


def _default_comment(
    *,
    bet_decision: str,
    participation_level: str,
    risk_tilt: str,
    has_longshot: bool,
    has_value: bool,
) -> str:
    decision = str(bet_decision or "").strip().lower()
    level = str(participation_level or "").strip().lower()
    tilt = str(risk_tilt or "").strip().lower()
    if decision != "bet":
        if has_value:
            return "気になる目はあるけど、ここは見送る。"
        return "今日はここで無理しない。"
    if has_longshot:
        if level == "small_bet":
            return "穴は気になるけど、軽くつまむ程度。"
        return "穴も一枚だけ押さえて入る。"
    if tilt == "high":
        return "配当狙いで、点数は絞って勝負。"
    if level == "small_bet":
        return "広げすぎず、良さそうなところだけ軽く。"
    if has_value:
        return "軸はこれで、相手を絞って入る。"
    return "形は決まるので、絞って入る。"


def _normalize_comment_text(
    value: Any,
    *,
    bet_decision: str,
    participation_level: str,
    risk_tilt: str,
    has_longshot: bool,
    has_value: bool,
) -> str:
    text = _trim_text(value or "", limit=100)
    if text and _looks_like_human_bet_comment(text):
        return text
    return _default_comment(
        bet_decision=bet_decision,
        participation_level=participation_level,
        risk_tilt=risk_tilt,
        has_longshot=has_longshot,
        has_value=has_value,
    )


def _compact_race_summary(input_obj: RacePolicyInput) -> Dict[str, Any]:
    race_context = dict(input_obj.race_context or {})
    out = {
        "race_id": str(input_obj.race_id or ""),
        "scope_key": str(input_obj.scope_key or ""),
        "field_size": int(input_obj.field_size or 0),
    }
    for key in (
        "race_date",
        "track_name",
        "venue",
        "surface",
        "distance",
        "race_class",
        "track_condition",
        "weather",
        "pace_label",
        "course",
    ):
        value = race_context.get(key)
        if value in (None, "", []):
            continue
        out[key] = value
    return out


def _compact_model_summary(input_obj: RacePolicyInput) -> Dict[str, Any]:
    multi_predictor = dict(input_obj.multi_predictor or {})
    consensus_rows = []
    for row in list(multi_predictor.get("consensus", []) or [])[: min(10, max(1, int(input_obj.field_size or 0)))]:
        consensus_rows.append(
            {
                "horse_no": _normalize_horse_no_text(row.get("horse_no", "")),
                "horse_name": str(row.get("horse_name", "") or ""),
                "top1_votes": int(row.get("top1_votes", 0) or 0),
                "top3_votes": int(row.get("top3_votes", 0) or 0),
                "avg_pred_rank": round(float(row.get("avg_pred_rank", 0.0) or 0.0), 3),
                "avg_top3_prob_model": round(float(row.get("avg_top3_prob_model", 0.0) or 0.0), 6),
                "avg_rank_score_norm": round(float(row.get("avg_rank_score_norm", 0.0) or 0.0), 6),
            }
        )
    performance = dict(multi_predictor.get("performance", {}) or {})
    scope_history = list(performance.get("current_scope_history", []) or [])
    compact_perf = []
    for row in scope_history[:6]:
        compact_perf.append(
            {
                "predictor_id": str(row.get("predictor_id", "") or row.get("name", "") or ""),
                "hit_rate": row.get("hit_rate"),
                "top3_rate": row.get("top3_rate"),
                "roi": row.get("roi"),
                "sample_size": row.get("sample_size"),
            }
        )
    marks_top5 = []
    for row in list(input_obj.marks_top5 or [])[:5]:
        marks_top5.append(
            {
                "horse_no": _normalize_horse_no_text(row.horse_no),
                "horse_name": str(row.horse_name or ""),
                "pred_rank": int(row.pred_rank or 0),
                "top3_prob_model": round(float(row.top3_prob_model or 0.0), 6),
            }
        )
    ai = _model_dump(input_obj.ai)
    return {
        "ai_confidence": {
            "gap": round(float(ai.get("gap", 0.0) or 0.0), 6),
            "confidence_score": round(float(ai.get("confidence_score", 0.0) or 0.0), 6),
            "stability_score": round(float(ai.get("stability_score", 0.0) or 0.0), 6),
            "risk_score": round(float(ai.get("risk_score", 0.0) or 0.0), 6),
        },
        "multi_model_ai": {
            str(key): value
            for key, value in dict(input_obj.multi_model_ai or {}).items()
            if str(key or "") in ("consensus_gap", "top1_vote_margin", "disagreement_score", "favorite_strength")
        },
        "consensus_top": consensus_rows,
        "marks_top5": marks_top5,
        "predictor_scope_performance": compact_perf,
    }


def _compact_horse_summary(input_obj: RacePolicyInput) -> List[Dict[str, Any]]:
    fact_map = {}
    for row in list(input_obj.horse_facts or []):
        horse_no = _normalize_horse_no_text(row.get("horse_no", ""))
        if horse_no:
            fact_map[horse_no] = dict(row or {})
    consensus_map = {}
    for row in list((input_obj.multi_predictor or {}).get("consensus", []) or []):
        horse_no = _normalize_horse_no_text(row.get("horse_no", ""))
        if horse_no:
            consensus_map[horse_no] = dict(row or {})
    out = []
    seen = set()
    for row in list(input_obj.predictions or []):
        horse_no = _normalize_horse_no_text(row.horse_no)
        if (not horse_no) or (horse_no in seen):
            continue
        seen.add(horse_no)
        fact = fact_map.get(horse_no, {})
        consensus = consensus_map.get(horse_no, {})
        out.append(
            {
                "horse_no": horse_no,
                "horse_name": str(row.horse_name or fact.get("horse_name", "") or ""),
                "pred_rank": int(row.pred_rank or 0),
                "top3_prob_model": round(float(row.top3_prob_model or 0.0), 6),
                "rank_score_norm": round(float(row.rank_score_norm or 0.0), 6),
                "win_odds": round(float(row.win_odds or fact.get("win_odds", 0.0) or 0.0), 6),
                "place_odds": round(float(row.place_odds or fact.get("place_odds", 0.0) or 0.0), 6),
                "top1_votes": int(consensus.get("top1_votes", 0) or 0),
                "top3_votes": int(consensus.get("top3_votes", 0) or 0),
                "avg_pred_rank": round(float(consensus.get("avg_pred_rank", 0.0) or 0.0), 3),
                "context_fit": _trim_text(
                    fact.get("context_fit")
                    or fact.get("fit_note")
                    or fact.get("style_note")
                    or fact.get("memo")
                    or "",
                    limit=60,
                ),
            }
        )
    return out


def _compact_portfolio_summary(input_obj: RacePolicyInput) -> Dict[str, Any]:
    portfolio_history = dict(input_obj.portfolio_history or {})
    today = dict(portfolio_history.get("today", {}) or {})
    lookback = dict(portfolio_history.get("lookback_summary", {}) or {})
    breakdown = list(portfolio_history.get("bet_type_breakdown", []) or [])
    compact_breakdown = []
    for row in breakdown[:6]:
        compact_breakdown.append(
            {
                "bet_type": str(row.get("bet_type", "") or ""),
                "roi": row.get("roi"),
                "hit_rate": row.get("hit_rate"),
                "count": row.get("count"),
            }
        )
    available_capital = int(today.get("available_capital", today.get("available_capital_yen", 0)) or 0)
    locked_capital = int(today.get("locked_capital", today.get("locked_capital_yen", 0)) or 0)
    streak = int(
        lookback.get("losing_streak")
        or lookback.get("consecutive_losses")
        or portfolio_history.get("losing_streak")
        or 0
    )
    return {
        "today_pnl": int(today.get("confirmed_pnl", today.get("pnl", 0)) or 0),
        "available_capital": available_capital,
        "locked_capital": locked_capital,
        "recent_overbet_flag": bool(locked_capital > max(0, available_capital)),
        "losing_streak_flag": bool(streak >= 3),
        "lookback_summary": {
            str(key): value
            for key, value in lookback.items()
            if str(key or "") in ("roi", "hit_rate", "count", "avg_stake_yen", "max_drawdown")
        },
        "bet_type_breakdown": compact_breakdown,
    }


def _compact_candidate_tickets(input_obj: RacePolicyInput) -> List[Dict[str, Any]]:
    allowed_types = {str(x).strip().lower() for x in list(input_obj.constraints.allowed_types or []) if str(x).strip()}
    out = []
    for cand in list(input_obj.candidates or []):
        bet_type = str(cand.bet_type or "").strip().lower()
        if allowed_types and bet_type not in allowed_types:
            continue
        out.append(
            {
                "id": str(cand.id or ""),
                "bet_type": bet_type,
                "legs": _canonical_legs_for_bet_type(bet_type, list(cand.legs or [])),
                "odds": round(float(cand.odds_used or 0.0), 6),
                "p_hit": round(float(cand.p_hit or 0.0), 6),
                "ev": round(float(cand.ev or 0.0), 6),
                "score": round(float(cand.score or 0.0), 6),
            }
        )
    return out


def _build_prompt_payload(input_obj: RacePolicyInput) -> Dict[str, Any]:
    constraints = input_obj.constraints
    return {
        "race_summary": _compact_race_summary(input_obj),
        "model_summary": _compact_model_summary(input_obj),
        "horse_summary": _compact_horse_summary(input_obj),
        "candidate_pool_meta": dict(input_obj.candidates_meta or {}),
        "candidate_tickets": _compact_candidate_tickets(input_obj),
        "portfolio_summary": _compact_portfolio_summary(input_obj),
        "constraints": {
            "bankroll_yen": int(constraints.bankroll_yen or 0),
            "race_budget_yen": int(constraints.race_budget_yen or 0),
            "max_tickets_per_race": int(constraints.max_tickets_per_race or 0),
            "high_odds_threshold": round(float(constraints.high_odds_threshold or 0.0), 6),
            "allowed_types": [str(x) for x in list(constraints.allowed_types or []) if str(x).strip()],
        },
    }


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

    multi_leg_types = {"wide", "quinella", "exacta", "trio"}
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
            "comment": _default_comment(
                bet_decision="no_bet",
                participation_level="no_bet",
                risk_tilt="low",
                has_longshot=False,
                has_value=False,
            ),
            "pick_ids": [],
            "selected_tickets": [],
            "ticket_plan": [],
            "warnings": warnings or None,
        },
    )


def deterministic_policy(input_obj: RacePolicyInput, fallback_reason: str = "") -> RacePolicyOutput:
    ai = input_obj.ai
    constraints = input_obj.constraints
    allowed_types = {str(x).strip().lower() for x in list(constraints.allowed_types or []) if str(x).strip()}
    candidate_by_id = dict(_build_candidate_maps(input_obj).get("by_id", {}) or {})
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
    has_combo_value = any(str(c.bet_type) in ("wide", "quinella", "exacta", "trio") and float(c.ev) > 0.0 for c in candidates)
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
                "comment": _default_comment(
                    bet_decision="no_bet",
                    participation_level="no_bet",
                    risk_tilt="low",
                    has_longshot=False,
                    has_value=has_value,
                ),
                "selected_tickets": [],
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
                "comment": _default_comment(
                    bet_decision="no_bet",
                    participation_level="no_bet",
                    risk_tilt="low",
                    has_longshot=False,
                    has_value=has_value,
                ),
                "selected_tickets": [],
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
    multi_leg_types = {"wide", "quinella", "exacta", "trio"}
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
    if top_type == "trio" and participation_level != "no_bet":
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
    budget_cap = int(constraints.race_budget_yen or 0) or int(constraints.bankroll_yen or 0)
    selected_tickets: List[Dict[str, Any]] = []
    if budget_cap >= 100 and prioritized:
        selected_candidates = prioritized[: max(1, min(int(max_ticket_count or 0), len(prioritized)))]
        unit_budget = min(
            max(1, budget_cap // 100),
            len(selected_candidates) if participation_level == "small_bet" else max(len(selected_candidates), min(len(selected_candidates) * 2, budget_cap // 100)),
        )
        per_ticket_units = max(1, unit_budget // max(1, len(selected_candidates)))
        remainder_units = max(0, unit_budget - per_ticket_units * len(selected_candidates))
        for idx, cand in enumerate(selected_candidates):
            units = per_ticket_units + (1 if idx < remainder_units else 0)
            selected_tickets.append({"id": str(cand.id), "stake_yen": int(units * 100)})
    pick_ids = [str(item.get("id", "")).strip() for item in selected_tickets if str(item.get("id", "")).strip()]
    ticket_plan = []
    for item in selected_tickets:
        candidate = dict(candidate_by_id.get(str(item.get("id", "")).strip(), {}) or {})
        if not candidate:
            continue
        ticket_plan.append(
            {
                "bet_type": str(candidate.get("bet_type", "") or "").strip().lower(),
                "legs": list(candidate.get("legs", []) or []),
                "stake_yen": int(item.get("stake_yen", 0) or 0),
            }
        )
    reason_codes = _reason_codes_for(
        ai,
        int(input_obj.field_size or 0),
        has_value,
        participation_level,
        buy_style,
        "bet",
        bool(longshot_horses),
    )
    comment = _default_comment(
        bet_decision="bet",
        participation_level=participation_level,
        risk_tilt=risk_tilt,
        has_longshot=bool(longshot_horses),
        has_value=has_value,
    )
    final_state = _coerce_final_execution_state(
        bet_decision="bet",
        participation_level=participation_level,
        enabled_bet_types=enabled,
        key_horses=key_horses,
        secondary_horses=secondary_horses,
        longshot_horses=longshot_horses,
        max_ticket_count=min(
            max(1, int(max_ticket_count)),
            max(1, int(constraints.max_tickets_per_race or max_ticket_count)),
        ),
        risk_tilt=risk_tilt,
        reason_codes=reason_codes,
        comment=comment,
        warnings=warnings,
        marks=[],
        selected_tickets=selected_tickets,
        pick_ids=pick_ids,
        ticket_plan=ticket_plan,
        focus_points=focus_points,
    )
    final_style_meta = _infer_internal_policy_style(
        bet_decision=str(final_state.get("bet_decision", "") or ""),
        participation_level=str(final_state.get("participation_level", "") or ""),
        enabled_bet_types=list(final_state.get("enabled_bet_types", []) or []),
        construction_style=construction_style,
    )
    final_state["reason_codes"] = _reason_codes_for(
        ai,
        int(input_obj.field_size or 0),
        has_value,
        str(final_state.get("participation_level", "") or ""),
        str(final_style_meta.get("buy_style", "no_bet") or "no_bet"),
        str(final_state.get("bet_decision", "") or ""),
        bool(final_state.get("longshot_horses", [])),
    )
    final_construction_style = (
        "conservative_single"
        if str(final_state.get("bet_decision", "") or "") == "no_bet"
        else construction_style
    )

    return _model_validate(
        RacePolicyOutput,
        {
            **final_state,
            "construction_style": final_construction_style,
        },
    )

def _make_prompt(input_obj: RacePolicyInput) -> str:
    schema = _model_json_schema(RacePolicyOutput)
    payload = _build_prompt_payload(input_obj)
    input_json = _stable_json_dumps(payload)
    schema_json = _stable_json_dumps(schema)
    constraints = input_obj.constraints
    race_context = dict(input_obj.race_context or {})
    budget_scope_note = str(race_context.get("budget_scope_note", "") or "").strip()
    planned_races_for_day = int(race_context.get("planned_races_for_day", 0) or 0)
    remaining_races_for_day = int(race_context.get("remaining_races_for_day", 0) or 0)
    race_sequence_for_day = int(race_context.get("race_sequence_for_day", 0) or 0)
    daily_plan_lines = []
    if budget_scope_note:
        daily_plan_lines.append(f"- {budget_scope_note}")
    if planned_races_for_day > 0:
        daily_plan_lines.append(f"- 当日の想定レース数: {planned_races_for_day}レース")
    if remaining_races_for_day > 0:
        daily_plan_lines.append(f"- このレースを含む残り想定レース数: {remaining_races_for_day}レース")
    if race_sequence_for_day > 0:
        daily_plan_lines.append(f"- 今日の進行上の位置づけ: {race_sequence_for_day}レース目")
    daily_plan_text = "\n".join(daily_plan_lines)
    if daily_plan_text:
        daily_plan_text += "\n"
    return (
        "あなたは当日の馬券ポートフォリオ最適化AIです。\n"
        "目的はこの1レース単体の的中ではなく、当日終了時の期待資金成長を最大化することです。\n"
        "無理に参加する必要はありません。優位が弱ければ no_bet を選んでください。\n\n"

        "== 役割 ==\n"
        "- あなたの仕事は candidate_tickets から選ぶことと、各候補への stake_yen 配分だけです\n"
        "- candidate_tickets に存在しない組み合わせを出力してはいけません\n"
        "- 候補探索は上流で完了済みです。あなたは候補池の最終選択者です\n"
        "- marks / focus_points / key_horses / enabled_bet_types / ticket_plan などの表示系フィールドは主目的ではありません\n"
        "- bet のときは selected_tickets を最優先で正しく返してください\n\n"

        "== 判断基準 ==\n"
        "1. candidate_tickets の edge（ev / score / p_hit）\n"
        "2. model_summary の合意度と不一致の質\n"
        "3. horse_summary の文脈適性\n"
        "4. portfolio_summary と予算制約に基づく当日資金配分\n\n"

        "== 制約 ==\n"
        f"- 現在の残り本金: {int(constraints.bankroll_yen)}円\n"
        f"- このレースの上限: {int(constraints.race_budget_yen)}円\n"
        f"{daily_plan_text}"
        f"- 最大購入点数: {int(constraints.max_tickets_per_race)}\n"
        "- 購入単位は100円刻み\n"
        "- selected_tickets の id は candidate_tickets の id と完全一致であること\n"
        "- selected_tickets の合計金額は race_budget_yen 以下\n"
        "- bankroll_yen は当日全体の運用資金であり、この1レースで大半を使い切らないこと\n"
        "- selected_tickets を実行可能に構成できないなら no_bet を返すこと\n"
        "- JSON のみ出力\n\n"

        "== 出力方針 ==\n"
        "- bet_decision が bet のときは selected_tickets を必須にする\n"
        "- selected_tickets は [{\"id\":\"candidate_id\",\"stake_yen\":300}] 形式で返す\n"
        "- comment は必ず自然な日本語で、短い1文にすること\n"
        "- comment は人間が馬券を買う前にメモするような口調にすること\n"
        "- comment では EV / score / candidate_tickets / 優位性 など機械的な言い回しを使わないこと\n"
        "- comment の例: 「軸はこれで、相手を絞って入る。」「穴は気になるけど軽く。」\n"
        "- それ以外の表示系フィールドは空やデフォルトでもよい\n\n"

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
        symbol = str((mark.get("symbol", "") if isinstance(mark, dict) else getattr(mark, "symbol", "")) or "").strip()
        horse_no = _normalize_horse_no_text(mark.get("horse_no", "") if isinstance(mark, dict) else getattr(mark, "horse_no", ""))
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
    candidate_by_ticket_key: Dict[str, Dict[str, Any]],
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
        expected_leg_count = {"win": 1, "place": 1, "wide": 2, "quinella": 2, "exacta": 2, "trio": 3}
        if len(legs) != expected_leg_count.get(bet_type, 0):
            continue
        ticket_key = _ticket_key_from_parts(bet_type, legs)
        candidate = dict(candidate_by_ticket_key.get(ticket_key, {}) or {})
        if not candidate:
            continue
        if ticket_key in seen_keys:
            continue
        stake_yen = int(stake_yen // 100) * 100
        if stake_yen <= 0:
            continue
        if budget_cap > 0 and used + stake_yen > budget_cap:
            continue
        seen_keys.add(ticket_key)
        used += stake_yen
        out.append({"bet_type": bet_type, "legs": list(candidate.get("legs", []) or legs), "stake_yen": stake_yen})
    return out


def _sanitize_selected_tickets(
    selected_tickets: List[PolicySelectedTicket],
    candidate_by_id: Dict[str, Dict[str, Any]],
    allowed_types: set,
    max_budget: int,
    max_tickets: int,
) -> List[Dict[str, Any]]:
    budget_cap = max(0, int(max_budget or 0))
    ticket_cap = max(0, int(max_tickets or 0))
    out: List[Dict[str, Any]] = []
    seen_ids = set()
    used = 0
    for item in list(selected_tickets or []):
        candidate_id = str((item.get("id", "") if isinstance(item, dict) else getattr(item, "id", "")) or "").strip()
        stake_yen = int((item.get("stake_yen", 0) if isinstance(item, dict) else getattr(item, "stake_yen", 0)) or 0)
        if (not candidate_id) or (candidate_id in seen_ids):
            continue
        candidate = dict(candidate_by_id.get(candidate_id, {}) or {})
        if not candidate:
            continue
        if allowed_types and str(candidate.get("bet_type", "") or "") not in allowed_types:
            continue
        stake_yen = int(stake_yen // 100) * 100
        if stake_yen <= 0:
            continue
        if budget_cap > 0 and used + stake_yen > budget_cap:
            continue
        if ticket_cap > 0 and len(out) >= ticket_cap:
            break
        seen_ids.add(candidate_id)
        used += stake_yen
        out.append({"id": candidate_id, "stake_yen": stake_yen})
    return out


def _selected_tickets_from_ticket_plan(
    ticket_plan: List[PolicyTicketPlan],
    candidate_by_ticket_key: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in list(ticket_plan or []):
        bet_type = str((item.get("bet_type", "") if isinstance(item, dict) else getattr(item, "bet_type", "")) or "").strip().lower()
        raw_legs = item.get("legs", []) if isinstance(item, dict) else getattr(item, "legs", [])
        legs = [_normalize_horse_no_text(x) for x in list(raw_legs or []) if _normalize_horse_no_text(x)]
        stake_yen = int((item.get("stake_yen", 0) if isinstance(item, dict) else getattr(item, "stake_yen", 0)) or 0)
        candidate = dict(candidate_by_ticket_key.get(_ticket_key_from_parts(bet_type, legs), {}) or {})
        candidate_id = str(candidate.get("id", "") or "").strip()
        if (not candidate_id) or stake_yen <= 0:
            continue
        out.append({"id": candidate_id, "stake_yen": stake_yen})
    return out


def _coerce_final_execution_state(
    *,
    bet_decision: str,
    participation_level: str,
    enabled_bet_types: List[str],
    key_horses: List[str],
    secondary_horses: List[str],
    longshot_horses: List[str],
    max_ticket_count: int,
    risk_tilt: str,
    reason_codes: List[str],
    comment: str,
    warnings: List[str],
    marks: List[Dict[str, str]],
    selected_tickets: List[Dict[str, Any]],
    pick_ids: List[str],
    ticket_plan: List[Dict[str, Any]],
    focus_points: List[Dict[str, str]],
) -> Dict[str, Any]:
    decision = str(bet_decision or "").strip().lower()
    level = str(participation_level or "").strip().lower()
    warning_list = _dedupe_text_items(list(warnings or []))
    focus_points_out = list(focus_points or [])
    if decision != "bet":
        final_reason_codes = _dedupe_text_items(list(reason_codes or []) + ["NO_BET"])
        return {
            "bet_decision": "no_bet",
            "participation_level": "no_bet",
            "enabled_bet_types": [],
            "key_horses": [],
            "secondary_horses": [],
            "longshot_horses": [],
            "max_ticket_count": 0,
            "risk_tilt": "low",
            "reason_codes": final_reason_codes,
            "comment": comment,
            "warnings": warning_list or None,
            "marks": [],
            "selected_tickets": [],
            "pick_ids": [],
            "ticket_plan": [],
            "focus_points": focus_points_out or [{"type": "concept", "value": "見送り"}],
        }
    if not list(selected_tickets or []):
        warning_list.append("NO_EXECUTABLE_TICKETS")
        warning_list = _dedupe_text_items(warning_list)
        final_reason_codes = _dedupe_text_items(list(reason_codes or []) + ["NO_BET"])
        return {
            "bet_decision": "no_bet",
            "participation_level": "no_bet",
            "enabled_bet_types": [],
            "key_horses": [],
            "secondary_horses": [],
            "longshot_horses": [],
            "max_ticket_count": 0,
            "risk_tilt": "low",
            "reason_codes": final_reason_codes,
            "comment": comment,
            "warnings": warning_list or None,
            "marks": [],
            "selected_tickets": [],
            "pick_ids": [],
            "ticket_plan": [],
            "focus_points": [{"type": "concept", "value": "見送り"}],
        }
    if level not in ("small_bet", "normal_bet"):
        level = "small_bet"
    return {
        "bet_decision": "bet",
        "participation_level": level,
        "enabled_bet_types": list(enabled_bet_types or []),
        "key_horses": list(key_horses or []),
        "secondary_horses": list(secondary_horses or []),
        "longshot_horses": list(longshot_horses or []),
        "max_ticket_count": int(max_ticket_count or 0),
        "risk_tilt": str(risk_tilt or "low").strip().lower() if str(risk_tilt or "").strip().lower() in ("low", "medium", "high") else "low",
        "reason_codes": list(reason_codes or []),
        "comment": comment,
        "warnings": warning_list or None,
        "marks": list(marks or []),
        "selected_tickets": list(selected_tickets or []),
        "pick_ids": list(pick_ids or []),
        "ticket_plan": list(ticket_plan or []),
        "focus_points": focus_points_out,
    }


def _execution_status(output: RacePolicyOutput) -> str:
    bet_decision = str(getattr(output, "bet_decision", "") or "").strip().lower()
    ticket_plan = list(getattr(output, "ticket_plan", []) or [])
    if bet_decision == "no_bet":
        return "voluntary_no_bet"
    if bet_decision == "bet" and ticket_plan:
        return "executed"
    if bet_decision == "bet" and not ticket_plan:
        return "invalid_bet_plan"
    return "unknown"


def _sanitize_output(output: RacePolicyOutput, input_obj: RacePolicyInput) -> RacePolicyOutput:
    allowed_types = {str(x).strip().lower() for x in list(input_obj.constraints.allowed_types or []) if str(x).strip()}
    allowed_horses = _horse_pool(input_obj)
    candidate_maps = _build_candidate_maps(input_obj)
    candidate_by_id = dict(candidate_maps.get("by_id", {}) or {})
    candidate_by_ticket_key = dict(candidate_maps.get("by_ticket_key", {}) or {})

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

    max_ticket_count = int(output.max_ticket_count or 0)
    if max_ticket_count < 0:
        max_ticket_count = 0
    hard_cap = int(input_obj.constraints.max_tickets_per_race or 0)
    if hard_cap > 0:
        max_ticket_count = min(max_ticket_count, hard_cap)
    budget_cap = int(input_obj.constraints.race_budget_yen or 0) or int(input_obj.constraints.bankroll_yen or 0)
    raw_ticket_plan = list(output.ticket_plan or [])
    raw_selected_tickets = list(output.selected_tickets or [])
    if (not raw_selected_tickets) and raw_ticket_plan:
        raw_selected_tickets = _selected_tickets_from_ticket_plan(raw_ticket_plan, candidate_by_ticket_key)
    selected_tickets = _sanitize_selected_tickets(
        raw_selected_tickets,
        candidate_by_id,
        allowed_types,
        budget_cap,
        hard_cap,
    )
    if not selected_tickets and list(output.pick_ids or []):
        pick_selected = [{"id": str(x).strip(), "stake_yen": 100} for x in list(output.pick_ids or []) if str(x).strip()]
        selected_tickets = _sanitize_selected_tickets(
            pick_selected,
            candidate_by_id,
            allowed_types,
            budget_cap,
            hard_cap,
        )
    pick_ids = [str(item.get("id", "")).strip() for item in selected_tickets if str(item.get("id", "")).strip()]
    ticket_plan = []
    for item in selected_tickets:
        candidate = dict(candidate_by_id.get(str(item.get("id", "")).strip(), {}) or {})
        if not candidate:
            continue
        ticket_plan.append(
            {
                "bet_type": str(candidate.get("bet_type", "") or "").strip().lower(),
                "legs": list(candidate.get("legs", []) or []),
                "stake_yen": int(item.get("stake_yen", 0) or 0),
            }
        )
    if not ticket_plan:
        ticket_plan = _sanitize_ticket_plan(
            raw_ticket_plan,
            allowed_types,
            allowed_horses,
            budget_cap,
            candidate_by_ticket_key,
        )
        if ticket_plan:
            pick_ids = [
                str(candidate_by_ticket_key.get(_ticket_key_from_parts(item.get("bet_type", ""), item.get("legs", [])), {}).get("id", "")).strip()
                for item in ticket_plan
            ]
            pick_ids = [x for x in pick_ids if x]
            selected_tickets = [
                {"id": candidate_id, "stake_yen": int(ticket.get("stake_yen", 0) or 0)}
                for candidate_id, ticket in zip(pick_ids, ticket_plan)
                if candidate_id
            ]
    horse_scores: Dict[str, int] = {}
    longshot_horses_set = set()
    selected_candidates = []
    for item in selected_tickets:
        candidate = dict(candidate_by_id.get(str(item.get("id", "")).strip(), {}) or {})
        if not candidate:
            continue
        selected_candidates.append(candidate)
        stake_yen = int(item.get("stake_yen", 0) or 0)
        for leg in list(candidate.get("legs", []) or []):
            horse_scores[leg] = int(horse_scores.get(leg, 0) or 0) + stake_yen
        if float(candidate.get("odds_used", 0.0) or 0.0) >= float(input_obj.constraints.high_odds_threshold or 0.0):
            for leg in list(candidate.get("legs", []) or []):
                longshot_horses_set.add(str(leg))
    ranked_horses = [horse for horse, _ in sorted(horse_scores.items(), key=lambda kv: (-int(kv[1]), kv[0]))]
    key_horses = ranked_horses[:1]
    secondary_horses = [horse for horse in ranked_horses[1:3] if horse not in set(key_horses)]
    longshot_horses = [horse for horse in ranked_horses if horse in longshot_horses_set and horse not in set(key_horses + secondary_horses)]
    marks = _sanitize_marks(
        [{"symbol": symbol, "horse_no": horse_no} for symbol, horse_no in zip(["◎", "○", "▲", "△", "☆"], ranked_horses[:5])],
        allowed_horses,
    )
    warnings = _dedupe_text_items(list(output.warnings or [])) if output.warnings else []

    participation_level = str(output.participation_level or "no_bet").strip().lower()
    bet_decision = str(output.bet_decision or "no_bet").strip().lower()
    derived_style = _infer_internal_policy_style(
        bet_decision=bet_decision,
        participation_level=participation_level,
        enabled_bet_types=enabled_bet_types or [str(item.get("bet_type", "")).strip().lower() for item in ticket_plan if str(item.get("bet_type", "")).strip()],
        construction_style=str(output.construction_style or "").strip(),
    )
    buy_style = str(derived_style.get("buy_style") or "no_bet").strip().lower()
    strategy_mode = str(derived_style.get("strategy_mode") or "no_bet").strip().lower()
    if len(ticket_plan) < len(raw_ticket_plan) or len(selected_tickets) < len(raw_selected_tickets):
        warnings.append("INVALID_TICKET_DROPPED")
    warnings = _dedupe_text_items(warnings)

    focus_points = []
    if key_horses:
        focus_points.append({"type": "horse", "value": key_horses[0]})
    if ticket_plan:
        focus_points.append({"type": "bet_type", "value": str(ticket_plan[0].get("bet_type", "")).strip()})
    top_pair_candidate = next((cand for cand in selected_candidates if len(list(cand.get("legs", []) or [])) >= 2), None)
    if top_pair_candidate:
        focus_points.append({"type": "pair", "value": "-".join(str(x) for x in list(top_pair_candidate.get("legs", []) or []))})
    elif bet_decision == "no_bet":
        focus_points.append({"type": "concept", "value": "見送り"})

    enabled_from_tickets = []
    seen_ticket_types = set()
    for item in ticket_plan:
        bet_type = str(item.get("bet_type", "")).strip().lower()
        if (not bet_type) or (bet_type in seen_ticket_types):
            continue
        seen_ticket_types.add(bet_type)
        enabled_from_tickets.append(bet_type)
    if enabled_from_tickets:
        enabled_bet_types = enabled_from_tickets
    if not max_ticket_count:
        max_ticket_count = len(selected_tickets)
    if ticket_plan and max_ticket_count > 0:
        max_ticket_count = min(max_ticket_count, len(ticket_plan))
    elif ticket_plan:
        max_ticket_count = len(ticket_plan)
    risk_tilt = str(output.risk_tilt or "").strip().lower()
    if risk_tilt not in ("low", "medium", "high"):
        risk_tilt = "low"
        if any(str(item.get("bet_type", "")).strip().lower() in ("exacta", "trio") for item in ticket_plan):
            risk_tilt = "high"
        elif any(float(cand.get("odds_used", 0.0) or 0.0) >= float(input_obj.constraints.high_odds_threshold or 0.0) for cand in selected_candidates):
            risk_tilt = "medium"
        elif participation_level == "normal_bet":
            risk_tilt = "medium"
    has_value = any(float(cand.get("ev", 0.0) or 0.0) > 0.0 for cand in selected_candidates) or any(
        float(c.ev or 0.0) > 0.0 for c in list(input_obj.candidates or [])
    )
    comment = _normalize_comment_text(
        output.comment,
        bet_decision=bet_decision,
        participation_level=participation_level,
        risk_tilt=risk_tilt,
        has_longshot=bool(longshot_horses),
        has_value=has_value,
    )
    reason_codes = _reason_codes_for(
        input_obj.ai,
        int(input_obj.field_size or 0),
        has_value,
        participation_level,
        buy_style,
        bet_decision,
        bool(longshot_horses),
    )
    final_state = _coerce_final_execution_state(
        bet_decision=bet_decision,
        participation_level=participation_level,
        enabled_bet_types=enabled_bet_types,
        key_horses=key_horses,
        secondary_horses=secondary_horses,
        longshot_horses=longshot_horses,
        max_ticket_count=max_ticket_count,
        risk_tilt=risk_tilt,
        reason_codes=reason_codes,
        comment=comment,
        warnings=warnings,
        marks=marks,
        selected_tickets=selected_tickets,
        pick_ids=pick_ids,
        ticket_plan=ticket_plan,
        focus_points=focus_points,
    )
    final_style_meta = _infer_internal_policy_style(
        bet_decision=str(final_state.get("bet_decision", "") or ""),
        participation_level=str(final_state.get("participation_level", "") or ""),
        enabled_bet_types=list(final_state.get("enabled_bet_types", []) or []),
        construction_style=str(output.construction_style or "").strip(),
    )
    final_state["reason_codes"] = _reason_codes_for(
        input_obj.ai,
        int(input_obj.field_size or 0),
        has_value,
        str(final_state.get("participation_level", "") or ""),
        str(final_style_meta.get("buy_style", "no_bet") or "no_bet"),
        str(final_state.get("bet_decision", "") or ""),
        bool(final_state.get("longshot_horses", [])),
    )
    final_construction_style = (
        "conservative_single"
        if str(final_state.get("bet_decision", "") or "") == "no_bet"
        else (
            _derive_construction_style(
                strategy_mode,
                buy_style,
                participation_level,
            )
            if not output.construction_style
            else str(output.construction_style).strip()
        )
    )

    return _model_validate(
        RacePolicyOutput,
        {
            **final_state,
            "construction_style": final_construction_style,
        },
    )


def _update_last_meta(meta: Dict[str, Any], output: RacePolicyOutput) -> None:
    global _LAST_CALL_META
    _LAST_CALL_META = {
        "cache_hit": bool(meta.get("cache_hit", False)),
        "llm_latency_ms": int(meta.get("llm_latency_ms", 0) or 0),
        "fallback_reason": str(meta.get("fallback_reason", "") or ""),
        "execution_status": str(meta.get("execution_status", "") or _execution_status(output)),
        "picked_count": int(max(len(output.selected_tickets or []), len(output.pick_ids or []), int(output.max_ticket_count or 0))),
        "requested_budget_yen": int(meta.get("requested_budget_yen", 0) or 0),
        "requested_race_budget_yen": int(meta.get("requested_race_budget_yen", 0) or 0),
        "reused": bool(meta.get("reused", False)),
        "source_budget_yen": int(meta.get("source_budget_yen", 0) or 0),
        "policy_version": str(meta.get("policy_version", POLICY_CACHE_VERSION) or POLICY_CACHE_VERSION),
    }
    print(
        "[gemini_policy] cache_hit={cache_hit} llm_latency_ms={llm_latency_ms} "
        "fallback_reason={fallback_reason} execution_status={execution_status} picked_count={picked_count} "
        "requested_budget_yen={requested_budget_yen} requested_race_budget_yen={requested_race_budget_yen} "
        "reused={reused} source_budget_yen={source_budget_yen} policy_version={policy_version}".format(
            cache_hit=int(_LAST_CALL_META["cache_hit"]),
            llm_latency_ms=_LAST_CALL_META["llm_latency_ms"],
            fallback_reason=_LAST_CALL_META["fallback_reason"],
            execution_status=_LAST_CALL_META["execution_status"],
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
        "execution_status": _execution_status(final_output),
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

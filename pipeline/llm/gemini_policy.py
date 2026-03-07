import concurrent.futures
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
POLICY_CACHE_VERSION = "gemini_policy_v6"
POLICY_PROMPT_VERSION = "gemini_policy_prompt_v6"
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
    bet_type: Literal["wide", "quinella"]
    pair: str
    odds: float


class PolicyCandidate(BaseModel):
    id: str
    bet_type: Literal["win", "place", "wide", "quinella"]
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
    allowed_types: List[Literal["win", "place", "wide", "quinella"]] = Field(default_factory=list)


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
    candidates: List[PolicyCandidate] = Field(default_factory=list)
    constraints: PolicyConstraints


class FocusPoint(BaseModel):
    type: Literal["horse", "pair", "bet_type", "concept"]
    value: str


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
    enabled_bet_types: List[Literal["win", "place", "wide", "quinella"]] = Field(default_factory=list)
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
    pick_ids: List[str] = Field(default_factory=list)
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
) -> Dict[str, str]:
    if bet_decision == "no_bet":
        return {
            "strategy_text_ja": "優位性が薄く、軽く入る形も作りにくいため、今回は見送りとします。\n券種を絞っても買う根拠が弱く、無理に参加しない判断を優先します。",
            "bet_tendency_ja": "買い目傾向：見送り",
        }
    if participation_level == "small_bet" and strategy_mode in ("place_only", "conservative_single", "small_probe"):
        return {
            "strategy_text_ja": "混戦寄りで上位の差は大きくありませんが、完全に見送るほどではないと判断しました。\n今回は点数を絞り、複勝中心で小さく入る方針です。",
            "bet_tendency_ja": "買い目傾向：複勝中心",
        }
    if buy_style == "win_focus":
        return {
            "strategy_text_ja": "上位の軸は比較的見えており、通常通り参加できるレースと見ています。\n単勝を少し前に置きつつ、複勝で土台を作る組み立てが自然です。",
            "bet_tendency_ja": "買い目傾向：単勝を少し厚め＋複勝で保険",
        }
    if strategy_mode in ("pair_focus", "spread"):
        return {
            "strategy_text_ja": "上位は混戦寄りですが、見送りではなく組み合わせの妙味で参加できると見ています。\n複勝で土台を残しつつ、ワイド・馬連を少点数で組む形が自然です。",
            "bet_tendency_ja": "買い目傾向：複勝・ワイド中心",
        }
    if has_longshot:
        return {
            "strategy_text_ja": "強気に広げる局面ではありませんが、見送りよりは軽く参加できると判断しました。\n本線は保守的に置きつつ、妙味のある高配当は1点だけ補助で狙います。",
            "bet_tendency_ja": "買い目傾向：複勝中心＋高配当は1点だけ",
        }
    return {
        "strategy_text_ja": "上位の信頼は極端ではありませんが、完全に見送るほどでもありません。\n今回は複勝を主軸に、必要な相手券だけを添える無理のない形で参加します。",
        "bet_tendency_ja": "買い目傾向：複勝＋ワイドを少額で",
    }


def deterministic_policy(input_obj: RacePolicyInput, fallback_reason: str = "") -> RacePolicyOutput:
    ai = input_obj.ai
    constraints = input_obj.constraints
    allowed_types = {str(x).strip().lower() for x in list(constraints.allowed_types or []) if str(x).strip()}
    candidates = [
        c for c in input_obj.candidates if (not allowed_types) or (str(c.bet_type) in allowed_types)
    ]
    horses = _horse_pool(input_obj)
    predictions = list(input_obj.predictions or [])
    high_odds_threshold = float(constraints.high_odds_threshold or 10.0)
    has_value = any(float(c.ev) > 0.0 for c in candidates)
    has_pair_value = any(str(c.bet_type) in ("wide", "quinella") and float(c.ev) > 0.0 for c in candidates)
    longshot_candidates = [c for c in candidates if float(c.odds_used) >= high_odds_threshold and float(c.ev) > 0.0]
    top_key = str(predictions[0].horse_no) if predictions else (horses[0] if horses else "")
    second_key = str(predictions[1].horse_no) if len(predictions) >= 2 else ""
    third_key = str(predictions[2].horse_no) if len(predictions) >= 3 else ""

    if not candidates:
        text = _render_strategy_text("no_bet", "no_bet", "no_bet", "no_bet", False)
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
        text = _render_strategy_text("no_bet", "no_bet", "no_bet", "no_bet", False)
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

    if float(ai.gap) >= 0.08 and float(ai.confidence_score) >= 0.62:
        participation_level = "normal_bet"
        buy_style = "win_focus"
        strategy_mode = "win_focus"
        enabled = ["win", "place"]
        max_ticket_count = 2
    elif has_pair_value and float(ai.stability_score) >= 0.5 and int(input_obj.field_size or 0) >= 10:
        participation_level = "normal_bet"
        buy_style = "balanced"
        strategy_mode = "pair_focus"
        enabled = ["place", "wide", "quinella"]
        max_ticket_count = 3
    else:
        weak_conf = float(ai.gap) < 0.03 or float(ai.confidence_score) < 0.56 or float(ai.stability_score) < 0.48
        participation_level = "small_bet" if weak_conf else "normal_bet"
        if participation_level == "small_bet":
            buy_style = "place_only" if (not has_pair_value or float(ai.stability_score) < 0.45) else "conservative"
            strategy_mode = "small_probe" if buy_style == "conservative" else "place_only"
            enabled = ["place"] if buy_style == "place_only" else ["place", "wide"]
            max_ticket_count = 1 if buy_style == "place_only" else 2
        else:
            buy_style = "place_focus" if float(ai.confidence_score) < 0.60 else "balanced"
            strategy_mode = "place_focus" if buy_style == "place_focus" else "balanced"
            enabled = ["place", "wide"]
            max_ticket_count = 3
        if buy_style == "balanced" and float(ai.confidence_score) >= 0.62:
            enabled.insert(0, "win")

    enabled = [x for x in enabled if ((not allowed_types) or (x in allowed_types))]
    if not enabled:
        enabled = [str(candidates[0].bet_type)] if candidates else []
    risk_tilt = "low" if participation_level == "small_bet" else ("low" if float(ai.stability_score) < 0.5 else "medium")

    key_horses = [top_key] if top_key else []
    secondary_horses = [x for x in [second_key, third_key] if x and x != top_key]
    longshot_horses: List[str] = []
    if longshot_candidates and participation_level != "no_bet":
        strategy_mode = "small_probe" if participation_level == "small_bet" else strategy_mode
        longshot_horses = [str(longshot_candidates[0].legs[0])] if list(longshot_candidates[0].legs or []) else []
        risk_tilt = "medium"

    construction_style = _derive_construction_style(strategy_mode, buy_style, participation_level)
    text = _render_strategy_text("bet", participation_level, buy_style, strategy_mode, bool(longshot_horses))
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
        "- あなたは購入金額を決める役割ではありません。\n"
        "- 予算の大小に応じた金額配分はローカル実行器が担当します。\n"
        "- あなたは、入力データだけをもとに、このレースで最も合理的な購入方針・券種構成・注目対象を決めてください。\n"
        "- 予算が小さいか大きいかに関わらず、まず「どう買うべきか」を決めてください。\n"
        "- 「いくら買うか」は考えなくて構いません。\n\n"
    )
    full_data_text = (
        "【入力データの読み方】\n"
        "- predictions は要約版、predictions_full は全馬・全列の予測テーブルです。\n"
        "- multi_predictor には v1-v4 全 predictor の要約・設計上の特徴・共識表が入っています。\n"
        "- multi_predictor.profiles は各 predictor の設計上の強みです。絶対評価ではなく、視点の違いとして扱ってください。\n"
        "- multi_predictor.summaries は predictor ごとの上位馬一覧です。\n"
        "- multi_predictor.consensus は馬番単位で揃えた共識表です。top1_votes / top3_votes / avg_pred_rank を優先的に見てください。\n"
        "- odds_full には win/place/wide/quinella の全量オッズが入っています。\n"
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
        "- v3 は市場融合・説明性寄りの視点です。値頃感や保守的な買い方の妥当性確認に向いています。\n"
        "- v4 は文脈適性ハイブリッド型です。Top3確率の分類と順位付けを混合し、コース・距離・馬場条件への適合を強く見ています。\n"
        "- 4 路の top1/top3 の共識が強い馬は軸候補です。ただし、オッズが過度に安く妙味がないなら買い方は保守化してください。\n"
        "- 4 路の見解差が大きい場合は、単勝偏重を避け、複勝・ワイド中心、または small_bet / conservative を優先してください。\n"
        "- 1 路だけが強く推す穴馬は、そのまま採用しないでください。オッズの裏付け、他 predictor の否定度、candidates の EV を合わせて判断してください。\n"
        "- 最終判断は『4 predictor の共識/見解差』と『現在のオッズ』の両方が必要です。どちらか片方だけで決めてはいけません。\n\n"
    )
    return (
        "あなたは「競馬AIの購入戦略責任者」です。\n\n"
        "あなたの役割は、入力として与えられる予測情報・オッズ・AI評価情報をもとに、"
        "このレースに対して最も合理的な購入方針を決めることです。\n\n"
        "あなたは単なるリスク管理官ではありません。"
        "あなたの仕事は「すぐ見送ること」ではなく、"
        "買わないべきレースは見送りつつ、買えるレースでは無理のない形で参加することです。\n\n"
        "--------------------------------\n"
        "【あなたの基本姿勢】\n"
        "--------------------------------\n"
        "あなたは次の3段階で判断してください。\n"
        "1. 明確に見送るべきレース\n"
        "2. 小さく保守的に参加すべきレース\n"
        "3. 通常通り参加できるレース\n\n"
        "重要：信頼度が低い = 直ちに no_bet ではありません。\n"
        "信頼度が低くても、点数を絞る、券種を保守化する、複勝中心にすることで参加できる場合があります。\n"
        "つまり、「見送る」か「軽く入る」かをきちんと分けてください。\n\n"
        "--------------------------------\n"
        "【絶対ルール】\n"
        "--------------------------------\n"
        "- 出力は JSON のみ\n"
        "- 入力に存在しない馬・組み合わせ・券種を創作しない\n"
        "- 無理に多点買いしない\n"
        "- 高オッズ狙いは補助的な扱いに留める\n"
        "- 混戦時に単勝へ寄りすぎない\n"
        "- 自信が弱いレースでは「小さく入る」という選択肢を優先して検討する\n"
        "- no_bet は最終手段であり、軽率に選ばない\n"
        "- enabled_bet_types は candidates に存在する bet_type のみ使用可能\n"
        "- focus_points に入れる horse / pair も入力に存在するもののみ使用可能\n"
        "- pick_ids は任意。使う場合も candidates[].id のみ使用可能\n\n"
        "--------------------------------\n"
        "【最初に考えること】\n"
        "--------------------------------\n"
        "まず次を判断してください：\n"
        "このレースは A. 完全に見送るべきか B. 軽く保守的に入るべきか C. 通常通り参加できるか。\n\n"
        "次のような場合だけ no_bet を選んでください：\n"
        "- 候補の質が全体的に低い\n"
        "- EV/score/p_hit のどれを見ても買う根拠が薄い\n"
        "- 高オッズに依存しないと成立しない\n"
        "- 券種を広げても優位性が作れない\n"
        "- 軽く保守的に参加する形すら不自然\n\n"
        "それ以外では、まず「小さく保守的に参加できないか」を優先して検討してください。\n\n"
        "--------------------------------\n"
        "【判断の優先順位】\n"
        "--------------------------------\n"
        "1. レースの性格\n"
        "入力の gap / confidence_score / stability_score / risk_score を見て、レースを混戦 / 通常 / 一本調子 / 見送り寄りのどれかに分類してください。\n\n"
        "2. 参加レベル\n"
        "participation_level は no_bet / small_bet / normal_bet です。\n"
        "- 明確な優位性が薄いが、複勝や保守的な組み方なら参加できる -> small_bet\n"
        "- ある程度自然に買える -> normal_bet\n"
        "- それすら難しい -> no_bet\n\n"
        "3. buy_style\n"
        "buy_style は no_bet / place_only / place_focus / balanced / win_focus / pair_focus / conservative から最も近いものを選んでください。\n"
        "重要：自信が弱いときは no_bet より先に place_only / conservative / place_focus を検討してください。\n\n"
        "4. strategy_mode\n"
        "strategy_mode は no_bet / place_only / place_focus / balanced / win_focus / pair_focus / spread / conservative_single / small_probe から選んでください。\n\n"
        "focus_points は固定の型に縛られず、そのレースで重要だと思う対象だけを出してください。\n"
        "type は horse / pair / bet_type / concept です。\n\n"
        "enabled_bet_types の考え方：\n"
        "- 混戦 / 低信頼 / 低安定 -> place を優先\n"
        "- 保守参加 -> place_only または place + wide\n"
        "- 通常 -> place + wide、状況次第で win\n"
        "- 一本調子 -> balanced か win_focus\n"
        "- pair は相手関係がはっきりしている時に限る\n"
        "- 券種を広げないと成立しない場合は、むしろ絞るか見送る\n\n"
        "max_ticket_count の考え方：\n"
        "- no_bet -> 0\n"
        "- small_bet / conservative_single / place_only -> 1〜2\n"
        "- place_focus / balanced -> 2〜3\n"
        "- spread / pair_focus -> 3〜4（本当に理由がある時だけ）\n\n"
        "risk_tilt の考え方：\n"
        "- small_bet / place_only / conservative_single -> low\n"
        "- balanced -> medium\n"
        "- win_focus / spread -> medium〜high\n"
        "- no_bet -> low\n\n"
        "strategy_text_ja は2〜4文で、自然な日本語にしてください。\n"
        "必ず、レースの見立て、参加レベル、主軸券種、見送りなら理由、small_bet なら見送りではなく軽く参加する理由を含めてください。\n"
        "bet_tendency_ja は1行のみで、たとえば「買い目傾向：複勝中心」「買い目傾向：複勝＋ワイドを少額で」「買い目傾向：見送り」のように書いてください。\n\n"
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
        '  "focus_points": [{"type": "horse | pair | bet_type | concept", "value": ""}],\n'
        '  "max_ticket_count": 0,\n'
        '  "risk_tilt": "low | medium | high",\n'
        '  "strategy_text_ja": "",\n'
        '  "bet_tendency_ja": "",\n'
        '  "reason_codes": [],\n'
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
        "temperature": 0.1,
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
            raise ValueError("pick_ids contains unknown id")
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

    participation_level = str(output.participation_level or "no_bet").strip().lower()
    bet_decision = str(output.bet_decision or "no_bet").strip().lower()
    buy_style = str(output.buy_style or "no_bet").strip().lower()
    strategy_mode = str(output.strategy_mode or "no_bet").strip().lower()
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
        pick_ids = []

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
            "warnings": [str(x) for x in list(output.warnings or []) if str(x).strip()] if output.warnings else None,
            "pick_ids": pick_ids,
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
    timeout_s: int = 20,
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
            start = time.perf_counter()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    fut = executor.submit(_call_gemini_once, prompt, model, api_key)
                    raw_text = fut.result(timeout=max(1, int(timeout_s or 1)))
                llm_latency_ms = int((time.perf_counter() - start) * 1000)
                payload = json.loads(raw_text)
                parsed = _model_validate(RacePolicyOutput, payload)
                output = _sanitize_output(parsed, input_obj)
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
                if "429" in text or "quota" in text or "rate" in text:
                    fallback_reason = "quota_or_429"
                elif "api key" in text or "permission" in text or "auth" in text:
                    fallback_reason = "auth_error"
                else:
                    fallback_reason = "network_or_sdk_error"

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

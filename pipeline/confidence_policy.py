import math
import os


HIGH_EVALUATION_THRESHOLD = 0.85
MEDIUM_EVALUATION_THRESHOLD = 0.70
MORNING_CONFIDENCE_VERY_HIGH_THRESHOLD = 0.90
MORNING_CONFIDENCE_LOW_THRESHOLD = 0.55

AGENT_CONFIDENCE_FALLBACK_SCORE = {
    "high": 0.88,
    "medium": 0.76,
    "low": 0.62,
}


def _safe_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def clamp01(value, default=0.0):
    number = _safe_float(value, default)
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def confidence_score_from_label(value, default=0.0):
    text = str(value or "").strip().lower()
    if text in AGENT_CONFIDENCE_FALLBACK_SCORE:
        return AGENT_CONFIDENCE_FALLBACK_SCORE[text]
    return clamp01(value, default)


def high_evaluation_threshold_from_env(environ=None):
    source = environ if environ is not None else os.environ
    raw = str(
        source.get("PIPELINE_AUTO_PREDICTION_NOTIFY_MIN_CONFIDENCE", "")
        or source.get("PIPELINE_HIGH_EVALUATION_NOTIFY_THRESHOLD", "")
        or ""
    ).strip()
    if not raw:
        return HIGH_EVALUATION_THRESHOLD
    value = clamp01(raw, HIGH_EVALUATION_THRESHOLD)
    if value <= 0:
        return HIGH_EVALUATION_THRESHOLD
    return value


def morning_confidence_label(score):
    value = clamp01(score)
    if value >= MORNING_CONFIDENCE_VERY_HIGH_THRESHOLD:
        return "かなり高い"
    if value >= HIGH_EVALUATION_THRESHOLD:
        return "高い"
    if value >= MEDIUM_EVALUATION_THRESHOLD:
        return "中"
    if value >= MORNING_CONFIDENCE_LOW_THRESHOLD:
        return "やや低い"
    return "低い"

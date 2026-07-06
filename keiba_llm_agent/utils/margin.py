from __future__ import annotations

import re
import unicodedata


SYMBOLIC_MARGIN_VALUES = {
    "同着": 0.0,
    "ハナ": 0.05,
    "鼻": 0.05,
    "アタマ": 0.1,
    "頭": 0.1,
    "クビ": 0.2,
    "首": 0.2,
    "大": 3.0,
    "大差": 4.0,
}


def _normalize_margin_text(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return " ".join(text.replace("\xa0", " ").split())


def _symbolic_margin_value(text: str) -> float | None:
    if text in SYMBOLIC_MARGIN_VALUES:
        return SYMBOLIC_MARGIN_VALUES[text]
    if text.endswith("差") and text[:-1] in SYMBOLIC_MARGIN_VALUES:
        return SYMBOLIC_MARGIN_VALUES[text[:-1]]
    return None


def _safe_fraction(numerator: str, denominator: str) -> float | None:
    try:
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    except (TypeError, ValueError):
        return None


def parse_margin_value(value: str | None) -> float | None:
    text = _normalize_margin_text(value)
    if not text:
        return None

    symbolic = _symbolic_margin_value(text)
    if symbolic is not None:
        return symbolic

    mixed_fraction = re.search(r"([+-]?\d+)(?:\s+|\.)(\d+)\s*/\s*(\d+)", text)
    if mixed_fraction:
        whole, numerator, denominator = mixed_fraction.groups()
        fraction = _safe_fraction(numerator, denominator)
        if fraction is not None:
            sign = -1.0 if whole.startswith("-") else 1.0
            return abs(float(whole) + sign * fraction)

    simple_fraction = re.search(r"([+-]?\d+)\s*/\s*(\d+)", text)
    if simple_fraction:
        fraction = _safe_fraction(*simple_fraction.groups())
        if fraction is not None:
            return abs(fraction)

    decimal = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    if not decimal:
        return None
    try:
        return abs(float(decimal.group(0)))
    except ValueError:
        return None

"""Parse explicit public performance fields without inferring missing values."""
from __future__ import annotations

import re
import unicodedata


def normalized(value: str | None) -> str:
    return unicodedata.normalize('NFKC', value or '').strip()


def parse_time_seconds(value: str | None) -> float | None:
    text = normalized(value)
    match = re.fullmatch(r'(?:(\d+):)?(\d{1,2}(?:\.\d+)?)', text)
    if not match:
        return None
    minutes, seconds = match.groups()
    seconds = float(seconds)
    if minutes and seconds >= 60:
        return None
    total = int(minutes or 0) * 60 + seconds
    return total if 10 <= total <= 1800 else None


def parse_body_weight(value: str | None) -> tuple[int | None, int | None]:
    match = re.fullmatch(r'(\d{3})\s*(?:\(\s*([+-]?\d+)\s*\))?', normalized(value))
    if not match:
        return None, None
    weight = int(match[1])
    change = int(match[2]) if match[2] is not None else None
    if not 250 <= weight <= 800 or (change is not None and abs(change) > 100):
        return None, None
    return weight, change


def parse_sex_age(value: str | None) -> tuple[str | None, int | None]:
    match = re.fullmatch(r'(牡|牝|セ|セン)\s*(\d{1,2})', normalized(value))
    if not match or not 1 <= int(match[2]) <= 30:
        return None, None
    return ('セ' if match[1] == 'セン' else match[1]), int(match[2])


def parse_pace(value: str | None) -> tuple[float | None, float | None]:
    match = re.fullmatch(r'(\d{2}(?:\.\d+)?)\s*-\s*(\d{2}(?:\.\d+)?)', normalized(value))
    if not match:
        return None, None
    first, last = map(float, match.groups())
    return (first, last) if 20 <= first <= 60 and 20 <= last <= 60 else (None, None)


def explicit_race_class(soup) -> tuple[str | None, str | None]:
    title = soup.select_one('.RaceName')
    if title is not None:
        for number in (1, 2, 3):
            if title.select_one(f'.Icon_GradeType{number}') is not None:
                return f'G{number}', 'RaceName.grade_icon'
    nodes = soup.select('.RaceData02')
    text = normalized(' '.join(n.get_text(' ', strip=True) for n in nodes))
    for label in ('3勝クラス', '2勝クラス', '1勝クラス', '未勝利', '新馬', 'リステッド', 'オープン'):
        if label in text:
            return label, 'RaceData02'
    return None, None

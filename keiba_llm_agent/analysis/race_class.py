from __future__ import annotations

from keiba_llm_agent.schemas.race_data import RaceInfo, RecentRun


CENTRAL_COURSES = {"札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"}
TARGET_RACE_NAME_TOKENS = ("S", "ステークス", "C", "カップ", "記念", "賞", "杯", "特別")


def normalize_race_text(value: str | None) -> str:
    text = str(value or "").strip().upper()
    replacements = {
        "Ｇ": "G",
        "Ｊ": "J",
        "Ｐ": "P",
        "Ｎ": "N",
        "Ｏ": "O",
        "Ｌ": "L",
        "Ｓ": "S",
        "Ｃ": "C",
        "Ⅰ": "I",
        "Ⅱ": "II",
        "Ⅲ": "III",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "０": "0",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def infer_race_class_level(race_name: str | None) -> float | None:
    text = normalize_race_text(race_name)
    if not text:
        return None

    if any(token in text for token in ("GIII", "JPNGIII", "JPNIII", "G3", "JPN3")):
        return 7.0
    if any(token in text for token in ("GII", "JPNGII", "JPNII", "G2", "JPN2")):
        return 8.0
    if any(token in text for token in ("GI", "JPNGI", "JPNI", "G1", "JPN1")):
        return 9.0
    if "(L)" in text or "リステッド" in text or "LISTED" in text:
        return 6.0
    if "OP" in text or "オープン" in text:
        return 5.5
    if "3勝" in text or "３勝" in text or "1600万" in text or "1600万円" in text:
        return 4.5
    if "2勝" in text or "２勝" in text or "1000万" in text or "1000万円" in text:
        return 3.5
    if "1勝" in text or "１勝" in text or "500万" in text or "500万円" in text:
        return 2.5
    if "未勝利" in text:
        return 1.5
    if "新馬" in text or "メイクデビュー" in text or "フレッシュチャレンジ" in text:
        return 1.0
    if "重賞" in text:
        return 7.0
    return None


def infer_target_race_class_level(race_info: RaceInfo | None) -> float | None:
    if race_info is None:
        return None
    explicit_level = infer_race_class_level(race_info.race_name)
    if explicit_level is not None:
        return explicit_level

    race_name = normalize_race_text(race_info.race_name)
    is_central = race_info.scope_key == "central" or race_info.course in CENTRAL_COURSES
    if is_central and race_name and any(token in race_name for token in TARGET_RACE_NAME_TOKENS):
        race_no = _infer_race_number(race_info.race_id)
        if race_no == 11:
            return 7.0
        return 6.0
    return None


def infer_run_race_class_level(run: RecentRun) -> float | None:
    return infer_race_class_level(run.race_name)


def _infer_race_number(race_id: str | None) -> int | None:
    text = str(race_id or "")
    if len(text) < 2 or not text[-2:].isdigit():
        return None
    return int(text[-2:])

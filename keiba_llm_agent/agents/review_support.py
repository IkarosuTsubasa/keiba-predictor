from __future__ import annotations

from typing import Any

from keiba_llm_agent.schemas.prediction import Prediction


PAYOUT_MISSING_WARNING = "Bet hit but payout data missing. ROI is unreliable."
PAYOUT_MISSING_NOTE_JA = "払戻データが不足しているためROIは暫定です。"

UNORDERED_BET_TYPES = {"wide", "quinella", "trio", "wakuren"}
ORDERED_BET_TYPES = {"exacta", "trifecta"}


def normalize_bet_type(value: object) -> str:
    mapping = {
        "ワイド": "wide",
        "wide": "wide",
        "複勝": "place",
        "place": "place",
        "単勝": "win",
        "win": "win",
        "馬連": "quinella",
        "quinella": "quinella",
        "馬単": "exacta",
        "exacta": "exacta",
        "三連複": "trio",
        "3連複": "trio",
        "trio": "trio",
        "三連単": "trifecta",
        "3連単": "trifecta",
        "trifecta": "trifecta",
        "枠連": "wakuren",
        "wakuren": "wakuren",
    }
    return mapping.get(str(value), str(value))


def _canonical_numbers(bet_type: str, horse_numbers: list[int]) -> list[int]:
    if bet_type in UNORDERED_BET_TYPES:
        return sorted(horse_numbers)
    return horse_numbers


def canonicalize_combination(bet_type: str, horse_numbers: list[int]) -> str:
    numbers = _canonical_numbers(bet_type, horse_numbers)
    return "-".join(str(number) for number in numbers)


def _parse_combination_text(combination_text: str) -> list[int]:
    parts = []
    for token in combination_text.replace("/", "-").split("-"):
        token = token.strip()
        if not token:
            continue
        try:
            parts.append(int(token))
        except ValueError:
            continue
    return parts


def extract_top3(result: dict[str, Any]) -> list[int]:
    if result.get("finish_order"):
        finish_order = sorted(
            result.get("finish_order", []),
            key=lambda item: item.get("finish", 999),
        )
        top3 = [item.get("horse_no") for item in finish_order[:3] if item.get("horse_no") is not None]
        if len(top3) >= 3:
            return [int(number) for number in top3[:3]]
    result_top3 = result.get("result", {})
    top3 = []
    for key in ("1st", "2nd", "3rd"):
        number = result_top3.get(key)
        if number is not None:
            top3.append(int(number))
    return top3[:3]


def extract_finish_map(result: dict[str, Any]) -> dict[int, int]:
    finish_map: dict[int, int] = {}
    if result.get("finish_order"):
        for item in result.get("finish_order", []):
            horse_no = item.get("horse_no")
            finish = item.get("finish")
            if horse_no is None or finish is None:
                continue
            finish_map[int(horse_no)] = int(finish)
        return finish_map

    top3 = extract_top3(result)
    for index, horse_no in enumerate(top3, start=1):
        finish_map[int(horse_no)] = index
    return finish_map


def extract_payout_lookup(result: dict[str, Any]) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for payout in result.get("payouts", []):
        bet_type = normalize_bet_type(payout.get("bet_type") or payout.get("type"))
        if payout.get("horse_numbers"):
            try:
                horse_numbers = [int(number) for number in payout.get("horse_numbers", [])]
            except Exception:
                horse_numbers = []
        else:
            horse_numbers = _parse_combination_text(str(payout.get("combination", "")))
        if not horse_numbers:
            continue
        combination = canonicalize_combination(bet_type, horse_numbers)
        payout_value = int(payout.get("payout") or 0)
        lookup[(bet_type, combination)] = payout_value
    return lookup


def _bet_hit(
    bet_type: str,
    horse_numbers: list[int],
    top3: list[int],
    first: int | None,
    second: int | None,
    third: int | None,
) -> bool:
    if bet_type == "wide" and len(horse_numbers) == 2:
        return all(number in top3 for number in horse_numbers)
    if bet_type == "place" and len(horse_numbers) == 1:
        return horse_numbers[0] in top3
    if bet_type == "win" and len(horse_numbers) == 1:
        return first is not None and horse_numbers[0] == first
    if bet_type == "quinella" and len(horse_numbers) == 2:
        return first is not None and second is not None and set(horse_numbers) == {first, second}
    if bet_type == "exacta" and len(horse_numbers) == 2:
        return first is not None and second is not None and horse_numbers == [first, second]
    if bet_type == "trio" and len(horse_numbers) == 3:
        return len(top3) == 3 and set(horse_numbers) == set(top3)
    if bet_type == "trifecta" and len(horse_numbers) == 3:
        return first is not None and second is not None and third is not None and horse_numbers == [first, second, third]
    return False


def calculate_review_metrics(prediction: Prediction, result: dict[str, Any]) -> dict[str, Any]:
    top3 = extract_top3(result)
    first = top3[0] if len(top3) >= 1 else None
    second = top3[1] if len(top3) >= 2 else None
    third = top3[2] if len(top3) >= 3 else None
    payout_lookup = extract_payout_lookup(result)

    total_stake = 0
    total_return = 0
    any_hit = False
    payout_warning = False
    review_warnings: list[str] = []
    bet_results: list[dict[str, Any]] = []

    for bet in prediction.bets:
        bet_type = normalize_bet_type(bet.bet_type)
        horse_numbers = [int(number) for number in bet.horse_numbers]
        amount = int(bet.amount or 0)
        total_stake += amount

        hit = _bet_hit(bet_type, horse_numbers, top3, first, second, third)
        payout = 0
        return_amount = 0
        if hit:
            any_hit = True
            combination = canonicalize_combination(bet_type, horse_numbers)
            payout = payout_lookup.get((bet_type, combination), 0)
            if payout > 0:
                return_amount = int((amount / 100) * payout) if amount > 0 else 0
                total_return += return_amount
            else:
                payout_warning = True

        bet_results.append(
            {
                "bet_type": bet.bet_type,
                "horse_numbers": horse_numbers,
                "amount": amount,
                "hit": hit,
                "payout": payout,
                "return_amount": return_amount,
            }
        )

    if payout_warning:
        review_warnings.append(PAYOUT_MISSING_WARNING)

    marks = prediction.marks or {}
    marked_numbers = [number for number in marks.values() if isinstance(number, int) and number > 0]
    marked_top3_count = len(set(marked_numbers) & set(top3))
    main_mark = marks.get("◎", 0)
    main_mark_top3 = bool(main_mark in top3)
    roi = round((total_return / total_stake), 2) if total_stake > 0 else 0.0

    return {
        "top3": top3,
        "bet_results": bet_results,
        "total_stake": total_stake,
        "total_return": total_return,
        "bet_hit": any_hit,
        "payout_warning": payout_warning,
        "review_warnings": review_warnings,
        "main_mark_top3": main_mark_top3,
        "marked_horses_top3_count": marked_top3_count,
        "roi": roi,
    }

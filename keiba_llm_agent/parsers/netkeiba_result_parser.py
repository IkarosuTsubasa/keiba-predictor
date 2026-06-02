from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, Tag

from keiba_llm_agent.schemas.result import FinishOrderItem, PayoutItem, ResultData


BET_TYPE_LABELS = {
    "単勝": "単勝",
    "複勝": "複勝",
    "枠連": "枠連",
    "馬連": "馬連",
    "ワイド": "ワイド",
    "馬単": "馬単",
    "3連複": "三連複",
    "三連複": "三連複",
    "3連単": "三連単",
    "三連単": "三連単",
}


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.replace("\xa0", " ").split())


def parse_int_safe(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^\d]", "", value)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_float_safe(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^\d.]", "", value)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_race_id_from_result_html(soup: BeautifulSoup) -> str | None:
    for selector in ("link[rel='canonical']", "meta[property='og:url']", "a[href*='race_id=']"):
        for node in soup.select(selector):
            candidate = node.get("href") or node.get("content")
            if not candidate:
                continue
            match = re.search(r"race_id=(\d+)", candidate)
            if match:
                return match.group(1)
    return None


def _find_finish_table(soup: BeautifulSoup) -> Tag | None:
    for table in soup.find_all("table"):
        headers = [normalize_text(th.get_text(" ", strip=True)).replace(" ", "") for th in table.find_all("th")]
        header_text = "".join(headers)
        class_text = " ".join(table.get("class", []))
        if "着順" not in header_text or "馬番" not in header_text or "馬名" not in header_text:
            continue
        if any(token in class_text for token in ("RaceTable", "Result", "Table_Show_All", "RaceCommon_Table")):
            return table
    return None


def _first_non_empty_text(nodes: Iterable[Tag | None]) -> str | None:
    for node in nodes:
        if node is None:
            continue
        text = normalize_text(node.get_text(" ", strip=True))
        if text:
            return text
    return None


def parse_finish_order(soup: BeautifulSoup) -> list[FinishOrderItem]:
    table = _find_finish_table(soup)
    if table is None:
        return []

    finish_order: list[FinishOrderItem] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 4:
            continue

        finish_text = _first_non_empty_text((row.select_one(".Result_Num .Rank"), cells[0]))
        finish = parse_int_safe(finish_text)
        if finish is None:
            continue

        horse_no_text = _first_non_empty_text((row.select_one("td.Num.Txt_C div"), cells[2]))
        horse_no = parse_int_safe(horse_no_text)
        if horse_no is None:
            continue

        horse_name = _first_non_empty_text(
            (
                row.select_one(".HorseNameSpan"),
                row.select_one(".Horse_Name"),
                row.select_one(".Horse_Info"),
                cells[3],
            )
        )
        if not horse_name:
            continue

        jockey = _first_non_empty_text((row.select_one(".JockeyNameSpan"), row.select_one("td.Jockey")))

        time_cells = row.select("td.Time")
        time_text = _first_non_empty_text((time_cells[0],)) if len(time_cells) >= 1 else None
        margin_text = _first_non_empty_text((time_cells[1],)) if len(time_cells) >= 2 else None

        popularity = parse_int_safe(_first_non_empty_text((row.select_one(".OddsPeople"),)))
        odds = parse_float_safe(_first_non_empty_text((row.select_one(".Odds_Ninki"),)))

        finish_order.append(
            FinishOrderItem(
                finish=finish,
                horse_no=horse_no,
                horse_name=horse_name,
                jockey=jockey,
                time=time_text,
                margin=margin_text,
                popularity=popularity,
                odds=odds,
            )
        )

    finish_order.sort(key=lambda item: item.finish)
    return finish_order


def _normalize_bet_type_label(label: str) -> str | None:
    normalized = normalize_text(label).replace(" ", "")
    return BET_TYPE_LABELS.get(normalized)


def _extract_combinations_from_result_cell(cell: Tag, bet_type: str) -> list[str]:
    combinations: list[str] = []

    grouped_lists = cell.find_all("ul", recursive=False)
    if grouped_lists:
        for grouped in grouped_lists:
            direct_lis = grouped.find_all("li", recursive=False)
            nodes = direct_lis if direct_lis else grouped.find_all("span", recursive=False)
            numbers = [normalize_text(node.get_text(" ", strip=True)) for node in nodes]
            cleaned_numbers = [value for value in numbers if parse_int_safe(value) is not None]
            if cleaned_numbers:
                combinations.append("-".join(cleaned_numbers))
        if combinations:
            return combinations

    grouped_divs = cell.find_all("div", recursive=False)
    if grouped_divs:
        for grouped in grouped_divs:
            direct_spans = grouped.find_all("span", recursive=False)
            nodes = direct_spans if direct_spans else grouped.find_all("li", recursive=False)
            values = [normalize_text(node.get_text(" ", strip=True)) for node in nodes]
            cleaned_values = [value for value in values if parse_int_safe(value) is not None]
            if not cleaned_values:
                continue
            if bet_type == "複勝":
                combinations.extend(cleaned_values)
            else:
                combinations.append("-".join(cleaned_values))
        if combinations:
            return combinations

    direct_numbers = [match.group(0) for match in re.finditer(r"\d+", cell.get_text(" ", strip=True))]
    if direct_numbers:
        if bet_type == "複勝":
            return direct_numbers
        return ["-".join(direct_numbers)]
    return []


def _extract_int_series(cell: Tag) -> list[int]:
    values: list[int] = []
    for text in cell.stripped_strings:
        parsed = parse_int_safe(text)
        if parsed is not None:
            values.append(parsed)
    return values


def parse_payouts(soup: BeautifulSoup) -> tuple[list[PayoutItem], list[str]]:
    payouts: list[PayoutItem] = []
    warnings: list[str] = []
    payout_tables = [table for table in soup.find_all("table") if "Payout_Detail_Table" in " ".join(table.get("class", []))]
    if not payout_tables:
        for table in soup.find_all("table"):
            if any(_normalize_bet_type_label(th.get_text(" ", strip=True)) for th in table.find_all("th")):
                payout_tables.append(table)
    if not payout_tables:
        return payouts, warnings

    for table in payout_tables:
        for row in table.find_all("tr"):
            header_cell = row.find("th")
            if header_cell is None:
                continue
            bet_type = _normalize_bet_type_label(header_cell.get_text(" ", strip=True))
            if bet_type is None:
                continue

            result_cell = row.find("td", class_="Result")
            payout_cell = row.find("td", class_="Payout")
            popularity_cell = row.find("td", class_="Ninki")
            if result_cell is None or payout_cell is None:
                cells = row.find_all("td", recursive=False)
                if len(cells) >= 2:
                    result_cell = cells[0]
                    payout_cell = cells[1]
            if result_cell is None or payout_cell is None:
                continue

            combinations = _extract_combinations_from_result_cell(result_cell, bet_type)
            payout_values = _extract_int_series(payout_cell)
            popularity_values = _extract_int_series(popularity_cell) if popularity_cell is not None else []

            if not combinations or not payout_values:
                warnings.append(f"{bet_type} の払戻解析に失敗しました。")
                continue

            if len(combinations) != len(payout_values):
                warnings.append(
                    f"{bet_type} の組番数({len(combinations)})と払戻数({len(payout_values)})が一致しません。"
                )

            pair_count = min(len(combinations), len(payout_values))
            for index in range(pair_count):
                payouts.append(
                    PayoutItem(
                        bet_type=bet_type,
                        combination=combinations[index],
                        payout=payout_values[index],
                        popularity=popularity_values[index] if index < len(popularity_values) else None,
                    )
                )

    if payout_tables and not payouts:
        warnings.append("払戻テーブルは存在しますが、払戻情報を解析できませんでした。")
    return payouts, warnings


def _build_top3_from_finish_order(finish_order: list[FinishOrderItem]) -> dict[str, int]:
    top3 = finish_order[:3]
    if len(top3) < 3:
        raise ValueError("failed to parse top3 result from netkeiba result HTML")
    return {
        "1st": top3[0].horse_no,
        "2nd": top3[1].horse_no,
        "3rd": top3[2].horse_no,
    }


def parse_netkeiba_result_html(html: str, race_id: str | None = None) -> ResultData:
    soup = BeautifulSoup(html, "html.parser")
    parsed_race_id = race_id or extract_race_id_from_result_html(soup)
    if parsed_race_id is None:
        raise ValueError("race_id not found in netkeiba result HTML")

    finish_order = parse_finish_order(soup)
    warnings: list[str] = []
    if not finish_order:
        warnings.append("着順テーブルを解析できませんでした。")
    top3_payload = _build_top3_from_finish_order(finish_order)

    payouts, payout_warnings = parse_payouts(soup)
    warnings.extend(payout_warnings)

    return ResultData.model_validate(
        {
            "race_id": parsed_race_id,
            "result": top3_payload,
            "payouts": [payout.model_dump() for payout in payouts],
            "finish_order": [item.model_dump() for item in finish_order],
            "warnings": warnings,
        }
    )

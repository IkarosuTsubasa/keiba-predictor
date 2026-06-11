from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from keiba_llm_agent.schemas.race_data import RecentRun


VENUE_NAMES = (
    "札幌",
    "函館",
    "福島",
    "新潟",
    "東京",
    "中山",
    "中京",
    "京都",
    "阪神",
    "小倉",
    "大井",
    "船橋",
    "川崎",
    "浦和",
    "門別",
    "盛岡",
    "水沢",
    "金沢",
    "笠松",
    "名古屋",
    "園田",
    "姫路",
    "高知",
    "佐賀",
)


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.replace("\xa0", " ").split())


def normalize_header_text(value: str | None) -> str:
    return normalize_text(value).replace(" ", "")


def parse_int_safe(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^\d-]", "", value)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_float_safe(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", value)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_corner_positions(value: str | None) -> list[int] | None:
    if not value:
        return None
    positions = [parse_int_safe(part) for part in re.split(r"[-→]", normalize_text(value))]
    filtered = [position for position in positions if position is not None]
    return filtered or None


def parse_surface_and_distance(text: str | None) -> tuple[str | None, int | None]:
    if not text:
        return None, None
    match = re.search(r"(芝|ダート|ダ|障害|障)\s*(\d+)", text)
    if not match:
        return None, None
    surface, distance = match.groups()
    surface_map = {"ダ": "ダート", "障": "障害"}
    return surface_map.get(surface, surface), int(distance)


def parse_course(text: str | None) -> str | None:
    if not text:
        return None
    for venue in VENUE_NAMES:
        if venue in text:
            return venue
    return None


def parse_date(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return normalize_text(text) or None


def parse_race_id_from_row(cells: list[Tag], header_index: dict[str, int]) -> str | None:
    race_name_index = header_index.get("レース名")
    if race_name_index is None or race_name_index >= len(cells):
        return None
    link = cells[race_name_index].find("a", href=re.compile(r"/race/"))
    if not link:
        return None
    href = link.get("href", "")
    match = re.search(r"/race/(\d+)/?", href)
    if match:
        return match.group(1)
    match = re.search(r"race_id=(\d+)", href)
    if match:
        return match.group(1)
    return None


def parse_results_table(soup: BeautifulSoup) -> Tag | None:
    candidates: list[Tag] = []
    for table in soup.find_all("table"):
        class_text = " ".join(table.get("class", []))
        if any(token in class_text for token in ("db_h_race_results", "nk_tb_common", "race_table_01")):
            candidates.append(table)
    candidates.extend(table for table in soup.find_all("table") if table not in candidates)

    for table in candidates:
        headers = [normalize_header_text(th.get_text(" ", strip=True)) for th in table.find_all("th")]
        if "日付" in headers and "着順" in headers and "距離" in headers:
            return table
    return None


def has_recent_runs_table(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    return parse_results_table(soup) is not None


def build_header_index(table: Tag) -> dict[str, int]:
    header_row = None
    for row in table.find_all("tr"):
        if row.find("th"):
            header_row = row
            break
    if header_row is None:
        return {}
    index: dict[str, int] = {}
    for idx, th in enumerate(header_row.find_all(["th", "td"], recursive=False)):
        name = normalize_header_text(th.get_text(" ", strip=True))
        if name:
            index[name] = idx
    return index


def get_cell_text(cells: list[Tag], header_index: dict[str, int], header_names: list[str]) -> str | None:
    for header_name in header_names:
        index = header_index.get(header_name)
        if index is None or index >= len(cells):
            continue
        text = normalize_text(cells[index].get_text(" ", strip=True))
        if text:
            return text
    return None


def is_valid_result_row(cells: list[Tag], header_index: dict[str, int]) -> bool:
    if not cells:
        return False
    date_text = get_cell_text(cells, header_index, ["日付"])
    finish_text = get_cell_text(cells, header_index, ["着順"])
    distance_text = get_cell_text(cells, header_index, ["距離"])
    return bool(date_text and finish_text and distance_text)


def parse_recent_run_row(cells: list[Tag], header_index: dict[str, int]) -> RecentRun:
    date_text = get_cell_text(cells, header_index, ["日付"])
    race_name_text = get_cell_text(cells, header_index, ["レース名"])
    course_text = get_cell_text(cells, header_index, ["開催"])
    distance_text = get_cell_text(cells, header_index, ["距離"])
    track_condition_text = get_cell_text(cells, header_index, ["馬場"])
    finish_text = get_cell_text(cells, header_index, ["着順"])
    field_size_text = get_cell_text(cells, header_index, ["頭数"])
    jockey_text = get_cell_text(cells, header_index, ["騎手"])
    odds_text = get_cell_text(cells, header_index, ["オッズ"])
    popularity_text = get_cell_text(cells, header_index, ["人気"])
    passing_order_text = get_cell_text(cells, header_index, ["通過", "通過順"])
    final_3f_text = get_cell_text(cells, header_index, ["上り", "上り3F", "上り3Fタイム"])
    margin_text = get_cell_text(cells, header_index, ["着差"])

    surface, distance = parse_surface_and_distance(distance_text)
    return RecentRun(
        race_id=parse_race_id_from_row(cells, header_index),
        date=parse_date(date_text),
        race_name=normalize_text(race_name_text) or None,
        course=parse_course(course_text),
        surface=surface,
        distance=distance,
        track_condition=normalize_text(track_condition_text) or None,
        finish=parse_int_safe(finish_text),
        field_size=parse_int_safe(field_size_text),
        jockey=normalize_text(jockey_text) or None,
        odds=parse_float_safe(odds_text),
        popularity=parse_int_safe(popularity_text),
        passing_order=normalize_text(passing_order_text) or None,
        corner_positions=parse_corner_positions(passing_order_text),
        final_3f=parse_float_safe(final_3f_text),
        margin=normalize_text(margin_text) or None,
    )


def parse_horse_recent_runs(html: str, limit: int | None = 5) -> list[RecentRun]:
    if limit is not None and limit <= 0:
        return []

    soup = BeautifulSoup(html, "html.parser")
    table = parse_results_table(soup)
    if table is None:
        return []

    header_index = build_header_index(table)
    if not header_index:
        return []

    runs: list[RecentRun] = []
    for row in table.find_all("tr"):
        if row.find("th"):
            continue
        cells = row.find_all("td", recursive=False)
        if not is_valid_result_row(cells, header_index):
            continue
        runs.append(parse_recent_run_row(cells, header_index))
        if limit is not None and len(runs) >= limit:
            break
    return runs

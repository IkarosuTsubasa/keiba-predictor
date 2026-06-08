from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from keiba_llm_agent.schemas.race_data import HorseEntry, RaceData, RaceInfo


CENTRAL_VENUE_MAP = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}
LOCAL_VENUE_MAP = {
    "30": "門別",
    "35": "盛岡",
    "36": "水沢",
    "42": "浦和",
    "43": "船橋",
    "44": "大井",
    "45": "川崎",
    "46": "金沢",
    "47": "笠松",
    "48": "名古屋",
    "50": "園田",
    "51": "姫路",
    "54": "高知",
    "55": "佐賀",
}
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
    "門別",
    "盛岡",
    "水沢",
    "浦和",
    "船橋",
    "大井",
    "川崎",
    "金沢",
    "笠松",
    "名古屋",
    "園田",
    "姫路",
    "高知",
    "佐賀",
)
LOCAL_VENUE_NAMES = set(LOCAL_VENUE_MAP.values())


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


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.replace("\xa0", " ").split())


def clean_header_text(value: str | None) -> str:
    return normalize_text(value).replace(" ", "").replace("\n", "")


def parse_horse_id_from_link(href: str | None) -> str | None:
    if not href:
        return None
    match = re.search(r"/horse/(\d+)/?", href)
    if match:
        return match.group(1)
    return None


def parse_race_date(text: str) -> str | None:
    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def parse_race_date_from_soup(soup: BeautifulSoup, race_id: str | None = None) -> str | None:
    for selector, attr in (
        ("meta[property='og:title']", "content"),
        ("meta[property='og:description']", "content"),
        ("meta[name='description']", "content"),
        ("title", None),
    ):
        nodes = soup.select(selector)
        for node in nodes:
            text = node.get(attr, "") if attr else node.get_text(" ", strip=True)
            parsed = parse_race_date(normalize_text(text))
            if parsed:
                return parsed

    refund_link = soup.select_one(".Refundlink a[href*='kaisai_date=']")
    if refund_link:
        href = refund_link.get("href", "")
        match = re.search(r"kaisai_date=(\d{4})(\d{2})(\d{2})", href)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"

    race_data_text = " ".join(
        normalize_text(node.get_text(" ", strip=True))
        for node in soup.select(".RaceData01, .RaceData02, .RaceName, .RaceList_Item02")
    )
    parsed = parse_race_date(race_data_text)
    if parsed:
        return parsed

    if race_id and len(race_id) >= 8:
        year = race_id[:4]
        for text in (
            normalize_text(node.get_text(" ", strip=True))
            for node in soup.select(".RaceData01, .RaceData02, .RaceName, .RaceList_Item02")
        ):
            match = re.search(r"(\d{1,2})/(\d{1,2})", text)
            if match:
                month, day = match.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def parse_surface_and_distance(text: str) -> tuple[str | None, int | None]:
    match = re.search(r"(芝|ダート|ダ|障害|障)\s*(\d+)m", text)
    if not match:
        return None, None
    surface, distance = match.groups()
    surface_map = {"ダ": "ダート", "障": "障害"}
    return surface_map.get(surface, surface), int(distance)


def parse_course(text: str) -> str | None:
    for venue in VENUE_NAMES:
        if venue in text:
            return venue
    return None


def infer_source_from_race_id(race_id: str | None) -> str | None:
    race_id_text = str(race_id or "").strip()
    if len(race_id_text) < 6:
        return None
    venue_code = race_id_text[4:6]
    if venue_code in CENTRAL_VENUE_MAP:
        return "central"
    if venue_code in LOCAL_VENUE_MAP:
        return "local"
    return None


def infer_scope_key(
    race_id: str | None,
    *,
    surface: str | None,
    course: str | None,
) -> str | None:
    source = infer_source_from_race_id(race_id)
    if source == "local" or course in LOCAL_VENUE_NAMES:
        return "local"
    if source != "central":
        return None
    if surface == "芝":
        return "central_turf"
    if surface and ("ダ" in surface or "砂" in surface):
        return "central_dirt"
    return None


def parse_weather(text: str) -> str | None:
    match = re.search(r"天候\s*[:：]\s*([^\s/]+)", text)
    return match.group(1) if match else None


def parse_track_condition(text: str) -> str | None:
    match = re.search(r"馬場\s*[:：]\s*([^\s/]+)", text)
    return match.group(1) if match else None


def select_first_text(row: Tag, selectors: list[str]) -> str | None:
    for selector in selectors:
        cell = row.select_one(selector)
        if cell:
            text = normalize_text(cell.get_text(" ", strip=True))
            if text:
                return text
    return None


def is_missing_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    normalized = normalize_text(value)
    return normalized in ("", "--", "---", "---.-", "**", "*")


def find_horse_tables(soup: BeautifulSoup) -> list[Tag]:
    tables: list[Tag] = []
    for table in soup.find_all("table"):
        class_names = table.get("class", [])
        class_text = " ".join(class_names)
        if not any(token in class_text for token in ("Shutuba", "RaceTable", "Shutuba_Table")):
            continue
        if "PredictRap_Table" in class_text:
            continue
        header_text = clean_header_text(" ".join(th.get_text(" ", strip=True) for th in table.find_all("th")))
        if table.select_one("th.HorseInfo, th.Horse_Name, th.Jockey") or (
            "馬名" in header_text and "騎手" in header_text
        ):
            tables.append(table)
    return tables


def find_horse_rows(table: Tag) -> list[Tag]:
    rows: list[Tag] = []
    for row in table.find_all("tr"):
        class_names = row.get("class", [])
        if any("HorseList" in class_name for class_name in class_names):
            rows.append(row)
    return rows


def build_header_index_map(table: Tag) -> dict[str, int]:
    header_map: dict[str, int] = {}
    header_row = table.select_one("thead tr.Header") or table.select_one("thead tr") or table.select_one("tr.Header")
    if header_row is None:
        return header_map
    index = 0
    for header in header_row.find_all("th", recursive=False):
        text = clean_header_text(header.get_text(" ", strip=True))
        colspan = parse_int_safe(header.get("colspan")) or 1
        if text:
            header_map[text] = index
        index += colspan
    return header_map


def parse_cell_by_index(row: Tag, header_map: dict[str, int], header_names: list[str]) -> str | None:
    if not header_map:
        return None
    cells = row.find_all("td", recursive=False)
    for header_name in header_names:
        cell_index = header_map.get(header_name)
        if cell_index is None or cell_index >= len(cells):
            continue
        text = normalize_text(cells[cell_index].get_text(" ", strip=True))
        if text:
            return text
    return None


def parse_odds_from_row(row: Tag, header_map: dict[str, int] | None = None) -> float | None:
    for selector in (
        "span[id^='odds-']",
        "td.Odds span",
        "td.Odds",
        "td[class*='Odds'] span",
        "td[class*='Odds']",
        "td.Txt_R.Popular span[id^='odds-']",
        "td.Txt_R.Popular span",
    ):
        node = row.select_one(selector)
        if node is None:
            continue
        text = normalize_text(node.get_text(" ", strip=True))
        if is_missing_placeholder(text):
            return None
        value = parse_float_safe(text)
        if value is not None:
            return value
    if header_map:
        text = parse_cell_by_index(row, header_map, ["オッズ", "予想オッズ"])
        if not is_missing_placeholder(text):
            return parse_float_safe(text)
    return None


def parse_popularity_from_row(row: Tag, header_map: dict[str, int] | None = None) -> int | None:
    for selector in (
        "span[id^='ninki-']",
        "td.Popular_Ninki span",
        "td[class*='Popular_Ninki'] span",
        "td.Ninki span",
        "td.Ninki",
        "td[class*='Ninki'] span",
        "td[class*='Ninki']",
    ):
        node = row.select_one(selector)
        if node is None:
            continue
        text = normalize_text(node.get_text(" ", strip=True))
        if is_missing_placeholder(text):
            return None
        value = parse_int_safe(text)
        if value is not None:
            return value
    if header_map:
        text = parse_cell_by_index(row, header_map, ["人気"])
        if not is_missing_placeholder(text):
            return parse_int_safe(text)
    return None


def parse_carried_weight_from_row(row: Tag) -> float | None:
    direct_value = parse_float_safe(
        select_first_text(
            row,
            [
                "td.Dredging",
                "td.CarriedWeight",
                "td.Weight",
                "td[class*='Dredging']",
                "td[class*='CarriedWeight']",
            ],
        )
    )
    if direct_value is not None and 45.0 <= direct_value <= 70.0:
        return direct_value

    cells = row.find_all("td", recursive=False)
    jockey_index = None
    for index, cell in enumerate(cells):
        class_names = cell.get("class", [])
        if any("Jockey" in class_name for class_name in class_names):
            jockey_index = index
            break
        if cell.find("a", href=re.compile(r"/jockey/")):
            jockey_index = index
            break

    if jockey_index is None:
        return None

    for index in range(jockey_index - 1, max(-1, jockey_index - 4), -1):
        if index < 0:
            continue
        text = normalize_text(cells[index].get_text(" ", strip=True))
        value = parse_float_safe(text)
        if value is not None and 45.0 <= value <= 70.0:
            return value
    return None


def parse_race_info(soup: BeautifulSoup, race_id: str | None = None) -> RaceInfo:
    race_name = None
    for selector in (".RaceName", "h1", "title"):
        node = soup.select_one(selector)
        if node:
            race_name = normalize_text(node.get_text(" ", strip=True))
            if race_name:
                break

    race_data_text_parts: list[str] = []
    for selector in (".RaceData01", ".RaceData02", ".RaceNum"):
        for node in soup.select(selector):
            text = normalize_text(node.get_text(" ", strip=True))
            if text:
                race_data_text_parts.append(text)
    race_data_text = " ".join(race_data_text_parts)

    parsed_race_id = race_id or extract_race_id_from_html(soup)
    if parsed_race_id is None:
        raise ValueError("race_id not found in shutuba HTML")

    surface, distance = parse_surface_and_distance(race_data_text)
    course = parse_course(race_data_text)
    source = infer_source_from_race_id(parsed_race_id)
    return RaceInfo(
        race_id=parsed_race_id,
        race_name=race_name,
        race_date=parse_race_date_from_soup(soup, parsed_race_id),
        course=course,
        surface=surface,
        distance=distance,
        track_condition=parse_track_condition(race_data_text),
        weather=parse_weather(race_data_text),
        source=source,
        scope_key=infer_scope_key(parsed_race_id, surface=surface, course=course),
    )


def extract_race_id_from_html(soup: BeautifulSoup) -> str | None:
    selectors = (
        "link[rel='canonical']",
        "meta[property='og:url']",
        "a[href*='race_id=']",
    )
    for selector in selectors:
        for node in soup.select(selector):
            candidate = node.get("href") or node.get("content")
            if not candidate:
                continue
            match = re.search(r"race_id=(\d+)", candidate)
            if match:
                return match.group(1)
    return None


def parse_horse_rows(soup: BeautifulSoup) -> list[HorseEntry]:
    rows: list[Tag] = []
    header_map: dict[str, int] = {}
    for table in find_horse_tables(soup):
        candidate_rows = find_horse_rows(table)
        if any(
            row.select_one(
                "td.HorseInfo a[href*='/horse/'], td.Horse_Name a[href*='/horse/'], span.HorseName a[href*='/horse/']"
            )
            for row in candidate_rows
        ):
            rows = candidate_rows
            header_map = build_header_index_map(table)
            break

    if not rows:
        rows = [
            row
            for row in soup.find_all("tr")
            if any("HorseList" in class_name for class_name in row.get("class", []))
        ]
        if rows:
            parent_table = rows[0].find_parent("table")
            if parent_table is not None:
                header_map = build_header_index_map(parent_table)

    horses: list[HorseEntry] = []
    for row in rows:
        horse_no = parse_int_safe(
            select_first_text(
                row,
                [
                    "td.Umaban",
                    "td.Horse_Num",
                    "td[class*='Umaban']",
                ],
            )
        )
        if horse_no is None:
            continue

        frame_no = parse_int_safe(
            select_first_text(
                row,
                [
                    "td.Waku",
                    "td.Frame",
                    "td[class*='Waku']",
                ],
            )
        )
        horse_link = row.select_one(
            "td.HorseInfo a[href*='/horse/'], td.Horse_Name a[href*='/horse/'], span.HorseName a[href*='/horse/']"
        )
        horse_name = normalize_text(horse_link.get_text(" ", strip=True) if horse_link else None)
        if not horse_name:
            horse_name = select_first_text(row, ["td.HorseInfo", "td.Horse_Name"]) or "unknown"

        jockey = None
        jockey_link = row.select_one("td.Jockey a, td.Jockey span, td[class*='Jockey'] a")
        if jockey_link:
            jockey = normalize_text(jockey_link.get_text(" ", strip=True))
        if not jockey:
            jockey = select_first_text(row, ["td.Jockey", "td[class*='Jockey']"]) or "unknown"

        carried_weight = parse_carried_weight_from_row(row)
        odds = parse_odds_from_row(row, header_map=header_map)
        popularity = parse_popularity_from_row(row, header_map=header_map)
        horses.append(
            HorseEntry(
                horse_no=horse_no,
                frame_no=frame_no,
                horse_id=parse_horse_id_from_link(horse_link.get("href") if horse_link else None),
                horse_name=horse_name,
                jockey=jockey,
                carried_weight=carried_weight,
                odds=odds,
                popularity=popularity,
                recent_runs=[],
            )
        )
    return horses


def parse_netkeiba_shutuba_html(html: str, race_id: str | None = None) -> RaceData:
    soup = BeautifulSoup(html, "html.parser")
    race_info = parse_race_info(soup, race_id=race_id)
    horses = parse_horse_rows(soup)
    return RaceData(race_info=race_info, horses=horses)

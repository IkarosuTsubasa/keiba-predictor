from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from keiba_llm_agent.schemas.pedigree import PedigreeInfo


LABEL_MAP = {
    "父": "sire",
    "母": "dam",
    "母父": "damsire",
}
ENTITY_KEYS = ("sire", "dam", "damsire", "sire_sire")


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.replace("\xa0", " ").split())


def _text_from_link_or_cell(node: Tag | None) -> str | None:
    if node is None:
        return None
    link = node.find("a")
    if link is not None:
        text = _normalize_text(link.get_text(" ", strip=True))
        if text:
            return text
    text = _normalize_text(node.get_text(" ", strip=True))
    return text or None


def _horse_id_from_href(href: str | None) -> str | None:
    if not href:
        return None
    match = re.search(r"/horse/(?:ped/|result/|[A-Za-z_]+/)?([0-9A-Za-z]+)/?$", href)
    if match:
        return match.group(1)
    return None


def _entity_from_link_or_cell(node: Tag | None) -> tuple[str | None, str | None]:
    if node is None:
        return None, None
    link = node.find("a")
    if link is not None:
        text = _normalize_text(link.get_text(" ", strip=True))
        horse_id = _horse_id_from_href(link.get("href"))
        if text or horse_id:
            return text or None, horse_id
    return _text_from_link_or_cell(node), None


def _class_tokens(node: Tag) -> list[str]:
    classes = node.get("class")
    if classes is None:
        return []
    if isinstance(classes, str):
        return classes.split()
    return [str(item) for item in classes]


def _find_blood_or_pedigree_table(soup: BeautifulSoup) -> Tag | None:
    for table in soup.find_all("table"):
        class_text = " ".join(_class_tokens(table)).lower()
        table_id = str(table.get("id") or "").lower()
        if any(token in class_text for token in ("blood", "pedigree")):
            return table
        if "blood" in table_id or "pedigree" in table_id:
            return table
    return None


def _empty_result() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for key in ENTITY_KEYS:
        result[key] = None
        result[f"{key}_id"] = None
    return result


def _set_entity(result: dict[str, str | None], key: str, node: Tag | None) -> None:
    name, horse_id = _entity_from_link_or_cell(node)
    if name:
        result[key] = name
    if horse_id:
        result[f"{key}_id"] = horse_id


def _parse_from_labeled_table(soup: BeautifulSoup) -> dict[str, str | None]:
    result = _empty_result()
    for row in soup.find_all("tr"):
        headers = row.find_all(["th", "td"], recursive=False)
        if len(headers) < 2:
            continue
        label = _normalize_text(headers[0].get_text(" ", strip=True))
        mapped = LABEL_MAP.get(label)
        if mapped is None:
            continue
        _set_entity(result, mapped, headers[1])
    return result


def extract_pedigree_url(html: str, horse_id: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    link = soup.find("a", href=re.compile(rf"/horse/ped/{re.escape(horse_id)}/?"))
    if link is None:
        link = soup.find("a", href=re.compile(r"/horse/ped/\d+/"))
    if link is None:
        return None
    href = link.get("href", "")
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return f"https://db.netkeiba.com{href}"
    return None


def _parse_from_blood_table(soup: BeautifulSoup) -> dict[str, str | None]:
    result = _empty_result()
    table = _find_blood_or_pedigree_table(soup)
    if table is None:
        return result

    rows = table.find_all("tr")
    top_level_cells: list[tuple[int, Tag]] = []
    for row_index, row in enumerate(rows):
        for cell in row.find_all("td", recursive=False):
            rowspan = str(cell.get("rowspan") or "")
            if rowspan == "16":
                top_level_cells.append((row_index, cell))

    if top_level_cells:
        sire_row_index, sire_cell = top_level_cells[0]
        _set_entity(result, "sire", sire_cell)
        sire_row_cells = rows[sire_row_index].find_all("td", recursive=False)
        if sire_cell in sire_row_cells:
            sire_position = sire_row_cells.index(sire_cell)
            subsequent_cells = sire_row_cells[sire_position + 1 :]
            sire_sire_cell = next(
                (
                    cell
                    for cell in subsequent_cells
                    if str(cell.get("rowspan") or "") in {"8", "16"}
                ),
                subsequent_cells[0] if subsequent_cells else None,
            )
            _set_entity(result, "sire_sire", sire_sire_cell)
    if len(top_level_cells) >= 2:
        _set_entity(result, "dam", top_level_cells[1][1])

    candidate_links: list[tuple[str, str | None]] = []
    for link in table.find_all("a", href=re.compile(r"/horse/")):
        text = _normalize_text(link.get_text(" ", strip=True))
        if text:
            candidate_links.append((text, _horse_id_from_href(link.get("href"))))
    unique_candidates: list[tuple[str, str | None]] = []
    seen_names: set[str] = set()
    for item in candidate_links:
        if item[0] not in seen_names:
            unique_candidates.append(item)
            seen_names.add(item[0])

    if result["sire"] is None and unique_candidates:
        result["sire"], result["sire_id"] = unique_candidates[0]

    if result["damsire"] is None and len(top_level_cells) >= 2:
        dam_row_index, dam_cell = top_level_cells[1]
        dam_row_cells = rows[dam_row_index].find_all("td", recursive=False)
        if dam_cell in dam_row_cells:
            dam_position = dam_row_cells.index(dam_cell)
            subsequent_cells = dam_row_cells[dam_position + 1 :]
            damsire_cell = next(
                (
                    cell
                    for cell in subsequent_cells
                    if str(cell.get("rowspan") or "") in {"8", "16"}
                ),
                subsequent_cells[0] if subsequent_cells else None,
            )
            _set_entity(result, "damsire", damsire_cell)

    if len(unique_candidates) >= 3:
        sire_name = result["sire"]
        dam_name = result["dam"]
        remaining = [item for item in unique_candidates if item[0] not in {sire_name, dam_name}]
        if result["dam"] is None:
            if remaining:
                result["dam"], result["dam_id"] = remaining[0]
        if result["damsire"] is None:
            if len(remaining) > 1:
                result["damsire"], result["damsire_id"] = remaining[1]
            elif remaining:
                result["damsire"], result["damsire_id"] = remaining[0]

    return result


def _parse_from_profile_table(soup: BeautifulSoup) -> dict[str, str | None]:
    result = _empty_result()
    tables = [table for table in soup.find_all("table") if "db_prof_table" in " ".join(_class_tokens(table))]
    for table in tables:
        parsed = _parse_from_labeled_table(BeautifulSoup(str(table), "html.parser"))
        for key in result:
            if result[key] is None and parsed.get(key):
                result[key] = parsed[key]
    return result


def _merge_non_null(*sources: dict[str, str | None]) -> dict[str, str | None]:
    merged = _empty_result()
    for key in merged:
        for source in sources:
            value = source.get(key)
            if value:
                merged[key] = value
                break
    return merged


def parse_pedigree_info(html: str, horse_id: str, horse_name: str | None = None) -> PedigreeInfo:
    soup = BeautifulSoup(html, "html.parser")
    labeled = _parse_from_labeled_table(soup)
    blood = _parse_from_blood_table(soup)
    profile = _parse_from_profile_table(soup)
    parsed = _merge_non_null(labeled, profile, blood)
    return PedigreeInfo(
        horse_id=horse_id,
        horse_name=horse_name,
        sire=parsed["sire"],
        sire_id=parsed["sire_id"],
        dam=parsed["dam"],
        dam_id=parsed["dam_id"],
        damsire=parsed["damsire"],
        damsire_id=parsed["damsire_id"],
        sire_sire=parsed["sire_sire"],
        sire_sire_id=parsed["sire_sire_id"],
    )

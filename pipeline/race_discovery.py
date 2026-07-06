import re
from datetime import datetime, timedelta
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from race_meta_fetcher import (
    LOCAL_VENUE_MAP,
    build_shutuba_url,
    fetch_html,
    normalize_race_id,
    parse_race_page,
)
from race_meta_fetcher import _decode_html_text


JST_OFFSET = timedelta(hours=9)
CENTRAL_TOP_URL = "https://race.netkeiba.com/top/"
CENTRAL_RACE_LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html"
LOCAL_TOP_URL = "https://nar.netkeiba.com/top/?rf=navi"
LOCAL_RACE_LIST_URL = "https://nar.netkeiba.com/top/race_list_sub.html"
LOCAL_DIRECT_PROBE_MAX_RACE_NO = 12
LOCAL_UNAVAILABLE_RACE_MARKERS = ("取止", "取り止め", "開催中止")


def _jst_today_text():
    return (datetime.utcnow() + JST_OFFSET).strftime("%Y-%m-%d")


def _normalize_date_text(date_text):
    text = str(date_text or "").strip()
    if not text:
        return _jst_today_text()
    compact = re.sub(r"\D", "", text)
    if len(compact) != 8:
        raise ValueError("date must be YYYY-MM-DD or YYYYMMDD")
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"


def _compact_date(date_text):
    return re.sub(r"\D", "", _normalize_date_text(date_text))


def _clean_text(value):
    text = str(value or "").replace("\xa0", " ").replace("\u3000", " ")
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"\s*-->\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _decoded_html(html_text):
    return _decode_html_text(html_text)


def _unique_urls(urls):
    out = []
    seen = set()
    for url in urls:
        text = str(url or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _race_number_value(race_id):
    race_id_text = normalize_race_id(race_id)
    if len(race_id_text) < 2:
        return None
    try:
        return int(race_id_text[-2:])
    except ValueError:
        return None


def _local_venue_code_from_race_id(race_id):
    race_id_text = normalize_race_id(race_id)
    if len(race_id_text) < 6:
        return ""
    venue_code = race_id_text[4:6]
    return venue_code if venue_code in LOCAL_VENUE_MAP else ""


def _build_local_race_id(date_compact, venue_code, race_no):
    return f"{date_compact[:4]}{venue_code}{date_compact[4:]}{int(race_no):02d}"


def _is_unavailable_race_page(html_text):
    text = _clean_text(BeautifulSoup(_decoded_html(html_text), "html.parser").get_text(" ", strip=True))
    return any(marker in text for marker in LOCAL_UNAVAILABLE_RACE_MARKERS)


def _extract_race_list_tab_urls(html_text, base_url, target_date_compact):
    soup = BeautifulSoup(_decoded_html(html_text), "html.parser")
    out = []
    for node in soup.select("a[href*='race_list_sub.html']"):
        href = str(node.get("href", "") or "").replace("&amp;", "&").strip()
        if f"kaisai_date={target_date_compact}" not in href:
            continue
        out.append(urljoin(base_url, href))
    return _unique_urls(out)


def _race_list_roots(soup):
    roots = soup.select(".RaceList_Body.RaceList_Top")
    return roots or [soup]


def parse_race_list_html(html_text, *, base_url, source):
    source_text = str(source or "").strip().lower()
    soup = BeautifulSoup(_decoded_html(html_text), "html.parser")
    rows_by_id = {}
    for root in _race_list_roots(soup):
        for node in root.select("a[href*='race_id=']"):
            href = str(node.get("href", "") or "").replace("&amp;", "&").strip()
            race_id = normalize_race_id(href)
            if not race_id:
                continue
            text = _clean_text(node.get_text(" ", strip=True))
            if not text:
                text = _clean_text(root.get_text(" ", strip=True))
            rows_by_id.setdefault(
                race_id,
                {
                    "race_id": race_id,
                    "source": source_text,
                    "list_text": text,
                    "source_url": urljoin(base_url, href),
                    "shutuba_url": build_shutuba_url(race_id, source=source_text),
                },
            )
    return [rows_by_id[key] for key in sorted(rows_by_id)]


def _fetch_races_from_urls(urls, *, source, timeout, fetcher):
    races_by_id = {}
    fetched_urls = []
    errors = []
    for url in _unique_urls(urls):
        try:
            html_bytes = fetcher(url, timeout=timeout)
            fetched_urls.append(url)
            for row in parse_race_list_html(html_bytes, base_url=url, source=source):
                races_by_id.setdefault(row["race_id"], row)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
    return {
        "races": [races_by_id[key] for key in sorted(races_by_id)],
        "fetched_urls": fetched_urls,
        "errors": errors,
    }


def _local_probe_row(html_text, *, race_id, source_url):
    if _is_unavailable_race_page(html_text):
        return None
    payload = parse_race_page(html_text, race_id=race_id, source="local", source_url=source_url)
    if not payload.get("meta_complete"):
        return None
    race_number = str(payload.get("race_number", "") or "").strip()
    race_name = str(payload.get("race_name", "") or "").strip()
    list_text = _clean_text(" ".join(part for part in (race_number, race_name) if part))
    return {
        "race_id": normalize_race_id(race_id),
        "source": "local",
        "list_text": list_text,
        "source_url": source_url,
        "shutuba_url": source_url,
    }


def _supplement_local_races_with_direct_probe(races, *, date_compact, timeout, fetcher):
    races_by_id = {
        normalize_race_id(row.get("race_id", "")): dict(row)
        for row in list(races or [])
        if normalize_race_id(row.get("race_id", ""))
    }
    venue_numbers = {}
    for race_id in races_by_id:
        venue_code = _local_venue_code_from_race_id(race_id)
        race_no = _race_number_value(race_id)
        if not venue_code or race_no is None:
            continue
        venue_numbers.setdefault(venue_code, set()).add(race_no)

    supplemented = []
    probe_urls = []
    probe_errors = []
    probe_warnings = []
    for venue_code in sorted(venue_numbers):
        existing_numbers = venue_numbers.get(venue_code, set())
        max_existing_no = max(existing_numbers) if existing_numbers else 0
        for race_no in range(1, LOCAL_DIRECT_PROBE_MAX_RACE_NO + 1):
            race_id = _build_local_race_id(date_compact, venue_code, race_no)
            if race_id in races_by_id:
                continue
            url = build_shutuba_url(race_id, source="local")
            is_internal_gap = race_no <= max_existing_no
            try:
                html_bytes = fetcher(url, timeout=timeout)
                probe_urls.append(url)
                row = _local_probe_row(html_bytes, race_id=race_id, source_url=url)
            except Exception as exc:
                if is_internal_gap:
                    probe_warnings.append({"url": url, "warning": str(exc), "race_id": race_id})
                continue
            if row is None:
                if is_internal_gap:
                    probe_warnings.append(
                        {
                            "url": url,
                            "warning": "race meta incomplete for missing internal race number",
                            "race_id": race_id,
                        }
                    )
                continue
            races_by_id[race_id] = row
            supplemented.append(row)

    return {
        "races": [races_by_id[key] for key in sorted(races_by_id)],
        "supplemented_races": supplemented,
        "probe_urls": probe_urls,
        "probe_errors": probe_errors,
        "probe_warnings": probe_warnings,
    }


def _central_candidate_urls(date_compact, fetcher, timeout):
    urls = []
    top_error = ""
    try:
        top_html = fetcher(CENTRAL_TOP_URL, timeout=timeout)
        urls.extend(_extract_race_list_tab_urls(top_html, CENTRAL_TOP_URL, date_compact))
    except Exception as exc:
        top_error = str(exc)
    urls.append(f"{CENTRAL_RACE_LIST_URL}?kaisai_date={date_compact}&current_group=10{date_compact}")
    return _unique_urls(urls), top_error


def _local_candidate_urls(date_compact, fetcher, timeout):
    urls = []
    top_error = ""
    try:
        top_html = fetcher(LOCAL_TOP_URL, timeout=timeout)
        urls.extend(_extract_race_list_tab_urls(top_html, LOCAL_TOP_URL, date_compact))
    except Exception as exc:
        top_error = str(exc)
    year = date_compact[:4]
    mmdd = date_compact[4:]
    for venue_code in sorted(LOCAL_VENUE_MAP):
        urls.append(f"{LOCAL_RACE_LIST_URL}?kaisai_date={date_compact}&kaisai_id={year}{venue_code}{mmdd}")
    return _unique_urls(urls), top_error


def discover_races_for_date(date_text="", *, include_central=True, include_local=True, timeout=30, fetcher=fetch_html):
    target_date = _normalize_date_text(date_text)
    date_compact = _compact_date(target_date)
    source_results = []
    races_by_id = {}
    warnings = []

    if include_central:
        urls, top_error = _central_candidate_urls(date_compact, fetcher, timeout)
        if top_error:
            warnings.append({"source": "central", "stage": "top", "error": top_error})
        result = _fetch_races_from_urls(urls, source="central", timeout=timeout, fetcher=fetcher)
        result.update({"source": "central", "candidate_count": len(urls), "top_error": top_error})
        source_results.append(result)
        for row in result["races"]:
            races_by_id.setdefault(row["race_id"], row)

    if include_local:
        urls, top_error = _local_candidate_urls(date_compact, fetcher, timeout)
        if top_error:
            warnings.append({"source": "local", "stage": "top", "error": top_error})
        result = _fetch_races_from_urls(urls, source="local", timeout=timeout, fetcher=fetcher)
        supplement = _supplement_local_races_with_direct_probe(
            result["races"],
            date_compact=date_compact,
            timeout=timeout,
            fetcher=fetcher,
        )
        result["races"] = supplement["races"]
        result["supplemented_races"] = supplement["supplemented_races"]
        result["supplemented_count"] = len(supplement["supplemented_races"])
        result["supplement_probe_urls"] = supplement["probe_urls"]
        result["supplement_errors"] = supplement["probe_errors"]
        result["supplement_warnings"] = supplement["probe_warnings"]
        if result["supplement_warnings"]:
            warnings.append(
                {
                    "source": "local",
                    "stage": "supplement",
                    "warnings": list(result["supplement_warnings"] or [])[:3],
                }
            )
        result.update({"source": "local", "candidate_count": len(urls), "top_error": top_error})
        source_results.append(result)
        for row in result["races"]:
            races_by_id.setdefault(row["race_id"], row)

    source_counts = {
        str(item.get("source", "") or ""): len(list(item.get("races", []) or []))
        for item in source_results
    }
    fatal_errors = []
    for item in source_results:
        if not item.get("fetched_urls"):
            fatal_errors.append(
                {
                    "source": str(item.get("source", "") or ""),
                    "errors": list(item.get("errors", []) or [])[:3],
                    "top_error": str(item.get("top_error", "") or ""),
                }
            )
        if item.get("supplement_errors"):
            fatal_errors.append(
                {
                    "source": str(item.get("source", "") or ""),
                    "stage": "supplement",
                    "errors": list(item.get("supplement_errors", []) or [])[:3],
                }
            )
    return {
        "target_date": target_date,
        "target_date_compact": date_compact,
        "races": [races_by_id[key] for key in sorted(races_by_id)],
        "source_counts": source_counts,
        "source_results": source_results,
        "warnings": warnings,
        "errors": fatal_errors,
    }


__all__ = [
    "discover_races_for_date",
    "parse_race_list_html",
]

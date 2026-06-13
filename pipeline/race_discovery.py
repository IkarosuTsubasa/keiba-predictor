import re
from datetime import datetime, timedelta
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from race_meta_fetcher import (
    LOCAL_VENUE_MAP,
    build_shutuba_url,
    fetch_html,
    normalize_race_id,
)
from race_meta_fetcher import _decode_html_text


JST_OFFSET = timedelta(hours=9)
CENTRAL_TOP_URL = "https://race.netkeiba.com/top/"
CENTRAL_RACE_LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html"
LOCAL_TOP_URL = "https://nar.netkeiba.com/top/?rf=navi"
LOCAL_RACE_LIST_URL = "https://nar.netkeiba.com/top/race_list_sub.html"


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

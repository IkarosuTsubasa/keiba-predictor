import re
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from bs4.dammit import UnicodeDammit


JST_OFFSET = timedelta(hours=9)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
CENTRAL_SHUTUBA_URL_BASE = "https://race.netkeiba.com/race/shutuba.html"
LOCAL_SHUTUBA_URL_BASE = "https://nar.netkeiba.com/race/shutuba.html"

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
KNOWN_VENUES = set(CENTRAL_VENUE_MAP.values()) | set(LOCAL_VENUE_MAP.values())
TRACK_CONDITION_MAP = {
    "良": "良",
    "稍": "稍重",
    "稍重": "稍重",
    "重": "重",
    "不": "不良",
    "不良": "不良",
    "-": "良",
    "－": "良",
    "": "良",
}


def _jst_now():
    return datetime.utcnow() + JST_OFFSET


def _jst_today_text():
    return _jst_now().strftime("%Y-%m-%d")


def normalize_race_id(value):
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"race_id=(\d+)", text)
    if match:
        return match.group(1)
    return re.sub(r"\D", "", text)


def race_number_text(race_id):
    race_id_text = normalize_race_id(race_id)
    if len(race_id_text) < 2:
        return ""
    try:
        return f"{int(race_id_text[-2:])}R"
    except ValueError:
        return ""


def venue_code_from_race_id(race_id):
    race_id_text = normalize_race_id(race_id)
    if len(race_id_text) < 6:
        return ""
    return race_id_text[4:6]


def infer_source_from_race_id(race_id):
    venue_code = venue_code_from_race_id(race_id)
    if venue_code in CENTRAL_VENUE_MAP:
        return "central"
    if venue_code in LOCAL_VENUE_MAP:
        return "local"
    return ""


def infer_location_from_race_id(race_id):
    venue_code = venue_code_from_race_id(race_id)
    if venue_code in CENTRAL_VENUE_MAP:
        return CENTRAL_VENUE_MAP[venue_code]
    if venue_code in LOCAL_VENUE_MAP:
        return LOCAL_VENUE_MAP[venue_code]
    return ""


def resolve_source_candidates(race_id, source=""):
    source_text = str(source or "").strip().lower()
    guessed = infer_source_from_race_id(race_id)
    candidates = []
    for item in (source_text, guessed, "central", "local"):
        if item and item not in candidates:
            candidates.append(item)
    return candidates


def build_shutuba_url(race_id, source=""):
    race_id_text = normalize_race_id(race_id)
    if not race_id_text:
        raise ValueError("race_id is required")
    source_text = str(source or "").strip().lower()
    base = LOCAL_SHUTUBA_URL_BASE if source_text == "local" else CENTRAL_SHUTUBA_URL_BASE
    query = urlencode({"race_id": race_id_text})
    return f"{base}?{query}"


def normalize_netkeiba_url(url, source=""):
    parsed = urlparse(str(url or "").strip())
    race_id = normalize_race_id(url)
    if not race_id:
        raise ValueError("race_id not found in url")
    query = parse_qs(parsed.query)
    query["race_id"] = [race_id]
    source_text = str(source or "").strip().lower()
    default_netloc = "nar.netkeiba.com" if source_text == "local" else "race.netkeiba.com"
    default_path = "/race/shutuba.html"
    return urlunparse(
        (
            parsed.scheme or "https",
            parsed.netloc or default_netloc,
            parsed.path or default_path,
            parsed.params,
            urlencode({key: values[-1] for key, values in query.items()}),
            parsed.fragment,
        )
    )


def fetch_html(url, timeout=30):
    req = Request(
        str(url or "").strip(),
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _decode_html_text(html_text):
    if isinstance(html_text, bytes):
        head = html_text[:4096]
        charset_match = re.search(br"charset=['\"]?\s*([A-Za-z0-9_\-]+)", head, flags=re.IGNORECASE)
        if charset_match:
            declared = charset_match.group(1).decode("ascii", errors="ignore").strip().lower()
            charset_aliases = {
                "shift-jis": "cp932",
                "shift_jis": "cp932",
                "sjis": "cp932",
                "windows-31j": "cp932",
                "x-sjis": "cp932",
                "euc-jp": "euc_jp",
                "euc_jp": "euc_jp",
                "utf-8": "utf-8",
                "utf8": "utf-8",
            }
            preferred_encoding = charset_aliases.get(declared, declared)
            try:
                return html_text.decode(preferred_encoding)
            except Exception:
                try:
                    return html_text.decode(preferred_encoding, errors="replace")
                except Exception:
                    pass
        dammit = UnicodeDammit(
            html_text,
            ["utf-8", "euc-jp", "cp932", "shift_jis"],
            is_html=True,
        )
        if dammit.unicode_markup:
            return dammit.unicode_markup
        return html_text.decode("utf-8", errors="replace")
    return str(html_text or "")


def _clean_text(value):
    text = str(value or "").replace("\xa0", " ").replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _node_text(node):
    if node is None:
        return ""
    return _clean_text(node.get_text(" ", strip=True))


def _select_first(root, selectors):
    if root is None:
        return None
    for selector in selectors:
        node = root.select_one(selector)
        if node is not None:
            return node
    return None


def _extract_time_text(text):
    match = re.search(r"(\d{1,2}:\d{2})\s*発走", str(text or ""))
    return match.group(1) if match else ""


def _extract_surface_distance(text):
    match = re.search(r"([芝ダ障])\s*(\d{3,4})[mｍ]", str(text or ""))
    if not match:
        return "", ""
    surface = match.group(1)
    distance = match.group(2)
    return surface, distance


def _normalize_track_condition(value):
    raw = _clean_text(value).replace("馬場:", "").replace("馬場：", "")
    return TRACK_CONDITION_MAP.get(raw, TRACK_CONDITION_MAP.get(raw[:1], "良"))


def _extract_track_condition(text):
    match = re.search(r"馬場[:：]\s*([^\s/]+)", str(text or ""))
    if not match:
        return "良"
    return _normalize_track_condition(match.group(1))


def _extract_location(data2_node, race_id):
    if data2_node is not None:
        spans = [_clean_text(item.get_text(" ", strip=True)) for item in data2_node.select("span")]
        for item in spans:
            if item in KNOWN_VENUES:
                return item
    return infer_location_from_race_id(race_id)


def _extract_location_from_text(text, race_id):
    cleaned = _clean_text(text)
    for venue in sorted(KNOWN_VENUES, key=len, reverse=True):
        if venue and venue in cleaned:
            return venue
    return infer_location_from_race_id(race_id)


def _extract_race_name(soup, container, race_id):
    selectors = (".RaceName", "h1.RaceName", "div.RaceName")
    for root in (container, soup):
        text = _node_text(_select_first(root, selectors))
        if text:
            return text
    title_text = _node_text(getattr(soup, "title", None))
    if title_text:
        title_text = re.split(r"[|｜]", title_text, maxsplit=1)[0]
        title_text = re.sub(r"^\d+\s*R\s*", "", title_text, flags=re.IGNORECASE)
        title_text = _clean_text(title_text)
        if title_text and title_text != race_number_text(race_id):
            return title_text
    return ""


def _extract_data_block_text(soup, container, class_name):
    selector = f".{class_name}"
    for root in (container, soup):
        text = _node_text(_select_first(root, (selector,)))
        if text:
            return text
    return ""


def _missing_meta_fields(payload):
    missing = []
    if not str(payload.get("race_name", "") or "").strip():
        missing.append("race_name")
    if not str(payload.get("location", "") or "").strip():
        missing.append("location")
    if not str(payload.get("scheduled_off_time", "") or "").strip():
        missing.append("scheduled_off_time")
    if not str(payload.get("target_distance", "") or "").strip():
        missing.append("target_distance")
    return missing


def _scope_key_for_source_and_surface(source, surface):
    source_text = str(source or "").strip().lower()
    if source_text == "local":
        return "local"
    return "central_turf" if surface == "芝" else "central_dirt"


def _target_surface_value(surface):
    if surface == "芝":
        return "turf"
    if surface == "ダ":
        return "dirt"
    return ""


def parse_race_page(html_text, race_id="", source="", source_url=""):
    decoded_text = _decode_html_text(html_text)
    soup = BeautifulSoup(decoded_text, "html.parser")
    container = _select_first(soup, (".RaceList_Item02", ".RaceList_Item01", "#RaceList"))
    page_text = _node_text(soup)
    race_name = _extract_race_name(soup, container, race_id)
    data1_text = _extract_data_block_text(soup, container, "RaceData01")
    data2_text = _extract_data_block_text(soup, container, "RaceData02")
    location = infer_location_from_race_id(race_id)
    if container is not None:
        location = _extract_location(container.select_one(".RaceData02"), race_id) or location
    if not location:
        location = _extract_location_from_text(data2_text or page_text, race_id)
    combined_text = _clean_text(" ".join(part for part in (data1_text, data2_text, page_text) if part))
    surface_text, distance_text = _extract_surface_distance(data1_text)
    if not distance_text:
        surface_text, distance_text = _extract_surface_distance(combined_text)
    source_text = str(source or "").strip().lower() or infer_source_from_race_id(race_id)
    track_condition = _extract_track_condition(data1_text or combined_text)
    race_date = _jst_today_text()
    time_text = _extract_time_text(data1_text)
    if not time_text:
        time_text = _extract_time_text(combined_text)
    scheduled_off_time = f"{race_date}T{time_text}:00" if time_text else ""
    payload = {
        "race_id": normalize_race_id(race_id),
        "race_name": race_name,
        "location": location,
        "race_date": race_date,
        "scheduled_off_time": scheduled_off_time,
        "target_distance": distance_text,
        "target_track_condition": track_condition or "良",
        "scope_key": _scope_key_for_source_and_surface(source_text, surface_text),
        "target_surface": _target_surface_value(surface_text),
        "race_number": race_number_text(race_id),
        "surface_text": surface_text,
        "source": source_text,
        "source_url": str(source_url or "").strip(),
        "meta_complete": False,
    }
    payload["meta_complete"] = not _missing_meta_fields(payload)
    return payload


def fetch_race_meta(race_id, source="", timeout=30):
    race_id_text = normalize_race_id(race_id)
    if not race_id_text:
        raise ValueError("race_id is required")
    last_error = None
    for candidate in resolve_source_candidates(race_id_text, source=source):
        try:
            url = build_shutuba_url(race_id_text, source=candidate)
            html_bytes = fetch_html(url, timeout=timeout)
            payload = parse_race_page(
                html_bytes,
                race_id=race_id_text,
                source=candidate,
                source_url=url,
            )
            if payload.get("meta_complete"):
                return payload
            missing_fields = ",".join(_missing_meta_fields(payload)) or "unknown"
            last_error = RuntimeError(f"race meta incomplete for source={candidate} missing={missing_fields}")
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to fetch race meta")


__all__ = [
    "build_shutuba_url",
    "fetch_html",
    "fetch_race_meta",
    "infer_location_from_race_id",
    "infer_source_from_race_id",
    "normalize_netkeiba_url",
    "normalize_race_id",
    "parse_race_page",
    "race_number_text",
]

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from bs4.dammit import UnicodeDammit


CENTRAL_RESULT_URL_BASE = "https://race.netkeiba.com/race/result.html"
LOCAL_RESULT_URL_BASE = "https://nar.netkeiba.com/race/result.html"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


def resolve_result_url_base(source="central"):
    source_text = str(source or "").strip().lower()
    if source_text == "local":
        return LOCAL_RESULT_URL_BASE
    return CENTRAL_RESULT_URL_BASE


def build_result_url(race_id="", url="", source="central"):
    raw_url = str(url or "").strip()
    if raw_url:
        return normalize_result_url(raw_url)
    race_id_text = extract_race_id(race_id)
    if not race_id_text:
        raise ValueError("race_id is required")
    query = urlencode({"race_id": race_id_text, "rf": "race_submenu"})
    return f"{resolve_result_url_base(source)}?{query}"


def normalize_result_url(url):
    parsed = urlparse(str(url or "").strip())
    race_id = extract_race_id(url)
    if not race_id:
        raise ValueError("race_id not found in url")
    query = parse_qs(parsed.query)
    query["race_id"] = [race_id]
    query.setdefault("rf", ["race_submenu"])
    return urlunparse(
        (
            parsed.scheme or "https",
            parsed.netloc or "race.netkeiba.com",
            parsed.path or "/race/result.html",
            parsed.params,
            urlencode({key: values[-1] for key, values in query.items()}),
            parsed.fragment,
        )
    )


def extract_race_id(value):
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"race_id=(\d+)", text)
    if match:
        return match.group(1)
    digits = re.sub(r"\D", "", text)
    return digits


def fetch_html(url, timeout=30):
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def text_or_empty(node):
    if node is None:
        return ""
    return node.get_text(" ", strip=True)


def parse_int_text(value):
    text = str(value or "").strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def parse_top3(soup):
    table = soup.select_one("#All_Result_Table")
    row_nodes = table.select("tbody tr") if table is not None else soup.select("tr")
    rows = []
    for tr in row_nodes:
        rank = parse_int_text(text_or_empty(tr.select_one("td.Result_Num .Rank")))
        num_values = [text_or_empty(node) for node in tr.select("td.Num div, td.Num span")]
        num_values = [value for value in num_values if value]
        horse_no = num_values[-1] if num_values else ""
        if not horse_no:
            horse_no = (
                text_or_empty(tr.select_one("td.Num.Txt_C div"))
                or text_or_empty(tr.select_one("td.Num.Waku div"))
                or text_or_empty(tr.select_one("td.Num.Txt_C"))
            )
        horse_name = (
            text_or_empty(tr.select_one("td.Horse_Info .Horse_Name .HorseNameSpan"))
            or text_or_empty(tr.select_one("td.Horse_Info .Horse_Name a"))
            or text_or_empty(tr.select_one("td.Horse_Info .Horse_Name"))
        )
        if rank is None or not horse_no or not horse_name:
            continue
        if rank > 3:
            continue
        rows.append(
            {
                "rank": rank,
                "horse_no": horse_no,
                "horse_name": horse_name,
            }
        )
    dedup = {}
    for item in rows:
        rank = int(item.get("rank", 99) or 99)
        if rank not in dedup:
            dedup[rank] = item
    out = [dedup[key] for key in sorted(dedup.keys())]
    return out[:3]


def parse_result_groups(result_cell):
    if result_cell is None:
        return []
    groups = []
    ul_nodes = result_cell.find_all("ul", recursive=False)
    if ul_nodes:
        for ul in ul_nodes:
            values = [text_or_empty(span) for span in ul.select("li > span")]
            values = [value for value in values if value]
            if values:
                groups.append(values)
        return groups
    div_nodes = result_cell.find_all("div", recursive=False)
    if div_nodes:
        for div in div_nodes:
            values = [text_or_empty(span) for span in div.find_all("span")]
            values = [value for value in values if value]
            if values:
                groups.append(values)
    if groups:
        return groups
    values = [text_or_empty(span) for span in result_cell.find_all("span")]
    values = [value for value in values if value]
    if values:
        groups.append(values)
    return groups


def parse_multiline_cell(cell):
    if cell is None:
        return []
    text = cell.get_text("\n", strip=True)
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_payout_row(tr):
    bet_type = text_or_empty(tr.select_one("th"))
    if not bet_type:
        return None
    result_groups = parse_result_groups(tr.select_one("td.Result"))
    payout_lines = parse_multiline_cell(tr.select_one("td.Payout"))
    popularity_lines = parse_multiline_cell(tr.select_one("td.Ninki"))
    count = max(len(result_groups), len(payout_lines), len(popularity_lines), 1)
    entries = []
    for idx in range(count):
        horse_numbers = result_groups[idx] if idx < len(result_groups) else []
        payout_text = payout_lines[idx] if idx < len(payout_lines) else ""
        popularity_text = popularity_lines[idx] if idx < len(popularity_lines) else ""
        entries.append(
            {
                "horse_numbers": horse_numbers,
                "payout_text": payout_text,
                "payout_yen": parse_int_text(payout_text),
                "popularity_text": popularity_text,
                "popularity_rank": parse_int_text(popularity_text),
            }
        )
    return {
        "bet_type": bet_type,
        "entries": entries,
    }


def parse_payouts(soup):
    payouts = {}
    wrap = soup.select_one("div.ResultPaybackLeftWrap")
    if wrap is None:
        return payouts
    for tr in wrap.select("table.Payout_Detail_Table tr"):
        row = parse_payout_row(tr)
        if not row:
            continue
        payouts[row["bet_type"]] = row["entries"]
    return payouts


def decode_html_text(html_text):
    if isinstance(html_text, bytes):
        dammit = UnicodeDammit(
            html_text,
            ["utf-8", "euc-jp", "cp932", "shift_jis"],
            is_html=True,
        )
        if dammit.unicode_markup:
            return dammit.unicode_markup
        return html_text.decode("utf-8", errors="replace")
    return str(html_text or "")


def parse_result_page(html_text, source_url=""):
    decoded_text = decode_html_text(html_text)
    soup = BeautifulSoup(decoded_text, "html.parser")
    race_id = extract_race_id(source_url)
    page_title = text_or_empty(soup.select_one("title"))
    top3 = parse_top3(soup)
    payouts = parse_payouts(soup)
    return {
        "race_id": race_id,
        "source_url": str(source_url or "").strip(),
        "page_title": page_title,
        "result_available": bool(top3 and payouts),
        "top3": top3,
        "payouts": payouts,
    }


def dump_failed_html(race_id, html_text):
    race_id_text = extract_race_id(race_id) or "unknown"
    path = Path(__file__).resolve().parent / f"_debug_central_result_{race_id_text}.html"
    path.write_text(decode_html_text(html_text), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Fetch netkeiba result and payouts.")
    parser.add_argument("--race-id", default="", help="race_id, e.g. 202606020608 or 202642031910")
    parser.add_argument("--url", default="", help="Full result URL")
    parser.add_argument("--source", choices=("central", "local"), default="central", help="result page source when using race_id")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    args = parser.parse_args()

    try:
        result_url = build_result_url(race_id=args.race_id, url=args.url, source=args.source)
        html_text = fetch_html(result_url, timeout=max(1, int(args.timeout or 30)))
        payload = parse_result_page(html_text, source_url=result_url)
        if not payload.get("result_available"):
            title = str(payload.get("page_title", "") or "").strip()
            race_id = str(payload.get("race_id", "") or "").strip()
            debug_path = dump_failed_html(race_id, html_text)
            raise RuntimeError(
                f"result not available yet for race_id={race_id or '-'}"
                + (f" title={title}" if title else "")
                + f" debug_html={debug_path}"
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if str(args.output or "").strip():
        output_path = Path(str(args.output).strip())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

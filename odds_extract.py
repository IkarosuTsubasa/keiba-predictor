import csv
import json
import os
import random
import re
import time
import sys
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


SLEEP_RANGE_SECONDS = (1.0, 3.5)
PAGE_LOAD_STRATEGY = os.environ.get("PIPELINE_PAGE_LOAD_STRATEGY", "eager").strip().lower() or "eager"
PAGE_LOAD_TIMEOUT_SECONDS = 15
BLOCK_PATTERNS = (
    "403 forbidden",
    "error 403",
    "http error 403",
    "429 too many requests",
    "too many requests",
    "access denied",
    "request blocked",
    "err_too_many_requests",
)

JRA_API_TYPE_WIN_PLACE = "1"
JRA_API_TYPE_QUINELLA = "4"
JRA_API_TYPE_WIDE = "5"
JRA_API_TYPE_EXACTA = "6"
JRA_API_TYPE_TRIO = "7"
JRA_API_TYPE_TRIFECTA = "8"
JRA_API_TYPES = (
    JRA_API_TYPE_WIN_PLACE,
    JRA_API_TYPE_QUINELLA,
    JRA_API_TYPE_WIDE,
    JRA_API_TYPE_EXACTA,
    JRA_API_TYPE_TRIO,
    JRA_API_TYPE_TRIFECTA,
)


def configure_utf8_io():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_io()


def sleep_jitter():
    time.sleep(random.uniform(*SLEEP_RANGE_SECONDS))


def get_env_float(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if value <= 0:
        return float(default)
    return value


def stop_loading(driver):
    try:
        driver.execute_script("window.stop();")
    except Exception:
        pass


def load_cookies_from_json_file(path="cookie.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cookie_json = json.load(f)
    except FileNotFoundError:
        return []
    cookies = []
    for item in cookie_json:
        name = item.get("name")
        value = item.get("value")
        if name and value:
            cookies.append({"name": name, "value": value})
    return cookies


def should_inject_cookies():
    raw = os.environ.get("PIPELINE_SKIP_COOKIE_INJECTION", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return True
    return False


def should_headless():
    raw = os.environ.get("PIPELINE_HEADLESS", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def get_chrome_profile():
    profile_dir = (
        os.environ.get("PIPELINE_CHROME_PROFILE", "").strip()
        or os.environ.get("CHROME_USER_DATA_DIR", "").strip()
        or os.environ.get("CHROME_PROFILE", "").strip()
    )
    profile_name = (
        os.environ.get("PIPELINE_CHROME_PROFILE_NAME", "").strip()
        or os.environ.get("CHROME_PROFILE_NAME", "").strip()
    )
    return profile_dir, profile_name


def assert_not_blocked(page_source, title, url):
    haystack = f"{title}\n{page_source}".lower()
    for pattern in BLOCK_PATTERNS:
        if pattern in haystack:
            print(f"[ERROR] Access blocked (pattern={pattern}) url={url}")
            raise SystemExit(2)


def get_page_source(url, driver, wait_css=None, timeout=10):
    if not driver:
        raise RuntimeError("Selenium driver is required for odds extraction.")
    try:
        driver.get(url)
    except TimeoutException:
        print(f"[WARN] Page load timeout; continue with partial DOM: {url}")
        stop_loading(driver)
    if wait_css:
        try:
            def has_numeric_text(_driver):
                elems = _driver.find_elements(By.CSS_SELECTOR, wait_css)
                for el in elems:
                    if re.search(r"\d", el.text or ""):
                        return True
                return False

            WebDriverWait(driver, timeout).until(has_numeric_text)
            stop_loading(driver)
        except Exception:
            print(f"[WARN] Timeout waiting for odds values: {wait_css}")
    page_source = driver.page_source
    assert_not_blocked(page_source, driver.title or "", url)
    return page_source


def parse_horse_num_from_odds_id(odds_id):
    match = re.search(r"odds-\d+_(\d+)", odds_id or "")
    if not match:
        return None
    num = match.group(1).lstrip("0")
    return num or match.group(1)


def parse_horse_num_from_input(row):
    input_el = row.select_one('input[id^="chk_b"], input[name="tan[]"]')
    if not input_el:
        return None
    raw = input_el.get("value", "")
    match = re.search(r"_(\d+)$", raw)
    if not match:
        return None
    num = match.group(1).lstrip("0")
    return num or match.group(1)


def parse_horse_num_from_umaban(row):
    umaban_el = row.select_one('td[class^="Umaban"], td[class*="Umaban"]')
    if not umaban_el:
        return None
    raw = umaban_el.get_text(strip=True)
    num = re.sub(r"\D", "", raw)
    return num or None


def extract_numbers(text):
    raw = (text or "").replace(",", "")
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", raw)]


def extract_json_payload(text):
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("<"):
        raw = BeautifulSoup(raw, "html.parser").get_text("\n", strip=True)
    raw = raw.strip()
    if raw.startswith("jQuery") or raw.endswith(");"):
        start = raw.find("(")
        end = raw.rfind(")")
        if start != -1 and end != -1 and end > start:
            raw = raw[start + 1 : end]
    if not (raw.startswith("{") or raw.startswith("[")):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def fetch_text_url(url, timeout=15):
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    for enc in ("utf-8", "euc-jp", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_float_text(value):
    raw = str(value or "").strip().replace(",", "")
    if raw in ("", "-", "---.-"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def normalize_horse_no(value):
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw = raw.lstrip("0")
    return raw or "0"


def split_combo_key(combo_key):
    raw = re.sub(r"\D", "", str(combo_key or ""))
    if not raw or len(raw) % 2 != 0:
        return []
    return [normalize_horse_no(raw[idx : idx + 2]) for idx in range(0, len(raw), 2)]


def dedupe_horse_results(items):
    seen = set()
    deduped = []
    for item in items:
        key = (item.get("horse_no") or "", item.get("name") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def parse_row(row, fallback_index):
    name_el = row.select_one(".HorseName a")
    if name_el is None:
        name_el = row.select_one(".Horse_Name")
    if name_el is None:
        name_el = row.select_one(".HorseName")
    name = name_el.get_text(strip=True) if name_el else ""
    if not name:
        return None

    odds_el = row.select_one('td.Txt_R.Popular span[id^="odds-"]')
    if odds_el is None:
        odds_el = row.select_one("td.Odds span.Odds")
    if odds_el is None:
        odds_el = row.select_one('td.Odds span[id^="odds-"]')
    if odds_el is None:
        odds_el = row.select_one('span[id^="odds-"]')
    if odds_el is None:
        odds_el = row.select_one("td.Txt_R.Popular span.Odds_Ninki")
    if odds_el is None:
        odds_el = row.select_one("td.Txt_R.Popular")
    if odds_el is None:
        odds_el = row.select_one("td.Odds")

    odds_text = odds_el.get_text(strip=True) if odds_el else ""
    if odds_text and odds_el and odds_el.name == "td":
        nums = extract_numbers(odds_text)
        odds_text = str(nums[0]) if nums else odds_text
    odds = "" if odds_text in ("", "-") else odds_text.replace(",", "")

    horse_no = parse_horse_num_from_odds_id(odds_el.get("id") if odds_el else "")
    if not horse_no:
        horse_no_el = row.select_one(".HorseNum")
        if horse_no_el:
            horse_no = horse_no_el.get_text(strip=True)
    if not horse_no:
        horse_no = parse_horse_num_from_input(row)
    if not horse_no:
        horse_no = parse_horse_num_from_umaban(row)
    if not horse_no:
        horse_no = str(fallback_index)

    return {"horse_no": horse_no, "name": name, "odds": odds}


def parse_odds_from_page(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("table.RaceTable01.ShutubaTable tbody tr.HorseList")
    if not rows:
        rows = soup.select(".HorseList")
    results = []
    for idx, row in enumerate(rows, 1):
        item = parse_row(row, idx)
        if item:
            results.append(item)
    if results:
        return dedupe_horse_results(results)

    rows = soup.select("table.RaceOdds_HorseList_Table tbody tr")
    results = []
    for idx, row in enumerate(rows, 1):
        item = parse_row(row, idx)
        if item:
            results.append(item)
    if results:
        return dedupe_horse_results(results)

    results = []
    for idx, odds_el in enumerate(soup.select('span[id^="odds-"], span.Odds_Ninki, td.Txt_R.Popular'), 1):
        odds_text = odds_el.get_text(strip=True)
        if odds_text and odds_el.name == "td":
            nums = extract_numbers(odds_text)
            odds_text = str(nums[0]) if nums else odds_text
        odds = "" if odds_text in ("", "-") else odds_text.replace(",", "")
        horse_no = parse_horse_num_from_odds_id(odds_el.get("id"))
        if not horse_no:
            horse_no = str(idx)
        results.append({"horse_no": horse_no, "name": "", "odds": odds})
    return dedupe_horse_results(results)


def extract_race_id(url):
    try:
        parsed = urlparse(url)
        race_id = parse_qs(parsed.query).get("race_id", [""])[0]
        race_id = re.sub(r"\D", "", race_id)
        return race_id
    except Exception:
        return ""


def build_tan_odds_url(race_url):
    race_id = extract_race_id(race_url)
    if not race_id:
        return ""
    host = urlparse(race_url).netloc.lower()
    if "nar.netkeiba.com" in host:
        return f"https://nar.netkeiba.com/odds/?race_id={race_id}&type=b1"
    if "race.netkeiba.com" in host:
        return f"https://race.netkeiba.com/odds/index.html?type=b1&race_id={race_id}"
    return ""


def build_wide_odds_url(race_url):
    race_id = extract_race_id(race_url)
    if not race_id:
        return ""
    host = urlparse(race_url).netloc.lower()
    if "nar.netkeiba.com" in host:
        return f"https://nar.netkeiba.com/odds/?race_id={race_id}&type=b5"
    if "race.netkeiba.com" in host:
        return f"https://race.netkeiba.com/odds/index.html?type=b5&race_id={race_id}&housiki=c0"
    return ""


def build_quinella_odds_url(race_url):
    race_id = extract_race_id(race_url)
    if not race_id:
        return ""
    host = urlparse(race_url).netloc.lower()
    if "nar.netkeiba.com" in host:
        return f"https://nar.netkeiba.com/odds/?race_id={race_id}&type=b4"
    if "race.netkeiba.com" in host:
        return f"https://race.netkeiba.com/odds/index.html?type=b4&race_id={race_id}&housiki=c0"
    return ""


def build_exacta_odds_url(race_url):
    race_id = extract_race_id(race_url)
    if not race_id:
        return ""
    host = urlparse(race_url).netloc.lower()
    if "nar.netkeiba.com" in host:
        return f"https://nar.netkeiba.com/odds/?race_id={race_id}&type=b6"
    if "race.netkeiba.com" in host:
        return f"https://race.netkeiba.com/odds/index.html?type=b6&race_id={race_id}&housiki=c0"
    return ""


def build_nar_odds_page_url(race_url, odds_type):
    race_id = extract_race_id(race_url)
    if not race_id:
        return ""
    host = urlparse(race_url).netloc.lower()
    if "nar.netkeiba.com" not in host:
        return ""
    return f"https://nar.netkeiba.com/odds/?race_id={race_id}&type={odds_type}"


def build_nar_odds_fragment_url(race_url, odds_type, jiku):
    race_id = extract_race_id(race_url)
    if not race_id:
        return ""
    host = urlparse(race_url).netloc.lower()
    if "nar.netkeiba.com" not in host:
        return ""
    return f"https://nar.netkeiba.com/odds/odds_get_form.html?type={odds_type}&race_id={race_id}&jiku={jiku}"


def build_race_card_url(race_url):
    race_id = extract_race_id(race_url)
    if not race_id:
        return ""
    host = urlparse(race_url).netloc.lower()
    if "race.netkeiba.com" in host:
        return f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    if "nar.netkeiba.com" in host:
        return f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    return race_url


def build_jra_odds_api_url(race_id, odds_type):
    return (
        "https://race.netkeiba.com/api/api_get_jra_odds.html"
        f"?pid=api_get_jra_odds&input=UTF-8&output=jsonp&race_id={race_id}"
        f"&type={odds_type}&action=init&sort=odds&compress=0"
    )


def fetch_jra_api_payload(race_id, odds_type):
    payload = extract_json_payload(fetch_text_url(build_jra_odds_api_url(race_id, odds_type)))
    if not payload:
        raise RuntimeError(f"Failed to parse JRA odds api payload: type={odds_type}")
    status = str(payload.get("status") or "").strip().lower()
    odds_root = (((payload or {}).get("data") or {}).get("odds") or {})
    if status not in {"result", "middle"}:
        raise RuntimeError(
            f"JRA odds api returned status={payload.get('status')} type={odds_type}"
        )
    if not odds_root:
        raise RuntimeError(
            f"JRA odds api returned no odds payload: status={payload.get('status')} type={odds_type}"
        )
    return payload


def build_horse_name_map_from_race_page(race_url):
    card_url = build_race_card_url(race_url)
    if not card_url:
        return {}
    try:
        html = fetch_text_url(card_url)
    except Exception:
        return {}
    entries = parse_odds_from_page(html)
    out = {}
    for item in entries:
        horse_no = normalize_horse_no(item.get("horse_no"))
        name = str(item.get("name", "")).strip()
        if horse_no and name and horse_no not in out:
            out[horse_no] = name
    return out


def parse_jra_type1_payload(payload, horse_name_map):
    odds_root = (((payload or {}).get("data") or {}).get("odds") or {})
    win_map = odds_root.get("1") or {}
    place_map = odds_root.get("2") or {}
    win_results = []
    place_results = []
    for horse_key in sorted(win_map.keys(), key=lambda x: int(re.sub(r"\D", "", x) or "0")):
        horse_no = normalize_horse_no(horse_key)
        row = win_map.get(horse_key) or []
        odds = parse_float_text(row[0] if len(row) > 0 else "")
        if odds is None:
            continue
        win_results.append(
            {
                "horse_no": horse_no,
                "name": horse_name_map.get(horse_no, ""),
                "odds": str(odds),
            }
        )
    for horse_key in sorted(
        place_map.keys(), key=lambda x: int(re.sub(r"\D", "", x) or "0")
    ):
        horse_no = normalize_horse_no(horse_key)
        row = place_map.get(horse_key) or []
        low = parse_float_text(row[0] if len(row) > 0 else "")
        high = parse_float_text(row[1] if len(row) > 1 else "")
        if low is None and high is None:
            continue
        if low is None:
            low = high
        if high is None:
            high = low
        mid = round((float(low) + float(high)) / 2.0, 3)
        place_results.append(
            {
                "horse_no": horse_no,
                "name": horse_name_map.get(horse_no, ""),
                "odds_low": float(low),
                "odds_high": float(high),
                "odds_mid": mid,
            }
        )
    return win_results, place_results


def parse_jra_pair_payload(payload, odds_type):
    odds_root = (((payload or {}).get("data") or {}).get("odds") or {})
    pair_map = odds_root.get(str(odds_type)) or {}
    results = []
    for combo_key in sorted(pair_map.keys()):
        parts = split_combo_key(combo_key)
        if len(parts) != 2:
            continue
        row = pair_map.get(combo_key) or []
        if odds_type == JRA_API_TYPE_WIDE:
            low = parse_float_text(row[0] if len(row) > 0 else "")
            high = parse_float_text(row[1] if len(row) > 1 else "")
            if low is None and high is None:
                continue
            if low is None:
                low = high
            if high is None:
                high = low
            mid = round((float(low) + float(high)) / 2.0, 3)
            results.append(
                {
                    "horse_no_a": parts[0],
                    "horse_no_b": parts[1],
                    "odds_low": float(low),
                    "odds_high": float(high),
                    "odds_mid": mid,
                }
            )
            continue
        odds = parse_float_text(row[0] if len(row) > 0 else "")
        if odds is None:
            continue
        results.append({"horse_no_a": parts[0], "horse_no_b": parts[1], "odds": float(odds)})
    return results


def parse_jra_triple_payload(payload, odds_type):
    odds_root = (((payload or {}).get("data") or {}).get("odds") or {})
    triple_map = odds_root.get(str(odds_type)) or {}
    results = []
    for combo_key in sorted(triple_map.keys()):
        parts = split_combo_key(combo_key)
        if len(parts) != 3:
            continue
        row = triple_map.get(combo_key) or []
        odds = parse_float_text(row[0] if len(row) > 0 else "")
        if odds is None:
            continue
        results.append(
            {
                "horse_no_a": parts[0],
                "horse_no_b": parts[1],
                "horse_no_c": parts[2],
                "odds": float(odds),
            }
        )
    return results


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def fetch_and_save_jra_api_odds(race_url):
    race_id = extract_race_id(race_url)
    if not race_id:
        return False
    horse_name_map = build_horse_name_map_from_race_page(race_url)

    payloads = {}
    for odds_type in JRA_API_TYPES:
        payloads[odds_type] = fetch_jra_api_payload(race_id, odds_type)

    official_time = (
        (((payloads.get(JRA_API_TYPE_WIN_PLACE) or {}).get("data") or {}).get("official_datetime"))
        or ""
    )
    if official_time:
        print(f"Official time: {official_time}")

    win_results, place_results = parse_jra_type1_payload(
        payloads[JRA_API_TYPE_WIN_PLACE], horse_name_map
    )
    if win_results:
        for item in win_results:
            print(f"{item['horse_no']}\t{item['name']}\t{item['odds']}")
        save_csv("odds.csv", ["horse_no", "name", "odds"], win_results)
    if place_results:
        save_csv(
            "fuku_odds.csv",
            ["horse_no", "name", "odds_low", "odds_high", "odds_mid"],
            place_results,
        )

    quinella_results = parse_jra_pair_payload(
        payloads[JRA_API_TYPE_QUINELLA], JRA_API_TYPE_QUINELLA
    )
    if quinella_results:
        save_csv("quinella_odds.csv", ["horse_no_a", "horse_no_b", "odds"], quinella_results)

    wide_results = parse_jra_pair_payload(payloads[JRA_API_TYPE_WIDE], JRA_API_TYPE_WIDE)
    if wide_results:
        save_csv(
            "wide_odds.csv",
            ["horse_no_a", "horse_no_b", "odds_low", "odds_high", "odds_mid"],
            wide_results,
        )

    exacta_results = parse_jra_pair_payload(payloads[JRA_API_TYPE_EXACTA], JRA_API_TYPE_EXACTA)
    if exacta_results:
        save_csv("exacta_odds.csv", ["horse_no_a", "horse_no_b", "odds"], exacta_results)

    trio_results = parse_jra_triple_payload(payloads[JRA_API_TYPE_TRIO], JRA_API_TYPE_TRIO)
    if trio_results:
        save_csv(
            "trio_odds.csv", ["horse_no_a", "horse_no_b", "horse_no_c", "odds"], trio_results
        )

    trifecta_results = parse_jra_triple_payload(
        payloads[JRA_API_TYPE_TRIFECTA], JRA_API_TYPE_TRIFECTA
    )
    if trifecta_results:
        save_csv(
            "trifecta_odds.csv",
            ["horse_no_a", "horse_no_b", "horse_no_c", "odds"],
            trifecta_results,
        )

    return bool(
        win_results
        or place_results
        or quinella_results
        or wide_results
        or exacta_results
        or trio_results
        or trifecta_results
    )


def parse_tan_odds_from_page(html):
    soup = BeautifulSoup(html, "html.parser")
    block = soup.select_one("#odds_tan_block") or soup
    results = []
    for row in block.select("table.RaceOdds_HorseList_Table tr"):
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        horse_no = re.sub(r"\D", "", cells[1].get_text(strip=True))
        name_el = row.select_one(".Horse_Name") or row.select_one(".HorseName")
        odds_el = (
            row.select_one("td.Odds span.Odds")
            or row.select_one('td.Odds span[id^="odds-"]')
            or row.select_one('span[id^="odds-"]')
        )
        if odds_el is None:
            odds_el = row.select_one("td.Odds")
        name = name_el.get_text(strip=True) if name_el else ""
        odds_text = odds_el.get_text(strip=True) if odds_el else ""
        nums = extract_numbers(odds_text)
        odds = str(nums[0]) if nums else ""
        if not horse_no and odds_el is not None:
            horse_no = parse_horse_num_from_odds_id(odds_el.get("id"))
        if not horse_no:
            horse_no = parse_horse_num_from_input(row)
        if not horse_no or not name or not odds:
            continue
        results.append({"horse_no": horse_no, "name": name, "odds": odds})
    return results


def parse_fuku_odds_from_page(html):
    soup = BeautifulSoup(html, "html.parser")
    block = soup.select_one("#odds_fuku_block") or soup
    results = []
    for row in block.select("table.RaceOdds_HorseList_Table tr"):
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        horse_no = re.sub(r"\D", "", cells[1].get_text(strip=True))
        name_el = row.select_one(".Horse_Name") or row.select_one(".HorseName")
        odds_el = (
            row.select_one("td.Odds span.Odds")
            or row.select_one('td.Odds span[id^="odds-"]')
            or row.select_one('span[id^="odds-"]')
        )
        if odds_el is None:
            odds_el = row.select_one("td.Odds")
        name = name_el.get_text(strip=True) if name_el else ""
        odds_text = odds_el.get_text(strip=True) if odds_el else ""
        odds_nums = extract_numbers(odds_text)
        if len(odds_nums) >= 2:
            low, high = sorted(odds_nums[:2])
            mid = round((low + high) / 2.0, 3)
        elif len(odds_nums) == 1:
            low = high = float(odds_nums[0])
            mid = float(odds_nums[0])
        else:
            continue
        if not horse_no and odds_el is not None:
            horse_no = parse_horse_num_from_odds_id(odds_el.get("id"))
        if not horse_no:
            horse_no = parse_horse_num_from_input(row)
        if not horse_no or not name:
            continue
        results.append(
            {
                "horse_no": horse_no,
                "name": name,
                "odds_low": low,
                "odds_high": high,
                "odds_mid": mid,
            }
        )
    return results


def parse_quinella_odds_from_page(html):
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select(".GraphOdds table.Odds_Table")
    if not tables:
        tables = soup.select("table.Odds_Table")
    results = []
    seen = set()

    def normalize_pair(a, b):
        if not a or not b:
            return "", ""
        a_i = int(a)
        b_i = int(b)
        if a_i == b_i:
            return "", ""
        if a_i > b_i:
            a_i, b_i = b_i, a_i
        return str(a_i), str(b_i)

    def parse_pair_from_token(token):
        if not token:
            return "", ""
        match = re.search(r"_([0-9]+)_([0-9]+)$", token)
        if match:
            left = match.group(1).lstrip("0") or match.group(1)
            right = match.group(2).lstrip("0") or match.group(2)
            return normalize_pair(left, right)
        match = re.search(r"odds-4-(\d+)", token)
        if match:
            digits = match.group(1)
            if len(digits) % 2 == 0:
                mid = len(digits) // 2
                left = digits[:mid].lstrip("0") or digits[:mid]
                right = digits[mid:].lstrip("0") or digits[mid:]
                return normalize_pair(left, right)
        return "", ""
    for table in tables:
        head = table.select_one("tr.col_label th")
        base_no = head.get_text(strip=True) if head else ""
        base_no = re.sub(r"\D", "", base_no)
        if not base_no:
            continue
        for row in table.select("tr"):
            cells = row.select("td")
            if not cells:
                continue
            horse_no = cells[0].get_text(strip=True)
            horse_no = re.sub(r"\D", "", horse_no)
            odds_cell = row.select_one("td.Odds") or (cells[1] if len(cells) > 1 else None)
            odds_span = odds_cell.select_one('span[id^="odds-4-"]') if odds_cell else None
            odds_text = odds_span.get_text(" ", strip=True) if odds_span else ""
            if not odds_text and odds_cell:
                odds_text = odds_cell.get_text(" ", strip=True)
            odds_nums = extract_numbers(odds_text)
            if not odds_nums:
                continue
            odds = float(odds_nums[0])
            a, b = normalize_pair(base_no, horse_no)
            if not a or not b:
                token = ""
                if odds_cell is not None:
                    token = odds_cell.get("cart-item") or odds_cell.get("name") or odds_cell.get("id") or ""
                if not token:
                    token = row.get("id") or ""
                a, b = parse_pair_from_token(token)
            if not a or not b:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            results.append({"horse_no_a": a, "horse_no_b": b, "odds": odds})
    if results:
        return results
    for odds_cell in soup.select('.GraphOdds td[cart-item], td[cart-item]'):
        odds_span = odds_cell.select_one('span[id^="odds-4-"]') or odds_cell.select_one(
            'span[id^="odds-"]'
        )
        odds_text = odds_span.get_text(" ", strip=True) if odds_span else ""
        if not odds_text:
            odds_text = odds_cell.get_text(" ", strip=True)
        odds_nums = extract_numbers(odds_text)
        if not odds_nums:
            continue
        token = odds_cell.get("cart-item") or odds_cell.get("name") or odds_cell.get("id") or ""
        a, b = parse_pair_from_token(token)
        if not a or not b:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        results.append({"horse_no_a": a, "horse_no_b": b, "odds": float(odds_nums[0])})
    return results


def parse_exacta_odds_from_page(html):
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select(".GraphOdds table.Odds_Table")
    if not tables:
        tables = soup.select("table.Odds_Table")
    results = []
    seen = set()

    def parse_pair_from_token(token):
        if not token:
            return "", ""
        match = re.search(r"_([0-9]+)_([0-9]+)$", token)
        if match:
            left = normalize_horse_no(match.group(1))
            right = normalize_horse_no(match.group(2))
            return left, right
        match = re.search(r"odds-6-(\d+)", token)
        if match:
            digits = match.group(1)
            if len(digits) % 2 == 0:
                mid = len(digits) // 2
                return normalize_horse_no(digits[:mid]), normalize_horse_no(digits[mid:])
        return "", ""

    for table in tables:
        head = table.select_one("tr.col_label th")
        base_no = re.sub(r"\D", "", head.get_text(strip=True) if head else "")
        if not base_no:
            continue
        for row in table.select("tr"):
            cells = row.select("td")
            if not cells:
                continue
            horse_no = re.sub(r"\D", "", cells[0].get_text(strip=True))
            odds_cell = row.select_one("td.Odds") or (cells[1] if len(cells) > 1 else None)
            odds_span = odds_cell.select_one('span[id^="odds-6-"]') if odds_cell else None
            odds_text = odds_span.get_text(" ", strip=True) if odds_span else ""
            if not odds_text and odds_cell:
                odds_text = odds_cell.get_text(" ", strip=True)
            odds_nums = extract_numbers(odds_text)
            if not odds_nums:
                continue
            a = normalize_horse_no(base_no)
            b = normalize_horse_no(horse_no)
            if not a or not b or a == b:
                token = ""
                if odds_cell is not None:
                    token = odds_cell.get("cart-item") or odds_cell.get("name") or odds_cell.get("id") or ""
                if not token:
                    token = row.get("id") or ""
                a, b = parse_pair_from_token(token)
            if not a or not b or a == b:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            results.append({"horse_no_a": a, "horse_no_b": b, "odds": float(odds_nums[0])})

    if results:
        return results

    for odds_cell in soup.select('.GraphOdds td[cart-item], td[cart-item]'):
        odds_span = odds_cell.select_one('span[id^="odds-6-"]') or odds_cell.select_one(
            'span[id^="odds-"]'
        )
        odds_text = odds_span.get_text(" ", strip=True) if odds_span else ""
        if not odds_text:
            odds_text = odds_cell.get_text(" ", strip=True)
        odds_nums = extract_numbers(odds_text)
        if not odds_nums:
            continue
        token = odds_cell.get("cart-item") or odds_cell.get("name") or odds_cell.get("id") or ""
        a, b = parse_pair_from_token(token)
        if not a or not b or a == b:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        results.append({"horse_no_a": a, "horse_no_b": b, "odds": float(odds_nums[0])})
    return results


def parse_nar_jiku_list(html):
    soup = BeautifulSoup(html, "html.parser")
    values = []
    seen = set()
    for option in soup.select("#list_select_horse option"):
        value = normalize_horse_no(option.get("value", ""))
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def parse_nar_triple_odds_from_page(html, keep_order):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    seen = set()

    def parse_triple_from_token(token):
        if not token:
            return []
        match = re.search(r"_([0-9]+)_([0-9]+)_([0-9]+)$", token)
        if not match:
            return []
        parts = [
            normalize_horse_no(match.group(1)),
            normalize_horse_no(match.group(2)),
            normalize_horse_no(match.group(3)),
        ]
        if any(not part for part in parts):
            return []
        return parts

    for odds_cell in soup.select(".GraphOdds td.Odds[cart-item], td.Odds[cart-item]"):
        odds_text = odds_cell.get_text(" ", strip=True)
        odds_nums = extract_numbers(odds_text)
        if not odds_nums:
            continue
        token = odds_cell.get("cart-item") or odds_cell.get("name") or odds_cell.get("id") or ""
        parts = parse_triple_from_token(token)
        if len(parts) != 3:
            continue
        if keep_order:
            key = tuple(parts)
        else:
            key = tuple(sorted(parts, key=lambda x: int(x)))
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "horse_no_a": key[0],
                "horse_no_b": key[1],
                "horse_no_c": key[2],
                "odds": float(odds_nums[0]),
            }
        )
    return results


def fetch_and_save_nar_triple_odds(race_url, odds_type, out_path, keep_order):
    page_url = build_nar_odds_page_url(race_url, odds_type)
    if not page_url:
        return []
    initial_html = fetch_text_url(page_url)
    jiku_list = parse_nar_jiku_list(initial_html)
    if not jiku_list:
        return []

    results = []
    seen = set()
    for idx, jiku in enumerate(jiku_list):
        if idx > 0:
            sleep_jitter()
        fragment_url = build_nar_odds_fragment_url(race_url, odds_type, jiku)
        if not fragment_url:
            continue
        html = fetch_text_url(fragment_url)
        for item in parse_nar_triple_odds_from_page(html, keep_order=keep_order):
            key = (item["horse_no_a"], item["horse_no_b"], item["horse_no_c"])
            if key in seen:
                continue
            seen.add(key)
            results.append(item)

    if results:
        save_csv(out_path, ["horse_no_a", "horse_no_b", "horse_no_c", "odds"], results)
    return results


def extract_wide_odds_numbers(odds_cell):
    if odds_cell is None:
        return []
    low_el = odds_cell.select_one('span[id^="odds-5-"]')
    high_el = odds_cell.select_one('span[id^="oddsmin-5-"]')
    nums = []
    for el in (low_el, high_el):
        if el is None:
            continue
        values = extract_numbers(el.get_text(" ", strip=True))
        if values:
            nums.append(values[0])
    if nums:
        return nums
    return extract_numbers(odds_cell.get_text(" ", strip=True))


def parse_wide_odds_from_page(html):
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select(".GraphOdds table.Odds_Table")
    results = []
    for table in tables:
        head = table.select_one("tr.col_label th")
        base_no = head.get_text(strip=True) if head else ""
        base_no = re.sub(r"\D", "", base_no)
        if not base_no:
            continue
        for row in table.select("tr"):
            cells = row.select("td")
            if not cells:
                continue
            horse_no = cells[0].get_text(strip=True)
            horse_no = re.sub(r"\D", "", horse_no)
            odds_cell = row.select_one("td.Odds")
            odds_nums = extract_wide_odds_numbers(odds_cell)
            if len(odds_nums) >= 2:
                low, high = sorted(odds_nums[:2])
                mid = round((low + high) / 2.0, 3)
            elif len(odds_nums) == 1:
                low = high = float(odds_nums[0])
                mid = float(odds_nums[0])
            else:
                continue
            a = base_no
            b = horse_no
            if not a or not b:
                continue
            a_i = int(a)
            b_i = int(b)
            if a_i == b_i:
                continue
            if a_i > b_i:
                a_i, b_i = b_i, a_i
            results.append(
                {
                    "horse_no_a": str(a_i),
                    "horse_no_b": str(b_i),
                    "odds_low": low,
                    "odds_high": high,
                    "odds_mid": mid,
                }
            )
    return results


def safe_parse(label, func, *args):
    try:
        return func(*args)
    except Exception as exc:
        print(f"[WARN] {label} parse failed: {exc}")
        return []


def is_jra_host(host):
    return "race.netkeiba.com" in host


def is_nar_host(host):
    return "nar.netkeiba.com" in host


def build_webdriver():
    options = Options()
    if PAGE_LOAD_STRATEGY in ("normal", "eager", "none"):
        options.page_load_strategy = PAGE_LOAD_STRATEGY
    debugger_address = os.environ.get("CHROME_DEBUGGER_ADDRESS", "").strip()
    shared_driver = bool(debugger_address)
    if shared_driver:
        options.add_experimental_option("debuggerAddress", debugger_address)
    else:
        if should_headless():
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
        options.add_argument("--lang=ja-JP")
        profile_dir, profile_name = get_chrome_profile()
        if profile_dir:
            options.add_argument(f"--user-data-dir={profile_dir}")
            if profile_name:
                options.add_argument(f"--profile-directory={profile_name}")
    driver = webdriver.Chrome(options=options)
    try:
        page_load_timeout = get_env_float("PIPELINE_PAGE_LOAD_TIMEOUT", PAGE_LOAD_TIMEOUT_SECONDS)
        driver.set_page_load_timeout(page_load_timeout)
    except Exception:
        pass
    return driver, shared_driver


def prepare_driver_session(driver):
    if should_inject_cookies():
        cookies = load_cookies_from_json_file("cookie.txt")
        driver.get("https://db.netkeiba.com")
        assert_not_blocked(driver.page_source, driver.title or "", "https://db.netkeiba.com")
        for cookie in cookies:
            driver.add_cookie(cookie)
        return
    print("Skipping cookie injection (PIPELINE_SKIP_COOKIE_INJECTION=1).")


def fetch_primary_odds_via_browser(race_url, driver):
    tan_url = build_tan_odds_url(race_url)
    fuku_results = []
    if tan_url:
        sleep_jitter()
        page = get_page_source(tan_url, driver, wait_css="span[id^='odds-']")
        results = safe_parse("tan", parse_tan_odds_from_page, page)
        fuku_results = safe_parse("fuku", parse_fuku_odds_from_page, page)
        if not results:
            results = safe_parse("odds", parse_odds_from_page, page)
        return results, fuku_results

    sleep_jitter()
    page = get_page_source(race_url, driver, wait_css="span[id^='odds-']")
    results = safe_parse("odds", parse_odds_from_page, page)
    if not results:
        results = safe_parse("tan", parse_tan_odds_from_page, page)
    fuku_results = safe_parse("fuku", parse_fuku_odds_from_page, page)
    return results, fuku_results


def save_primary_odds_results(results, fuku_results):
    for item in results:
        print(f"{item['horse_no']}\t{item['name']}\t{item['odds']}")
    save_csv("odds.csv", ["horse_no", "name", "odds"], results)
    if fuku_results:
        save_csv(
            "fuku_odds.csv",
            ["horse_no", "name", "odds_low", "odds_high", "odds_mid"],
            fuku_results,
        )


def fetch_and_save_html_odds(race_url, driver, label, build_url, wait_css, parse_func, out_path, fieldnames):
    odds_url = build_url(race_url)
    if not odds_url:
        return []
    sleep_jitter()
    results = safe_parse(label, parse_func, get_page_source(odds_url, driver, wait_css=wait_css))
    if results:
        save_csv(out_path, fieldnames, results)
    return results


def fetch_and_save_nar_triple_series(race_url):
    try:
        fetch_and_save_nar_triple_odds(race_url, "b7", "trio_odds.csv", keep_order=False)
    except Exception as exc:
        print(f"[WARN] trio fetch failed: {exc}")
    try:
        fetch_and_save_nar_triple_odds(race_url, "b8", "trifecta_odds.csv", keep_order=True)
    except Exception as exc:
        print(f"[WARN] trifecta fetch failed: {exc}")


def run_browser_odds_flow(race_url, host):
    try:
        driver, shared_driver = build_webdriver()
    except Exception as exc:
        print(f"[ERROR] Selenium unavailable: {exc}")
        return

    try:
        prepare_driver_session(driver)
        results, fuku_results = fetch_primary_odds_via_browser(race_url, driver)
        if not results:
            print("No odds found.")
            return

        save_primary_odds_results(results, fuku_results)

        fetch_and_save_html_odds(
            race_url,
            driver,
            "wide",
            build_wide_odds_url,
            "span[id^='odds-5-']",
            parse_wide_odds_from_page,
            "wide_odds.csv",
            ["horse_no_a", "horse_no_b", "odds_low", "odds_high", "odds_mid"],
        )
        fetch_and_save_html_odds(
            race_url,
            driver,
            "quinella",
            build_quinella_odds_url,
            "span[id^='odds-4-']",
            parse_quinella_odds_from_page,
            "quinella_odds.csv",
            ["horse_no_a", "horse_no_b", "odds"],
        )
        fetch_and_save_html_odds(
            race_url,
            driver,
            "exacta",
            build_exacta_odds_url,
            "span[id^='odds-6-'], td.Odds",
            parse_exacta_odds_from_page,
            "exacta_odds.csv",
            ["horse_no_a", "horse_no_b", "odds"],
        )

        if is_nar_host(host):
            fetch_and_save_nar_triple_series(race_url)
    finally:
        if driver and not shared_driver:
            driver.quit()


def main():
    url = input("Race URL: ").strip()
    if not url:
        print("No URL provided.")
        return

    host = urlparse(url).netloc.lower()
    if is_jra_host(host):
        try:
            if fetch_and_save_jra_api_odds(url):
                return
        except Exception as exc:
            print(f"[WARN] JRA api odds fetch failed, fallback to browser flow: {exc}")
    run_browser_odds_flow(url, host)


if __name__ == "__main__":
    main()

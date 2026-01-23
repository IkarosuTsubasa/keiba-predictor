import csv
import json
import os
import random
import re
import time
import sys
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


SLEEP_RANGE_SECONDS = (1.0, 3.5)
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


def assert_not_blocked(page_source, title, url):
    haystack = f"{title}\n{page_source}".lower()
    for pattern in BLOCK_PATTERNS:
        if pattern in haystack:
            print(f"[ERROR] Access blocked (pattern={pattern}) url={url}")
            raise SystemExit(2)


def get_page_source(url, driver, wait_css=None, timeout=10):
    if not driver:
        raise RuntimeError("Selenium driver is required for odds extraction.")
    driver.get(url)
    if wait_css:
        try:
            def has_numeric_text(_driver):
                elems = _driver.find_elements(By.CSS_SELECTOR, wait_css)
                for el in elems:
                    if re.search(r"\d", el.text or ""):
                        return True
                return False

            WebDriverWait(driver, timeout).until(has_numeric_text)
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


def main():
    url = input("Race URL: ").strip()
    if not url:
        print("No URL provided.")
        return

    options = Options()
    debugger_address = os.environ.get("CHROME_DEBUGGER_ADDRESS", "").strip()
    shared_driver = bool(debugger_address)
    if shared_driver:
        options.add_experimental_option("debuggerAddress", debugger_address)
    else:
        if should_headless():
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
        options.add_argument("--lang=ja-JP")

    try:
        driver = webdriver.Chrome(options=options)
    except Exception as exc:
        print(f"[ERROR] Selenium unavailable: {exc}")
        return
    try:
        if should_inject_cookies():
            cookies = load_cookies_from_json_file("cookie.txt")
            driver.get("https://db.netkeiba.com")
            assert_not_blocked(driver.page_source, driver.title or "", "https://db.netkeiba.com")
            for cookie in cookies:
                driver.add_cookie(cookie)
        else:
            print("Skipping cookie injection (PIPELINE_SKIP_COOKIE_INJECTION=1).")
        tan_url = build_tan_odds_url(url)
        fuku_results = []
        if tan_url:
            sleep_jitter()
            page = get_page_source(tan_url, driver, wait_css="span[id^='odds-']")
            results = safe_parse("tan", parse_tan_odds_from_page, page)
            fuku_results = safe_parse("fuku", parse_fuku_odds_from_page, page)
            if not results:
                results = safe_parse("odds", parse_odds_from_page, page)
        else:
            sleep_jitter()
            page = get_page_source(url, driver, wait_css="span[id^='odds-']")
            results = safe_parse("odds", parse_odds_from_page, page)
            if not results:
                results = safe_parse("tan", parse_tan_odds_from_page, page)
            fuku_results = safe_parse("fuku", parse_fuku_odds_from_page, page)
        if not results:
            print("No odds found.")
            return

        for item in results:
            print(f"{item['horse_no']}\t{item['name']}\t{item['odds']}")

        out_path = "odds.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["horse_no", "name", "odds"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved: {out_path}")

        if fuku_results:
            fuku_path = "fuku_odds.csv"
            with open(fuku_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["horse_no", "name", "odds_low", "odds_high", "odds_mid"],
                )
                writer.writeheader()
                writer.writerows(fuku_results)
            print(f"Saved: {fuku_path}")

        wide_url = build_wide_odds_url(url)
        if wide_url:
            sleep_jitter()
            wide_results = safe_parse(
                "wide",
                parse_wide_odds_from_page,
                get_page_source(wide_url, driver, wait_css="span[id^='odds-5-']"),
            )
            if wide_results:
                wide_path = "wide_odds.csv"
                with open(wide_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["horse_no_a", "horse_no_b", "odds_low", "odds_high", "odds_mid"],
                    )
                    writer.writeheader()
                    writer.writerows(wide_results)
                print(f"Saved: {wide_path}")

        quinella_url = build_quinella_odds_url(url)
        if quinella_url:
            sleep_jitter()
            quinella_results = safe_parse(
                "quinella",
                parse_quinella_odds_from_page,
                get_page_source(quinella_url, driver, wait_css="span[id^='odds-4-']"),
            )
            if quinella_results:
                quinella_path = "quinella_odds.csv"
                with open(quinella_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["horse_no_a", "horse_no_b", "odds"],
                    )
                    writer.writeheader()
                    writer.writerows(quinella_results)
                print(f"Saved: {quinella_path}")
    finally:
        if driver and not shared_driver:
            driver.quit()


if __name__ == "__main__":
    main()

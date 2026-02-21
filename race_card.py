import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import random
import time
import re
import json
import sys
from urllib.parse import urljoin

def configure_utf8_io():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_io()

HORSE_LINK_PATTERN = re.compile(r"/horse/\d+/?")


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
SLEEP_RANGE_SECONDS = (0.6, 1.6)
# ===== 从 cookie.txt 加载 JSON 格式的 cookie =====
def load_cookies_from_json_file(path="cookie.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cookie_json = json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    cookies = []
    for item in cookie_json:
        if "name" in item and "value" in item:
            cookies.append({
                "name": item["name"],
                "value": item["value"]
            })
    return cookies


def assert_not_blocked(driver, url):
    page_source = driver.page_source or ""
    title = driver.title or ""
    haystack = f"{title}\n{page_source}".lower()
    for pattern in BLOCK_PATTERNS:
        if pattern in haystack:
            print(f"[ERROR] Access blocked (pattern={pattern}) url={url}")
            raise SystemExit(2)


def sleep_jitter():
    time.sleep(random.uniform(*SLEEP_RANGE_SECONDS))


def wait_for_horse_rows(driver, timeout=12):
    selectors = [
        "table.RaceTable01.ShutubaTable tbody tr.HorseList",
        "table.ShutubaTable tbody tr.HorseList",
        "table.Shutuba_Table tbody tr.HorseList",
        "table.RaceTable01 tbody tr.HorseList",
        "tr.HorseList",
    ]
    end_time = time.time() + timeout
    while time.time() < end_time:
        for selector in selectors:
            if driver.find_elements(By.CSS_SELECTOR, selector):
                return selector
        time.sleep(0.25)
    return ""


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

# ===== 用户输入 URL =====
url = input("Race URL: ")
trigger_race_name = os.environ.get("TRIGGER_RACE", "").strip()
exclude_trigger_race = os.environ.get("RACE_CARD_EXCLUDE_TRIGGER", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

debug_enabled = os.environ.get("RACE_CARD_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

# ===== 启动 Selenium 浏览器 =====
chrome_options = Options()
debugger_address = os.environ.get("CHROME_DEBUGGER_ADDRESS", "").strip()
shared_driver = bool(debugger_address)
if shared_driver:
    chrome_options.add_experimental_option("debuggerAddress", debugger_address)
else:
    if should_headless():
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--lang=ja-JP")
driver = webdriver.Chrome(options=chrome_options)

# ===== 注入 cookie（先访问一次主域名）=====
if should_inject_cookies():
    cookies = load_cookies_from_json_file("cookie.txt")
    if cookies:
        driver.get("https://db.netkeiba.com")
        assert_not_blocked(driver, "https://db.netkeiba.com")
        for cookie in cookies:
            driver.add_cookie(cookie)
        print("Loaded login cookies.")
    else:
        print("No cookie file found; continue without cookies.")
else:
    print("Skipping cookie injection (PIPELINE_SKIP_COOKIE_INJECTION=1).")

# ===== 打开出走马页面 =====
driver.get(url)
wait_selector = wait_for_horse_rows(driver)
if debug_enabled:
    if wait_selector:
        print(f"DEBUG: horse rows detected via {wait_selector}")
    else:
        print("DEBUG: horse rows not detected before parse.")
assert_not_blocked(driver, url)
soup = BeautifulSoup(driver.page_source, 'html.parser')

# 提取比赛名称
race_name_element = (
    soup.find("div", {"class": "RaceName"})
    or soup.find("h1", {"class": "RaceName"})
    or soup.select_one(".RaceData01 .RaceName")
)
race_name = race_name_element.text.strip() if race_name_element else "未知比赛"
print(f"Fetching race data: {race_name}")

# 日期和文件名
current_date = datetime.now().strftime('%Y%m%d')
file_name = f'{current_date}_{race_name}.csv'

# 获取出走马信息
def select_horse_rows(soup):
    selectors = [
        "table.RaceTable01.ShutubaTable tbody tr.HorseList",
        "table.ShutubaTable tbody tr.HorseList",
        "table.Shutuba_Table tbody tr.HorseList",
        "table.RaceTable01 tbody tr.HorseList",
    ]
    for selector in selectors:
        rows = soup.select(selector)
        if rows:
            return rows
    return soup.select("tr.HorseList")


horse_rows = select_horse_rows(soup)
print(f"Horse rows found: {len(horse_rows)}")

def normalize_horse_url(href):
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        href = f"https:{href}"
    if href.startswith("/"):
        href = urljoin("https://db.netkeiba.com", href)
    match = re.search(r"/horse/(\d+)", href)
    if match:
        return f"https://db.netkeiba.com/horse/{match.group(1)}"
    return href.rstrip("/")


def extract_horse_link(row):
    link_el = row.select_one('a[href*="/horse/"]')
    if link_el:
        name = link_el.get_text(strip=True)
        return name, normalize_horse_url(link_el.get("href", ""))
    name_el = row.select_one(".HorseName") or row.select_one(".Horse_Name")
    if name_el:
        name = name_el.get_text(strip=True)
        link_el = name_el.find("a", href=True)
        if link_el:
            return name, normalize_horse_url(link_el.get("href", ""))
    return "", ""


def parse_sex_age(text):
    if not text:
        return "", None, ""
    cleaned = re.sub(r"\s+", "", str(text))
    match = re.search(r"([牡牝セ騸])(\d+)", cleaned)
    if not match:
        return "", None, ""
    sex = match.group(1)
    age = int(match.group(2))
    return sex, age, f"{sex}{age}"

BIRTHDATE_RE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")
RACE_DATE_RE = re.compile(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})")
JOCKEY_ID_RE = re.compile(r"/jockey/result/recent/([^/]+)/?")


def extract_jockey_ids_from_table(table):
    if table is None:
        return []
    ids = []
    for row in table.find_all("tr"):
        if row.find("th"):
            continue
        link = row.find("a", href=JOCKEY_ID_RE)
        if link:
            match = JOCKEY_ID_RE.search(link.get("href", ""))
            if match:
                ids.append(match.group(1))
                continue
        ids.append("")
    return ids


def build_jockey_id_series(table, df):
    ids = extract_jockey_ids_from_table(table)
    if len(ids) < len(df):
        ids.extend([""] * (len(df) - len(ids)))
    elif len(ids) > len(df):
        ids = ids[:len(df)]
    return pd.Series(ids, index=df.index)


def insert_jockey_id_column(df, jockey_series):
    if jockey_series is None:
        return
    if "JockeyId" in df.columns:
        df.drop(columns=["JockeyId"], inplace=True)
    insert_at = len(df.columns)
    for col in ("騎手", "Jockey"):
        if col in df.columns:
            insert_at = df.columns.get_loc(col) + 1
            break
    df.insert(insert_at, "JockeyId", jockey_series.reindex(df.index).tolist())


def insert_current_jockey_id_column(df, jockey_id):
    if "JockeyId_current" in df.columns:
        df.drop(columns=["JockeyId_current"], inplace=True)
    insert_at = len(df.columns)
    for col in ("JockeyId", "騎手", "Jockey"):
        if col in df.columns:
            insert_at = df.columns.get_loc(col) + 1
            break
    df.insert(insert_at, "JockeyId_current", [jockey_id or ""] * len(df))


def extract_birthdate(soup):
    th = soup.find("th", string=re.compile(r"生年月日"))
    if not th:
        return None
    td = th.find_next_sibling("td")
    if not td:
        return None
    text = td.get_text(strip=True)
    match = BIRTHDATE_RE.search(text)
    if not match:
        return None
    try:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        ).date()
    except ValueError:
        return None


def parse_race_date(value):
    if not value:
        return None
    match = RACE_DATE_RE.search(str(value))
    if not match:
        return None
    try:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        ).date()
    except ValueError:
        return None


def compute_age(birthdate, race_date):
    if not birthdate or not race_date:
        return None
    age = race_date.year - birthdate.year
    if (race_date.month, race_date.day) < (birthdate.month, birthdate.day):
        age -= 1
    return age if age >= 0 else None


def normalize_sex(sex, sex_age):
    if sex:
        return sex
    if sex_age:
        text = str(sex_age).strip()
        return text[:1] if text else ""
    return ""


def compute_age_series(df, birthdate):
    if birthdate is None or "日付" not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    return df["日付"].apply(lambda x: compute_age(birthdate, parse_race_date(x)))


def format_sex_age(sex, age):
    if not sex or age is None or pd.isna(age):
        return ""
    try:
        age_i = int(age)
    except (TypeError, ValueError):
        return ""
    return f"{sex}{age_i}"


def upsert_column(df, idx, name, value):
    if name in df.columns:
        df.drop(columns=[name], inplace=True)
    df.insert(idx, name, value)


def extract_sex_age(row):
    candidates = []
    for selector in ("td.Barei", ".Barei", "span.Age"):
        el = row.select_one(selector)
        if el:
            candidates.append(el.get_text(strip=True))
    if not candidates:
        candidates.append(row.get_text(" ", strip=True))
    for raw in candidates:
        sex, age, sex_age = parse_sex_age(raw)
        if sex or age is not None:
            return sex, age, sex_age
    return "", None, ""


def extract_jockey_id_from_row(row):
    link = row.select_one('td.Jockey a[href*="/jockey/result/recent/"]')
    if link:
        match = JOCKEY_ID_RE.search(link.get("href", ""))
        if match:
            return match.group(1)
    return ""


horses_info = {}
skipped_cancel = 0
skipped_missing = 0
missing_sex_age = 0
horse_index = 1
seen_horses = set()
for row in horse_rows:
    row_text = row.get_text(" ", strip=True)
    if (
        row.select_one(".Cancel_Txt")
        or "Cancel" in (row.get("class") or [])
        or "除外" in row_text
        or "取消" in row_text
        or "中止" in row_text
    ):
        skipped_cancel += 1
        continue
    name, href = extract_horse_link(row)
    if not name or not href:
        skipped_missing += 1
        if debug_enabled and skipped_missing <= 3:
            print(f"DEBUG: missing horse link row_text={row_text}")
        continue
    sex, age, sex_age = extract_sex_age(row)
    jockey_id = extract_jockey_id_from_row(row)
    if not sex_age:
        missing_sex_age += 1
        if debug_enabled and missing_sex_age <= 3:
            print(f"DEBUG: missing sex/age for {name} row_text={row_text}")
    horse_id = ""
    if href:
        match = re.search(r"/horse/(\d+)", href)
        if match:
            horse_id = match.group(1)
    key = horse_id or href or name
    if key in seen_horses:
        continue
    seen_horses.add(key)
    horses_info[f"Horse {horse_index}"] = {
        "name": name,
        "url": href,
        "sex": sex,
        "age": age,
        "sex_age": sex_age,
        "jockey_id": jockey_id,
    }
    print(f"Horse {horse_index}: Name - {name}, URL - {href}")
    horse_index += 1

if skipped_cancel:
    print(f"Skipped canceled horses: {skipped_cancel}")
if skipped_missing and debug_enabled:
    print(f"DEBUG: rows missing horse link/name: {skipped_missing}")
if missing_sex_age and debug_enabled:
    print(f"DEBUG: rows missing sex/age: {missing_sex_age}")
if not horses_info:
    print("WARN: No horses parsed from race card. Page layout may have changed.")

# ===== 遍历每匹马并抓取表格 =====
all_frames = []

def extract_kanji(text):
    return ''.join(re.findall(r'[一-鿿]+', text))

for horse_number, horse_data in horses_info.items():
    print(f"\nFetching data for {horse_data['name']}...")

    driver.get(horse_data["url"])
    assert_not_blocked(driver, horse_data["url"])
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "db_h_race_results"))
        )
    except:
        print(f"WARN: {horse_data['name']} results table not found")
        continue

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    birthdate = extract_birthdate(soup)
    table = soup.find('table', {'class': 'db_h_race_results nk_tb_common'})
    if table is None:
        print(f"WARN: {horse_data['name']} table is None")
        continue

    try:
        df = pd.read_html(str(table), header=0)[0]
    except Exception as e:
        print(f"ERROR: {horse_data['name']} table parse failed: {e}")
        continue
    jockey_series = build_jockey_id_series(table, df)

    if exclude_trigger_race and trigger_race_name:
        if "レース名" not in df.columns:
            print(f"WARN: {horse_data['name']} missing race name column.")
            continue
        race_names = df["レース名"].astype(str).fillna("")
        match = race_names.str.contains(re.escape(trigger_race_name), na=False)
        if match.any():
            match_pos = match.to_numpy().nonzero()[0][0]
            start_pos = match_pos + 1
            if start_pos >= len(df):
                print(f"WARN: {horse_data['name']} has no rows after trigger race.")
                continue
            df = df.iloc[start_pos:]
            race_names = df["レース名"].astype(str).fillna("")
            df = df.loc[
                ~race_names.str.contains(re.escape(trigger_race_name), na=False)
            ]
            if df.empty:
                print(f"WARN: {horse_data['name']} has no rows after trigger race.")
                continue
            jockey_series = jockey_series.loc[df.index]
        else:
            print(f"WARN: {horse_data['name']} did not run {trigger_race_name}. Skipping.")
            continue

    # 过滤 'ﾀｲﾑ指数' 列（如存在）
    if 'ﾀｲﾑ指数' in df.columns:
        df = df[df['ﾀｲﾑ指数'].notnull() & (df['ﾀｲﾑ指数'] != 0)]
        jockey_series = jockey_series.loc[df.index]
    else:
        print(f"WARN: {horse_data['name']} missing TimeIndex column")

    if df.empty:
        print(f"WARN: {horse_data['name']} has no valid race data")
        continue

    sex_value = normalize_sex(horse_data.get("sex", ""), horse_data.get("sex_age", ""))
    age_series = compute_age_series(df, birthdate)
    sex_age_series = age_series.apply(lambda a: format_sex_age(sex_value, a))

    df.drop(columns=["Sex", "Age"], errors="ignore", inplace=True)
    upsert_column(df, 0, "HorseName", horse_data["name"])
    upsert_column(df, 1, "SexAge", sex_age_series)
    insert_jockey_id_column(df, jockey_series)
    insert_current_jockey_id_column(df, horse_data.get("jockey_id", ""))
    all_frames.append(df)
    print(f"OK: {horse_data['name']} data extracted.")
    sleep_jitter()

# ===== 写入 CSV =====
if all_frames:
    all_data = pd.concat(all_frames, ignore_index=True)
    all_data.to_csv("shutuba.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved output CSV (rows={len(all_data)}).")
else:
    print("\nWARN: No valid data collected.")

# ===== 关闭浏览器 =====
if not shared_driver:
    driver.quit()

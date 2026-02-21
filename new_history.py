from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re
import json
import random
import sys
from urllib.parse import urljoin
from datetime import datetime

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


def configure_utf8_io():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_io()
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
JOCKEY_ID_RE = re.compile(r"/jockey/result/recent/([^/]+)/")


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


def parse_race_year_from_row(tr, race_href):
    date_text = ""
    date_link = tr.find("a", href=re.compile(r"^/race/list/\d{8}/$"))
    if date_link:
        date_text = date_link.get_text(strip=True)
    else:
        tds = tr.find_all("td")
        if tds:
            date_text = tds[0].get_text(strip=True)
    match = re.search(r"(\d{4})[./-]\d{2}[./-]\d{2}", date_text)
    if match:
        return int(match.group(1))
    race_id_match = re.search(r"/race/(\d{12})/", race_href)
    if race_id_match:
        return int(race_id_match.group(1)[:4])
    return None


def parse_race_date_from_row(tr):
    date_text = ""
    date_link = tr.find("a", href=re.compile(r"^/race/list/\d{8}/$"))
    if date_link:
        date_text = date_link.get_text(strip=True)
    else:
        tds = tr.find_all("td")
        if tds:
            date_text = tds[0].get_text(strip=True)
    return parse_race_date(date_text)


def find_headcount_index(soup):
    for tr in soup.find_all("tr"):
        ths = tr.find_all("th")
        if not ths:
            continue
        header_texts = [th.get_text(strip=True) for th in ths]
        for idx, text in enumerate(header_texts):
            if "頭数" in text:
                return idx
    return None


def parse_headcount_from_row(tr, headcount_index):
    if headcount_index is None:
        return None
    tds = tr.find_all("td")
    if headcount_index >= len(tds):
        return None
    text = tds[headcount_index].get_text(strip=True)
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    return int(digits)


def find_new_results_table(soup):
    for t in soup.find_all("table", class_=lambda x: x and "nk_tb_common" in x):
        header = t.find("tr")
        if not header:
            continue
        th_texts = [th.get_text(strip=True) for th in header.find_all("th")]
        if not th_texts:
            continue
        if any("レース名" in text for text in th_texts) and any("日付" in text for text in th_texts):
            return t
    return None


def find_date_column(df):
    for col in df.columns:
        if "日付" in str(col):
            return col
    for col in df.columns:
        series = df[col].astype(str)
        for value in series.head(5):
            if parse_race_date(value):
                return col
    return None


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

# ===== 用户输入 =====
url = input("History search URL: ")

# ===== 启动 Selenium =====
options = Options()
debugger_address = os.environ.get("CHROME_DEBUGGER_ADDRESS", "").strip()
shared_driver = bool(debugger_address)
if shared_driver:
    options.add_experimental_option("debuggerAddress", debugger_address)
else:
    if should_headless():
        options.add_argument("--headless")
    options.add_argument("--lang=ja-JP")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

# ===== 注入 cookie（需先访问主域）=====
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

# ===== 打开目标列表页 =====
driver.get(url)
assert_not_blocked(driver, url)
sleep_jitter()
soup = BeautifulSoup(driver.page_source, 'html.parser')

# ===== 解析前20场比赛 =====
race_links = []
seen_race_urls = set()
headcount_index = find_headcount_index(soup)
for tr in soup.find_all("tr"):
    race_a = tr.find("a", href=re.compile(r"^/race/\d{12}/$"))
    if not race_a:
        continue
    headcount = parse_headcount_from_row(tr, headcount_index)
    if headcount == 0:
        race_name = race_a.text.strip()
        date_text = ""
        date_link = tr.find("a", href=re.compile(r"^/race/list/\d{8}/$"))
        if date_link:
            date_text = date_link.get_text(strip=True)
        print(f"Skip race with headcount 0: {date_text} {race_name}")
        continue
    href = race_a.get("href", "")
    race_year = parse_race_year_from_row(tr, href)
    if race_year is not None and race_year < 2007:
        print(f"Reached races before 2007 (year={race_year}); stopping list scan.")
        break
    race_name = race_a.text.strip()
    full_url = urljoin("https://db.netkeiba.com/", href)
    if full_url not in seen_race_urls:
        race_date = parse_race_date_from_row(tr)
        race_links.append((race_name, full_url, race_date))
        seen_race_urls.add(full_url)
    if len(race_links) >= 20:
        break

if not race_links:
    print("WARN: No race links found. Check URL or login status.")
    driver.quit()
    exit()

print(f"Found {len(race_links)} races.")

# ===== 用于存储所有马匹数据 =====
all_frames = []

# ===== 主循环 =====
for race_idx, (race_name, race_url, race_date) in enumerate(race_links, start=1):
    if race_date is None:
        print(f"\n>>> Processing race {race_idx}/{len(race_links)}: {race_name}")
        print("WARN: Race date not found; skipping.")
        continue
    race_date_str = race_date.strftime("%Y/%m/%d")
    print(f"\n>>> Processing race {race_idx}/{len(race_links)}: {race_name} ({race_date_str})")
    race_id_match = re.search(r"/race/(\d{12})/", race_url)
    race_id = race_id_match.group(1) if race_id_match else ""
    driver.get(race_url)
    assert_not_blocked(driver, race_url)
    sleep_jitter()
    soup_race = BeautifulSoup(driver.page_source, 'html.parser')

    # 找到比赛表格并取前3匹马
    race_table = soup_race.find("table", class_="race_table_01 nk_tb_common")
    if race_table is None:
        print("WARN: Entry table not found. Skipping race.")
        continue

    sex_age_col = None
    header_cells = []
    for tr in race_table.find_all("tr"):
        ths = tr.find_all("th")
        if ths:
            header_cells = ths
            break
    if header_cells:
        header_texts = [th.get_text(strip=True) for th in header_cells]
        for idx, text in enumerate(header_texts):
            if re.search(r"性齢|性龄|性令", text):
                sex_age_col = idx
                break

    horse_links = []
    rows = race_table.find_all("tr")[1:]
    for row_idx, tr in enumerate(rows, start=1):
        a_h = tr.find("a", href=re.compile(r"^/horse/\d{10}/$"))
        if a_h:
            horse_name = a_h.text.strip()
            horse_url = urljoin("https://db.netkeiba.com/", a_h["href"])
            finish_pos = row_idx
            is_top3 = 1 if finish_pos <= 3 else 0
            sex = ""
            age = None
            sex_age = ""
            if sex_age_col is not None:
                tds = tr.find_all("td")
                if sex_age_col < len(tds):
                    raw_sex_age = tds[sex_age_col].get_text(strip=True)
                    sex, age, sex_age = parse_sex_age(raw_sex_age)
            if not sex_age:
                sex, age, sex_age = parse_sex_age(tr.get_text(" ", strip=True))
            horse_links.append((horse_name, horse_url, finish_pos, is_top3, sex, age, sex_age))

    if not horse_links:
        print("WARN: No horses found. Skipping race.")
        continue

    # 抓每匹马的战绩表
    for h_idx, (horse_name, horse_url, finish_pos, is_top3, sex, age, sex_age) in enumerate(horse_links, start=1):
        print(f"  - Horse {h_idx}: {horse_name}")
        driver.get(horse_url)
        assert_not_blocked(driver, horse_url)
        sleep_jitter()
        soup_horse = BeautifulSoup(driver.page_source, 'html.parser')
        birthdate = extract_birthdate(soup_horse)

        # 查找新版战绩表（以列名“レース名”为标志）
        table = find_new_results_table(soup_horse)

        if table is None:
            print(f"    WARN: {horse_name} results table not found")
            continue

        try:
            df = pd.read_html(str(table), header=0)[0]
        except Exception as e:
            print(f"    ERROR: Failed to parse {horse_name} table: {e}")
            continue
        jockey_series = build_jockey_id_series(table, df)

        date_col = find_date_column(df)
        if not date_col:
            print(f"    WARN: {horse_name} date column not found. Skipping.")
            continue
        date_series = df[date_col].apply(parse_race_date)
        match = date_series == race_date
        if match.any():
            start_index = match[match].index[0]
            df = df.loc[start_index:]
            jockey_series = jockey_series.loc[df.index]
        else:
            print(f"    WARN: {horse_name} no race on {race_date_str}. Skipping.")
            continue

        # 过滤有效“ﾀｲﾑ指数”
        if 'ﾀｲﾑ指数' in df.columns:
            df = df[df['ﾀｲﾑ指数'].notnull() & (df['ﾀｲﾑ指数'] != 0)]
            jockey_series = jockey_series.loc[df.index]
        else:
            print(f"    WARN: {horse_name} missing TimeIndex column.")
            continue

        if df.empty:
            print(f"    WARN: {horse_name} has no valid data.")
            continue

        # 加入马名列并删除レース名
        sex_value = normalize_sex(sex, sex_age)
        age_series = compute_age_series(df, birthdate)
        sex_age_series = age_series.apply(lambda a: format_sex_age(sex_value, a))

        df.drop(columns=["Sex", "Age"], errors="ignore", inplace=True)
        upsert_column(df, 0, "race_id", race_id)
        upsert_column(df, 1, "HorseName", horse_name)
        upsert_column(df, 2, "SexAge", sex_age_series)
        insert_jockey_id_column(df, jockey_series)
        df.insert(3, 'finish_pos', finish_pos)
        df.insert(4, 'is_top3', is_top3)
        #df = df.drop(columns=['レース名'], errors='ignore')

        all_frames.append(df)
        print(f"    OK: {horse_name} data extracted.")
        sleep_jitter()

# ===== 保存结果 =====
if all_frames:
    all_data = pd.concat(all_frames, ignore_index=True)
    filename = f"kachiuma.csv"
    all_data.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\nSaved output CSV: {filename} (rows={len(all_data)}).")
else:
    print("\nWARN: No valid data collected.")
    print("\nWARN: No valid data collected.")

if not shared_driver:
    driver.quit()

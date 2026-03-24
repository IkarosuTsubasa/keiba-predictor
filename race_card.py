import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import random
import time
import re
import sys
from urllib.parse import urljoin, urlparse

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
PAGE_LOAD_STRATEGY = os.environ.get("PIPELINE_PAGE_LOAD_STRATEGY", "eager").strip().lower() or "eager"
PAGE_LOAD_TIMEOUT_SECONDS = 15.0
PAGE_WAIT_TIMEOUT_SECONDS = 10.0


def get_env_int(name, default, minimum=1, maximum=None):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    if value < minimum:
        value = minimum
    if maximum is not None:
        value = min(value, maximum)
    return int(value)


def get_sleep_range():
    min_raw = os.environ.get("PIPELINE_HORSE_DELAY_MIN", "").strip()
    max_raw = os.environ.get("PIPELINE_HORSE_DELAY_MAX", "").strip()
    try:
        min_value = float(min_raw) if min_raw else float(SLEEP_RANGE_SECONDS[0])
    except ValueError:
        min_value = float(SLEEP_RANGE_SECONDS[0])
    try:
        max_value = float(max_raw) if max_raw else float(SLEEP_RANGE_SECONDS[1])
    except ValueError:
        max_value = float(SLEEP_RANGE_SECONDS[1])
    min_value = max(0.0, min_value)
    max_value = max(min_value, max_value)
    return min_value, max_value


def assert_not_blocked(driver, url):
    page_source = driver.page_source or ""
    title = driver.title or ""
    haystack = f"{title}\n{page_source}".lower()
    for pattern in BLOCK_PATTERNS:
        if pattern in haystack:
            print(f"[ERROR] Access blocked (pattern={pattern}) url={url}")
            raise SystemExit(2)


def sleep_jitter():
    sleep_min, sleep_max = get_sleep_range()
    if sleep_max <= 0:
        return
    time.sleep(random.uniform(sleep_min, sleep_max))


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


def expand_cell_texts(cells):
    values = []
    for cell in cells:
        text = cell.get_text(" ", strip=True)
        colspan_raw = cell.get("colspan", 1)
        try:
            colspan = max(1, int(colspan_raw))
        except (TypeError, ValueError):
            colspan = 1
        values.extend([text] * colspan)
    return values


def normalize_header_text(text):
    return re.sub(r"\s+", "", str(text or "")).strip()


def table_to_dataframe(table):
    header = []
    rows = []
    for tr in table.find_all("tr"):
        ths = tr.find_all("th")
        tds = tr.find_all("td")
        if ths and not header:
            header = [normalize_header_text(x) for x in expand_cell_texts(ths)]
            continue
        if not tds:
            continue
        row = expand_cell_texts(tds)
        if header:
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))
            elif len(row) > len(header):
                row = row[: len(header)]
        rows.append(row)
    if not header and rows:
        width = max(len(row) for row in rows)
        header = [f"col_{idx}" for idx in range(width)]
        rows = [row + [""] * (width - len(row)) for row in rows]
    if not header:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=header)


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
                stop_loading(driver)
                return selector
        time.sleep(0.25)
    return ""


def get_page_source_fast(driver, url, wait_css=None, timeout=None, require_numeric=False):
    timeout = timeout if timeout is not None else get_env_float("PIPELINE_PAGE_WAIT_TIMEOUT", PAGE_WAIT_TIMEOUT_SECONDS)
    try:
        driver.get(url)
    except TimeoutException:
        print(f"[WARN] Page load timeout; continue with partial DOM: {url}")
        stop_loading(driver)
    if wait_css:
        try:
            if require_numeric:
                def has_numeric_text(_driver):
                    elems = _driver.find_elements(By.CSS_SELECTOR, wait_css)
                    for el in elems:
                        if re.search(r"\d", el.text or ""):
                            return True
                    return False
                WebDriverWait(driver, timeout).until(has_numeric_text)
            else:
                WebDriverWait(driver, timeout).until(
                    lambda d: bool(d.find_elements(By.CSS_SELECTOR, wait_css))
                )
            stop_loading(driver)
        except Exception:
            print(f"[WARN] Timeout waiting for selector: {wait_css}")
    return driver.page_source


def should_headless():
    raw = os.environ.get("PIPELINE_HEADLESS", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def create_driver(allow_shared=True):
    chrome_options = Options()
    if PAGE_LOAD_STRATEGY in ("normal", "eager", "none"):
        chrome_options.page_load_strategy = PAGE_LOAD_STRATEGY
    if allow_shared and debugger_address:
        chrome_options.add_experimental_option("debuggerAddress", debugger_address)
    else:
        if should_headless():
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--lang=ja-JP")
    created_driver = webdriver.Chrome(options=chrome_options)
    try:
        page_load_timeout = get_env_float("PIPELINE_PAGE_LOAD_TIMEOUT", PAGE_LOAD_TIMEOUT_SECONDS)
        created_driver.set_page_load_timeout(page_load_timeout)
    except Exception:
        pass
    return created_driver


def split_work_items(items, workers):
    groups = [[] for _ in range(max(1, workers))]
    for idx, item in enumerate(items):
        groups[idx % len(groups)].append(item)
    return [group for group in groups if group]


def capture_netkeiba_cookies(active_driver):
    cookies = []
    allowed_keys = ("name", "value", "path", "domain", "secure", "httpOnly", "expiry", "sameSite")
    try:
        raw_cookies = active_driver.get_cookies()
    except Exception:
        return cookies
    for cookie in raw_cookies:
        domain = str(cookie.get("domain", "") or "").lstrip(".").lower()
        if not domain.endswith("netkeiba.com"):
            continue
        cookies.append({key: cookie[key] for key in allowed_keys if key in cookie and cookie[key] is not None})
    return cookies


def apply_cookies_for_url(active_driver, target_url, cookies):
    if not cookies:
        return
    parsed = urlparse(target_url)
    host = str(parsed.hostname or "").lower()
    if not host:
        return
    base_url = f"{parsed.scheme or 'https'}://{host}/"
    try:
        active_driver.get(base_url)
    except Exception:
        return
    for cookie in cookies:
        domain = str(cookie.get("domain", "") or "").lstrip(".").lower()
        if domain and host != domain and not host.endswith(f".{domain}"):
            continue
        try:
            active_driver.add_cookie(cookie)
        except Exception:
            continue

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
debugger_address = os.environ.get("CHROME_DEBUGGER_ADDRESS", "").strip()
shared_driver = bool(debugger_address)
driver = create_driver(allow_shared=True)

# ===== 打开出走马页面 =====
get_page_source_fast(driver, url)
wait_selector = wait_for_horse_rows(
    driver,
    timeout=get_env_float("PIPELINE_PAGE_WAIT_TIMEOUT", PAGE_WAIT_TIMEOUT_SECONDS),
)
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


def find_matching_column(df, candidates=(), contains=()):
    columns = list(df.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for col in columns:
        text = str(col)
        for token in contains:
            if token and token in text:
                return col
    return ""


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
horse_fetch_workers = get_env_int("PIPELINE_HORSE_FETCH_WORKERS", 1, minimum=1, maximum=8)


def fetch_race_card_horse_frame(active_driver, horse_data):
    logs = [f"\nFetching data for {horse_data['name']}..."]
    page = get_page_source_fast(
        active_driver,
        horse_data["url"],
        wait_css=".db_h_race_results",
    )
    assert_not_blocked(active_driver, horse_data["url"])
    if not active_driver.find_elements(By.CLASS_NAME, "db_h_race_results"):
        logs.append(f"WARN: {horse_data['name']} results table not found")
        return None, logs

    soup = BeautifulSoup(page, 'html.parser')
    birthdate = extract_birthdate(soup)
    table = soup.find('table', {'class': 'db_h_race_results nk_tb_common'})
    if table is None:
        logs.append(f"WARN: {horse_data['name']} table is None")
        return None, logs

    try:
        df = table_to_dataframe(table)
        if df.empty:
            raise ValueError("parsed table is empty")
    except Exception as e:
        logs.append(f"ERROR: {horse_data['name']} table parse failed: {e}")
        return None, logs
    jockey_series = build_jockey_id_series(table, df)

    race_name_col = find_matching_column(
        df,
        candidates=("レース名", "繝ｬ繝ｼ繧ｹ蜷・"),
        contains=("レース",),
    )
    if exclude_trigger_race and trigger_race_name:
        if race_name_col not in df.columns:
            logs.append(f"WARN: {horse_data['name']} missing race name column.")
            return None, logs
        race_names = df[race_name_col].astype(str).fillna("")
        match = race_names.str.contains(re.escape(trigger_race_name), na=False)
        if match.any():
            match_pos = match.to_numpy().nonzero()[0][0]
            start_pos = match_pos + 1
            if start_pos >= len(df):
                logs.append(f"WARN: {horse_data['name']} has no rows after trigger race.")
                return None, logs
            df = df.iloc[start_pos:]
            race_names = df[race_name_col].astype(str).fillna("")
            df = df.loc[
                ~race_names.str.contains(re.escape(trigger_race_name), na=False)
            ]
            if df.empty:
                logs.append(f"WARN: {horse_data['name']} has no rows after trigger race.")
                return None, logs
            jockey_series = jockey_series.loc[df.index]
        else:
            logs.append(f"WARN: {horse_data['name']} did not run {trigger_race_name}. Skipping.")
            return None, logs

    time_index_col = find_matching_column(
        df,
        candidates=("ﾀｲﾑ指数", "タイム指数", "TimeIndex"),
        contains=("指数",),
    )
    if time_index_col:
        df = df[df[time_index_col].notnull() & (df[time_index_col] != 0)]
        jockey_series = jockey_series.loc[df.index]
    else:
        logs.append(f"WARN: {horse_data['name']} missing TimeIndex column. columns={list(df.columns)}")

    if df.empty:
        logs.append(f"WARN: {horse_data['name']} has no valid race data")
        return None, logs

    sex_value = normalize_sex(horse_data.get("sex", ""), horse_data.get("sex_age", ""))
    age_series = compute_age_series(df, birthdate)
    sex_age_series = age_series.apply(lambda a: format_sex_age(sex_value, a))

    df.drop(columns=["Sex", "Age"], errors="ignore", inplace=True)
    upsert_column(df, 0, "HorseName", horse_data["name"])
    upsert_column(df, 1, "SexAge", sex_age_series)
    insert_jockey_id_column(df, jockey_series)
    insert_current_jockey_id_column(df, horse_data.get("jockey_id", ""))
    logs.append(f"OK: {horse_data['name']} data extracted.")
    sleep_jitter()
    return df, logs


def process_race_card_horse_batch(items, session_cookies):
    batch_results = []
    batch_logs = []
    batch_driver = create_driver(allow_shared=False)
    try:
        first_target_url = items[0][1]["url"] if items else ""
        if first_target_url:
            apply_cookies_for_url(batch_driver, first_target_url, session_cookies)
        for order, horse_data in items:
            frame, logs = fetch_race_card_horse_frame(batch_driver, horse_data)
            batch_results.append((order, frame))
            batch_logs.extend(logs)
    finally:
        batch_driver.quit()
    return batch_results, batch_logs


horse_items = list(enumerate(horses_info.values(), start=1))
effective_workers = min(horse_fetch_workers, len(horse_items))
if effective_workers > 1:
    print(f"Parallel horse fetch enabled: workers={effective_workers}")
    session_cookies = capture_netkeiba_cookies(driver)
    if not session_cookies and horse_items:
        get_page_source_fast(
            driver,
            horse_items[0][1]["url"],
            wait_css=".db_h_race_results",
        )
        assert_not_blocked(driver, horse_items[0][1]["url"])
        session_cookies = capture_netkeiba_cookies(driver)
    grouped_items = split_work_items(horse_items, effective_workers)
    parallel_results = []
    parallel_logs = []
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [executor.submit(process_race_card_horse_batch, group, session_cookies) for group in grouped_items]
        for future in as_completed(futures):
            batch_results, batch_logs = future.result()
            parallel_results.extend(batch_results)
            parallel_logs.extend(batch_logs)
    for line in parallel_logs:
        print(line)
    for _, frame in sorted(parallel_results, key=lambda item: item[0]):
        if frame is not None:
            all_frames.append(frame)
    horses_info = {}

def extract_kanji(text):
    return ''.join(re.findall(r'[一-鿿]+', text))

for horse_number, horse_data in horses_info.items():
    print(f"\nFetching data for {horse_data['name']}...")

    page = get_page_source_fast(
        driver,
        horse_data["url"],
        wait_css=".db_h_race_results",
    )
    assert_not_blocked(driver, horse_data["url"])
    if not driver.find_elements(By.CLASS_NAME, "db_h_race_results"):
        print(f"WARN: {horse_data['name']} results table not found")
        continue

    soup = BeautifulSoup(page, 'html.parser')
    birthdate = extract_birthdate(soup)
    table = soup.find('table', {'class': 'db_h_race_results nk_tb_common'})
    if table is None:
        print(f"WARN: {horse_data['name']} table is None")
        continue

    try:
        df = table_to_dataframe(table)
        if df.empty:
            raise ValueError("parsed table is empty")
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

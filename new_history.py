from selenium import webdriver
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re
import random
import sys
from urllib.parse import urljoin, urlparse
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


def configure_utf8_io():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_io()
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


def should_headless():
    raw = os.environ.get("PIPELINE_HEADLESS", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def create_driver(allow_shared=True):
    options = Options()
    if PAGE_LOAD_STRATEGY in ("normal", "eager", "none"):
        options.page_load_strategy = PAGE_LOAD_STRATEGY
    if allow_shared and debugger_address:
        options.add_experimental_option("debuggerAddress", debugger_address)
    else:
        if should_headless():
            options.add_argument("--headless")
        options.add_argument("--lang=ja-JP")
    created_driver = webdriver.Chrome(options=options)
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

# ===== 用户输入 =====
url = input("History search URL: ")

# ===== 启动 Selenium =====
debugger_address = os.environ.get("CHROME_DEBUGGER_ADDRESS", "").strip()
shared_driver = bool(debugger_address)
driver = create_driver(allow_shared=True)

# ===== 打开目标列表页 =====
page = get_page_source_fast(driver, url, wait_css='a[href^="/race/"]')
assert_not_blocked(driver, url)
soup = BeautifulSoup(page, 'html.parser')

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
horse_fetch_workers = get_env_int("PIPELINE_HORSE_FETCH_WORKERS", 1, minimum=1, maximum=8)


def fetch_history_horse_frame(active_driver, horse_entry, race_id, race_date, race_date_str):
    horse_name, horse_url, finish_pos, is_top3, sex, age, sex_age = horse_entry
    logs = [f"  - {horse_name}"]
    horse_page = get_page_source_fast(
        active_driver,
        horse_url,
        wait_css=".db_h_race_results",
    )
    assert_not_blocked(active_driver, horse_url)
    soup_horse = BeautifulSoup(horse_page, 'html.parser')
    birthdate = extract_birthdate(soup_horse)
    table = find_new_results_table(soup_horse)

    if table is None:
        logs.append(f"    WARN: {horse_name} results table not found")
        return None, logs

    try:
        df = table_to_dataframe(table)
        if df.empty:
            raise ValueError("parsed table is empty")
    except Exception as e:
        logs.append(f"    ERROR: Failed to parse {horse_name} table: {e}")
        return None, logs
    jockey_series = build_jockey_id_series(table, df)

    date_col = find_date_column(df)
    if not date_col:
        logs.append(f"    WARN: {horse_name} date column not found. Skipping.")
        return None, logs
    date_series = df[date_col].apply(parse_race_date)
    match = date_series == race_date
    if match.any():
        start_index = match[match].index[0]
        df = df.loc[start_index:]
        jockey_series = jockey_series.loc[df.index]
    else:
        logs.append(f"    WARN: {horse_name} no race on {race_date_str}. Skipping.")
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
        logs.append(f"    WARN: {horse_name} missing TimeIndex column. columns={list(df.columns)}")
        return None, logs

    if df.empty:
        logs.append(f"    WARN: {horse_name} has no valid data.")
        return None, logs

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
    logs.append(f"    OK: {horse_name} data extracted.")
    sleep_jitter()
    return df, logs


def process_history_horse_batch(items, race_id, race_date, race_date_str, session_cookies):
    batch_results = []
    batch_logs = []
    batch_driver = create_driver(allow_shared=False)
    try:
        first_target_url = items[0][1][1] if items else ""
        if first_target_url:
            apply_cookies_for_url(batch_driver, first_target_url, session_cookies)
        for order, horse_entry in items:
            frame, logs = fetch_history_horse_frame(batch_driver, horse_entry, race_id, race_date, race_date_str)
            batch_results.append((order, frame))
            batch_logs.extend(logs)
    finally:
        batch_driver.quit()
    return batch_results, batch_logs


def collect_history_frames_for_race(horse_links, race_id, race_date, race_date_str, session_cookies):
    ordered_items = list(enumerate(horse_links, start=1))
    effective_workers = min(horse_fetch_workers, len(ordered_items))
    if effective_workers <= 1:
        race_frames = []
        race_logs = []
        for _, horse_entry in ordered_items:
            frame, logs = fetch_history_horse_frame(driver, horse_entry, race_id, race_date, race_date_str)
            race_logs.extend(logs)
            if frame is not None:
                race_frames.append(frame)
        return race_frames, race_logs

    race_logs = [f"  Parallel horse fetch enabled: workers={effective_workers}"]
    grouped_items = split_work_items(ordered_items, effective_workers)
    parallel_results = []
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(process_history_horse_batch, group, race_id, race_date, race_date_str, session_cookies)
            for group in grouped_items
        ]
        for future in as_completed(futures):
            batch_results, batch_logs = future.result()
            parallel_results.extend(batch_results)
            race_logs.extend(batch_logs)
    race_frames = []
    for _, frame in sorted(parallel_results, key=lambda item: item[0]):
        if frame is not None:
            race_frames.append(frame)
    return race_frames, race_logs

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
    race_page = get_page_source_fast(
        driver,
        race_url,
        wait_css="table.race_table_01",
    )
    assert_not_blocked(driver, race_url)
    soup_race = BeautifulSoup(race_page, 'html.parser')

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

    session_cookies = capture_netkeiba_cookies(driver)
    race_frames, race_logs = collect_history_frames_for_race(
        horse_links,
        race_id,
        race_date,
        race_date_str,
        session_cookies,
    )
    for line in race_logs:
        print(line)
    all_frames.extend(race_frames)
    continue

    # 抓每匹马的战绩表
    for h_idx, (horse_name, horse_url, finish_pos, is_top3, sex, age, sex_age) in enumerate(horse_links, start=1):
        print(f"  - Horse {h_idx}: {horse_name}")
        horse_page = get_page_source_fast(
            driver,
            horse_url,
            wait_css=".db_h_race_results",
        )
        assert_not_blocked(driver, horse_url)
        soup_horse = BeautifulSoup(horse_page, 'html.parser')
        birthdate = extract_birthdate(soup_horse)

        # 查找新版战绩表（以列名“レース名”为标志）
        table = find_new_results_table(soup_horse)

        if table is None:
            print(f"    WARN: {horse_name} results table not found")
            continue

        try:
            df = table_to_dataframe(table)
            if df.empty:
                raise ValueError("parsed table is empty")
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

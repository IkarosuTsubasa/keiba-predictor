import argparse
import sys
from pathlib import Path

from run_pipeline import ROOT_DIR, extract_race_id, run_script, sleep_between_scrapes, start_shared_chrome


def prompt_value(label):
    value = input(label).strip()
    if not value:
        print("缺少输入，已取消。")
        sys.exit(1)
    return value


def ensure_updated(path, previous_mtime):
    if not path.exists():
        raise RuntimeError(f"{path.name} 未生成。")
    if previous_mtime is None:
        return
    try:
        current_mtime = path.stat().st_mtime
    except OSError as exc:
        raise RuntimeError(f"{path.name} 无法读取时间戳: {exc}") from exc
    if current_mtime <= previous_mtime:
        raise RuntimeError(f"{path.name} 没有更新，请检查抓取是否成功。")


def main():
    parser = argparse.ArgumentParser(
        description="只生成本地的 kachiuma.csv 和 shutuba.csv。"
    )
    parser.add_argument("--race-url", help="race_card.py 使用的比赛 URL")
    parser.add_argument("--history-url", help="new_history.py 使用的历史检索 URL")
    args = parser.parse_args()

    race_url = (args.race_url or "").strip() or prompt_value("Race URL: ")
    history_url = (args.history_url or "").strip() or prompt_value("History search URL: ")

    race_id = extract_race_id(race_url)
    if not race_id:
        print("Race URL 里缺少 race_id，已取消。")
        sys.exit(1)

    shutuba_path = ROOT_DIR / "shutuba.csv"
    kachiuma_path = ROOT_DIR / "kachiuma.csv"
    shutuba_before = shutuba_path.stat().st_mtime if shutuba_path.exists() else None
    kachiuma_before = kachiuma_path.stat().st_mtime if kachiuma_path.exists() else None

    print(f"开始生成输入文件，race_id={race_id}")
    start_shared_chrome()
    run_script(
        ROOT_DIR / "race_card.py",
        [race_url],
        "race_card",
        ROOT_DIR,
    )
    sleep_between_scrapes()
    run_script(
        ROOT_DIR / "new_history.py",
        [history_url],
        "new_history",
        ROOT_DIR,
    )

    ensure_updated(shutuba_path, shutuba_before)
    ensure_updated(kachiuma_path, kachiuma_before)

    print("")
    print("已生成以下文件：")
    print(f"- {shutuba_path}")
    print(f"- {kachiuma_path}")


if __name__ == "__main__":
    main()

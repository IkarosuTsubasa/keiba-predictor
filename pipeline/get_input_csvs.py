import argparse
import os
import shutil
import sys
import zipfile

from run_pipeline import ROOT_DIR, extract_race_id, run_script, sleep_between_scrapes, start_shared_chrome


def prompt_value(label):
    value = input(label).strip()
    if not value:
        print("必填值不能为空。")
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
        raise RuntimeError(f"{path.name} 修改时间检查失败: {exc}") from exc
    if current_mtime <= previous_mtime:
        raise RuntimeError(f"{path.name} 没有更新，请检查抓取流程是否实际执行。")


def save_outputs_to_race_dir(race_id, *paths):
    race_dir = ROOT_DIR / race_id
    race_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for path in paths:
        dest_path = race_dir / path.name
        shutil.copy2(path, dest_path)
        saved_paths.append(dest_path)
    return race_dir, saved_paths


def save_outputs_to_zip(race_id, *paths):
    zip_path = ROOT_DIR / f"{race_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=path.name)
    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="只生成本地的 kachiuma.csv 和 shutuba.csv，并最终打包为 race_id.zip。",
    )
    parser.add_argument("--race-url", help="race_card.py 使用的比赛 URL")
    parser.add_argument("--history-url", help="new_history.py 使用的历史检索 URL")
    args = parser.parse_args()

    race_url = (args.race_url or "").strip() or prompt_value("Race URL: ")
    history_url = (args.history_url or "").strip() or prompt_value("History search URL: ")

    race_id = extract_race_id(race_url)
    if not race_id:
        print("Race URL 里缺少 race_id，无法继续。")
        sys.exit(1)

    shutuba_path = ROOT_DIR / "shutuba.csv"
    kachiuma_path = ROOT_DIR / "kachiuma.csv"
    shutuba_before = shutuba_path.stat().st_mtime if shutuba_path.exists() else None
    kachiuma_before = kachiuma_path.stat().st_mtime if kachiuma_path.exists() else None

    print(f"开始抓取输入 CSV，race_id={race_id}")
    start_shared_chrome()
    scrape_env = {}
    if not os.environ.get("PIPELINE_HORSE_FETCH_WORKERS", "").strip():
        scrape_env["PIPELINE_HORSE_FETCH_WORKERS"] = "3"
    if not os.environ.get("PIPELINE_HORSE_DELAY_MIN", "").strip():
        scrape_env["PIPELINE_HORSE_DELAY_MIN"] = "0.6"
    if not os.environ.get("PIPELINE_HORSE_DELAY_MAX", "").strip():
        scrape_env["PIPELINE_HORSE_DELAY_MAX"] = "1.5"
    run_script(
        ROOT_DIR / "race_card.py",
        [race_url],
        "race_card",
        ROOT_DIR,
        extra_env=scrape_env,
    )
    sleep_between_scrapes()
    run_script(
        ROOT_DIR / "new_history.py",
        [history_url],
        "new_history",
        ROOT_DIR,
        extra_env=scrape_env,
    )

    ensure_updated(shutuba_path, shutuba_before)
    ensure_updated(kachiuma_path, kachiuma_before)
    race_dir, saved_paths = save_outputs_to_race_dir(race_id, shutuba_path, kachiuma_path)
    zip_path = save_outputs_to_zip(race_id, shutuba_path, kachiuma_path)

    print("")
    print("生成成功:")
    print(f"- {shutuba_path}")
    print(f"- {kachiuma_path}")
    print("")
    print(f"ZIP 输出: {zip_path}")
    print("")
    print(f"Race CSV 目录副本: {race_dir}")
    for saved_path in saved_paths:
        print(f"- {saved_path}")


if __name__ == "__main__":
    main()

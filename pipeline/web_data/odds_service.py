import csv
from pathlib import Path


def parse_odds_value(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_odds_snapshot(normalize_name, path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                out = {}
                for row in reader:
                    horse_no = str(row.get("horse_no", "")).strip()
                    name = str(row.get("name", "")).strip()
                    odds_raw = str(row.get("odds", "")).strip()
                    key = ""
                    if horse_no:
                        key = f"no:{horse_no}"
                    elif name:
                        key = f"name:{normalize_name(name)}"
                    if not key:
                        continue
                    out[key] = {
                        "horse_no": horse_no,
                        "name": name,
                        "odds_raw": odds_raw,
                        "odds_val": parse_odds_value(odds_raw),
                    }
                return out
        except UnicodeDecodeError:
            continue
    return {}


def odds_changed(prev_item, curr_item):
    if prev_item.get("odds_val") is not None and curr_item.get("odds_val") is not None:
        return abs(prev_item["odds_val"] - curr_item["odds_val"]) > 1e-9
    return str(prev_item.get("odds_raw", "")) != str(curr_item.get("odds_raw", ""))


def odds_item_label(item):
    horse_no = str(item.get("horse_no", "")).strip()
    name = str(item.get("name", "")).strip()
    if horse_no and name:
        return f"{horse_no} {name}"
    return horse_no or name or "(unknown)"


def odds_sort_key(item):
    horse_no = str(item.get("horse_no", "")).strip()
    name = str(item.get("name", "")).strip()
    try:
        return (0, int(float(horse_no)), name)
    except (TypeError, ValueError):
        return (1, horse_no, name)


def format_odds_diff(prev_snapshot, curr_snapshot, limit=50):
    if not prev_snapshot:
        return ["odds_diff: no previous odds"]
    if not curr_snapshot:
        return ["odds_diff: no new odds"]
    added = []
    removed = []
    changed = []
    for key, curr in curr_snapshot.items():
        prev = prev_snapshot.get(key)
        if not prev:
            added.append(curr)
        elif odds_changed(prev, curr):
            changed.append((prev, curr))
    for key, prev in prev_snapshot.items():
        if key not in curr_snapshot:
            removed.append(prev)
    lines = [
        f"odds_diff: changed={len(changed)} added={len(added)} removed={len(removed)}"
    ]
    for prev, curr in sorted(changed, key=lambda pair: odds_sort_key(pair[1])):
        lines.append(
            f"~ {odds_item_label(curr)}: {prev.get('odds_raw', '')} -> {curr.get('odds_raw', '')}"
        )
    for item in sorted(added, key=odds_sort_key):
        lines.append(f"+ {odds_item_label(item)}: {item.get('odds_raw', '')}")
    for item in sorted(removed, key=odds_sort_key):
        lines.append(f"- {odds_item_label(item)}: {item.get('odds_raw', '')}")
    if len(lines) > limit + 1:
        more = len(lines) - (limit + 1)
        return lines[: limit + 1] + [f"... truncated {more} lines"]
    return lines

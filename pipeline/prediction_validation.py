import csv
import re
from pathlib import Path


def normalize_name(value):
    return "".join(str(value or "").split())


_INTEGER_LIKE_NUMBER_RE = re.compile(r"^\s*([0-9]+)(?:\.0+)?\s*$")


def normalize_horse_no(value):
    text = str(value or "").strip()
    if not text:
        return ""
    matched = _INTEGER_LIKE_NUMBER_RE.match(text)
    if matched:
        return str(int(matched.group(1)))
    return text


def load_entry_sets(path, name_fields, no_fields):
    src = Path(path)
    if not src.exists():
        return None, None, f"{src} not found."
    with open(src, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        name_field = next((field for field in name_fields if field in fieldnames), None)
        no_field = next((field for field in no_fields if field in fieldnames), None)
        if name_field is None and no_field is None:
            expected = ", ".join(sorted(set(name_fields + no_fields)))
            return None, None, f"{src} missing columns: {expected}"
        names = set()
        numbers = set()
        for row in reader:
            if name_field:
                name = normalize_name(row.get(name_field, ""))
                if name:
                    names.add(name)
            if no_field:
                horse_no = normalize_horse_no(row.get(no_field, ""))
                if horse_no:
                    numbers.add(horse_no)
    if not names and not numbers:
        return None, None, f"{src} has no usable rows."
    return names, numbers, ""


def format_entry_mismatch(expected, actual, label):
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    parts = []
    if missing:
        parts.append(f"missing_{label}={','.join(missing[:8])}")
    if extra:
        parts.append(f"extra_{label}={','.join(extra[:8])}")
    return "; ".join(parts)


def validate_odds_predictions(odds_path, pred_path):
    odds_names, odds_numbers, err = load_entry_sets(
        odds_path,
        ["name", "HorseName", "horse_name"],
        ["horse_no", "HorseNo", "horse_number", "馬番"],
    )
    if err:
        return False, err
    pred_names, pred_numbers, err = load_entry_sets(
        pred_path,
        ["HorseName", "horse_name", "name"],
        ["horse_no", "HorseNo", "horse_number", "馬番"],
    )
    if err:
        return False, err

    if odds_numbers and pred_numbers:
        if odds_numbers != pred_numbers:
            mismatch = format_entry_mismatch(odds_numbers, pred_numbers, "horse_no")
            return False, f"odds/predictions horse_no mismatch: {mismatch}"
        return True, ""

    if not odds_names or not pred_names:
        return False, "odds/predictions missing comparable entrant fields."
    if odds_names != pred_names:
        mismatch = format_entry_mismatch(odds_names, pred_names, "horse_name")
        return False, f"odds/predictions horse_name mismatch: {mismatch}"
    return True, ""

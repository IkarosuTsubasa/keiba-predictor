import csv


def normalize_csv_fieldnames(fieldnames):
    if not fieldnames:
        return []
    return [name.lstrip("\ufeff") if name else name for name in fieldnames]


def normalize_csv_rows(rows, fieldnames):
    norm_fieldnames = normalize_csv_fieldnames(fieldnames)
    if norm_fieldnames == fieldnames:
        return rows, fieldnames
    mapping = dict(zip(fieldnames, norm_fieldnames))
    out = []
    for row in rows:
        new_row = {}
        for key, value in row.items():
            new_key = mapping.get(key, key.lstrip("\ufeff") if key else key)
            new_row[new_key] = value
        out.append(new_row)
    return out, norm_fieldnames


def load_runs_with_header(migrate_legacy_data, get_data_dir, base_dir, scope_key):
    migrate_legacy_data(base_dir, scope_key)
    runs_path = get_data_dir(base_dir, scope_key) / "runs.csv"
    if not runs_path.exists():
        return [], [], "utf-8"
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            with open(runs_path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
                rows, fieldnames = normalize_csv_rows(rows, fieldnames)
                return rows, fieldnames, enc
        except UnicodeDecodeError:
            continue
    return [], [], "utf-8"


def update_run_row_fields(
    get_data_dir,
    base_dir,
    load_runs_with_header_func,
    normalize_race_id,
    scope_key,
    run_row,
    updates,
):
    if not run_row or not updates:
        return False
    runs_path = get_data_dir(base_dir, scope_key) / "runs.csv"
    rows, fieldnames, enc = load_runs_with_header_func(scope_key)
    if not rows or not fieldnames:
        return False
    for key in updates:
        if key not in fieldnames:
            fieldnames.append(key)
    target_run_id = str(run_row.get("run_id", "")).strip()
    target_race_id = normalize_race_id(run_row.get("race_id", ""))
    target_timestamp = str(run_row.get("timestamp", "")).strip()
    target_pred_path = str(run_row.get("predictions_path", "")).strip()
    matched = False
    for row in rows:
        row_run_id = str(row.get("run_id", "")).strip()
        if target_run_id and row_run_id == target_run_id:
            row.update(updates)
            matched = True
            break
        row_race_id = normalize_race_id(row.get("race_id", ""))
        row_ts = str(row.get("timestamp", "")).strip()
        if target_race_id and target_timestamp:
            if row_race_id != target_race_id or row_ts != target_timestamp:
                continue
            if target_pred_path:
                row_pred = str(row.get("predictions_path", "")).strip()
                if row_pred != target_pred_path:
                    continue
            row.update(updates)
            matched = True
            break
    if not matched:
        return False
    with open(runs_path, "w", newline="", encoding=enc, errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return True


def update_run_plan_path(get_data_dir, base_dir, load_runs_with_header_func, scope_key, run_id, plan_path):
    runs_path = get_data_dir(base_dir, scope_key) / "runs.csv"
    rows, fieldnames, enc = load_runs_with_header_func(scope_key)
    if not rows or not fieldnames:
        return False
    updated = False
    if "plan_path" not in fieldnames:
        fieldnames.append("plan_path")
    for row in rows:
        if row.get("run_id") == run_id:
            row["plan_path"] = str(plan_path)
            updated = True
            break
    if not updated:
        return False
    with open(runs_path, "w", newline="", encoding=enc, errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return True

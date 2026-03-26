from pathlib import Path


PREDICTOR_DISPLAY_LABELS = {
    "main": "ゲート",
    "v2_opus": "ストライド",
    "v3_premium": "伯楽",
    "v4_gemini": "馬場眼",
    "v5_stacking": "フュージョン",
    "v6_kiwami": "極 KIWAMI",
}


PREDICTOR_SPECS = [
    {
        "id": "main",
        "label": PREDICTOR_DISPLAY_LABELS["main"],
        "script_name": "predictor.py",
        "run_field": "predictions_path",
        "latest_filename": "predictions.csv",
        "snapshot_prefix": "predictions",
        "is_primary": True,
    },
    {
        "id": "v2_opus",
        "label": PREDICTOR_DISPLAY_LABELS["v2_opus"],
        "script_name": "predictor_v2_opus.py",
        "run_field": "predictions_v2_opus_path",
        "latest_filename": "predictions_v2_opus.csv",
        "snapshot_prefix": "predictions_v2_opus",
        "is_primary": False,
    },
    {
        "id": "v3_premium",
        "label": PREDICTOR_DISPLAY_LABELS["v3_premium"],
        "script_name": "predictor_v3_premium.py",
        "run_field": "predictions_v3_premium_path",
        "latest_filename": "predictions_v3_premium.csv",
        "snapshot_prefix": "predictions_v3_premium",
        "is_primary": False,
    },
    {
        "id": "v4_gemini",
        "label": PREDICTOR_DISPLAY_LABELS["v4_gemini"],
        "script_name": "predictor_v4_gemini.py",
        "run_field": "predictions_v4_gemini_path",
        "latest_filename": "predictions_v4_gemini.csv",
        "snapshot_prefix": "predictions_v4_gemini",
        "is_primary": False,
    },
    {
        "id": "v5_stacking",
        "label": PREDICTOR_DISPLAY_LABELS["v5_stacking"],
        "script_name": "predictor_v5_omc.py",
        "run_field": "predictions_v5_stacking_path",
        "latest_filename": "predictions_v5_stacking.csv",
        "snapshot_prefix": "predictions_v5_stacking",
        "is_primary": False,
    },
    {
        "id": "v6_kiwami",
        "label": PREDICTOR_DISPLAY_LABELS["v6_kiwami"],
        "script_name": "predictor_v6_gptpro.py",
        "run_field": "predictions_v6_kiwami_path",
        "latest_filename": "predictions_v6_kiwami.csv",
        "snapshot_prefix": "predictions_v6_kiwami",
        "is_primary": False,
    },
]


def list_predictors():
    return list(PREDICTOR_SPECS)


def get_predictor_spec(predictor_id):
    predictor_id = str(predictor_id or "").strip()
    for spec in PREDICTOR_SPECS:
        if spec["id"] == predictor_id:
            return spec
    return None


def canonical_predictor_id(value):
    raw = str(value or "").strip()
    return raw if get_predictor_spec(raw) else "main"


def predictor_label(value):
    spec = get_predictor_spec(canonical_predictor_id(value))
    if spec:
        return spec["label"]
    return PREDICTOR_DISPLAY_LABELS["main"]


def latest_prediction_path(root_dir, predictor_id):
    spec = get_predictor_spec(predictor_id)
    if not spec:
        return None
    return Path(root_dir) / spec["latest_filename"]


def snapshot_prediction_path(data_dir, race_id, run_id, predictor_id):
    spec = get_predictor_spec(predictor_id)
    if not spec:
        return None
    race_id = str(race_id or "").strip()
    run_id = str(run_id or "").strip()
    if race_id:
        return Path(data_dir) / race_id / f"{spec['snapshot_prefix']}_{run_id}_{race_id}.csv"
    return Path(data_dir) / f"{spec['snapshot_prefix']}_{run_id}.csv"


def resolve_run_prediction_path(run_row, data_dir, root_dir, run_id="", race_id="", predictor_id="main", last_run_id=""):
    spec = get_predictor_spec(predictor_id)
    if not spec:
        return None
    run_row = run_row or {}
    run_id = str(run_id or run_row.get("run_id") or "").strip()
    race_id = str(race_id or run_row.get("race_id") or "").strip()
    last_run_id = str(last_run_id or "").strip()
    candidates = []
    run_path = str(run_row.get(spec["run_field"], "") or "").strip()
    if run_path:
        candidates.append(Path(run_path))
    if spec["is_primary"]:
        legacy_path = str(run_row.get("predictions_path", "") or "").strip()
        if legacy_path:
            candidates.append(Path(legacy_path))
    if run_id:
        if race_id:
            candidates.append(snapshot_prediction_path(data_dir, race_id, run_id, predictor_id))
        if spec["is_primary"]:
            candidates.append(Path(data_dir) / f"predictions_{run_id}.csv")
        else:
            candidates.append(Path(data_dir) / f"{spec['snapshot_prefix']}_{run_id}.csv")
    latest_path = latest_prediction_path(root_dir, predictor_id)
    # For explicit historical/current run resolution, do not fall back to the
    # root-level latest file. That file may belong to a different race and can
    # leak stale predictions into workspace/admin views.
    if latest_path is not None and not run_id:
        candidates.append(latest_path)
    for path in candidates:
        if path and path.exists():
            return path
    return candidates[0] if candidates else latest_path

import os
from pathlib import Path


def normalize_scope_key(value):
    if value is None:
        return ""
    raw = str(value).strip().lower()
    raw = raw.replace(" ", "_").replace("-", "_").replace("/", "_")
    if raw in ("central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"):
        return "central_turf"
    if raw in ("central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"):
        return "central_dirt"
    if raw in ("local", "l", "3"):
        return "local"
    return ""


def get_scope_key(prompt_label="Data scope (central_turf/central_dirt/local) [central_dirt]: "):
    env = normalize_scope_key(os.environ.get("SCOPE_KEY", ""))
    if not env:
        env = normalize_scope_key(os.environ.get("SURFACE_KEY", ""))
    if env:
        return env
    try:
        raw = input(prompt_label).strip()
    except EOFError:
        raw = ""
    key = normalize_scope_key(raw)
    return key or "central_dirt"


def get_data_dir(base_dir, scope_key):
    return base_dir / "data" / scope_key


def get_config_path(base_dir, scope_key):
    return base_dir / f"config_{scope_key}.json"


def get_predictor_config_path(base_dir, scope_key):
    return base_dir / f"predictor_config_{scope_key}.json"


def get_predictor_prev_path(base_dir, scope_key):
    return base_dir / f"predictor_config_prev_{scope_key}.json"


def ensure_scope_dir(base_dir, scope_key):
    scope_dir = get_data_dir(base_dir, scope_key)
    scope_dir.mkdir(parents=True, exist_ok=True)
    return scope_dir


def migrate_legacy_data(base_dir, scope_key):
    legacy_root = base_dir / "data"
    if not legacy_root.exists():
        return
    target_dir = ensure_scope_dir(base_dir, scope_key)
    legacy_map = {"central_dirt": "dirt", "central_turf": "turf"}

    if scope_key in legacy_map:
        legacy_surface = legacy_root / legacy_map[scope_key]
        if legacy_surface.exists():
            source_dirs = []
            legacy_default = legacy_surface / "default"
            if legacy_default.exists():
                source_dirs.append(legacy_default)
            source_dirs.append(legacy_surface)
            for source_dir in source_dirs:
                for item in source_dir.iterdir():
                    if item.is_dir():
                        continue
                    target = target_dir / item.name
                    if target.exists():
                        continue
                    try:
                        item.replace(target)
                    except OSError:
                        continue

    if scope_key == "central_dirt":
        for item in legacy_root.iterdir():
            if item.is_dir():
                continue
            target = target_dir / item.name
            if target.exists():
                continue
            try:
                item.replace(target)
            except OSError:
                continue

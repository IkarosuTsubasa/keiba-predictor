import json
import os
from pathlib import Path


def load_local_env(base_dir):
    base_path = Path(base_dir)
    candidates = [
        base_path / "data" / "_shared" / "local_env.json",
        base_path / "data" / "_shared" / ".env.json",
    ]
    loaded = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            name = str(key or "").strip()
            if not name:
                continue
            text = str(value or "")
            if name not in os.environ or not str(os.environ.get(name, "")).strip():
                os.environ[name] = text
                loaded[name] = True
    return loaded


__all__ = ["load_local_env"]

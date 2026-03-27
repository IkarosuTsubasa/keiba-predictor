import json
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SITE_COPY_PATH = BASE_DIR.parent / "frontend" / "src" / "content" / "siteCopy.json"


@lru_cache(maxsize=1)
def load_site_copy():
    with open(SITE_COPY_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)

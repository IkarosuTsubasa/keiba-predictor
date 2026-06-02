from __future__ import annotations

from urllib.parse import parse_qs, urlparse


SUPPORTED_HOST = "race.netkeiba.com"


def extract_race_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname != SUPPORTED_HOST:
        raise ValueError("not a supported netkeiba URL")

    query = parse_qs(parsed.query)
    race_ids = query.get("race_id", [])
    if not race_ids or not race_ids[0]:
        raise ValueError("race_id not found in netkeiba URL")

    return race_ids[0]

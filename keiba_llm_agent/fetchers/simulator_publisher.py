"""Publish collected inputs before expensive analysis; failures never block Agent."""
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def publish_collected_race(race_data):
    from pipeline.simulator_input_store import publish_race
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = publish_race(Path(__file__).resolve().parents[2] / "pipeline", race_data.model_dump(), source_as_of=stamp)
    callback = os.environ.get("SIMULATOR_INPUT_CALLBACK_URL", "")
    secret = os.environ.get("PIPELINE_CALLBACK_SECRET", "")
    if callback and secret:
        body = json.dumps({"race_data": race_data.model_dump(), "source_as_of": stamp}, ensure_ascii=False).encode()
        signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        for attempt in range(3):
            try:
                req = Request(callback, data=body, headers={"Content-Type": "application/json", "X-Hub-Signature-256": signature}, method="POST")
                with urlopen(req, timeout=15) as result:
                    result.read()
                break
            except Exception:
                if attempt == 2:
                    logging.getLogger(__name__).warning("Simulator input callback failed; prediction continues")
    return payload

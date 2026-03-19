from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main():
    trigger_url = str(os.environ.get("RUN_DUE_TRIGGER_URL", "") or "").strip()
    admin_token = str(os.environ.get("ADMIN_TOKEN", "") or "").strip()
    timeout_s = int(str(os.environ.get("RUN_DUE_TRIGGER_TIMEOUT_SECONDS", "60") or "60").strip() or "60")

    if not trigger_url:
        raise RuntimeError("RUN_DUE_TRIGGER_URL is required")
    if not admin_token:
        raise RuntimeError("ADMIN_TOKEN is required")

    req = urllib.request.Request(
        trigger_url,
        method="POST",
        headers={
            "Authorization": f"Bearer {admin_token}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            print(body)
            if resp.status >= 400:
                raise RuntimeError(f"run_due trigger failed: http {resp.status}")
            try:
                payload = json.loads(body or "{}")
            except Exception:
                payload = {}
            if payload.get("ok") is False:
                raise RuntimeError(f"run_due trigger failed: {body}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"run_due trigger failed: http {exc.code} {detail}".strip()) from exc


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

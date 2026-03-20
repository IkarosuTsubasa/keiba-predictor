from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def main():
    trigger_url = str(os.environ.get("RUN_DUE_TRIGGER_URL", "") or "").strip()
    admin_token = str(os.environ.get("ADMIN_TOKEN", "") or "").strip()
    timeout_s = int(str(os.environ.get("RUN_DUE_TRIGGER_TIMEOUT_SECONDS", "60") or "60").strip() or "60")
    max_attempts = int(str(os.environ.get("RUN_DUE_TRIGGER_RETRY_COUNT", "3") or "3").strip() or "3")

    if not trigger_url:
        raise RuntimeError("RUN_DUE_TRIGGER_URL is required")
    if not admin_token:
        raise RuntimeError("ADMIN_TOKEN is required")
    if max_attempts <= 0:
        max_attempts = 1

    req = urllib.request.Request(
        trigger_url,
        method="POST",
        headers={
            "Authorization": f"Bearer {admin_token}",
            "Accept": "application/json",
        },
    )
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                print(body)
                try:
                    payload = json.loads(body or "{}")
                except Exception:
                    payload = {}
                if payload.get("ok") is False:
                    raise RuntimeError(f"run_due trigger failed: {body}")
                return
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            detail_preview = " ".join(str(detail or "").split())[:400]
            last_error = RuntimeError(f"run_due trigger failed: http {exc.code} {detail_preview}".strip())
            if exc.code not in (502, 503, 504) or attempt >= max_attempts:
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"run_due trigger failed: url error {exc}")
            if attempt >= max_attempts:
                raise last_error from exc
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise
        time.sleep(min(5 * attempt, 15))
    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

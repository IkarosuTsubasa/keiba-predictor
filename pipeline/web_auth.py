import hashlib
import hmac as hmac_mod
import os
import secrets


def admin_token_expected():
    return str(os.environ.get("ADMIN_TOKEN", "") or "").strip()


def verify_callback_hmac(body: bytes, sig_header: str) -> bool:
    secret = str(os.environ.get("PIPELINE_CALLBACK_SECRET", "") or "").strip()
    if not secret or not sig_header:
        return False
    expected = "sha256=" + hmac_mod.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return secrets.compare_digest(sig_header, expected)


def admin_token_enabled():
    return bool(admin_token_expected())


def admin_token_valid(token=""):
    expected = admin_token_expected()
    if not expected:
        return True
    supplied = str(token or "").strip()
    return bool(supplied) and secrets.compare_digest(supplied, expected)

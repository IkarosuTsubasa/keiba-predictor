"""Association documents require real release identities; no fake certificates."""
import os
import re
from fastapi.responses import JSONResponse


def register_routes(app):
    @app.get("/.well-known/assetlinks.json")
    def android_links():
        fingerprints = [v.strip().upper() for v in os.environ.get("SIMULATOR_ANDROID_SHA256", "").split(",") if v.strip()]
        if not fingerprints or not all(re.fullmatch(r"(?:[A-F0-9]{2}:){31}[A-F0-9]{2}", v) for v in fingerprints):
            return JSONResponse({"error": "release_certificate_not_configured"}, status_code=503, headers={"Cache-Control": "no-store"})
        return JSONResponse([{"relation": ["delegate_permission/common.handle_all_urls"], "target": {"namespace": "android_app", "package_name": "com.ikaimo.keiba.app", "sha256_cert_fingerprints": fingerprints}}], headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/.well-known/apple-app-site-association")
    def apple_links():
        team = os.environ.get("SIMULATOR_IOS_TEAM_ID", "")
        if not re.fullmatch(r"[A-Z0-9]{10}", team):
            return JSONResponse({"error": "release_team_not_configured"}, status_code=503, headers={"Cache-Control": "no-store"})
        return JSONResponse({"applinks": {"details": [{"appIDs": [f"{team}.com.ikaimo.keiba.app"], "components": [{"/": "/keiba/open/simulator/*"}]}]}}, headers={"Cache-Control": "public, max-age=3600"})

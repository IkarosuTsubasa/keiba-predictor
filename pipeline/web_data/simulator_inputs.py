"""HTTP contract for the shared simulator, separate from prediction APIs."""
import hashlib
import json
import os
from datetime import datetime

from fastapi.responses import JSONResponse, Response

try:
    from simulator_input_store import MODEL_VERSION, JST, import_saved_race, list_races, read_race
except ImportError:
    from pipeline.simulator_input_store import MODEL_VERSION, JST, import_saved_race, list_races, read_race


def enabled():
    default = "0" if os.environ.get("RENDER") else "1"
    return os.environ.get("SIMULATOR_ENABLED", default).lower() not in ("0", "false", "off")


def config():
    return {"enabled": enabled(), "schema_version": 1, "bridge_version": 1,
            "active_model_version": MODEL_VERSION,
            "models": [{"version": MODEL_VERSION, "status": "active", "label": "標準モデル"}],
            "android_store_url": os.environ.get("SIMULATOR_ANDROID_STORE_URL", "https://play.google.com/store/apps/details?id=com.ikaimo.keiba.app"),
            "ios_store_url": os.environ.get("SIMULATOR_IOS_STORE_URL", ""),
            "app_link_base": "/keiba/open/simulator"}


def response(request, payload, *, immutable=False):
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    headers = {"ETag": f'"{digest}"', "Cache-Control": "public, max-age=31536000, immutable" if immutable else "public, max-age=15, must-revalidate"}
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload, headers=headers)


def day_text(value):
    value = value or datetime.now(JST).date().isoformat()
    parsed = datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    if parsed != value:
        raise ValueError("invalid_date")
    return value


def get_race(base_dir, rid, day, version=""):
    payload = read_race(base_dir, rid, day, version)
    if payload is None and not version:
        payload = import_saved_race(base_dir, rid, day)
    return payload


def lightweight_states(base_dir, day):
    if not enabled():
        return {}
    try:
        return {r["race_id"]: {k: r.get(k) for k in ("availability", "input_version", "published_at")} for r in list_races(base_dir, day)}
    except ValueError:
        return {}


def register_routes(app, base_dir, load_schedule):
    from fastapi import Request
    from pathlib import Path
    try:
        from simulator_input_store import _json
    except ImportError:
        from pipeline.simulator_input_store import _json
    def schedule(day):
        return [*load_schedule(), *(_json(Path(base_dir) / "data/simulator_schedule" / f"{day}.json") or [])]
    # Register explicit prefixes; neither conflicts with /races/{race_path:path}.
    for prefix in ("/keiba/api/public/simulator", "/api/public/simulator"):
        def models(request: Request):
            return response(request, {"ok": True, "data": config()})

        def races(request: Request, date: str = "", scope_key: str = "", state: str = "all"):
            if not enabled():
                return JSONResponse({"ok": False, "error": "disabled"}, status_code=503)
            try:
                day = day_text(date)
                if state not in ("all", "upcoming"):
                    raise ValueError("invalid_state")
                rows = list_races(base_dir, day, schedule(day), scope_key=scope_key, state=state)
                return response(request, {"ok": True, "data": {"race_date": day, "races": rows}})
            except ValueError:
                return JSONResponse({"ok": False, "error": "invalid_request"}, status_code=400)

        def detail(race_id: str, request: Request, date: str = "", input_version: str = ""):
            if not enabled():
                return JSONResponse({"ok": False, "error": "disabled"}, status_code=503)
            try:
                day = day_text(date)
                payload = get_race(base_dir, race_id, day, input_version)
            except ValueError:
                return JSONResponse({"ok": False, "error": "invalid_request"}, status_code=400)
            if payload is None:
                known = any(r["race_id"] == race_id for r in list_races(base_dir, day, schedule(day)))
                error = "version_not_found" if input_version else "preparing" if known else "not_found"
                return JSONResponse({"ok": False, "error": error, "race_id": race_id, "race_date": day}, status_code=409 if error == "preparing" else 404)
            return response(request, {"ok": True, "data": payload}, immutable=bool(input_version))

        app.add_api_route(f"{prefix}/models", models, methods=["GET"])
        app.add_api_route(f"{prefix}/races", races, methods=["GET"])
        app.add_api_route(f"{prefix}/races/{{race_id}}", detail, methods=["GET"])

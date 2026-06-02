import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
PUBLIC_FRONTEND_INDEX = BASE_DIR / "public_frontend_dist" / "index.html"


def ensure_public_frontend_dist():
    if PUBLIC_FRONTEND_INDEX.exists():
        return
    package_json = FRONTEND_DIR / "package.json"
    if not package_json.exists():
        print(f"Missing public frontend dist: {PUBLIC_FRONTEND_INDEX}", flush=True)
        print(f"Missing frontend package.json: {package_json}", flush=True)
        sys.exit(1)
    print("Public frontend dist is missing; building frontend before starting web server.", flush=True)
    try:
        subprocess.run(["npm", "install", "--include=dev"], cwd=FRONTEND_DIR, check=True)
        subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
    except FileNotFoundError:
        print("Missing dependency: npm", flush=True)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"Failed to build public frontend dist: command exited with {exc.returncode}", flush=True)
        sys.exit(exc.returncode or 1)
    if not PUBLIC_FRONTEND_INDEX.exists():
        print(f"Frontend build finished but index is still missing: {PUBLIC_FRONTEND_INDEX}", flush=True)
        sys.exit(1)


def main():
    try:
        import uvicorn
    except ImportError:
        print("Missing dependency: uvicorn")
        print("Install with: pip install uvicorn fastapi")
        sys.exit(1)
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError:
        port = 8000
    ensure_public_frontend_dist()
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()

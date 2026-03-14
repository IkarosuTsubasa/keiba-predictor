import os
import sys


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
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()

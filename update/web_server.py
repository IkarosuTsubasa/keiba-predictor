import sys


def main():
    try:
        import uvicorn
    except ImportError:
        print("Missing dependency: uvicorn")
        print("Install with: pip install uvicorn fastapi")
        sys.exit(1)
    uvicorn.run(
        "web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()

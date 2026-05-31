"""Convenience launcher for the FastAPI app.

Usage:
    python scripts/run_api.py
        -> http://127.0.0.1:8000
        -> http://127.0.0.1:8000/docs   (interactive Swagger UI)

This is the same as:
    uvicorn regflow.api.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "regflow.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()

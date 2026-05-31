"""FastAPI app entry. Mount routers here."""

from __future__ import annotations

from fastapi import FastAPI

from regflow import __version__
from regflow.api.routes import reviews
from regflow.common.logging import configure_logging

configure_logging()

app = FastAPI(
    title="RegFlow API",
    version=__version__,
    description=(
        "RegFlow human-in-the-loop endpoints. "
        "The /reviews/corrections endpoint closes the correction-retrieval loop "
        "central to the architecture (lines 131-170)."
    ),
)

app.include_router(reviews.router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}

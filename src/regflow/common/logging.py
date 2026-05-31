from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from regflow.common.settings import get_settings

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    # Force UTF-8 on stdout/stderr. Windows PowerShell defaults to cp1252 which can't
    # encode characters common in EU regulatory text (U+202F narrow no-break space, etc.).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    level = getattr(logging, settings.log_level)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout, force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "weaviate"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()

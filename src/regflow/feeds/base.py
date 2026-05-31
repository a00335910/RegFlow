"""Common types and the FeedConnector protocol for all regulatory sources.

Every connector implements two steps:
1. `discover()` — list candidate source_doc_ids (e.g. CELEX numbers, FCA handbook refs).
2. `fetch(source_doc_id)` — download + parse one document into a ParsedDocument.

The ingestion pipeline (in ingestion/pipeline.py) wraps every connector identically:
fetched bytes -> MinIO, metadata -> Postgres, article segments -> Postgres + Weaviate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass
class FetchedDocument:
    """Raw bytes + content type as returned by the connector before parsing."""

    source_doc_id: str
    source_url: str
    content: bytes
    content_type: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ParsedArticle:
    article_ref: str       # e.g. "Article 1", "Annex II"
    sequence: int          # order within document
    text: str
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class ParsedDocument:
    source: str                    # connector identifier ("eur_lex", "fca", ...)
    source_doc_id: str             # canonical id within source
    title: str
    jurisdiction: str              # "EU", "UK", "US-FED", ...
    regulator: str                 # "European Parliament", "FCA", ...
    document_type: str | None = None
    published_date: datetime | None = None
    source_url: str | None = None
    language: str = "en"
    canonical_text: str = ""       # plain text used for hashing / search
    articles: list[ParsedArticle] = field(default_factory=list)
    raw_content: bytes = b""
    raw_content_type: str = "text/html"
    extra_metadata: dict[str, str] = field(default_factory=dict)


class FeedConnector(Protocol):
    source: str

    def discover(self) -> list[str]:
        """Return source_doc_ids to ingest. Connectors may use feeds.yaml, search APIs, etc."""
        ...

    def fetch(self, source_doc_id: str) -> ParsedDocument:
        """Download and parse one document. Raises on hard failure."""
        ...

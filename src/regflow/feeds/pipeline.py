"""Ingestion pipeline (Architecture lines 8-16).

Given a FeedConnector, for each source_doc_id:
  1. connector.fetch()                  -> ParsedDocument
  2. MinIO.put_object(raw bytes)        -> raw doc store
  3. Postgres upsert Document + Article rows
  4. Embed each article -> Weaviate corpus collection

This is "pure engineering" with no LLM, exactly as described in the architecture.
Agent 1 (Regulatory Radar) consumes from these stores and runs the diff.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from regflow.common.logging import get_logger
from regflow.db import minio_client
from regflow.db.postgres import Article, Document, get_session
from regflow.db.vector import get_embedder, get_vector_store
from regflow.feeds.base import FeedConnector, ParsedDocument

log = get_logger(__name__)


@dataclass
class IngestResult:
    document_id: UUID
    source_doc_id: str
    is_new_version: bool
    article_count: int
    content_hash: str


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _object_key(source: str, source_doc_id: str, content_hash: str, ext: str) -> str:
    return f"{source}/{source_doc_id}/{content_hash}.{ext}"


def ingest_document(connector: FeedConnector, source_doc_id: str) -> IngestResult:
    """Convenience wrapper: connector.fetch(...) -> persist."""
    return ingest_document_from_parsed(connector.fetch(source_doc_id))


def ingest_document_from_parsed(parsed: ParsedDocument) -> IngestResult:
    """Persist an already-parsed document into all stores.

    Split out so callers (local-file ingestion, future Kafka consumers) can hand in a
    ParsedDocument they obtained any way they like — without going through a FeedConnector.
    """
    doc_hash = _sha256(parsed.canonical_text)

    with get_session() as session:
        existing = session.execute(
            select(Document).where(
                Document.source == parsed.source,
                Document.source_doc_id == parsed.source_doc_id,
                Document.content_hash == doc_hash,
            )
        ).scalar_one_or_none()

        if existing is not None:
            log.info(
                "ingest.unchanged",
                source=parsed.source,
                source_doc_id=parsed.source_doc_id,
                document_id=str(existing.id),
            )
            return IngestResult(
                document_id=existing.id,
                source_doc_id=parsed.source_doc_id,
                is_new_version=False,
                article_count=len(existing.articles),
                content_hash=doc_hash,
            )

        ext = "html" if "html" in parsed.raw_content_type else "xml" if "xml" in parsed.raw_content_type else "bin"
        object_key = _object_key(parsed.source, parsed.source_doc_id, doc_hash, ext)
        minio_client.put_object(object_key, parsed.raw_content, content_type=parsed.raw_content_type)

        document = _persist_document(session, parsed, doc_hash, object_key)
        article_rows = _persist_articles(session, document, parsed)
        session.flush()      # populates document.id + article.id and resolves FK via relationship

        _index_articles(parsed, document.id, article_rows)

        log.info(
            "ingest.new_version",
            source=parsed.source,
            source_doc_id=parsed.source_doc_id,
            document_id=str(document.id),
            articles=len(article_rows),
        )
        return IngestResult(
            document_id=document.id,
            source_doc_id=parsed.source_doc_id,
            is_new_version=True,
            article_count=len(article_rows),
            content_hash=doc_hash,
        )


def _persist_document(session: Session, parsed: ParsedDocument, doc_hash: str, object_key: str) -> Document:
    document = Document(
        source=parsed.source,
        source_doc_id=parsed.source_doc_id,
        title=parsed.title,
        jurisdiction=parsed.jurisdiction,
        regulator=parsed.regulator,
        document_type=parsed.document_type,
        published_date=parsed.published_date,
        source_url=parsed.source_url,
        language=parsed.language,
        raw_object_key=object_key,
        content_hash=doc_hash,
        content_length=len(parsed.canonical_text),
        extra_metadata=parsed.extra_metadata,
    )
    session.add(document)
    return document


def _persist_articles(session: Session, document: Document, parsed: ParsedDocument) -> list[Article]:
    """Use the `Document.articles` relationship — SQLAlchemy fills in document_id at flush.
    This avoids the chicken-and-egg of needing document.id before the Document row is flushed."""
    rows: list[Article] = []
    for art in parsed.articles:
        row = Article(
            article_ref=art.article_ref,
            sequence=art.sequence,
            text=art.text,
            char_start=art.char_start,
            char_end=art.char_end,
            content_hash=_sha256(art.text),
        )
        document.articles.append(row)
        rows.append(row)
    return rows


def _index_articles(parsed: ParsedDocument, document_id: UUID, article_rows: list[Article]) -> None:
    if not article_rows:
        return
    embedder = get_embedder()
    vectors = embedder.embed([a.text for a in article_rows])
    store = get_vector_store()
    store.ensure_schema()
    for row, vec in zip(article_rows, vectors, strict=True):
        store.upsert_article(
            article_uuid=str(row.id),
            document_uuid=str(document_id),
            source=parsed.source,
            source_doc_id=parsed.source_doc_id,
            article_ref=row.article_ref,
            jurisdiction=parsed.jurisdiction,
            regulator=parsed.regulator,
            text=row.text,
            content_hash=row.content_hash,
            vector=vec,
        )



"""Prefect ingestion flow — the autonomous-polling answer.

What it does per run:
  1. For each configured source, call connector.discover() to find candidate documents.
  2. For each document, call ingest_document() — idempotent (content-hash dedup), so
     unchanged docs are no-ops.
  3. If ingest reported is_new_version=True, trigger run_workflow() to push the new
     version through Agent 1 + Orchestrator + Agent 2.

Failure handling:
  - One bad doc doesn't kill the run. Exceptions are caught per doc, logged, and the
    flow continues with the next.
  - One bad source doesn't kill the flow. Discovery exceptions are caught per source.
  - Anti-bot challenges on EUR-Lex are caught explicitly and skipped silently.

Two ways to run:
  - One-shot:    `python scripts/run_ingestion_flow.py`
  - Scheduled:   `python scripts/serve_ingestion_schedule.py`  (long-lived, cron-based)
"""

from __future__ import annotations

from prefect import flow, get_run_logger, task

from regflow.feeds.base import FeedConnector
from regflow.feeds.connectors.eur_lex import EurLexAntiBotError, EurLexConnector
from regflow.feeds.connectors.federal_register import FederalRegisterConnector
from regflow.feeds.pipeline import IngestResult, ingest_document
from regflow.orchestrator import run_workflow

_SOURCES = ("federal_register", "eur_lex")


def _make_connector(source: str) -> FeedConnector:
    if source == "federal_register":
        return FederalRegisterConnector()
    if source == "eur_lex":
        return EurLexConnector()
    raise ValueError(f"Unknown source: {source}")


@task(retries=2, retry_delay_seconds=30)
def discover_source(source: str) -> list[str]:
    """Run the source's discovery query. Returns list of source_doc_ids."""
    logger = get_run_logger()
    connector = _make_connector(source)
    try:
        ids = connector.discover()
        logger.info(f"[{source}] discovered {len(ids)} candidate documents")
        return ids
    finally:
        connector.close()


@task(retries=2, retry_delay_seconds=60)
def ingest_one(source: str, source_doc_id: str) -> IngestResult | None:
    """Fetch + ingest one document. Returns None on per-doc failure so the flow continues."""
    logger = get_run_logger()
    connector = _make_connector(source)
    try:
        result = ingest_document(connector, source_doc_id)
        logger.info(
            f"[{source}] {source_doc_id}: ingest done "
            f"(new_version={result.is_new_version}, articles={result.article_count})"
        )
        return result
    except EurLexAntiBotError as exc:
        logger.warning(f"[{source}] {source_doc_id}: anti-bot challenge — skipping. {exc}")
        return None
    except Exception as exc:        # noqa: BLE001 — per-doc isolation: log + continue
        logger.error(f"[{source}] {source_doc_id}: ingest failed: {exc}")
        return None
    finally:
        connector.close()


@task(retries=1, retry_delay_seconds=30)
def run_downstream(document_id) -> None:
    """For a freshly-ingested document, run Agent 1 + Orchestrator + Agent 2.

    No include_prefixes filter: it was EUR-Lex-specific (art_*) and silently zeroed out
    every Federal Register document (whose article_refs are heading text, not `art_N`).
    Cosmetic-noise filtering still happens inside Agent 1's severity classifier — the
    architecturally correct place for it — not at the article-ref level.
    """
    logger = get_run_logger()
    result = run_workflow(document_id)
    logger.info(
        f"workflow done: events={len(result.events)} "
        f"obligations={len(result.obligations)} review_log_entries={result.review_log_entries_written}"
    )


@flow(name="regflow_ingestion", log_prints=True)
def regflow_ingestion_flow(sources: tuple[str, ...] = _SOURCES) -> None:
    """Discovery -> ingest -> trigger downstream workflow for substantive changes only.

    Sequential by source; sequential within a source. Add parallelism via task.submit()
    when source counts and document counts grow.
    """
    logger = get_run_logger()
    triggered = 0
    skipped = 0

    for source in sources:
        try:
            doc_ids = discover_source(source)
        except Exception as exc:        # noqa: BLE001 — bad source isolated
            logger.error(f"[{source}] discovery failed: {exc}. Skipping source.")
            continue

        for doc_id in doc_ids:
            result = ingest_one(source, doc_id)
            if result is None:
                skipped += 1
                continue
            if result.is_new_version:
                run_downstream(result.document_id)
                triggered += 1
            else:
                logger.info(f"[{source}] {doc_id}: unchanged — no downstream work")

    logger.info(f"flow summary: triggered_workflows={triggered} skipped_failures={skipped}")

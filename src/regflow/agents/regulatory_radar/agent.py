"""Agent 1: Regulatory Radar. Wires delta detection (Phase 1) + severity classification (Phase 2).

Public entry: `run_radar(document_id)`. Given a newly-ingested document, returns a list of
RegulatoryChangeEvent — one per substantive change that survived filtering.

The orchestrator calls this and then routes events into the downstream pipeline.
"""

from __future__ import annotations

import hashlib
from uuid import UUID

from sqlalchemy import select

from regflow.agents.regulatory_radar.delta_detector import ArticleDelta, compute_deltas
from regflow.agents.regulatory_radar.severity_classifier import (
    SeverityClassification,
    classify_delta,
    fallback_classification,
)
from regflow.common.llm import LLMError
from regflow.common.logging import get_logger
from regflow.common.types import RegulatoryChangeEvent, Severity
from regflow.db.postgres import Article, Document, get_session
from regflow.rag import retrieve_corrections

log = get_logger(__name__)

_DROP_SEVERITIES = {Severity.COSMETIC}   # architecture line 27: "FILTERS OUT: cosmetic changes"

# Default exclusions for EUR-Lex umbrella blocks that contain (and duplicate) their children.
# The 'pbl_' (preamble) block in particular bundles all citations + recitals into a single
# multi-thousand-token chunk that would blow the LLM's context window.
_DEFAULT_EXCLUDE_PREFIXES: tuple[str, ...] = ("pbl_",)


def run_radar(
    document_id: UUID,
    *,
    limit: int | None = None,
    include_prefixes: tuple[str, ...] | None = None,
    exclude_prefixes: tuple[str, ...] = _DEFAULT_EXCLUDE_PREFIXES,
) -> list[RegulatoryChangeEvent]:
    """Diff the given document against its prior version, classify changes, emit events.

    Args:
        document_id: which document to analyze.
        limit: process at most N deltas after filtering. None = no limit.
        include_prefixes: if set, keep only deltas whose article_ref starts with one of these
                          (e.g. ("art_",) to restrict to article blocks only).
        exclude_prefixes: drop deltas whose article_ref starts with one of these.
                          Default excludes the preamble umbrella (`pbl_`).
    On first ingest (no prior version), every article is treated as ADDED.
    """
    with get_session() as session:
        current = session.get(Document, document_id)
        if current is None:
            raise ValueError(f"document_id {document_id} not found")

        prior = _find_prior_version(session, current)
        deltas = compute_deltas(current.articles, prior.articles if prior else None)
        deltas = _apply_filters(deltas, include_prefixes, exclude_prefixes, limit)

        log.info(
            "radar.deltas_computed",
            document_id=str(document_id),
            source=current.source,
            source_doc_id=current.source_doc_id,
            prior_version_id=str(prior.id) if prior else None,
            delta_count=len(deltas),
            change_types={d.change_type.value for d in deltas},
            limit=limit,
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
        )

        if not deltas:
            return []

        events: list[RegulatoryChangeEvent] = []
        for delta in deltas:
            classification = _classify_safely(delta)
            if classification.severity in _DROP_SEVERITIES:
                continue
            events.append(_build_event(current, prior, delta, classification))

        log.info(
            "radar.events_emitted",
            document_id=str(document_id),
            emitted=len(events),
            filtered_out=len(deltas) - len(events),
        )
        return events


def _apply_filters(
    deltas: list[ArticleDelta],
    include_prefixes: tuple[str, ...] | None,
    exclude_prefixes: tuple[str, ...],
    limit: int | None,
) -> list[ArticleDelta]:
    out = deltas
    if include_prefixes:
        out = [d for d in out if d.article_ref.startswith(include_prefixes)]
    if exclude_prefixes:
        out = [d for d in out if not d.article_ref.startswith(exclude_prefixes)]
    if limit is not None:
        out = out[:limit]
    return out


def _find_prior_version(session, current: Document) -> Document | None:
    """Latest row for the same (source, source_doc_id) with a different content_hash."""
    stmt = (
        select(Document)
        .where(
            Document.source == current.source,
            Document.source_doc_id == current.source_doc_id,
            Document.content_hash != current.content_hash,
            Document.id != current.id,
        )
        .order_by(Document.fetched_at.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def _classify_safely(delta: ArticleDelta) -> SeverityClassification:
    # Override Store retrieval: past reviewer corrections on similar diffs become
    # few-shot anti-examples in the classification prompt. Empty result is fine
    # (cold-start case); classifier degrades to vanilla behavior.
    corrections = retrieve_corrections(
        delta.diff_text or delta.new_text or delta.old_text or "",
        agent_id="agent_1",
        top_k=3,
    )
    if corrections:
        log.debug(
            "radar.corrections_retrieved",
            article_ref=delta.article_ref,
            count=len(corrections),
            top_distance=corrections[0].distance,
        )
    try:
        return classify_delta(delta, corrections=corrections)
    except LLMError as exc:
        log.warning(
            "radar.classification_fallback",
            article_ref=delta.article_ref,
            change_type=delta.change_type.value,
            error=str(exc),
        )
        return fallback_classification(delta)


def _build_event(
    current: Document,
    prior: Document | None,
    delta: ArticleDelta,
    classification: SeverityClassification,
) -> RegulatoryChangeEvent:
    """ADDED / REMOVED deltas may not have a current article_id — fall back to a synthetic id."""
    article_id = delta.article_id or _synthetic_article_id(current.id, delta.article_ref)
    return RegulatoryChangeEvent(
        article_id=article_id,
        document_id=str(current.id),
        severity=classification.severity,
        jurisdiction=current.jurisdiction,
        regulator=current.regulator,
        confidence=classification.confidence,
        diff_summary=classification.diff_summary,
        prior_version_hash=prior.content_hash if prior else None,
        new_version_hash=current.content_hash,
    )


def _synthetic_article_id(document_id: UUID, article_ref: str) -> str:
    """Stable id for REMOVED articles (no current row exists) so events still reference something."""
    return hashlib.sha1(f"{document_id}:{article_ref}".encode()).hexdigest()

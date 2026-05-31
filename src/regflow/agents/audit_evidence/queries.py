"""Data fetches for Agent 6: assemble everything an auditor would want for one obligation."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select

from regflow.common.logging import get_logger
from regflow.db.postgres import (
    Article,
    Document,
    GapRow,
    ObligationRow,
    ReviewLogEntry,
    get_session,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class EvidenceContext:
    obligation: ObligationRow
    article: Article | None              # source article — for the exact text span
    document: Document | None             # source document — for citation header
    latest_gap: GapRow | None             # may be None if Agent 4 hasn't run yet
    review_log_entries: list[ReviewLogEntry]


def load_evidence_context(obligation_id: UUID) -> EvidenceContext | None:
    with get_session() as session:
        obligation = session.get(ObligationRow, obligation_id)
        if obligation is None:
            log.warning("audit_evidence.obligation_not_found", obligation_id=str(obligation_id))
            return None

        article = session.get(Article, obligation.article_id)
        document = session.get(Document, obligation.document_id)

        # Newest Gap (if any) — Agent 4's output, source of matching_controls.
        latest_gap = session.execute(
            select(GapRow)
            .where(GapRow.obligation_id == obligation_id)
            .order_by(GapRow.analyzed_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        # Review log entries directly tied to this obligation (Agent 1/2 triggers).
        review_entries = list(
            session.execute(
                select(ReviewLogEntry)
                .where(ReviewLogEntry.subject_id == obligation_id)
                .order_by(ReviewLogEntry.created_at.asc())
            ).scalars()
        )

        session.expunge_all()
        return EvidenceContext(
            obligation=obligation,
            article=article,
            document=document,
            latest_gap=latest_gap,
            review_log_entries=review_entries,
        )

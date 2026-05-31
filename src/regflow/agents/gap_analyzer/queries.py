"""Data fetches for Agent 4: the enterprise context bridges regulation -> company.

For v0.1 we load ALL controls and ALL findings every call. That's fine when the
enterprise context fits in the LLM's context window (~10-30 controls, dozens of
findings). When this exceeds ~50 controls or audit-finding context overflows, switch
to RAG retrieval over Weaviate (a separate `EnterpriseControl` collection).
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select

from regflow.common.logging import get_logger
from regflow.db.postgres import EnterpriseControl, ObligationRow, PriorAuditFinding, get_session

log = get_logger(__name__)


@dataclass(frozen=True)
class ControlView:
    name: str
    description: str
    category: str
    control_owner: str | None
    evidence_exists: bool          # True if evidence_uri set


@dataclass(frozen=True)
class FindingView:
    finding_ref: str
    finding_text: str
    category: str
    status: str
    year: int


def load_obligation(obligation_id: UUID) -> ObligationRow | None:
    with get_session() as session:
        row = session.get(ObligationRow, obligation_id)
        if row is None:
            return None
        session.expunge(row)
        return row


def load_all_controls() -> list[ControlView]:
    with get_session() as session:
        rows = session.execute(select(EnterpriseControl)).scalars().all()
        return [
            ControlView(
                name=r.name,
                description=r.description,
                category=r.category,
                control_owner=r.control_owner,
                evidence_exists=bool(r.evidence_uri),
            )
            for r in rows
        ]


def load_findings_in_categories(categories: set[str]) -> list[FindingView]:
    """Return findings whose category matches any of the supplied categories.
    We narrow by category (not just dump all findings) to keep the prompt focused.
    """
    if not categories:
        return []
    with get_session() as session:
        rows = (
            session.execute(
                select(PriorAuditFinding).where(PriorAuditFinding.category.in_(categories))
            )
            .scalars()
            .all()
        )
        return [
            FindingView(
                finding_ref=r.finding_ref,
                finding_text=r.finding_text,
                category=r.category,
                status=r.status,
                year=r.year,
            )
            for r in rows
        ]

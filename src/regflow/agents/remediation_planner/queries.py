"""Data fetches for Agent 5: gap + originating obligation + available owners."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select

from regflow.common.logging import get_logger
from regflow.db.postgres import (
    EnterpriseControl,
    GapRow,
    ObligationRow,
    get_session,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class GapContext:
    """Everything Agent 5 needs as input — one immutable bundle."""

    gap: GapRow
    obligation: ObligationRow
    available_owners: list[str]
    matching_controls: list[str]
    missing_or_weak_controls: list[str]


def load_gap_context(gap_id: UUID) -> GapContext | None:
    with get_session() as session:
        gap = session.get(GapRow, gap_id)
        if gap is None:
            log.warning("remediation.gap_not_found", gap_id=str(gap_id))
            return None

        obligation = session.get(ObligationRow, gap.obligation_id)
        if obligation is None:
            log.warning("remediation.obligation_not_found", obligation_id=str(gap.obligation_id))
            return None

        # Distinct owners across the enterprise context — the LLM picks from this universe.
        owners = (
            session.execute(
                select(EnterpriseControl.control_owner)
                .where(EnterpriseControl.control_owner.isnot(None))
                .distinct()
            )
            .scalars()
            .all()
        )

        session.expunge_all()
        return GapContext(
            gap=gap,
            obligation=obligation,
            available_owners=sorted(o for o in owners if o),
            matching_controls=list(gap.matching_controls or []),
            missing_or_weak_controls=list(gap.missing_or_weak_controls or []),
        )

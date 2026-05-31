"""Agent 5 — Remediation Planner (architecture lines 102-108).

Public contract:
    plan_remediation_for_gap(gap_id) -> list[RemediationAction]

Per-gap execution. Produces a small ordered set of concrete actions, each with
an owner (from the enterprise's universe of control owners), deadline, priority,
proposed control updates, and dependency notes.

Persistence: Postgres RemediationActionRow + Neo4j RemediationAction nodes with
(Gap)-[:HAS_ACTION]->(Action) edges.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from regflow.agents.remediation_planner.extractor import (
    PlannedAction,
    RemediationPlan,
    plan,
)
from regflow.agents.remediation_planner.queries import GapContext, load_gap_context
from regflow.common.llm import LLMError
from regflow.common.logging import get_logger
from regflow.common.types import CorrectionType, RemediationAction
from regflow.db.neo4j import upsert_remediation_action_node
from regflow.db.postgres import RemediationActionRow, get_session
from regflow.rag import retrieve_corrections

log = get_logger(__name__)


def plan_remediation_for_gap(gap_id: UUID) -> list[RemediationAction]:
    ctx = load_gap_context(gap_id)
    if ctx is None:
        return []

    # Override Store retrieval — past reviewer corrections to remediation plans,
    # especially `owner_reassignment` type (architecture line 106: "incorporates
    # past reviewer overrides for owner suggestions").
    correction_query = f"{ctx.obligation.obligation_text}\n{ctx.gap.rationale or ''}"
    corrections = retrieve_corrections(
        correction_query,
        agent_id="agent_5",
        top_k=3,
        correction_type=CorrectionType.OWNER_REASSIGNMENT,
    )
    if corrections:
        log.info(
            "remediation.corrections_retrieved",
            gap_id=str(gap_id),
            count=len(corrections),
            top_distance=corrections[0].distance,
        )

    try:
        result: RemediationPlan = plan(ctx, corrections=corrections)
    except LLMError as exc:
        log.warning("remediation.llm_failure", gap_id=str(gap_id), error=str(exc))
        return []

    if not result.actions:
        log.info("remediation.no_actions", gap_id=str(gap_id))
        return []

    # Constrain owners to the supplied universe (defensive: if the LLM invented one,
    # null it out so the reviewer can manually assign).
    valid_owners = set(ctx.available_owners)
    domain_actions: list[tuple[RemediationAction, PlannedAction]] = []
    for pa in result.actions:
        owner = pa.suggested_owner if pa.suggested_owner in valid_owners else None
        action = RemediationAction(
            gap_id=ctx.gap.id,
            description=pa.description,
            suggested_owner=owner,
            suggested_deadline=None,                    # raw string kept in DB; datetime parse is v2
            proposed_control_updates=pa.proposed_control_updates,
            dependencies=[],                            # description-only for v0.1
            priority=pa.priority,
            confidence=pa.confidence,
        )
        domain_actions.append((action, pa))

    _persist(ctx.gap.id, ctx.gap.obligation_id, domain_actions)

    log.info(
        "remediation.planned",
        gap_id=str(gap_id),
        action_count=len(domain_actions),
        owners_assigned=sum(1 for a, _ in domain_actions if a.suggested_owner),
        avg_priority=round(sum(a.priority for a, _ in domain_actions) / len(domain_actions), 2),
        avg_confidence=round(sum(a.confidence for a, _ in domain_actions) / len(domain_actions), 2),
    )
    return [a for a, _ in domain_actions]


def _persist(
    gap_id: UUID,
    obligation_id: UUID,
    actions: list[tuple[RemediationAction, PlannedAction]],
) -> None:
    """Postgres rows first (source of truth), then Neo4j edges best-effort."""
    created_at = datetime.utcnow()
    with get_session() as session:
        for action, pa in actions:
            session.add(
                RemediationActionRow(
                    id=action.action_id,
                    gap_id=gap_id,
                    obligation_id=obligation_id,
                    description=action.description,
                    suggested_owner=action.suggested_owner,
                    suggested_deadline=pa.suggested_deadline,
                    proposed_control_updates=action.proposed_control_updates,
                    dependency_descriptions=pa.dependency_descriptions,
                    priority=action.priority,
                    confidence=action.confidence,
                    rationale=pa.rationale,
                    created_at=created_at,
                )
            )

    iso = created_at.isoformat()
    for action, pa in actions:
        try:
            upsert_remediation_action_node(
                action_id=str(action.action_id),
                gap_id=str(gap_id),
                description=action.description,
                suggested_owner=action.suggested_owner,
                suggested_deadline=pa.suggested_deadline,
                priority=action.priority,
                confidence=action.confidence,
                created_at=iso,
            )
        except Exception as exc:        # noqa: BLE001
            log.warning(
                "remediation.neo4j_edge_failed",
                action_id=str(action.action_id),
                error=str(exc),
            )

"""Neo4j writes — v0.1 scope: Obligation nodes only.

Edges (Document)-[:CONTAINS]->(Article)-[:HAS_OBLIGATION]->(Obligation)-[:REQUIRES]->(Control)
come in a follow-up iteration. For now, nodes are enough to demonstrate that the graph layer
is wired in and Agent 2 writes are landing.
"""

from __future__ import annotations

from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.common.types import Obligation
from regflow.db.neo4j.connection import get_driver

log = get_logger(__name__)


_CONSTRAINTS = [
    "CREATE CONSTRAINT obligation_id         IF NOT EXISTS FOR (o:Obligation)         REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT article_id            IF NOT EXISTS FOR (a:Article)            REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT document_id           IF NOT EXISTS FOR (d:Document)           REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT gap_id                IF NOT EXISTS FOR (g:Gap)                REQUIRE g.id IS UNIQUE",
    "CREATE CONSTRAINT remediation_action_id IF NOT EXISTS FOR (a:RemediationAction)  REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT audit_evidence_id     IF NOT EXISTS FOR (e:AuditEvidence)      REQUIRE e.id IS UNIQUE",
]


def init_constraints() -> None:
    """Idempotent — safe to run on every boot of init_infra.py."""
    settings = get_settings().neo4j
    with get_driver().session(database=settings.database) as session:
        for stmt in _CONSTRAINTS:
            session.run(stmt)
    log.info("neo4j.constraints_initialized")


def upsert_conflict_edge(
    obligation_a_id: str,
    obligation_b_id: str,
    *,
    conflict_type: str,
    severity: str,
    confidence: float,
    description: str,
    detected_at: str,
) -> None:
    """Write or update a CONFLICTS_WITH edge between two existing Obligation nodes.

    MATCH (both must exist) -> MERGE the edge -> SET its properties. If either
    obligation node is missing, the edge isn't created (no orphan edges).
    """
    settings = get_settings().neo4j
    with get_driver().session(database=settings.database) as session:
        session.run(
            """
            MATCH (a:Obligation {id: $a_id})
            WITH a
            MATCH (b:Obligation {id: $b_id})
            MERGE (a)-[r:CONFLICTS_WITH]->(b)
            SET r.type        = $type,
                r.severity    = $severity,
                r.confidence  = $confidence,
                r.description = $description,
                r.detected_at = $detected_at
            """,
            {
                "a_id": obligation_a_id,
                "b_id": obligation_b_id,
                "type": conflict_type,
                "severity": severity,
                "confidence": confidence,
                "description": description,
                "detected_at": detected_at,
            },
        )


def upsert_gap_node(
    gap_id: str,
    obligation_id: str,
    *,
    risk_score: float,
    risk_level: str,
    missing_or_weak_controls: list[str],
    matching_controls: list[str],
    enforcement_severity: float,
    business_impact: float,
    deadline_urgency: float,
    evidence_exists: bool,
    confidence: float,
    rationale: str,
    analyzed_at: str,
) -> None:
    """Create the Gap node AND the (Obligation)-[:HAS_GAP]->(Gap) edge in one Cypher.
    Pipelined with `WITH` to avoid cartesian-product planner warnings.
    """
    settings = get_settings().neo4j
    with get_driver().session(database=settings.database) as session:
        session.run(
            """
            MATCH (o:Obligation {id: $obligation_id})
            WITH o
            MERGE (g:Gap {id: $gap_id})
            SET g.risk_score              = $risk_score,
                g.risk_level              = $risk_level,
                g.missing_or_weak_controls= $missing_or_weak_controls,
                g.matching_controls       = $matching_controls,
                g.enforcement_severity    = $enforcement_severity,
                g.business_impact         = $business_impact,
                g.deadline_urgency        = $deadline_urgency,
                g.evidence_exists         = $evidence_exists,
                g.confidence              = $confidence,
                g.rationale               = $rationale,
                g.analyzed_at             = $analyzed_at
            MERGE (o)-[:HAS_GAP]->(g)
            """,
            {
                "gap_id": gap_id,
                "obligation_id": obligation_id,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "missing_or_weak_controls": missing_or_weak_controls,
                "matching_controls": matching_controls,
                "enforcement_severity": enforcement_severity,
                "business_impact": business_impact,
                "deadline_urgency": deadline_urgency,
                "evidence_exists": evidence_exists,
                "confidence": confidence,
                "rationale": rationale,
                "analyzed_at": analyzed_at,
            },
        )


def upsert_remediation_action_node(
    action_id: str,
    gap_id: str,
    *,
    description: str,
    suggested_owner: str | None,
    suggested_deadline: str | None,
    priority: int,
    confidence: float,
    created_at: str,
) -> None:
    """Create RemediationAction node + (Gap)-[:HAS_ACTION]->(Action) edge.
    `WITH g` pipeline avoids cartesian-product planner warnings."""
    settings = get_settings().neo4j
    with get_driver().session(database=settings.database) as session:
        session.run(
            """
            MATCH (g:Gap {id: $gap_id})
            WITH g
            MERGE (a:RemediationAction {id: $action_id})
            SET a.description         = $description,
                a.suggested_owner     = $suggested_owner,
                a.suggested_deadline  = $suggested_deadline,
                a.priority            = $priority,
                a.confidence          = $confidence,
                a.created_at          = $created_at
            MERGE (g)-[:HAS_ACTION]->(a)
            """,
            {
                "action_id": action_id,
                "gap_id": gap_id,
                "description": description,
                "suggested_owner": suggested_owner,
                "suggested_deadline": suggested_deadline,
                "priority": priority,
                "confidence": confidence,
                "created_at": created_at,
            },
        )


def upsert_audit_evidence_node(
    evidence_id: str,
    obligation_id: str,
    *,
    justification: str,
    confidence: float,
    control_links: list[str],
    generated_at: str,
) -> None:
    """Create AuditEvidence node + (Obligation)-[:HAS_EVIDENCE]->(AuditEvidence) edge."""
    settings = get_settings().neo4j
    with get_driver().session(database=settings.database) as session:
        session.run(
            """
            MATCH (o:Obligation {id: $obligation_id})
            WITH o
            MERGE (e:AuditEvidence {id: $evidence_id})
            SET e.justification = $justification,
                e.confidence    = $confidence,
                e.control_links = $control_links,
                e.generated_at  = $generated_at
            MERGE (o)-[:HAS_EVIDENCE]->(e)
            """,
            {
                "evidence_id": evidence_id,
                "obligation_id": obligation_id,
                "justification": justification[:2000],
                "confidence": confidence,
                "control_links": control_links,
                "generated_at": generated_at,
            },
        )


def upsert_obligation_node(obligation: Obligation) -> None:
    """MERGE so reruns don't create duplicates; SET writes properties on every call."""
    settings = get_settings().neo4j
    with get_driver().session(database=settings.database) as session:
        session.run(
            """
            MERGE (o:Obligation {id: $id})
            SET o.text          = $text,
                o.type          = $type,
                o.scope         = $scope,
                o.jurisdiction  = $jurisdiction,
                o.regulator     = $regulator,
                o.confidence    = $confidence,
                o.deadlines     = $deadlines,
                o.penalties     = $penalties,
                o.exemptions    = $exemptions,
                o.article_id    = $article_id,
                o.document_id   = $document_id,
                o.extracted_at  = $extracted_at
            """,
            {
                "id": str(obligation.obligation_id),
                "text": obligation.obligation_text,
                "type": obligation.obligation_type,
                "scope": obligation.scope,
                "jurisdiction": obligation.jurisdiction,
                "regulator": obligation.regulator,
                "confidence": obligation.confidence,
                "deadlines": obligation.deadlines,
                "penalties": obligation.penalties,
                "exemptions": obligation.exemptions,
                "article_id": obligation.article_id,
                "document_id": obligation.document_id,
                "extracted_at": obligation.extracted_at.isoformat(),
            },
        )

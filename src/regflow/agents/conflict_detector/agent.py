"""Agent 3 — Cross-Jurisdiction Conflict Detector (architecture lines 78-86).

Public contract:
    detect_conflicts_for_article(article_id) -> list[Conflict]

Batch mode (architecture line 66): receives all N obligations for an article as a SET,
plus their cross-jurisdiction neighbors, makes ONE LLM call, returns all detected
conflicts. Different shape from Agent 2 (per-obligation) by design.

Persistence:
  - Postgres: one ConflictRow per detected conflict pair
  - Neo4j:    (Obligation)-[:CONFLICTS_WITH]->(Obligation) edge per pair
  - Review log: a high-severity conflict (architecture line 86) gets logged with
                trigger='high_severity_conflict' for legal signoff
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from regflow.agents.conflict_detector.detector import detect_conflicts
from regflow.agents.conflict_detector.queries import (
    find_cross_jurisdiction_neighbors,
    load_source_obligations,
)
from regflow.common.llm import LLMError
from regflow.common.logging import get_logger
from regflow.common.types import Conflict, CorrectionType, Severity
from regflow.db.neo4j import upsert_conflict_edge
from regflow.db.postgres import ConflictRow, ReviewLogEntry, get_session
from regflow.rag import retrieve_corrections

log = get_logger(__name__)


def detect_conflicts_for_article(article_id: UUID) -> list[Conflict]:
    sources = load_source_obligations(article_id)
    if len(sources) == 0:
        log.info("conflict.no_source_obligations", article_id=str(article_id))
        return []

    neighbors = find_cross_jurisdiction_neighbors(sources)
    if not neighbors:
        log.info("conflict.no_neighbors_found", article_id=str(article_id))
        return []

    # Override Store retrieval — past reviewer corrections on similar conflict
    # candidates. We embed the joined source obligation texts so similar batches
    # retrieve the same lessons.
    correction_query = "\n".join(s.obligation_text for s in sources[:5])[:2000]
    corrections = retrieve_corrections(
        correction_query,
        agent_id="agent_3",
        top_k=3,
        correction_type=CorrectionType.FALSE_POSITIVE_CONFLICT,
    )
    if corrections:
        log.info(
            "conflict.corrections_retrieved",
            article_id=str(article_id),
            count=len(corrections),
            top_distance=corrections[0].distance,
        )

    try:
        result = detect_conflicts(sources, neighbors, corrections=corrections)
    except LLMError as exc:
        log.warning("conflict.llm_failure", article_id=str(article_id), error=str(exc))
        return []

    if not result.conflicts:
        log.info("conflict.none_detected", article_id=str(article_id))
        return []

    by_id = {c.obligation_id: c for c in (sources + neighbors)}
    conflicts: list[Conflict] = []
    high_severity: list[Conflict] = []

    for det in result.conflicts:
        a_data = by_id.get(det.obligation_a_id)
        b_data = by_id.get(det.obligation_b_id)
        if a_data is None or b_data is None:
            log.warning(
                "conflict.unknown_id_in_llm_output",
                a_id=det.obligation_a_id,
                b_id=det.obligation_b_id,
            )
            continue

        conflict = Conflict(
            conflict_type=det.conflict_type,
            obligation_ids=[UUID(det.obligation_a_id), UUID(det.obligation_b_id)],
            description=det.description,
            severity=Severity(det.severity),
            confidence=det.confidence,
        )
        conflicts.append(conflict)
        if conflict.severity == Severity.MAJOR:
            high_severity.append(conflict)

    _persist(conflicts, by_id, high_severity)

    log.info(
        "conflict.detected",
        article_id=str(article_id),
        total=len(conflicts),
        high_severity=len(high_severity),
    )
    return conflicts


def _persist(
    conflicts: list[Conflict],
    by_id: dict[str, "CandidateObligation"],  # type: ignore[name-defined]
    high_severity: list[Conflict],
) -> None:
    """Write to Postgres + Neo4j; route MAJOR conflicts to review_log (architecture line 86)."""
    if not conflicts:
        return

    with get_session() as session:
        for c in conflicts:
            a_id, b_id = c.obligation_ids
            a_data = by_id[str(a_id)]
            b_data = by_id[str(b_id)]
            session.add(
                ConflictRow(
                    obligation_a_id=a_id,
                    obligation_b_id=b_id,
                    conflict_type=c.conflict_type,
                    severity=c.severity.value,
                    description=c.description,
                    confidence=c.confidence,
                    jurisdiction_a=a_data.jurisdiction,
                    jurisdiction_b=b_data.jurisdiction,
                )
            )
        # Route high-severity to mandatory legal review per architecture line 86.
        for c in high_severity:
            session.add(
                ReviewLogEntry(
                    trigger="high_severity_conflict",
                    agent_id="agent_3",
                    subject_type="conflict",
                    subject_id=uuid4(),
                    payload={
                        "conflict_type": c.conflict_type,
                        "severity": c.severity.value,
                        "confidence": c.confidence,
                        "description": c.description,
                        "obligation_a_id": str(c.obligation_ids[0]),
                        "obligation_b_id": str(c.obligation_ids[1]),
                    },
                )
            )

    # Neo4j edges (best-effort; conflicts already persisted in Postgres above).
    detected_at_iso = datetime.utcnow().isoformat()
    for c in conflicts:
        a_id, b_id = c.obligation_ids
        try:
            upsert_conflict_edge(
                str(a_id),
                str(b_id),
                conflict_type=c.conflict_type,
                severity=c.severity.value,
                confidence=c.confidence,
                description=c.description,
                detected_at=detected_at_iso,
            )
        except Exception as exc:    # noqa: BLE001
            log.warning("conflict.neo4j_edge_failed", a_id=str(a_id), b_id=str(b_id), error=str(exc))


# Re-import for the type annotation in _persist (kept at bottom to avoid circular import noise).
from regflow.agents.conflict_detector.queries import CandidateObligation  # noqa: E402

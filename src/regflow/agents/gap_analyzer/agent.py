"""Agent 4 — Gap Analyzer & Risk Scorer (architecture lines 89-99).

Public contract:
    analyze_gap_for_obligation(obligation_id) -> Gap | None

Per-obligation execution (parallel-eligible across obligations). Inputs:
  - the obligation
  - ALL enterprise controls (small corpus today; switch to RAG when it grows)
  - prior audit findings filtered to the obligation's likely category space

Output: a Gap object with:
  - matching_controls + missing_or_weak_controls
  - risk factors (enforcement_severity * business_impact * deadline_urgency)
  - HIGH / MEDIUM / LOW risk_level (thresholded)
  - related_audit_findings

Persistence: Postgres GapRow + Neo4j Gap node with (Obligation)-[:HAS_GAP]->(Gap).
High-risk gaps routed to review_log for compliance approval (architecture line 99).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from regflow.agents.gap_analyzer.extractor import GapAnalysis, analyze_gap
from regflow.agents.gap_analyzer.queries import (
    ControlView,
    FindingView,
    load_all_controls,
    load_findings_in_categories,
    load_obligation,
)
from regflow.common.llm import LLMError
from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.common.types import CorrectionType, Gap, Obligation, RiskLevel, SourceCitation
from regflow.db.neo4j import upsert_gap_node
from regflow.db.postgres import GapRow, ObligationRow, ReviewLogEntry, get_session
from regflow.rag import retrieve_corrections

log = get_logger(__name__)


def analyze_gap_for_obligation(obligation_id: UUID) -> Gap | None:
    obligation_row = load_obligation(obligation_id)
    if obligation_row is None:
        log.warning("gap.obligation_not_found", obligation_id=str(obligation_id))
        return None

    obligation = _row_to_obligation(obligation_row)

    controls = load_all_controls()
    if not controls:
        log.warning("gap.no_controls_loaded — load enterprise context first (scripts/load_enterprise_context.py)")
        return None

    # Findings: narrow by the obligation's likely category space so the LLM stays focused.
    categories_of_interest = _infer_categories(obligation)
    findings = load_findings_in_categories(categories_of_interest)

    # Override Store retrieval — past reviewer corrections to gap analyses (e.g.,
    # `risk_override` type when the reviewer adjusted a risk_level).
    corrections = retrieve_corrections(
        obligation.obligation_text,
        agent_id="agent_4",
        top_k=3,
        correction_type=CorrectionType.RISK_OVERRIDE,
    )
    if corrections:
        log.info(
            "gap.corrections_retrieved",
            obligation_id=str(obligation_id),
            count=len(corrections),
            top_distance=corrections[0].distance,
        )

    try:
        analysis: GapAnalysis = analyze_gap(obligation, controls, findings, corrections=corrections)
    except LLMError as exc:
        log.warning("gap.llm_failure", obligation_id=str(obligation_id), error=str(exc))
        return None

    gap = _to_gap(obligation, analysis)
    _persist(gap, analysis)

    log.info(
        "gap.analyzed",
        obligation_id=str(obligation_id),
        risk_level=gap.risk_level.value,
        risk_score=round(gap.risk_score, 3),
        missing_controls=len(gap.missing_or_weak_controls),
        confidence=gap.confidence,
    )
    return gap


# ---------- helpers ----------


def _row_to_obligation(row: ObligationRow) -> Obligation:
    """Bridge ORM row -> Pydantic Obligation. Same shape Agent 2 produces."""
    citations = [SourceCitation(**c) for c in (row.citations or [])] if row.citations else []
    return Obligation(
        obligation_id=row.id,
        article_id=str(row.article_id),
        document_id=str(row.document_id),
        jurisdiction=row.jurisdiction,
        regulator=row.regulator,
        obligation_text=row.obligation_text,
        obligation_type=row.obligation_type,
        scope=row.scope,
        deadlines=row.deadlines or [],
        penalties=row.penalties or [],
        exemptions=row.exemptions or [],
        citations=citations,
        confidence=row.confidence,
        extracted_at=row.extracted_at,
    )


def _infer_categories(obligation: Obligation) -> set[str]:
    """Map the obligation's regulator + obligation_type to enterprise-control categories.
    Keeps findings retrieval focused; broader than strictly necessary on purpose so the
    LLM doesn't miss adjacent findings.
    """
    t = obligation.obligation_type.lower()
    categories: set[str] = set()
    if t in {"reporting", "disclosure"}:
        categories.update({"financial_reporting", "data_protection", "governance"})
    if t in {"retention", "consent", "governance"}:
        categories.update({"data_protection", "governance"})
    if t == "security":
        categories.update({"information_security", "data_protection", "operational_resilience"})
    # Always include the catch-alls so cross-cutting findings are visible.
    categories.update({"governance", "data_protection"})
    return categories


def _to_gap(obligation: Obligation, analysis: GapAnalysis) -> Gap:
    risk_score = analysis.enforcement_severity * analysis.business_impact * analysis.deadline_urgency
    risk_level = _risk_level_for_score(risk_score)
    return Gap(
        obligation_id=obligation.obligation_id,
        missing_or_weak_controls=analysis.missing_or_weak_controls,
        risk_score=risk_score,
        risk_level=risk_level,
        enforcement_severity=analysis.enforcement_severity,
        business_impact=analysis.business_impact,
        deadline_urgency=analysis.deadline_urgency,
        related_audit_findings=analysis.related_audit_findings,
        evidence_exists=analysis.evidence_exists,
        confidence=analysis.confidence,
    )


def _risk_level_for_score(risk_score: float) -> RiskLevel:
    """Thresholds: HIGH >= 0.4, MEDIUM >= 0.15, else LOW.
    A product of three [0,1] factors is naturally small (max 1.0, typical 0.05-0.5).
    Calibrated empirically against architecture line 99: HIGH triggers compliance approval.
    """
    s = get_settings().orchestrator
    if risk_score >= s.gap_high_risk_threshold * 0.55:    # ~0.41 with default threshold 0.75
        return RiskLevel.HIGH
    if risk_score >= 0.15:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _persist(gap: Gap, analysis: GapAnalysis) -> None:
    """Postgres + Neo4j + review_log routing for HIGH risk."""
    with get_session() as session:
        session.add(
            GapRow(
                id=gap.gap_id,
                obligation_id=gap.obligation_id,
                matching_controls=analysis.matching_controls,
                missing_or_weak_controls=gap.missing_or_weak_controls,
                related_audit_findings=gap.related_audit_findings,
                evidence_exists=gap.evidence_exists,
                enforcement_severity=gap.enforcement_severity,
                business_impact=gap.business_impact,
                deadline_urgency=gap.deadline_urgency,
                risk_score=gap.risk_score,
                risk_level=gap.risk_level.value,
                confidence=gap.confidence,
                rationale=analysis.rationale,
            )
        )
        if gap.risk_level == RiskLevel.HIGH:
            session.add(
                ReviewLogEntry(
                    trigger="high_risk_gap",
                    agent_id="agent_4",
                    subject_type="gap",
                    subject_id=gap.gap_id,
                    payload={
                        "obligation_id": str(gap.obligation_id),
                        "risk_score": gap.risk_score,
                        "risk_level": gap.risk_level.value,
                        "missing_or_weak_controls": gap.missing_or_weak_controls,
                        "related_audit_findings": gap.related_audit_findings,
                        "rationale": analysis.rationale,
                    },
                )
            )

    try:
        upsert_gap_node(
            gap_id=str(gap.gap_id),
            obligation_id=str(gap.obligation_id),
            risk_score=gap.risk_score,
            risk_level=gap.risk_level.value,
            missing_or_weak_controls=gap.missing_or_weak_controls,
            matching_controls=analysis.matching_controls,
            enforcement_severity=gap.enforcement_severity,
            business_impact=gap.business_impact,
            deadline_urgency=gap.deadline_urgency,
            evidence_exists=gap.evidence_exists,
            confidence=gap.confidence,
            rationale=analysis.rationale,
            analyzed_at=datetime.utcnow().isoformat(),
        )
    except Exception as exc:        # noqa: BLE001 — Postgres has it; graph is best-effort
        log.warning("gap.neo4j_write_failed", gap_id=str(gap.gap_id), error=str(exc))

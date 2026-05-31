"""Relational schema for the raw document store, review log, and override-store correction records.

Architecture mapping:
- Document / Article: raw document store (line 218-221)
- CorrectionRecordRow: override store structured-record half (line 233-243)
                       The embedded vector half lives in Weaviate (override collection).
- ReviewLogEntry: human review log (line 116, 128)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy import Column  # noqa: F401  (used by some auto-mapping helpers downstream)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSONB,
        list[Any]: JSONB,
    }


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source: Mapped[str] = mapped_column(String(64), index=True)              # "eur_lex", "fca", etc.
    source_doc_id: Mapped[str] = mapped_column(String(128), index=True)      # e.g. CELEX number
    title: Mapped[str] = mapped_column(Text)
    jurisdiction: Mapped[str] = mapped_column(String(32), index=True)        # "EU", "UK", "US"
    regulator: Mapped[str] = mapped_column(String(64))                       # e.g. "European Parliament"
    document_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    published_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(8), default="en")

    raw_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)  # MinIO key
    content_hash: Mapped[str] = mapped_column(String(64), index=True)        # sha256 of canonical text
    content_length: Mapped[int] = mapped_column(BigInteger, default=0)

    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    articles: Mapped[list[Article]] = relationship(back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_source_doc", "source", "source_doc_id"),
    )


class Article(Base):
    """Article-level segmentation. One row per article version (line 12, 13)."""

    __tablename__ = "articles"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    article_ref: Mapped[str] = mapped_column(String(64))                     # "Article 1", "Annex II", etc.
    sequence: Mapped[int] = mapped_column(Integer)                           # order within document
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)

    document: Mapped[Document] = relationship(back_populates="articles")

    __table_args__ = (
        Index("ix_articles_doc_ref", "document_id", "article_ref"),
    )


class ObligationRow(Base):
    """Relational mirror of Agent 2's output. The canonical store is Neo4j (graph);
    this table exists for fast tabular queries ('show me everything with a deadline this quarter')
    and for foreign-key joins with the rest of the relational schema.
    """

    __tablename__ = "obligations"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    article_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    obligation_text: Mapped[str] = mapped_column(Text)
    obligation_type: Mapped[str] = mapped_column(String(32), index=True)
    scope: Mapped[str | None] = mapped_column(Text, nullable=True)
    jurisdiction: Mapped[str] = mapped_column(String(32), index=True)
    regulator: Mapped[str] = mapped_column(String(64))

    deadlines: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    penalties: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    exemptions: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    citations: Mapped[list[Any]] = mapped_column(JSONB, default=list)

    confidence: Mapped[float] = mapped_column(Float, index=True)
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)


class ConflictRow(Base):
    """Relational mirror of Agent 3's output (architecture lines 78-86).

    The canonical store is Neo4j (graph edges between Obligation nodes); this row
    exists for tabular queries and audit. obligation_a_id and obligation_b_id together
    identify a directed conflict pair.
    """

    __tablename__ = "conflicts"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    obligation_a_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("obligations.id", ondelete="CASCADE"), index=True)
    obligation_b_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("obligations.id", ondelete="CASCADE"), index=True)

    conflict_type: Mapped[str] = mapped_column(String(32), index=True)   # contradiction / overlap / stricter_standard
    severity: Mapped[str] = mapped_column(String(16), index=True)        # minor / substantive / major
    description: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, index=True)

    jurisdiction_a: Mapped[str] = mapped_column(String(32), index=True)
    jurisdiction_b: Mapped[str] = mapped_column(String(32), index=True)

    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_conflicts_pair", "obligation_a_id", "obligation_b_id"),
    )


class EnterpriseControl(Base):
    """Architecture lines 223-231 — Enterprise Context Layer (manual upload)."""

    __tablename__ = "enterprise_controls"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), index=True)         # e.g. "data_protection", "financial_reporting"
    control_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    business_unit: Mapped[str | None] = mapped_column(String(128), nullable=True)
    evidence_uri: Mapped[str | None] = mapped_column(Text, nullable=True)  # link to evidence doc (S3 URI, ticket, etc.)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class PriorAuditFinding(Base):
    """Past findings — Agent 4 weights gaps higher when prior audits flagged the area."""

    __tablename__ = "prior_audit_findings"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    finding_ref: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # e.g. "2024-Q3-A07"
    finding_text: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), index=True)         # mirrors EnterpriseControl.category
    status: Mapped[str] = mapped_column(String(32), default="open")        # open | resolved | accepted_risk
    year: Mapped[int] = mapped_column(Integer, index=True)
    related_control_names: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class GapRow(Base):
    """Agent 4 output — relational mirror of the Gap. Neo4j has the graph view."""

    __tablename__ = "gaps"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    obligation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("obligations.id", ondelete="CASCADE"), index=True
    )

    matching_controls: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    missing_or_weak_controls: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    related_audit_findings: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    evidence_exists: Mapped[bool] = mapped_column(default=False)

    enforcement_severity: Mapped[float] = mapped_column(Float)
    business_impact: Mapped[float] = mapped_column(Float)
    deadline_urgency: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)        # HIGH | MEDIUM | LOW

    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)


class RemediationActionRow(Base):
    """Agent 5 output. One action in a remediation plan attached to a Gap."""

    __tablename__ = "remediation_actions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    gap_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("gaps.id", ondelete="CASCADE"), index=True
    )
    obligation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("obligations.id", ondelete="CASCADE"), index=True
    )

    description: Mapped[str] = mapped_column(Text)
    suggested_owner: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    suggested_deadline: Mapped[str | None] = mapped_column(String(64), nullable=True)
    proposed_control_updates: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    dependency_descriptions: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    priority: Mapped[int] = mapped_column(Integer, default=3, index=True)   # 1=highest, 5=lowest
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)


class AuditEvidenceRow(Base):
    """Agent 6 output. Self-contained 'evidence pack' for one obligation — what an
    auditor would actually want to see when reviewing compliance for that obligation."""

    __tablename__ = "audit_evidence"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    obligation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("obligations.id", ondelete="CASCADE"), unique=True, index=True
    )

    clause_citations: Mapped[list[Any]] = mapped_column(JSONB, default=list)         # from Obligation.citations
    control_links: Mapped[list[Any]] = mapped_column(JSONB, default=list)            # control names
    related_audit_findings: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    related_review_log_refs: Mapped[list[Any]] = mapped_column(JSONB, default=list)  # ReviewLogEntry ids
    open_questions: Mapped[list[Any]] = mapped_column(JSONB, default=list)           # auditor-targeted questions

    justification: Mapped[str] = mapped_column(Text)
    evidence_summary: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    confidence: Mapped[float] = mapped_column(Float, index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)


class CorrectionRecordRow(Base):
    """Structured half of the Override Store (architecture lines 138-146).

    The embedding of `input_context` lives in Weaviate (override collection).
    `vector_uuid` is the cross-store join key.
    """

    __tablename__ = "correction_records"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id: Mapped[str] = mapped_column(String(32), index=True)            # "agent_1" ... "agent_6"
    correction_type: Mapped[str] = mapped_column(String(64), index=True)
    input_context: Mapped[str] = mapped_column(Text)
    original_output: Mapped[dict[str, Any]] = mapped_column(JSONB)
    corrected_output: Mapped[dict[str, Any]] = mapped_column(JSONB)
    reviewer_id: Mapped[str] = mapped_column(String(128))
    vector_uuid: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)


class ReviewLogEntry(Base):
    """Append-only audit log of human review actions (architecture lines 116, 128, 192)."""

    __tablename__ = "review_log"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    trigger: Mapped[str] = mapped_column(String(64))                         # "low_confidence", "high_severity_conflict", "high_risk_gap"
    agent_id: Mapped[str] = mapped_column(String(32))
    subject_type: Mapped[str] = mapped_column(String(64))                    # "obligation", "conflict", "gap", "remediation"
    subject_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), index=True)
    reviewer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)  # "approved", "rejected", "modified"
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)

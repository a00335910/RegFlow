"""Shared domain types — pipeline contracts between layers.

These mirror the architecture PDF exactly:
- RegulatoryChangeEvent (Agent 1 output, lines 28-30)
- Obligation (Agent 2 output, lines 46-56)
- ConflictReport (Agent 3 output, lines 78-86)
- Gap / RiskScore (Agent 4 output, lines 89-98)
- RemediationPlan (Agent 5 output, lines 102-108)
- AuditEvidence (Agent 6 output, lines 111-116)
- CorrectionRecord (Override Store schema, lines 138-146)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    COSMETIC = "cosmetic"
    MINOR = "minor"
    SUBSTANTIVE = "substantive"
    MAJOR = "major"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CorrectionType(str, Enum):
    """Override Store schema, architecture line 143-144."""

    WRONG_EXTRACTION = "wrong_extraction"
    FALSE_POSITIVE_CONFLICT = "false_positive_conflict"
    RISK_OVERRIDE = "risk_override"
    OWNER_REASSIGNMENT = "owner_reassignment"


class RouteDecision(str, Enum):
    """Orchestrator routing, architecture line 40."""

    AUTO = "auto"
    NOTIFY = "notify"
    BLOCK = "block"


class BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    emitted_at: datetime = Field(default_factory=datetime.utcnow)


class RegulatoryChangeEvent(BaseEvent):
    """Emitted by Agent 1 (Regulatory Radar). Architecture lines 28-30."""

    article_id: str
    document_id: str
    severity: Severity
    jurisdiction: str            # e.g. "EU", "UK", "US-FED"
    regulator: str               # e.g. "ESMA", "FCA", "SEC"
    confidence: float = Field(ge=0.0, le=1.0)
    diff_summary: str
    prior_version_hash: str | None = None
    new_version_hash: str


class SourceCitation(BaseModel):
    document_id: str
    article_id: str
    clause_ref: str | None = None
    text_span: str
    char_start: int | None = None
    char_end: int | None = None


class Obligation(BaseModel):
    """Agent 2 output. Architecture lines 46-56."""

    obligation_id: UUID = Field(default_factory=uuid4)
    article_id: str
    document_id: str
    jurisdiction: str
    regulator: str

    obligation_text: str
    obligation_type: str         # e.g. "reporting", "retention", "disclosure"
    scope: str | None = None
    deadlines: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    exemptions: list[str] = Field(default_factory=list)

    citations: list[SourceCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class Conflict(BaseModel):
    """Agent 3 output. Architecture lines 78-86."""

    conflict_id: UUID = Field(default_factory=uuid4)
    conflict_type: Literal["contradiction", "overlap", "stricter_standard"]
    obligation_ids: list[UUID]
    description: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class Gap(BaseModel):
    """Agent 4 output. Architecture lines 89-98."""

    gap_id: UUID = Field(default_factory=uuid4)
    obligation_id: UUID
    missing_or_weak_controls: list[str]
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    enforcement_severity: float
    business_impact: float
    deadline_urgency: float
    related_audit_findings: list[str] = Field(default_factory=list)
    evidence_exists: bool = False
    confidence: float = Field(ge=0.0, le=1.0)


class RemediationAction(BaseModel):
    """Agent 5 output element. Architecture lines 102-108."""

    action_id: UUID = Field(default_factory=uuid4)
    gap_id: UUID
    description: str
    suggested_owner: str | None = None
    suggested_deadline: datetime | None = None
    proposed_control_updates: list[str] = Field(default_factory=list)
    dependencies: list[UUID] = Field(default_factory=list)
    priority: int = 0
    confidence: float = Field(ge=0.0, le=1.0)


class AuditEvidence(BaseModel):
    """Agent 6 output. Architecture lines 111-116."""

    evidence_id: UUID = Field(default_factory=uuid4)
    obligation_id: UUID
    clause_citations: list[SourceCitation]
    control_links: list[str] = Field(default_factory=list)
    justification: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CorrectionRecord(BaseModel):
    """Override Store schema. Architecture lines 138-146.

    Embedding of input_context is stored in Weaviate (override collection); this
    model represents the structured record (also persisted in Postgres for audit).
    """

    correction_id: UUID = Field(default_factory=uuid4)
    agent_id: Literal["agent_1", "agent_2", "agent_3", "agent_4", "agent_5", "agent_6"]
    correction_type: CorrectionType
    input_context: str
    original_output: dict[str, Any]
    corrected_output: dict[str, Any]
    reviewer_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

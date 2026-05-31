"""Agent 6 — Audit Evidence Generator (architecture lines 111-118).

Public contract:
    generate_evidence_for_obligation(obligation_id) -> AuditEvidence | None

Synthesizes an obligation's source citations, mapped controls, prior findings,
and review log into a structured audit-ready evidence pack. Most fields are
deterministically composed from existing agent output; the LLM's job is the
human-readable justification narrative and the proactive auditor questions.

Persistence: Postgres AuditEvidenceRow + Neo4j AuditEvidence node with
(Obligation)-[:HAS_EVIDENCE]->(AuditEvidence) edge.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from regflow.agents.audit_evidence.extractor import EvidenceSynthesis, synthesize
from regflow.agents.audit_evidence.queries import EvidenceContext, load_evidence_context
from regflow.agents.audit_evidence.validator import ValidationResult, validate_citations
from regflow.agents.audit_evidence.verifier import VerifierResult, verify
from regflow.common.llm import LLMError
from regflow.common.logging import get_logger
from regflow.common.types import AuditEvidence, CorrectionType, SourceCitation
from regflow.db.neo4j import upsert_audit_evidence_node
from regflow.db.postgres import AuditEvidenceRow, get_session
from regflow.rag.override_retriever import retrieve_corrections

log = get_logger(__name__)


def generate_evidence_for_obligation(obligation_id: UUID) -> AuditEvidence | None:
    ctx = load_evidence_context(obligation_id)
    if ctx is None:
        return None

    # ===== LAYER 3 — pull past reviewer corrections (anti-examples) =====
    corrections = retrieve_corrections(
        ctx.obligation.obligation_text,
        agent_id="agent_6",
        top_k=3,
        correction_type=CorrectionType.WRONG_EXTRACTION,
    )
    if corrections:
        log.info(
            "audit_evidence.corrections_retrieved",
            count=len(corrections),
            top_distance=corrections[0].distance,
        )

    # ===== Generate (with Layer 1 = [ref: X] requirement enforced in prompt) =====
    try:
        synth: EvidenceSynthesis = synthesize(ctx, corrections)
    except LLMError as exc:
        log.warning("audit_evidence.llm_failure", obligation_id=str(obligation_id), error=str(exc))
        return None

    # ===== LAYER 1 — validate citations against allow-lists =====
    allowed = _build_allowed_set(ctx)
    just_v = validate_citations(synth.justification, allowed)
    summary_validations: list[ValidationResult] = [
        validate_citations(b, allowed) for b in synth.evidence_summary
    ]
    synth.justification = just_v.clean_text
    synth.evidence_summary = [v.clean_text for v in summary_validations]
    all_invalid_refs = just_v.refs_invalid + [r for v in summary_validations for r in v.refs_invalid]

    if all_invalid_refs:
        log.info(
            "audit_evidence.citation_validation",
            invalid_refs=all_invalid_refs[:10],     # log first 10 to keep logs sane
            invalid_count=len(all_invalid_refs),
        )

    # ===== LAYER 2 — verifier second-pass over the cleaned justification =====
    verifier_summary = _summarize_input_for_verifier(ctx)
    verifier_result = verify(synth.justification, verifier_summary)
    unsupported_claims: list[str] = []
    if verifier_result is not None:
        unsupported_claims = [
            f"{j.claim}  [reason: {j.reason or 'not in input data'}]"
            for j in verifier_result.judgments
            if not j.supported
        ]
        log.info(
            "audit_evidence.verifier_pass",
            total_claims=len(verifier_result.judgments),
            unsupported=len(unsupported_claims),
        )

    evidence, row = _assemble(ctx, synth, all_invalid_refs, unsupported_claims)
    _persist(evidence, row)

    log.info(
        "audit_evidence.generated",
        obligation_id=str(obligation_id),
        control_links=len(row.control_links),
        review_log_refs=len(row.related_review_log_refs),
        invalid_refs=len(all_invalid_refs),
        unsupported_claims=len(unsupported_claims),
        confidence=synth.confidence,
    )
    return evidence


def _build_allowed_set(ctx: EvidenceContext) -> set[str]:
    """The universe of valid [ref: X] targets for this evidence pack."""
    allowed: set[str] = set()
    if ctx.latest_gap:
        allowed.update(ctx.latest_gap.matching_controls or [])
        allowed.update(ctx.latest_gap.related_audit_findings or [])
    return allowed


def _summarize_input_for_verifier(ctx: EvidenceContext) -> str:
    """Compact input summary the verifier consults to check claims against."""
    o = ctx.obligation
    a = ctx.article
    g = ctx.latest_gap
    parts = [
        f"OBLIGATION TEXT: {o.obligation_text}",
        f"JURISDICTION: {o.jurisdiction}",
        f"REGULATOR: {o.regulator}",
        f"ARTICLE: {a.article_ref if a else '(unknown)'}",
    ]
    if a and a.text:
        parts.append(f"ARTICLE TEXT EXCERPT: {a.text[:1200]}")
    if g:
        parts.append(f"MATCHING CONTROLS: {list(g.matching_controls or [])}")
        parts.append(f"MISSING CONTROLS: {list(g.missing_or_weak_controls or [])}")
        parts.append(f"RELATED FINDING REFS: {list(g.related_audit_findings or [])}")
    if ctx.review_log_entries:
        parts.append(
            "REVIEW LOG TRIGGERS: "
            + ", ".join({entry.trigger for entry in ctx.review_log_entries})
        )
    return "\n".join(parts)


def _assemble(
    ctx: EvidenceContext,
    synth: EvidenceSynthesis,
    invalid_refs: list[str],
    unsupported_claims: list[str],
) -> tuple[AuditEvidence, AuditEvidenceRow]:
    """Build both the domain object (AuditEvidence) and the Postgres row.

    Layer 1 + 2 outputs (invalid refs from citation validator + unsupported claims
    from verifier) are appended into open_questions with a [VERIFIER] prefix so the
    auditor sees the unverified material instead of it hiding in the body text.
    """
    o = ctx.obligation
    citations_payload = list(o.citations or [])
    control_links = list((ctx.latest_gap.matching_controls if ctx.latest_gap else []) or [])
    related_findings = list((ctx.latest_gap.related_audit_findings if ctx.latest_gap else []) or [])
    review_refs = [str(entry.id) for entry in ctx.review_log_entries]

    # Merge unverified material into open_questions with prefixes so the renderer
    # can group them. (Avoids a Postgres schema change for this iteration.)
    augmented_questions = list(synth.open_questions)
    if invalid_refs:
        augmented_questions.append(
            f"[VERIFIER] Citations not in supplied data: {', '.join(sorted(set(invalid_refs))[:8])}"
        )
    for claim in unsupported_claims:
        augmented_questions.append(f"[VERIFIER] Unsupported claim: {claim}")

    pydantic_citations = [
        SourceCitation(**c) if isinstance(c, dict) else SourceCitation.model_validate(c)
        for c in citations_payload
    ]
    evidence = AuditEvidence(
        obligation_id=o.id,
        clause_citations=pydantic_citations,
        control_links=control_links,
        justification=synth.justification,
    )

    row = AuditEvidenceRow(
        id=evidence.evidence_id,
        obligation_id=o.id,
        clause_citations=citations_payload,
        control_links=control_links,
        related_audit_findings=related_findings,
        related_review_log_refs=review_refs,
        open_questions=augmented_questions,
        justification=synth.justification,
        evidence_summary=synth.evidence_summary,
        confidence=synth.confidence,
    )
    return evidence, row


def _persist(evidence: AuditEvidence, row: AuditEvidenceRow) -> None:
    with get_session() as session:
        # One evidence pack per obligation: replace any prior one for this obligation.
        from sqlalchemy import delete

        session.execute(
            delete(AuditEvidenceRow).where(AuditEvidenceRow.obligation_id == row.obligation_id)
        )
        session.add(row)

    try:
        upsert_audit_evidence_node(
            evidence_id=str(row.id),
            obligation_id=str(row.obligation_id),
            justification=row.justification,
            confidence=row.confidence,
            control_links=row.control_links,
            generated_at=datetime.utcnow().isoformat(),
        )
    except Exception as exc:    # noqa: BLE001
        log.warning("audit_evidence.neo4j_edge_failed", obligation_id=str(row.obligation_id), error=str(exc))


# Used by the markdown export script to format a pack human-readably.
def render_markdown(row: AuditEvidenceRow, obligation_row, document_row, article_row) -> str:
    """Render one evidence pack as markdown — auditor-friendly artifact."""
    o = obligation_row
    d = document_row
    a = article_row

    citations = "\n".join(
        f"- **{c.get('clause_ref') or c.get('article_ref') or 'cite'}**: "
        f"\"{(c.get('text_span') or '')[:300]}...\""
        for c in (row.clause_citations or [])
    ) or "_(none)_"

    controls = "\n".join(f"- {_strip_leading_bullets(c)}" for c in row.control_links) or "_(no matched controls)_"
    findings = "\n".join(f"- {_strip_leading_bullets(f)}" for f in row.related_audit_findings) or "_(none)_"
    bullets = "\n".join(f"- {_strip_leading_bullets(b)}" for b in row.evidence_summary) or "_(none)_"

    # Split verifier output from real auditor questions so the renderer can show them separately.
    verifier_items: list[str] = []
    auditor_questions: list[str] = []
    for q in row.open_questions or []:
        cleaned = _strip_leading_bullets(q)
        if cleaned.startswith("[VERIFIER]"):
            verifier_items.append(cleaned.replace("[VERIFIER]", "").strip())
        else:
            auditor_questions.append(cleaned)
    questions = "\n".join(f"- {q}" for q in auditor_questions) or "_(none — pack believed complete)_"
    verifier_block = (
        "\n".join(f"- {v}" for v in verifier_items)
        or "_(none — all citations resolved and all claims supported)_"
    )

    return _MARKDOWN_TEMPLATE.format(
        obligation_text=o.obligation_text,
        jurisdiction=o.jurisdiction,
        regulator=o.regulator,
        document_title=d.title if d else "(unknown)",
        document_id=d.source_doc_id if d else "?",
        article_ref=a.article_ref if a else "(unknown)",
        obligation_type=o.obligation_type,
        deadlines=", ".join(o.deadlines) if o.deadlines else "_(none)_",
        penalties=", ".join(o.penalties) if o.penalties else "_(none)_",
        confidence=row.confidence,
        generated_at=row.generated_at.isoformat(),
        citations=citations,
        controls=controls,
        findings=findings,
        justification=row.justification,
        bullets=bullets,
        questions=questions,
        verifier_findings=verifier_block,
        pack_id=row.id,
    )


def _strip_leading_bullets(text: str) -> str:
    """LLMs occasionally emit list items that already start with '- ' or '* '. Strip
    those so the markdown renderer doesn't produce double bullets."""
    s = str(text).strip()
    while s.startswith(("- ", "* ", "• ")):
        s = s[2:].lstrip()
    return s


_MARKDOWN_TEMPLATE = """# Audit Evidence Pack

**Obligation:** {obligation_text}

| | |
|---|---|
| Jurisdiction | {jurisdiction} |
| Regulator | {regulator} |
| Source document | {document_title} (`{document_id}`) |
| Source article | {article_ref} |
| Obligation type | {obligation_type} |
| Deadlines | {deadlines} |
| Penalties | {penalties} |
| Confidence | {confidence:.2f} |
| Generated at | {generated_at} |

---

## Clause Citations

{citations}

## Matching Controls

{controls}

## Related Audit Findings & Enforcement Precedents

{findings}

---

## Compliance Justification

{justification}

---

## Evidence Summary

{bullets}

## Open Questions for Auditor

{questions}

## Verifier Findings (unsupported claims / unresolved citations)

> Items below are claims the Verifier could not ground in the supplied input data, or citations to entities not on the allow-list. They are NOT removed from the narrative — they remain visible so the reviewer can decide whether to accept, edit, or strip them.

{verifier_findings}

---

_Generated by RegFlow Agent 6 — Audit Evidence Generator._
_Defense layers active: citation-required generation (L1), verifier second pass (L2), Override Store anti-examples (L3)._
_Pack ID: {pack_id}_
"""

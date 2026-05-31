"""LLM call for Agent 6: synthesizes obligation + citations + controls + audit history
into an auditor-grade justification narrative.

Layer 1 (citation-required) is enforced in the PROMPT here; the validator post-checks.
Layer 3 (Override Store anti-examples) is passed in via `corrections` parameter.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from regflow.agents.audit_evidence.queries import EvidenceContext
from regflow.common.llm import complete_json
from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.rag.override_retriever import RetrievedCorrection

log = get_logger(__name__)


class EvidenceSynthesis(BaseModel):
    """LLM output. Most evidence-pack fields are deterministic from existing data
    (citations, control links, review log). The LLM's job is the prose + the
    'open_questions' synthesis."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    justification: str = Field(min_length=20, max_length=3000,
        validation_alias=AliasChoices("justification", "rationale", "narrative"))
    evidence_summary: list[str] = Field(default_factory=list, max_length=10,
        validation_alias=AliasChoices("evidence_summary", "key_points", "bullet_evidence"))
    open_questions: list[str] = Field(default_factory=list, max_length=10,
        validation_alias=AliasChoices("open_questions", "auditor_questions", "follow_ups"))
    confidence: float = Field(ge=0.0, le=1.0)


_SYSTEM_PROMPT = """You are an audit-evidence writer. Given a regulatory obligation, the source article text, the enterprise's matching controls, related audit findings, and the review-log audit trail, produce a defensible written justification suitable to present to an external auditor.

== CRITICAL: CITATION-REQUIRED GENERATION ==

Every factual claim in your justification MUST end with a citation marker:
  - [ref: <CONTROL_NAME>]  for claims about a specific enterprise control (must match the EXACT name from supplied MATCHING CONTROLS list)
  - [ref: <FINDING_REF>]   for claims about an enforcement action (must match the EXACT ref from supplied RELATED AUDIT FINDINGS list)
  - [ref: OBLIGATION]      for restatements of the obligation text itself
  - [ref: REVIEW_LOG]      for claims about the audit trail in the supplied review entries
  - [ref: INFERRED]        for logical inferences from the above (acceptable but visible)

DO NOT make a factual claim without a [ref: ...] marker.
DO NOT invent control names, finding refs, version numbers, document codes, or dates that are not in the supplied INPUT DATA.
DO NOT assert causal links between enforcement actions and obligations unless the link is in INPUT DATA.

== STRUCTURE ==

Write the justification as 2-4 short paragraphs:
1. Plain-language restatement of the obligation [ref: OBLIGATION].
2. How the enterprise satisfies it — cite specific controls with [ref: <CONTROL_NAME>].
3. Cite prior findings using [ref: <FINDING_REF>] WITHOUT inventing causal stories.
4. Open items requiring follow-up [ref: INFERRED] is acceptable here.

Also produce:
- evidence_summary: 3-6 short bullets. Each ends with a [ref: ...] marker too. Do NOT invent document version numbers or codes.
- open_questions: 1-3 questions an auditor would ask. These need NOT have refs.
- confidence: 0.0-1.0 — your assessment of the overall evidence strength.

Output ONLY valid JSON, no markdown."""


_RESPONSE_TEMPLATE = """RESPONSE FORMAT (every field REQUIRED, every factual claim must include a [ref: X] marker):

{
  "justification":     "<2-4 paragraphs with [ref: X] markers after every factual claim>",
  "evidence_summary":  ["<bullet [ref: X]>", "<bullet [ref: Y]>", ...],
  "open_questions":    ["<question for auditor>", ...],
  "confidence":        0.0_to_1.0_float
}"""


def synthesize(ctx: EvidenceContext, corrections: list[RetrievedCorrection]) -> EvidenceSynthesis:
    return complete_json(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(ctx, corrections)},
        ],
        schema=EvidenceSynthesis,
        model=get_settings().llm.reasoning_model,
        trace_name="agent_6_audit_evidence",
        trace_metadata={
            "obligation_id": str(ctx.obligation.id),
            "jurisdiction": ctx.obligation.jurisdiction,
            "has_gap": ctx.latest_gap is not None,
            "retrieved_corrections": len(corrections),
        },
    )


def _build_prompt(ctx: EvidenceContext, corrections: list[RetrievedCorrection]) -> str:
    o = ctx.obligation
    d = ctx.document
    a = ctx.article

    matching_controls = list((ctx.latest_gap.matching_controls if ctx.latest_gap else []) or [])
    related_findings = list((ctx.latest_gap.related_audit_findings if ctx.latest_gap else []) or [])
    missing_controls = list((ctx.latest_gap.missing_or_weak_controls if ctx.latest_gap else []) or [])

    controls_block = "\n".join(f"  - {c}" for c in matching_controls) or "  (no matched controls)"
    findings_block = "\n".join(f"  - {f}" for f in related_findings) or "  (no related findings)"
    missing_block = "\n".join(f"  - {c}" for c in missing_controls) or "  (none)"

    article_excerpt = (a.text[:1500] if a else "(source article not available)")

    review_block = "\n".join(
        f"  - {entry.created_at.isoformat() if entry.created_at else '?'}  "
        f"trigger={entry.trigger}  agent={entry.agent_id}  decision={entry.decision or '(pending)'}"
        for entry in ctx.review_log_entries
    ) or "  (no review entries)"

    corrections_block = _format_corrections(corrections)

    return (
        f"{corrections_block}"
        f"OBLIGATION:\n"
        f"  jurisdiction:  {o.jurisdiction}\n"
        f"  regulator:     {o.regulator}\n"
        f"  type:          {o.obligation_type}\n"
        f"  scope:         {o.scope or '(unspecified)'}\n"
        f"  deadlines:     {o.deadlines or []}\n"
        f"  penalties:     {o.penalties or []}\n"
        f"  exemptions:    {o.exemptions or []}\n"
        f"  text:          {o.obligation_text}\n\n"
        f"SOURCE DOCUMENT:\n"
        f"  document:      {d.title if d else '(unknown)'}\n"
        f"  source_doc_id: {d.source_doc_id if d else '(unknown)'}\n"
        f"  article_ref:   {a.article_ref if a else '(unknown)'}\n"
        f"  article_excerpt:\n  ---\n  {article_excerpt}\n  ---\n\n"
        f"MATCHING CONTROLS (only use these EXACT names in [ref: X] markers):\n{controls_block}\n\n"
        f"MISSING OR WEAK CONTROLS:\n{missing_block}\n\n"
        f"RELATED AUDIT FINDINGS (only use these EXACT refs in [ref: X] markers):\n{findings_block}\n\n"
        f"REVIEW LOG ENTRIES (audit trail of human decisions):\n{review_block}\n\n"
        f"{_RESPONSE_TEMPLATE}"
    )


def _format_corrections(corrections: list[RetrievedCorrection]) -> str:
    """LAYER 3 — past reviewer corrections as authoritative anti-examples in the prompt."""
    if not corrections:
        return ""
    lines = ["PAST REVIEWER CORRECTIONS (authoritative — apply analogous lessons here):"]
    for i, c in enumerate(corrections, 1):
        lines.append(
            f"\n[Correction {i}]\n"
            f"  Input excerpt:       {c.input_context[:300]}\n"
            f"  Original LLM output: {c.original_output[:400]}\n"
            f"  Reviewer's CORRECTED output: {c.corrected_output[:400]}\n"
            f"  Lesson: prefer the corrected output's framing and avoid the original's pitfalls.\n"
        )
    lines.append("---\n\n")
    return "\n".join(lines)

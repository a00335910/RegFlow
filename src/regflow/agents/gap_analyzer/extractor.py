"""LLM call for Agent 4: given an obligation + enterprise context, produce a Gap."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from regflow.agents.gap_analyzer.queries import ControlView, FindingView
from regflow.common.llm import complete_json
from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.common.types import Obligation
from regflow.rag import RetrievedCorrection, format_corrections_as_fewshot

log = get_logger(__name__)


class GapAnalysis(BaseModel):
    """LLM output schema. Aliases tolerate common renamings (same defensive posture
    used by Agent 3's DetectedConflict)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    matching_controls: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("matching_controls", "controls_matched", "covered_by"),
    )
    missing_or_weak_controls: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "missing_or_weak_controls", "missing_controls", "gaps", "weak_controls"
        ),
    )
    related_audit_findings: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("related_audit_findings", "audit_findings", "findings"),
    )
    evidence_exists: bool = Field(
        default=False,
        validation_alias=AliasChoices("evidence_exists", "evidence_available"),
    )
    enforcement_severity: float = Field(ge=0.0, le=1.0)
    business_impact: float = Field(ge=0.0, le=1.0)
    deadline_urgency: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(
        max_length=800,
        validation_alias=AliasChoices("rationale", "reasoning", "explanation", "summary"),
    )


_SYSTEM_PROMPT = """You are a compliance gap analyzer. Given (a) a regulatory obligation and (b) the enterprise's existing controls + prior audit findings, determine:

1. matching_controls  — control names that already address this obligation (use exact names from the list).
2. missing_or_weak_controls — short descriptions of controls that SHOULD exist but don't, OR existing controls that need strengthening to fully satisfy the obligation.
3. related_audit_findings — finding_refs (e.g. "2025-Q1-A02") of prior findings relevant to this obligation.
4. evidence_exists — true if at least one matching control has evidence on file, else false.
5. Score the gap on three factors (each 0.0-1.0):
   - enforcement_severity: how harshly does the regulator punish non-compliance? Look at the obligation's `penalties` field.
   - business_impact: how broadly does the obligation touch the enterprise's operations? (one team vs every business unit)
   - deadline_urgency: how time-pressured is implementation? Hard deadlines = high; ongoing principles = medium; future-effective = lower.
6. confidence: 0.0-1.0 in your overall analysis.
7. rationale: one or two sentences explaining the assessment (max 800 chars).

If the obligation is fully covered with strong evidence and no related findings, set the three risk factors LOW and explain why.
If there is NO matching control AND a related open audit finding, set deadline_urgency HIGH and explain.

Output ONLY valid JSON. Do not wrap in markdown."""


_RESPONSE_TEMPLATE = """RESPONSE FORMAT (use these EXACT field names, every field REQUIRED):

{
  "matching_controls":         ["<control name>", ...],
  "missing_or_weak_controls":  ["<short description of missing/weak control>", ...],
  "related_audit_findings":    ["<finding_ref>", ...],
  "evidence_exists":           true | false,
  "enforcement_severity":      0.0_to_1.0,
  "business_impact":           0.0_to_1.0,
  "deadline_urgency":          0.0_to_1.0,
  "confidence":                0.0_to_1.0,
  "rationale":                 "<one-to-two sentence summary, max 800 chars>"
}"""


def analyze_gap(
    obligation: Obligation,
    controls: list[ControlView],
    findings: list[FindingView],
    corrections: list[RetrievedCorrection] | None = None,
) -> GapAnalysis:
    user_prompt = _build_prompt(obligation, controls, findings, corrections or [])
    return complete_json(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        schema=GapAnalysis,
        model=get_settings().llm.reasoning_model,
        trace_name="agent_4_gap_analyzer",
        trace_metadata={
            "obligation_id": str(obligation.obligation_id),
            "jurisdiction": obligation.jurisdiction,
            "controls_in_scope": len(controls),
            "findings_in_scope": len(findings),
            "retrieved_corrections": len(corrections or []),
        },
    )


def _build_prompt(
    obligation: Obligation,
    controls: list[ControlView],
    findings: list[FindingView],
    corrections: list[RetrievedCorrection],
) -> str:
    controls_block = "\n".join(
        f"  - name: {c.name}\n    category: {c.category}\n    owner: {c.control_owner or '(unassigned)'}\n"
        f"    evidence_on_file: {c.evidence_exists}\n    description: {c.description}"
        for c in controls
    ) or "  (no controls loaded)"

    findings_block = "\n".join(
        f"  - {f.finding_ref} [{f.status}, {f.year}, {f.category}]: {f.finding_text}"
        for f in findings
    ) or "  (no relevant findings)"

    return (
        f"{format_corrections_as_fewshot(corrections)}"
        f"OBLIGATION:\n"
        f"  jurisdiction: {obligation.jurisdiction}\n"
        f"  regulator:    {obligation.regulator}\n"
        f"  type:         {obligation.obligation_type}\n"
        f"  scope:        {obligation.scope or '(unspecified)'}\n"
        f"  deadlines:    {obligation.deadlines or []}\n"
        f"  penalties:    {obligation.penalties or []}\n"
        f"  exemptions:   {obligation.exemptions or []}\n"
        f"  text:         {obligation.obligation_text}\n\n"
        f"ENTERPRISE CONTROLS:\n{controls_block}\n\n"
        f"PRIOR AUDIT FINDINGS (in related categories):\n{findings_block}\n\n"
        f"{_RESPONSE_TEMPLATE}"
    )

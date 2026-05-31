"""LLM call for Agent 5: turns a Gap into a list of concrete actions."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from regflow.agents.remediation_planner.queries import GapContext
from regflow.common.llm import complete_json
from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.rag import RetrievedCorrection, format_corrections_as_fewshot

log = get_logger(__name__)


class PlannedAction(BaseModel):
    """One action in a remediation plan. The LLM emits these; Python wraps each
    in a fresh action_id UUID before persistence."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    description: str = Field(min_length=1, max_length=600)
    suggested_owner: str | None = Field(
        default=None,
        validation_alias=AliasChoices("suggested_owner", "owner", "assignee"),
    )
    suggested_deadline: str | None = Field(
        default=None, max_length=64,
        validation_alias=AliasChoices("suggested_deadline", "deadline", "due"),
    )
    proposed_control_updates: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "proposed_control_updates", "control_updates", "new_or_updated_controls"
        ),
    )
    dependency_descriptions: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("dependency_descriptions", "dependencies", "depends_on"),
    )
    priority: int = Field(ge=1, le=5)        # 1=highest, 5=lowest
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = Field(default=None, max_length=400)


class RemediationPlan(BaseModel):
    actions: list[PlannedAction]


_SYSTEM_PROMPT = """You are a compliance remediation planner. Given a Gap (an obligation the enterprise does not yet fully satisfy) plus its matching controls, missing controls, and originating obligation, produce a remediation plan as an ordered list of concrete actions.

For each action:
- description: a single sentence describing what to do (max 600 chars).
- suggested_owner: pick from the SUPPLIED list of available owners. If none fit perfectly, pick the closest match. Do NOT invent new owners.
- suggested_deadline: a natural-language target (e.g., "Q3 2025", "within 60 days", "before 2025-09-30"). Tie this to the obligation's deadline if it has one.
- proposed_control_updates: short descriptions of new controls to add OR existing controls to amend. Reference existing control names where applicable.
- dependency_descriptions: list of dependencies in natural language (e.g., "after policy is approved by Legal"). Empty list if independent.
- priority: integer 1 (highest) to 5 (lowest).
- confidence: 0.0-1.0 — how sure you are this action is correct and feasible.
- rationale: one-sentence justification.

Aim for 2-5 actions per plan: enough to actually close the gap, not so many that the plan is unactionable. Be specific and operational — "Draft a new SOP for X" not "Improve our process."

Output ONLY valid JSON. No markdown."""


_RESPONSE_TEMPLATE = """RESPONSE FORMAT (every action field is REQUIRED unless explicitly nullable):

{
  "actions": [
    {
      "description":              "<one sentence, max 600 chars>",
      "suggested_owner":          "<MUST be one of the supplied available_owners, or null>",
      "suggested_deadline":       "<natural language e.g. 'Q3 2025' or null>",
      "proposed_control_updates": ["<short description>", ...],
      "dependency_descriptions":  ["<short description>", ...],
      "priority":                 1 | 2 | 3 | 4 | 5,
      "confidence":               0.0_to_1.0_float,
      "rationale":                "<short justification or null>"
    }
  ]
}"""


def plan(ctx: GapContext, corrections: list[RetrievedCorrection] | None = None) -> RemediationPlan:
    return complete_json(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(ctx, corrections or [])},
        ],
        schema=RemediationPlan,
        model=get_settings().llm.reasoning_model,
        trace_name="agent_5_remediation_planner",
        trace_metadata={
            "gap_id": str(ctx.gap.id),
            "obligation_id": str(ctx.obligation.id),
            "available_owners": len(ctx.available_owners),
            "retrieved_corrections": len(corrections or []),
        },
    )


def _build_prompt(ctx: GapContext, corrections: list[RetrievedCorrection]) -> str:
    g = ctx.gap
    o = ctx.obligation
    owners_block = "\n".join(f"  - {o}" for o in ctx.available_owners) or "  (no owners on file)"
    matching = "\n".join(f"  - {c}" for c in ctx.matching_controls) or "  (none — full gap)"
    missing = "\n".join(f"  - {c}" for c in ctx.missing_or_weak_controls) or "  (none)"
    return (
        f"{format_corrections_as_fewshot(corrections)}"
        f"OBLIGATION (what regulation requires):\n"
        f"  jurisdiction:  {o.jurisdiction}\n"
        f"  regulator:     {o.regulator}\n"
        f"  type:          {o.obligation_type}\n"
        f"  scope:         {o.scope or '(unspecified)'}\n"
        f"  deadlines:     {o.deadlines or []}\n"
        f"  penalties:     {o.penalties or []}\n"
        f"  text:          {o.obligation_text}\n\n"
        f"GAP (what's missing from the enterprise):\n"
        f"  risk_level:               {g.risk_level}\n"
        f"  risk_score:               {round(g.risk_score, 3)}\n"
        f"  enforcement_severity:     {round(g.enforcement_severity, 2)}\n"
        f"  business_impact:          {round(g.business_impact, 2)}\n"
        f"  deadline_urgency:         {round(g.deadline_urgency, 2)}\n"
        f"  matching_controls:\n{matching}\n"
        f"  missing_or_weak_controls:\n{missing}\n"
        f"  related_audit_findings:   {g.related_audit_findings or []}\n"
        f"  rationale:                {g.rationale or '(none)'}\n\n"
        f"AVAILABLE OWNERS (pick suggested_owner from this list):\n{owners_block}\n\n"
        f"{_RESPONSE_TEMPLATE}"
    )

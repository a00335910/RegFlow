"""LLM call for Agent 3. Given a set of source obligations + neighbor candidates,
ask the LLM to return all conflict pairs with type + severity + confidence.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from regflow.agents.conflict_detector.queries import CandidateObligation
from regflow.common.llm import complete_json
from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.rag import RetrievedCorrection, format_corrections_as_fewshot

log = get_logger(__name__)


class DetectedConflict(BaseModel):
    """LLM output schema for one conflict pair.

    Aliases accept the most common alternate names the model produces (we observed
    `source_id`/`neighbor_id`/`type` instead of the canonical names on gpt-oss:20b).
    populate_by_name=True means the canonical names also work.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    obligation_a_id: str = Field(
        validation_alias=AliasChoices(
            "obligation_a_id", "source_id", "source_obligation_id", "a_id"
        )
    )
    obligation_b_id: str = Field(
        validation_alias=AliasChoices(
            "obligation_b_id", "neighbor_id", "neighbor_obligation_id", "b_id"
        )
    )
    conflict_type: str = Field(
        pattern=r"^(contradiction|overlap|stricter_standard)$",
        validation_alias=AliasChoices("conflict_type", "type"),
    )
    severity: str = Field(pattern=r"^(minor|substantive|major)$")
    confidence: float = Field(
        ge=0.0, le=1.0,
        validation_alias=AliasChoices("confidence", "score"),
    )
    description: str = Field(
        max_length=600,
        validation_alias=AliasChoices("description", "reason", "explanation", "summary"),
    )


class ConflictResult(BaseModel):
    conflicts: list[DetectedConflict]


_SYSTEM_PROMPT = """You are a cross-jurisdiction regulatory conflict detector for a financial-compliance system.

Given (a) a set of SOURCE obligations from one article and (b) a set of NEIGHBOR obligations from other jurisdictions retrieved by semantic search, identify every meaningful conflict between any (source, neighbor) pair.

Three conflict types:
- "contradiction": The two obligations require incompatible actions (e.g., one says retain data for 5 years, the other says delete after 3 years).
- "overlap": The two obligations require essentially the same action under two regimes (duplicative compliance burden).
- "stricter_standard": One obligation is a stricter version of the other (e.g., 24-hour breach notification vs 72-hour); the stricter one supersedes in its jurisdiction.

Severity scale (the conflict's importance, not the underlying obligation's):
- "minor": affects narrow scope, easy to satisfy both
- "substantive": affects compliance design, requires reconciliation
- "major": legal-risk-grade, requires explicit policy decision

Be conservative. Return only conflicts with confidence >= 0.6. If two obligations are about different topics entirely, that is NOT a conflict — skip them.

Output ONLY valid JSON. Do not wrap in markdown. Use the EXACT field names shown in the response template — do NOT rename, abbreviate, or omit any field."""


_RESPONSE_TEMPLATE = """RESPONSE FORMAT (use these exact field names — every field is REQUIRED):

{
  "conflicts": [
    {
      "obligation_a_id": "<UUID string copied verbatim from one of the SOURCE obligations>",
      "obligation_b_id": "<UUID string copied verbatim from one of the NEIGHBOR obligations>",
      "conflict_type":   "contradiction" | "overlap" | "stricter_standard",
      "severity":        "minor" | "substantive" | "major",
      "confidence":      0.0_to_1.0_float,
      "description":     "<one sentence explaining the conflict, max 600 characters>"
    }
  ]
}

If no meaningful conflicts exist, return exactly: {"conflicts": []}"""


def detect_conflicts(
    sources: list[CandidateObligation],
    neighbors: list[CandidateObligation],
    corrections: list[RetrievedCorrection] | None = None,
) -> ConflictResult:
    """`corrections` are past reviewer corrections retrieved from the Override Store
    for agent_3 (typically `false_positive_conflict` type)."""
    if not sources or not neighbors:
        return ConflictResult(conflicts=[])

    user_prompt = _build_prompt(sources, neighbors, corrections or [])
    return complete_json(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        schema=ConflictResult,
        model=get_settings().llm.reasoning_model,
        trace_name="agent_3_conflict_detector",
        trace_metadata={
            "sources": len(sources),
            "neighbors": len(neighbors),
            "retrieved_corrections": len(corrections or []),
        },
    )


def _build_prompt(
    sources: list[CandidateObligation],
    neighbors: list[CandidateObligation],
    corrections: list[RetrievedCorrection],
) -> str:
    src_block = "\n".join(_fmt(c, label="SOURCE") for c in sources)
    nbr_block = "\n".join(_fmt(c, label="NEIGHBOR") for c in neighbors)
    return (
        f"{format_corrections_as_fewshot(corrections)}"
        f"SOURCE OBLIGATIONS (from one article):\n{src_block}\n\n"
        f"NEIGHBOR OBLIGATIONS (semantically similar from other jurisdictions):\n{nbr_block}\n\n"
        f"{_RESPONSE_TEMPLATE}"
    )


def _fmt(c: CandidateObligation, *, label: str) -> str:
    return (
        f"  [{label}] id={c.obligation_id} jurisdiction={c.jurisdiction} regulator={c.regulator} "
        f"type={c.obligation_type}\n"
        f"      text: {c.obligation_text[:400]}"
    )

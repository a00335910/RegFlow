"""LAYER 2 — Verifier second-pass model.

A separate LLM call grades each material claim in the generated justification
against a compact summary of the input data. Returns per-claim judgments:
  { claim: str, supported: bool, evidence: str | None }

The agent then surfaces unsupported claims to the reviewer rather than letting
them sit inside the prose untagged.

Design choice: we use the SAME model as the generator (gpt-oss:20b-cloud) for
v0.1. A production system would use a smaller, faster verifier specifically
fine-tuned for NLI/entailment — architecture line 188-189 anticipates this.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from regflow.common.llm import LLMError, complete_json
from regflow.common.logging import get_logger
from regflow.common.settings import get_settings

log = get_logger(__name__)


class ClaimJudgment(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    claim: str = Field(min_length=1, max_length=600)
    supported: bool
    evidence: str | None = Field(
        default=None,
        validation_alias=AliasChoices("evidence", "evidence_span", "quote", "source_quote"),
    )
    reason: str | None = Field(default=None, max_length=300)


class VerifierResult(BaseModel):
    judgments: list[ClaimJudgment]


_SYSTEM_PROMPT = """You are a verifier. Given INPUT DATA and a TEXT generated from it, decide which factual claims in the text are SUPPORTED by the input.

For each material claim in TEXT, output:
  - claim: the exact sentence (or sub-claim)
  - supported: true ONLY if the claim is directly stated in or trivially inferable from INPUT DATA
  - evidence: a short quote from INPUT DATA that supports it, or null
  - reason: when supported=false, one short sentence explaining why (max 300 chars)

A claim is UNSUPPORTED when it asserts any of:
  - specific document version numbers, document codes, dates, or identifiers not present verbatim in INPUT DATA
  - causal connections between enforcement cases and obligations that INPUT DATA does not explicitly state
  - internal company artifacts (training records, signed copies, test plans) not mentioned in INPUT DATA
  - any quantitative figure not present in INPUT DATA

Plain-language restatements of obligation text ARE supported (obligation text IS in input). Common-sense regulatory knowledge expressed without specific unverifiable details is acceptable.

Be strict. When in doubt, mark unsupported. Output ONLY valid JSON. No markdown."""


_RESPONSE_TEMPLATE = """RESPONSE FORMAT:

{
  "judgments": [
    {
      "claim":     "<exact sentence or sub-claim>",
      "supported": true | false,
      "evidence":  "<short quote from INPUT DATA, or null>",
      "reason":    "<one short sentence if supported=false, else null>"
    }
  ]
}"""


def verify(generated_text: str, input_summary: str) -> VerifierResult | None:
    """Run the verifier. Returns None on persistent LLM failure (caller treats as 'unable to verify')."""
    if not generated_text.strip():
        return VerifierResult(judgments=[])
    try:
        return complete_json(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"INPUT DATA:\n{input_summary}\n\n"
                    f"TEXT TO VERIFY:\n{generated_text}\n\n"
                    f"{_RESPONSE_TEMPLATE}"
                )},
            ],
            schema=VerifierResult,
            model=get_settings().llm.reasoning_model,
            trace_name="agent_6_verifier",
        )
    except LLMError as exc:
        log.warning("audit_evidence.verifier_failed", error=str(exc))
        return None

"""Phase 2 of Agent 1: LLM severity classification.

Each `ArticleDelta` is sent to the LLM with a structured-output prompt. The model
returns a `SeverityClassification` (validated by Pydantic). Cosmetic results are
filtered out by the caller (agent.py), not here — this module's only job is to
produce a clean classification given a delta.

If the LLM call fails after retries, the caller decides what to do (currently:
emit as MINOR with confidence=0.0, per the design preface).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from regflow.common.llm import complete_json
from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.common.types import Severity
from regflow.agents.regulatory_radar.delta_detector import ArticleDelta, DeltaType
from regflow.rag import RetrievedCorrection, format_corrections_as_fewshot

log = get_logger(__name__)


class SeverityClassification(BaseModel):
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    topic: str = Field(max_length=120)
    diff_summary: str = Field(max_length=400)


_SYSTEM_PROMPT = """You are a regulatory change classifier for a financial-compliance system.

Given a diff between two versions of a regulation article (or a newly added / removed article), classify the change.

Severity scale:
- "cosmetic":   formatting, punctuation, numbering, citation updates, no semantic change
- "minor":      clarifying language, examples added, non-binding guidance changes
- "substantive": changed deadline, scope, threshold, exemption; new or removed obligation
- "major":      sanctions added, scope expansion across products/entities, fundamental restructure

Be conservative. Only call something "major" if it materially expands what regulated entities must do.
Output ONLY valid JSON matching the schema. Do not include any prose outside the JSON object."""


_RESPONSE_SCHEMA_HINT = """Respond with JSON of this exact shape:
{
  "severity": "cosmetic" | "minor" | "substantive" | "major",
  "confidence": <float between 0.0 and 1.0>,
  "topic": "<short label, max 120 chars>",
  "diff_summary": "<one sentence, max 400 chars>"
}"""


def classify_delta(
    delta: ArticleDelta,
    corrections: list[RetrievedCorrection] | None = None,
) -> SeverityClassification:
    """Returns a classification. May raise LLMError / LLMSchemaError on persistent failure.

    `corrections` are past reviewer corrections retrieved from the Override Store; they
    are injected into the prompt as authoritative few-shot anti-examples.
    """
    user_prompt = _build_user_prompt(delta, corrections or [])
    settings = get_settings().llm

    classification = complete_json(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        schema=SeverityClassification,
        model=settings.classifier_model,
        trace_name="agent_1_severity_classifier",
        trace_metadata={
            "article_ref": delta.article_ref,
            "change_type": delta.change_type.value,
            "retrieved_corrections": len(corrections or []),
        },
    )
    log.debug(
        "radar.classified",
        article_ref=delta.article_ref,
        change_type=delta.change_type.value,
        severity=classification.severity.value,
        confidence=classification.confidence,
    )
    return classification


def fallback_classification(delta: ArticleDelta) -> SeverityClassification:
    """Used when the LLM call fails after all retries (design choice: MINOR + log)."""
    return SeverityClassification(
        severity=Severity.MINOR,
        confidence=0.0,
        topic="llm_failure_fallback",
        diff_summary=(
            f"[automatic fallback] LLM classification failed for {delta.change_type.value} "
            f"article {delta.article_ref}; flagged for human review."
        ),
    )


def _build_user_prompt(delta: ArticleDelta, corrections: list[RetrievedCorrection]) -> str:
    header = (
        f"Article: {delta.article_ref}\n"
        f"Change type: {delta.change_type.value}\n"
    )
    if delta.change_type is DeltaType.MODIFIED:
        body = f"Diff (unified, 2-line context):\n```diff\n{delta.diff_text}\n```"
    elif delta.change_type is DeltaType.ADDED:
        body = f"New article body:\n```\n{delta.new_text}\n```"
    else:  # REMOVED
        body = f"Removed article body:\n```\n{delta.old_text}\n```"

    corrections_block = format_corrections_as_fewshot(corrections)
    return f"{corrections_block}{header}\n{body}\n\n{_RESPONSE_SCHEMA_HINT}"

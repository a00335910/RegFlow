"""Agent 2's LLM call — prompt building + structured-output extraction.

Lives in its own module so the orchestration in agent.py stays readable.
Override Store corrections are passed in as a list of `RetrievedCorrection` and
formatted into the prompt as few-shot examples.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from regflow.common.llm import complete_json
from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.rag.override_retriever import RetrievedCorrection

log = get_logger(__name__)


class ExtractedObligation(BaseModel):
    """Schema the LLM must produce for each obligation. Pydantic validates on parse."""

    obligation_text: str = Field(min_length=1, max_length=2000)
    obligation_type: str = Field(max_length=64)            # reporting / retention / disclosure / consent / security / governance / other
    scope: str | None = Field(default=None, max_length=500)
    deadlines: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    exemptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    obligations: list[ExtractedObligation]


_SYSTEM_PROMPT = """You are a regulatory obligation extractor for a financial-compliance system.

Given one article from a regulation, extract structured OBLIGATIONS. An obligation is anything that creates a duty, deadline, prohibition, or required action for a regulated entity (controller, processor, supervisory authority, etc.).

For each obligation, output:
- obligation_text: a clear action-focused sentence describing what must be done
- obligation_type: one of "reporting", "retention", "disclosure", "consent", "security", "governance", "other"
- scope: who must comply (e.g., "controllers established in the Union"), or null if all-encompassing
- deadlines: list of time-bound requirements (e.g., "within 72 hours of becoming aware", "by 25 May 2018")
- penalties: list of consequences for non-compliance (cross-references to fine provisions are fine)
- exemptions: list of circumstances where the obligation does NOT apply
- confidence: 0.0–1.0 (how sure you are this is a real obligation vs informational text)

If the article is purely informational, definitional, or procedural (e.g. "this Regulation enters into force on X", "the following definitions apply"), return an empty obligations array. Not every article contains an obligation; that is normal and expected.

Output ONLY a single JSON object of the form: {"obligations": [...]}. Do not wrap the JSON in markdown."""


def extract_from_article(
    article_text: str,
    article_ref: str,
    retrieved_corrections: list[RetrievedCorrection],
) -> ExtractionResult:
    user_prompt = _build_user_prompt(article_text, article_ref, retrieved_corrections)
    return complete_json(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        schema=ExtractionResult,
        model=get_settings().llm.extraction_model,
        trace_name="agent_2_obligation_extractor",
        trace_metadata={
            "article_ref": article_ref,
            "retrieved_corrections": len(retrieved_corrections),
        },
    )


def _build_user_prompt(
    article_text: str,
    article_ref: str,
    corrections: list[RetrievedCorrection],
) -> str:
    parts: list[str] = []

    if corrections:
        parts.append("PAST REVIEWER CORRECTIONS for similar inputs (treat as authoritative guidance):")
        for i, c in enumerate(corrections, 1):
            parts.append(
                f"\n[Example {i}]\n"
                f"Input excerpt: {c.input_context[:400]}\n"
                f"Original LLM output: {c.original_output[:600]}\n"
                f"Reviewer's CORRECTED output: {c.corrected_output[:600]}\n"
                f"Lesson: prefer the corrected output for analogous inputs.\n"
            )
        parts.append("---\n")

    parts.append(
        f"Now extract obligations from this article.\n\n"
        f"Article: {article_ref}\n"
        f"Text:\n{article_text}\n\n"
        f'Respond with: {{"obligations": [ ... ]}}'
    )
    return "".join(parts)

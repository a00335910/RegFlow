"""Shared formatter that turns retrieved Override Store records into a few-shot
prompt block. Used by every agent's extractor — keeps the framing consistent
across the system so all retrieved corrections look the same to the LLM.

Architecture line 153: 'Inject retrieved corrections into prompt as few-shot examples.'
"""

from __future__ import annotations

from regflow.rag.override_retriever import RetrievedCorrection


def format_corrections_as_fewshot(
    corrections: list[RetrievedCorrection],
    *,
    input_excerpt_chars: int = 300,
    output_excerpt_chars: int = 400,
) -> str:
    """Renders a uniform '[Correction N]' block. Returns '' when no corrections —
    callers concat directly without an extra branch.

    The framing matters: we tell the LLM these are AUTHORITATIVE reviewer guidance,
    not just curious examples. Models treat unframed examples as 'interesting';
    framed examples as 'definitive.'
    """
    if not corrections:
        return ""

    lines: list[str] = [
        "PAST REVIEWER CORRECTIONS (authoritative — apply analogous lessons here):"
    ]
    for i, c in enumerate(corrections, 1):
        lines.append(
            f"\n[Correction {i}]\n"
            f"  Input excerpt:              {c.input_context[:input_excerpt_chars]}\n"
            f"  Original LLM output:        {c.original_output[:output_excerpt_chars]}\n"
            f"  Reviewer's CORRECTED output:{c.corrected_output[:output_excerpt_chars]}\n"
            f"  Lesson: prefer the corrected output's framing and avoid the original's pitfalls.\n"
        )
    lines.append("---\n")
    return "\n".join(lines)

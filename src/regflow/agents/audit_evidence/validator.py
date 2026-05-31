"""LAYER 1 — Citation-required generation validator.

Every factual claim in the LLM's justification must end with a `[ref: X]` marker.
This module parses those markers and validates each against the allow-lists:
  - Control names from the matched-controls list
  - Finding refs from the related-findings list
  - "OBLIGATION" (the regulation text itself, always allowed)
  - "INFERRED" (logical inference from supplied facts, allowed but visible)

Unsupported refs are NOT silently dropped — they're rewritten to `[⚠ unverified ref: X]`
so the auditor sees what was challenged. Transparent rather than naive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_REF_PATTERN = re.compile(r"\[ref:\s*([^\]]+)\]", re.IGNORECASE)

# Always-allowed pseudo-refs the model can use for material from supplied data.
_PSEUDO_REFS = {"OBLIGATION", "INFERRED", "REVIEW_LOG"}


@dataclass(frozen=True)
class ValidationResult:
    clean_text: str            # original with invalid refs rewritten as [⚠ unverified ref: X]
    refs_found: list[str]      # every ref the LLM emitted
    refs_invalid: list[str]    # subset of refs_found that didn't resolve
    refs_valid: list[str]      # subset that resolved


def validate_citations(text: str, allowed: set[str]) -> ValidationResult:
    """Find all `[ref: X]` markers in `text`. Allowed refs are passed through;
    unsupported refs are rewritten to `[⚠ unverified ref: X]`."""

    refs_found: list[str] = []
    refs_invalid: list[str] = []
    refs_valid: list[str] = []

    allowed_set = {*allowed, *_PSEUDO_REFS}

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        refs_found.append(ref)
        # Case-insensitive allow-list check + pseudo-ref fast path.
        if ref in allowed_set or ref.upper() in _PSEUDO_REFS:
            refs_valid.append(ref)
            return f"[ref: {ref}]"
        # Try a case-insensitive match against allowed names.
        for a in allowed:
            if a.lower() == ref.lower():
                refs_valid.append(a)
                return f"[ref: {a}]"
        refs_invalid.append(ref)
        return f"[⚠ unverified ref: {ref}]"

    cleaned = _REF_PATTERN.sub(replace, text)

    return ValidationResult(
        clean_text=cleaned,
        refs_found=refs_found,
        refs_invalid=refs_invalid,
        refs_valid=refs_valid,
    )

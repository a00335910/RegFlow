"""End-to-end demonstration of the correction-retrieval loop.

What it does:
    1. Runs Agent 2 on a given article (defaults to GDPR Article 5 — the
       'principles for processing' article — which the model under-extracts
       on day 1).
    2. Submits a reviewer correction via the /reviews/corrections API.
    3. Re-runs Agent 2 on the SAME article.
    4. Prints a side-by-side diff so the influence of the correction is visible.

Usage:
    python scripts/run_api.py                      # in one terminal
    python scripts/demo_correction_loop.py         # in another terminal

    # Defaults to article_ref="art_5" of CELEX 32016R0679. Override with --article-ref / --celex.
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx
from sqlalchemy import select

from regflow.agents.obligation_extractor import extract_obligations
from regflow.common.logging import configure_logging, get_logger
from regflow.common.types import CorrectionType, RegulatoryChangeEvent, Severity
from regflow.db.postgres import Article, Document, get_session


# A defensible reviewer-supplied correction for GDPR Article 5 — the seven principles
# every controller must apply. This is what a real compliance reviewer would say
# Agent 2 should have extracted.
_ARTICLE_5_CORRECTED_OUTPUT = {
    "obligations": [
        {
            "obligation_text": "Controllers must process personal data lawfully, fairly, and in a transparent manner.",
            "obligation_type": "governance",
            "scope": "controllers",
            "deadlines": [],
            "penalties": ["administrative fines under Article 83"],
            "exemptions": [],
            "confidence": 0.95,
        },
        {
            "obligation_text": "Controllers must collect personal data only for specified, explicit and legitimate purposes (purpose limitation).",
            "obligation_type": "governance",
            "scope": "controllers",
            "deadlines": [],
            "penalties": ["administrative fines under Article 83"],
            "exemptions": [],
            "confidence": 0.95,
        },
        {
            "obligation_text": "Controllers must ensure personal data are adequate, relevant and limited to what is necessary (data minimisation).",
            "obligation_type": "governance",
            "scope": "controllers",
            "deadlines": [],
            "penalties": ["administrative fines under Article 83"],
            "exemptions": [],
            "confidence": 0.95,
        },
        {
            "obligation_text": "Controllers must keep personal data accurate and up to date; inaccurate data must be erased or rectified without delay.",
            "obligation_type": "governance",
            "scope": "controllers",
            "deadlines": ["without delay"],
            "penalties": ["administrative fines under Article 83"],
            "exemptions": [],
            "confidence": 0.95,
        },
        {
            "obligation_text": "Controllers must retain personal data only for as long as necessary for the purpose for which it was collected (storage limitation).",
            "obligation_type": "retention",
            "scope": "controllers",
            "deadlines": [],
            "penalties": ["administrative fines under Article 83"],
            "exemptions": ["archiving in public interest, scientific or historical research"],
            "confidence": 0.95,
        },
        {
            "obligation_text": "Controllers must ensure appropriate security of personal data, including protection against unauthorised or unlawful processing (integrity and confidentiality).",
            "obligation_type": "security",
            "scope": "controllers",
            "deadlines": [],
            "penalties": ["administrative fines under Article 83"],
            "exemptions": [],
            "confidence": 0.95,
        },
        {
            "obligation_text": "Controllers must be able to demonstrate compliance with all the above principles (accountability).",
            "obligation_type": "governance",
            "scope": "controllers",
            "deadlines": [],
            "penalties": ["administrative fines under Article 83"],
            "exemptions": [],
            "confidence": 0.95,
        },
    ]
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--celex", default="32016R0679")
    p.add_argument("--article-ref", default="art_5")
    p.add_argument("--api-url", default="http://127.0.0.1:8000")
    return p.parse_args()


def _load_article(celex: str, article_ref: str):
    with get_session() as session:
        doc = session.execute(
            select(Document).where(Document.source_doc_id == celex).order_by(Document.fetched_at.desc()).limit(1)
        ).scalar_one_or_none()
        if doc is None:
            print(f"No ingested document found for CELEX {celex}.")
            sys.exit(1)
        art = session.execute(
            select(Article).where(Article.document_id == doc.id, Article.article_ref == article_ref).limit(1)
        ).scalar_one_or_none()
        if art is None:
            print(f"No article '{article_ref}' in document {celex}.")
            sys.exit(1)
        session.expunge_all()
        return doc, art


def _event_for(art, doc) -> RegulatoryChangeEvent:
    """Construct a RegulatoryChangeEvent identical to what Agent 1 would have emitted."""
    return RegulatoryChangeEvent(
        article_id=str(art.id),
        document_id=str(doc.id),
        severity=Severity.SUBSTANTIVE,
        jurisdiction=doc.jurisdiction,
        regulator=doc.regulator,
        confidence=0.90,
        diff_summary=f"Demo invocation for article {art.article_ref}",
        prior_version_hash=None,
        new_version_hash=doc.content_hash,
    )


def _print_obligations(label: str, obligations: list) -> None:
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"  Count: {len(obligations)}")
    for i, o in enumerate(obligations, 1):
        print(f"\n  [{i}] type={o.obligation_type:<11} confidence={o.confidence:.2f}")
        print(f"      {o.obligation_text[:140]}")


def main() -> int:
    configure_logging()
    log = get_logger("demo_correction_loop")
    args = _parse_args()

    doc, art = _load_article(args.celex, args.article_ref)
    print(f"\nTarget: {doc.source_doc_id} / {art.article_ref}")
    print(f"Text excerpt: {art.text[:200]}...\n")

    event = _event_for(art, doc)

    # ===== Day 1 =====
    print("\n[Step 1/4] Day-1 run — Override Store is empty for this input.")
    day1_obligations = extract_obligations(event)
    _print_obligations("Day-1 Agent 2 output (no corrections influencing)", day1_obligations)

    # ===== Submit correction =====
    print("\n[Step 2/4] Submitting reviewer correction to /reviews/corrections ...")
    payload = {
        "agent_id": "agent_2",
        "correction_type": CorrectionType.WRONG_EXTRACTION.value,
        "input_context": art.text,
        "original_output": {"obligations": [_to_dict(o) for o in day1_obligations]},
        "corrected_output": _ARTICLE_5_CORRECTED_OUTPUT,
        "reviewer_id": "demo_reviewer",
        "note": "Article 5 establishes the 7 GDPR principles; each is an obligation.",
    }
    resp = httpx.post(f"{args.api_url}/reviews/corrections", json=payload, timeout=30)
    if resp.status_code != 201:
        print(f"\n  ERROR: API returned {resp.status_code}: {resp.text}")
        return 1
    print(f"  -> correction_id: {resp.json()['correction_id']}")

    # ===== Day 2 =====
    print("\n[Step 3/4] Day-2 run — Override Store now contains the reviewer correction.")
    day2_obligations = extract_obligations(event)
    _print_obligations("Day-2 Agent 2 output (correction retrieved as few-shot)", day2_obligations)

    # ===== Diff =====
    print("\n[Step 4/4] Diff summary:")
    print(f"  Day-1 count: {len(day1_obligations)}")
    print(f"  Day-2 count: {len(day2_obligations)}")
    delta = len(day2_obligations) - len(day1_obligations)
    direction = "more" if delta > 0 else "fewer" if delta < 0 else "same"
    print(f"  Delta:       {direction} ({delta:+d}) obligations after correction.")
    if delta > 0:
        print("\n  The correction influenced the LLM. Override Store loop is closed.")
    elif delta == 0 and day1_obligations == day2_obligations:
        print("\n  No change — the LLM may have ignored the few-shot. Investigate prompt formatting.")
    else:
        print("\n  Output changed; review side-by-side to assess whether the correction helped.")

    return 0


def _to_dict(obligation) -> dict:
    return {
        "obligation_text": obligation.obligation_text,
        "obligation_type": obligation.obligation_type,
        "scope": obligation.scope,
        "deadlines": obligation.deadlines,
        "penalties": obligation.penalties,
        "exemptions": obligation.exemptions,
        "confidence": obligation.confidence,
    }


if __name__ == "__main__":
    sys.exit(main())

"""Drive the LangGraph orchestrator end-to-end on the latest ingested version of a document.

Usage:
    # Full document
    python scripts/run_workflow.py 32016R0679

    # Smoke test on first 5 article-blocks
    python scripts/run_workflow.py 32016R0679 --only-articles --limit 5

What it does:
    1. Loads the latest ingested Document for `source_doc_id`.
    2. Invokes the LangGraph workflow:
         radar -> route -> dispatch -> finalize
    3. Prints a routing summary and the BLOCK list (the events humans must review).
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import select

from regflow.common.logging import configure_logging, get_logger
from regflow.db.postgres import Document, get_session
from regflow.orchestrator import run_workflow


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_doc_id")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only-articles", action="store_true")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("run_workflow")
    args = _parse_args(argv)

    with get_session() as session:
        latest = session.execute(
            select(Document)
            .where(Document.source_doc_id == args.source_doc_id)
            .order_by(Document.fetched_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest is None:
            log.error("document_not_found", source_doc_id=args.source_doc_id)
            print(f"\nNo ingested document for {args.source_doc_id!r}.")
            return 1

        document_id = latest.id
        print(
            f"\nRunning Orchestrator on:"
            f"\n  source_doc_id:    {latest.source_doc_id}"
            f"\n  jurisdiction:     {latest.jurisdiction}"
            f"\n  --limit:          {args.limit}"
            f"\n  --only-articles:  {args.only_articles}"
            f"\n"
        )

    result = run_workflow(
        document_id=document_id,
        radar_limit=args.limit,
        radar_include_prefixes=("art_",) if args.only_articles else None,
    )

    print("\n=== Routing summary ===")
    print(f"  Total events     : {len(result.events)}")
    print(f"  AUTO             : {len(result.auto)}")
    print(f"  NOTIFY           : {len(result.notify)}")
    print(f"  BLOCK            : {len(result.block)}")
    print(f"  Review log writes: {result.review_log_entries_written}")
    print(f"  Obligations (stub): {len(result.obligations)}")

    if result.block:
        print("\n=== BLOCKED events (mandatory human review) ===")
        for i, evt in enumerate(result.block, 1):
            print(
                f"  [{i}] severity={evt.severity.value:<11} confidence={evt.confidence:.2f}  "
                f"article={evt.article_id[:12]}"
            )
            print(f"      {evt.diff_summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

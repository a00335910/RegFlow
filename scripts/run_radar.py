"""Run Agent 1 (Regulatory Radar) against the latest ingested version of a CELEX/source doc id.

Usage:
    # Process the full document (will hit Ollama ~N times)
    python scripts/run_radar.py 32016R0679

    # Smoke test: classify only 5 deltas (good for first LLM run)
    python scripts/run_radar.py 32016R0679 --limit 5

    # Restrict to real articles (skip recitals, citations, preamble umbrella)
    python scripts/run_radar.py 32016R0679 --only-articles --limit 5

Prereqs:
    - docker compose up -d (Postgres + Weaviate + MinIO + Ollama running)
    - python scripts/init_infra.py
    - The document already ingested via scripts/ingest_eurlex.py
    - The Ollama model pulled: docker exec -it regflow-ollama ollama pull llama3.1:8b
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import select

from regflow.agents.regulatory_radar import run_radar
from regflow.common.logging import configure_logging, get_logger
from regflow.db.postgres import Document, get_session


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_doc_id", help="CELEX number or other source_doc_id")
    p.add_argument("--limit", type=int, default=None,
                   help="Classify at most N deltas (good for smoke-testing).")
    p.add_argument("--only-articles", action="store_true",
                   help="Restrict to article blocks (article_ref starting with 'art_').")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("run_radar")
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
            print(f"Run `python scripts/ingest_eurlex.py {args.source_doc_id}` first.")
            return 1

        document_id = latest.id
        print(
            f"\nRunning Regulatory Radar on:"
            f"\n  source_doc_id: {latest.source_doc_id}"
            f"\n  jurisdiction:  {latest.jurisdiction}"
            f"\n  total articles in store: {len(latest.articles)}"
            f"\n  --limit:          {args.limit}"
            f"\n  --only-articles:  {args.only_articles}"
            f"\n"
        )

    include_prefixes = ("art_",) if args.only_articles else None

    events = run_radar(
        document_id,
        limit=args.limit,
        include_prefixes=include_prefixes,
    )

    print(f"\n--- {len(events)} RegulatoryChangeEvent(s) emitted ---\n")
    for i, evt in enumerate(events, 1):
        print(
            f"[{i}] severity={evt.severity.value:<11} confidence={evt.confidence:.2f}  "
            f"article={evt.article_id[:12]}\n"
            f"    {evt.diff_summary}\n"
        )

    if not events:
        print("(no substantive changes emitted — all filtered as cosmetic or matched no articles)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

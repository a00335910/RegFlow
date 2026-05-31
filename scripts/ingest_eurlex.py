"""Drive EUR-Lex ingestion.

Three usage patterns:

    # 1. Ingest everything listed in config/feeds.yaml (HTTP fetch)
    python scripts/ingest_eurlex.py

    # 2. Ingest one specific CELEX over HTTP
    python scripts/ingest_eurlex.py 32016R0679

    # 3. Ingest from a locally-saved HTML file (use when EUR-Lex's Cloudflare blocks us)
    python scripts/ingest_eurlex.py 32016R0679 --from-file data/samples/gdpr.html
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from regflow.common.logging import configure_logging, get_logger
from regflow.feeds.connectors.eur_lex import EurLexConnector
from regflow.feeds.pipeline import ingest_document_from_parsed


@dataclass
class Args:
    celex_ids: list[str]
    from_file: Path | None


def _parse_args(argv: list[str]) -> Args:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("celex_ids", nargs="*", help="CELEX number(s) to ingest. Omit to use feeds.yaml.")
    p.add_argument("--from-file", type=Path, default=None,
                   help="Path to a locally-saved EUR-Lex HTML file (requires exactly one CELEX arg).")
    ns = p.parse_args(argv[1:])

    if ns.from_file is not None and len(ns.celex_ids) != 1:
        p.error("--from-file requires exactly one CELEX argument")
    if ns.from_file is not None and not ns.from_file.exists():
        p.error(f"file not found: {ns.from_file}")

    return Args(celex_ids=ns.celex_ids, from_file=ns.from_file)


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("ingest_eurlex")
    args = _parse_args(argv)

    connector = EurLexConnector()
    try:
        # Local-file path (single CELEX, skip HTTP entirely)
        if args.from_file is not None:
            celex = args.celex_ids[0]
            parsed = connector.fetch_from_local_file(celex, args.from_file)
            result = ingest_document_from_parsed(parsed)
            log.info(
                "done",
                celex=celex,
                document_id=str(result.document_id),
                new_version=result.is_new_version,
                articles=result.article_count,
            )
            return 0

        # HTTP path: one or more CELEX, or feeds.yaml
        targets = args.celex_ids or connector.discover()
        if not targets:
            log.error("no CELEX ids on the command line and none configured in config/feeds.yaml")
            return 1

        for celex in targets:
            try:
                parsed = connector.fetch(celex)
                result = ingest_document_from_parsed(parsed)
                log.info(
                    "done",
                    celex=celex,
                    document_id=str(result.document_id),
                    new_version=result.is_new_version,
                    articles=result.article_count,
                )
            except Exception as exc:
                log.error("ingest.failed", celex=celex, error=str(exc))
                raise
        return 0
    finally:
        connector.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Drive Federal Register ingestion.

Three usage patterns (mirrors scripts/ingest_eurlex.py):

    # 1. Use config/feeds.yaml (discover via search OR explicit document_numbers)
    python scripts/ingest_federal_register.py

    # 2. One or more specific document numbers
    python scripts/ingest_federal_register.py 2024-12345 2024-67890

Find document numbers at https://www.federalregister.gov/ — every rule has a
document number in its URL (e.g. .../documents/2024/05/15/2024-10739/...).
"""

from __future__ import annotations

import argparse
import sys

from regflow.common.logging import configure_logging, get_logger
from regflow.feeds.connectors.federal_register import FederalRegisterConnector
from regflow.feeds.pipeline import ingest_document


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("document_numbers", nargs="*",
                   help="Document numbers to ingest. Omit to use config/feeds.yaml.")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("ingest_federal_register")
    args = _parse_args(argv)

    connector = FederalRegisterConnector()
    try:
        targets = args.document_numbers or connector.discover()
        if not targets:
            log.error("no document_numbers on command line and search returned 0 hits")
            return 1

        log.info("ingest.starting", count=len(targets), document_numbers=targets)
        for doc_num in targets:
            try:
                result = ingest_document(connector, doc_num)
                log.info(
                    "done",
                    document_number=doc_num,
                    document_id=str(result.document_id),
                    new_version=result.is_new_version,
                    articles=result.article_count,
                )
            except Exception as exc:
                log.error("ingest.failed", document_number=doc_num, error=str(exc))
                # Continue with the next doc rather than aborting the batch
                continue
        return 0
    finally:
        connector.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv))

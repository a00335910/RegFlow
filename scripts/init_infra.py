"""Initialize all persistent stores (idempotent). Run once after `docker compose up`.

Creates:
- Postgres tables (documents, articles, correction_records, review_log)
- Weaviate collections (RegulatoryCorpus, OverrideStore)
- MinIO bucket (regflow-raw-docs)
- Neo4j constraints (uniqueness on Obligation.id, Document.id, etc.)
"""

from __future__ import annotations

import sys

from regflow.common.logging import configure_logging, get_logger
from regflow.db import minio_client
from regflow.db.neo4j import init_constraints as init_neo4j_constraints
from regflow.db.postgres import Base, get_engine
from regflow.db.vector import get_vector_store


def main() -> int:
    configure_logging()
    log = get_logger("init_infra")

    log.info("postgres.create_all")
    Base.metadata.create_all(get_engine())

    log.info("minio.ensure_bucket")
    minio_client.ensure_bucket()

    log.info("weaviate.ensure_schema")
    with get_vector_store() as store:
        store.ensure_schema()

    log.info("neo4j.init_constraints")
    init_neo4j_constraints()

    log.info("init_infra.done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

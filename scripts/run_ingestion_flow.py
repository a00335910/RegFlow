"""Run the Prefect ingestion flow ONCE. Useful for manual triggers or smoke tests.

Usage:
    python scripts/run_ingestion_flow.py

Prefect will use a local SQLite-backed ephemeral state if no PREFECT_API_URL is set —
so this works without `prefect server start`. For the scheduled long-lived version
see scripts/serve_ingestion_schedule.py.
"""

from __future__ import annotations

import sys

from regflow.common.logging import configure_logging
from regflow.flows.ingestion_flow import regflow_ingestion_flow


def main() -> int:
    configure_logging()
    regflow_ingestion_flow()
    return 0


if __name__ == "__main__":
    sys.exit(main())

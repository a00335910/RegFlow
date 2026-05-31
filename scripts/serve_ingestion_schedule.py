"""Serve the ingestion flow as a long-running scheduled deployment.

Usage:
    # In one terminal:
    prefect server start                          # http://localhost:4200 (UI)

    # In another terminal:
    python scripts/serve_ingestion_schedule.py    # runs forever; fires on cron schedule

The flow executes on its cron schedule (default: every 4 hours on the hour) until
this process is stopped (Ctrl+C). Open the Prefect UI to see every run, retry,
and log line.

Override the schedule via env vars:
    REGFLOW_INGESTION_CRON="0 0 * * *"            # daily at midnight
    REGFLOW_INGESTION_DEPLOYMENT_NAME="my-name"
"""

from __future__ import annotations

import os
import sys

from regflow.common.logging import configure_logging
from regflow.flows.ingestion_flow import regflow_ingestion_flow


def main() -> int:
    configure_logging()

    cron = os.environ.get("REGFLOW_INGESTION_CRON", "0 */4 * * *")  # every 4 hours
    name = os.environ.get("REGFLOW_INGESTION_DEPLOYMENT_NAME", "regflow-ingestion-cron")

    print(f"Serving regflow_ingestion_flow as '{name}' with cron='{cron}'")
    print("Press Ctrl+C to stop.")
    regflow_ingestion_flow.serve(
        name=name,
        cron=cron,
        tags=["regflow", "ingestion"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

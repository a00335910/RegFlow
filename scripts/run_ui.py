"""Launch the Gradio dashboard.

Usage:
    python scripts/run_ui.py

Opens http://127.0.0.1:7860 — see four tabs:
    Overview · Obligation Explorer · Cross-Jurisdiction Conflicts · Submit Correction

Requires the Postgres/Weaviate/Neo4j services to be running (docker compose up -d)
and `python scripts/init_infra.py` to have run at least once.
"""

from __future__ import annotations

import sys

from regflow.common.logging import configure_logging
from regflow.ui import build_app


def main() -> int:
    configure_logging()
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,                # set True to get a public ngrok-style URL
        inbrowser=True,             # auto-open the browser
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

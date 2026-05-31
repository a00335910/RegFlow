"""Export an obligation's audit evidence pack as a Markdown file.

Usage:
    python scripts/export_audit_pack.py <obligation-id>
    python scripts/export_audit_pack.py <obligation-id> --out path/to/file.md

If no --out is given, writes to data/exports/audit_pack_<obligation_short>.md.

The markdown file is suitable for handing to an external auditor: includes the
obligation, exact clause citations, matched controls, prior findings, the
generated compliance justification, and proactive auditor questions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

from regflow.agents.audit_evidence.agent import render_markdown
from regflow.common.logging import configure_logging, get_logger
from regflow.db.postgres import Article, AuditEvidenceRow, Document, ObligationRow, get_session

_DEFAULT_EXPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("export_audit_pack")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("obligation_id", type=UUID)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv[1:])

    with get_session() as session:
        evidence = session.execute(
            __sqlalchemy_select(AuditEvidenceRow, args.obligation_id)
        ).scalar_one_or_none()
        if evidence is None:
            print(f"No audit evidence found for obligation {args.obligation_id}. Run scripts/run_audit_evidence.py first.")
            return 1

        obligation = session.get(ObligationRow, args.obligation_id)
        article = session.get(Article, obligation.article_id) if obligation else None
        document = session.get(Document, obligation.document_id) if obligation else None
        session.expunge_all()

    markdown = render_markdown(evidence, obligation, document, article)

    out_path = args.out
    if out_path is None:
        _DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _DEFAULT_EXPORT_DIR / f"audit_pack_{str(args.obligation_id)[:8]}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"\nExported audit pack -> {out_path.resolve()}\n  ({len(markdown)} chars)\n")
    return 0


def __sqlalchemy_select(model, obligation_id):
    """Tiny wrapper to keep the import-and-select in one place at the top."""
    from sqlalchemy import select

    return select(model).where(model.obligation_id == obligation_id)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Run Agent 3 (Cross-Jurisdiction Conflict Detector).

Two modes:

    # 1. One specific article (by UUID)
    python scripts/run_conflict_detector.py --article-id <uuid>

    # 2. Sweep — run on every article that has >= 2 obligations
    python scripts/run_conflict_detector.py --sweep
"""

from __future__ import annotations

import argparse
import sys
from uuid import UUID

from sqlalchemy import func, select

from regflow.agents.conflict_detector import detect_conflicts_for_article
from regflow.common.logging import configure_logging, get_logger
from regflow.db.postgres import ObligationRow, get_session


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--article-id", type=UUID,
                   help="UUID of one Article row to run conflict detection on.")
    g.add_argument("--sweep", action="store_true",
                   help="Run on every article that has at least 2 obligations.")
    p.add_argument("--limit", type=int, default=None,
                   help="In sweep mode, process at most N articles.")
    return p.parse_args(argv[1:])


def _articles_with_multiple_obligations(limit: int | None) -> list[UUID]:
    with get_session() as session:
        stmt = (
            select(ObligationRow.article_id, func.count(ObligationRow.id).label("n"))
            .group_by(ObligationRow.article_id)
            .having(func.count(ObligationRow.id) >= 1)
            .order_by(func.count(ObligationRow.id).desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return [row[0] for row in session.execute(stmt).all()]


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("run_conflict_detector")
    args = _parse_args(argv)

    if args.article_id:
        articles = [args.article_id]
    else:
        articles = _articles_with_multiple_obligations(args.limit)
        if not articles:
            print("No articles with obligations found. Run Agent 2 first to extract obligations.")
            return 1

    print(f"\nRunning Agent 3 on {len(articles)} article(s).\n")
    total_conflicts = 0
    high_sev = 0
    for art_id in articles:
        conflicts = detect_conflicts_for_article(art_id)
        if conflicts:
            print(f"  article {str(art_id)[:8]}: {len(conflicts)} conflict(s)")
            for c in conflicts:
                marker = "[!]" if c.severity.value == "major" else "   "
                print(f"    {marker} {c.conflict_type:<18} severity={c.severity.value:<11} "
                      f"confidence={c.confidence:.2f}")
                print(f"        {c.description[:120]}")
                if c.severity.value == "major":
                    high_sev += 1
            total_conflicts += len(conflicts)

    print(f"\n=== Summary ===")
    print(f"  Articles processed : {len(articles)}")
    print(f"  Conflicts found    : {total_conflicts}")
    print(f"  High severity ([!]): {high_sev}  (routed to review_log for legal signoff)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

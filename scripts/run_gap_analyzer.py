"""Run Agent 4 (Gap Analyzer & Risk Scorer).

Two modes:

    # 1. One specific obligation
    python scripts/run_gap_analyzer.py --obligation-id <uuid>

    # 2. Sweep — run on N obligations across jurisdictions
    python scripts/run_gap_analyzer.py --sweep --limit 10
"""

from __future__ import annotations

import argparse
import sys
from uuid import UUID

from sqlalchemy import select

from regflow.agents.gap_analyzer import analyze_gap_for_obligation
from regflow.common.logging import configure_logging, get_logger
from regflow.db.postgres import ObligationRow, get_session


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--obligation-id", type=UUID,
                   help="UUID of a single ObligationRow to analyze.")
    g.add_argument("--sweep", action="store_true",
                   help="Analyze obligations in order of descending confidence.")
    p.add_argument("--limit", type=int, default=10,
                   help="Max obligations to process in sweep mode (default 10).")
    return p.parse_args(argv[1:])


def _pick_obligations_for_sweep(limit: int) -> list[UUID]:
    with get_session() as session:
        stmt = (
            select(ObligationRow.id)
            .order_by(ObligationRow.confidence.desc())
            .limit(limit)
        )
        return [row[0] for row in session.execute(stmt).all()]


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("run_gap_analyzer")
    args = _parse_args(argv)

    targets = [args.obligation_id] if args.obligation_id else _pick_obligations_for_sweep(args.limit)
    if not targets:
        print("No obligations to analyze. Run Agent 2 to extract some first.")
        return 1

    print(f"\nRunning Agent 4 on {len(targets)} obligation(s).\n")
    high = medium = low = none = 0
    for oid in targets:
        gap = analyze_gap_for_obligation(oid)
        if gap is None:
            none += 1
            continue
        marker = "[!]" if gap.risk_level.value == "high" else "   "
        print(
            f"  {marker} obligation {str(oid)[:8]}  "
            f"risk={gap.risk_level.value:<6} score={gap.risk_score:.2f}  "
            f"missing_controls={len(gap.missing_or_weak_controls)}  "
            f"findings={len(gap.related_audit_findings)}"
        )
        if gap.risk_level.value == "high":
            high += 1
        elif gap.risk_level.value == "medium":
            medium += 1
        else:
            low += 1

    print(f"\n=== Summary ===")
    print(f"  Processed   : {len(targets)}")
    print(f"  HIGH risk   : {high}   (routed to review_log for compliance approval)")
    print(f"  MEDIUM risk : {medium}")
    print(f"  LOW risk    : {low}")
    print(f"  Skipped     : {none}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

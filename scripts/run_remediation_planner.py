"""Run Agent 5 (Remediation Planner).

Two modes:

    # 1. One specific gap
    python scripts/run_remediation_planner.py --gap-id <uuid>

    # 2. Sweep — plan remediation for the HIGH-risk gaps first, then MEDIUM, then LOW
    python scripts/run_remediation_planner.py --sweep --limit 10

    # 3. Only HIGH-risk gaps (the most useful demo subset)
    python scripts/run_remediation_planner.py --sweep --risk high --limit 10
"""

from __future__ import annotations

import argparse
import sys
from uuid import UUID

from sqlalchemy import select

from regflow.agents.remediation_planner import plan_remediation_for_gap
from regflow.common.logging import configure_logging, get_logger
from regflow.db.postgres import GapRow, get_session


_RISK_ORDER = {"high": 0, "medium": 1, "low": 2}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--gap-id", type=UUID)
    g.add_argument("--sweep", action="store_true")
    p.add_argument("--risk", choices=["high", "medium", "low"], default=None,
                   help="In sweep mode, restrict to gaps of this risk level.")
    p.add_argument("--limit", type=int, default=10)
    return p.parse_args(argv[1:])


def _pick_gaps_for_sweep(risk: str | None, limit: int) -> list[UUID]:
    with get_session() as session:
        stmt = select(GapRow.id, GapRow.risk_level, GapRow.risk_score)
        if risk:
            stmt = stmt.where(GapRow.risk_level == risk)
        rows = session.execute(stmt).all()

    rows = sorted(rows, key=lambda r: (_RISK_ORDER.get(r[1], 9), -r[2]))
    return [r[0] for r in rows[:limit]]


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("run_remediation_planner")
    args = _parse_args(argv)

    if args.gap_id:
        targets = [args.gap_id]
    else:
        targets = _pick_gaps_for_sweep(args.risk, args.limit)

    if not targets:
        print("No gaps to remediate. Run Agent 4 to generate some first.")
        return 1

    print(f"\nRunning Agent 5 on {len(targets)} gap(s)" + (f" (risk={args.risk})" if args.risk else "") + ".\n")
    total_actions = 0
    owners_unassigned = 0
    for gap_id in targets:
        actions = plan_remediation_for_gap(gap_id)
        if not actions:
            print(f"  gap {str(gap_id)[:8]}: 0 actions (LLM declined or failure)")
            continue
        print(f"  gap {str(gap_id)[:8]}: {len(actions)} action(s)")
        for a in sorted(actions, key=lambda x: x.priority):
            owner = a.suggested_owner or "(UNASSIGNED)"
            if a.suggested_owner is None:
                owners_unassigned += 1
            print(f"    [P{a.priority}] conf={a.confidence:.2f}  owner={owner}")
            print(f"          {a.description[:140]}")
        total_actions += len(actions)

    print(f"\n=== Summary ===")
    print(f"  Gaps processed       : {len(targets)}")
    print(f"  Total actions        : {total_actions}")
    print(f"  Actions w/o owner    : {owners_unassigned}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

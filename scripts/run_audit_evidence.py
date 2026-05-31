"""Run Agent 6 (Audit Evidence Generator).

Two modes:

    # 1. One obligation
    python scripts/run_audit_evidence.py --obligation-id <uuid>

    # 2. Sweep — generate packs for obligations that have a Gap from Agent 4
    python scripts/run_audit_evidence.py --sweep --limit 10
"""

from __future__ import annotations

import argparse
import sys
from uuid import UUID

from sqlalchemy import select

from regflow.agents.audit_evidence import generate_evidence_for_obligation
from regflow.common.logging import configure_logging, get_logger
from regflow.db.postgres import GapRow, get_session


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--obligation-id", type=UUID)
    g.add_argument("--sweep", action="store_true")
    p.add_argument("--limit", type=int, default=10)
    return p.parse_args(argv[1:])


def _pick_obligations(limit: int) -> list[UUID]:
    """Obligations that already have a Gap analysis (so there's something to cite)."""
    with get_session() as session:
        rows = session.execute(
            select(GapRow.obligation_id).distinct().limit(limit)
        ).all()
        return [r[0] for r in rows]


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("run_audit_evidence")
    args = _parse_args(argv)

    targets = [args.obligation_id] if args.obligation_id else _pick_obligations(args.limit)
    if not targets:
        print("No obligations to process. Run Agent 4 (gap analyzer) first.")
        return 1

    print(f"\nRunning Agent 6 on {len(targets)} obligation(s).\n")
    generated = 0
    for oid in targets:
        evidence = generate_evidence_for_obligation(oid)
        if evidence is None:
            print(f"  obligation {str(oid)[:8]}: skipped (LLM failure or missing data)")
            continue
        generated += 1
        print(
            f"  obligation {str(oid)[:8]}  "
            f"citations={len(evidence.clause_citations)}  "
            f"controls={len(evidence.control_links)}  "
            f"justification={len(evidence.justification)}chars"
        )

    print(f"\n=== Summary ===")
    print(f"  Targets processed   : {len(targets)}")
    print(f"  Evidence packs made : {generated}")
    print(f"  Skipped             : {len(targets) - generated}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

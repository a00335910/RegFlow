"""Load enterprise controls + prior audit findings from YAML into Postgres.

This is the architecture's 'Enterprise Context Layer [Manual Upload — Approach A]'
(lines 223-231) — the company-specific data Agents 3, 4, 5 read.

Usage:
    python scripts/load_enterprise_context.py
    python scripts/load_enterprise_context.py --controls-file data/sample_controls/controls.yaml \
                                              --findings-file data/sample_controls/audit_findings.yaml

Idempotent: upserts by `name` (controls) or `finding_ref` (findings). Safe to re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from sqlalchemy import select

from regflow.common.logging import configure_logging, get_logger
from regflow.db.postgres import EnterpriseControl, PriorAuditFinding, get_session

_DEFAULT_CONTROLS = Path(__file__).resolve().parents[1] / "data" / "sample_controls" / "controls.yaml"
_DEFAULT_FINDINGS = Path(__file__).resolve().parents[1] / "data" / "sample_controls" / "audit_findings.yaml"


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("load_enterprise_context")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controls-file", type=Path, default=_DEFAULT_CONTROLS)
    p.add_argument("--findings-file", type=Path, default=_DEFAULT_FINDINGS)
    args = p.parse_args(argv[1:])

    if not args.controls_file.exists():
        log.error("controls_file_missing", path=str(args.controls_file))
        return 1
    if not args.findings_file.exists():
        log.error("findings_file_missing", path=str(args.findings_file))
        return 1

    controls_payload = yaml.safe_load(args.controls_file.read_text(encoding="utf-8")) or {}
    findings_payload = yaml.safe_load(args.findings_file.read_text(encoding="utf-8")) or {}

    inserted_controls = 0
    updated_controls = 0
    inserted_findings = 0
    updated_findings = 0

    with get_session() as session:
        for entry in controls_payload.get("controls", []):
            existing = session.execute(
                select(EnterpriseControl).where(EnterpriseControl.name == entry["name"])
            ).scalar_one_or_none()
            if existing is None:
                session.add(EnterpriseControl(**entry))
                inserted_controls += 1
            else:
                for k, v in entry.items():
                    setattr(existing, k, v)
                updated_controls += 1

        for entry in findings_payload.get("findings", []):
            existing = session.execute(
                select(PriorAuditFinding).where(PriorAuditFinding.finding_ref == entry["finding_ref"])
            ).scalar_one_or_none()
            if existing is None:
                session.add(PriorAuditFinding(**entry))
                inserted_findings += 1
            else:
                for k, v in entry.items():
                    setattr(existing, k, v)
                updated_findings += 1

    log.info(
        "enterprise_context_loaded",
        controls_inserted=inserted_controls,
        controls_updated=updated_controls,
        findings_inserted=inserted_findings,
        findings_updated=updated_findings,
    )
    print(
        f"\nControls:  inserted={inserted_controls}  updated={updated_controls}\n"
        f"Findings:  inserted={inserted_findings}  updated={updated_findings}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

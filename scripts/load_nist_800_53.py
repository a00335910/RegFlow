"""Import a subset of the NIST SP 800-53 Rev 5 control catalog into Postgres as
EnterpriseControl rows.

Source: NIST OSCAL public repository
    https://github.com/usnistgov/oscal-content/blob/main/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json

Auto-downloads + caches the OSCAL JSON locally on first run (~5MB).

Usage:
    # Default: imports the families most relevant to GDPR + SEC obligations
    python scripts/load_nist_800_53.py

    # All 20 families (~1000 controls — big prompt context for Agent 4)
    python scripts/load_nist_800_53.py --all

    # Custom families
    python scripts/load_nist_800_53.py --families AC IR PT RA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

from regflow.common.logging import configure_logging, get_logger
from regflow.db.postgres import EnterpriseControl, get_session

_CATALOG_URL = (
    "https://raw.githubusercontent.com/usnistgov/oscal-content/main/"
    "nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"
)
_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_controls" / "nist_800_53_rev5_catalog.json"

# Families most relevant to RegFlow's regulatory focus (GDPR + SEC + cybersecurity).
# Roughly 50-100 controls; well within the "stuff all controls in the prompt" budget.
_DEFAULT_FAMILIES = ("AC", "AT", "AU", "IA", "IR", "PT", "RA", "SI")

# Map NIST family -> our internal category taxonomy (matches controls.yaml choices).
_FAMILY_TO_CATEGORY = {
    "AC": "information_security",      # Access Control
    "AT": "governance",                 # Awareness and Training
    "AU": "financial_reporting",        # Audit and Accountability
    "CA": "governance",                 # Assessment, Authorization, and Monitoring
    "CM": "information_security",       # Configuration Management
    "CP": "operational_resilience",     # Contingency Planning
    "IA": "information_security",       # Identification and Authentication
    "IR": "information_security",       # Incident Response
    "MA": "information_security",       # Maintenance
    "MP": "information_security",       # Media Protection
    "PE": "operational_resilience",     # Physical and Environmental Protection
    "PL": "governance",                 # Planning
    "PM": "governance",                 # Program Management
    "PS": "governance",                 # Personnel Security
    "PT": "data_protection",            # PII Processing and Transparency
    "RA": "governance",                 # Risk Assessment
    "SA": "vendor_risk",                # System and Services Acquisition
    "SC": "information_security",       # System and Communications Protection
    "SI": "information_security",       # System and Information Integrity
    "SR": "vendor_risk",                # Supply Chain Risk Management
}


def main(argv: list[str]) -> int:
    configure_logging()
    log = get_logger("load_nist_800_53")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-file", type=Path, default=_CACHE_PATH)
    p.add_argument("--catalog-url", default=_CATALOG_URL)
    p.add_argument("--families", nargs="+", default=None,
                   help=f"Family codes to import (default: {' '.join(_DEFAULT_FAMILIES)}).")
    p.add_argument("--all", action="store_true",
                   help="Import all 20 NIST families (~1000 controls). Slows Agent 4 prompts.")
    args = p.parse_args(argv[1:])

    if args.all:
        family_filter = None
    else:
        family_filter = {f.upper() for f in (args.families or _DEFAULT_FAMILIES)}

    catalog = _fetch_catalog(args.catalog_url, args.catalog_file, log)
    controls = list(_iter_controls(catalog, family_filter))

    log.info("nist.controls_parsed", count=len(controls),
             families_filter=sorted(family_filter) if family_filter else "ALL")

    inserted, updated = _upsert_controls(controls)
    log.info("nist.load_done", inserted=inserted, updated=updated)
    print(f"\nNIST 800-53 import:  inserted={inserted}  updated={updated}\n")
    return 0


def _fetch_catalog(url: str, cache_path: Path, log) -> dict[str, Any]:
    if cache_path.exists():
        log.info("nist.catalog_cache_hit", path=str(cache_path))
        return json.loads(cache_path.read_text(encoding="utf-8"))

    log.info("nist.catalog_download", url=url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    log.info("nist.catalog_cached", path=str(cache_path), bytes=cache_path.stat().st_size)
    return json.loads(resp.text)


def _iter_controls(catalog: dict[str, Any], family_filter: set[str] | None):
    """Yield {name, description, category, business_unit, ...} dicts.

    OSCAL Catalog -> groups (NIST families) -> controls -> (parts: statement, guidance).
    We skip control enhancements (nested controls) — those would multiply count by ~3.
    """
    for group in catalog.get("catalog", {}).get("groups", []) or []:
        family_id = (group.get("id") or "").upper()
        if family_filter and family_id not in family_filter:
            continue
        category = _FAMILY_TO_CATEGORY.get(family_id, "governance")
        business_unit = _family_business_unit(family_id)

        for control in group.get("controls", []) or []:
            yield _control_to_row(control, family_id, category, business_unit)


def _control_to_row(
    control: dict[str, Any], family_id: str, category: str, business_unit: str
) -> dict[str, Any]:
    cid = (control.get("id") or "").upper()
    title = (control.get("title") or "").strip()
    name = f"{cid} {title}".strip() or cid

    description_pieces: list[str] = []
    for part in control.get("parts", []) or []:
        if part.get("name") in {"statement", "guidance"}:
            prose = (part.get("prose") or "").strip()
            if prose:
                description_pieces.append(prose)
            # Nested statement parts often hold the meaningful enumerated requirements.
            for sub in part.get("parts", []) or []:
                sub_prose = (sub.get("prose") or "").strip()
                if sub_prose:
                    description_pieces.append(f"- {sub_prose}")

    description = "\n".join(description_pieces).strip() or f"NIST {cid} — {title}"
    description = description[:2000]   # keep individual control under ~500 tokens

    return {
        "name": name,
        "description": description,
        "category": category,
        "control_owner": None,
        "business_unit": business_unit,
        "evidence_uri": None,
    }


def _family_business_unit(family_id: str) -> str:
    if family_id in {"AC", "AU", "CM", "IA", "IR", "MA", "MP", "SC", "SI"}:
        return "Information Security"
    if family_id in {"PT"}:
        return "Privacy / DPO Office"
    if family_id in {"PE", "CP"}:
        return "Operations / Resilience"
    if family_id in {"SA", "SR"}:
        return "Procurement / Vendor Risk"
    return "Compliance / GRC"


def _upsert_controls(rows: list[dict[str, Any]]) -> tuple[int, int]:
    inserted = 0
    updated = 0
    with get_session() as session:
        for row in rows:
            existing = session.execute(
                select(EnterpriseControl).where(EnterpriseControl.name == row["name"])
            ).scalar_one_or_none()
            if existing is None:
                session.add(EnterpriseControl(**row))
                inserted += 1
            else:
                for k, v in row.items():
                    setattr(existing, k, v)
                updated += 1
    return inserted, updated


if __name__ == "__main__":
    sys.exit(main(sys.argv))

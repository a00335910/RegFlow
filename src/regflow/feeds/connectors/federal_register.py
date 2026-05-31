"""Federal Register connector.

Public, no-auth REST API: https://www.federalregister.gov/developers/documentation/api/v1

The connector supports two discovery modes via config/feeds.yaml:
  - Explicit list of `document_numbers` (e.g. ["2024-12345", "2024-67890"])
  - Search filters: agency, document type, date range — returns matching numbers

`fetch(document_number)` does two HTTP calls:
  1. /documents/{document_number}.json   -> metadata (title, agencies, body_html_url)
  2. body_html_url                       -> full HTML for parsing
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.feeds.base import FeedConnector, ParsedDocument
from regflow.feeds.parsers.federal_register_html import parse_federal_register_html

log = get_logger(__name__)

_BASE = "https://www.federalregister.gov/api/v1"
_FEEDS_CONFIG = Path(__file__).resolve().parents[3].parent / "config" / "feeds.yaml"


class FederalRegisterConnector(FeedConnector):
    source = "federal_register"

    def __init__(self, client: httpx.Client | None = None) -> None:
        s = get_settings().ingestion
        self._client = client or httpx.Client(
            headers={
                "User-Agent": s.user_agent,
                "Accept": "application/json, text/html;q=0.9",
            },
            timeout=s.request_timeout_s,
            follow_redirects=True,
        )

    # ---------- discover ----------

    def discover(self) -> list[str]:
        config = self._load_config()
        explicit = list(config.get("document_numbers") or [])
        if explicit:
            log.info("federal_register.discover_explicit", count=len(explicit))
            return explicit
        return self._search(config)

    def _search(self, config: dict[str, Any]) -> list[str]:
        params: list[tuple[str, str]] = []
        for agency in config.get("agencies", []) or []:
            params.append(("conditions[agencies][]", agency))
        for doc_type in config.get("document_types", ["RULE"]) or ["RULE"]:
            params.append(("conditions[type][]", doc_type))
        if config.get("date_after"):
            params.append(("conditions[publication_date][gte]", config["date_after"]))
        if config.get("date_before"):
            params.append(("conditions[publication_date][lte]", config["date_before"]))
        params.append(("per_page", str(config.get("max_results", 10))))
        params.append(("fields[]", "document_number"))
        params.append(("fields[]", "title"))
        params.append(("order", "newest"))

        url = f"{_BASE}/documents.json"
        log.info("federal_register.search", url=url, params=params)
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", []) or []
        numbers = [r["document_number"] for r in results if r.get("document_number")]
        log.info("federal_register.search_results", count=len(numbers))
        return numbers

    # ---------- fetch ----------

    def fetch(self, source_doc_id: str) -> ParsedDocument:
        meta = self._fetch_metadata(source_doc_id)
        body_url = meta.get("body_html_url")
        if not body_url:
            raise ValueError(f"Federal Register document {source_doc_id} has no body_html_url")

        log.info("federal_register.fetch_body", document_number=source_doc_id, url=body_url)
        body_resp = self._client.get(body_url)
        body_resp.raise_for_status()
        raw = body_resp.content

        title, articles, canonical = parse_federal_register_html(raw)
        # Prefer the API-supplied title; fall back to the parsed one.
        title = meta.get("title") or title

        return ParsedDocument(
            source=self.source,
            source_doc_id=source_doc_id,
            title=title,
            jurisdiction="US-FED",
            regulator=_primary_agency_name(meta),
            document_type=meta.get("type") or "RULE",
            published_date=_parse_date(meta.get("publication_date")),
            source_url=meta.get("html_url") or body_url,
            language="en",
            canonical_text=canonical,
            articles=articles,
            raw_content=raw,
            raw_content_type="text/html; charset=utf-8",
            extra_metadata={
                "document_number": source_doc_id,
                "regulation_id_numbers": meta.get("regulation_id_numbers", []) or [],
                "agencies": [a.get("slug") for a in meta.get("agencies", []) if a.get("slug")],
                "type": meta.get("type"),
                "effective_on": meta.get("effective_on"),
            },
        )

    def _fetch_metadata(self, document_number: str) -> dict[str, Any]:
        url = f"{_BASE}/documents/{document_number}.json"
        log.info("federal_register.fetch_metadata", url=url)
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.json()

    # ---------- housekeeping ----------

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _load_config() -> dict[str, Any]:
        if not _FEEDS_CONFIG.exists():
            return {}
        data = yaml.safe_load(_FEEDS_CONFIG.read_text(encoding="utf-8")) or {}
        return data.get("federal_register", {}) or {}


# ---------- helpers ----------


def _primary_agency_name(meta: dict[str, Any]) -> str:
    agencies = meta.get("agencies") or []
    if agencies and isinstance(agencies, list):
        first = agencies[0]
        if isinstance(first, dict):
            return first.get("name") or first.get("raw_name") or "US Federal Agency"
    return "US Federal Agency"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    # FR API uses "YYYY-MM-DD"
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None

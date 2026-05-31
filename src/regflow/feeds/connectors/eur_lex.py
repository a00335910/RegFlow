"""EUR-Lex connector. Fetches consolidated/published acts by CELEX number.

EUR-Lex is the EU's official legal text repository. We use the public HTML rendering
served from the ELI/CELLAR URLs. No API key required.

CELEX format reference (the 4 fields we care about for v1):
    32016R0679  -> 3 = legislation, 2016 = year, R = regulation, 0679 = number  (GDPR)
    32014L0065  -> L = directive  (MiFID II)
    32014R0596  -> Market Abuse Regulation
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import httpx
import yaml

from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.feeds.base import FeedConnector, ParsedDocument
from regflow.feeds.parsers.eurlex_html import parse_eurlex_html

log = get_logger(__name__)

# Primary URL: eur-lex.europa.eu HTML rendering. Behind Cloudflare; may serve a challenge.
_BASE_EURLEX = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
# Fallback URL: publications.europa.eu Cellar resource. Machine-harvesting endpoint,
# returns 303 redirect to actual data; less likely to challenge non-browser clients.
_BASE_CELLAR = "http://publications.europa.eu/resource/celex/"
_FEEDS_CONFIG = Path(__file__).resolve().parents[3].parent / "config" / "feeds.yaml"

# Markers that indicate Cloudflare / EUR-Lex anti-bot returned a challenge page
# instead of the regulation. If any of these appear in the response body, we abort.
_ANTI_BOT_MARKERS = (
    "JavaScript is disabled",
    "verify that you're not a robot",
    "Just a moment...",                   # Cloudflare interstitial
    "cf-browser-verification",
)


class EurLexAntiBotError(RuntimeError):
    """Raised when EUR-Lex serves a Cloudflare/anti-bot challenge instead of the document."""


class EurLexConnector(FeedConnector):
    source = "eur_lex"

    def __init__(self, client: httpx.Client | None = None) -> None:
        s = get_settings().ingestion
        self._client = client or httpx.Client(
            headers={
                "User-Agent": s.user_agent,
                # Mirror what a real browser sends so Cloudflare's heuristics pass.
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
            timeout=s.request_timeout_s,
            follow_redirects=True,
            http2=False,
        )

    def discover(self) -> list[str]:
        """Read CELEX numbers to ingest from config/feeds.yaml."""
        if not _FEEDS_CONFIG.exists():
            return []
        data = yaml.safe_load(_FEEDS_CONFIG.read_text(encoding="utf-8")) or {}
        return list(data.get("eur_lex", {}).get("celex_ids", []))

    def fetch(self, source_doc_id: str) -> ParsedDocument:
        """Try eur-lex.europa.eu first; on anti-bot challenge, fall back to Cellar."""
        try:
            return self._fetch_from(source_doc_id, f"{_BASE_EURLEX}?uri=CELEX:{source_doc_id}")
        except EurLexAntiBotError as exc:
            log.warning("eurlex.antibot_hit_trying_cellar", celex=source_doc_id, reason=str(exc))
            cellar_url = f"{_BASE_CELLAR}{source_doc_id}/EN"
            return self._fetch_from(source_doc_id, cellar_url)

    def fetch_from_local_file(self, source_doc_id: str, html_path: Path) -> ParsedDocument:
        """Skip HTTP — parse a previously-downloaded HTML file. Use when EUR-Lex blocks us."""
        log.info("eurlex.fetch_local", celex=source_doc_id, path=str(html_path))
        raw = html_path.read_bytes()
        _guard_anti_bot(raw, source_doc_id, f"file://{html_path}")
        return _build_parsed_document(
            source=self.source,
            source_doc_id=source_doc_id,
            raw=raw,
            source_url=f"file://{html_path.resolve()}",
        )

    def _fetch_from(self, source_doc_id: str, url: str) -> ParsedDocument:
        log.info("eurlex.fetch", celex=source_doc_id, url=url)
        resp = self._client.get(url)
        resp.raise_for_status()
        raw = resp.content
        _guard_anti_bot(raw, source_doc_id, url)
        return _build_parsed_document(
            source=self.source,
            source_doc_id=source_doc_id,
            raw=raw,
            source_url=str(resp.url),  # final URL after redirects (Cellar returns 303)
        )

    def close(self) -> None:
        self._client.close()


def _build_parsed_document(*, source: str, source_doc_id: str, raw: bytes, source_url: str) -> ParsedDocument:
    """Single shared builder used by HTTP + local-file paths."""
    title, articles, canonical = parse_eurlex_html(raw)
    return ParsedDocument(
        source=source,
        source_doc_id=source_doc_id,
        title=title,
        jurisdiction="EU",
        regulator=_infer_regulator(source_doc_id),
        document_type=_infer_doc_type(source_doc_id),
        published_date=_infer_published_date(source_doc_id),
        source_url=source_url,
        language="en",
        canonical_text=canonical,
        articles=articles,
        raw_content=raw,
        raw_content_type="text/html; charset=utf-8",
        extra_metadata={"celex": source_doc_id},
    )


def _guard_anti_bot(raw: bytes, celex: str, url: str) -> None:
    """Raise EurLexAntiBotError if the response is a Cloudflare/JS challenge page."""
    sample = raw[:8192].decode("utf-8", errors="ignore")
    hit = next((m for m in _ANTI_BOT_MARKERS if m in sample), None)
    if hit:
        raise EurLexAntiBotError(
            f"EUR-Lex returned an anti-bot challenge for {celex}. Marker: {hit!r}. "
            f"URL: {url}. Try a fresh browser User-Agent, or use the Cellar/Formex endpoint."
        )


def _infer_doc_type(celex: str) -> str | None:
    """CELEX position 5 (0-indexed 5) is the act type code: R=regulation, L=directive, D=decision."""
    if len(celex) < 6:
        return None
    code = celex[5]
    return {
        "R": "regulation",
        "L": "directive",
        "D": "decision",
        "H": "recommendation",
    }.get(code)


def _infer_published_date(celex: str) -> datetime | None:
    """Positions 1-4 encode the year. We can't infer the exact date without parsing the doc."""
    if len(celex) < 5 or not celex[1:5].isdigit():
        return None
    return datetime(int(celex[1:5]), 1, 1)


def _infer_regulator(celex: str) -> str:
    """Sector 3 (Parliament + Council legislation) is the EU legislature. Other sectors map differently;
    we keep this minimal for v1."""
    if not celex:
        return "European Union"
    sector = celex[0]
    return {
        "3": "European Parliament and Council",
        "0": "Founding treaties",
        "1": "External agreements",
    }.get(sector, "European Union")

"""Parse Federal Register `body_html` into article-level segments.

FR HTML uses BOTH legacy `<HD SOURCE="HDx">` tags (older XML-derived rendering) AND
modern `<h1>...<h4>` headings. We accept both. Between headings we collect `<p>`,
`<li>`, `<fp>`, `<blockquote>` content.

Two passes after collection:
  1. Build raw sections (one per heading).
  2. Merge sections smaller than _MIN_SECTION_CHARS into the previous section. This
     prevents over-segmentation where a heading like "I." or "(a)" with minimal body
     would become its own Article row — those should be absorbed into context.

Falls back to "whole body as one Article" when no headings are present (typical for
short FR notices).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from regflow.feeds.base import ParsedArticle

_WS = re.compile(r"\s+")

# Below this size in chars, a section gets merged into the previous one rather than
# becoming its own Article. Empirically tuned: numeric markers ("I.", "(a)", short
# enumerations) sit under 250 chars; real sub-sections of substance are well above.
_MIN_SECTION_CHARS = 250


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


@dataclass
class _Section:
    header: str
    content_parts: list[str] = field(default_factory=list)

    def size(self) -> int:
        return len(self.header) + sum(len(p) for p in self.content_parts)


def parse_federal_register_html(html: bytes | str) -> tuple[str, list[ParsedArticle], str]:
    """Returns (title, articles, canonical_text)."""
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("body") or soup

    # Title: prefer the first PRTPAGE or first HD; fall back to <title>.
    title_node = soup.find(["hd", "title"])
    title = _clean(title_node.get_text(" ")) if title_node else "(untitled)"

    # Walk every relevant inline element in document order, accumulating content
    # under the current heading. When we hit a new heading, flush the previous section.
    # FR HTML uses BOTH legacy `<HD>` tags (older XML-derived rendering) AND modern
    # `<h1>...<h4>` tags depending on the document; we accept both.
    heading_tags = {"hd", "h1", "h2", "h3", "h4"}
    content_tags = {"p", "li", "fp", "blockquote"}
    sections: list[_Section] = []
    current: _Section | None = None
    for el in body.find_all(heading_tags | content_tags):
        if not isinstance(el, Tag):
            continue
        name = el.name.lower()
        text = _clean(el.get_text(" "))
        if not text:
            continue
        if name in heading_tags:
            if current is not None:
                sections.append(current)
            current = _Section(header=text, content_parts=[])
        else:
            if current is None:
                current = _Section(header="(preamble)", content_parts=[])
            current.content_parts.append(text)
    if current is not None:
        sections.append(current)

    # If FR returned a document with no recognizable structure, dump the whole body.
    if not sections:
        body_text = _clean(soup.get_text(" "))
        return title, _single_article(body_text), body_text

    sections = _merge_tiny_sections(sections)

    articles: list[ParsedArticle] = []
    canonical_parts: list[str] = []
    cursor = 0
    for seq, sec in enumerate(sections):
        body_text = sec.header + "\n" + "\n".join(sec.content_parts)
        canonical_parts.append(body_text)
        articles.append(
            ParsedArticle(
                article_ref=sec.header[:64] or f"Section {seq + 1}",
                sequence=seq,
                text=body_text,
                char_start=cursor,
                char_end=cursor + len(body_text),
            )
        )
        cursor += len(body_text) + 2

    return title, articles, "\n\n".join(canonical_parts)


def _merge_tiny_sections(sections: list[_Section], min_chars: int = _MIN_SECTION_CHARS) -> list[_Section]:
    """A section smaller than min_chars is merged into the previous (larger) one.
    Its header becomes a line inside the previous section's content so the heading
    text is preserved but it doesn't generate its own Article row."""
    if not sections:
        return sections

    merged: list[_Section] = [sections[0]]
    for sec in sections[1:]:
        if sec.size() < min_chars and merged:
            merged[-1].content_parts.append(sec.header)
            merged[-1].content_parts.extend(sec.content_parts)
        else:
            merged.append(sec)
    return merged


def _single_article(body_text: str) -> list[ParsedArticle]:
    return [
        ParsedArticle(
            article_ref="Document",
            sequence=0,
            text=body_text,
            char_start=0,
            char_end=len(body_text),
        )
    ]

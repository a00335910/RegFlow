"""Parse the EUR-Lex 'EN/TXT/HTML' rendering into article-level segments.

EUR-Lex HTML uses ELI-aligned structural tags: each article is wrapped in
`<div class="eli-subdivision" id="art_<n>">` with a header `<p class="title-article-norm">`.
Recitals, annexes, and the preamble are also `eli-subdivision` blocks with different ids.

We segment everything below the preamble; the preamble itself is kept as one block
with article_ref="Preamble" so it remains searchable and diffable.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from regflow.feeds.base import ParsedArticle

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def parse_eurlex_html(html: bytes | str) -> tuple[str, list[ParsedArticle], str]:
    """Returns (title, articles, canonical_text)."""
    soup = BeautifulSoup(html, "lxml")

    title_node = soup.find("p", class_="oj-doc-ti") or soup.find("title")
    title = _clean(title_node.get_text(" ")) if title_node else "(untitled)"

    subdivisions: list[Tag] = soup.find_all("div", class_="eli-subdivision")
    articles: list[ParsedArticle] = []
    canonical_parts: list[str] = []
    cursor = 0
    seq = 0

    if not subdivisions:
        body_text = _clean(soup.get_text(" "))
        articles.append(
            ParsedArticle(
                article_ref="Document",
                sequence=0,
                text=body_text,
                char_start=0,
                char_end=len(body_text),
            )
        )
        return title, articles, body_text

    for div in subdivisions:
        div_id = div.get("id", "")
        if isinstance(div_id, list):
            div_id = div_id[0] if div_id else ""
        header = div.find(["p", "h2", "h3"], class_=re.compile(r"(title-article|title-section|title-annex|title-chapter)"))
        article_ref = _clean(header.get_text(" ")) if header else (div_id or f"Section {seq + 1}")

        text = _clean(div.get_text(" "))
        if not text:
            continue

        char_start = cursor
        canonical_parts.append(text)
        cursor += len(text) + 2  # "\n\n" separator
        char_end = char_start + len(text)

        articles.append(
            ParsedArticle(
                article_ref=article_ref[:64],
                sequence=seq,
                text=text,
                char_start=char_start,
                char_end=char_end,
            )
        )
        seq += 1

    canonical_text = "\n\n".join(canonical_parts)
    return title, articles, canonical_text

"""Phase 1 of Agent 1: deterministic article-level diff. No LLM.

Compares the current document's articles to the prior version's articles via
content_hash. Produces an `ArticleDelta` for every ADD / REMOVE / MODIFY.
UNCHANGED articles are dropped (we don't even waste a Python object on them).

For MODIFIED articles, we build a unified diff with 2 lines of context — the same
format git/PRs use — so the downstream LLM has a familiar representation to reason
over.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import Enum

from regflow.db.postgres import Article


class DeltaType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass(frozen=True)
class ArticleDelta:
    article_ref: str
    sequence: int
    change_type: DeltaType
    old_text: str | None      # None for ADDED
    new_text: str | None      # None for REMOVED
    diff_text: str            # unified diff for MODIFIED; full body otherwise
    article_id: str | None    # UUID of the current-version Article row (None for REMOVED)


def compute_deltas(
    current_articles: list[Article],
    prior_articles: list[Article] | None,
) -> list[ArticleDelta]:
    """Build the delta list. If `prior_articles` is None, every current article is ADDED."""
    if prior_articles is None:
        return [_added(a) for a in current_articles]

    prior_by_ref = {a.article_ref: a for a in prior_articles}
    current_by_ref = {a.article_ref: a for a in current_articles}

    deltas: list[ArticleDelta] = []

    for ref, curr in current_by_ref.items():
        prior = prior_by_ref.get(ref)
        if prior is None:
            deltas.append(_added(curr))
        elif prior.content_hash != curr.content_hash:
            deltas.append(_modified(prior, curr))
        # else: unchanged, skip

    for ref, prior in prior_by_ref.items():
        if ref not in current_by_ref:
            deltas.append(_removed(prior))

    deltas.sort(key=lambda d: d.sequence)
    return deltas


def _added(curr: Article) -> ArticleDelta:
    return ArticleDelta(
        article_ref=curr.article_ref,
        sequence=curr.sequence,
        change_type=DeltaType.ADDED,
        old_text=None,
        new_text=curr.text,
        diff_text=f"+++ NEW ARTICLE: {curr.article_ref}\n{curr.text}",
        article_id=str(curr.id),
    )


def _removed(prior: Article) -> ArticleDelta:
    return ArticleDelta(
        article_ref=prior.article_ref,
        sequence=prior.sequence,
        change_type=DeltaType.REMOVED,
        old_text=prior.text,
        new_text=None,
        diff_text=f"--- REMOVED ARTICLE: {prior.article_ref}\n{prior.text}",
        article_id=None,
    )


def _modified(prior: Article, curr: Article) -> ArticleDelta:
    diff_lines = difflib.unified_diff(
        prior.text.splitlines(),
        curr.text.splitlines(),
        fromfile=f"prior:{prior.article_ref}",
        tofile=f"new:{curr.article_ref}",
        n=2,                   # 2 lines of context — see design preface
        lineterm="",
    )
    return ArticleDelta(
        article_ref=curr.article_ref,
        sequence=curr.sequence,
        change_type=DeltaType.MODIFIED,
        old_text=prior.text,
        new_text=curr.text,
        diff_text="\n".join(diff_lines),
        article_id=str(curr.id),
    )

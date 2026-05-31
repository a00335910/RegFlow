"""Unit tests for Agent 1's deterministic Phase 1 (delta detector).

These tests use plain Python — no Postgres, no LLM, no Docker required.
Just `pytest tests/unit/agents/test_regulatory_radar.py -v`.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from regflow.agents.regulatory_radar.delta_detector import (
    ArticleDelta,
    DeltaType,
    compute_deltas,
)


def _article(ref: str, sequence: int, text: str) -> SimpleNamespace:
    """Build a stand-in for a SQLAlchemy Article row. delta_detector only reads attributes,
    so a SimpleNamespace works perfectly and keeps the test free of DB setup."""
    import hashlib
    return SimpleNamespace(
        id=uuid4(),
        article_ref=ref,
        sequence=sequence,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


class TestComputeDeltas:
    def test_first_ingest_marks_every_article_as_added(self) -> None:
        current = [
            _article("Article 1", 0, "First article body."),
            _article("Article 2", 1, "Second article body."),
        ]

        deltas = compute_deltas(current, prior_articles=None)

        assert len(deltas) == 2
        assert all(d.change_type is DeltaType.ADDED for d in deltas)
        assert [d.article_ref for d in deltas] == ["Article 1", "Article 2"]

    def test_identical_articles_produce_no_deltas(self) -> None:
        articles = [_article("Article 1", 0, "Unchanged body.")]

        deltas = compute_deltas(articles, prior_articles=list(articles))

        assert deltas == []

    def test_modified_article_produces_unified_diff(self) -> None:
        prior = [_article("Article 1", 0, "Reports must be filed within 5 days.")]
        current = [_article("Article 1", 0, "Reports must be filed within 7 days.")]

        deltas = compute_deltas(current, prior)

        assert len(deltas) == 1
        delta = deltas[0]
        assert delta.change_type is DeltaType.MODIFIED
        assert "5 days" in delta.diff_text
        assert "7 days" in delta.diff_text
        assert delta.diff_text.startswith("---") or "@@" in delta.diff_text  # unified-diff markers

    def test_added_article_appears_with_full_body(self) -> None:
        prior = [_article("Article 1", 0, "Old body")]
        current = [
            _article("Article 1", 0, "Old body"),
            _article("Article 2", 1, "Brand new article body"),
        ]

        deltas = compute_deltas(current, prior)

        assert len(deltas) == 1
        assert deltas[0].change_type is DeltaType.ADDED
        assert deltas[0].article_ref == "Article 2"
        assert "Brand new article body" in deltas[0].diff_text

    def test_removed_article_preserves_old_text(self) -> None:
        prior = [
            _article("Article 1", 0, "Kept"),
            _article("Article 2", 1, "About to be removed"),
        ]
        current = [_article("Article 1", 0, "Kept")]

        deltas = compute_deltas(current, prior)

        assert len(deltas) == 1
        assert deltas[0].change_type is DeltaType.REMOVED
        assert deltas[0].article_ref == "Article 2"
        assert "About to be removed" in deltas[0].diff_text
        assert deltas[0].article_id is None       # removed articles have no current row

    def test_mixed_changes_in_one_document(self) -> None:
        prior = [
            _article("Article 1", 0, "Original A"),
            _article("Article 2", 1, "Original B"),
            _article("Article 3", 2, "Original C"),
        ]
        current = [
            _article("Article 1", 0, "Original A"),               # unchanged
            _article("Article 2", 1, "MODIFIED B"),               # modified
            # Article 3 removed
            _article("Article 4", 3, "Brand new D"),              # added
        ]

        deltas = compute_deltas(current, prior)
        by_ref = {d.article_ref: d for d in deltas}

        assert len(deltas) == 3
        assert "Article 1" not in by_ref                         # unchanged is dropped
        assert by_ref["Article 2"].change_type is DeltaType.MODIFIED
        assert by_ref["Article 3"].change_type is DeltaType.REMOVED
        assert by_ref["Article 4"].change_type is DeltaType.ADDED

    def test_deltas_returned_in_sequence_order(self) -> None:
        current = [
            _article("Article 5", 4, "fifth"),
            _article("Article 1", 0, "first"),
            _article("Article 3", 2, "third"),
        ]

        deltas = compute_deltas(current, prior_articles=None)

        assert [d.sequence for d in deltas] == [0, 2, 4]


class TestArticleDelta:
    def test_delta_is_immutable(self) -> None:
        delta = ArticleDelta(
            article_ref="Article 1",
            sequence=0,
            change_type=DeltaType.ADDED,
            old_text=None,
            new_text="hello",
            diff_text="+++ NEW ARTICLE: Article 1\nhello",
            article_id=str(uuid4()),
        )
        with pytest.raises((AttributeError, TypeError)):
            delta.diff_text = "tampered"        # type: ignore[misc]

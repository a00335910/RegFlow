"""Helpers to fetch a source obligation set and its candidate cross-jurisdiction neighbors.

Two stores are queried:
  - Postgres: source obligations for the article + obligation row hydration by id
  - Weaviate: semantic search in RegulatoryCorpus for cross-jurisdiction neighbors

The retrieved neighbors are FILTERED to a different jurisdiction than the source,
so Agent 3 looks for cross-jurisdiction conflicts rather than self-overlaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select

from regflow.common.logging import get_logger
from regflow.db.postgres import ObligationRow, get_session
from regflow.db.vector import get_embedder, get_vector_store

log = get_logger(__name__)


@dataclass(frozen=True)
class CandidateObligation:
    """Trimmed view of an obligation for the LLM prompt — only what the LLM needs."""

    obligation_id: str
    jurisdiction: str
    regulator: str
    obligation_text: str
    obligation_type: str
    distance: float = 0.0      # set for neighbors; 0.0 for source obligations


def load_source_obligations(article_id: UUID) -> list[CandidateObligation]:
    """Return all obligations Agent 2 extracted for the given article."""
    with get_session() as session:
        rows = session.execute(
            select(ObligationRow).where(ObligationRow.article_id == article_id)
        ).scalars().all()
        return [
            CandidateObligation(
                obligation_id=str(r.id),
                jurisdiction=r.jurisdiction,
                regulator=r.regulator,
                obligation_text=r.obligation_text,
                obligation_type=r.obligation_type,
                distance=0.0,
            )
            for r in rows
        ]


def find_cross_jurisdiction_neighbors(
    sources: list[CandidateObligation],
    *,
    top_k_per_source: int = 8,
    max_distance: float = 0.75,
) -> list[CandidateObligation]:
    """For each source obligation, search RegulatoryCorpus for similar OBLIGATIONS
    from OTHER jurisdictions. Deduplicates by obligation_id; keeps the smallest distance.

    Filters applied directly in Weaviate (not in Python):
      - source = "agent_2_obligation"  -> only Agent-2-emitted obligations, never raw articles
      - jurisdiction != source jurisdictions  -> cross-jurisdiction by construction

    Defaults loosened (top_k 5->8, distance 0.55->0.75) because BGE-M3 distances between
    related-but-not-identical regulatory obligations tend to land in the 0.5-0.7 range;
    0.55 was too tight on real data.
    """
    if not sources:
        return []

    embedder = get_embedder()
    store = get_vector_store()
    source_jurisdictions = tuple({s.jurisdiction for s in sources})
    source_ids = {s.obligation_id for s in sources}
    seen: dict[str, CandidateObligation] = {}

    for src in sources:
        try:
            vec = embedder.embed_one(src.obligation_text)
        except Exception as exc:        # noqa: BLE001
            log.warning("conflict.embed_failed", obligation_id=src.obligation_id, error=str(exc))
            continue

        hits = store.search_corpus(
            vec,
            top_k=top_k_per_source,
            source="agent_2_obligation",
            exclude_jurisdictions=source_jurisdictions,
        )
        for hit in hits:
            uuid = hit.get("uuid")
            dist = float(hit.get("distance") or 0.0)

            if not uuid or uuid in source_ids:
                continue
            if dist > max_distance:
                continue

            # Hydrate the structured fields from Postgres for the LLM prompt.
            obligation = _hydrate(uuid)
            if obligation is None:
                continue

            existing = seen.get(uuid)
            if existing is None or dist < existing.distance:
                seen[uuid] = CandidateObligation(
                    obligation_id=uuid,
                    jurisdiction=obligation.jurisdiction,
                    regulator=obligation.regulator,
                    obligation_text=obligation.obligation_text,
                    obligation_type=obligation.obligation_type,
                    distance=dist,
                )

    neighbors = sorted(seen.values(), key=lambda c: c.distance)
    log.info(
        "conflict.neighbors_found",
        source_count=len(sources),
        neighbor_count=len(neighbors),
        top_distance=neighbors[0].distance if neighbors else None,
        max_distance=max_distance,
    )
    return neighbors


def _hydrate(obligation_uuid: str) -> ObligationRow | None:
    """Look up an obligation row by UUID. Returns None if it isn't actually an obligation
    (e.g. the Weaviate hit was a regulatory article, not an Agent-2-produced obligation)."""
    try:
        oid = UUID(obligation_uuid)
    except ValueError:
        return None
    with get_session() as session:
        row = session.get(ObligationRow, oid)
        if row is None:
            return None
        session.expunge(row)
        return row

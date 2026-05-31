"""Agent 2: Obligation Extractor (architecture lines 46-57).

Public contract (unchanged from the stub — orchestrator never has to know it's now real):
    extract_obligations(event: RegulatoryChangeEvent) -> list[Obligation]

Flow per call:
    1. Load Article + Document from Postgres (we need text + jurisdiction + regulator).
    2. Override Store retrieval — embed the article text, fetch top-k corrections for agent_2.
    3. LLM call (via the shared LiteLLM wrapper) with retrieved corrections as few-shot context.
    4. Convert ExtractedObligation -> Obligation domain type.
    5. Persist: Postgres ObligationRow + Neo4j Obligation node + Weaviate re-embed for downstream RAG.
    6. Return the list.
"""

from __future__ import annotations

from uuid import UUID

from regflow.agents.obligation_extractor.extractor import (
    ExtractedObligation,
    ExtractionResult,
    extract_from_article,
)
from regflow.common.llm import LLMError
from regflow.common.logging import get_logger
from regflow.common.settings import get_settings
from regflow.common.types import (
    Obligation,
    RegulatoryChangeEvent,
    SourceCitation,
)
from regflow.db.neo4j import upsert_obligation_node
from regflow.db.postgres import Article, Document, ObligationRow, get_session
from regflow.db.vector import get_embedder, get_vector_store
from regflow.rag.override_retriever import retrieve_corrections

log = get_logger(__name__)


def extract_obligations(event: RegulatoryChangeEvent) -> list[Obligation]:
    """Real Agent 2 — replaces the previous stub. Same signature, fully wired."""
    article, document = _load_article_for_event(event)
    if article is None or document is None:
        return []

    # ---- Override Store retrieval (the headline novelty) -----------------------
    corrections = retrieve_corrections(article.text, agent_id="agent_2", top_k=3)
    if corrections:
        log.info(
            "agent_2.corrections_retrieved",
            count=len(corrections),
            article_ref=article.article_ref,
            top_distance=corrections[0].distance,
        )

    # ---- LLM extraction --------------------------------------------------------
    try:
        result: ExtractionResult = extract_from_article(article.text, article.article_ref, corrections)
    except LLMError as exc:
        log.warning("agent_2.llm_failure", article_ref=article.article_ref, error=str(exc))
        return []

    if not result.obligations:
        log.info("agent_2.no_obligations", article_ref=article.article_ref)
        return []

    obligations = [_to_obligation(item, article, document) for item in result.obligations]

    _persist(obligations)

    log.info(
        "agent_2.extracted",
        article_ref=article.article_ref,
        count=len(obligations),
        confidence_avg=sum(o.confidence for o in obligations) / len(obligations),
    )
    return obligations


# ---------- helpers ----------


def _load_article_for_event(event: RegulatoryChangeEvent) -> tuple[Article | None, Document | None]:
    try:
        article_uuid = UUID(event.article_id)
    except ValueError:
        log.info("agent_2.skip_non_uuid_article", article_id=event.article_id)
        return None, None

    with get_session() as session:
        article = session.get(Article, article_uuid)
        if article is None:
            log.warning("agent_2.article_not_found", article_id=event.article_id)
            return None, None
        document = session.get(Document, article.document_id)
        if document is None:
            return None, None
        # Detach from session so we can use these objects after the context exits.
        session.expunge_all()
        return article, document


def _to_obligation(item: ExtractedObligation, article: Article, document: Document) -> Obligation:
    citation = SourceCitation(
        document_id=str(document.id),
        article_id=str(article.id),
        clause_ref=article.article_ref,
        text_span=article.text[:300],
        char_start=article.char_start,
        char_end=article.char_end,
    )
    return Obligation(
        article_id=str(article.id),
        document_id=str(document.id),
        jurisdiction=document.jurisdiction,
        regulator=document.regulator,
        obligation_text=item.obligation_text,
        obligation_type=item.obligation_type,
        scope=item.scope,
        deadlines=item.deadlines,
        penalties=item.penalties,
        exemptions=item.exemptions,
        citations=[citation],
        confidence=item.confidence,
    )


def _persist(obligations: list[Obligation]) -> None:
    """Postgres + Neo4j + Weaviate re-embed, all in one pass."""
    if not obligations:
        return

    # Postgres rows
    with get_session() as session:
        for obl in obligations:
            session.add(
                ObligationRow(
                    id=obl.obligation_id,
                    article_id=UUID(obl.article_id),
                    document_id=UUID(obl.document_id),
                    obligation_text=obl.obligation_text,
                    obligation_type=obl.obligation_type,
                    scope=obl.scope,
                    jurisdiction=obl.jurisdiction,
                    regulator=obl.regulator,
                    deadlines=obl.deadlines,
                    penalties=obl.penalties,
                    exemptions=obl.exemptions,
                    citations=[c.model_dump() for c in obl.citations],
                    confidence=obl.confidence,
                    extracted_at=obl.extracted_at,
                )
            )

    # Neo4j nodes (one transaction per obligation — small volume; can batch later if hot)
    for obl in obligations:
        try:
            upsert_obligation_node(obl)
        except Exception as exc:    # noqa: BLE001 — best-effort; never fail the agent on graph write
            log.warning("agent_2.neo4j_write_failed", obligation_id=str(obl.obligation_id), error=str(exc))

    # Weaviate re-embed (architecture line 54-56: re-embed new obligations for downstream search)
    try:
        _reembed_for_downstream_rag(obligations)
    except Exception as exc:        # noqa: BLE001 — graph + relational already persisted; vector is best-effort
        log.warning("agent_2.weaviate_reembed_failed", error=str(exc))


def _reembed_for_downstream_rag(obligations: list[Obligation]) -> None:
    texts = [o.obligation_text for o in obligations]
    vectors = get_embedder().embed(texts)
    store = get_vector_store()
    for obl, vec in zip(obligations, vectors, strict=True):
        # Reuse the corpus collection so semantic search hits BOTH source articles and extracted obligations.
        store.upsert_article(
            article_uuid=str(obl.obligation_id),
            document_uuid=obl.document_id,
            source="agent_2_obligation",
            source_doc_id=obl.document_id,
            article_ref=f"obligation:{obl.obligation_type}",
            jurisdiction=obl.jurisdiction,
            regulator=obl.regulator,
            text=obl.obligation_text,
            content_hash=str(obl.obligation_id),
            vector=vec,
        )

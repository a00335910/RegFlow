"""Weaviate v4 client wrapper for two collections:

- RegulatoryCorpus  — embedded article chunks (written by Ingestion + Agent 2)
- OverrideStore    — embedded `input_context` of correction records (architecture lines 233-241)

The Override Store collection has a deliberately different schema, retention, and access pattern
(architecture lines 162-166). We co-locate in Weaviate but logically separate via collections.
"""

from __future__ import annotations

import atexit
from functools import lru_cache
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter, MetadataQuery

from regflow.common.logging import get_logger
from regflow.common.settings import get_settings

log = get_logger(__name__)


class VectorStore:
    def __init__(self) -> None:
        s = get_settings().weaviate
        self._client = weaviate.connect_to_local(
            host=s.host,
            port=s.http_port,
            grpc_port=s.grpc_port,
            additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
        )
        self._corpus = s.corpus_collection
        self._overrides = s.override_collection

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ---------- schema ----------

    def ensure_schema(self) -> None:
        """Idempotent: create collections if missing. Vectorizer is `none` — we provide vectors."""
        if not self._client.collections.exists(self._corpus):
            self._client.collections.create(
                name=self._corpus,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="document_id", data_type=DataType.UUID),
                    Property(name="article_id", data_type=DataType.UUID),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="source_doc_id", data_type=DataType.TEXT),
                    Property(name="article_ref", data_type=DataType.TEXT),
                    Property(name="jurisdiction", data_type=DataType.TEXT),
                    Property(name="regulator", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="content_hash", data_type=DataType.TEXT),
                ],
            )
            log.info("weaviate.collection_created", name=self._corpus)

        if not self._client.collections.exists(self._overrides):
            self._client.collections.create(
                name=self._overrides,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="correction_id", data_type=DataType.UUID),
                    Property(name="agent_id", data_type=DataType.TEXT),
                    Property(name="correction_type", data_type=DataType.TEXT),
                    Property(name="input_context", data_type=DataType.TEXT),
                    Property(name="original_output", data_type=DataType.TEXT),
                    Property(name="corrected_output", data_type=DataType.TEXT),
                ],
            )
            log.info("weaviate.collection_created", name=self._overrides)

    # ---------- corpus writes ----------

    def upsert_article(
        self,
        *,
        article_uuid: str,
        document_uuid: str,
        source: str,
        source_doc_id: str,
        article_ref: str,
        jurisdiction: str,
        regulator: str,
        text: str,
        content_hash: str,
        vector: list[float],
    ) -> None:
        col = self._client.collections.get(self._corpus)
        col.data.insert(
            uuid=article_uuid,
            properties={
                "document_id": document_uuid,
                "article_id": article_uuid,
                "source": source,
                "source_doc_id": source_doc_id,
                "article_ref": article_ref,
                "jurisdiction": jurisdiction,
                "regulator": regulator,
                "text": text,
                "content_hash": content_hash,
            },
            vector=vector,
        )

    # ---------- corpus reads (RAG) ----------

    def search_corpus(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        jurisdiction: str | None = None,
        source: str | None = None,
        exclude_jurisdictions: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        col = self._client.collections.get(self._corpus)

        clauses: list[Any] = []
        if jurisdiction:
            clauses.append(Filter.by_property("jurisdiction").equal(jurisdiction))
        if source:
            clauses.append(Filter.by_property("source").equal(source))
        if exclude_jurisdictions:
            # Weaviate v4: chain individual not-equal filters with AND.
            for j in exclude_jurisdictions:
                clauses.append(Filter.by_property("jurisdiction").not_equal(j))

        filters: Any | None = None
        if clauses:
            filters = clauses[0]
            for c in clauses[1:]:
                filters = filters & c

        res = col.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            {**obj.properties, "uuid": str(obj.uuid), "distance": obj.metadata.distance}
            for obj in res.objects
        ]

    # ---------- override store ----------

    def upsert_correction(
        self,
        *,
        correction_uuid: str,
        agent_id: str,
        correction_type: str,
        input_context: str,
        original_output: str,
        corrected_output: str,
        vector: list[float],
    ) -> None:
        col = self._client.collections.get(self._overrides)
        col.data.insert(
            uuid=correction_uuid,
            properties={
                "correction_id": correction_uuid,
                "agent_id": agent_id,
                "correction_type": correction_type,
                "input_context": input_context,
                "original_output": original_output,
                "corrected_output": corrected_output,
            },
            vector=vector,
        )

    def search_overrides(
        self,
        query_vector: list[float],
        *,
        agent_id: str,
        top_k: int = 3,
        correction_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Architecture lines 148-154: retrieve top-k corrections filtered by agent_id."""
        col = self._client.collections.get(self._overrides)
        filters = Filter.by_property("agent_id").equal(agent_id)
        if correction_type:
            filters = filters & Filter.by_property("correction_type").equal(correction_type)
        res = col.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            {**obj.properties, "uuid": str(obj.uuid), "distance": obj.metadata.distance}
            for obj in res.objects
        ]


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    store = VectorStore()
    atexit.register(store.close)   # mirrors Postgres engine.dispose() pattern
    return store

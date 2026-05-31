"""Override Store retrieval — the system's central novelty.

For any input (article text, obligation candidate, conflict candidate, …) and any
agent_id, return the top-k past human corrections most semantically similar to the
input. Those corrections become few-shot examples injected into the LLM prompt.

Architecture lines 148-154:
    1. Embed current input
    2. Retrieve top-k by cosine similarity, filtered by agent_id (and optional correction_type)
    3. Inject into prompt as in-context guidance
    4. Agent reasons with corrections as guidance

This module is shared by all agents that consume corrections (2, 3, 4, 5, 6).
"""

from __future__ import annotations

from dataclasses import dataclass

from regflow.common.logging import get_logger
from regflow.common.types import CorrectionType
from regflow.db.vector import get_embedder, get_vector_store

log = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedCorrection:
    correction_id: str
    agent_id: str
    correction_type: str
    input_context: str
    original_output: str         # stored as JSON string in Weaviate
    corrected_output: str        # stored as JSON string in Weaviate
    distance: float              # cosine distance from the query vector (lower = more similar)


def retrieve_corrections(
    input_context: str,
    agent_id: str,
    *,
    top_k: int = 3,
    correction_type: CorrectionType | None = None,
) -> list[RetrievedCorrection]:
    """Returns empty list when the Override Store has no entries (first run / cold start).
    The caller treats an empty list as 'no few-shot examples available' and proceeds.
    """
    if not input_context.strip():
        return []

    query_vec = get_embedder().embed_one(input_context)
    raw = get_vector_store().search_overrides(
        query_vec,
        agent_id=agent_id,
        top_k=top_k,
        correction_type=correction_type.value if correction_type else None,
    )

    results = [
        RetrievedCorrection(
            correction_id=str(obj.get("correction_id", obj.get("uuid", ""))),
            agent_id=obj.get("agent_id", ""),
            correction_type=obj.get("correction_type", ""),
            input_context=obj.get("input_context", ""),
            original_output=obj.get("original_output", ""),
            corrected_output=obj.get("corrected_output", ""),
            distance=float(obj.get("distance", 0.0) or 0.0),
        )
        for obj in raw
    ]

    log.debug(
        "override.retrieved",
        agent_id=agent_id,
        correction_type=correction_type.value if correction_type else None,
        hit_count=len(results),
        top_distance=results[0].distance if results else None,
    )
    return results

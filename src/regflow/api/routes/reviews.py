"""Human Review endpoints.

POST /reviews/corrections  — submit a reviewer correction; writes to Override Store.

This is the human-side endpoint that CLOSES the correction-retrieval loop.
Every record submitted here becomes retrievable few-shot guidance for the
corresponding agent at its next inference (architecture lines 131-170).
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from regflow.api.schemas.reviews import CorrectionResponse, CorrectionSubmission
from regflow.common.logging import get_logger
from regflow.db.postgres import CorrectionRecordRow, ReviewLogEntry, get_session
from regflow.db.vector import get_embedder, get_vector_store

log = get_logger(__name__)
router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.post("/corrections", response_model=CorrectionResponse, status_code=201)
def submit_correction(payload: CorrectionSubmission) -> CorrectionResponse:
    """Persist a reviewer correction across both halves of the Override Store.

    Step-by-step (architecture lines 138-166):
      1. Embed input_context (this is the key the agent will search by next time).
      2. Write the vector + properties to Weaviate's OverrideStore collection.
      3. Write the structured row to Postgres for audit + tabular query.
      4. Append a ReviewLogEntry for the audit trail.
    """
    correction_id = uuid4()
    vector_uuid = str(correction_id)   # reuse the id; both stores join cleanly.

    # 1. Embed.
    try:
        embedding = get_embedder().embed_one(payload.input_context)
    except Exception as exc:           # noqa: BLE001
        log.error("review.embed_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"embedding failed: {exc}") from exc

    # 2. Weaviate write — vector half of the Override Store.
    try:
        get_vector_store().upsert_correction(
            correction_uuid=vector_uuid,
            agent_id=payload.agent_id,
            correction_type=payload.correction_type.value,
            input_context=payload.input_context,
            original_output=json.dumps(payload.original_output),
            corrected_output=json.dumps(payload.corrected_output),
            vector=embedding,
        )
    except Exception as exc:           # noqa: BLE001
        log.error("review.weaviate_write_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"vector store write failed: {exc}") from exc

    # 3. Postgres write — structured / audit half. Same correction_id so the two halves join.
    created_at = datetime.utcnow()
    with get_session() as session:
        session.add(
            CorrectionRecordRow(
                id=correction_id,
                agent_id=payload.agent_id,
                correction_type=payload.correction_type.value,
                input_context=payload.input_context,
                original_output=payload.original_output,
                corrected_output=payload.corrected_output,
                reviewer_id=payload.reviewer_id,
                vector_uuid=vector_uuid,
                created_at=created_at,
            )
        )
        # 4. Audit trail.
        session.add(
            ReviewLogEntry(
                trigger="correction_submitted",
                agent_id=payload.agent_id,
                subject_type="correction_record",
                subject_id=correction_id,
                reviewer_id=payload.reviewer_id,
                decision="modified",
                notes=payload.note,
                payload={"correction_type": payload.correction_type.value},
            )
        )

    log.info(
        "review.correction_persisted",
        correction_id=str(correction_id),
        agent_id=payload.agent_id,
        correction_type=payload.correction_type.value,
        reviewer_id=payload.reviewer_id,
    )

    return CorrectionResponse(
        correction_id=correction_id,
        vector_uuid=vector_uuid,
        agent_id=payload.agent_id,
        correction_type=payload.correction_type.value,
        created_at=created_at,
    )

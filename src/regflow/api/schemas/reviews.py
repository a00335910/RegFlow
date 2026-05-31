"""Request/response schemas for the human review endpoints.

Pydantic-validated wire shapes — kept separate from the domain types in
common/types.py so the API surface can evolve independently of the internal
data model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from regflow.common.types import CorrectionType


class CorrectionSubmission(BaseModel):
    """POST /reviews/corrections body.

    The reviewer says: 'agent X looked at this input, produced this output, but it
    should have produced this corrected output instead.' We embed input_context
    so future agent calls with semantically similar inputs can retrieve this lesson.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: Literal["agent_1", "agent_2", "agent_3", "agent_4", "agent_5", "agent_6"]
    correction_type: CorrectionType
    input_context: str = Field(min_length=10, max_length=20000)
    original_output: dict[str, Any]
    corrected_output: dict[str, Any]
    reviewer_id: str = Field(min_length=1, max_length=128)
    note: str | None = Field(default=None, max_length=2000)


class CorrectionResponse(BaseModel):
    correction_id: UUID
    vector_uuid: str
    agent_id: str
    correction_type: str
    created_at: datetime

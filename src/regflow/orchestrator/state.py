"""Shared state for the LangGraph workflow.

Every node in the graph receives this dict, mutates it, and returns it. LangGraph
threads it through the graph automatically. Keeping all transient pipeline state
in one TypedDict makes the workflow easy to reason about and easy to serialize
for checkpointing later.
"""

from __future__ import annotations

from typing import TypedDict
from uuid import UUID

from regflow.common.types import Obligation, RegulatoryChangeEvent


class WorkflowState(TypedDict, total=False):
    # ---- inputs (set by caller) ----
    document_id: UUID
    radar_limit: int | None
    radar_include_prefixes: tuple[str, ...] | None

    # ---- populated by radar_node ----
    events: list[RegulatoryChangeEvent]

    # ---- populated by route_node ----
    # Keys are RouteDecision.value strings ("auto" / "notify" / "block") — strings
    # rather than enums so LangGraph's optional checkpointing can serialize them.
    routing: dict[str, list[RegulatoryChangeEvent]]

    # ---- populated by dispatch_node ----
    obligations: list[Obligation]
    review_log_entries_written: int

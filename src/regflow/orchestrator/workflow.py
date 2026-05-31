"""LangGraph workflow — the Control Plane (architecture lines 35-43).

Graph shape:
                              ┌─> AUTO dispatch
   radar ─> route ─> dispatch ─┼─> NOTIFY dispatch + review_log write
                              └─> BLOCK dispatch (review_log write only)
                       │
                       ▼
                   finalize

The dispatch node currently loops in Python. When Agent 2 is real, we'll
refactor the fan-out using LangGraph's `Send` API for true parallelism per
architecture lines 60-75. The state shape and routing logic don't change —
only the dispatch node body.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from langgraph.graph import END, StateGraph

from regflow.agents.obligation_extractor import extract_obligations
from regflow.agents.regulatory_radar import run_radar
from regflow.common.logging import get_logger
from regflow.common.types import Obligation, RegulatoryChangeEvent, RouteDecision
from regflow.db.postgres import ReviewLogEntry, get_session
from regflow.orchestrator.router import route_event
from regflow.orchestrator.state import WorkflowState

log = get_logger(__name__)

_ROUTE_KEYS = (RouteDecision.AUTO.value, RouteDecision.NOTIFY.value, RouteDecision.BLOCK.value)


# ---------- nodes ----------


def radar_node(state: WorkflowState) -> WorkflowState:
    """Phase 1 of the workflow: invoke Agent 1, collect RegulatoryChangeEvents."""
    events = run_radar(
        state["document_id"],
        limit=state.get("radar_limit"),
        include_prefixes=state.get("radar_include_prefixes"),
    )
    state["events"] = events
    log.info("orchestrator.radar_done", event_count=len(events))
    return state


def route_node(state: WorkflowState) -> WorkflowState:
    """Bucket each event into AUTO / NOTIFY / BLOCK."""
    buckets: dict[str, list[RegulatoryChangeEvent]] = {k: [] for k in _ROUTE_KEYS}
    for evt in state["events"]:
        decision = route_event(evt)
        buckets[decision.value].append(evt)

    state["routing"] = buckets
    log.info(
        "orchestrator.routed",
        auto=len(buckets[RouteDecision.AUTO.value]),
        notify=len(buckets[RouteDecision.NOTIFY.value]),
        block=len(buckets[RouteDecision.BLOCK.value]),
    )
    return state


def dispatch_node(state: WorkflowState) -> WorkflowState:
    """Walk each routing bucket. BLOCK: review_log only. NOTIFY: review_log + Agent 2. AUTO: Agent 2."""
    routing = state["routing"]
    obligations: list[Obligation] = []
    log_writes = 0

    with get_session() as session:
        # BLOCK: log only, do NOT dispatch to Agent 2.
        for evt in routing[RouteDecision.BLOCK.value]:
            session.add(_review_log_for(evt, trigger="orchestrator_block"))
            log_writes += 1

        # NOTIFY: log AND dispatch to Agent 2 (process now, flag humans).
        for evt in routing[RouteDecision.NOTIFY.value]:
            session.add(_review_log_for(evt, trigger="orchestrator_notify"))
            log_writes += 1
            obligations.extend(extract_obligations(evt))

        # AUTO: dispatch only.
        for evt in routing[RouteDecision.AUTO.value]:
            obligations.extend(extract_obligations(evt))

    state["obligations"] = obligations
    state["review_log_entries_written"] = log_writes
    log.info(
        "orchestrator.dispatched",
        obligations=len(obligations),
        review_log_entries=log_writes,
    )
    return state


def finalize_node(state: WorkflowState) -> WorkflowState:
    """Placeholder for any post-pipeline aggregation (metrics, status flips, etc.)."""
    return state


def _review_log_for(evt: RegulatoryChangeEvent, *, trigger: str) -> ReviewLogEntry:
    return ReviewLogEntry(
        trigger=trigger,
        agent_id="agent_1",
        subject_type="regulatory_change_event",
        subject_id=evt.event_id,
        payload={
            "severity": evt.severity.value,
            "confidence": evt.confidence,
            "diff_summary": evt.diff_summary,
            "article_id": evt.article_id,
            "document_id": evt.document_id,
            "jurisdiction": evt.jurisdiction,
            "regulator": evt.regulator,
        },
    )


# ---------- graph assembly ----------


def build_graph():
    g = StateGraph(WorkflowState)
    g.add_node("radar", radar_node)
    g.add_node("route", route_node)
    g.add_node("dispatch", dispatch_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("radar")
    g.add_edge("radar", "route")
    g.add_edge("route", "dispatch")
    g.add_edge("dispatch", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


# Module-level compilation: build once, reuse across invocations.
_COMPILED_GRAPH = build_graph()


# ---------- public API ----------


@dataclass
class WorkflowResult:
    document_id: UUID
    events: list[RegulatoryChangeEvent]
    auto: list[RegulatoryChangeEvent]
    notify: list[RegulatoryChangeEvent]
    block: list[RegulatoryChangeEvent]
    obligations: list[Obligation]
    review_log_entries_written: int


def run_workflow(
    document_id: UUID,
    *,
    radar_limit: int | None = None,
    radar_include_prefixes: tuple[str, ...] | None = None,
) -> WorkflowResult:
    initial: WorkflowState = {
        "document_id": document_id,
        "radar_limit": radar_limit,
        "radar_include_prefixes": radar_include_prefixes,
        "events": [],
        "routing": {k: [] for k in _ROUTE_KEYS},
        "obligations": [],
        "review_log_entries_written": 0,
    }

    final: WorkflowState = _COMPILED_GRAPH.invoke(initial)

    return WorkflowResult(
        document_id=document_id,
        events=final["events"],
        auto=final["routing"][RouteDecision.AUTO.value],
        notify=final["routing"][RouteDecision.NOTIFY.value],
        block=final["routing"][RouteDecision.BLOCK.value],
        obligations=final["obligations"],
        review_log_entries_written=final["review_log_entries_written"],
    )

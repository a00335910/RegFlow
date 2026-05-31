"""Confidence-and-severity routing logic.

Pure function — no I/O, no LLM. Unit-testable in isolation. Called by route_node
inside the LangGraph workflow.

Policy (matches the design preface):
  - severity == MAJOR (any confidence)            -> BLOCK   (architecture line 86)
  - confidence < extraction_min_confidence        -> BLOCK
  - confidence >= auto_confidence_threshold       -> AUTO
  - otherwise (medium confidence, non-major)      -> NOTIFY
"""

from __future__ import annotations

from regflow.common.settings import OrchestratorSettings, get_settings
from regflow.common.types import RegulatoryChangeEvent, RouteDecision, Severity


def route_event(
    event: RegulatoryChangeEvent,
    settings: OrchestratorSettings | None = None,
) -> RouteDecision:
    s = settings or get_settings().orchestrator

    if event.severity == Severity.MAJOR:
        return RouteDecision.BLOCK
    if event.confidence < s.extraction_min_confidence:
        return RouteDecision.BLOCK
    if event.confidence >= s.auto_confidence_threshold:
        return RouteDecision.AUTO
    return RouteDecision.NOTIFY

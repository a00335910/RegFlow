"""LiteLLM wrapper used by every agent.

Single function: `complete_json(messages, schema)` — calls the configured LLM, parses
its JSON response, validates against a Pydantic schema, and retries on bad output.

Why this exists:
  - All agents call the LLM through this module, not litellm directly. That means
    swapping providers (Ollama -> vLLM -> OpenAI-compatible) is one config change.
  - Centralizes JSON-mode setup. Ollama uses `format="json"`; OpenAI-style endpoints
    use `response_format={"type": "json_object"}`. We send both — providers ignore
    the one they don't understand.
  - Centralizes retry policy (tenacity). If validation fails, we retry the LLM call
    up to `settings.llm.max_retries`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from regflow.common.logging import get_logger
from regflow.common.settings import get_settings

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

# Some models wrap JSON in ```json ... ``` even when told not to. Strip it defensively.
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

# Module-state: configure LiteLLM callbacks (Langfuse) exactly once per process.
_LANGFUSE_CONFIGURED = False


def _ensure_langfuse_configured() -> None:
    """Idempotent wiring of LiteLLM's Langfuse callback.

    LiteLLM reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from
    the process environment. We project our pydantic-settings values into env
    here so the user doesn't have to set both YAML and env.

    If keys are not configured, Langfuse stays disabled — every LLM call still
    works, just no observability data is emitted. Graceful degradation.
    """
    global _LANGFUSE_CONFIGURED
    if _LANGFUSE_CONFIGURED:
        return

    settings = get_settings().langfuse
    if not (settings.enabled and settings.secret_key and settings.public_key):
        _LANGFUSE_CONFIGURED = True
        return

    os.environ.setdefault("LANGFUSE_HOST", settings.host)
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.public_key.get_secret_value())
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.secret_key.get_secret_value())
    os.environ.setdefault("LANGFUSE_FLUSH_AT", str(settings.flush_at))
    os.environ.setdefault("LANGFUSE_FLUSH_INTERVAL", str(settings.flush_interval))

    # LiteLLM hooks: every successful AND failed completion is shipped to Langfuse.
    if "langfuse" not in (litellm.success_callback or []):
        litellm.success_callback = [*(litellm.success_callback or []), "langfuse"]
    if "langfuse" not in (litellm.failure_callback or []):
        litellm.failure_callback = [*(litellm.failure_callback or []), "langfuse"]

    log.info("langfuse.configured", host=settings.host)
    _LANGFUSE_CONFIGURED = True


def _extract_json(raw: str) -> str:
    """Strip markdown code fences if present; trim whitespace."""
    return _CODE_FENCE.sub("", raw).strip()


class LLMError(RuntimeError):
    """Raised when the LLM call fails after all retries."""


class LLMSchemaError(LLMError):
    """Raised when the model's output never matches the requested schema."""


def complete_json(
    messages: list[dict[str, str]],
    schema: type[T],
    *,
    model: str | None = None,
    temperature: float | None = None,
    trace_name: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> T:
    """Call the LLM and return a validated instance of `schema`.

    Args:
        messages: OpenAI-style chat messages, e.g. [{"role": "system", "content": "..."}, ...].
        schema:   A Pydantic model class. The model's JSON output is validated against it.
        model:    Override the configured model (e.g., use a smaller classifier model).
        temperature: Override the configured temperature.
        trace_name: Logical name for this LLM call (e.g. "agent_2_extractor") — used to
                    group calls in the Langfuse dashboard. Optional but recommended.
        trace_metadata: Extra fields attached to the Langfuse trace (e.g. obligation_id,
                        article_ref). Useful for filtering in the Langfuse UI.

    Raises:
        LLMError / LLMSchemaError after `settings.llm.max_retries` failed attempts.
    """
    _ensure_langfuse_configured()

    settings = get_settings().llm
    target_model = model or settings.extraction_model
    target_temp = settings.temperature if temperature is None else temperature

    # LiteLLM forwards `metadata` to its observability callbacks (Langfuse here).
    # Recognized keys: generation_name, trace_name, trace_user_id, tags, session_id.
    obs_metadata: dict[str, Any] = {
        "generation_name": trace_name or "complete_json",
        "trace_name": trace_name or "regflow_llm_call",
        "tags": [trace_name] if trace_name else ["regflow"],
        "trace_user_id": "regflow",
    }
    if trace_metadata:
        obs_metadata.update(trace_metadata)

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((json.JSONDecodeError, ValidationError, litellm.exceptions.APIError)),
        reraise=True,
    )
    def _call() -> T:
        response = litellm.completion(
            model=target_model,
            messages=messages,
            api_base=settings.base_url,
            temperature=target_temp,
            timeout=settings.request_timeout_s,
            response_format={"type": "json_object"},   # OpenAI-style hint
            format="json",                              # Ollama-specific hint
            metadata=obs_metadata,
        )
        raw = response.choices[0].message.content or ""
        cleaned = _extract_json(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("llm.invalid_json", model=target_model, raw_preview=raw[:200])
            raise
        try:
            return schema.model_validate(parsed)
        except ValidationError:
            log.warning("llm.schema_mismatch", model=target_model, raw_preview=raw[:200])
            raise

    try:
        return _call()
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMSchemaError(
            f"LLM output failed schema validation after {settings.max_retries} attempts: {exc}"
        ) from exc
    except RetryError as exc:
        raise LLMError(f"LLM call failed after {settings.max_retries} attempts: {exc}") from exc
    except Exception as exc:
        raise LLMError(f"LLM call failed: {exc}") from exc

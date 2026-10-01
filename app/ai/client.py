"""The one place a model is named.

Every AI task calls through here; no task constructs a chat model of its own.
That is what made swapping provider a one-section documentation change, and it
is enforced by a test rather than by convention -- ``grep`` for a model id
should find this file and ``app/core/config.py`` and nothing else.

Three things this module owns:

*   **Role -> model.** Tasks ask for ``primary`` or ``cheap``, never for
    ``openai/gpt-oss-120b``. A task that names a model has to be edited when
    the model changes; a task that names a role never does.
*   **Structured output.** ``method="json_schema"`` is Groq's Structured Output
    API. Whether that means *guaranteed* adherence depends on the installed
    client -- see the note below. ``docs/ai/README.md`` section 7 has the two
    standing consequences: every field must be required, and structured output
    cannot be combined with streaming or tool use.
*   **Accounting.** Every call returns an :class:`AIRun` beside its result and
    logs one structured line. ``docs/schema/README.md`` section 10 deliberately
    has no ``ai_runs`` table, so this log *is* the run record.

``langchain_groq`` is imported normally, at module scope. Nothing in
``app.main`` imports this package, so the API is unaffected either way; what
gates a model call is :func:`_require_enabled` -- ``ai_enabled`` plus a key --
not the availability of the import.

**Constrained decoding is version-dependent.** ``method="json_schema"`` is
Groq's Structured Output API, but only its ``strict: true`` mode uses
constrained decoding and therefore guarantees the schema. langchain-groq 0.3.8
(what pip resolves on Python 3.9) builds ``response_format`` without ``strict``
and has no parameter to set it, so there the mode is best-effort: valid JSON,
adherence not guaranteed. 1.x adds ``strict``. :data:`_SUPPORTS_STRICT` picks
the stronger mode where it exists, which means the *validation* below is not
belt-and-braces -- on 0.3.8 it is the only thing standing between a malformed
response and a written claim.
"""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from langchain_groq import ChatGroq

from app.core.config import settings
from app.ai.governor import RateLimitGovernor, estimate_tokens

logger = logging.getLogger("dealpilot.ai")

# Whether the installed langchain-groq can request Groq's strict mode. Probed
# once rather than pinned to a version string: the answer is "does this
# function take this argument", and asking directly cannot drift.
_SUPPORTS_STRICT = (
    "strict" in inspect.signature(ChatGroq.with_structured_output).parameters
)

# Roles, not sizes. docs/ai/README.md section 7 assigns them.
ROLE_PRIMARY = "primary"
ROLE_CHEAP = "cheap"
ROLE_CHALLENGER = "challenger"

_MAX_TOKENS_BY_TASK = {
    "extract": lambda: settings.ai_max_tokens_extract,
    "validate": lambda: settings.ai_max_tokens_validate,
    "detect": lambda: settings.ai_max_tokens_detect,
    "synthesize": lambda: settings.ai_max_tokens_synthesize,
    "chat": lambda: settings.ai_max_tokens_chat,
}


class AIDisabled(RuntimeError):
    """Raised when a task is invoked with ``ai_enabled=False`` or no API key.

    A distinct type rather than a generic error: a route can turn this into a
    503 "AI is not configured", which is a true and actionable answer, where a
    500 is neither.
    """


class TokenCeilingExceeded(RuntimeError):
    """One pipeline run tried to spend more than ``ai_token_ceiling_per_run``.

    Bounds the cost of a prompt bug or a runaway loop to a single run.
    """


@dataclass
class AIRun:
    """What one model call cost and whether it worked.

    Written to the log rather than to a table -- see the module docstring.
    """

    task: str
    model: str
    prompt_version: str
    outcome: str = "ok"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: int = 0
    attempts: int = 1
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> Optional[int]:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    def log(self) -> None:
        logger.info(
            "ai.run task=%s model=%s version=%s outcome=%s in=%s out=%s ms=%s attempts=%s%s",
            self.task,
            self.model,
            self.prompt_version,
            self.outcome,
            self.input_tokens,
            self.output_tokens,
            self.latency_ms,
            self.attempts,
            " error=%r" % self.error if self.error else "",
            extra={"ai_run": self.__dict__},
        )


class RunBudget:
    """Per-run token ceiling, passed down a pipeline.

    A pipeline creates one of these and hands it to every stage, so the limit
    applies to the run rather than to each call -- which is the level at which
    a runaway actually costs money.
    """

    def __init__(self, ceiling: Optional[int] = None) -> None:
        self.ceiling = settings.ai_token_ceiling_per_run if ceiling is None else ceiling
        self.spent = 0

    def charge(self, tokens: Optional[int]) -> None:
        self.spent += tokens or 0
        if self.spent > self.ceiling:
            raise TokenCeilingExceeded(
                "run spent %d tokens, ceiling is %d" % (self.spent, self.ceiling)
            )


_governor: Optional[RateLimitGovernor] = None


def governor() -> RateLimitGovernor:
    """Process-wide, created on first use.

    Built lazily because it holds an ``asyncio.Semaphore``, which binds to the
    running loop -- constructing it at import time couples it to whichever loop
    happened to exist then.
    """
    global _governor
    if _governor is None:
        _governor = RateLimitGovernor(
            max_concurrency=settings.groq_max_concurrency,
            tokens_per_minute=settings.groq_tokens_per_minute,
            requests_per_minute=settings.groq_requests_per_minute,
        )
    return _governor


def model_for(role: str) -> str:
    if role == ROLE_PRIMARY:
        return settings.groq_model_primary
    if role == ROLE_CHEAP:
        return settings.groq_model_cheap
    if role == ROLE_CHALLENGER:
        return settings.groq_model_challenger
    raise ValueError("unknown role %r" % role)


def max_tokens_for(task: str) -> int:
    getter = _MAX_TOKENS_BY_TASK.get(task)
    return getter() if getter else settings.ai_max_tokens_default


def _require_enabled() -> None:
    if not settings.ai_enabled:
        raise AIDisabled("ai_enabled is false")
    if not settings.groq_api_key:
        raise AIDisabled("GROQ_API_KEY is not set")


def chat_model(
    *,
    role: str = ROLE_PRIMARY,
    task: str = "default",
    reasoning_effort: Optional[str] = None,
    model: Optional[str] = None,
):
    """Build a configured ``ChatGroq``.

    ``max_retries=0`` on purpose. LangChain's own retry would re-send without
    passing back through the governor, which is precisely the behaviour that
    turns one 429 into several -- retries are handled in :func:`_invoke`, where
    the rate limiter can see them.

    ``temperature=0``: every task here is extraction, classification or
    entailment. None of them wants sampling variance, and stability across runs
    is a product requirement (docs/ai/README.md section 10, the flicker metric).
    """
    _require_enabled()

    kwargs: Dict[str, Any] = {
        "model": model or model_for(role),
        "api_key": settings.groq_api_key,
        "temperature": 0,
        "max_tokens": max_tokens_for(task),
        "timeout": settings.ai_request_timeout_seconds,
        "max_retries": 0,
    }
    # Only the gpt-oss models accept it; passing it to another model is a 400.
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


def _is_retryable(exc: BaseException) -> bool:
    """Whether one more attempt is worth making.

    Matched loosely and on purpose: the exception type depends on which
    langchain-groq version pip resolved, so keying on a concrete class would
    silently stop retrying after an upgrade. A 4xx that is not 429 is the
    caller's fault and retrying it only burns budget.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "timeout", "timed out", "connection", "503", "502", "overloaded")
    )


def _usage(raw: Any) -> Tuple[Optional[int], Optional[int]]:
    """Pull input/output tokens off a response, whatever shape it arrived in."""
    usage = getattr(raw, "usage_metadata", None) or {}
    if not usage:
        meta = getattr(raw, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
    return (
        usage.get("input_tokens", usage.get("prompt_tokens")),
        usage.get("output_tokens", usage.get("completion_tokens")),
    )


async def _invoke(runnable: Any, messages: Any, run: AIRun, estimated: int) -> Any:
    """One call, governed, with at most ``ai_max_retries`` further attempts."""
    gov = governor()
    attempts = settings.ai_max_retries + 1
    started = time.monotonic()
    last_exc: Optional[BaseException] = None

    for attempt in range(1, attempts + 1):
        run.attempts = attempt
        try:
            async with gov.reserve(estimated):
                result = await runnable.ainvoke(messages)
            run.latency_ms = int((time.monotonic() - started) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001 -- classified by _is_retryable
            last_exc = exc
            if attempt >= attempts or not _is_retryable(exc):
                break
            # Linear, not exponential: the governor is already pacing us, so a
            # long backoff here just idles the worker.
            await asyncio.sleep(1.0 * attempt)

    run.latency_ms = int((time.monotonic() - started) * 1000)
    run.outcome = "error"
    run.error = "%s: %s" % (type(last_exc).__name__, last_exc)
    run.log()
    raise last_exc  # type: ignore[misc]


def _render(messages: Sequence[Tuple[str, str]]) -> str:
    return "\n".join(content for _, content in messages)


async def structured(
    schema: Any,
    messages: Sequence[Tuple[str, str]],
    *,
    task: str,
    prompt_version: str,
    role: str = ROLE_PRIMARY,
    reasoning_effort: Optional[str] = None,
    model: Optional[str] = None,
    budget: Optional[RunBudget] = None,
) -> Tuple[Any, AIRun]:
    """Call a model and get back an instance of ``schema``.

    ``include_raw=True`` is not optional here: without it the parsed object
    arrives alone and the usage metadata -- which is the whole of the run
    record -- is discarded. The result is a dict of ``raw`` / ``parsed`` /
    ``parsing_error``.

    A ``parsing_error`` is raised rather than returned: a task that silently
    receives ``None`` would write a claim with no content. Under constrained
    decoding it should be unreachable, so it means the stronger mode was not
    applied; on 0.3.8, where it never is, it simply means the model returned
    the wrong shape. Either way the claim must not land.
    """
    llm = chat_model(role=role, task=task, reasoning_effort=reasoning_effort, model=model)
    structured_kwargs: Dict[str, Any] = {
        "method": settings.structured_output_method,
        "include_raw": True,
    }
    if _SUPPORTS_STRICT and settings.structured_output_method == "json_schema":
        # Constrained decoding. Silently absent on 0.3.8 -- see the module
        # docstring; do not pass it blind, it would reach Groq as an unknown
        # top-level parameter.
        structured_kwargs["strict"] = True
    runnable = llm.with_structured_output(schema, **structured_kwargs)

    run = AIRun(task=task, model=model or model_for(role), prompt_version=prompt_version)
    estimated = estimate_tokens(_render(messages)) + max_tokens_for(task)

    result = await _invoke(runnable, list(messages), run, estimated)

    raw = result.get("raw") if isinstance(result, dict) else None
    parsed = result.get("parsed") if isinstance(result, dict) else result
    run.input_tokens, run.output_tokens = _usage(raw)
    await governor().reconcile(estimated, run.total_tokens)
    if budget is not None:
        budget.charge(run.total_tokens)

    error = result.get("parsing_error") if isinstance(result, dict) else None
    if error is not None or parsed is None:
        run.outcome = "parse_error"
        run.error = str(error) if error else "parsed output was None"
        run.log()
        raise ValueError(
            "structured output failed for task=%s method=%s: %s"
            % (task, settings.structured_output_method, run.error)
        )

    run.log()
    return parsed, run


async def text(
    messages: Sequence[Tuple[str, str]],
    *,
    task: str,
    prompt_version: str,
    role: str = ROLE_PRIMARY,
    reasoning_effort: Optional[str] = None,
    budget: Optional[RunBudget] = None,
) -> Tuple[str, AIRun]:
    """Unstructured completion, for the few places a schema would be noise."""
    llm = chat_model(role=role, task=task, reasoning_effort=reasoning_effort)
    run = AIRun(task=task, model=model_for(role), prompt_version=prompt_version)
    estimated = estimate_tokens(_render(messages)) + max_tokens_for(task)

    message = await _invoke(llm, list(messages), run, estimated)

    run.input_tokens, run.output_tokens = _usage(message)
    await governor().reconcile(estimated, run.total_tokens)
    if budget is not None:
        budget.charge(run.total_tokens)
    run.log()
    return getattr(message, "content", ""), run

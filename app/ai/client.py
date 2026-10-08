"""The one place a model is named.

Every AI task calls through here; no task constructs a chat model of its own.
That is what made swapping provider a one-section documentation change, and it
is enforced by a test rather than by convention -- ``grep`` for a model id
should find this file and ``app/core/config.py`` and nothing else.

Three things this module owns:

*   **Role -> model.** Tasks ask for ``primary`` or ``cheap``, never for
    ``gpt-4o``. A task that names a model has to be edited when the model
    changes; a task that names a role never does.
*   **Structured output.** ``method="json_schema"`` with ``strict`` is OpenAI's
    Structured Outputs -- constrained decoding, so the schema is guaranteed.
    ``docs/ai/README.md`` section 7 has the standing consequence: every field
    must be required.
*   **Accounting.** Every call returns an :class:`AIRun` beside its result and
    logs one structured line. ``docs/schema/README.md`` section 10 deliberately
    has no ``ai_runs`` table, so this log *is* the run record.

``langchain_openai`` is imported normally, at module scope. Nothing in
``app.main`` imports this package, so the API is unaffected either way; what
gates a model call is :func:`_require_enabled` -- ``ai_enabled`` plus a key --
not the availability of the import.

**Two kinds of model.** OpenAI's reasoning models (o-series, ``gpt-5*``) take
``reasoning_effort`` and reject ``temperature``; gpt-4o and the other chat
models are the reverse, and either mistake is a 400. :func:`is_reasoning_model`
decides which one :func:`chat_model` sends, so call sites always pass their
per-task ``reasoning_effort`` and pointing a role at either kind is config-only.
"""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.ai.governor import RateLimitGovernor, estimate_tokens

logger = logging.getLogger("cognideal.ai")

# Model families that take `reasoning_effort` and reject `temperature`.
# Anything else -- gpt-4o, gpt-4o-mini, gpt-4.1 -- is the reverse. `gpt-5-chat*`
# is the exception inside the family: a non-reasoning snapshot.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def is_reasoning_model(model: str) -> bool:
    name = model.lower()
    return name.startswith(_REASONING_MODEL_PREFIXES) and "-chat" not in name


# Roles, not sizes. docs/ai/README.md section 7 assigns them.
ROLE_PRIMARY = "primary"
ROLE_CHEAP = "cheap"
ROLE_CHALLENGER = "challenger"

_MAX_TOKENS_BY_TASK = {
    "extract": lambda: settings.ai_max_tokens_extract,
    "validate": lambda: settings.ai_max_tokens_validate,
    "detect": lambda: settings.ai_max_tokens_detect,
    "synthesize": lambda: settings.ai_max_tokens_synthesize,
    "brief": lambda: settings.ai_max_tokens_synthesize,
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
            max_concurrency=settings.ai_max_concurrency,
            tokens_per_minute=settings.ai_tokens_per_minute,
            requests_per_minute=settings.ai_requests_per_minute,
        )
    return _governor


def model_for(role: str) -> str:
    if role == ROLE_PRIMARY:
        return settings.ai_model_primary
    if role == ROLE_CHEAP:
        return settings.ai_model_cheap
    if role == ROLE_CHALLENGER:
        return settings.ai_model_challenger
    raise ValueError("unknown role %r" % role)


def max_tokens_for(task: str) -> int:
    getter = _MAX_TOKENS_BY_TASK.get(task)
    return getter() if getter else settings.ai_max_tokens_default


def _require_enabled() -> None:
    if not settings.ai_enabled:
        raise AIDisabled("ai_enabled is false")
    if not settings.openai_api_key:
        raise AIDisabled("OPENAI_API_KEY is not set")


def chat_model(
    *,
    role: str = ROLE_PRIMARY,
    task: str = "default",
    reasoning_effort: Optional[str] = None,
    model: Optional[str] = None,
    rate_limiter: Optional[BaseRateLimiter] = None,
):
    """Build a configured ``ChatOpenAI``.

    ``max_retries=0`` on purpose. LangChain's own retry would re-send without
    passing back through the governor, which is precisely the behaviour that
    turns one 429 into several -- retries are handled in :func:`_invoke`, where
    the rate limiter can see them.

    ``temperature=0`` on non-reasoning models: every task here is extraction,
    classification or entailment. None of them wants sampling variance, and
    stability across runs is a product requirement (docs/ai/README.md section
    10, the flicker metric). Reasoning models reject any temperature but the
    default and get ``reasoning_effort`` instead -- decided by the model, not
    by whether the caller passed an effort, so a reasoning model called without
    one still gets no ``temperature``.

    ``stream_usage=True``: OpenAI omits token usage from a streamed response
    unless asked, and the chat agent streams -- without it ``track_usage``
    would charge every chat turn as zero tokens.
    """
    _require_enabled()

    name = model or model_for(role)
    kwargs: Dict[str, Any] = {
        "model": name,
        "api_key": settings.openai_api_key,
        "temperature": 0,
        "max_tokens": max_tokens_for(task),
        "timeout": settings.ai_request_timeout_seconds,
        "max_retries": 0,
        "stream_usage": True,
    }
    if is_reasoning_model(name):
        del kwargs["temperature"]
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
    # Left unset for the `structured` path, which takes from the same bucket
    # itself -- attaching it there would charge every request twice.
    if rate_limiter is not None:
        kwargs["rate_limiter"] = rate_limiter
    return ChatOpenAI(**kwargs)


def _is_retryable(exc: BaseException) -> bool:
    """Whether one more attempt is worth making.

    Matched loosely and on purpose: the exception type depends on which
    langchain-openai version pip resolved, so keying on a concrete class would
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


_RETRY_AFTER = re.compile(r"try again in ((?:[0-9.]+(?:ms|h|m|s))+)", re.IGNORECASE)
_DURATION_PART = re.compile(r"([0-9.]+)(ms|h|m|s)")
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _duration_seconds(raw: Any) -> Optional[float]:
    """``"1.5"``, ``"20ms"``, ``"6m0s"`` -> seconds; None when unparseable.

    ``retry-after`` is plain seconds; OpenAI's ``x-ratelimit-reset-*`` headers
    and its 429 messages use the compound form.
    """
    text_value = str(raw).strip().lower()
    try:
        return float(text_value)
    except ValueError:
        pass
    parts = _DURATION_PART.findall(text_value)
    if not parts or "".join(n + u for n, u in parts) != text_value:
        return None
    try:
        return sum(float(n) * _DURATION_UNITS[u] for n, u in parts)
    except ValueError:
        return None


def _retry_after(exc: BaseException, attempt: int) -> float:
    """How long to wait, preferring the server's own answer.

    A 429 says when the window reopens -- "Please try again in 1.3s" or
    "in 20ms" in the message, ``retry-after`` / ``x-ratelimit-reset-tokens``
    in the headers -- and a flat one-second backoff ignores it, retries into
    the same exhausted window and burns the attempt.

    Capped: a server asking for several minutes is better handled by failing
    the stage than by holding a worker.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(key)
        seconds = _duration_seconds(raw) if raw else None
        if seconds is not None:
            return min(seconds + 0.25, 60.0)
    match = _RETRY_AFTER.search(str(exc))
    if match:
        seconds = _duration_seconds(match.group(1))
        if seconds is not None:
            return min(seconds + 0.5, 60.0)
    return 1.0 * attempt


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


#: How many stray array elements a salvage will drop before giving up. A
#: response with more than a couple is not "one bad token", it is a model
#: that lost the shape, and keeping whatever parsed out of it would be
#: guesswork presented as data.
_MAX_SALVAGED_DROPS = 3


def _strip_stray_elements(payload: Any) -> Tuple[Any, List[str]]:
    """Drop non-object elements from lists of objects, recursively.

    The defect this exists for, measured by task 11.9: the model emits a
    spurious scalar *inside* an array of objects -- a bare ``""`` between two
    valid risks, or a truncated ``'{""risk_type"'`` -- and the whole response
    is rejected, discarding two or three perfectly good objects with it. Three
    detect runs in ten failed this way.

    The rule is deliberately narrow. An element is dropped only when its
    siblings are objects and it is not, which is exactly the observed shape. A
    list of scalars is left alone, an object is never repaired field by field,
    and a missing required field still fails: this recovers a response the
    model nearly got right, and must not manufacture one it did not.
    """
    dropped: List[str] = []

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            return {k: walk(v, "%s.%s" % (path, k)) for k, v in node.items()}
        if isinstance(node, list):
            objects = [i for i in node if isinstance(i, dict)]
            # Only when the list is *meant* to hold objects: at least one does,
            # and at least one does not.
            if objects and len(objects) != len(node):
                for item in node:
                    if not isinstance(item, dict):
                        dropped.append("%s[%r]" % (path, _clip(item)))
                node = objects
            return [walk(i, "%s[%d]" % (path, n)) for n, i in enumerate(node)]
        return node

    return walk(payload, "$"), dropped


def _clip(value: Any, limit: int = 40) -> str:
    text_value = str(value)
    return text_value if len(text_value) <= limit else text_value[:limit] + "..."


def _salvage(schema: Any, raw_json: Optional[str], task: str) -> Tuple[Any, List[str]]:
    """Re-validate a rejected completion with stray array elements removed.

    Returns ``(None, [])`` when there is nothing to salvage, so the caller
    raises exactly as it did before. Only ever called on a response that has
    already failed, so it cannot change the outcome of a good one.
    """
    if not raw_json:
        return None, []
    try:
        payload = json.loads(raw_json)
    except ValueError:
        # Truncated or otherwise unparseable as JSON at all. Out of scope:
        # repairing that means guessing where the object ended.
        return None, []

    cleaned, dropped = _strip_stray_elements(payload)
    if not dropped or len(dropped) > _MAX_SALVAGED_DROPS:
        return None, dropped
    try:
        return schema.model_validate(cleaned), dropped
    except Exception as exc:  # noqa: BLE001 -- the salvage simply did not work
        logger.info("ai.salvage_rejected task=%s error=%s", task, _clip(exc, 160))
        return None, dropped


def _content_of(raw: Any) -> Optional[str]:
    """The response text, for a completion that arrived and failed validation."""
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return content
    # Some providers return content as a list of parts.
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ) or None
    return None


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
            delay = _retry_after(exc, attempt)
            logger.info(
                "ai.retry task=%s attempt=%d/%d sleeping=%.1fs reason=%s",
                run.task, attempt, attempts, delay, type(exc).__name__,
            )
            await asyncio.sleep(delay)

    run.latency_ms = int((time.monotonic() - started) * 1000)
    run.outcome = "error"
    run.error = "%s: %s" % (type(last_exc).__name__, last_exc)
    run.log()
    raise last_exc  # type: ignore[misc]


def _render(messages: Sequence[Tuple[str, str]]) -> str:
    return "\n".join(content for _, content in messages)


@asynccontextmanager
async def track_usage(
    budget: Optional[RunBudget] = None,
) -> AsyncIterator[Any]:
    """Count every model call made inside this block -- task 3.8.

    ``_usage(raw)`` reads only the response *this* module holds, which is
    enough while every call goes through :func:`structured`. It stops being
    enough the moment a LangGraph node invokes a runnable directly or a ReAct
    agent loops: the agent makes five calls and we see one, so ``RunBudget``
    under-counts exactly where a runaway is likeliest.

    ``get_usage_metadata_callback`` registers a context-scoped handler, so it
    captures anything LangChain runs inside the block without that code having
    to know about it. On exit the aggregate is charged to the budget once.

    Use it to wrap a *run* -- a pipeline, an agent turn -- not an individual
    call, which :func:`structured` already accounts for::

        async with track_usage(budget) as usage:
            await agent.ainvoke(...)
    """
    with get_usage_metadata_callback() as callback:
        try:
            yield callback
        finally:
            if budget is not None:
                budget.charge(usage_total(callback))


def usage_total(callback: Any) -> int:
    """Total tokens a usage callback observed, across every model it saw."""
    total = 0
    for usage in (getattr(callback, "usage_metadata", None) or {}).values():
        if isinstance(usage, dict):
            total += int(usage.get("total_tokens") or 0)
        else:
            total += int(getattr(usage, "total_tokens", 0) or 0)
    return total


class GovernorRateLimiter(BaseRateLimiter):
    """The governor's request ceiling, in the shape LangChain understands -- task 3.9.

    For calls that never pass through :func:`structured`: a runnable invoked
    inside a graph node, or an agent's own loop. ``ChatOpenAI(rate_limiter=...)``
    is the only hook that reaches those.

    **Requests only.** ``BaseRateLimiter.acquire(*, blocking)`` takes no
    argument describing the request, so a token-aware limiter cannot be
    expressed through this interface at all -- which is also why LangChain's
    own ``InMemoryRateLimiter`` is time-based and says so. Tokens per minute
    stays in :func:`structured`, the only place an estimate exists. Concurrency
    stays there too: a limiter gates entry and is never told the call finished,
    so it has nothing to release.

    Attach it with :func:`agent_model`, never to a model used by
    :func:`structured` -- that path already takes from the same bucket and
    would be charged twice.
    """

    def acquire(self, *, blocking: bool = True) -> bool:
        if not blocking:
            return governor().try_take_request()
        raise NotImplementedError(
            "synchronous acquire would block the event loop; use aacquire"
        )

    async def aacquire(self, *, blocking: bool = True) -> bool:
        if not blocking:
            return governor().try_take_request()
        await governor().take_request()
        return True


def agent_model(
    *,
    role: str = ROLE_PRIMARY,
    task: str = "chat",
    reasoning_effort: Optional[str] = None,
):
    """A model for code that does not call :func:`structured`.

    Same configuration, plus the request limiter -- so a graph node or an agent
    loop is paced even though it never reaches the governor directly.
    """
    return chat_model(
        role=role,
        task=task,
        reasoning_effort=reasoning_effort,
        rate_limiter=GovernorRateLimiter(),
    )


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
    receives ``None`` would write a claim with no content. Under strict
    constrained decoding the shape is guaranteed, so what remains is a refusal,
    a response cut off by ``max_tokens``, or a non-constraining fallback
    method. Either way the claim must not land.
    """
    llm = chat_model(role=role, task=task, reasoning_effort=reasoning_effort, model=model)
    structured_kwargs: Dict[str, Any] = {
        "method": settings.structured_output_method,
        "include_raw": True,
    }
    if settings.structured_output_method in ("json_schema", "function_calling"):
        # Constrained decoding. `json_mode` has no strict variant.
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
        # The completion came back and failed validation. Rare under strict
        # mode; what it still catches is a stray element inside an array of
        # objects from a non-constraining fallback -- see
        # `_strip_stray_elements`.
        salvaged, dropped = _salvage(schema, _content_of(raw), task)
        if salvaged is not None:
            run.outcome = "ok_salvaged"
            run.extra["salvaged_drops"] = dropped
            logger.warning(
                "ai.salvaged task=%s source=parsing_error dropped=%d %s",
                task, len(dropped), dropped,
            )
            run.log()
            return salvaged, run
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

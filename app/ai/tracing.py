"""Langfuse tracing for the chat assistant.

One trace per chat turn. The ReAct agent's model calls, tool calls and the
LangGraph nodes between them nest under it on their own, because Langfuse
attaches as a LangChain callback and LangChain already reports that tree --
nothing here walks the graph or instruments a tool.

**Additive by construction.** :func:`callbacks` returns ``[]`` whenever tracing
is off, a key is missing, or the SDK failed to start, and
:func:`agent_config` then hands back the caller's config unchanged. A turn with
tracing disabled takes exactly the path it took before this module existed, so
a broken Langfuse account cannot break a conversation.

**Not an accounting source.** ``client.py`` owns the per-call ``AIRun`` and the
structured log line that ``docs/schema/README.md`` section 10 leaves as the
only run record, and ``chat_messages.token_usage`` is what the product reads
back. Langfuse is for inspecting a turn -- which tools ran, what the model saw,
where the latency went -- not for counting one. Deliberately no second number
anybody could cite.

**Why the chat assistant only.** Every AI task could be traced from
``client.py`` in one place, and that is the obvious extension. Chat is where a
trace pays for itself first: it is the one task with tool use, multiple model
steps per turn, and a user waiting, so "why did it answer that" is a question
somebody actually asks. The pipeline stages are single-shot and already have
their stage records.

Session grouping is by ``chat_sessions.id`` -- the same key
``checkpointer.thread_id`` uses -- so a Langfuse session and a checkpoint
thread are the same conversation under the same id.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("cognideal.ai.tracing")

#: Resolved once. ``None`` means unavailable, and that is a terminal answer --
#: a missing key or a failed import will not start working later in the
#: process, and retrying per turn would log the same failure on every message.
_client = None
_resolved = False


def enabled() -> bool:
    """True only with tracing on *and* both keys present.

    Both halves matter: ``LANGFUSE_ENABLED=true`` with no keys is a
    misconfiguration, and the honest response is to run untraced rather than
    to fail a turn over telemetry.
    """
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


def client():
    """The process-wide Langfuse client, or ``None`` if unavailable.

    Constructed on first use rather than at import, so importing this module
    never opens a connection and ``app.main`` stays as free of AI imports as
    ``app/ai/__init__.py`` claims.

    The SDK reads its own environment variables, but the keys are passed
    explicitly: ``app/core/config.py`` is the one place this project's
    configuration is declared, and a setting that worked only when exported
    under a second name would be a trap.
    """
    global _client, _resolved
    if _resolved:
        return _client
    _resolved = True
    if not enabled():
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            environment=settings.langfuse_environment,
            sample_rate=settings.langfuse_sample_rate,
            # Chat is a streaming endpoint and the generator may be cancelled
            # mid-turn, so a trace that waits for a full batch can be lost
            # with the process. Short interval, small batch: this is low
            # volume by nature -- one trace per question a human typed.
            flush_at=1,
            flush_interval=1.0,
        )
        logger.info(
            "langfuse.enabled host=%s environment=%s sample_rate=%s",
            settings.langfuse_host,
            settings.langfuse_environment,
            settings.langfuse_sample_rate,
        )
    except Exception:  # noqa: BLE001 -- telemetry must never fail a turn
        _client = None
        logger.exception("langfuse.init_failed -- continuing untraced")
    return _client


def callbacks() -> List[Any]:
    """LangChain callback handlers to attach, or ``[]`` when tracing is off.

    The handler takes no credentials in the v3 SDK: it finds the client
    :func:`client` constructed above, which is why that has to be built first
    and why its return value is checked before the handler is.
    """
    if client() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        # A handler per turn rather than a module-level one. It keeps
        # per-run bookkeeping in dicts keyed by run id, and chat turns are
        # concurrent by nature; a fresh handler keeps one turn's state from
        # outliving it. The object is cheap -- the client underneath is shared.
        #
        # `update_trace` lifts the agent run's input and output onto the trace
        # itself, so a trace reads as the question and the answer rather than
        # as an unnamed root with the real content one level down.
        return [CallbackHandler(update_trace=True)]
    except Exception:  # noqa: BLE001
        logger.exception("langfuse.handler_failed -- continuing untraced")
        return []


def agent_config(
    config: Dict[str, Any],
    *,
    session_id: Any,
    deal_id: Any = None,
    scope: Optional[str] = None,
) -> Dict[str, Any]:
    """The agent's ``config`` with tracing merged in, or it unchanged.

    Returns a new dict; the caller's ``thread`` config is not mutated, so the
    checkpointer's ``configurable`` is still the only thing deciding which
    thread resumes.

    The ``langfuse_*`` metadata keys are the SDK's documented contract for
    trace attributes -- they are stripped from what it records rather than
    stored as metadata. The rest (``deal_id``, ``scope``) are ours and do show
    up, which is the point: a trace should say which deal it was about.
    """
    handlers = callbacks()
    if not handlers:
        return config
    merged = dict(config)
    merged["callbacks"] = list(merged.get("callbacks") or []) + handlers
    metadata = dict(merged.get("metadata") or {})
    metadata.update(
        {
            "langfuse_session_id": str(session_id),
            "langfuse_tags": [tag for tag in ["chat", scope] if tag],
            "deal_id": str(deal_id) if deal_id else None,
            "scope": scope,
        }
    )
    merged["metadata"] = metadata
    # Names the trace. Without it every chat trace is called `LangGraph`, and
    # a list of identical names is a list you cannot scan.
    merged["run_name"] = "chat_turn"
    return merged


def _flush_blocking() -> None:
    """The executor target for :func:`flush_soon`. Never raises.

    A named function rather than passing ``client.flush`` straight to the
    executor, for the ``except``: nobody awaits the Future this runs on, so an
    exception escaping here would sit on it unretrieved and resurface as an
    "exception was never retrieved" warning whenever it was collected --
    telemetry noise in a log that should only carry the app's own failures.
    """
    existing = _client
    if existing is None:
        return
    try:
        existing.flush()
    except Exception:  # noqa: BLE001 -- telemetry must never fail a turn
        logger.exception("langfuse.flush_failed")


def flush_soon() -> None:
    """Push buffered traces, off the event loop and without waiting.

    The SDK buffers spans and exports them on a timer (``flush_interval``
    above), so a trace already recorded is still in memory for up to a second.
    This closes that window at the end of a turn -- most usefully for a
    cancelled or failed one, which is the trace somebody actually goes looking
    for.

    Not awaited, and not named ``flush``:

    *   ``Langfuse.flush()`` blocks on an HTTP round trip and chat turns run on
        the event loop, so calling it inline would stall every other request
        on this worker for its duration.
    *   Not awaiting is also what makes this safe in a ``finally`` reached by
        ``CancelledError``, where an ``await`` would re-raise at once and lose
        the rest of the cleanup.
    *   ``flush`` alone would read as ``db.flush`` to anyone who works in this
        codebase, where that is what the word means everywhere else.
    """
    if _client is None:
        return
    try:
        asyncio.get_running_loop().run_in_executor(None, _flush_blocking)
    except RuntimeError:
        # No running loop (a sync caller, or a test). Flush inline -- there is
        # no event loop to protect.
        _flush_blocking()


def shutdown() -> None:
    """Flush and stop the exporter, for application shutdown."""
    existing = _client
    if existing is None:
        return
    try:
        existing.shutdown()
    except Exception:  # noqa: BLE001
        logger.exception("langfuse.shutdown_failed")

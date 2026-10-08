"""LangGraph's conversation memory, in Postgres.

Replaces the hand-built history in ``chat.py``. The agent used to be stateless
per turn: every turn re-read ``chat_messages`` and rebuilt a list of
``HumanMessage``/``AIMessage``. That worked, but it could only replay the
*prose* -- the tool calls and tool results of earlier turns were dropped, so a
tool-using agent could see that it had answered and not what it had looked up.
A checkpointer stores the whole message graph, tool traffic included.

**``chat_messages`` is still authoritative and still written.** This is a
second store, deliberately, and the division is:

    chat_messages   the product's record -- what `GET /chat/sessions/{id}/
                    messages` serves, what `claim_evidence.claim_id` points at
                    for `claim_type='chat_message'`, and what 11.6's reaper
                    finalizes. Durable, migrated, ours.
    checkpoints     the agent's working memory -- the message graph LangGraph
                    needs to resume a thread. Disposable: delete it and the
                    conversation is still intact in chat_messages, the agent
                    just forgets the tool traffic.

That asymmetry is the whole reason this is safe to add. ``docs/schema/
README.md`` warns that two copies of the same thing can disagree, which is why
``documents.raw_text`` was dropped in 0008 -- but here only one copy is load
bearing. If they ever disagree, ``chat_messages`` wins by definition.

**The tables are not Alembic's.** ``AsyncPostgresSaver.setup()`` creates and
migrates ``checkpoints``, ``checkpoint_writes``, ``checkpoint_blobs`` and
``checkpoint_migrations`` itself, so they are outside the migration chain and
will not appear in an autogenerate diff. ``models/__init__.py``'s registry test
only walks ``Base.metadata``, so it is unaffected. The consequence to know: a
schema rebuild from empty needs ``setup()`` to have run. ``app.main`` runs it at
startup so no chat turn waits on DDL, and :func:`saver` runs it again if that
did not finish.

**A second Postgres driver.** The app talks to Postgres over asyncpg via
SQLAlchemy; ``langgraph-checkpoint-postgres`` requires psycopg 3 and will not
accept an asyncpg connection. Both now ship. ``settings.psycopg_database_url``
converts the asyncpg-form URL back to libpq's, which psycopg can parse.
"""

import asyncio
import logging
from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    trim_messages,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

logger = logging.getLogger("cognideal.ai.checkpointer")

_pool: Optional[AsyncConnectionPool] = None
_saver: Optional[AsyncPostgresSaver] = None
# Created lazily for the same event-loop reason as the pool.
_lock: Optional[asyncio.Lock] = None

#: What the model is shown in place of a tool result that never arrived.
_INTERRUPTED_RESULT = "Tool call interrupted before it returned; no result is available."


async def saver() -> AsyncPostgresSaver:
    """The process-wide saver, created and migrated on first use.

    Lazy for the same reason :func:`client.governor` is: the pool binds to the
    running event loop, so building it at import time couples it to whichever
    loop happened to exist then.

    ``from_conn_string`` is deliberately not used -- it is an async context
    manager that closes the connection on exit, which is right for a script and
    wrong for a long-lived API process that must not reconnect per turn.

    ``autocommit=True`` and ``row_factory=dict_row`` are required by
    ``AsyncPostgresSaver``, not preferences: without autocommit its migration
    statements sit in an open transaction, and it indexes result rows by name.
    ``prepare_threshold=0`` is required by Neon's ``-pooler`` host: PgBouncer in
    transaction mode hands each statement to whichever server connection is
    free, so a server-side prepared statement made on one is missing on the
    next.

    **Published only after ``setup()`` succeeds.** It used to be assigned
    first, so a setup cut short -- a client disconnect cancelled the first chat
    turn mid-migration -- left a saver every later turn reused against tables
    still missing ``checkpoint_writes.task_path``. Now a failed or cancelled
    setup closes its pool and the next call starts over; the lock stops two
    concurrent first calls from each building a pool.
    """
    global _pool, _saver, _lock
    if _saver is not None:
        return _saver
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        if _saver is not None:
            return _saver

        size = settings.chat_checkpoint_pool_size
        pool = AsyncConnectionPool(
            conninfo=settings.psycopg_database_url,
            min_size=min(1, size),
            max_size=size,
            open=False,
            # Neon closes idle connections; check one before handing it out.
            check=AsyncConnectionPool.check_connection,
            kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        )
        await pool.open()
        try:
            candidate = AsyncPostgresSaver(pool)
            # Idempotent: it keeps its own `checkpoint_migrations` table and
            # resumes from the last applied step, so a half-finished setup is
            # completed rather than repeated.
            await candidate.setup()
        except BaseException:
            await pool.close()
            raise
        _pool, _saver = pool, candidate
        logger.info("checkpointer.ready pool_max=%d", size)
        return _saver


async def close() -> None:
    """Release the pool. For tests and a clean shutdown."""
    global _pool, _saver, _lock
    if _pool is not None:
        await _pool.close()
    _pool, _saver, _lock = None, None, None


def thread_id(session_id) -> str:
    """One checkpoint thread per chat session.

    The session id rather than a derived key, so a checkpoint row is traceable
    back to a `chat_sessions` row by eye in psql.
    """
    return str(session_id)


async def forget(session_id) -> None:
    """Drop a session's checkpoint thread.

    Called when a chat session is deleted. Without this the checkpoint rows
    outlive the session that explains them -- the same orphan shape
    ``claim_evidence`` has, and the reason `services/claims.py` exists. Failure
    is logged rather than raised: the session delete is the user's intent, and
    leaving stale agent memory behind is not a reason to fail it.
    """
    try:
        store = await saver()
        await store.adelete_thread(thread_id(session_id))
    except Exception:
        logger.exception("checkpointer.forget_failed session=%s", session_id)


def trim_hook(limit: Optional[int] = None):
    """A ``pre_model_hook`` that caps what the model sees, not what is stored.

    Returns ``llm_input_messages`` rather than ``messages`` -- the key that
    feeds the model *without* updating graph state. Writing ``messages`` here
    would make trimming destructive: the checkpointer would forget the dropped
    turns permanently, and the next turn would trim an already-trimmed history.
    This way the thread keeps everything and each turn re-trims from the full
    record.

    Counted in messages, not tokens (``token_counter=len``). A token count
    would be more precise and needs the tokenizer; message count is predictable
    and the per-call ceiling is already enforced by ``RunBudget`` and
    ``ai_max_tokens_chat``, so this limit only has to stop unbounded growth.

    ``include_system=True`` and ``start_on="human"`` matter: the system prompt
    must survive trimming, and a window that begins on a tool result -- an
    orphaned result whose tool call was trimmed away -- is a shape providers
    reject.

    Two repairs on top of the trim, both on the model's view only:

    *   **The current question always survives.** One question that takes more
        tool calls than the window holds pushes its own human message out, and
        ``start_on="human"`` then trims to nothing -- the model would answer
        with only the system prompt. The window instead falls back to
        everything since the latest human message.
    *   **Unanswered tool calls get a stub result.** A turn cancelled or failed
        between the model asking for a tool and the result being saved leaves
        an ``AIMessage`` with ``tool_calls`` and no ``ToolMessage``. OpenAI
        rejects every later request on that thread, so the session would be
        dead for good; a stub result keeps it usable.
    """
    window = settings.chat_history_messages if limit is None else limit

    def hook(state):
        messages = state["messages"]
        trimmed = trim_messages(
            messages,
            max_tokens=window,
            token_counter=len,
            strategy="last",
            include_system=True,
            start_on="human",
            allow_partial=False,
        )
        if not any(isinstance(m, HumanMessage) for m in trimmed):
            trimmed = _since_last_human(messages)
        return {"llm_input_messages": _answer_dangling_tool_calls(trimmed)}

    return hook


def _since_last_human(messages: List[BaseMessage]) -> List[BaseMessage]:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return list(messages[index:])
    return list(messages)


def _answer_dangling_tool_calls(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Insert a stub ``ToolMessage`` after any tool call that has no result."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    repaired: List[BaseMessage] = []
    for message in messages:
        repaired.append(message)
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if call.get("id") and call["id"] not in answered:
                    repaired.append(ToolMessage(
                        content=_INTERRUPTED_RESULT,
                        tool_call_id=call["id"],
                        name=call.get("name"),
                    ))
    return repaired

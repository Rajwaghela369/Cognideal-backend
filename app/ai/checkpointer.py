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
schema rebuild from empty needs ``setup()`` to have run, which is why it is
called on first use rather than at import.

**A second Postgres driver.** The app talks to Postgres over asyncpg via
SQLAlchemy; ``langgraph-checkpoint-postgres`` requires psycopg 3 and will not
accept an asyncpg connection. Both now ship. :func:`_conninfo` converts the
asyncpg-form URL back to libpq's, which psycopg can parse.
"""

import logging
from typing import Optional

from langchain_core.messages import trim_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

logger = logging.getLogger("cognideal.ai.checkpointer")

_pool: Optional[AsyncConnectionPool] = None
_saver: Optional[AsyncPostgresSaver] = None


def _conninfo() -> str:
    """``postgresql+asyncpg://...?ssl=`` -> ``postgresql://...?sslmode=``.

    SQLAlchemy's URL carries the driver in the scheme and asyncpg's own SSL
    parameter name; psycopg rejects both. One setting, two consumers, rather
    than a second URL in config that could drift from the first.
    """
    return settings.psycopg_database_url


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
    """
    global _pool, _saver
    if _saver is not None:
        return _saver

    _pool = AsyncConnectionPool(
        conninfo=_conninfo(),
        max_size=settings.chat_checkpoint_pool_size,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await _pool.open()
    _saver = AsyncPostgresSaver(_pool)
    # Idempotent: it keeps its own `checkpoint_migrations` table and skips what
    # has already been applied, so calling it on every cold start is cheap and
    # means a fresh database needs no extra deploy step.
    await _saver.setup()
    logger.info("checkpointer.ready pool_max=%d", settings.chat_checkpoint_pool_size)
    return _saver


async def close() -> None:
    """Release the pool. For tests and a clean shutdown."""
    global _pool, _saver
    if _pool is not None:
        await _pool.close()
    _pool, _saver = None, None


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
    """
    window = settings.chat_history_messages if limit is None else limit

    def hook(state):
        messages = state["messages"] if isinstance(state, dict) else state.messages
        return {
            "llm_input_messages": trim_messages(
                messages,
                max_tokens=window,
                token_counter=len,
                strategy="last",
                include_system=True,
                start_on="human",
                allow_partial=False,
            )
        }

    return hook

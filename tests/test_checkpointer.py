"""Conversation memory: the trim hook, handle seeding, and the real saver.

The chat agent no longer rebuilds history by hand. These cover the three things
that change as a result -- what the model is sent, what the thread keeps, and
whether evidence handles still mean one thing each -- plus one test against
Postgres, because ``conftest`` swaps in the in-memory saver everywhere else and
something has to exercise the driver that actually ships.
"""

import uuid

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import select

from app.ai import checkpointer
from app.ai.tools import ToolRegistry
from app.ai import chat as chat_mod
from app.models import ChatMessage, ChatSession
from app.models.enums import ChatRole, ChatScope, MessageStatus


# --------------------------------------------------------------------------
# The trim hook
# --------------------------------------------------------------------------


def _conversation(turns):
    """`turns` question/answer pairs, oldest first."""
    messages = [SystemMessage(content="you are cognideal")]
    for i in range(turns):
        messages.append(HumanMessage(content="q%d" % i))
        messages.append(AIMessage(content="a%d" % i))
    return messages


def test_trim_caps_what_the_model_sees():
    hook = checkpointer.trim_hook(limit=5)
    out = hook({"messages": _conversation(10)})
    assert len(out["llm_input_messages"]) <= 5


def test_trim_returns_llm_input_messages_and_never_messages():
    """The distinction that makes trimming non-destructive.

    Returning `messages` would UPDATE graph state, so the checkpointer would
    forget the dropped turns permanently and each turn would re-trim an
    already-trimmed history. `llm_input_messages` feeds the model only.
    """
    out = checkpointer.trim_hook(limit=4)(({"messages": _conversation(10)}))
    assert "llm_input_messages" in out
    assert "messages" not in out


def test_trim_keeps_the_system_prompt():
    """Scope enforcement lives in the system prompt -- it must survive."""
    out = checkpointer.trim_hook(limit=4)({"messages": _conversation(10)})
    kept = out["llm_input_messages"]
    assert any(isinstance(m, SystemMessage) for m in kept)


def test_trim_does_not_start_the_window_on_a_tool_result():
    """An orphaned ToolMessage -- one whose tool call was trimmed away -- is a
    shape providers reject, so the window must not open on one."""
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[
            {"name": "list_risks", "args": {}, "id": "call_1"}
        ]),
        ToolMessage(content="risk", tool_call_id="call_1"),
        AIMessage(content="a"),
        HumanMessage(content="q2"),
        AIMessage(content="a2"),
    ]
    kept = checkpointer.trim_hook(limit=3)({"messages": messages})["llm_input_messages"]
    non_system = [m for m in kept if not isinstance(m, SystemMessage)]
    if non_system:
        assert not isinstance(non_system[0], ToolMessage)


def test_a_short_conversation_is_untouched():
    messages = _conversation(2)
    kept = checkpointer.trim_hook(limit=50)({"messages": messages})["llm_input_messages"]
    assert len(kept) == len(messages)


# --------------------------------------------------------------------------
# Evidence handles across turns -- the bug the checkpointer would have made worse
# --------------------------------------------------------------------------


def test_handles_restart_without_an_offset():
    """Documents the old behaviour, so the fix cannot be removed silently."""
    a = ToolRegistry(None, uuid.uuid4())
    b = ToolRegistry(None, uuid.uuid4())
    assert a._handle({"snippet": "x"}) == "e1"
    assert b._handle({"snippet": "y"}) == "e1"


def test_an_offset_continues_the_numbering():
    """Turn 2 must not reissue a handle turn 1 already used.

    With history in play the thread carries turn 1's tool results, so two live
    `e1`s would mean two different citations in one context -- and
    `_attach_citations` resolves whatever the model wrote against *this* turn's
    registry, attaching a claim to the wrong evidence.
    """
    later = ToolRegistry(None, uuid.uuid4(), handle_offset=3)
    assert later._handle({"snippet": "x"}) == "e4"
    assert later._handle({"snippet": "y"}) == "e5"
    assert "e1" not in later.evidence


@pytest.mark.asyncio
async def test_handles_issued_reads_back_what_was_recorded(db, deal_id):
    session = ChatSession(scope=ChatScope.DEAL, deal_id=deal_id)
    db.add(session)
    await db.commit()

    assert await chat_mod._handles_issued(db, session.id) == 0

    db.add(ChatMessage(
        session_id=session.id,
        role=ChatRole.ASSISTANT,
        content="answer [e1] and [e2]",
        status=MessageStatus.COMPLETE,
        token_usage={"citation_handles": [
            {"handle": "e1", "evidence_id": str(uuid.uuid4())},
            {"handle": "e2", "evidence_id": str(uuid.uuid4())},
        ]},
    ))
    await db.commit()

    assert await chat_mod._handles_issued(db, session.id) == 2

    await db.delete(session)
    await db.commit()


@pytest.mark.asyncio
async def test_handles_issued_ignores_turns_that_cited_nothing(db, deal_id):
    """A turn with no citations records an empty list, not a gap."""
    session = ChatSession(scope=ChatScope.DEAL, deal_id=deal_id)
    db.add(session)
    await db.flush()   # session.id is None until this runs
    db.add(ChatMessage(
        session_id=session.id,
        role=ChatRole.ASSISTANT,
        content="just advice, no citations",
        status=MessageStatus.COMPLETE,
        token_usage={"total_tokens": 10, "citation_handles": []},
    ))
    await db.commit()

    assert await chat_mod._handles_issued(db, session.id) == 0

    await db.delete(session)
    await db.commit()


# --------------------------------------------------------------------------
# The thread itself
# --------------------------------------------------------------------------


def test_thread_id_is_the_session_id():
    """Traceable back to a chat_sessions row by eye in psql."""
    session_id = uuid.uuid4()
    assert checkpointer.thread_id(session_id) == str(session_id)


def test_conninfo_drops_the_sqlalchemy_dialect():
    """psycopg cannot parse `+asyncpg`, and one setting must serve both."""
    info = checkpointer.settings.psycopg_database_url
    assert info.startswith("postgresql://")
    assert "+asyncpg" not in info


@pytest.mark.asyncio
async def test_the_in_memory_saver_is_what_tests_get(db):
    """Guards conftest's swap: a chat test must not create checkpoint tables."""
    saver = await checkpointer.saver()
    assert type(saver).__name__ == "InMemorySaver"


# --------------------------------------------------------------------------
# Against real Postgres
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_postgres_saver_round_trips_a_thread(monkeypatch):
    """The driver that actually ships, exercised once.

    `conftest` swaps in the in-memory saver for every other test, which is
    right -- a unit test should not create `checkpoints*` tables as a side
    effect. But that means nothing else here touches psycopg, and the asyncpg/
    psycopg split is exactly the kind of thing that works in a unit test and
    fails in the process.
    """
    monkeypatch.undo()  # drop conftest's in-memory swap for this test
    import importlib

    importlib.reload(checkpointer)
    try:
        saver = await checkpointer.saver()
        assert type(saver).__name__ == "AsyncPostgresSaver"

        # `checkpoint_ns` is required by aput, not optional -- LangGraph pops it
        # unguarded. An empty namespace is what the agent uses at the top level.
        thread = {"configurable": {
            "thread_id": "test-%s" % uuid.uuid4(), "checkpoint_ns": "",
        }}
        assert await saver.aget_tuple(thread) is None

        await saver.aput(
            thread,
            {"v": 1, "id": str(uuid.uuid4()), "ts": "2026-10-05T00:00:00+00:00",
             "channel_values": {"messages": []}, "channel_versions": {},
             "versions_seen": {}},
            {"source": "input", "step": 0, "parents": {}},
            {},
        )
        assert await saver.aget_tuple(thread) is not None

        await saver.adelete_thread(thread["configurable"]["thread_id"])
        assert await saver.aget_tuple(thread) is None
    finally:
        await checkpointer.close()


# --------------------------------------------------------------------------
# Handle extraction -- the glyph the model happens to use
# --------------------------------------------------------------------------


@pytest.mark.parametrize("answer, expected", [
    ("SecureFlow is the deal [e1].", ["e1"]),
    # Observed in a real two-turn run: fullwidth CJK brackets inside a markdown
    # table. An ASCII-only pattern dropped every citation in that answer and
    # wrote the claim with no evidence, which still reads as cited.
    ("Priya Raman 【e4】【e5】", ["e4", "e5"]),
    ("Marcus Webb ［e9］", ["e9"]),
    ("mixed [e1] and 【e2】", ["e1", "e2"]),
    ("no handles here", []),
    # The handle itself stays strict: brackets are not a licence for anything.
    ("[event] and [e] and [3]", []),
])
def test_handles_are_found_whatever_bracket_the_model_used(answer, expected):
    assert chat_mod._HANDLE.findall(answer) == expected

"""Phase 9 conversational agent with durable turns and grounded citations."""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import AsyncIterator

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.prebuilt import create_react_agent
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import checkpointer, client
from app.ai.prompts import get as get_prompt
from app.ai.schemas import ChatTitleOut
from app.ai.tools import ToolRegistry
from app.db.session import SessionLocal
from app.models import ChatMessage, ChatSession
from app.models.enums import (
    ChatRole,
    ChatScope,
    ClaimType,
    MessageStatus,
)
from app.services import claims, gate0

logger = logging.getLogger("dealpilot.ai.chat")
# Handles as the model writes them. ASCII brackets are what the prompt asks
# for, but the bracket is the model's choice of glyph and it does not always
# choose ours: a real turn emitted `【e4】【e5】【e6】` -- fullwidth CJK brackets
# -- inside a markdown table, and an ASCII-only pattern matched none of them, so
# every citation in that answer was silently dropped and the claim was written
# with no evidence at all. Failing open like that is the worst available
# outcome: the answer still looks cited to a reader.
#
# Accepts the bracket families models actually use. The handle itself stays
# strict (`e` plus digits), so this widens which punctuation is tolerated and
# not what counts as a handle.
_HANDLE = re.compile(r"[\[【［](e\d+)[\]】］]")
#: The number inside a handle, for seeding the next turn past it.
_HANDLE_NUMBER = re.compile(r"^e(\d+)$")


def _text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


async def stream_turn(
    db: AsyncSession, session: ChatSession, question: str
) -> AsyncIterator[str]:
    """Persist first, stream deltas, then ground only handles actually cited."""
    session_id = session.id
    user = ChatMessage(
        session_id=session.id,
        role=ChatRole.USER,
        content=question.strip(),
        status=MessageStatus.COMPLETE,
    )
    assistant = ChatMessage(
        session_id=session.id,
        role=ChatRole.ASSISTANT,
        content="",
        status=MessageStatus.STREAMING,
        model=client.model_for(client.ROLE_PRIMARY),
    )
    db.add_all([user, assistant])
    session.last_message_at = func.now()
    await db.commit()
    await db.refresh(assistant)
    assistant_id = assistant.id

    # History is the checkpointer's job now. Only the new question is sent; the
    # thread's earlier messages -- including the tool calls and tool results the
    # old hand-built history could not represent -- come from the saver, keyed
    # on this session. `ai/checkpointer.py` has the chat_messages-vs-checkpoints
    # division of labour.
    store = await checkpointer.saver()
    thread = {"configurable": {"thread_id": checkpointer.thread_id(session_id)}}

    registry = ToolRegistry(
        db,
        session.deal_id if session.scope == ChatScope.DEAL else None,
        # Evidence handles must not restart at e1 each turn. They used to, and
        # with history in play that is a correctness bug rather than a cosmetic
        # one: the thread now carries turn 1's tool results verbatim, so `e1`
        # would appear twice in one context meaning two different citations, and
        # `_attach_citations` would resolve a handle the model copied from
        # earlier to whatever this turn's registry put at that number. Seeded
        # past every handle the session has already issued.
        handle_offset=await _handles_issued(db, session_id),
    )
    scope_note = (
        "This conversation is locked to deal %s. Never ask for or use another deal id."
        % session.deal_id
        if session.scope == ChatScope.DEAL
        else "This is a global pipeline conversation."
    )
    prompt = (
        "You are DealPilot, an evidence-grounded sales copilot. Use the read-only "
        "tools for factual claims. Tool results contain evidence handles such as e3. "
        "Cite every factual assertion inline as [e3], using only handles returned by "
        "tools. Clearly label advice or inference as such. Never claim a write occurred. "
        + scope_note
    )
    agent = create_react_agent(
        client.agent_model(task="chat", reasoning_effort="medium"),
        registry.langchain_tools(),
        prompt=prompt,
        checkpointer=store,
        pre_model_hook=checkpointer.trim_hook(),
    )
    started = time.monotonic()
    answer = ""
    last_checkpoint = 0
    budget = client.RunBudget()

    try:
        async with client.track_usage(budget) as usage:
            async for event in agent.astream(
                {"messages": [HumanMessage(content=question.strip())]},
                config=thread,
                stream_mode="messages",
            ):
                message, metadata = event
                if not isinstance(message, AIMessageChunk):
                    continue
                if metadata.get("langgraph_node") not in ("agent", "model"):
                    continue
                delta = _text(message)
                if not delta:
                    continue
                answer += delta
                yield json.dumps({"type": "delta", "content": delta})
                if len(answer) - last_checkpoint >= 500:
                    assistant.content = answer
                    await db.commit()
                    last_checkpoint = len(answer)

        assistant.content = answer
        assistant.status = MessageStatus.COMPLETE
        assistant.latency_ms = int((time.monotonic() - started) * 1000)
        assistant.token_usage = {
            "total_tokens": client.usage_total(usage),
            "citation_handles": await _attach_citations(
                db, session, assistant, answer, registry
            ),
        }
        await db.commit()
        yield json.dumps({"type": "done", "message_id": str(assistant.id)})
        if session.title is None:
            asyncio.create_task(_title_session(session.id, question, answer))
    except Exception as exc:
        await db.rollback()
        row = await db.get(ChatMessage, assistant_id)
        if row is not None:
            row.content = answer
            row.status = MessageStatus.ERROR
            row.latency_ms = int((time.monotonic() - started) * 1000)
            await db.commit()
        logger.exception("chat.turn_failed session=%s", session_id)
        yield json.dumps({"type": "error", "detail": str(exc)})


async def _handles_issued(db, session_id) -> int:
    """The highest evidence-handle number this session has already used.

    Read back from what was recorded rather than counted separately, so there
    is no second source of truth to drift: ``_attach_citations`` writes the
    handles it attached into ``token_usage['citation_handles']``, and this is
    the same list read in reverse.

    Returns 0 for a first turn, which makes the first handle ``e1`` exactly as
    before -- the numbering only changes from turn 2 onward, where it had to.
    """
    rows = (await db.scalars(
        select(ChatMessage.token_usage).where(
            ChatMessage.session_id == session_id,
            ChatMessage.role == ChatRole.ASSISTANT,
        )
    )).all()
    highest = 0
    for usage in rows:
        for entry in (usage or {}).get("citation_handles") or []:
            match = _HANDLE_NUMBER.match(entry.get("handle", ""))
            if match:
                highest = max(highest, int(match.group(1)))
    return highest


async def _attach_citations(db, session, message, answer, registry):
    handles = list(dict.fromkeys(_HANDLE.findall(answer)))
    citations = [registry.evidence[h] for h in handles if h in registry.evidence]
    if not citations:
        return []
    check = await gate0.check_claim(db, answer, citations)
    attached = []
    for handle, citation, link in zip(
        [h for h in handles if h in registry.evidence], citations, check.links
    ):
        evidence = await claims.attach_evidence(
            db,
            claim_type=ClaimType.CHAT_MESSAGE,
            claim_id=message.id,
            deal_id=uuid.UUID(citation["deal_id"]),
            source_kind=citation["source_kind"],
            snippet=citation["snippet"],
            record_ref=citation.get("record_ref"),
            document_id=citation.get("document_id"),
            chunk_id=citation.get("chunk_id"),
            char_start=citation.get("char_start"),
            char_end=citation.get("char_end"),
            speaker=citation.get("speaker"),
            occurred_at=citation.get("occurred_at"),
            verification_status=link.status,
        )
        attached.append({"handle": handle, "evidence_id": str(evidence.id)})
    return attached


async def _title_session(session_id, question, answer) -> None:
    """Best-effort and independent of the request transaction."""
    try:
        prompt = get_prompt("chat_title")
        result, _ = await client.structured(
            ChatTitleOut,
            prompt.messages(question=question, answer=answer[:2000]),
            task="chat_title",
            prompt_version=prompt.version,
            role=client.ROLE_CHEAP,
            reasoning_effort="low",
        )
        async with SessionLocal() as db:
            session = await db.get(ChatSession, session_id)
            if session is not None and session.title is None:
                session.title = result.title.strip()[:255]
                await db.commit()
    except Exception:
        logger.exception("chat.title_failed session=%s", session_id)

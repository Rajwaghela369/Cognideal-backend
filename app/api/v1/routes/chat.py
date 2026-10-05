"""Durable chat sessions and SSE turns."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import chat as chat_agent
from app.core.config import settings
from app.db.session import get_db
from app.models import ChatMessage, ChatSession, Deal
from app.models.enums import ClaimType
from app.schemas.v1.chat import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
)
from app.services import claims

router = APIRouter(prefix="/chat/sessions", tags=["chat"])


async def _session_or_404(db: AsyncSession, session_id: uuid.UUID) -> ChatSession:
    session = await db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session


@router.post("", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: ChatSessionCreate, db: AsyncSession = Depends(get_db)
) -> Any:
    if body.deal_id is not None and await db.get(Deal, body.deal_id) is None:
        raise HTTPException(status_code=404, detail="Deal not found")
    session = ChatSession(scope=body.scope, deal_id=body.deal_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.get("", response_model=list[ChatSessionResponse])
async def list_sessions(db: AsyncSession = Depends(get_db)) -> Any:
    return list((await db.scalars(
        select(ChatSession).order_by(ChatSession.updated_at.desc())
    )).all())


@router.get("/{session_id}", response_model=ChatSessionResponse)
async def get_session(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Any:
    return await _session_or_404(db, session_id)


@router.patch("/{session_id}", response_model=ChatSessionResponse)
async def rename_session(
    session_id: uuid.UUID,
    body: ChatSessionUpdate,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Rename a session.

    Task 9.5 titles a session from its first turn with the cheap model, which is
    a guess made from one message. Without this there was no way to correct it.
    """
    session = await _session_or_404(db, session_id)
    session.title = body.title.strip()
    await db.commit()
    await db.refresh(session)
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Response:
    """Delete a session and its turns.

    `chat_messages` cascades on the foreign key, but the citations do not:
    `claim_evidence.claim_id` is polymorphic and carries no foreign key, so the
    links have to be cleared explicitly or they outlive the messages they point
    at. This is the orphan trap `docs/api/README.md` section 7 describes, and
    `services/claims.py` is the only correct way through it.
    """
    session = await _session_or_404(db, session_id)
    message_ids = list(
        (
            await db.scalars(
                select(ChatMessage.id).where(ChatMessage.session_id == session_id)
            )
        ).all()
    )
    if message_ids:
        await claims.delete_claim_links(db, ClaimType.CHAT_MESSAGE, message_ids)
    await db.delete(session)
    await db.commit()
    # The agent's memory of this session, which LangGraph keeps in its own
    # tables outside the foreign key graph -- the same orphan shape as the
    # citations above, one layer out. Deliberately after the commit and
    # non-fatal: the session is already gone, and stale checkpoint rows are not
    # a reason to fail a delete that succeeded.
    if settings.ai_enabled:
        from app.ai import checkpointer

        await checkpointer.forget(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{session_id}/messages", response_model=list[ChatMessageResponse])
async def list_messages(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Any:
    await _session_or_404(db, session_id)
    rows = list((await db.scalars(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )).all())
    out = []
    for row in rows:
        payload = {column.name: getattr(row, column.name) for column in row.__table__.columns}
        evidence = await claims.evidence_for(db, ClaimType.CHAT_MESSAGE, row.id)
        handle_by_evidence = {
            item["evidence_id"]: item["handle"]
            for item in (row.token_usage or {}).get("citation_handles", [])
        }
        payload["citations"] = [
            {
                "handle": handle_by_evidence.get(
                    str(item.id), "citation-%d" % (index + 1)
                ),
                "source_kind": item.source_kind,
                "snippet": item.snippet,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "record_ref": item.record_ref,
                "char_start": item.char_start,
                "char_end": item.char_end,
            }
            for index, item in enumerate(evidence)
        ]
        out.append(payload)
    return out


@router.post("/{session_id}/messages")
async def send_message(
    session_id: uuid.UUID,
    body: ChatMessageCreate,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    session = await _session_or_404(db, session_id)

    async def events():
        async for event in chat_agent.stream_turn(db, session, body.content):
            yield "data: %s\n\n" % event

    return StreamingResponse(events(), media_type="text/event-stream")

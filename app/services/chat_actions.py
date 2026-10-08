"""Turning the chat assistant's drafts into rows -- only on a person's say-so.

The assistant's ``draft_task`` / ``draft_meeting`` tools write nothing; they
record a draft on the answer (``chat_messages.actions``, migration 0017) and
the UI shows it as a card. This module is what runs when the user clicks
Create or Cancel on that card.

Creation goes through ``services.task.create_task`` and
``services.meeting.create_meeting`` -- the same code as the REST routes -- so a
confirmed draft gets exactly the validation, status handling and analysis
triggers a hand-made task or meeting gets.

Create is idempotent. The message row is locked for the duration, and a draft
already ``created`` returns the row it made instead of making another, so a
double click or a retried request cannot duplicate work.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChatMessage, ChatSession, Contact, Deal, MeetingAttendee
from app.models.enums import Origin
from app.schemas.v1.deal.meeting import MeetingCreate
from app.schemas.v1.task import TaskCreate
from app.services import meeting as meeting_service
from app.services import roster
from app.services import task as task_service

logger = logging.getLogger("cognideal.chat.actions")

KIND_TASK = "task"
KIND_MEETING = "meeting"

PROPOSED = "proposed"
CREATED = "created"
CANCELLED = "cancelled"

#: What the user may change on the card before creating. Attendees are not
#: editable here: they were matched to contacts when drafted, and a free-text
#: edit would bypass that.
EDITABLE = {
    KIND_TASK: {"title", "description", "due_date", "priority"},
    KIND_MEETING: {"title", "meeting_type", "scheduled_at"},
}


async def _locked_message(
    db: AsyncSession, session: ChatSession, message_id: uuid.UUID
) -> ChatMessage:
    message = await db.scalar(
        select(ChatMessage)
        .where(ChatMessage.id == message_id, ChatMessage.session_id == session.id)
        .with_for_update()
    )
    if message is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found")
    return message


def _find(actions: List[Dict[str, Any]], action_id: str) -> int:
    for index, action in enumerate(actions):
        if action.get("id") == action_id:
            return index
    raise HTTPException(status.HTTP_404_NOT_FOUND, "No such action on this message")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _invalid(exc: ValidationError) -> HTTPException:
    return HTTPException(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[
            {"loc": list(error.get("loc", ())), "msg": error.get("msg")}
            for error in exc.errors()
        ],
    )


async def apply_action(
    db: AsyncSession,
    session: ChatSession,
    message_id: uuid.UUID,
    action_id: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the task or meeting a draft describes. Commits."""
    message = await _locked_message(db, session, message_id)
    actions = [dict(action) for action in (message.actions or [])]
    index = _find(actions, action_id)
    action = actions[index]

    if action.get("status") == CREATED:
        return action
    if action.get("status") == CANCELLED:
        raise HTTPException(status.HTTP_409_CONFLICT, "This draft was cancelled")

    kind = action.get("kind")
    if kind not in EDITABLE:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unknown action kind %r" % kind)
    unknown = set(overrides or {}) - EDITABLE[kind]
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Not editable on a %s draft: %s" % (kind, ", ".join(sorted(unknown))),
        )

    deal_id = uuid.UUID(str(action["deal_id"]))
    # Drafted under the session's scope already; checked again because the
    # action list is data, and data can be stale.
    if session.deal_id is not None and deal_id != session.deal_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "Draft belongs to another deal")
    deal = await db.get(Deal, deal_id)
    if deal is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The draft's deal no longer exists")

    fields = {**(action.get("fields") or {}), **(overrides or {})}

    if kind == KIND_TASK:
        try:
            body = TaskCreate(
                deal_id=deal_id,
                title=fields.get("title"),
                description=fields.get("description"),
                due_date=fields.get("due_date"),
                priority=fields.get("priority") or "medium",
            )
        except ValidationError as exc:
            raise _invalid(exc) from exc
        created = await task_service.create_task(db, body, origin=Origin.AI)
    else:
        try:
            body = MeetingCreate(
                title=fields.get("title"),
                meeting_type=fields.get("meeting_type") or "other",
                scheduled_at=fields.get("scheduled_at"),
            )
        except ValidationError as exc:
            raise _invalid(exc) from exc
        created = await meeting_service.create_meeting(db, deal_id, body)
        await _add_attendees(db, deal, created.id, fields.get("attendees") or [])

    action.update(
        status=CREATED,
        created_id=str(created.id),
        decided_at=_now(),
        fields=fields,
    )
    actions[index] = action
    # Replaced, not mutated: JSONB columns are not change-tracked in place.
    message.actions = actions
    await db.commit()
    logger.info(
        "chat.action_created kind=%s action=%s message=%s id=%s deal=%s",
        kind, action_id, message_id, created.id, deal_id,
    )
    return action


async def _add_attendees(
    db: AsyncSession, deal: Deal, meeting_id: uuid.UUID, attendees: List[Dict[str, Any]]
) -> None:
    """Attendee rows for a confirmed meeting draft.

    A contact matched when drafting is re-checked against the deal's account --
    it may have been deleted or moved since -- and dropped back to the raw name
    if it no longer fits, which is the same unresolved shape a transcript
    speaker takes. One contact appears once: ``(meeting_id, contact_id)`` is
    unique.
    """
    seen_contacts = set()
    for attendee in attendees:
        raw_name = roster.normalise(str(attendee.get("raw_name") or ""))[:200]
        if not raw_name:
            continue
        contact_id = attendee.get("contact_id")
        if contact_id:
            contact = await db.get(Contact, uuid.UUID(str(contact_id)))
            if contact is None or contact.account_id != deal.account_id:
                contact_id = None
            else:
                contact_id = contact.id
        if contact_id in seen_contacts and contact_id is not None:
            continue
        seen_contacts.add(contact_id)
        db.add(
            MeetingAttendee(
                meeting_id=meeting_id,
                raw_name=raw_name,
                contact_id=contact_id,
                is_internal=roster.is_internal(raw_name),
                # The REST route's default. `False` would read as a no-show
                # once the meeting is marked held, and attendance-based risk
                # rules would fire on people who simply were not edited.
                attended=True,
            )
        )
    await db.flush()


async def cancel_action(
    db: AsyncSession, session: ChatSession, message_id: uuid.UUID, action_id: str
) -> Dict[str, Any]:
    """Mark a draft cancelled. Commits. Cancelling a created draft is refused --
    the row exists; delete it where it lives."""
    message = await _locked_message(db, session, message_id)
    actions = [dict(action) for action in (message.actions or [])]
    index = _find(actions, action_id)
    action = actions[index]
    if action.get("status") == CREATED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Already created. Delete the %s itself to undo it." % action.get("kind"),
        )
    if action.get("status") != CANCELLED:
        action.update(status=CANCELLED, decided_at=_now())
        actions[index] = action
        message.actions = actions
        await db.commit()
        logger.info("chat.action_cancelled action=%s message=%s", action_id, message_id)
    return action

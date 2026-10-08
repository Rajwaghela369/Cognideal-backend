"""Meeting writes that are more than one statement, or that have a rule.

Same reasoning as services/deal.py and services/task.py: these rules hold no
matter which route wrote the row, so they live in one place.
"""

import uuid
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Contact, Deal, DealContact, Meeting, MeetingAttendee
from app.models.enums import AnalysisStatus, MeetingStatus
from app.schemas.v1.deal.meeting import MeetingCreate
from app.services import activity


async def get_meeting_or_404(
    db: AsyncSession, deal_id: uuid.UUID, meeting_id: uuid.UUID
) -> Meeting:
    """Resolve a meeting *within* a deal.

    The routes are nested, so a client can name a meeting that exists but
    belongs to a different deal. Answering that with the meeting would let
    deal A's URL serve deal B's data; answering with 404 is correct, because
    within this deal that meeting genuinely does not exist.

    One dependency rather than the same check copied into eight handlers.
    """
    meeting = await db.scalar(
        select(Meeting).where(Meeting.id == meeting_id, Meeting.deal_id == deal_id)
    )
    if meeting is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meeting_id} not found on deal {deal_id}",
        )
    return meeting


async def get_attendee_or_404(
    db: AsyncSession, meeting_id: uuid.UUID, attendee_id: uuid.UUID
) -> MeetingAttendee:
    attendee = await db.scalar(
        select(MeetingAttendee).where(
            MeetingAttendee.id == attendee_id,
            MeetingAttendee.meeting_id == meeting_id,
        )
    )
    if attendee is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attendee {attendee_id} not found on meeting {meeting_id}",
        )
    return attendee


async def apply_status_change(
    db: AsyncSession, meeting: Meeting, to_status: MeetingStatus
) -> None:
    """Move a meeting to a new status, maintaining its timestamps.

    The same shape as deal.apply_stage_change and task.apply_status_change:

    *   -> completed  stamp ended_at if the caller did not supply one, so a
                      meeting marked done always has an end.
    *   -> scheduled  clear started_at and ended_at. A meeting moved back to
                      scheduled did not happen, and leaving the timestamps
                      makes it read as held to every query that checks them.
    *   -> cancelled  leave the timestamps alone. A meeting that started and
                      was then abandoned really did start.
    """
    if meeting.status == to_status:
        return

    was_completed = meeting.status == MeetingStatus.COMPLETED
    meeting.status = to_status

    if to_status == MeetingStatus.COMPLETED:
        if meeting.ended_at is None:
            meeting.ended_at = func.now()
        # A held meeting is the strongest form of "someone talked to them".
        # ended_at may be a SQL expression rather than a value at this point,
        # so the touch defaults to now() -- within a request the two are the
        # same moment for every purpose this column serves.
        await activity.touch_deal(db, meeting.deal_id)
        from app.services import analysis

        await analysis.record_change(
            db,
            meeting.deal_id,
            "meeting.completed",
            table="meetings",
            row_id=meeting.id,
            fields=("status", "ended_at"),
        )
    elif to_status == MeetingStatus.SCHEDULED and was_completed:
        meeting.started_at = None
        meeting.ended_at = None


async def create_meeting(db: AsyncSession, deal_id: uuid.UUID, body: MeetingCreate) -> Meeting:
    """Create one meeting: the single path for ``POST .../meetings`` and chat
    drafts.

    Built as scheduled, then moved -- so a meeting logged as already held gets
    its ``ended_at`` (and its analysis triggers) from :func:`apply_status_change`.
    """
    payload = body.model_dump()
    to_status = payload.pop("status")
    meeting = Meeting(deal_id=deal_id, status=MeetingStatus.SCHEDULED, **payload)
    db.add(meeting)
    await apply_status_change(db, meeting, to_status)
    await db.flush()
    return meeting


async def queue_analysis(
    db: AsyncSession, meeting: Meeting, transcript_document_id: Optional[uuid.UUID]
) -> None:
    """Mark a meeting as awaiting analysis.

    Consumed by ``backend/worker.py``, which claims queued meetings with
    ``FOR UPDATE SKIP LOCKED`` and runs the pipeline in ``app/ai/pipeline.py``.
    The pipeline's stages are still no-ops (docs/ai/TASKS.md phases 2-7), so a
    run currently reaches ``complete`` having written nothing -- the queue,
    the claim and the crash semantics are what exist.

    ``analyzed_at`` is cleared here: a re-analysis has not happened yet, and
    leaving the old timestamp makes the Analyzer screen read as done.
    """
    if transcript_document_id is not None:
        meeting.transcript_document_id = transcript_document_id
    meeting.analysis_status = AnalysisStatus.QUEUED
    meeting.analyzed_at = None


async def assert_contact_on_deal_account(
    db: AsyncSession, deal: Deal, contact_id: uuid.UUID
) -> Contact:
    """Contacts belong to accounts and meetings belong to deals, but nothing
    stops an attendee pointing at a contact from a different company. No
    foreign key can express the rule."""
    contact = await db.get(Contact, contact_id)
    if contact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contact {contact_id} not found",
        )
    if contact.account_id != deal.account_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Contact {contact_id} belongs to a different account than "
                f"deal {deal.id}"
            ),
        )
    return contact


async def create_contact_from_identity(
    db: AsyncSession, deal: Deal, identity
) -> Contact:
    """Create the person an unresolved attendee turned out to be.

    account_id comes from the deal, never from the client, so a contact cannot
    land on the wrong company.

    The duplicate-email rule is `services/account.py`'s, not a second copy of
    it: the `/accounts/{id}/contacts` collection creates contacts too, and two
    writers disagreeing about what a repeat email means is how one path starts
    returning 500s the other handles.
    """
    from app.services import account as account_service

    await account_service.assert_email_free(
        db, deal.account_id, identity.email
    )

    contact = Contact(account_id=deal.account_id, **identity.model_dump())
    db.add(contact)
    await db.flush()
    return contact


async def upsert_stakeholder(
    db: AsyncSession, deal_id: uuid.UUID, contact_id: uuid.UUID, identity
) -> DealContact:
    """Add or update the deal_contacts row for a resolved attendee.

    Upsert rather than insert: a person can already be a stakeholder and still
    appear as an unresolved name in a transcript, so resolving them must not
    409 on a link that already exists.
    """
    from app.services import deal as deal_service

    payload = {k: v for k, v in identity.model_dump().items() if v is not None}

    if payload.get("is_primary"):
        # Must precede the write: uq_deal_contacts_deal_id_primary is a partial
        # unique index, checked per statement and never deferrable.
        await deal_service.demote_primary_stakeholder(db, deal_id, keeping=contact_id)

    link = await db.scalar(
        select(DealContact).where(
            DealContact.deal_id == deal_id, DealContact.contact_id == contact_id
        )
    )
    if link is None:
        link = DealContact(deal_id=deal_id, contact_id=contact_id, **payload)
        db.add(link)
    else:
        for field, value in payload.items():
            setattr(link, field, value)
    await db.flush()
    return link

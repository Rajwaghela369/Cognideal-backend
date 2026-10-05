"""Who was on a call, and turning an unknown name into a tracked person.

Two lists, and the difference is the whole point:

*   /meetings/{id}/attendees      who was on THIS call
*   /deals/{deal_id}/participants who has been on ANY call for this deal

Only the second shows the gap between who you listed as a stakeholder and who
actually turns up -- which is the signal this table exists to produce.
"""

import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import false, func, literal, null, select, true
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from typing_extensions import Annotated

from app.api.deps import get_deal_or_404
from app.db.session import get_db
from app.models import Contact, Deal, DealContact, Meeting, MeetingAttendee
from app.schemas.v1.deal.meeting import (
    AttendeeCreate,
    AttendeeFilters,
    AttendeeResolve,
    AttendeeUpdate,
    DealParticipant,
    MeetingAttendee as MeetingAttendeeOut,
    ParticipantFilters,
)
from app.services import meeting as meeting_service
from app.services import analysis as analysis_service

router = APIRouter(
    prefix="/deals/{deal_id}/meetings/{meeting_id}/attendees", tags=["attendees"]
)
participants = APIRouter(prefix="/deals/{deal_id}", tags=["participants"])

_RESOLVED = MeetingAttendee.contact_id.is_not(None).label("resolved")


async def get_meeting(
    deal_id: uuid.UUID,
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> Meeting:
    return await meeting_service.get_meeting_or_404(db, deal_id, meeting_id)


def _attendee_stmt(meeting_id: uuid.UUID) -> Select:
    return (
        select(
            MeetingAttendee.id,
            MeetingAttendee.raw_name,
            MeetingAttendee.contact_id,
            (Contact.first_name + " " + Contact.last_name).label("contact_name"),
            Contact.email.label("contact_email"),
            Contact.title.label("contact_title"),
            MeetingAttendee.is_internal,
            MeetingAttendee.attended,
            _RESOLVED,
        )
        # outerjoin, not join: an unresolved attendee has no contact, and those
        # are exactly the rows worth surfacing. An inner join would hide them.
        .outerjoin(Contact, Contact.id == MeetingAttendee.contact_id)
        .where(MeetingAttendee.meeting_id == meeting_id)
        # Unresolved first -- they are the ones needing action -- then by name.
        .order_by(_RESOLVED.asc(), MeetingAttendee.raw_name.asc())
    )


@router.get("", response_model=List[MeetingAttendeeOut])
async def list_attendees(
    filters: Annotated[AttendeeFilters, Query()],
    meeting: Meeting = Depends(get_meeting),
    db: AsyncSession = Depends(get_db),
) -> Any:
    stmt = _attendee_stmt(meeting.id)
    if filters.resolved is not None:
        stmt = stmt.where(
            MeetingAttendee.contact_id.is_not(None)
            if filters.resolved
            else MeetingAttendee.contact_id.is_(None)
        )
    if filters.is_internal is not None:
        stmt = stmt.where(MeetingAttendee.is_internal.is_(filters.is_internal))
    if filters.attended is not None:
        stmt = stmt.where(MeetingAttendee.attended.is_(filters.attended))
    return (await db.execute(stmt)).all()


async def _one_attendee(db: AsyncSession, meeting_id: uuid.UUID, attendee_id: uuid.UUID) -> Any:
    """Re-read through the contact outerjoin after a write -- the ORM object
    just written carries no name or email."""
    return (
        await db.execute(
            _attendee_stmt(meeting_id).where(MeetingAttendee.id == attendee_id)
        )
    ).first()


@router.post("", response_model=MeetingAttendeeOut, status_code=status.HTTP_201_CREATED)
async def add_attendee(
    body: AttendeeCreate,
    deal: Deal = Depends(get_deal_or_404),
    meeting: Meeting = Depends(get_meeting),
    db: AsyncSession = Depends(get_db),
) -> Any:
    if body.contact_id is not None:
        await meeting_service.assert_contact_on_deal_account(db, deal, body.contact_id)

    meeting_id = meeting.id
    attendee = MeetingAttendee(meeting_id=meeting_id, **body.model_dump())
    db.add(attendee)
    try:
        await db.flush()
        await analysis_service.record_change(
            db,
            deal.id,
            "meeting.attendee_added",
            table="meeting_attendees",
            row_id=attendee.id,
            fields=("raw_name", "contact_id", "is_internal", "attended"),
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Contact {body.contact_id} is already an attendee on this "
                f"meeting"
            ),
        )
    return await _one_attendee(db, meeting_id, attendee.id)


@router.patch("/{attendee_id}", response_model=MeetingAttendeeOut)
async def update_attendee(
    attendee_id: uuid.UUID,
    body: AttendeeUpdate,
    deal: Deal = Depends(get_deal_or_404),
    meeting: Meeting = Depends(get_meeting),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Plain field edits, including linking to an existing contact.

    To create the contact at the same time, use POST /{attendee_id}/resolve.
    """
    attendee = await meeting_service.get_attendee_or_404(db, meeting.id, attendee_id)
    changes = body.model_dump(exclude_unset=True)
    if changes.get("contact_id") is not None:
        await meeting_service.assert_contact_on_deal_account(
            db, deal, changes["contact_id"]
        )
    for field, value in changes.items():
        setattr(attendee, field, value)

    meeting_id = meeting.id
    try:
        await analysis_service.record_change(
            db,
            deal.id,
            "meeting.attendee_updated",
            table="meeting_attendees",
            row_id=attendee.id,
            fields=changes.keys(),
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That contact is already an attendee on this meeting",
        )
    return await _one_attendee(db, meeting_id, attendee_id)


@router.delete("/{attendee_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_attendee(
    attendee_id: uuid.UUID,
    deal: Deal = Depends(get_deal_or_404),
    meeting: Meeting = Depends(get_meeting),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Removes the attendance record, never the contact."""
    attendee = await meeting_service.get_attendee_or_404(db, meeting.id, attendee_id)
    await db.delete(attendee)
    await analysis_service.record_change(
        db,
        deal.id,
        "meeting.attendee_deleted",
        table="meeting_attendees",
        row_id=attendee_id,
        fields=("raw_name", "contact_id", "is_internal", "attended"),
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{attendee_id}/resolve", response_model=MeetingAttendeeOut)
async def resolve_attendee(
    attendee_id: uuid.UUID,
    body: AttendeeResolve,
    deal: Deal = Depends(get_deal_or_404),
    meeting: Meeting = Depends(get_meeting),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Turn "Dana (procurement)" into a tracked person, in one transaction.

    Up to three writes for one human action -- create the contact, point the
    attendee at it, add the deal_contacts row -- so it cannot half-fail with a
    contact created and no stakeholder link.

    The same shape as extracted_facts promotion: a raw signal, a human
    approving it, and Layer A rows created as a result. Here the attendee row
    *is* the staging record, so no fact is involved.
    """
    attendee = await meeting_service.get_attendee_or_404(db, meeting.id, attendee_id)

    if attendee.is_internal:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Internal attendees are your own people, not customer "
                "contacts. Clear is_internal first if this is a mistake."
            ),
        )
    if attendee.contact_id is not None and not body.force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Already resolved to contact {attendee.contact_id}; pass "
                f"force=true to re-point it."
            ),
        )

    if body.contact_id is not None:
        contact = await meeting_service.assert_contact_on_deal_account(
            db, deal, body.contact_id
        )
    else:
        contact = await meeting_service.create_contact_from_identity(
            db, deal, body.contact
        )

    attendee.contact_id = contact.id
    if body.stakeholder is not None:
        await meeting_service.upsert_stakeholder(
            db, deal.id, contact.id, body.stakeholder
        )

    meeting_id = meeting.id
    try:
        await analysis_service.record_change(
            db,
            deal.id,
            "meeting.attendee_resolved",
            table="meeting_attendees",
            row_id=attendee.id,
            fields=("contact_id",),
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That contact is already an attendee on this meeting",
        )
    return await _one_attendee(db, meeting_id, attendee_id)


# --------------------------------------------------------------------------
# Participants -- every person across every meeting on the deal
# --------------------------------------------------------------------------


@participants.get("/participants", response_model=List[DealParticipant])
async def list_participants(
    filters: Annotated[ParticipantFilters, Query()],
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Who is actually involved in this deal.

    Two populations unioned, because each answers half the question:

    *   everyone who attended a meeting -- including names that resolve to no
        contact, which is the missing-stakeholder signal
    *   every stakeholder, including those who have never attended anything,
        which for an economic buyer is the NO_ECONOMIC_BUYER risk

    An inner join either way would hide exactly the rows worth seeing.
    """
    attended = (
        select(
            MeetingAttendee.contact_id.label("contact_id"),
            # Unresolved attendees have no contact, so the snapshotted
            # raw_name is the only name available -- which is why the column
            # is NOT NULL.
            func.coalesce(
                Contact.first_name + " " + Contact.last_name,
                MeetingAttendee.raw_name,
            ).label("name"),
            Contact.email.label("email"),
            Contact.title.label("title"),
            MeetingAttendee.is_internal.label("is_internal"),
            func.count(Meeting.id).label("meetings_attended"),
            func.max(Meeting.scheduled_at).label("last_seen_at"),
        )
        .select_from(MeetingAttendee)
        .join(Meeting, Meeting.id == MeetingAttendee.meeting_id)
        .outerjoin(Contact, Contact.id == MeetingAttendee.contact_id)
        .where(Meeting.deal_id == deal.id, MeetingAttendee.attended.is_(True))
        .group_by(
            MeetingAttendee.contact_id,
            Contact.first_name,
            Contact.last_name,
            Contact.email,
            Contact.title,
            MeetingAttendee.raw_name,
            MeetingAttendee.is_internal,
        )
        .subquery()
    )

    stmt = (
        select(
            attended.c.contact_id,
            attended.c.name,
            attended.c.email,
            attended.c.title,
            attended.c.is_internal,
            attended.c.contact_id.is_not(None).label("resolved"),
            DealContact.id.is_not(None).label("is_stakeholder"),
            DealContact.buying_role,
            DealContact.influence,
            attended.c.meetings_attended,
            attended.c.last_seen_at,
        )
        .select_from(attended)
        .outerjoin(
            DealContact,
            (DealContact.contact_id == attended.c.contact_id)
            & (DealContact.deal_id == deal.id),
        )
    )

    # Stakeholders who have never attended anything. Unreachable from the
    # attendee side -- they have no meeting_attendees row at all -- so they
    # need their own SELECT with literal columns to match the shape above.
    never_attended = (
        select(
            DealContact.contact_id.label("contact_id"),
            (Contact.first_name + " " + Contact.last_name).label("name"),
            Contact.email.label("email"),
            Contact.title.label("title"),
            false().label("is_internal"),
            true().label("resolved"),
            true().label("is_stakeholder"),
            DealContact.buying_role,
            DealContact.influence,
            literal(0).label("meetings_attended"),
            null().label("last_seen_at"),
        )
        .join(Contact, Contact.id == DealContact.contact_id)
        .where(
            DealContact.deal_id == deal.id,
            ~select(MeetingAttendee.id)
            .join(Meeting, Meeting.id == MeetingAttendee.meeting_id)
            .where(
                MeetingAttendee.contact_id == DealContact.contact_id,
                Meeting.deal_id == deal.id,
            )
            .exists(),
        )
    )

    rows = list((await db.execute(stmt)).all())
    rows += list((await db.execute(never_attended)).all())

    out = []
    for r in rows:
        d = dict(r._mapping)
        if filters.is_internal is not None and bool(d["is_internal"]) != filters.is_internal:
            continue
        if filters.resolved is not None and bool(d["resolved"]) != filters.resolved:
            continue
        if (
            filters.is_stakeholder is not None
            and bool(d["is_stakeholder"]) != filters.is_stakeholder
        ):
            continue
        out.append(d)

    # Most-engaged first, then the never-attended stakeholders at the bottom
    # where they read as the anomaly they are.
    out.sort(key=lambda d: (-int(d["meetings_attended"] or 0), d["name"] or ""))
    return out

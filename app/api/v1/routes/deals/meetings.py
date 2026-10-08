"""Meetings on a deal, and the analysis screen's endpoints.

Fully nested under /deals/{deal_id}: meetings are read from the deal detail
page and nowhere else, so there is no cross-deal collection. A meeting does
have its own uuid, so the nesting is a deliberate consistency choice rather
than a necessity -- the cost is that every handler must confirm the meeting
belongs to the deal named in the path, which get_meeting_or_404 does once.
"""

import logging
import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from typing_extensions import Annotated

from app.api.deps import get_deal_or_404
from app.db.session import get_db
from app.models import (
    Account,
    ClaimEvidence,
    Deal,
    ExtractedFact,
    Meeting,
    MeetingAttendee,
    Risk,
)
from app.models.enums import AnalysisStatus, ClaimType, MeetingStatus, RiskStatus
from app.schemas.v1.deal.meeting import (
    AnalysisRequest,
    BriefRequest,
    MeetingAnalysis,
    MeetingCreate,
    MeetingDetail,
    MeetingFilters,
    MeetingListItem,
    MeetingBriefResponse,
    MeetingUpdate,
)
from app.services import meeting as meeting_service
from app.services import analysis as analysis_service
from app.ai import brief as brief_service

logger = logging.getLogger("cognideal.api.meetings")

router = APIRouter(prefix="/deals/{deal_id}/meetings", tags=["meetings"])

_SORT_KEYS = {"scheduled_at", "title", "status", "created_at"}

# Derived per request, never stored -- a cached attendee count is wrong the
# moment someone is added.
_ATTENDEE_COUNT = (
    select(func.count())
    .select_from(MeetingAttendee)
    .where(MeetingAttendee.meeting_id == Meeting.id)
    .correlate(Meeting)
    .scalar_subquery()
    .label("attendee_count")
)
_HAS_TRANSCRIPT = Meeting.transcript_document_id.is_not(None).label("has_transcript")


async def get_meeting(
    deal_id: uuid.UUID,
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> Meeting:
    return await meeting_service.get_meeting_or_404(db, deal_id, meeting_id)


def _list_stmt(deal_id: uuid.UUID, f: MeetingFilters) -> Select:
    stmt = select(
        Meeting.id,
        Meeting.title,
        Meeting.meeting_type,
        Meeting.status,
        Meeting.scheduled_at,
        Meeting.started_at,
        Meeting.ended_at,
        Meeting.sentiment,
        Meeting.analysis_status,
        Meeting.analyzed_at,
        _HAS_TRANSCRIPT,
        _ATTENDEE_COUNT,
    ).where(Meeting.deal_id == deal_id)

    if f.status:
        stmt = stmt.where(Meeting.status.in_(f.status))
    if f.meeting_type:
        stmt = stmt.where(Meeting.meeting_type.in_(f.meeting_type))
    if f.analysis_status:
        stmt = stmt.where(Meeting.analysis_status.in_(f.analysis_status))
    if f.upcoming is not None:
        still_ahead = (
            (Meeting.status == MeetingStatus.SCHEDULED)
            & Meeting.scheduled_at.is_not(None)
            & (Meeting.scheduled_at >= func.now())
        )
        stmt = stmt.where(still_ahead if f.upcoming else ~still_ahead)
    if f.has_transcript is not None:
        stmt = stmt.where(
            Meeting.transcript_document_id.is_not(None)
            if f.has_transcript
            else Meeting.transcript_document_id.is_(None)
        )
    if f.q:
        pattern = f"%{f.q}%"
        stmt = stmt.where(
            or_(Meeting.title.ilike(pattern), Meeting.summary.ilike(pattern))
        )

    descending = f.sort.startswith("-")
    column = getattr(Meeting, f.sort.lstrip("-"))
    order = column.desc() if descending else column.asc()
    # A meeting with no scheduled_at is unscheduled, not earliest or latest.
    return stmt.order_by(order.nulls_last(), Meeting.id)


@router.get("", response_model=List[MeetingListItem])
async def list_meetings(
    filters: Annotated[MeetingFilters, Query()],
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """The meeting track on the deal page.

    Unpaginated: one deal has tens of meetings at worst, and an envelope would
    cost the caller an unwrap for nothing.
    """
    if filters.sort.lstrip("-") not in _SORT_KEYS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown sort key. Valid: {', '.join(sorted(_SORT_KEYS))}",
        )
    return (await db.execute(_list_stmt(deal.id, filters))).all()


async def _detail(db: AsyncSession, deal_id: uuid.UUID, meeting_id: uuid.UUID) -> Any:
    stmt = (
        select(
            Meeting,
            Deal.name.label("deal_name"),
            Deal.account_id,
            Account.name.label("account_name"),
            _HAS_TRANSCRIPT,
            _ATTENDEE_COUNT,
        )
        .join(Deal, Deal.id == Meeting.deal_id)
        .join(Account, Account.id == Deal.account_id)
        .where(Meeting.id == meeting_id, Meeting.deal_id == deal_id)
    )
    row = (await db.execute(stmt)).first()
    m = row.Meeting
    return {
        "id": m.id,
        "deal_id": m.deal_id,
        "deal_name": row.deal_name,
        "account_id": row.account_id,
        "account_name": row.account_name,
        "title": m.title,
        "meeting_type": m.meeting_type,
        "status": m.status,
        "scheduled_at": m.scheduled_at,
        "started_at": m.started_at,
        "ended_at": m.ended_at,
        "summary": m.summary,
        "sentiment": m.sentiment,
        "analysis_status": m.analysis_status,
        "analyzed_at": m.analyzed_at,
        "transcript_document_id": m.transcript_document_id,
        "has_transcript": row.has_transcript,
        "attendee_count": row.attendee_count,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
    }


@router.get("/{meeting_id}", response_model=MeetingDetail)
async def get_one(
    meeting: Meeting = Depends(get_meeting), db: AsyncSession = Depends(get_db)
) -> Any:
    return await _detail(db, meeting.deal_id, meeting.id)


@router.post("", response_model=MeetingDetail, status_code=status.HTTP_201_CREATED)
async def create_meeting(
    body: MeetingCreate,
    response: Response,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    # The one creation path, shared with the chat's confirmed drafts.
    meeting = await meeting_service.create_meeting(db, deal.id, body)
    await db.commit()

    response.headers["Location"] = f"/deals/{deal.id}/meetings/{meeting.id}"
    return await _detail(db, deal.id, meeting.id)


@router.patch("/{meeting_id}", response_model=MeetingDetail)
async def update_meeting(
    body: MeetingUpdate,
    meeting: Meeting = Depends(get_meeting),
    db: AsyncSession = Depends(get_db),
) -> Any:
    changes = body.model_dump(exclude_unset=True)
    to_status = changes.pop("status", None)

    for field, value in changes.items():
        setattr(meeting, field, value)
    if to_status is not None:
        # Not a plain column write: also maintains started_at / ended_at.
        await meeting_service.apply_status_change(db, meeting, to_status)

    deal_id, meeting_id = meeting.deal_id, meeting.id
    await db.commit()
    return await _detail(db, deal_id, meeting_id)


@router.delete("/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_meeting(
    meeting: Meeting = Depends(get_meeting), db: AsyncSession = Depends(get_db)
) -> Response:
    """Cascades to this meeting's attendees and its brief.

    The transcript document is NOT deleted -- transcript_document_id is
    ON DELETE SET NULL in the other direction, and the document is an
    independent record that may already be chunked, embedded and cited.
    """
    deal_id, meeting_id = meeting.deal_id, meeting.id
    attendee_ids = list(
        (
            await db.scalars(
                select(MeetingAttendee.id).where(
                    MeetingAttendee.meeting_id == meeting_id
                )
            )
        ).all()
    )
    await db.delete(meeting)
    # The database cascades the attendee rows, but record_ref is JSON rather
    # than a foreign key. Re-resolve those citations after the cascade flushes
    # so they become span_missing in this same transaction.
    for attendee_id in attendee_ids:
        await analysis_service.reverify_record_refs(
            db,
            table="meeting_attendees",
            row_id=attendee_id,
            fields=("raw_name", "contact_id", "is_internal", "attended"),
        )
    await analysis_service.record_change(
        db,
        deal_id,
        "meeting deleted",
        table="meetings",
        row_id=meeting_id,
        fields=("status", "ended_at"),
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------
# Analysis
#
# NOT YET IMPLEMENTED BEYOND THE CONTRACT. GET reads the columns and is real.
# POST sets analysis_status to `queued` and returns 202 -- but no worker
# consumes that queue, so a meeting stays queued until the Meeting Analyzer is
# built. The endpoints exist now so the screen can be written against the shape
# it will eventually have; anyone wondering why nothing happens should read
# this comment rather than the code.
# --------------------------------------------------------------------------


async def _analysis_payload(db: AsyncSession, meeting: Meeting) -> dict:
    payload = {
        "meeting_id": meeting.id,
        "analysis_status": meeting.analysis_status,
        "analyzed_at": meeting.analyzed_at,
        "summary": meeting.summary,
        "sentiment": meeting.sentiment,
        "has_transcript": meeting.transcript_document_id is not None,
        "analysis_error": meeting.analysis_error,
    }
    if meeting.analysis_status == AnalysisStatus.COMPLETE:
        fact_ids = select(ExtractedFact.id).where(ExtractedFact.meeting_id == meeting.id)
        payload["facts_count"] = await db.scalar(
            select(func.count()).select_from(fact_ids.subquery())
        )
        payload["evidence_count"] = await db.scalar(
            select(func.count()).select_from(ClaimEvidence).where(
                ClaimEvidence.claim_type == ClaimType.FACT,
                ClaimEvidence.claim_id.in_(fact_ids),
            )
        )
        payload["open_risks_count"] = await db.scalar(
            select(func.count()).select_from(Risk).where(
                Risk.deal_id == meeting.deal_id, Risk.status == RiskStatus.OPEN
            )
        )
    return payload


@router.get("/{meeting_id}/analysis", response_model=MeetingAnalysis)
async def get_analysis(
    meeting: Meeting = Depends(get_meeting), db: AsyncSession = Depends(get_db)
) -> Any:
    """What the Meeting Analyzer screen polls."""
    return await _analysis_payload(db, meeting)


@router.post(
    "/{meeting_id}/analysis",
    response_model=MeetingAnalysis,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_analysis(
    body: AnalysisRequest,
    meeting: Meeting = Depends(get_meeting),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Queue this meeting for analysis.

    202 rather than 200: the work is accepted, not done. See the section
    comment above -- nothing drains this queue yet.
    """
    if meeting.analysis_status == AnalysisStatus.COMPLETE and not body.force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Analysis is already complete; pass force=true to re-run",
        )
    await meeting_service.queue_analysis(db, meeting, body.transcript_document_id)
    await db.commit()
    await db.refresh(meeting)
    return await _analysis_payload(db, meeting)


@router.get("/{meeting_id}/brief", response_model=MeetingBriefResponse)
async def get_brief(
    meeting: Meeting = Depends(get_meeting),
    db: AsyncSession = Depends(get_db),
) -> Any:
    from app.models import MeetingBrief

    brief = await db.scalar(
        select(MeetingBrief).where(MeetingBrief.meeting_id == meeting.id)
    )
    if brief is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Brief not generated")
    return brief


@router.post("/{meeting_id}/brief", response_model=MeetingBriefResponse)
async def generate_brief(
    body: BriefRequest,
    meeting: Meeting = Depends(get_meeting),
    db: AsyncSession = Depends(get_db),
) -> Any:
    from app.ai.client import AIDisabled

    meeting_id = meeting.id
    try:
        brief = await brief_service.generate(db, meeting, force=body.force)
        await db.commit()
        await db.refresh(brief)
    except AIDisabled as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        # A model or database failure used to escape as a bare 500, which the
        # UI can only show as "Request failed with status 500". Logged with
        # the traceback, and returned with its cause so the screen says why.
        await db.rollback()
        logger.exception("brief.generate_failed meeting=%s", meeting_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate the brief: %s: %s" % (type(exc).__name__, exc),
        )
    logger.info("brief.generated meeting=%s force=%s", meeting_id, body.force)
    return brief

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    AnalysisStatus,
    MeetingStatus,
    MeetingType,
    Origin,
    Sentiment,
    analysis_status_enum,
    check_in,
    meeting_status_enum,
    origin_enum,
    sentiment_enum,
)


class Meeting(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "meetings"
    __table_args__ = (
        check_in("meeting_type", MeetingType, "meeting_type"),
        # Every read of this table is "one deal's meetings, newest first" --
        # the track on the deal page. A bare deal_id index finds the rows and
        # then sorts them separately. deal_id-leading, so it serves the plain
        # lookup the old index served.
        Index("ix_meetings_deal_id_scheduled_at", "deal_id", "scheduled_at"),
    )

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    meeting_type: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default=MeetingType.OTHER.value
    )
    status: Mapped[MeetingStatus] = mapped_column(
        meeting_status_enum, nullable=False, server_default=MeetingStatus.SCHEDULED.value
    )
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # The transcript lives in `documents` like every other unstructured input,
    # so it is chunked, embedded and citable by the same code path as an email
    # or a contract.
    transcript_document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment: Mapped[Optional[Sentiment]] = mapped_column(sentiment_enum, nullable=True)
    # What the Meeting Analyzer screen polls.
    analysis_status: Mapped[AnalysisStatus] = mapped_column(
        analysis_status_enum,
        nullable=False,
        server_default=AnalysisStatus.NOT_STARTED.value,
    )
    analyzed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Who wrote `summary` and `sentiment`, and what it cost. Scoped to the
    # fields the pipeline owns rather than the whole row -- the title and the
    # timestamps are the user's. See migration 0011.
    analysis_origin: Mapped[Optional[Origin]] = mapped_column(origin_enum, nullable=True)
    analysis_confidence: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), nullable=True
    )
    analysis_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # Why a run failed. Without it `analysis_status='failed'` can say that it
    # failed but not why, which is not much use to the person looking at it.
    analysis_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class MeetingAttendee(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Who was on the call.

    ``raw_name`` is required and ``contact_id`` is the optional *resolution* of
    that name to a known person -- not the other way round. Two reasons:

    *   Transcript speakers are frequently not in ``contacts`` yet, and those
        are exactly the people worth surfacing: a name that maps to no known
        contact is the raw signal for "missing stakeholder".
    *   Attendance is a historical fact. Deleting a contact sets ``contact_id``
        to NULL, and a row whose only identity was that link would be left
        anonymous -- so the name is snapshotted at insert time and the row
        degrades to "unresolved attendee" instead of breaking.
    """

    __tablename__ = "meeting_attendees"
    __table_args__ = (
        # "Has this person attended any meeting?" walks contact_id -> meetings,
        # and that is the direction the product is built on: the missing
        # stakeholder list and the NO_ECONOMIC_BUYER risk both run it once per
        # person. It was the one direction with no index.
        Index("ix_meeting_attendees_contact_id", "contact_id"),
        # One row per resolved person per meeting. Without this the same
        # contact can be added twice -- once from the invite, once from the
        # transcript -- and every attendance count is silently wrong.
        #
        # Partial, because unresolved attendees all carry contact_id IS NULL
        # and a meeting may legitimately have many of those.
        Index(
            "uq_meeting_attendees_meeting_id_contact_id",
            "meeting_id",
            "contact_id",
            unique=True,
            postgresql_where=text("contact_id IS NOT NULL"),
        ),
    )

    meeting_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    contact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contacts.id", ondelete="SET NULL"),
        nullable=True,
    )
    raw_name: Mapped[str] = mapped_column(String(200), nullable=False)
    is_internal: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    attended: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )


class MeetingBrief(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Pre-meeting preparation, persisted so it survives a page reload instead
    of costing a regeneration on every view."""

    __tablename__ = "meeting_briefs"
    __table_args__ = (UniqueConstraint("meeting_id", name="one_per_meeting"),)

    meeting_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
    )
    objectives: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    context_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_risks: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    recommended_questions: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

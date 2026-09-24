import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    AnalysisStatus,
    MeetingStatus,
    MeetingType,
    Sentiment,
    analysis_status_enum,
    check_in,
    meeting_status_enum,
    sentiment_enum,
)


class Meeting(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "meetings"
    __table_args__ = (check_in("meeting_type", MeetingType, "meeting_type"),)

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
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


class MeetingAttendee(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Who was on the call.

    ``contact_id`` is nullable on purpose. Transcript speakers are frequently
    not in ``contacts`` yet, and those are exactly the people worth surfacing:
    a name in a transcript that maps to no known contact is the raw signal for
    "missing stakeholder". Storing only known contacts would discard it.
    """

    __tablename__ = "meeting_attendees"
    __table_args__ = (
        CheckConstraint(
            "contact_id IS NOT NULL OR raw_name IS NOT NULL",
            name="identified_somehow",
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
    raw_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
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

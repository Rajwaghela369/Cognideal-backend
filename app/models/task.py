import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    ActivityType,
    Origin,
    Priority,
    TaskStatus,
    check_in,
    origin_enum,
    priority_enum,
    task_status_enum,
)


class Task(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Actionable work. Powers the dashboard's overdue-actions panel and is the
    source of a deal's "next action" -- which is why ``deals`` has no
    ``next_action`` column of its own."""

    __tablename__ = "tasks"
    __table_args__ = (
        # Partial index: the overdue-actions panel only ever scans open tasks,
        # and closed ones will eventually outnumber them many times over.
        Index(
            "ix_tasks_open_due_date",
            "due_date",
            postgresql_where=text("status = 'open'"),
        ),
    )

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    due_date: Mapped[Optional[date]] = mapped_column(nullable=True)
    status: Mapped[TaskStatus] = mapped_column(
        task_status_enum, nullable=False, server_default=TaskStatus.OPEN.value
    )
    priority: Mapped[Priority] = mapped_column(
        priority_enum, nullable=False, server_default=Priority.MEDIUM.value
    )
    origin: Mapped[Origin] = mapped_column(
        origin_enum, nullable=False, server_default=Origin.USER.value
    )
    # Set when this task was promoted from an approved extraction (Gate 3).
    source_fact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("extracted_facts.id", ondelete="SET NULL"),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Activity(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Append-only deal timeline."""

    __tablename__ = "activities"
    __table_args__ = (
        check_in("activity_type", ActivityType, "activity_type"),
        Index("ix_activities_deal_id_occurred_at", "deal_id", "occurred_at"),
    )

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
    )
    contact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contacts.id", ondelete="SET NULL"),
        nullable=True,
    )
    meeting_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=True,
    )
    activity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

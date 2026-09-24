import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    BuyingRole,
    DealStage,
    InfluenceLevel,
    RiskLevel,
    Sentiment,
    buying_role_enum,
    deal_stage_enum,
    influence_level_enum,
    risk_level_enum,
    sentiment_enum,
)


class Deal(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An opportunity.

    Deliberately absent:

    *   ``status`` -- ``stage`` carries the whole lifecycle, closed_won and
        closed_lost included. Two columns encoding one lifecycle always drift.
    *   ``next_action`` / ``next_action_due_date`` -- derived from the oldest
        open row in ``tasks``. A stored copy is a copy that goes stale.
    """

    __tablename__ = "deals"
    __table_args__ = (
        Index("ix_deals_stage_expected_close_date", "stage", "expected_close_date"),
        CheckConstraint(
            "win_probability IS NULL OR (win_probability >= 0 AND win_probability <= 100)",
            name="win_probability_range",
        ),
    )

    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2), nullable=True)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default="USD"
    )
    stage: Mapped[DealStage] = mapped_column(deal_stage_enum, nullable=False)
    win_probability: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    risk_level: Mapped[Optional[RiskLevel]] = mapped_column(risk_level_enum, nullable=True)
    expected_close_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class DealContact(TimestampMixin, Base):
    """Which people matter on which deal, and how.

    This table is what makes "no economic buyer has ever attended a meeting" a
    SQL query rather than a model's guess. Missing-stakeholder detection lives
    or dies on ``buying_role``.
    """

    __tablename__ = "deal_contacts"

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        primary_key=True,
    )
    contact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contacts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    buying_role: Mapped[BuyingRole] = mapped_column(
        buying_role_enum, nullable=False, server_default=BuyingRole.UNKNOWN.value
    )
    influence: Mapped[InfluenceLevel] = mapped_column(
        influence_level_enum,
        nullable=False,
        server_default=InfluenceLevel.UNKNOWN.value,
    )
    sentiment: Mapped[Sentiment] = mapped_column(
        sentiment_enum, nullable=False, server_default=Sentiment.UNKNOWN.value
    )
    is_primary: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class DealStageHistory(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Append-only stage transitions. Feeds stalled-deal detection -- cheap to
    maintain, impossible to reconstruct later if skipped."""

    __tablename__ = "deal_stage_history"

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_stage: Mapped[Optional[DealStage]] = mapped_column(deal_stage_enum, nullable=True)
    to_stage: Mapped[DealStage] = mapped_column(deal_stage_enum, nullable=False)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

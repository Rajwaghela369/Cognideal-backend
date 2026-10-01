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
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    Origin,
    origin_enum,
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


class DealContact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Which people matter on which deal, and how.

    This table is what makes "no economic buyer has ever attended a meeting" a
    SQL query rather than a model's guess. Missing-stakeholder detection lives
    or dies on ``buying_role``.

    ``(deal_id, contact_id)`` is still the logical identity -- it is what the
    API addresses a row by, and the UNIQUE constraint enforces it. The
    surrogate ``id`` exists because ``extracted_facts.promoted_to_id`` is a
    single uuid: a human accepting a ``stakeholder`` fact had nowhere to record
    which link it became, this being the one promotion target whose primary key
    was a pair. Every other target already had one.
    """

    __tablename__ = "deal_contacts"
    __table_args__ = (
        UniqueConstraint("deal_id", "contact_id"),
        # The UNIQUE index above is deal_id-leading, so "who is on this deal?"
        # is covered. The reverse -- "which deals is this person on?", which is
        # GET /contacts/{id}/deals -- is not.
        Index("ix_deal_contacts_contact_id", "contact_id"),
        # At most one primary contact per deal; zero is fine. Without this,
        # "who do we talk to here?" is answered by whichever of two flagged
        # rows the planner happens to return, and that can change between two
        # identical requests.
        #
        # Partial uniqueness can only ever be an index, never a constraint, so
        # it cannot be DEFERRABLE -- a write setting is_primary must therefore
        # demote the incumbent in an *earlier statement*, not merely the same
        # transaction. See services.deal.set_primary_stakeholder.
        Index(
            "uq_deal_contacts_deal_id_primary",
            "deal_id",
            unique=True,
            postgresql_where=text("is_primary"),
        ),
    )

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
    )
    contact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contacts.id", ondelete="CASCADE"),
        nullable=False,
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
    # Who set `buying_role` and the rest: a human, or a promoted `stakeholder`
    # fact. The README carried this as a known gap. See migration 0011.
    origin: Mapped[Origin] = mapped_column(
        origin_enum, nullable=False, server_default=Origin.USER.value
    )


class DealStageHistory(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Append-only stage transitions. Feeds stalled-deal detection -- cheap to
    maintain, impossible to reconstruct later if skipped."""

    __tablename__ = "deal_stage_history"
    __table_args__ = (
        # Every read of this table is "one deal's transitions, in order" -- the
        # timeline, and the LEAD() that derives time-in-stage from it. A bare
        # deal_id index finds the rows and then sorts them in memory. Composite
        # and deal_id-leading, so it serves the plain lookup too. Mirrors
        # ix_activities_deal_id_occurred_at.
        Index("ix_deal_stage_history_deal_id_changed_at", "deal_id", "changed_at"),
    )

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
    )
    from_stage: Mapped[Optional[DealStage]] = mapped_column(deal_stage_enum, nullable=True)
    to_stage: Mapped[DealStage] = mapped_column(deal_stage_enum, nullable=False)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

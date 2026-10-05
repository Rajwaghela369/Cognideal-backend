"""The four tables holding model assertions about a deal.

Every row here carries ``origin``, ``confidence`` and ``status``. Without
``origin`` you can never answer "how many of these eight commitments did the
model write?", which is the first question anyone asks the moment they stop
trusting the output.

``confidence`` is the *generator's self-report* -- a weak, poorly calibrated
signal. It is not a validation result. Verdicts live in ``claim_validations``,
and the UI must never render one as if it were the other.
"""

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    ActionType,
    DismissalReason,
    CommitmentStatus,
    FactStatus,
    FactType,
    Origin,
    OwnerSide,
    Priority,
    RecommendationStatus,
    RiskStatus,
    RiskType,
    Severity,
    check_in,
    commitment_status_enum,
    fact_status_enum,
    origin_enum,
    owner_side_enum,
    priority_enum,
    recommendation_status_enum,
    risk_status_enum,
    severity_enum,
)


class ExtractedFact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Raw extraction output -- the intake queue, not the finished record.

    Extraction always writes here with ``status='pending'``. A human accepting
    a fact (Gate 3) is what creates the ``commitments`` / ``tasks`` row, and
    ``promoted_to_type`` / ``promoted_to_id`` record where it went. That is why
    facts and commitments both exist without duplicating each other.
    """

    __tablename__ = "extracted_facts"
    __table_args__ = (check_in("fact_type", FactType, "fact_type"),)

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    meeting_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("meetings.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    fact_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(3, 2), nullable=True)
    status: Mapped[FactStatus] = mapped_column(
        fact_status_enum, nullable=False, server_default=FactStatus.PENDING.value
    )
    promoted_to_type: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    promoted_to_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Commitment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Something someone said they would do -- on either side of the table."""

    __tablename__ = "commitments"

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_fact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("extracted_facts.id", ondelete="SET NULL"),
        nullable=True,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    owner_side: Mapped[OwnerSide] = mapped_column(owner_side_enum, nullable=False)
    owner_contact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contacts.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Free text for a speaker not yet in `contacts` -- same reasoning as
    # meeting_attendees.raw_name.
    owner_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    due_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[CommitmentStatus] = mapped_column(
        commitment_status_enum,
        nullable=False,
        server_default=CommitmentStatus.PENDING.value,
    )
    origin: Mapped[Origin] = mapped_column(
        origin_enum, nullable=False, server_default=Origin.AI.value
    )
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(3, 2), nullable=True)


class Risk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "risks"
    __table_args__ = (
        check_in("risk_type", RiskType, "risk_type"),
        # Re-running the analyzer must bump last_seen_at, not insert a fifth
        # copy of "single-threaded". Without this the risk panel fills with
        # duplicates inside a week. Scoped to open risks so a resolved one can
        # legitimately recur later.
        Index(
            "uq_risks_deal_id_risk_type_open",
            "deal_id",
            "risk_type",
            unique=True,
            postgresql_where=text("status = 'open'"),
        ),
    )

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    risk_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # Empty string for the ten known types, a canonicalised slug for `other`.
    # Empty rather than NULL: a NULL in a unique index is distinct from every
    # other NULL, which would let two open rows of one type coexist and quietly
    # remove the guarantee. See migration 0013.
    risk_key: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    severity: Mapped[Severity] = mapped_column(severity_enum, nullable=False)
    status: Mapped[RiskStatus] = mapped_column(
        risk_status_enum, nullable=False, server_default=RiskStatus.OPEN.value
    )
    origin: Mapped[Origin] = mapped_column(
        origin_enum, nullable=False, server_default=Origin.AI.value
    )
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(3, 2), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    detector_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    first_detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Recommendation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Next best action. Accepting one creates a task, linked back via
    ``created_task_id``."""

    __tablename__ = "recommendations"
    __table_args__ = (
        check_in("action_type", ActionType, "action_type"),
        check_in("dismissal_reason", DismissalReason, "dismissal_reason"),
        Index("ix_recommendations_source_risk_id", "source_risk_id"),
        # One live suggestion per risk. The detector runs repeatedly, and
        # without this it re-creates the same advice every pass.
        #
        # Keyed on the risk, NOT on action_type: two risks legitimately share
        # one action_type -- "engage the economic buyer" and "broaden beyond
        # one contact" are both `engage_stakeholder` but are different advice,
        # and keying on action_type silently dropped the second.
        #
        # Scoped to `suggested` so a dismissal does not block a fresh
        # suggestion later, and to source_risk_id IS NOT NULL so proactive
        # recommendations -- which have no risk -- are not capped at one.
        Index(
            "uq_recommendations_deal_id_source_risk_id_suggested",
            "deal_id",
            "source_risk_id",
            unique=True,
            postgresql_where=text(
                "status = 'suggested' AND source_risk_id IS NOT NULL"
            ),
        ),
        Index("ix_recommendations_source_commitment_id", "source_commitment_id"),
        # The same guarantee for `correct_record` proposals, which have no
        # source risk and so fall outside the index above. Without it stage 7
        # files a duplicate on every analysis run that sees the commitment.
        Index(
            "uq_recommendations_deal_id_source_commitment_id_suggested",
            "deal_id",
            "source_commitment_id",
            unique=True,
            postgresql_where=text(
                "status = 'suggested' AND source_commitment_id IS NOT NULL"
            ),
        ),
    )

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # The risk this answers. Nullable because a *proactive* recommendation has
    # none -- "their fiscal year ends in six weeks, push to close" is good
    # advice that is not fixing anything.
    #
    # SET NULL rather than CASCADE: if the risk goes, an accepted
    # recommendation and the task it created are still real work.
    #
    # Without this column there is no link between risks and recommendations at
    # all -- they shared only their evidence rows, which is a terrible thing to
    # join a UI on. Follows tasks.source_fact_id and commitments.source_fact_id.
    source_risk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("risks.id", ondelete="SET NULL"),
        nullable=True,
    )
    # The commitment a `correct_record` proposal is about. Null for every other
    # action type. SET NULL for the same reason as source_risk_id, and for one
    # more: claim_evidence.claim_id has no foreign key, so a CASCADE that
    # deleted recommendation rows would orphan their evidence links behind
    # services/claims.py's back. Migration 0015.
    source_commitment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("commitments.id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    priority: Mapped[Priority] = mapped_column(
        priority_enum, nullable=False, server_default=Priority.MEDIUM.value
    )
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(3, 2), nullable=True)
    status: Mapped[RecommendationStatus] = mapped_column(
        recommendation_status_enum,
        nullable=False,
        server_default=RecommendationStatus.SUGGESTED.value,
    )
    origin: Mapped[Origin] = mapped_column(
        origin_enum, nullable=False, server_default=Origin.AI.value
    )
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    detector_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    decided_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_task_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Structured so dismissals can be counted; the note carries the detail.
    dismissal_reason: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    dismissal_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    ClaimType,
    SourceKind,
    ValidationMethod,
    Verdict,
    VerificationStatus,
    claim_type_enum,
    source_kind_enum,
    validation_method_enum,
    verdict_enum,
    verification_status_enum,
)


class Evidence(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A locatable span backing a claim -- not a description of one.

    Two flavours:

    *   ``source_kind='document'`` -- ``chunk_id`` plus ``char_start``/
        ``char_end`` point at an exact stretch of a transcript or contract.
    *   ``source_kind='record'`` -- ``record_ref`` points at a field in our own
        database, e.g. ``{"table": "deals", "id": "...",
        "field": "expected_close_date"}``.

    The record flavour is not an afterthought. Most risk detection reasons over
    structured state ("stage has not moved in 58 days", "no economic buyer has
    attended a meeting"), not over quotes. Restricted to document spans, every
    one of those risks would render uncited and read as a hunch.

    Evidence rows are immutable, like the chunks they point at.
    """

    __tablename__ = "evidence"
    __table_args__ = (
        CheckConstraint(
            "source_kind <> 'document' OR chunk_id IS NOT NULL",
            name="document_needs_chunk",
        ),
        CheckConstraint(
            "source_kind <> 'record' OR record_ref IS NOT NULL",
            name="record_needs_ref",
        ),
    )

    deal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_kind: Mapped[SourceKind] = mapped_column(source_kind_enum, nullable=False)
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
    )
    chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=True,
    )
    record_ref: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    char_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    speaker: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    occurred_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ClaimEvidence(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """The spine: which evidence backs which claim.

    Five tables hold model assertions -- extracted_facts, commitments, risks,
    recommendations, chat_messages. A plain foreign key cannot express the
    relationship, because one claim usually rests on several pieces of evidence
    *and* one piece of evidence supports several claims. Many-to-many on both
    sides needs a join table.

    ``claim_id`` deliberately has **no foreign key**: the row it points at
    lives in one of five tables, and ``claim_type`` says which. This is a
    polymorphic association, and the cost is real -- Postgres will not stop you
    orphaning rows.

    **Contract for callers:** whatever deletes a claim must delete its
    ``claim_evidence`` rows in the same transaction. Prefer soft-deleting
    claims (dismissed risks want that anyway). The ``evidence_id`` side is a
    real FK, so dropping a document cascades cleanly on its own.
    """

    __tablename__ = "claim_evidence"
    __table_args__ = (
        UniqueConstraint(
            "claim_type", "claim_id", "evidence_id", name="claim_evidence_link"
        ),
        Index("ix_claim_evidence_claim_type_claim_id", "claim_type", "claim_id"),
    )

    claim_type: Mapped[ClaimType] = mapped_column(claim_type_enum, nullable=False)
    claim_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # How strongly THIS piece supports THIS claim -- lets the UI sort citations
    # so the strongest quote shows first and marginal ones collapse behind
    # "show 2 more".
    relevance: Mapped[Optional[float]] = mapped_column(Numeric(3, 2), nullable=True)
    # Gate 0 outcome: does the cited span still exist and still say this?
    verification_status: Mapped[VerificationStatus] = mapped_column(
        verification_status_enum,
        nullable=False,
        server_default=VerificationStatus.UNVERIFIED.value,
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ClaimValidation(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Gate 1 verdicts. Append-only: one row per validation run.

    Keeping history is what lets you tell whether a prompt change improved or
    regressed grounding on the *same* claims.

    ``validator_version`` matters more than it looks -- after a validator
    prompt change every historical verdict came from a different judge, and
    without the column the coverage metric silently mixes two populations.
    """

    __tablename__ = "claim_validations"
    __table_args__ = (
        Index(
            "ix_claim_validations_claim_type_claim_id_checked_at",
            "claim_type",
            "claim_id",
            "checked_at",
        ),
    )

    claim_type: Mapped[ClaimType] = mapped_column(claim_type_enum, nullable=False)
    claim_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    verdict: Mapped[Verdict] = mapped_column(verdict_enum, nullable=False)
    method: Mapped[ValidationMethod] = mapped_column(
        validation_method_enum, nullable=False
    )
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    validator_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

"""Commitments -- promises made, by either side.

Distinct from `tasks`, and the distinction is `owner_side`. A task is work *you*
committed to; a commitment is a promise tracked, and the valuable half is
`owner_side='customer'` -- "they will send the SOC 2 report by Friday" is not
something you can do, but it is something that going unmet kills the deal.
`MISSED_COMMITMENT` detects exactly that.

Human-entered in MVP. Once extraction exists a `commitment` fact accepted by a
person promotes into this table, which is why `source_fact_id` and `confidence`
are absent from every request model -- the promotion path owns them, along with
`origin`.
"""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.models.enums import CommitmentStatus, OwnerSide
from app.schemas.common import ORM, WRITE

# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class CommitmentListItem(BaseModel):
    model_config = ORM

    id: uuid.UUID
    description: str
    owner_side: str
    owner_contact_id: Optional[uuid.UUID] = None
    # The name as given. Snapshotted like meeting_attendees.raw_name, so the row
    # degrades to "someone called Dana promised this" rather than becoming
    # anonymous if the contact is deleted.
    owner_name: Optional[str] = None
    owner_contact_name: Optional[str] = None
    due_date: Optional[date] = None
    status: str
    origin: str
    confidence: Optional[Decimal] = None
    # Pending and past due. Computed -- a stored flag is wrong the day after it
    # is written. Undated is unscheduled, not late.
    is_overdue: bool = False
    source_fact_id: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


class CommitmentCreate(BaseModel):
    model_config = WRITE

    description: str = Field(min_length=1)
    owner_side: OwnerSide
    # Either a known contact or just a name. Both may be absent for an
    # organisation-level promise ("their legal team will review by Friday").
    owner_contact_id: Optional[uuid.UUID] = None
    owner_name: Optional[str] = Field(default=None, max_length=200)
    due_date: Optional[date] = None
    status: CommitmentStatus = CommitmentStatus.PENDING

    @model_validator(mode="after")
    def _owner_is_identifiable(self) -> "CommitmentCreate":
        """A customer promise with no owner at all is untrackable.

        Our own side can be unowned -- "we will send the questionnaire" is the
        team -- but "they will send it" with nobody named cannot be chased, and
        MISSED_COMMITMENT would have no one to point at.
        """
        if (
            self.owner_side == OwnerSide.CUSTOMER
            and self.owner_contact_id is None
            and not self.owner_name
        ):
            raise ValueError(
                "a customer commitment needs owner_contact_id or owner_name -- "
                "a promise with nobody attached cannot be followed up"
            )
        return self


class CommitmentUpdate(BaseModel):
    """Partial update, read with ``model_dump(exclude_unset=True)``.

    `owner_side` is absent: flipping it turns "they owe us" into "we owe them",
    which is a different promise. Delete and recreate.
    """

    model_config = WRITE

    description: Optional[str] = Field(default=None, min_length=1)
    owner_contact_id: Optional[uuid.UUID] = None
    owner_name: Optional[str] = Field(default=None, max_length=200)
    due_date: Optional[date] = None
    status: Optional[CommitmentStatus] = None


class CommitmentFilters(BaseModel):
    """Query parameters of GET /deals/{deal_id}/commitments."""

    model_config = WRITE

    status: List[CommitmentStatus] = Field(default_factory=list)
    owner_side: Optional[OwnerSide] = None
    overdue: Optional[bool] = Field(
        default=None,
        description="Pending and past due. An overdue customer commitment is "
        "what MISSED_COMMITMENT will detect once extraction exists.",
    )
    due_before: Optional[date] = None
    due_after: Optional[date] = None
    q: Optional[str] = Field(default=None, description="Matches the description")
    sort: str = "due_date"

    @model_validator(mode="after")
    def _range_is_the_right_way_round(self) -> "CommitmentFilters":
        if (
            self.due_after is not None
            and self.due_before is not None
            and self.due_after > self.due_before
        ):
            raise ValueError("due_after cannot be later than due_before")
        return self

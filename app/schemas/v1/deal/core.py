"""The /deals contract, both directions.

Responses first, then the request bodies and query parameters. They live in one
file because they are read together: POST /deals takes DealCreate and answers
with DealDetail, and GET /deals takes DealFilters and answers with
Page[DealListItem]. Splitting by direction would mean opening two files to
understand one endpoint.

Sub-resources are siblings in this package -- stakeholder.py,
stage_history.py -- mirroring api/routes/deals/.

Columns absent from every request model below, and why:

    id                  server-generated (gen_random_uuid)
    risk_level          a rollup of open risks -- the analyzer owns it
    closed_at           derived from the stage transition
    last_activity_at    touched by meeting / document / task writes
    days_in_stage       derived per request from deal_stage_history
    created_at          server
    updated_at          server
"""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.enums import DealStage, RiskLevel
from app.schemas.v1.account import AccountRef
from app.schemas.common import ORM, WRITE


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class DealListItem(BaseModel):
    """One row of the pipeline table."""

    model_config = ORM

    id: uuid.UUID
    name: str
    account_id: uuid.UUID
    account_name: str
    value: Optional[Decimal] = None
    currency: str
    stage: str
    win_probability: Optional[int] = None
    risk_level: Optional[str] = None
    expected_close_date: Optional[date] = None
    last_activity_at: Optional[datetime] = None
    # Derived from the newest deal_stage_history row. Not the same signal as
    # last_activity_at: that one is "has anyone talked to them", this is "is
    # the deal moving".
    days_in_stage: int = 0
    # Derived from the oldest open task -- `deals` stores no next_action.
    next_action: Optional[str] = None
    next_action_due_date: Optional[date] = None


class DealCounts(BaseModel):
    open_risks: int
    open_tasks: int
    open_commitments: int
    stakeholders: int
    meetings: int
    documents: int


class DealDetail(BaseModel):
    model_config = ORM

    id: uuid.UUID
    name: str
    account: AccountRef
    value: Optional[Decimal] = None
    currency: str
    stage: str
    win_probability: Optional[int] = None
    risk_level: Optional[str] = None
    expected_close_date: Optional[date] = None
    closed_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    days_in_stage: int = 0
    next_action: Optional[str] = None
    next_action_due_date: Optional[date] = None
    counts: DealCounts
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------
# Requests -- bodies and query parameters
#
# Enums are typed as the enums themselves rather than `str` (as the responses
# above are), so a bad value inbound is a 422 listing the legal ones and
# OpenAPI hands the frontend the vocabulary.
# --------------------------------------------------------------------------


class DealCreate(BaseModel):
    model_config = WRITE

    account_id: uuid.UUID
    name: str = Field(min_length=1, max_length=255)
    # Required. There is no defensible default -- a deal created at the wrong
    # stage silently corrupts both the pipeline view and stage history.
    stage: DealStage
    value: Optional[Decimal] = Field(
        default=None, ge=0, max_digits=14, decimal_places=2
    )
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    # Mirrors the win_probability_range CHECK on the model. Duplicated on
    # purpose: the constraint is the guarantee, this is the error message.
    win_probability: Optional[int] = Field(default=None, ge=0, le=100)
    expected_close_date: Optional[date] = None


class DealUpdate(BaseModel):
    """Partial update. Read it with ``model_dump(exclude_unset=True)``.

    Never ``exclude_none``: Pydantic tracks which keys the client actually
    sent, so ``{"expected_close_date": null}`` (clear it) and an omitted
    ``expected_close_date`` (leave it alone) are distinguishable -- but only
    through ``exclude_unset``. ``exclude_none`` collapses both into "do
    nothing" and makes every nullable column permanently unclearable.

    ``account_id`` is absent: contacts belong to accounts, so re-parenting a
    deal would strand every one of its ``deal_contacts`` rows. Immutable after
    create.
    """

    model_config = WRITE

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    value: Optional[Decimal] = Field(
        default=None, ge=0, max_digits=14, decimal_places=2
    )
    currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")
    stage: Optional[DealStage] = None
    win_probability: Optional[int] = Field(default=None, ge=0, le=100)
    expected_close_date: Optional[date] = None
    # Not a column on `deals`. Writing `stage` is not a column update -- it
    # also appends to deal_stage_history, and that row wants a note. This is
    # the only field here whose meaning depends on another being present.
    stage_note: Optional[str] = None

    @model_validator(mode="after")
    def _stage_note_needs_a_stage(self) -> "DealUpdate":
        if "stage_note" in self.model_fields_set and "stage" not in self.model_fields_set:
            raise ValueError("stage_note is only meaningful alongside a stage change")
        return self


# The vocabulary of sort keys, defined here so the request contract owns it.
# routes/deals.py maps each name to its SQL expression and asserts at import
# time that the two agree -- the mapping has to live next to the columns, but
# the list of legal values belongs with the model that validates them.
DEAL_SORT_KEYS = frozenset(
    {
        "name",
        "value",
        "stage",
        "risk",
        "days_in_stage",
        "expected_close_date",
        "last_activity_at",
        "created_at",
    }
)


class DealFilters(BaseModel):
    """Every query parameter of GET /deals, as one model.

    Bound with ``Annotated[DealFilters, Query()]``, which is what keeps these
    query parameters rather than a request body -- a bare Pydantic model in a
    handler signature is a *body*, and a GET with a body is not something
    clients or proxies handle reliably.
    """

    model_config = WRITE

    account_id: Optional[uuid.UUID] = None
    contact_id: Optional[uuid.UUID] = Field(
        default=None, description="Deals this person is a stakeholder on"
    )
    stage: List[DealStage] = Field(default_factory=list)
    risk_level: List[RiskLevel] = Field(default_factory=list)
    open: Optional[bool] = Field(
        default=None,
        description="true = open pipeline, false = closed. Shorthand for the "
        "stage filter so callers need not enumerate the closed stages -- and "
        "need not remember to revisit that list when a stage is added.",
    )
    q: Optional[str] = Field(default=None, description="Matches deal or account name")
    value_min: Optional[Decimal] = Field(default=None, ge=0)
    value_max: Optional[Decimal] = Field(default=None, ge=0)
    close_before: Optional[date] = None
    close_after: Optional[date] = None
    stale_days: Optional[int] = Field(
        default=None, ge=0, description="No activity in the last N days"
    )
    stalled_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Has not changed stage in the last N days. Not the same as "
        "stale_days: a deal with weekly check-ins and no stage movement for two "
        "months is invisible to that filter and is exactly what this one finds.",
    )
    stalled: Optional[bool] = Field(
        default=None,
        description="Stuck by the per-stage thresholds in "
        "expressions.STALL_THRESHOLD_DAYS, rather than one flat number -- a "
        "security review at 40 days is normal where a discovery deal at 40 days "
        "is not. Use stalled_days instead to ask with your own threshold.",
    )
    sort: str = "-last_activity_at"

    @field_validator("sort")
    @classmethod
    def _known_sort_key(cls, value: str) -> str:
        if value.lstrip("-") not in DEAL_SORT_KEYS:
            raise ValueError(
                f"unknown sort key; valid: {', '.join(sorted(DEAL_SORT_KEYS))}"
            )
        return value

    @model_validator(mode="after")
    def _ranges_are_the_right_way_round(self) -> "DealFilters":
        """An inverted range returns zero rows and looks like "no such deals".

        Neither pair was checked before, so ?value_min=100000&value_max=1000
        answered 200 with an empty page -- indistinguishable from a correct
        query that simply matched nothing.
        """
        if (
            self.value_min is not None
            and self.value_max is not None
            and self.value_min > self.value_max
        ):
            raise ValueError("value_min cannot exceed value_max")
        if (
            self.close_after is not None
            and self.close_before is not None
            and self.close_after > self.close_before
        ):
            raise ValueError("close_after cannot be later than close_before")
        return self

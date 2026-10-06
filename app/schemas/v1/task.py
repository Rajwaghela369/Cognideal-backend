"""The /tasks contract, both directions.

A flat module rather than a package, unlike routes_models/deal/: tasks have no
sub-resources. Promote it to a package if they ever gain one.

Tasks are addressed top-level even though ``tasks.deal_id`` is NOT NULL. The
tasks page is a cross-deal table, which a route nested under /deals cannot
serve; ``?deal_id=`` covers the per-deal case, so there is one collection
rather than two.

Columns absent from every request model below, and why:

    id              server-generated (gen_random_uuid)
    completed_at    derived from the status change
    origin          anything created through this API is `user` by definition;
                    an `ai` task arrives by promotion from an accepted fact,
                    not by POST
    source_fact_id  set only by that promotion path (Gate 3)
    created_at      server
    updated_at      server

``source_recommendation_id`` / ``source_recommendation`` are absent from the
*model* as well as the requests: they are **derived, not stored**. The edge
already exists as ``recommendations.created_task_id``, and a second column on
``tasks`` pointing the other way would be a duplicate of one relationship that
could drift -- the same reason ``is_completed``, ``is_overdue`` and
``days_in_stage`` are computed rather than kept. See ``routes/tasks.py`` for how
it is read, and why it is a correlated subquery rather than a join.
"""

import uuid
from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.enums import Origin, Priority, TaskStatus
from app.schemas.common import ListQuery, ORM, WRITE
from app.schemas.v1.deal.risk import RecommendationRef

# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class TaskListItem(BaseModel):
    """One row of the tasks table.

    Carries the deal and account names because this list is cross-deal:
    "Send security questionnaire, due 3 days ago" is not actionable without
    knowing which deal it belongs to.
    """

    model_config = ORM

    id: uuid.UUID
    deal_id: uuid.UUID
    deal_name: str
    account_id: uuid.UUID
    account_name: str
    title: str
    due_date: Optional[date] = None
    status: str
    priority: str
    origin: str
    # Open and past due. Computed, not stored -- a stored flag would be wrong
    # every day after the one it was written on.
    is_overdue: bool = False
    # The recommendation a human accepted to create this task, if any. Derived
    # from `recommendations.created_task_id`; the id alone, because the list
    # needs only enough to link. `origin='ai'` is not a substitute: a task
    # promoted from a fact is also `ai` and has no recommendation.
    source_recommendation_id: Optional[uuid.UUID] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class TaskDetail(BaseModel):
    model_config = ORM

    id: uuid.UUID
    deal_id: uuid.UUID
    deal_name: str
    account_id: uuid.UUID
    account_name: str
    title: str
    description: Optional[str] = None
    due_date: Optional[date] = None
    status: str
    priority: str
    origin: str
    is_overdue: bool = False
    # Present when a human accepted an extracted fact and this task is what it
    # became. Read-only here: the promotion path owns it.
    source_fact_id: Optional[uuid.UUID] = None
    # The suggestion this task came from. A different provenance path from
    # `source_fact_id` above: that one is Gate 3 promoting a fact, this one is
    # a human accepting a recommendation. Both can be null and they are not
    # alternatives -- a task can have neither.
    source_recommendation: Optional[RecommendationRef] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------
# Requests -- bodies and query parameters
#
# Enums are typed as the enums themselves rather than `str` (as the responses
# above are), so a bad value inbound is a 422 listing the legal ones and
# OpenAPI hands the frontend the vocabulary.
# --------------------------------------------------------------------------


class TaskCreate(BaseModel):
    model_config = WRITE

    deal_id: uuid.UUID
    title: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    due_date: Optional[date] = None
    priority: Priority = Priority.MEDIUM
    # Accepted so an already-finished piece of work can be logged in one call.
    # Routed through the same status handling as PATCH, so completed_at is set
    # rather than left null on a task created as `done`.
    status: TaskStatus = TaskStatus.OPEN


class TaskUpdate(BaseModel):
    """Partial update. Read it with ``model_dump(exclude_unset=True)``.

    Never ``exclude_none`` -- that collapses "clear the due date" and "leave
    the due date alone" into one thing and makes the column unclearable.

    ``deal_id`` is absent: moving a task between deals would silently change
    which deal's "next action" it is. Delete and recreate instead.
    """

    model_config = WRITE

    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    due_date: Optional[date] = None
    priority: Optional[Priority] = None
    status: Optional[TaskStatus] = None


TASK_SORT_KEYS = frozenset(
    {
        "due_date",
        "priority",
        "status",
        "title",
        "deal_name",
        "created_at",
    }
)


class TaskFilters(ListQuery):
    """Every query parameter of GET /tasks.

    Bound with ``Annotated[TaskFilters, Query()]`` so these stay query
    parameters rather than becoming a request body.
    """

    model_config = WRITE

    deal_id: Optional[uuid.UUID] = None
    account_id: Optional[uuid.UUID] = Field(
        default=None, description="Every task across this account's deals"
    )
    status: List[TaskStatus] = Field(default_factory=list)
    priority: List[Priority] = Field(default_factory=list)
    origin: Optional[Origin] = Field(
        default=None, description="Who authored the task -- user or ai"
    )
    open: Optional[bool] = Field(
        default=None,
        description="true = status is open, false = done or cancelled. "
        "Shorthand so callers need not enumerate the closed statuses.",
    )
    overdue: Optional[bool] = Field(
        default=None,
        description="Open and past its due date. A task with no due date is "
        "never overdue -- undated is unscheduled, not late.",
    )
    due_before: Optional[date] = None
    due_after: Optional[date] = None
    has_due_date: Optional[bool] = Field(
        default=None, description="false finds the unscheduled backlog"
    )
    q: Optional[str] = Field(default=None, description="Matches title or description")
    # due_date first, nulls last -- deliberately the same ordering as
    # expressions.NEXT_ACTION, so the top row of ?deal_id=X&open=true is always
    # that deal's next action. If the two diverge, the deal page and the task
    # table disagree about what is next.
    sort: str = "due_date"

    @field_validator("sort")
    @classmethod
    def _known_sort_key(cls, value: str) -> str:
        if value.lstrip("-") not in TASK_SORT_KEYS:
            raise ValueError(
                f"unknown sort key; valid: {', '.join(sorted(TASK_SORT_KEYS))}"
            )
        return value

    @model_validator(mode="after")
    def _range_is_the_right_way_round(self) -> "TaskFilters":
        if (
            self.due_after is not None
            and self.due_before is not None
            and self.due_after > self.due_before
        ):
            raise ValueError("due_after cannot be later than due_before")
        return self

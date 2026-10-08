"""Tasks -- the cross-deal task table.

A flat module, not a package like routes/deals/: tasks have no sub-resources.

Top-level rather than nested under /deals even though every task has a deal.
The tasks page is one table across the whole pipeline, which a nested route
cannot serve, and ?deal_id= covers the per-deal case -- so there is one
collection instead of two that drift.
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from typing_extensions import Annotated

from app.api.deps import paginate
from app.queries import IS_OVERDUE, PRIORITY_ORDER
from app.db.session import get_db
from app.models import Account, Deal, Recommendation, Task
from app.models.enums import TaskStatus
from app.schemas.common import Page
from app.schemas.v1.task import (
    TASK_SORT_KEYS,
    TaskCreate,
    TaskDetail,
    TaskFilters,
    TaskListItem,
    TaskUpdate,
)
from app.services import task as task_service
from app.services import analysis as analysis_service

router = APIRouter(prefix="/tasks", tags=["tasks"])

# Statuses that mean "no longer to be done". Mirrors CLOSED_STAGES on deals --
# `open=false` must not make callers enumerate these, and must not need
# revisiting here when a fourth status appears.
CLOSED_STATUSES = (TaskStatus.DONE.value, TaskStatus.CANCELLED.value)

# Sort keys mapped to their expression. The vocabulary lives in TASK_SORT_KEYS,
# which validates the request; the mapping lives here because it names columns.
_SORTS = {
    "due_date": Task.due_date,
    "priority": PRIORITY_ORDER,
    "status": Task.status,
    "title": Task.title,
    "deal_name": Deal.name,
    "created_at": Task.created_at,
}

assert set(_SORTS) == set(TASK_SORT_KEYS), (
    f"_SORTS and TASK_SORT_KEYS disagree: {set(_SORTS) ^ set(TASK_SORT_KEYS)}"
)

# The recommendation a human accepted to create this task, if any.
#
# A correlated scalar subquery rather than an outer join, and that is not a
# style choice. `recommendations.created_task_id` carries no unique constraint,
# so nothing stops two rows pointing at one task -- and `paginate()` runs the
# list statement twice, counted and windowed. A join would therefore duplicate
# the task *and* inflate `total`, which is the multiplication trap
# `list_accounts` documents for exactly this reason. `limit(1)` means a
# duplicate degrades to "picks one" instead of corrupting the page.
#
# Note the direction: risks.py joins Task to Recommendation safely, because it
# walks the foreign key forward (many recommendations -> one task). Only the
# reverse walk can multiply rows.
_SOURCE_RECOMMENDATION_ID = (
    select(Recommendation.id)
    .where(Recommendation.created_task_id == Task.id)
    .correlate(Task)
    .limit(1)
    .scalar_subquery()
    .label("source_recommendation_id")
)

# Columns every task read returns, joined up to the deal and account so the
# cross-deal table has something to label each row with.
_COLUMNS = (
    Task.id,
    Task.deal_id,
    Deal.name.label("deal_name"),
    Deal.account_id,
    Account.name.label("account_name"),
    Task.title,
    Task.due_date,
    Task.status,
    Task.priority,
    Task.origin,
    IS_OVERDUE.label("is_overdue"),
    _SOURCE_RECOMMENDATION_ID,
    Task.completed_at,
    Task.created_at,
)


def _base_stmt() -> Select:
    return select(*_COLUMNS).join(Deal, Deal.id == Task.deal_id).join(
        Account, Account.id == Deal.account_id
    )


def _task_list_stmt(f: TaskFilters) -> Select:
    """Build the filtered statement once -- paginate() runs it twice, counted
    and windowed, and filters applied in two places eventually disagree."""
    stmt = _base_stmt()

    if f.deal_id is not None:
        stmt = stmt.where(Task.deal_id == f.deal_id)
    if f.account_id is not None:
        stmt = stmt.where(Deal.account_id == f.account_id)
    if f.status:
        stmt = stmt.where(Task.status.in_(f.status))
    if f.priority:
        stmt = stmt.where(Task.priority.in_(f.priority))
    if f.origin is not None:
        stmt = stmt.where(Task.origin == f.origin)
    if f.open is not None:
        stmt = stmt.where(
            Task.status.notin_(CLOSED_STATUSES)
            if f.open
            else Task.status.in_(CLOSED_STATUSES)
        )
    if f.overdue is not None:
        stmt = stmt.where(IS_OVERDUE if f.overdue else ~IS_OVERDUE)
    if f.due_after is not None:
        stmt = stmt.where(Task.due_date >= f.due_after)
    if f.due_before is not None:
        stmt = stmt.where(Task.due_date <= f.due_before)
    if f.has_due_date is not None:
        stmt = stmt.where(
            Task.due_date.is_not(None) if f.has_due_date else Task.due_date.is_(None)
        )
    if f.q:
        pattern = f"%{f.q}%"
        stmt = stmt.where(
            or_(Task.title.ilike(pattern), Task.description.ilike(pattern))
        )

    descending = f.sort.startswith("-")
    column = _SORTS[f.sort.lstrip("-")]
    order = column.desc() if descending else column.asc()
    # nulls_last in either direction: an undated task is not the soonest and
    # not the latest, it is unscheduled. created_at breaks ties so paging is
    # stable when many tasks share a due date.
    return stmt.order_by(order.nulls_last(), Task.created_at.asc(), Task.id)


@router.get("", response_model=Page[TaskListItem])
async def list_tasks(
    filters: Annotated[TaskFilters, Query()],
    db: AsyncSession = Depends(get_db),
) -> Any:
    """The tasks table."""
    rows, total = await paginate(db, _task_list_stmt(filters), filters)
    # Raw Rows -- response_model validates them once on the way out.
    return {"items": rows, "total": total, "limit": filters.limit, "offset": filters.offset}


async def _task_detail(db: AsyncSession, task_id: uuid.UUID) -> Any:
    """One task, with the detail-only columns and its provenance.

    Built from `_base_stmt()` plus `add_columns` rather than a second hand-rolled
    select: the two used to repeat the Deal/Account joins and their own column
    lists, so anything added to `_COLUMNS` reached the list and silently missed
    the detail.
    """
    stmt = _base_stmt().add_columns(
        Task.description, Task.source_fact_id, Task.updated_at
    ).where(Task.id == task_id)
    row = (await db.execute(stmt)).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found"
        )

    # The id is already on the row; the detail view earns the rest of the
    # recommendation, because "why does this task exist?" is answered by its
    # rationale rather than by a uuid. A second small query rather than more
    # correlated subqueries: three more of those on every list row would be
    # paid by the table that does not use them.
    source_recommendation = None
    if row.source_recommendation_id is not None:
        source_recommendation = (
            await db.execute(
                select(
                    Recommendation.id,
                    Recommendation.title,
                    Recommendation.action_type,
                    Recommendation.rationale,
                ).where(Recommendation.id == row.source_recommendation_id)
            )
        ).first()

    return {**row._mapping, "source_recommendation": source_recommendation}


async def get_task_or_404(
    task_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Task:
    task = await db.get(Task, task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found"
        )
    return task


@router.get("/{task_id}", response_model=TaskDetail)
async def get_task(task_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Any:
    return await _task_detail(db, task_id)


@router.post("", response_model=TaskDetail, status_code=status.HTTP_201_CREATED)
async def create_task(
    body: TaskCreate,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> Any:
    # The one creation path, shared with the chat's confirmed drafts.
    task = await task_service.create_task(db, body)
    await db.commit()

    response.headers["Location"] = f"/tasks/{task.id}"
    return await _task_detail(db, task.id)


@router.patch("/{task_id}", response_model=TaskDetail)
async def update_task(
    body: TaskUpdate,
    task: Task = Depends(get_task_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    changes = body.model_dump(exclude_unset=True)
    to_status = changes.pop("status", None)

    for field, value in changes.items():
        setattr(task, field, value)
    if to_status is not None:
        # Not a plain column write: also sets or clears completed_at.
        await task_service.apply_status_change(db, task, to_status)

    task_id = task.id
    await db.commit()
    return await _task_detail(db, task_id)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task: Task = Depends(get_task_or_404), db: AsyncSession = Depends(get_db)
) -> Response:
    """Hard delete, for a task that should not have existed.

    Distinct from ``status='cancelled'``, which is the soft delete and is what
    "we decided not to do this" should use -- it keeps the record.

    Deleting a task promoted from an extracted fact also releases that fact
    back to `pending`; see services.task.release_source_fact for why leaving it
    `accepted` would strand it.
    """
    await task_service.release_source_fact(db, task)
    deal_id = task.deal_id
    await db.delete(task)
    await analysis_service.record_change(
        db,
        deal_id,
        "task deleted",
        tier1=True,
        tier2=False,
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

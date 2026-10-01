"""The deal record itself: pipeline list, detail, create, update, delete.

Sub-resources hanging off /deals/{deal_id} are siblings in this package --
stakeholders.py, stage_history.py -- each with its own router. __init__.py
assembles them.
"""

import uuid
from typing import Any
from datetime import date, datetime, timedelta, timezone
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.api.deps import Pagination, get_deal_or_404, paginate, pagination
from app.queries import (
    CLOSED_STAGES,
    DAYS_IN_STAGE,
    IS_STALLED,
    NEXT_ACTION,
    NEXT_ACTION_DUE_DATE,
    RISK_PRIORITY,
    STAGE_CHANGED_AT,
)
from app.db.session import get_db
from app.models import (
    Account,
    Commitment,
    Deal,
    DealContact,
    DealStageHistory,
    Document,
    Meeting,
    Risk,
    Task,
)
from app.schemas.v1.account import AccountRef
from app.schemas.common import Page
from app.schemas.v1.deal.core import (
    DEAL_SORT_KEYS,
    DealCounts,
    DealCreate,
    DealDetail,
    DealFilters,
    DealListItem,
    DealUpdate,
)
from app.services import claims as claims_service
from app.services import deal as deal_service

router = APIRouter(prefix="/deals", tags=["deals"])


# --------------------------------------------------------------------------
# Deals
# --------------------------------------------------------------------------

# Each sort key mapped to its expression. The mapping lives here because it
# names columns; the *vocabulary* lives in schemas as DEAL_SORT_KEYS, which is
# what validates the request. A whitelist either way rather than
# getattr(Deal, name), which would turn a query string into arbitrary column
# access and expose internals like closed_at as sortable by accident.
_SORTS = {
    "name": Deal.name,
    "value": Deal.value,
    "stage": Deal.stage,
    "risk": RISK_PRIORITY,
    "days_in_stage": DAYS_IN_STAGE,
    "expected_close_date": Deal.expected_close_date,
    "last_activity_at": Deal.last_activity_at,
    "created_at": Deal.created_at,
}


def _deal_list_stmt(f: DealFilters) -> Select:
    """Build the filtered statement once.

    paginate() runs this twice -- once wrapped in a count, once windowed. If the
    filters were applied in two places the total and the page could disagree,
    and a table would render "137 results" over a page drawn from a different
    set.
    """
    stmt = select(
        Deal.id,
        Deal.name,
        Deal.account_id,
        Account.name.label("account_name"),
        Deal.value,
        Deal.currency,
        Deal.stage,
        Deal.win_probability,
        Deal.risk_level,
        Deal.expected_close_date,
        Deal.last_activity_at,
        DAYS_IN_STAGE,
        NEXT_ACTION,
        NEXT_ACTION_DUE_DATE,
    ).join(Account, Account.id == Deal.account_id)

    if f.account_id is not None:
        stmt = stmt.where(Deal.account_id == f.account_id)
    if f.contact_id is not None:
        stmt = stmt.where(
            select(DealContact.id)
            .where(
                DealContact.deal_id == Deal.id,
                DealContact.contact_id == f.contact_id,
            )
            .exists()
        )
    if f.stage:
        stmt = stmt.where(Deal.stage.in_(f.stage))
    if f.risk_level:
        stmt = stmt.where(Deal.risk_level.in_(f.risk_level))
    if f.open is not None:
        stmt = stmt.where(
            Deal.stage.notin_(CLOSED_STAGES)
            if f.open
            else Deal.stage.in_(CLOSED_STAGES)
        )
    if f.q:
        pattern = f"%{f.q}%"
        stmt = stmt.where(or_(Deal.name.ilike(pattern), Account.name.ilike(pattern)))
    if f.value_min is not None:
        stmt = stmt.where(Deal.value >= f.value_min)
    if f.value_max is not None:
        stmt = stmt.where(Deal.value <= f.value_max)
    if f.close_after is not None:
        stmt = stmt.where(Deal.expected_close_date >= f.close_after)
    if f.close_before is not None:
        stmt = stmt.where(Deal.expected_close_date <= f.close_before)
    if f.stale_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=f.stale_days)
        # A deal with no activity at all is the stalest thing there is, so NULL
        # counts as stale rather than dropping out of the result.
        stmt = stmt.where(
            or_(Deal.last_activity_at < cutoff, Deal.last_activity_at.is_(None))
        )
    if f.stalled_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=f.stalled_days)
        # STAGE_CHANGED_AT coalesces to created_at, so this never silently
        # drops a deal that has no history rows yet.
        stmt = stmt.where(STAGE_CHANGED_AT < cutoff)
    if f.stalled is not None:
        stmt = stmt.where(IS_STALLED if f.stalled else ~IS_STALLED)

    descending = f.sort.startswith("-")
    column = _SORTS[f.sort.lstrip("-")]
    order = column.desc() if descending else column.asc()
    # nulls_last in either direction: a deal with no close date is not "the
    # soonest", and it is not the latest either -- it is simply unknown. The id
    # tiebreak keeps paging stable when the sort column has duplicates.
    return stmt.order_by(order.nulls_last(), Deal.id)


@router.get("", response_model=Page[DealListItem])
async def list_deals(
    filters: Annotated[DealFilters, Query()],
    page: Pagination = Depends(pagination),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """The pipeline table.

    Annotated[..., Query()] is what makes DealFilters a set of query
    parameters rather than a request body: a bare Pydantic model in a handler
    signature is a body, and a GET with a body is not reliably handled by
    clients or proxies. On the wire nothing changed -- ?q=acme&stalled=true
    still works exactly as it did.

    The one cost: OpenAPI describes this as a single `filters` parameter
    $ref-ing the DealFilters schema, rather than fourteen named query
    parameters. Requests are unaffected, but a generated client would get the
    serialisation wrong. Nothing here generates one today (the frontend has no
    codegen step), so this is a note for whoever adds it, not a live problem.

    limit/offset stay outside the model as the shared Pagination dependency:
    every list endpoint takes them, and they are not filters.
    """
    rows, total = await paginate(db, _deal_list_stmt(filters), page)
    # Raw Rows, not DealListItem instances. FastAPI validates the return value
    # against response_model with from_attributes=True, so building the models
    # here would mean every row is validated twice -- once by us, once by
    # FastAPI on the way out.
    return {"items": rows, "total": total, "limit": page.limit, "offset": page.offset}


# The sort vocabulary is validated in DealFilters, but the expressions live
# here -- so a key added to one and not the other would be a KeyError at
# request time, on whichever deployment happened to hit it first. Fail at
# import instead.
assert set(_SORTS) == set(DEAL_SORT_KEYS), (
    f"_SORTS and DEAL_SORT_KEYS disagree: {set(_SORTS) ^ set(DEAL_SORT_KEYS)}"
)


def _count_for_deal(model, *conditions) -> Select:
    """A correlated count of one child table for the deal in scope."""
    return (
        select(func.count())
        .select_from(model)
        .where(model.deal_id == Deal.id, *conditions)
        .correlate(Deal)
        .scalar_subquery()
    )


async def _deal_detail(db: AsyncSession, deal_id: uuid.UUID) -> DealDetail:
    """One round trip for the record, its account and all six counts.

    Six separate COUNT queries would be six round trips for what is already the
    most-visited screen in the product.
    """
    stmt = (
        select(
            Deal,
            Account.id.label("acct_id"),
            Account.name.label("acct_name"),
            Account.industry.label("acct_industry"),
            DAYS_IN_STAGE,
            NEXT_ACTION,
            NEXT_ACTION_DUE_DATE,
            _count_for_deal(Risk, Risk.status == "open").label("open_risks"),
            _count_for_deal(Task, Task.status == "open").label("open_tasks"),
            _count_for_deal(Commitment, Commitment.status == "pending").label(
                "open_commitments"
            ),
            _count_for_deal(DealContact).label("stakeholders"),
            _count_for_deal(Meeting).label("meetings"),
            _count_for_deal(Document).label("documents"),
        )
        .join(Account, Account.id == Deal.account_id)
        .where(Deal.id == deal_id)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Deal {deal_id} not found"
        )

    deal = row.Deal
    return DealDetail(
        id=deal.id,
        name=deal.name,
        account=AccountRef(
            id=row.acct_id, name=row.acct_name, industry=row.acct_industry
        ),
        value=deal.value,
        currency=deal.currency,
        stage=deal.stage,
        win_probability=deal.win_probability,
        risk_level=deal.risk_level,
        expected_close_date=deal.expected_close_date,
        closed_at=deal.closed_at,
        last_activity_at=deal.last_activity_at,
        days_in_stage=row.days_in_stage,
        next_action=row.next_action,
        next_action_due_date=row.next_action_due_date,
        counts=DealCounts(
            open_risks=row.open_risks,
            open_tasks=row.open_tasks,
            open_commitments=row.open_commitments,
            stakeholders=row.stakeholders,
            meetings=row.meetings,
            documents=row.documents,
        ),
        created_at=deal.created_at,
        updated_at=deal.updated_at,
    )


@router.get("/{deal_id}", response_model=DealDetail)
async def get_deal(deal_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> DealDetail:
    return await _deal_detail(db, deal_id)


@router.post("", response_model=DealDetail, status_code=status.HTTP_201_CREATED)
async def create_deal(
    body: DealCreate,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> DealDetail:
    # A bad account_id would otherwise surface as a foreign-key IntegrityError
    # at commit, which reaches the client as a 500.
    if await db.get(Account, body.account_id) is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Account {body.account_id} not found",
        )

    deal = Deal(**body.model_dump())
    db.add(deal)
    await db.flush()
    # The opening stage is a transition from nothing and belongs in the history
    # like any other. Without it the first entry for every deal is missing, and
    # "how long has it sat in this stage?" has no start date to measure from.
    # Written directly rather than through apply_stage_change, which is a no-op
    # when from and to agree -- and here the deal already holds its stage.
    db.add(
        DealStageHistory(
            deal_id=deal.id,
            from_stage=None,
            to_stage=deal.stage,
            note="Deal created",
        )
    )
    await db.commit()

    response.headers["Location"] = f"/deals/{deal.id}"
    return await _deal_detail(db, deal.id)


@router.patch("/{deal_id}", response_model=DealDetail)
async def update_deal(
    body: DealUpdate,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> DealDetail:
    # exclude_unset, never exclude_none -- see the DealUpdate docstring.
    changes = body.model_dump(exclude_unset=True)
    stage = changes.pop("stage", None)
    stage_note = changes.pop("stage_note", None)

    for field, value in changes.items():
        setattr(deal, field, value)
    if stage is not None:
        # Not a plain column write: also appends to deal_stage_history and
        # sets or clears closed_at.
        await deal_service.apply_stage_change(db, deal, stage, note=stage_note)

    await db.commit()
    return await _deal_detail(db, deal.id)


@router.delete("/{deal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deal(
    deal: Deal = Depends(get_deal_or_404), db: AsyncSession = Depends(get_db)
) -> Response:
    """Hard delete. Cascades to stakeholders, meetings, tasks, documents and
    their chunks.

    claim_evidence links are cleared first. claim_id carries no foreign key --
    it is polymorphic over five claim tables -- so without this the deal's
    risks, recommendations, commitments and facts all vanish while the links
    pointing at them survive, referencing nothing. See README section 4.1 and
    services/claims.py.
    """
    await claims_service.delete_links_for_deal(db, deal.id)
    await db.delete(deal)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

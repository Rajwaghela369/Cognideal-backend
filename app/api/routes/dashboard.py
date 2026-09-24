from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models import Account, Deal, Task
from app.schemas.dashboard import PriorityDeal

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

CLOSED_STAGES = ("closed_won", "closed_lost")

RISK_PRIORITY = case(
    (Deal.risk_level == "high", 0),
    (Deal.risk_level == "medium", 1),
    (Deal.risk_level == "low", 2),
    else_=3,
)

# `deals` deliberately has no next_action column -- it is derived from the
# oldest open task so the two can never drift. Soonest due date wins; tasks
# with no due date fall to the back.
NEXT_ACTION = (
    select(Task.title)
    .where(Task.deal_id == Deal.id, Task.status == "open")
    .order_by(Task.due_date.asc().nulls_last(), Task.created_at.asc())
    .limit(1)
    .correlate(Deal)
    .scalar_subquery()
    .label("next_action")
)


@router.get("/priority-deals", response_model=list[PriorityDeal])
async def get_priority_deals(
    within_days: Optional[int] = Query(
        default=None, ge=0, description="Only include deals closing within N days"
    ),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[PriorityDeal]:
    """Open deals needing intervention, ranked by risk level then close date.

    Ordering: high risk before medium before low; within the same risk
    tier, the soonest expected_close_date comes first (deals with none
    sort last).
    """
    stmt = (
        select(
            Deal.id,
            Deal.account_id,
            Account.name.label("account_name"),
            Deal.name,
            Deal.value,
            Deal.stage,
            Deal.risk_level,
            Deal.expected_close_date,
            NEXT_ACTION,
            Deal.last_activity_at,
        )
        .join(Account, Account.id == Deal.account_id)
        .where(Deal.stage.notin_(CLOSED_STAGES))
    )

    if within_days is not None:
        cutoff = date.today() + timedelta(days=within_days)
        stmt = stmt.where(Deal.expected_close_date.is_not(None),
            Deal.expected_close_date <= cutoff,)

    stmt = stmt.order_by(RISK_PRIORITY, Deal.expected_close_date.asc().nulls_last()).limit(
        limit
    )

    result = await db.execute(stmt)
    return [PriorityDeal.model_validate(row) for row in result.all()]

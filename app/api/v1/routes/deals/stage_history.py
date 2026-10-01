"""Stage transitions for one deal.

Read-only, and that is not an omission -- it is the point of the module, which
is why it is its own file rather than a tail section of deals.py.

The only legitimate writer is services.deal.apply_stage_change, reached through
PATCH /deals/{id} with a `stage`. It writes the history row, deals.stage and
closed_at in one transaction. An endpoint that appended here independently
would let the history say a deal reached negotiation while deals.stage still
said discovery -- the same drift the schema avoids by having no `status`
column, and a second write path into one invariant is how that drift gets in.

No PATCH or DELETE either: the table carries created_at and no updated_at
because a transition is a thing that happened. A fumbled stage change leaves
two honest rows rather than one edited one.
"""

from typing import Any, List

from fastapi import APIRouter, Depends
from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_deal_or_404
from app.db.session import get_db
from app.models import Deal, DealStageHistory
from app.schemas.v1.deal.stage_history import StageHistoryEntry

# Prefix stops at /deals: the handler below carries {deal_id} in its own
# path, and putting the parameter in both is a duplicated-param error.
router = APIRouter(prefix="/deals", tags=["stage history"])


@router.get("/{deal_id}/stage-history", response_model=List[StageHistoryEntry])
async def list_stage_history(
    deal: Deal = Depends(get_deal_or_404), db: AsyncSession = Depends(get_db)
) -> Any:
    """Every transition this deal has made, oldest first.

    Ascending, unlike the activity feed -- this reads as a progression, and
    days_in_stage only makes sense forwards.

    Unpaginated: there are seven stages, and even a thrashing deal produces a
    list that fits on one screen.
    """
    # Each row's successor, so time-in-stage is one pass rather than a
    # self-join. The newest row has no successor and is measured against now():
    # "still here, 58 days" is what the timeline renders, and a null would make
    # every client reimplement the subtraction.
    #
    # The id tiebreak matters -- a seeder writing a whole history in one
    # statement gives every row the same changed_at, and without it LEAD picks
    # arbitrarily between them.
    next_changed_at = func.lead(DealStageHistory.changed_at).over(
        partition_by=DealStageHistory.deal_id,
        order_by=(DealStageHistory.changed_at, DealStageHistory.id),
    )
    days_in_stage = (
        func.floor(
            func.extract(
                "epoch",
                func.coalesce(next_changed_at, func.now())
                - DealStageHistory.changed_at,
            )
            / 86400
        )
        .cast(Integer)
        .label("days_in_stage")
    )

    stmt = (
        select(
            DealStageHistory.id,
            DealStageHistory.from_stage,
            DealStageHistory.to_stage,
            DealStageHistory.changed_at,
            DealStageHistory.note,
            days_in_stage,
        )
        .where(DealStageHistory.deal_id == deal.id)
        .order_by(DealStageHistory.changed_at.asc(), DealStageHistory.id.asc())
    )
    return (await db.execute(stmt)).all()

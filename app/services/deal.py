"""Writes that are more than one statement, or that have ordering rules.

Kept out of the route modules for the same reason ``db/expressions.py`` keeps
the read expressions out: a stage change is three writes that must agree, and
the future seeder and fact-acceptance flow will need to make exactly the same
three. Two implementations of "what happens when a deal closes" is two
implementations that drift.
"""

import uuid
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.queries import CLOSED_STAGES
from app.models import Contact, Deal, DealContact, DealStageHistory
from app.models.enums import DealStage


async def apply_stage_change(
    db: AsyncSession, deal: Deal, to_stage: DealStage, note: Optional[str] = None
) -> None:
    """Move a deal to a new stage, with the two side-effects that implies.

    Writing ``deals.stage`` alone is never correct:

    *   ``deal_stage_history`` gets a row. It is append-only and feeds
        stalled-deal detection -- impossible to reconstruct if skipped.
    *   ``closed_at`` is set on entering a closed stage and *cleared* on
        leaving one. A deal reopened from closed_lost with a stale closed_at
        reads as closed to every query that checks the timestamp.

    A no-op stage write is ignored rather than logged, so a PATCH that merely
    echoes the current stage does not litter the history.
    """
    from_stage = deal.stage
    if from_stage == to_stage:
        return

    deal.stage = to_stage
    db.add(
        DealStageHistory(
            deal_id=deal.id,
            from_stage=from_stage,
            to_stage=to_stage,
            note=note,
        )
    )

    now_closed = to_stage.value in CLOSED_STAGES
    was_closed = from_stage.value in CLOSED_STAGES
    if now_closed and not was_closed:
        deal.closed_at = func.now()
    elif was_closed and not now_closed:
        deal.closed_at = None


async def demote_primary_stakeholder(
    db: AsyncSession, deal_id: uuid.UUID, keeping: Optional[uuid.UUID] = None
) -> None:
    """Clear is_primary on every other link for this deal.

    ``uq_deal_contacts_deal_id_primary`` is a *partial* unique index, and
    partial uniqueness cannot be expressed as a constraint -- only as an index.
    An index is checked at the end of each statement and can never be made
    DEFERRABLE. So this must run as its own statement *before* the row that
    claims the flag, not merely inside the same transaction. Reverse the order
    and every primary swap fails with an IntegrityError.
    """
    stmt = (
        update(DealContact)
        .where(DealContact.deal_id == deal_id, DealContact.is_primary.is_(True))
        .values(is_primary=False)
    )
    if keeping is not None:
        stmt = stmt.where(DealContact.contact_id != keeping)
    await db.execute(stmt)


async def assert_contact_on_deal_account(
    db: AsyncSession, deal: Deal, contact_id: uuid.UUID
) -> Contact:
    """Contacts belong to accounts, deals belong to accounts -- but nothing
    stops a link between a deal and a contact at a *different* company. No
    foreign key can express the rule, so the service layer is the only place it
    can live.
    """
    contact = await db.get(Contact, contact_id)
    if contact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contact {contact_id} not found",
        )
    if contact.account_id != deal.account_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Contact {contact_id} belongs to a different account than "
                f"deal {deal.id}"
            ),
        )
    return contact


async def get_stakeholder_or_404(
    db: AsyncSession, deal_id: uuid.UUID, contact_id: uuid.UUID
) -> DealContact:
    """Resolve a link by its logical identity -- the pair, never the surrogate
    id, which is internal and never leaves the database."""
    link = await db.scalar(
        select(DealContact).where(
            DealContact.deal_id == deal_id, DealContact.contact_id == contact_id
        )
    )
    if link is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contact {contact_id} is not a stakeholder on deal {deal_id}",
        )
    return link

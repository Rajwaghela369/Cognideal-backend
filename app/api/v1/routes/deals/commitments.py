"""Promises made on a deal, by either side.

Nested under the deal like meetings and risks: a commitment only means anything
in the context of the deal it was made about.
"""

import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from typing_extensions import Annotated

from app.api.deps import get_deal_or_404
from app.db.session import get_db
from app.models import Commitment, Contact, Deal
from app.models.enums import ClaimType, Origin
from app.queries import COMMITMENT_OVERDUE
from app.schemas.v1.deal.commitment import (
    CommitmentCreate,
    CommitmentFilters,
    CommitmentListItem,
    CommitmentUpdate,
)
from app.services import claims as claims_service
from app.services import analysis as analysis_service
from app.services import meeting as meeting_service

router = APIRouter(prefix="/deals/{deal_id}/commitments", tags=["commitments"])

_SORT_KEYS = {"due_date", "status", "owner_side", "created_at"}

_COLUMNS = (
    Commitment.id,
    Commitment.description,
    Commitment.owner_side,
    Commitment.owner_contact_id,
    Commitment.owner_name,
    (Contact.first_name + " " + Contact.last_name).label("owner_contact_name"),
    Commitment.due_date,
    Commitment.status,
    Commitment.origin,
    Commitment.confidence,
    COMMITMENT_OVERDUE.label("is_overdue"),
    Commitment.source_fact_id,
    Commitment.created_at,
    Commitment.updated_at,
)


def _base() -> Select:
    # outerjoin: a commitment may name an owner who is not a known contact, or
    # no individual at all ("their legal team will review by Friday").
    return select(*_COLUMNS).outerjoin(
        Contact, Contact.id == Commitment.owner_contact_id
    )


def _list_stmt(deal_id: uuid.UUID, f: CommitmentFilters) -> Select:
    stmt = _base().where(Commitment.deal_id == deal_id)

    if f.status:
        stmt = stmt.where(Commitment.status.in_(f.status))
    if f.owner_side is not None:
        stmt = stmt.where(Commitment.owner_side == f.owner_side)
    if f.overdue is not None:
        stmt = stmt.where(COMMITMENT_OVERDUE if f.overdue else ~COMMITMENT_OVERDUE)
    if f.due_after is not None:
        stmt = stmt.where(Commitment.due_date >= f.due_after)
    if f.due_before is not None:
        stmt = stmt.where(Commitment.due_date <= f.due_before)
    if f.q:
        stmt = stmt.where(Commitment.description.ilike(f"%{f.q}%"))

    descending = f.sort.startswith("-")
    column = getattr(Commitment, f.sort.lstrip("-"))
    order = column.desc() if descending else column.asc()
    return stmt.order_by(order.nulls_last(), Commitment.created_at.asc())


async def _one(db: AsyncSession, commitment_id: uuid.UUID) -> Any:
    return (
        await db.execute(_base().where(Commitment.id == commitment_id))
    ).first()


async def get_commitment_or_404(
    db: AsyncSession, deal_id: uuid.UUID, commitment_id: uuid.UUID
) -> Commitment:
    commitment = await db.scalar(
        select(Commitment).where(
            Commitment.id == commitment_id, Commitment.deal_id == deal_id
        )
    )
    if commitment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Commitment {commitment_id} not found on deal {deal_id}",
        )
    return commitment


@router.get("", response_model=List[CommitmentListItem])
async def list_commitments(
    filters: Annotated[CommitmentFilters, Query()],
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    if filters.sort.lstrip("-") not in _SORT_KEYS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown sort key. Valid: {', '.join(sorted(_SORT_KEYS))}",
        )
    return (await db.execute(_list_stmt(deal.id, filters))).all()


@router.post("", response_model=CommitmentListItem, status_code=status.HTTP_201_CREATED)
async def create_commitment(
    body: CommitmentCreate,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Log a promise.

    `origin='user'`: anything created here was typed by a person. An `ai`
    commitment arrives by a human accepting an extracted fact, not through this
    route -- which is why `source_fact_id`, `confidence` and `origin` are not in
    the request model.
    """
    if body.owner_contact_id is not None:
        # Contacts belong to accounts and commitments to deals, but nothing
        # stops an owner from a different company. No foreign key says this.
        await meeting_service.assert_contact_on_deal_account(
            db, deal, body.owner_contact_id
        )

    commitment = Commitment(deal_id=deal.id, origin=Origin.USER, **body.model_dump())
    db.add(commitment)
    await db.flush()
    await analysis_service.record_change(
        db,
        deal.id,
        "commitment created",
        table="commitments",
        row_id=commitment.id,
        fields=("status", "due_date"),
    )
    await db.commit()
    return await _one(db, commitment.id)


@router.patch("/{commitment_id}", response_model=CommitmentListItem)
async def update_commitment(
    commitment_id: uuid.UUID,
    body: CommitmentUpdate,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Partial update, including the status decision.

    Unlike deals, tasks, meetings and risks, `commitments` has no timestamp to
    maintain alongside `status` -- no `met_at` column exists. `updated_at` is
    the only record of when it was marked, which is imprecise but not worth a
    migration until something needs to measure it.
    """
    commitment = await get_commitment_or_404(db, deal.id, commitment_id)
    changes = body.model_dump(exclude_unset=True)

    if changes.get("owner_contact_id") is not None:
        await meeting_service.assert_contact_on_deal_account(
            db, deal, changes["owner_contact_id"]
        )
    for field, value in changes.items():
        setattr(commitment, field, value)

    relevant = set(changes) & {"status", "due_date"}
    if relevant:
        await analysis_service.record_change(
            db,
            deal.id,
            "commitment changed",
            table="commitments",
            row_id=commitment.id,
            fields=relevant,
        )
    await db.commit()
    return await _one(db, commitment_id)


@router.delete("/{commitment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_commitment(
    commitment_id: uuid.UUID,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Hard delete, for a commitment logged in error.

    `status='waived'` is the soft version and is what "we agreed to drop it"
    should use -- it keeps the record, which is what MISSED_COMMITMENT reasons
    over.

    Citation links are cleared first: `claim_evidence.claim_id` has no foreign
    key, so a commitment promoted from an extracted fact would otherwise leave
    links pointing at nothing.
    """
    commitment = await get_commitment_or_404(db, deal.id, commitment_id)
    await claims_service.delete_claim_links(db, ClaimType.COMMITMENT, [commitment_id])
    await db.delete(commitment)
    await analysis_service.record_change(
        db,
        deal.id,
        "commitment deleted",
        table="commitments",
        row_id=commitment_id,
        fields=("status", "due_date"),
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

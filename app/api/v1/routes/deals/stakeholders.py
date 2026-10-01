"""Stakeholders -- rows of `deal_contacts`, joined to the person they point at.

Named for the resource rather than the table. The URL is /stakeholders, the
schema is DealStakeholder, the service function is set_primary_stakeholder; a
module called deal_contacts.py would be the only place those disagree. And
routes/contacts.py is reserved for the real /contacts resource, whose schemas
are already written in schemas/contact.py.

A row here is addressed by its logical identity, the (deal_id, contact_id)
pair. The surrogate `id` added in 0004 exists only so extracted_facts can point
at the row, and never leaves the database.
"""

import uuid
from typing import Any, List

from typing_extensions import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.api.deps import get_deal_or_404
from app.db.session import get_db
from app.models import Contact, Deal, DealContact
from app.schemas.v1.deal.stakeholder import (
    DealStakeholder,
    DealStakeholderCreate,
    DealStakeholderFilters,
    DealStakeholderUpdate,
)
from app.services import deal as deal_service

router = APIRouter(prefix="/deals/{deal_id}/stakeholders", tags=["stakeholders"])


def _stakeholder_stmt(deal_id: uuid.UUID) -> Select:
    return (
        select(
            DealContact.contact_id,
            Contact.first_name,
            Contact.last_name,
            Contact.email,
            Contact.title,
            Contact.phone,
            DealContact.buying_role,
            DealContact.influence,
            DealContact.sentiment,
            DealContact.is_primary,
            DealContact.notes,
        )
        .join(Contact, Contact.id == DealContact.contact_id)
        .where(DealContact.deal_id == deal_id)
        # Primary contact first, then everyone else alphabetically -- the order
        # the stakeholder panel renders.
        .order_by(
            DealContact.is_primary.desc(),
            Contact.last_name.asc(),
            Contact.first_name.asc(),
        )
    )


async def _one_stakeholder(
    db: AsyncSession, deal_id: uuid.UUID, contact_id: uuid.UUID
) -> Any:
    """Re-read through the join after a write.

    The response is the link *joined to the contact*, so the ORM object just
    written cannot be serialised directly -- it carries no name or email.
    """
    row = (
        await db.execute(
            _stakeholder_stmt(deal_id).where(DealContact.contact_id == contact_id)
        )
    ).first()
    return row


@router.get("", response_model=List[DealStakeholder])
async def list_stakeholders(
    filters: Annotated[DealStakeholderFilters, Query()],
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Unpaginated on purpose -- a deal has a handful of stakeholders, and an
    envelope would cost the caller an unwrap for nothing.

    Annotated[..., Query()] keeps DealStakeholderFilters a set of query
    parameters; a bare model in the signature would be a request body.
    """
    stmt = _stakeholder_stmt(deal.id)
    if filters.buying_role:
        stmt = stmt.where(DealContact.buying_role.in_(filters.buying_role))
    if filters.is_primary is not None:
        stmt = stmt.where(DealContact.is_primary.is_(filters.is_primary))

    return (await db.execute(stmt)).all()


@router.post(
    "", response_model=DealStakeholder, status_code=status.HTTP_201_CREATED
)
async def add_stakeholder(
    body: DealStakeholderCreate,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> DealStakeholder:
    await deal_service.assert_contact_on_deal_account(db, deal, body.contact_id)

    # Read the id out now. rollback() expires every ORM object in the session --
    # expire_on_commit=False governs commit, not rollback -- so touching
    # deal.id afterwards fires a lazy refresh, which is blocking I/O inside the
    # async handler and dies with MissingGreenlet. The 409 below turns into a
    # 500 for exactly that reason if the attribute is read in the message.
    deal_id = deal.id

    payload = body.model_dump()
    if payload["is_primary"]:
        # Must precede the insert: the partial unique index is checked at the
        # end of each statement and cannot be deferred to commit.
        await deal_service.demote_primary_stakeholder(db, deal_id)

    db.add(DealContact(deal_id=deal_id, **payload))
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Contact {body.contact_id} is already a stakeholder on deal {deal_id}"
            ),
        )
    return await _one_stakeholder(db, deal_id, body.contact_id)


@router.patch("/{contact_id}", response_model=DealStakeholder)
async def update_stakeholder(
    contact_id: uuid.UUID,
    body: DealStakeholderUpdate,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> DealStakeholder:
    link = await deal_service.get_stakeholder_or_404(db, deal.id, contact_id)

    changes = body.model_dump(exclude_unset=True)
    if changes.get("is_primary"):
        await deal_service.demote_primary_stakeholder(db, deal.id, keeping=contact_id)
    for field, value in changes.items():
        setattr(link, field, value)

    await db.commit()
    return await _one_stakeholder(db, deal.id, contact_id)


@router.delete("/{contact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_stakeholder(
    contact_id: uuid.UUID,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Removes the link, never the contact -- the person still belongs to the
    account and may be a stakeholder on other deals."""
    link = await deal_service.get_stakeholder_or_404(db, deal.id, contact_id)
    await db.delete(link)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

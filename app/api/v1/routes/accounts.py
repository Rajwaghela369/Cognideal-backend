"""Accounts, and the contacts nested under them.

A flat module rather than a package: contacts are the only sub-resource and
they are small, so splitting this in two would mean two files of thirty lines
and an `__init__` to join them.

**Why this exists.** `DealCreate` requires an `account_id` and nothing in the
API could produce one, so a fresh install could not create its first deal
without psql -- and `DealStakeholderCreate` requires an existing `contact_id`,
which only the transcript-resolution path could mint. In practice that meant a
stakeholder had to have spoken on a recorded call before they could be tracked,
so the economic buyer could not be added until after the meeting you needed
them in.

Contacts are addressed as `/accounts/{account_id}/contacts` because
`contacts.account_id` is the foreign key -- deliberately not `deal_id`, since
one person can appear in several deals with the same company. Everything
deal-specific lives on `deal_contacts` and is reached through
`/deals/{id}/stakeholders`.
"""

import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated

from app.api.deps import paginate
from app.db.session import get_db
from app.models import Account, Contact, Deal
from app.schemas.common import Page
from app.schemas.v1.account import (
    AccountCreate,
    AccountFilters,
    AccountListItem,
    AccountResponse,
    AccountUpdate,
    ContactCreate,
    ContactFilters,
    ContactResponse,
    ContactUpdate,
)
from app.services import account as account_service

router = APIRouter(prefix="/accounts", tags=["accounts"])


# --------------------------------------------------------------------------
# Accounts
# --------------------------------------------------------------------------


@router.post("", response_model=AccountResponse, status_code=status.HTTP_201_CREATED)
async def create_account(
    body: AccountCreate, db: AsyncSession = Depends(get_db)
) -> Any:
    """Create an account.

    No uniqueness check on `name`: two real companies can share one, and
    `accounts` carries no unique constraint on it. Deduplication is the
    caller's judgement, which is why the list endpoint takes `?q=`.
    """
    account = Account(**body.model_dump())
    db.add(account)
    await db.commit()
    await db.refresh(account)
    return account


@router.get("", response_model=Page[AccountListItem])
async def list_accounts(
    filters: Annotated[AccountFilters, Query()],
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Accounts with their deal and contact counts, newest first.

    The counts are correlated scalar subqueries rather than joins with a
    GROUP BY: two independent one-to-many relationships joined in one statement
    multiply each other, and the contact count would come back wrong by a
    factor of the deal count.
    """
    deal_count = (
        select(func.count())
        .select_from(Deal)
        .where(Deal.account_id == Account.id)
        .correlate(Account)
        .scalar_subquery()
    )
    contact_count = (
        select(func.count())
        .select_from(Contact)
        .where(Contact.account_id == Account.id)
        .correlate(Account)
        .scalar_subquery()
    )

    stmt = select(
        Account.id,
        Account.name,
        Account.industry,
        Account.website,
        Account.employee_band,
        Account.hq_region,
        deal_count.label("deal_count"),
        contact_count.label("contact_count"),
    ).order_by(Account.created_at.desc())

    if filters.q:
        stmt = stmt.where(Account.name.ilike(f"%{filters.q}%"))

    rows, total = await paginate(db, stmt, filters)
    return Page[AccountListItem](
        items=[AccountListItem.model_validate(row) for row in rows],
        total=total,
        limit=filters.limit,
        offset=filters.offset,
    )


@router.get("/{account_id}", response_model=AccountResponse)
async def get_account(
    account_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Any:
    return await account_service.get_account_or_404(db, account_id)


@router.patch("/{account_id}", response_model=AccountResponse)
async def update_account(
    account_id: uuid.UUID,
    body: AccountUpdate,
    db: AsyncSession = Depends(get_db),
) -> Any:
    account = await account_service.get_account_or_404(db, account_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(account, field, value)
    await db.commit()
    await db.refresh(account)
    return account


@router.delete("/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    account_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Response:
    """Delete an account that has no deals.

    Refused rather than cascaded. The foreign keys *would* take the deals with
    it, and their meetings, documents, facts, risks and every piece of evidence
    behind them -- and `claim_evidence.claim_id` has no foreign key, so those
    links would survive as orphans pointing at rows that no longer exist (the
    trap `services/claims.py` exists for). Deleting the deals first makes the
    scale of what is being destroyed visible, one deal at a time.
    """
    await account_service.get_account_or_404(db, account_id)

    deals = await db.scalar(
        select(func.count()).select_from(Deal).where(Deal.account_id == account_id)
    )
    if deals:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Account has {deals} deal(s). Delete them first -- cascading "
                f"here would take their meetings, documents and evidence with it."
            ),
        )

    # Contacts do cascade: a contact with no deals on the account has nothing
    # pointing at it except `deal_contacts` rows, which went with the deals.
    await db.execute(Contact.__table__.delete().where(Contact.account_id == account_id))
    account = await db.get(Account, account_id)
    await db.delete(account)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------
# Contacts
# --------------------------------------------------------------------------


@router.post(
    "/{account_id}/contacts",
    response_model=ContactResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_contact(
    account_id: uuid.UUID,
    body: ContactCreate,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Add a person to this account.

    `account_id` comes from the path, so a body disagreeing with the URL cannot
    arise. A repeat email on the account is a 409 naming the existing contact --
    see `services/account.py`, which the transcript-resolution path shares.
    """
    await account_service.get_account_or_404(db, account_id)
    await account_service.assert_email_free(db, account_id, body.email)

    contact = Contact(account_id=account_id, **body.model_dump())
    db.add(contact)
    await db.commit()
    await db.refresh(contact)
    return contact


@router.get("/{account_id}/contacts", response_model=List[ContactResponse])
async def list_contacts(
    account_id: uuid.UUID,
    filters: Annotated[ContactFilters, Query()],
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Everyone known at this account.

    Unpaginated: this is the picker behind "add a stakeholder", and an account
    with more contacts than one response can hold is not a shape this product
    has. `?q=` is here for narrowing by name, not for paging.
    """
    await account_service.get_account_or_404(db, account_id)

    stmt = (
        select(Contact)
        .where(Contact.account_id == account_id)
        .order_by(Contact.last_name, Contact.first_name)
    )
    if filters.q:
        term = f"%{filters.q}%"
        stmt = stmt.where(
            or_(
                Contact.first_name.ilike(term),
                Contact.last_name.ilike(term),
                Contact.email.ilike(term),
            )
        )
    return list((await db.scalars(stmt)).all())


@router.get("/{account_id}/contacts/{contact_id}", response_model=ContactResponse)
async def get_contact(
    account_id: uuid.UUID,
    contact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> Any:
    return await account_service.get_contact_or_404(db, account_id, contact_id)


@router.patch("/{account_id}/contacts/{contact_id}", response_model=ContactResponse)
async def update_contact(
    account_id: uuid.UUID,
    contact_id: uuid.UUID,
    body: ContactUpdate,
    db: AsyncSession = Depends(get_db),
) -> Any:
    contact = await account_service.get_contact_or_404(db, account_id, contact_id)
    payload = body.model_dump(exclude_unset=True)

    if "email" in payload:
        # `excluding` the contact itself, so resending an unchanged email is not
        # a conflict with its own row.
        await account_service.assert_email_free(
            db, account_id, payload["email"], excluding=contact_id
        )

    for field, value in payload.items():
        setattr(contact, field, value)
    await db.commit()
    await db.refresh(contact)
    return contact


@router.delete(
    "/{account_id}/contacts/{contact_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_contact(
    account_id: uuid.UUID,
    contact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Remove a contact.

    `deal_contacts` and `meeting_attendees` both reference contacts. The
    stakeholder links cascade -- a person who is gone is not a stakeholder --
    but `meeting_attendees.contact_id` is **nullable by design**: the row
    records that a name spoke in a transcript, which stays true whether or not
    we still track the person. Those are set back to NULL, returning the
    attendee to the unresolved state Phase 2 treats as a signal rather than a
    failure.
    """
    from app.models import DealContact, MeetingAttendee

    await account_service.get_contact_or_404(db, account_id, contact_id)

    await db.execute(
        MeetingAttendee.__table__.update()
        .where(MeetingAttendee.contact_id == contact_id)
        .values(contact_id=None)
    )
    await db.execute(
        DealContact.__table__.delete().where(DealContact.contact_id == contact_id)
    )
    contact = await db.get(Contact, contact_id)
    await db.delete(contact)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

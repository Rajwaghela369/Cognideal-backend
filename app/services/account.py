"""Accounts and contacts -- the rules that must hold for every writer.

Two writers create contacts and they must agree: the ``/accounts/{id}/contacts``
collection, and ``services/meeting.py``'s transcript-resolution path, which
turns an unresolved attendee into a person. The duplicate-email answer is the
part worth sharing, so it lives here and both call it.
"""

import uuid
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Account, Contact


async def get_account_or_404(db: AsyncSession, account_id: uuid.UUID) -> Account:
    account = await db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return account


async def assert_email_free(
    db: AsyncSession,
    account_id: uuid.UUID,
    email: Optional[str],
    *,
    excluding: Optional[uuid.UUID] = None,
) -> None:
    """Refuse a duplicate email on this account, usefully.

    ``contacts`` carries ``UNIQUE(account_id, email)``. Left to the database a
    repeat is an ``IntegrityError`` at commit, which reaches the client as a
    500 -- technically a rejection, practically unactionable. Caught here it
    becomes the answer the caller needs: this person already exists, here is
    their id, link to them instead of creating a second row.

    A null email is always free: the constraint is per-account and Postgres
    permits many NULLs under a unique constraint, which is deliberate -- the
    transcript path creates contacts from a spoken name with no email at all.

    ``excluding`` is for updates, so a PATCH that resends a contact's own
    unchanged email does not collide with itself.
    """
    if not email:
        return

    stmt = select(Contact).where(
        Contact.account_id == account_id, Contact.email == email
    )
    if excluding is not None:
        stmt = stmt.where(Contact.id != excluding)

    existing = await db.scalar(stmt)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"{existing.first_name} {existing.last_name} already uses "
                f"{email} on this account (contact {existing.id}). "
                f"Link to that contact instead of creating a duplicate."
            ),
        )


async def get_contact_or_404(
    db: AsyncSession, account_id: uuid.UUID, contact_id: uuid.UUID
) -> Contact:
    """A contact, checked against the account in the path.

    Both ids are matched rather than just the contact's. Fetching by id alone
    would let ``/accounts/{A}/contacts/{belongs-to-B}`` return and update B's
    contact through A's URL -- a 200 that silently edits someone else's record.
    """
    contact = await db.scalar(
        select(Contact).where(
            Contact.id == contact_id, Contact.account_id == account_id
        )
    )
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found on this account")
    return contact

"""Accounts, and the contacts that hang off them.

``AccountRef`` is the embedded form, used wherever a record points at an
account. The rest is the ``/accounts`` collection, which exists because
``DealCreate`` requires an ``account_id`` and nothing could produce one: a
fresh install could not create its first deal without reaching for psql.

Contacts live here rather than in their own module because they are addressed
as ``/accounts/{account_id}/contacts``. That nesting is not a style choice --
``contacts.account_id`` is the foreign key, deliberately not ``deal_id``, since
the same person can appear in several deals with one company and everything
deal-specific (buying role, influence, sentiment) belongs on ``deal_contacts``.
A contact is therefore a child of an account and of nothing else.
"""

import uuid
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.common import ListQuery, ORM, WRITE


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class AccountRef(BaseModel):
    """Minimal account identity, embedded wherever a record points at one."""

    model_config = ORM

    id: uuid.UUID
    name: str
    industry: Optional[str] = None


class AccountResponse(BaseModel):
    model_config = ORM

    id: uuid.UUID
    name: str
    industry: Optional[str] = None
    website: Optional[str] = None
    employee_band: Optional[str] = None
    hq_region: Optional[str] = None


class AccountListItem(AccountResponse):
    """A row in the accounts table, with the counts that make it worth reading.

    Counted in SQL rather than by loading the collections: an account with
    forty contacts should not cost forty rows to list.
    """

    deal_count: int = 0
    contact_count: int = 0


class ContactResponse(BaseModel):
    model_config = ORM

    id: uuid.UUID
    account_id: uuid.UUID
    first_name: str
    last_name: str
    email: Optional[str] = None
    title: Optional[str] = None
    phone: Optional[str] = None


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


class AccountCreate(BaseModel):
    model_config = WRITE

    name: str = Field(min_length=1, max_length=255)
    industry: Optional[str] = Field(default=None, max_length=255)
    website: Optional[str] = Field(default=None, max_length=255)
    employee_band: Optional[str] = Field(default=None, max_length=50)
    hq_region: Optional[str] = Field(default=None, max_length=100)


class AccountUpdate(BaseModel):
    """Partial update. Read it with ``model_dump(exclude_unset=True)``.

    Every field optional, including ``name`` -- but ``min_length=1`` still
    applies when it is sent, so a rename to the empty string is a 422 rather
    than an account with no name.
    """

    model_config = WRITE

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    industry: Optional[str] = Field(default=None, max_length=255)
    website: Optional[str] = Field(default=None, max_length=255)
    employee_band: Optional[str] = Field(default=None, max_length=50)
    hq_region: Optional[str] = Field(default=None, max_length=100)


class ContactCreate(BaseModel):
    """A new person at this account.

    ``account_id`` is absent on purpose: it comes from the path, so a body that
    disagreed with the URL could not arise. The same reason chat tools take
    their scope from ``chat_sessions.deal_id`` rather than from an argument.

    ``email`` is optional because the transcript-resolution path
    (``POST .../attendees/{id}/resolve``) legitimately creates a contact from a
    spoken name alone. When present it must be unique within the account --
    ``UniqueConstraint("account_id", "email")``.

    Typed ``str`` rather than ``EmailStr`` to match that other path
    (``schemas/v1/deal/meeting.py``), which creates rows in this same table
    under the same constraint. ``EmailStr`` here was the first draft and it was
    wrong: it rejects reserved TLDs such as ``.test``, so a contact the resolve
    path accepts would be refused by this one -- two rules for one table, which
    is the drift `services/account.py` exists to prevent on the duplicate-email
    side. Tightening both is a separate decision.
    """

    model_config = WRITE

    first_name: str = Field(min_length=1, max_length=120)
    last_name: str = Field(min_length=1, max_length=120)
    email: Optional[str] = Field(default=None, max_length=320)
    title: Optional[str] = Field(default=None, max_length=200)
    phone: Optional[str] = Field(default=None, max_length=50)


class ContactUpdate(BaseModel):
    """Partial update. ``account_id`` is not updatable.

    Moving a contact between accounts would silently invalidate every
    ``deal_contacts`` row pointing at them -- those deals belong to the old
    account. Delete and recreate is the honest path, and it fails loudly.
    """

    model_config = WRITE

    first_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    last_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    email: Optional[str] = Field(default=None, max_length=320)
    title: Optional[str] = Field(default=None, max_length=200)
    phone: Optional[str] = Field(default=None, max_length=50)


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


class AccountFilters(ListQuery):
    model_config = WRITE

    #: Case-insensitive substring match on name. `q` rather than `name` because
    #: it is a search, not an equality filter, and the two read differently to
    #: a caller.
    q: Optional[str] = Field(default=None, min_length=1, max_length=255)


class ContactFilters(BaseModel):
    model_config = WRITE

    q: Optional[str] = Field(default=None, min_length=1, max_length=255)

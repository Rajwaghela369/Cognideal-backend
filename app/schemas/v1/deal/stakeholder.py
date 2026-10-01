"""The /deals/{deal_id}/stakeholders contract, both directions.

A row of `deal_contacts`, addressed by the (deal_id, contact_id) pair -- the
surrogate id added in 0004 is internal and appears in none of these models.

`deal_id` is in none of the request models either: it is in the path, and two
sources of truth for the same value is how a row lands on the wrong deal."""

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.enums import BuyingRole, InfluenceLevel, Sentiment
from app.schemas.common import ORM, WRITE


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class DealStakeholder(BaseModel):
    """A row of `deal_contacts` joined to the person it points at."""

    model_config = ORM

    contact_id: uuid.UUID
    first_name: str
    last_name: str
    email: Optional[str] = None
    title: Optional[str] = None
    phone: Optional[str] = None
    buying_role: str
    influence: str
    sentiment: str
    is_primary: bool
    notes: Optional[str] = None


# --------------------------------------------------------------------------
# Requests -- bodies and query parameters
#
# Enums are typed as the enums themselves rather than `str` (as the responses
# above are), so a bad value inbound is a 422 listing the legal ones and
# OpenAPI hands the frontend the vocabulary.
# --------------------------------------------------------------------------


class DealStakeholderBase(BaseModel):
    """The `deal_contacts` payload, minus the identifying pair.

    Defaults mirror the model's server_defaults, so a minimal body produces the
    same row the database would.
    """

    model_config = WRITE

    buying_role: BuyingRole = BuyingRole.UNKNOWN
    influence: InfluenceLevel = InfluenceLevel.UNKNOWN
    sentiment: Sentiment = Sentiment.UNKNOWN
    is_primary: bool = False
    notes: Optional[str] = None


class DealStakeholderCreate(DealStakeholderBase):
    contact_id: uuid.UUID


class DealStakeholderUpdate(BaseModel):
    model_config = WRITE

    buying_role: Optional[BuyingRole] = None
    influence: Optional[InfluenceLevel] = None
    sentiment: Optional[Sentiment] = None
    is_primary: Optional[bool] = None
    notes: Optional[str] = None


class DealStakeholderFilters(BaseModel):
    """Query parameters of GET /deals/{deal_id}/stakeholders.

    Only two fields, so this is not about shortening a signature -- it is about
    `extra="forbid"`. A loose query parameter is silently dropped by FastAPI,
    so ?is_primry=true used to return every stakeholder on the deal and look
    like a working filter.

    No pagination: a deal has a handful of stakeholders. No `sort` either --
    the order is fixed in _stakeholder_stmt (primary first, then alphabetical),
    which is the one order the stakeholder panel renders.
    """

    model_config = WRITE

    buying_role: List[BuyingRole] = Field(
        default_factory=list, description="Repeatable; any of the listed roles"
    )
    is_primary: Optional[bool] = None

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.account import AccountRef

ORM = ConfigDict(from_attributes=True)


class ContactListItem(BaseModel):
    model_config = ORM

    id: uuid.UUID
    account_id: uuid.UUID
    account_name: str
    first_name: str
    last_name: str
    email: Optional[str] = None
    title: Optional[str] = None
    phone: Optional[str] = None
    deal_count: int


class ContactCounts(BaseModel):
    deals: int
    open_deals: int
    meetings_attended: int
    commitments_owned: int


class ContactDetail(BaseModel):
    model_config = ORM

    id: uuid.UUID
    first_name: str
    last_name: str
    email: Optional[str] = None
    title: Optional[str] = None
    phone: Optional[str] = None
    account: AccountRef
    counts: ContactCounts
    created_at: datetime
    updated_at: datetime


class ContactDeal(BaseModel):
    """A deal this person is involved in, with their role *on that deal*.

    The per-deal shape is why this route exists rather than deferring to
    GET /deals?account_id= -- buying_role, influence and sentiment live on
    `deal_contacts` and differ per deal for the same person.
    """

    model_config = ORM

    deal_id: uuid.UUID
    deal_name: str
    stage: str
    value: Optional[Decimal] = None
    currency: str
    risk_level: Optional[str] = None
    expected_close_date: Optional[date] = None
    buying_role: str
    influence: str
    sentiment: str
    is_primary: bool

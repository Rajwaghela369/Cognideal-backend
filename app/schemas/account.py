import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict

ORM = ConfigDict(from_attributes=True)


class AccountRef(BaseModel):
    """Minimal account identity, embedded wherever a record points at one."""

    model_config = ORM

    id: uuid.UUID
    name: str
    industry: Optional[str] = None


class AccountListItem(BaseModel):
    """One row of the accounts table. Aggregates are flat rather than nested so
    each is a sortable column."""

    model_config = ORM

    id: uuid.UUID
    name: str
    industry: Optional[str] = None
    website: Optional[str] = None
    employee_band: Optional[str] = None
    hq_region: Optional[str] = None
    deal_count: int
    open_deal_count: int
    open_pipeline_value: Optional[Decimal] = None
    contact_count: int


class AccountCounts(BaseModel):
    deals: int
    open_deals: int
    won_deals: int
    lost_deals: int
    contacts: int
    documents: int


class AccountDetail(BaseModel):
    model_config = ORM

    id: uuid.UUID
    name: str
    industry: Optional[str] = None
    website: Optional[str] = None
    employee_band: Optional[str] = None
    hq_region: Optional[str] = None
    open_pipeline_value: Optional[Decimal] = None
    # Most recent activity across every deal on the account -- the "is anything
    # happening here?" signal that a per-deal field cannot answer.
    last_activity_at: Optional[datetime] = None
    counts: AccountCounts
    created_at: datetime
    updated_at: datetime

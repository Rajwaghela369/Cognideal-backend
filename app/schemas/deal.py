import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.account import AccountRef

ORM = ConfigDict(from_attributes=True)

__all__ = [
    "AccountRef",
    "ActivityEntry",
    "DealCounts",
    "DealDetail",
    "DealDocument",
    "DealListItem",
    "DealMeeting",
    "DealStakeholder",
    "StageHistoryEntry",
]


class DealListItem(BaseModel):
    """One row of the pipeline table."""

    model_config = ORM

    id: uuid.UUID
    name: str
    account_id: uuid.UUID
    account_name: str
    value: Optional[Decimal] = None
    currency: str
    stage: str
    win_probability: Optional[int] = None
    risk_level: Optional[str] = None
    expected_close_date: Optional[date] = None
    last_activity_at: Optional[datetime] = None
    # Derived from the oldest open task -- `deals` stores no next_action.
    next_action: Optional[str] = None
    next_action_due_date: Optional[date] = None


class DealCounts(BaseModel):
    open_risks: int
    open_tasks: int
    open_commitments: int
    stakeholders: int
    meetings: int
    documents: int


class DealDetail(BaseModel):
    model_config = ORM

    id: uuid.UUID
    name: str
    account: AccountRef
    value: Optional[Decimal] = None
    currency: str
    stage: str
    win_probability: Optional[int] = None
    risk_level: Optional[str] = None
    expected_close_date: Optional[date] = None
    closed_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    next_action: Optional[str] = None
    next_action_due_date: Optional[date] = None
    counts: DealCounts
    created_at: datetime
    updated_at: datetime


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


class DealMeeting(BaseModel):
    model_config = ORM

    id: uuid.UUID
    title: str
    meeting_type: str
    status: str
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    sentiment: Optional[str] = None
    analysis_status: str
    analyzed_at: Optional[datetime] = None
    has_transcript: bool
    attendee_count: int


class DealDocument(BaseModel):
    """Deliberately omits `raw_text` -- the list view must not carry whole
    transcripts. Fetch it from GET /documents/{id}/text."""

    model_config = ORM

    id: uuid.UUID
    title: str
    source_type: str
    original_filename: Optional[str] = None
    mime_type: Optional[str] = None
    byte_size: Optional[int] = None
    occurred_at: datetime
    uploaded_at: datetime
    ingest_status: str
    ingest_error: Optional[str] = None
    chunk_count: int


class StageHistoryEntry(BaseModel):
    model_config = ORM

    id: uuid.UUID
    from_stage: Optional[str] = None
    to_stage: str
    changed_at: datetime
    note: Optional[str] = None


class ActivityEntry(BaseModel):
    model_config = ORM

    id: uuid.UUID
    activity_type: str
    summary: str
    occurred_at: datetime
    contact_id: Optional[uuid.UUID] = None
    contact_name: Optional[str] = None
    meeting_id: Optional[uuid.UUID] = None

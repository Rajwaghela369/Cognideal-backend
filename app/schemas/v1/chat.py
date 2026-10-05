import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.models.enums import ChatScope
from app.schemas.common import ORM, WRITE


class ChatSessionCreate(BaseModel):
    model_config = WRITE

    scope: ChatScope
    deal_id: Optional[uuid.UUID] = None

    @model_validator(mode="after")
    def _scope_matches_deal(self):
        if (self.scope == ChatScope.DEAL) != (self.deal_id is not None):
            raise ValueError("deal scope requires deal_id; global scope forbids it")
        return self


class ChatSessionResponse(BaseModel):
    model_config = ORM

    id: uuid.UUID
    scope: str
    deal_id: Optional[uuid.UUID] = None
    title: Optional[str] = None
    last_message_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ChatSessionUpdate(BaseModel):
    """Rename only. `scope` and `deal_id` are immutable: tool scoping reads
    `deal_id` off this row (task 9.2), so a session that could be re-pointed at
    another deal would make every citation in its history ambiguous."""

    model_config = WRITE

    title: str = Field(min_length=1, max_length=255)


class ChatMessageCreate(BaseModel):
    model_config = WRITE

    content: str = Field(min_length=1, max_length=20_000)


class ChatCitation(BaseModel):
    handle: str
    source_kind: str
    snippet: str
    document_id: Optional[uuid.UUID] = None
    chunk_id: Optional[uuid.UUID] = None
    record_ref: Optional[dict] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class ChatMessageResponse(BaseModel):
    model_config = ORM

    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    status: str
    model: Optional[str] = None
    token_usage: Optional[dict] = None
    latency_ms: Optional[int] = None
    created_at: datetime
    citations: List[ChatCitation] = Field(default_factory=list)

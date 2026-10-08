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
    #: The browser's IANA timezone, e.g. "Europe/Berlin". The assistant needs
    #: it to turn "Friday at 3pm" into a date and time; unknown values fall
    #: back to UTC rather than failing the message.
    timezone: Optional[str] = Field(default=None, max_length=64)


class ChatCitation(BaseModel):
    handle: str
    source_kind: str
    snippet: str
    document_id: Optional[uuid.UUID] = None
    chunk_id: Optional[uuid.UUID] = None
    record_ref: Optional[dict] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class ChatAction(BaseModel):
    """A task or meeting the assistant drafted, and what became of it.

    ``fields`` is the draft as the card shows it: for a task ``title``,
    ``description``, ``due_date``, ``priority``; for a meeting ``title``,
    ``meeting_type``, ``scheduled_at`` and ``attendees``.
    """

    id: str
    kind: str
    status: str
    deal_id: uuid.UUID
    deal_name: Optional[str] = None
    fields: dict
    created_id: Optional[uuid.UUID] = None
    decided_at: Optional[datetime] = None


class ChatActionApply(BaseModel):
    """Create a drafted action, optionally with the user's edits.

    ``fields`` overrides the draft's values, key by key. Only the editable
    keys are accepted -- see ``services/chat_actions.EDITABLE``.
    """

    model_config = WRITE

    fields: Optional[dict] = None


class ChatUsage(BaseModel):
    """Today's chat questions against the shared daily limit."""

    #: 0 when the limit is switched off.
    limit: int
    used: int
    #: None when unlimited.
    remaining: Optional[int] = None
    #: Next midnight in ``timezone``, when ``used`` starts again from 0.
    resets_at: datetime
    timezone: str


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
    actions: List[ChatAction] = Field(default_factory=list)

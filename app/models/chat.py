import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    ChatRole,
    ChatScope,
    MessageStatus,
    chat_role_enum,
    chat_scope_enum,
    message_status_enum,
)


class ChatSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A conversation, either pipeline-wide or scoped to one deal."""

    __tablename__ = "chat_sessions"
    __table_args__ = (
        CheckConstraint(
            "(scope = 'deal') = (deal_id IS NOT NULL)", name="deal_scope_needs_deal"
        ),
    )

    scope: Mapped[ChatScope] = mapped_column(chat_scope_enum, nullable=False)
    deal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_message_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ChatMessage(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """One turn.

    Citations are not a column here and there is no separate citations table --
    they reuse ``claim_evidence`` with ``claim_type='chat_message'``, so a
    cited answer is validated by exactly the same gates as a risk or a
    commitment.

    ``status='streaming'`` gives SSE a row to write into, so a refresh
    mid-answer does not lose the turn.
    """

    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_session_id_created_at", "session_id", "created_at"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[ChatRole] = mapped_column(chat_role_enum, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[MessageStatus] = mapped_column(
        message_status_enum, nullable=False, server_default=MessageStatus.COMPLETE.value
    )
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    token_usage: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

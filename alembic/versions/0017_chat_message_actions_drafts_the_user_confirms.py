"""chat_messages.actions -- tasks and meetings the assistant drafts for confirmation

The chat assistant can now set up work: "a pricing call with Dana on Tuesday,
and a task to send the SOC 2 report by Friday". It does not create either.
Its `draft_task` / `draft_meeting` tools record a draft on the answer, the UI
shows each as a card, and only a person clicking Create writes the row --
through the same service code as the REST routes. The rule the chat tools were
built on stands: the model proposes, a human decides.

The drafts live on the message that proposed them, as JSONB, because that is
their whole lifetime: proposed in one answer, then created or cancelled from
it. Each records its outcome (`created_id`), which is what makes Create
idempotent and lets a reloaded conversation show "Created" with a link rather
than a live button. A table would add a join to every message read for a list
that is almost always empty.

Revision ID: 0017_chat_message_actions
Revises: 0016_fact_content_trgm
Create Date: 2026-10-08

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0017_chat_message_actions"
down_revision: Union[str, None] = "0016_fact_content_trgm"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("actions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_messages", "actions")

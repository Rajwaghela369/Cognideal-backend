"""index user chat messages by time -- the daily question limit counts them

`services/chat_usage.py` caps the assistant at a number of questions per day,
counted as `chat_messages` rows with `role='user'` since local midnight. That
count runs on every question sent and on every load of the usage bar, and the
only existing index leads with `session_id`, so it would scan the table. A
partial index on `created_at` for user rows keeps it an index range scan.

Revision ID: 0018_chat_question_count
Revises: 0017_chat_message_actions
Create Date: 2026-10-09

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0018_chat_question_count"
down_revision: Union[str, None] = "0017_chat_message_actions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX ix_chat_messages_user_created_at ON chat_messages (created_at)
          WHERE role = 'user'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chat_messages_user_created_at")

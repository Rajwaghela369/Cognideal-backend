"""dirty deals and sweep cadence

Revision ID: 0014_analysis_triggers
Revises: 0013_open_risk_taxonomy
Create Date: 2026-10-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014_analysis_triggers"
down_revision: Union[str, None] = "0013_open_risk_taxonomy"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("deals", sa.Column("analysis_dirty_first_at", sa.DateTime(timezone=True)))
    op.add_column("deals", sa.Column("analysis_dirty_last_at", sa.DateTime(timezone=True)))
    op.add_column("deals", sa.Column("analysis_dirty_reason", sa.Text()))
    op.add_column("deals", sa.Column("analysis_swept_at", sa.DateTime(timezone=True)))
    op.create_index(
        "ix_deals_analysis_dirty_last_at_pending",
        "deals",
        ["analysis_dirty_last_at"],
        postgresql_where=sa.text("analysis_dirty_first_at IS NOT NULL"),
    )
    op.create_index("ix_deals_analysis_swept_at", "deals", ["analysis_swept_at"])


def downgrade() -> None:
    op.drop_index("ix_deals_analysis_swept_at", table_name="deals")
    op.drop_index("ix_deals_analysis_dirty_last_at_pending", table_name="deals")
    op.drop_column("deals", "analysis_swept_at")
    op.drop_column("deals", "analysis_dirty_reason")
    op.drop_column("deals", "analysis_dirty_last_at")
    op.drop_column("deals", "analysis_dirty_first_at")

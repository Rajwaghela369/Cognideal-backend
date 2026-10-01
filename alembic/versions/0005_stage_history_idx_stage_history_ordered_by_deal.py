"""stage history ordered by deal

Every read of deal_stage_history is "one deal's transitions, in order": the
timeline route, and the LEAD() window function that derives time-in-stage from
it. ix_deal_stage_history_deal_id found the rows and left the sort to be done
in memory.

The replacement is deal_id-leading, so it serves the plain lookup the old index
served and supplies the ordering for free. Mirrors the shape already used for
ix_activities_deal_id_occurred_at.

Revision ID: 0005_stage_history_idx
Revises: 0004_deal_contacts_id
Create Date: 2026-09-28 11:41:58.204773

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '0005_stage_history_idx'
down_revision: Union[str, None] = '0004_deal_contacts_id'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'ix_deal_stage_history_deal_id_changed_at',
        'deal_stage_history',
        ['deal_id', 'changed_at'],
        unique=False,
    )
    op.drop_index(
        op.f('ix_deal_stage_history_deal_id'), table_name='deal_stage_history'
    )


def downgrade() -> None:
    op.create_index(
        op.f('ix_deal_stage_history_deal_id'),
        'deal_stage_history',
        ['deal_id'],
        unique=False,
    )
    op.drop_index(
        'ix_deal_stage_history_deal_id_changed_at', table_name='deal_stage_history'
    )

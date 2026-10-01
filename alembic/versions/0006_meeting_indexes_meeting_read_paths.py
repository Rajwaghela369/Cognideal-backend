"""meeting read paths

Three indexes, all on the meeting tables.

ix_meeting_attendees_contact_id -- contact_id had no index, and that is the
direction the product is built on. "Has this person attended any meeting?" runs
once per person for the deal participants roll-up and again for the
NO_ECONOMIC_BUYER risk; without it each one is a sequential scan of every
attendee row ever recorded.

uq_meeting_attendees_meeting_id_contact_id -- nothing stopped the same contact
being attached to one meeting twice (once from the invite, once from the
transcript), which silently inflates every attendance count. Partial, because
unresolved attendees all carry contact_id IS NULL and one meeting may
legitimately have many of those.

ix_meetings_deal_id_scheduled_at -- replaces ix_meetings_deal_id. Every read is
"this deal's meetings, by date"; the old index found the rows and left the sort
to be done separately. deal_id-leading, so it serves the plain lookup too.

Revision ID: 0006_meeting_indexes
Revises: 0005_stage_history_idx
Create Date: 2026-09-29 10:18:44.512903

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0006_meeting_indexes'
down_revision: Union[str, None] = '0005_stage_history_idx'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        op.f('ix_meeting_attendees_contact_id'),
        'meeting_attendees',
        ['contact_id'],
        unique=False,
    )
    op.create_index(
        'uq_meeting_attendees_meeting_id_contact_id',
        'meeting_attendees',
        ['meeting_id', 'contact_id'],
        unique=True,
        postgresql_where=sa.text('contact_id IS NOT NULL'),
    )
    op.create_index(
        'ix_meetings_deal_id_scheduled_at',
        'meetings',
        ['deal_id', 'scheduled_at'],
        unique=False,
    )
    op.drop_index(op.f('ix_meetings_deal_id'), table_name='meetings')


def downgrade() -> None:
    op.create_index(
        op.f('ix_meetings_deal_id'), 'meetings', ['deal_id'], unique=False
    )
    op.drop_index('ix_meetings_deal_id_scheduled_at', table_name='meetings')
    op.drop_index(
        'uq_meeting_attendees_meeting_id_contact_id',
        table_name='meeting_attendees',
        postgresql_where=sa.text('contact_id IS NOT NULL'),
    )
    op.drop_index(
        op.f('ix_meeting_attendees_contact_id'), table_name='meeting_attendees'
    )

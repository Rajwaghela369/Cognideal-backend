"""timeline is derived

`activities` was a denormalized copy of four other tables, and every one of its
seven activity_type values already had a primary home:

    meeting          -> meetings
    stage_change     -> deal_stage_history
    task_completed   -> tasks.completed_at
    document_upload  -> documents.uploaded_at
    email            -> documents, source_type='email'
    note             -> documents, source_type='note'
    call             -> meetings, meeting_type='check_in', no transcript

Not one was a primary record, which makes the table the same pattern the schema
rejects twice elsewhere -- no `next_action` column, no `status` column -- for
the same reason: a stored copy drifts from what it copies.

The cost of keeping it was a write in every service that touches a deal
(apply_stage_change, task status, meeting create and status, document upload,
attendee resolve), any one of which could be forgotten, producing a timeline
that is silently incomplete -- worse than none, because it looks complete.

The deal timeline is a UNION ALL over the four source tables instead, already
served by ix_meetings_deal_id_scheduled_at,
ix_deal_stage_history_deal_id_changed_at and ix_tasks_open_due_date. It cannot
drift because there is nothing to keep in sync.

The ActivityType enum is deliberately KEPT in models/enums.py: it stops being a
column and becomes the discriminator that UNION emits per row.

activity_type was text + CHECK rather than a native enum, so there is no
CREATE TYPE left behind -- the enum-leftover trap documented in
docs/schema/TASKS.md does not apply here.

Revision ID: 0007_drop_activities
Revises: 0006_meeting_indexes
Create Date: 2026-09-29 11:02:17.339481

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0007_drop_activities'
down_revision: Union[str, None] = '0006_meeting_indexes'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index('ix_activities_deal_id_occurred_at', table_name='activities')
    op.drop_table('activities')


def downgrade() -> None:
    op.create_table(
        'activities',
        sa.Column(
            'id',
            sa.UUID(),
            server_default=sa.text('gen_random_uuid()'),
            nullable=False,
        ),
        sa.Column('deal_id', sa.UUID(), nullable=False),
        sa.Column('contact_id', sa.UUID(), nullable=True),
        sa.Column('meeting_id', sa.UUID(), nullable=True),
        sa.Column('activity_type', sa.String(length=50), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "activity_type IN ('call', 'email', 'meeting', 'note', "
            "'stage_change', 'document_upload', 'task_completed')",
            name=op.f('ck_activities_activity_type'),
        ),
        sa.ForeignKeyConstraint(
            ['contact_id'],
            ['contacts.id'],
            name=op.f('fk_activities_contact_id_contacts'),
            ondelete='SET NULL',
        ),
        sa.ForeignKeyConstraint(
            ['deal_id'],
            ['deals.id'],
            name=op.f('fk_activities_deal_id_deals'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['meeting_id'],
            ['meetings.id'],
            name=op.f('fk_activities_meeting_id_meetings'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_activities')),
    )
    op.create_index(
        'ix_activities_deal_id_occurred_at',
        'activities',
        ['deal_id', 'occurred_at'],
        unique=False,
    )

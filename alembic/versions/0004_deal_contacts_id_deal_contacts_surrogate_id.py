"""deal contacts surrogate id

``extracted_facts.promoted_to_id`` is a single uuid -- it is how a fact a human
accepted records the Layer A row it became. ``deal_contacts`` was the one
promotion target it could not address: the row's identity was the pair
``(deal_id, contact_id)``, and no single uuid names that. A ``stakeholder`` fact
could be accepted but never record where it went.

The pair becomes a UNIQUE constraint rather than the primary key -- the same
guarantee by a different mechanism -- and the surrogate id gives the row one
handle. It is internal: the API still addresses a link as
``/deals/{deal_id}/stakeholders/{contact_id}`` and never returns the id.

Two indexes ride along, both on this table and both cheap while it is empty:

*   ``ix_deal_contacts_contact_id`` -- the old composite PK was deal_id-leading,
    so "which deals is this person on?" had no index to use.
*   ``uq_deal_contacts_deal_id_primary`` -- at most one primary contact per
    deal. Partial uniqueness cannot be a constraint, only an index, so it can
    never be DEFERRABLE; writers must demote the incumbent in an earlier
    statement.

Revision ID: 0004_deal_contacts_id
Revises: 0003_attendee_name
Create Date: 2026-09-28 11:04:12.880431

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0004_deal_contacts_id'
down_revision: Union[str, None] = '0003_attendee_name'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # server_default fills the column for any pre-existing row, so the NOT NULL
    # and the primary key below hold without a backfill pass.
    op.add_column(
        'deal_contacts',
        sa.Column(
            'id',
            sa.UUID(),
            server_default=sa.text('gen_random_uuid()'),
            nullable=False,
        ),
    )
    # Dropping the PK drops its backing index with it -- nothing to clean up.
    op.drop_constraint(op.f('pk_deal_contacts'), 'deal_contacts', type_='primary')
    op.create_primary_key(op.f('pk_deal_contacts'), 'deal_contacts', ['id'])
    op.create_unique_constraint(
        op.f('uq_deal_contacts_deal_id_contact_id'),
        'deal_contacts',
        ['deal_id', 'contact_id'],
    )
    op.create_index(
        op.f('ix_deal_contacts_contact_id'),
        'deal_contacts',
        ['contact_id'],
        unique=False,
    )
    op.create_index(
        'uq_deal_contacts_deal_id_primary',
        'deal_contacts',
        ['deal_id'],
        unique=True,
        postgresql_where=sa.text('is_primary'),
    )


def downgrade() -> None:
    op.drop_index(
        'uq_deal_contacts_deal_id_primary',
        table_name='deal_contacts',
        postgresql_where=sa.text('is_primary'),
    )
    op.drop_index(op.f('ix_deal_contacts_contact_id'), table_name='deal_contacts')
    op.drop_constraint(
        op.f('uq_deal_contacts_deal_id_contact_id'), 'deal_contacts', type_='unique'
    )
    op.drop_constraint(op.f('pk_deal_contacts'), 'deal_contacts', type_='primary')
    op.create_primary_key(
        op.f('pk_deal_contacts'), 'deal_contacts', ['deal_id', 'contact_id']
    )
    op.drop_column('deal_contacts', 'id')

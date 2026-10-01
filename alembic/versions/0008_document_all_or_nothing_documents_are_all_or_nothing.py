"""documents are all or nothing

A document row now exists only when the document is fully ingested -- text
extracted, chunks written, bytes in object storage. Anything that fails leaves
no row at all, so `ingest_status` and `ingest_error` have nothing to represent:
there is no half-ingested state for the API to report.

`raw_text` goes for a different reason -- it was always a duplicate. The chunks,
ordered by chunk_index, *are* the text; the original file is served from object
storage for preview. Keeping both meant two copies of the same content that
could disagree after a re-ingest.

storage_uri and content_hash become NOT NULL, since every row is now backed by
stored bytes and the hash is both the idempotency key and the object key.

NOTE the DROP TYPE. `ingest_status` was a *native* Postgres enum, and
`DROP COLUMN` leaves the type behind -- the trap documented in
docs/schema/TASKS.md, where a surviving type made the next `upgrade head` fail
with "type already exists". `0007` needed no such cleanup because
`activity_type` was text + CHECK.

Revision ID: 0008_document_all_or_nothing
Revises: 0007_drop_activities
Create Date: 2026-09-29 11:41:06.772918

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0008_document_all_or_nothing'
down_revision: Union[str, None] = '0007_drop_activities'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INGEST_STATUS_VALUES = (
    'pending', 'parsing', 'chunking', 'embedding', 'ready', 'failed',
)


def upgrade() -> None:
    op.drop_column('documents', 'ingest_error')
    op.drop_column('documents', 'ingest_status')
    op.drop_column('documents', 'raw_text')
    # DROP COLUMN does not drop the enum type the column implicitly created.
    op.execute('DROP TYPE IF EXISTS ingest_status')

    # Any pre-existing row predates object storage and has no bytes behind it,
    # so it cannot satisfy the NOT NULL below. The table is a staging area for
    # content that no longer exists in a usable form -- clear it rather than
    # inventing placeholder keys.
    op.execute('DELETE FROM documents WHERE storage_uri IS NULL OR content_hash IS NULL')

    op.alter_column('documents', 'storage_uri', existing_type=sa.Text(), nullable=False)
    op.alter_column(
        'documents', 'content_hash', existing_type=sa.String(length=64), nullable=False
    )


def downgrade() -> None:
    op.alter_column(
        'documents', 'content_hash', existing_type=sa.String(length=64), nullable=True
    )
    op.alter_column('documents', 'storage_uri', existing_type=sa.Text(), nullable=True)

    ingest_status = sa.Enum(*INGEST_STATUS_VALUES, name='ingest_status')
    ingest_status.create(op.get_bind(), checkfirst=True)
    op.add_column('documents', sa.Column('raw_text', sa.Text(), nullable=True))
    op.add_column(
        'documents',
        sa.Column(
            'ingest_status',
            ingest_status,
            server_default='pending',
            nullable=False,
        ),
    )
    op.add_column('documents', sa.Column('ingest_error', sa.Text(), nullable=True))

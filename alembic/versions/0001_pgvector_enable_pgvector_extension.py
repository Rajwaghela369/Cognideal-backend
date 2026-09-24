"""enable pgvector extension

Separate from the schema migration on purpose: the extension is a
database-level concern that must exist before any table declares a `vector`
column, and it is the one step that fails loudly if the Postgres image is
wrong (plain `postgres:16` cannot run this -- see docker-compose.yml).

Revision ID: 0001_pgvector
Revises:
Create Date: 2026-09-24

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0001_pgvector"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    # Only drops cleanly once every vector column is gone, which the schema
    # migration's own downgrade handles first.
    op.execute("DROP EXTENSION IF EXISTS vector")

"""enable pg_trgm for attendee name resolution

Same reasoning as 0001_pgvector: an extension is a database-level concern that
has to exist before anything depends on it, and keeping it in its own revision
means the failure is legible if the image lacks it. Unlike pgvector, pg_trgm
ships with stock Postgres, so there is no image requirement here.

Resolving a transcript speaker label to a `contacts` row is fuzzy by nature --
"Priya Raman" from an invite, "Priya" in a transcript, "Priya Ramen" from a
mis-hearing. Exact matching alone leaves every one of those unresolved, which
would make the missing-stakeholder roll-up useless by burying the genuine
signal (a name that really is nobody we know) under spelling noise.

`gin_trgm_ops` on the contacts name expression rather than the bare columns:
the thing being compared is the full name, and resolution reads it tens of
times per transcript -- once per speaker per meeting.

Revision ID: 0010_pg_trgm
Revises: 0009_recommendation_provenance
Create Date: 2026-10-01

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0010_pg_trgm"
down_revision: Union[str, None] = "0009_recommendation_provenance"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    # Trigram index on the concatenated name, matching how resolution queries
    # it. An index on first_name and last_name separately cannot serve a
    # similarity search against "Priya Raman".
    op.execute(
        """
        CREATE INDEX ix_contacts_full_name_trgm ON contacts
          USING gin ((first_name || ' ' || last_name) gin_trgm_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_contacts_full_name_trgm")
    # Dropped last: the index above depends on the operator class.
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")

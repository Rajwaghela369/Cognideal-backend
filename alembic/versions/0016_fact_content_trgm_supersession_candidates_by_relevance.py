"""index extracted_facts.content for trigram similarity

Supersession asks a model which older fact a new one replaces, choosing from a
list this codebase supplies. `app/ai/reconcile.py` built that list with
`ORDER BY extracted_at DESC LIMIT 10` -- recency standing in for relevance,
which holds only while a deal has fewer than ten accepted facts of a type.
Past that the one fact genuinely about the same subject can sit outside the
window, the model never sees it, and it stays `accepted` forever while a newer
fact contradicts it. Both then appear on the panel, each with a valid citation.
Nothing is wrong with the evidence; the shortlist was wrong.

So the limit moves onto relevance: `similarity(content, :new_content)`, with
`pg_trgm` already enabled by 0010. Trigram rather than embeddings deliberately
-- the extension is here, there is no provider to call, no dimension to pin, no
second code path to keep working, and `tests/test_reconcile.py` can measure it.

A GIN trigram index does not serve `ORDER BY similarity(...) DESC` as an index
scan; that needs GiST with `<->`. It is here for the `similarity(...) > t` floor
in the same query, which is what keeps a deal with hundreds of accepted facts
from sorting all of them per new fact. The ordering itself is a sort over the
rows the floor admits, and at this row count that is the right trade: GIN is
smaller and faster to build, and the floor is the selective half.

Revision ID: 0016_fact_content_trgm
Revises: 0015_commitment_correction
Create Date: 2026-10-05

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0016_fact_content_trgm"
down_revision: Union[str, None] = "0015_commitment_correction"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # No CREATE EXTENSION: 0010 owns pg_trgm, and repeating it here would mean
    # two revisions could each believe they may drop it.
    op.execute(
        """
        CREATE INDEX ix_extracted_facts_content_trgm ON extracted_facts
          USING gin (content gin_trgm_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_extracted_facts_content_trgm")

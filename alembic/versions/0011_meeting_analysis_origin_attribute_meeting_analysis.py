"""attribute meeting analysis, and record why it failed

`MeetingAnalysis` has promised `summary` and `sentiment` since the API layer
existed, and nothing could write them attributably: `meetings` had no `origin`,
so once both a human and the analyzer can set those fields there is no way to
tell which did. The root README has carried this as a known gap.

Four columns on `meetings`, and the fourth is not cosmetic. A critical-stage
failure sets `analysis_status='failed'` and the reason survived only in the
worker log, so the Analyzer screen could say *that* it failed but not *why* --
which makes the status useless to the person looking at it.

`deal_contacts.origin` closes the same gap on the other table the README names:
`buying_role` will eventually be written both by a human and by a promoted
`stakeholder` fact.

Deliberately NOT one `origin` column on `meetings`. The analysis writes
`summary` and `sentiment`; the title, type and timestamps are the user's. A
single `origin` would have to mean "who wrote some of this row", which is not a
question anyone asks -- hence the `analysis_` prefix, scoped to the fields the
pipeline owns.

Revision ID: 0011_meeting_analysis_origin
Revises: 0010_pg_trgm
Create Date: 2026-10-01

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_meeting_analysis_origin"
down_revision: Union[str, None] = "0010_pg_trgm"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The `origin` type already exists (0002 created it for the tables that carry
# it). create_type=False stops Alembic emitting a second CREATE TYPE, which
# fails with "type already exists" -- the same trap docs/schema/TASKS.md
# records for shared native enums.
ORIGIN = postgresql.ENUM("user", "ai", name="origin", create_type=False)


def upgrade() -> None:
    op.add_column("meetings", sa.Column("analysis_origin", ORIGIN, nullable=True))
    op.add_column(
        "meetings", sa.Column("analysis_confidence", sa.Numeric(3, 2), nullable=True)
    )
    op.add_column("meetings", sa.Column("analysis_model", sa.String(100), nullable=True))
    op.add_column("meetings", sa.Column("analysis_error", sa.Text(), nullable=True))

    op.add_column(
        "deal_contacts",
        sa.Column("origin", ORIGIN, nullable=False, server_default="user"),
    )


def downgrade() -> None:
    op.drop_column("deal_contacts", "origin")
    op.drop_column("meetings", "analysis_error")
    op.drop_column("meetings", "analysis_model")
    op.drop_column("meetings", "analysis_confidence")
    op.drop_column("meetings", "analysis_origin")
    # No DROP TYPE: `origin` is still used by commitments, risks,
    # recommendations and extracted_facts. The trap in docs/schema/TASKS.md is
    # the reverse case -- dropping the last column that uses a native enum
    # leaves the type behind -- and it does not apply here.

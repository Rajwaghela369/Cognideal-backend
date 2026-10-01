"""attribute detected risks and recommendations to the detector that made them

The same argument as `claim_validations.validator_version`, which the schema
already carries: after a prompt change, every row older than the change came
from a different detector. Without the column the precision metric silently
averages two populations, and "40% of dismissals are `wrong`" stops meaning
anything because you cannot tell which detector earned them.

Existing rows are backfilled `deterministic-1`, not left NULL. They were made
by the four SQL rules in `services/detect.py`, which is a real and nameable
detector -- and it makes the labelled "before" that task 7.4's shadow run
compares against.

Revision ID: 0012_detector_provenance
Revises: 0011_meeting_analysis_origin
Create Date: 2026-10-01

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012_detector_provenance"
down_revision: Union[str, None] = "0011_meeting_analysis_origin"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DETERMINISTIC = "deterministic-1"


def upgrade() -> None:
    for table in ("risks", "recommendations"):
        op.add_column(table, sa.Column("model", sa.String(100), nullable=True))
        op.add_column(table, sa.Column("detector_version", sa.String(50), nullable=True))
        # Backfill before anything new is written, so the AI detector's first
        # run has a clean population to be compared against.
        op.execute(
            sa.text(
                "UPDATE %s SET detector_version = :version WHERE detector_version IS NULL"
                % table
            ).bindparams(version=DETERMINISTIC)
        )


def downgrade() -> None:
    for table in ("risks", "recommendations"):
        op.drop_column(table, "detector_version")
        op.drop_column(table, "model")

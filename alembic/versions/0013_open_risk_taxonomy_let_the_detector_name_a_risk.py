"""let the detector name a risk the enum does not have

`risk_type` is a closed set of ten, which is what makes
`uq_risks_deal_id_risk_type_open` work: re-running detection bumps
`last_seen_at` instead of inserting a fifth "single-threaded". Prose titles
vary between runs and the index cannot see that two cards are one risk, so
identity has to be the key, never the text.

That constrains an AI detector to ten kinds of risk. `risk_key` lifts the
constraint without giving up identity: with `risk_type='other'` the model
proposes a slug, and the key becomes `(deal_id, risk_type, risk_key)`. For the
ten known types `risk_key` stays `''` and behaviour is exactly as before --
which is why the default is empty string rather than NULL. A NULL in a unique
index is distinct from every other NULL, so two open `stalled_stage` rows would
both be allowed and the guarantee would quietly disappear.

Canonicalisation of a proposed slug happens in Python before the upsert, not
here: `champion_going_quiet` and `champion_disengaged` are one problem, and no
constraint can know that.

`risk_type` is `text + CHECK` rather than a native enum precisely so this set
can churn (docs/schema/README.md section 1), so adding `other` is a CHECK swap.
Note the trap recorded in docs/schema/TASKS.md: `create_check_constraint` runs
the name through the metadata naming convention, so dropping one by the name it
actually has needs raw SQL or it gets double-prefixed.

Revision ID: 0013_open_risk_taxonomy
Revises: 0012_detector_provenance
Create Date: 2026-10-01

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013_open_risk_taxonomy"
down_revision: Union[str, None] = "0012_detector_provenance"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KNOWN = (
    "no_economic_buyer", "single_threaded", "stalled_stage", "close_date_at_risk",
    "unresolved_objection", "security_review_pending", "budget_unconfirmed",
    "competitor_pressure", "missed_commitment", "gone_quiet",
)


def _check(values) -> str:
    return "risk_type IN (%s)" % ", ".join("'%s'" % v for v in values)


def upgrade() -> None:
    op.add_column(
        "risks",
        sa.Column("risk_key", sa.Text(), nullable=False, server_default=""),
    )

    # Raw SQL for the drop: the constraint is named `ck_risks_risk_type` in the
    # database, and routing that through create/drop_check_constraint would
    # prefix it again.
    op.execute("ALTER TABLE risks DROP CONSTRAINT ck_risks_risk_type")
    op.execute(
        "ALTER TABLE risks ADD CONSTRAINT ck_risks_risk_type CHECK (%s)"
        % _check(KNOWN + ("other",))
    )

    op.execute("DROP INDEX uq_risks_deal_id_risk_type_open")
    op.execute(
        """
        CREATE UNIQUE INDEX uq_risks_open_key ON risks (deal_id, risk_type, risk_key)
          WHERE status = 'open'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX uq_risks_open_key")
    op.execute(
        """
        CREATE UNIQUE INDEX uq_risks_deal_id_risk_type_open ON risks (deal_id, risk_type)
          WHERE status = 'open'
        """
    )
    # Any `other` row would violate the narrower CHECK, and there is no
    # defensible automatic mapping back into the ten -- so they go. The
    # downgrade is for a mistaken upgrade, not for discarding real data.
    op.execute("DELETE FROM risks WHERE risk_type = 'other'")
    op.execute("ALTER TABLE risks DROP CONSTRAINT ck_risks_risk_type")
    op.execute(
        "ALTER TABLE risks ADD CONSTRAINT ck_risks_risk_type CHECK (%s)" % _check(KNOWN)
    )
    op.drop_column("risks", "risk_key")

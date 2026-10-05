"""give a proposed data correction somewhere to land

Stage 7 of the meeting pipeline asks whether this meeting's facts show an open
commitment was kept. It has worked since Phase 6 and has written nothing,
because the design said a proposal lands as a `recommendation` and that did not
fit: every `ActionType` names an action to *take* -- `send_document`,
`engage_stakeholder` -- while "this commitment now looks satisfied" is a
proposed **data correction**. Filing it under the nearest action would corrupt
the action-type and `dismissal_reason` distributions, which are two of the few
signals in this schema that are not self-reported.

So `action_type` gains `correct_record`. That much is a CHECK swap, which is
why `action_type` is `text + CHECK` rather than a native enum
(docs/schema/README.md section 1).

**`source_commitment_id` is the part that is not obvious.** A recommendation's
identity comes from `uq_recommendations_deal_id_source_risk_id_suggested`,
which is scoped to `source_risk_id IS NOT NULL` so that *proactive* advice --
which answers no risk -- is not capped at one per deal. A correction proposal
also has no source risk, so without a second key stage 7 would file a fresh
duplicate on every analysis run of every meeting touching that commitment, and
the detector would spam the one surface it just gained. The column plus its
partial unique index gives corrections the same one-live-proposal guarantee
risks already have.

`SET NULL` rather than `CASCADE`, for two reasons. It matches
`source_risk_id`'s precedent -- an accepted recommendation and the task it
created are still real work after the thing that prompted them is gone. More
pointedly, `claim_evidence.claim_id` carries no foreign key, so a database-level
CASCADE that deleted recommendation rows would walk straight into the orphan
trap documented in docs/api/README.md section 7: the links must be cleared
through `services/claims.py`, which a CASCADE cannot call. A nulled row stops
matching the partial index, which is exactly what happens to a suggestion whose
risk was deleted today.

Revision ID: 0015_commitment_correction
Revises: 0014_analysis_triggers
Create Date: 2026-10-01

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0015_commitment_correction"
down_revision: Union[str, None] = "0014_analysis_triggers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KNOWN = (
    "schedule_meeting", "send_document", "follow_up_email",
    "engage_stakeholder", "update_close_date", "address_objection",
    "internal_escalation",
)


def _check(values) -> str:
    return "action_type IN (%s)" % ", ".join("'%s'" % v for v in values)


def upgrade() -> None:
    # Raw SQL for the drop: the constraint is named `ck_recommendations_action_type`
    # in the database, and routing that through drop_check_constraint would run
    # it through the metadata naming convention and prefix it again. Same trap
    # as 0013; see docs/schema/TASKS.md.
    op.execute("ALTER TABLE recommendations DROP CONSTRAINT ck_recommendations_action_type")
    op.execute(
        "ALTER TABLE recommendations ADD CONSTRAINT ck_recommendations_action_type "
        "CHECK (%s)" % _check(KNOWN + ("correct_record",))
    )

    op.add_column(
        "recommendations",
        sa.Column("source_commitment_id", sa.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_recommendations_source_commitment_id_commitments",
        "recommendations",
        "commitments",
        ["source_commitment_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_recommendations_source_commitment_id",
        "recommendations",
        ["source_commitment_id"],
    )
    # One live correction proposal per commitment, mirroring the risk-keyed
    # index above it.
    op.execute(
        """
        CREATE UNIQUE INDEX uq_recommendations_deal_id_source_commitment_id_suggested
          ON recommendations (deal_id, source_commitment_id)
          WHERE status = 'suggested' AND source_commitment_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX uq_recommendations_deal_id_source_commitment_id_suggested")
    op.drop_index("ix_recommendations_source_commitment_id", table_name="recommendations")
    op.drop_constraint(
        "fk_recommendations_source_commitment_id_commitments",
        "recommendations",
        type_="foreignkey",
    )
    op.drop_column("recommendations", "source_commitment_id")

    # A `correct_record` row would violate the narrower CHECK and there is no
    # defensible remap into the seven -- none of them means "fix this field".
    # The downgrade undoes a mistaken upgrade; it is not a data-discarding tool,
    # which is why the delete is scoped to exactly the new value.
    #
    # claim_evidence.claim_id has no foreign key, so the links must go first or
    # they are orphaned. This is the one place a migration has to do what
    # services/claims.py does.
    op.execute(
        """
        DELETE FROM claim_evidence
         WHERE claim_type = 'recommendation'
           AND claim_id IN (SELECT id FROM recommendations WHERE action_type = 'correct_record')
        """
    )
    op.execute("DELETE FROM recommendations WHERE action_type = 'correct_record'")
    op.execute("ALTER TABLE recommendations DROP CONSTRAINT ck_recommendations_action_type")
    op.execute(
        "ALTER TABLE recommendations ADD CONSTRAINT ck_recommendations_action_type "
        "CHECK (%s)" % _check(KNOWN)
    )

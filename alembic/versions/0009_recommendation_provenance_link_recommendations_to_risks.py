"""link recommendations to risks, and record dismissals

`risks` and `recommendations` had no link at all -- not a foreign key in either
direction. Their only connection was that both cited the same `evidence` rows
through `claim_evidence`, which is indirect and a poor thing to assemble a UI
from. The risk panel shows one card per risk with its suggested action inline,
and that card cannot be built without a join.

source_risk_id is nullable because a *proactive* recommendation has no risk
behind it, and ON DELETE SET NULL rather than CASCADE because an accepted
recommendation and the task it produced are real work that should outlive the
risk that prompted them. Follows tasks.source_fact_id and
commitments.source_fact_id -- the same "where did this come from" pattern.

dismissal_reason is structured so dismissals can be counted ("40% are `wrong`"
says the detector needs work; "40% are `already_handled`" says it is right but
late); dismissal_note carries the detail a count cannot. text + CHECK rather
than a native enum because this set will churn as it becomes clear what people
actually click -- see docs/schema/README.md section 1.

uq_recommendations_deal_id_source_risk_id_suggested is what stops the detector
re-creating a suggestion it already made. Keyed on the *risk*, not the action
type: two risks legitimately share an action_type -- "engage the economic buyer"
and "broaden beyond one contact" are both `engage_stakeholder` but are different
pieces of advice, and keying on action_type silently dropped the second. Scoped
to `suggested` so a dismissal does not block a fresh suggestion later if
circumstances change, and to source_risk_id IS NOT NULL so proactive
recommendations, which have no risk, are not limited to one per deal.

Revision ID: 0009_recommendation_provenance
Revises: 0008_document_all_or_nothing
Create Date: 2026-09-29 12:24:51.108663

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0009_recommendation_provenance'
down_revision: Union[str, None] = '0008_document_all_or_nothing'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('recommendations', sa.Column('source_risk_id', sa.UUID(), nullable=True))
    op.add_column(
        'recommendations', sa.Column('dismissal_reason', sa.String(length=30), nullable=True)
    )
    op.add_column('recommendations', sa.Column('dismissal_note', sa.Text(), nullable=True))
    op.create_foreign_key(
        op.f('fk_recommendations_source_risk_id_risks'),
        'recommendations',
        'risks',
        ['source_risk_id'],
        ['id'],
        ondelete='SET NULL',
    )
    # Autogenerate does not detect CHECK constraints -- added by hand. The bare
    # logical name goes through the naming convention, which prefixes it with
    # ck_recommendations_.
    op.create_check_constraint(
        'dismissal_reason',
        'recommendations',
        "dismissal_reason IN ('already_handled', 'not_relevant', 'wrong', "
        "'bad_timing', 'other')",
    )
    op.create_index(
        op.f('ix_recommendations_source_risk_id'),
        'recommendations',
        ['source_risk_id'],
        unique=False,
    )
    op.create_index(
        'uq_recommendations_deal_id_source_risk_id_suggested',
        'recommendations',
        ['deal_id', 'source_risk_id'],
        unique=True,
        postgresql_where=sa.text(
            "status = 'suggested' AND source_risk_id IS NOT NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index(
        'uq_recommendations_deal_id_source_risk_id_suggested',
        table_name='recommendations',
        postgresql_where=sa.text(
            "status = 'suggested' AND source_risk_id IS NOT NULL"
        ),
    )
    op.drop_index(
        op.f('ix_recommendations_source_risk_id'), table_name='recommendations'
    )
    # Raw SQL rather than op.drop_constraint(): the metadata naming convention
    # would expand the name a second time into
    # ck_recommendations_ck_recommendations_dismissal_reason. Same trap as
    # 0003_attendee_name.
    op.execute(
        'ALTER TABLE recommendations DROP CONSTRAINT ck_recommendations_dismissal_reason'
    )
    op.drop_constraint(
        op.f('fk_recommendations_source_risk_id_risks'),
        'recommendations',
        type_='foreignkey',
    )
    op.drop_column('recommendations', 'dismissal_note')
    op.drop_column('recommendations', 'dismissal_reason')
    op.drop_column('recommendations', 'source_risk_id')

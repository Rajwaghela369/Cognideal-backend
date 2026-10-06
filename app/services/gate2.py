"""Gate 2's deterministic half: staleness. Task 6.3.

Gate 2 asks two questions and only one of them is arithmetic:

*   **Staleness** -- is this claim's newest evidence older than the last time
    anything happened on the deal? Pure SQL, and this module.
*   **Contradiction** -- does new evidence contradict a live claim? Needs
    semantic comparison, so it is a model task and lives in
    ``app/ai/reconcile.py``.

Why staleness matters at all: a claim can be perfectly grounded and no longer
*true*. "Security sign-off is outstanding" was cited correctly in August and is
wrong by September, and nothing about the citation changes when that happens.
Flagging it is the difference between a panel that ages gracefully and one that
confidently reports last month.

The comparison is against ``deals.last_activity_at`` rather than ``now()``,
which is the point: a deal nobody has touched for three months has no stale
claims, because nothing has happened that could have superseded them. It is
*activity* that ages a claim, not time.

``stale`` is written only over ``verified``. A link already marked
``span_missing`` or ``value_drifted`` has a worse problem, and overwriting it
would lose the more serious diagnosis.
"""

import logging
import uuid
from typing import List, Optional

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ClaimEvidence, Deal, Evidence
from app.models.enums import ClaimType, VerificationStatus

logger = logging.getLogger("cognideal.services.gate2")


async def mark_stale_claims(
    db: AsyncSession, deal_id: uuid.UUID, claim_type: Optional[ClaimType] = None
) -> List[uuid.UUID]:
    """Flag links whose claim's newest evidence predates the last activity.

    Returns the claim ids affected. Scoped to one deal because
    ``last_activity_at`` is per deal, and optionally to one claim type -- the
    pipeline runs it for facts, while a later sweep may want all five.
    """
    last_activity = await db.scalar(
        select(Deal.last_activity_at).where(Deal.id == deal_id)
    )
    if last_activity is None:
        # Nothing has happened on this deal, so nothing can have aged.
        return []

    # Newest evidence per claim, for claims on this deal.
    newest = (
        select(
            ClaimEvidence.claim_type,
            ClaimEvidence.claim_id,
            func.max(Evidence.occurred_at).label("newest"),
        )
        .join(Evidence, Evidence.id == ClaimEvidence.evidence_id)
        .where(Evidence.deal_id == deal_id)
        .group_by(ClaimEvidence.claim_type, ClaimEvidence.claim_id)
    )
    if claim_type is not None:
        newest = newest.where(ClaimEvidence.claim_type == claim_type)
    newest = newest.subquery()

    stale_claims = [
        row[1]
        for row in (
            await db.execute(
                select(newest.c.claim_type, newest.c.claim_id).where(
                    and_(newest.c.newest.is_not(None), newest.c.newest < last_activity)
                )
            )
        ).all()
    ]
    if not stale_claims:
        return []

    await db.execute(
        update(ClaimEvidence)
        .where(
            ClaimEvidence.claim_id.in_(stale_claims),
            # Only over `verified`: anything else has a worse problem already
            # diagnosed, and overwriting it would lose that.
            ClaimEvidence.verification_status == VerificationStatus.VERIFIED,
        )
        .values(verification_status=VerificationStatus.STALE, verified_at=func.now())
    )
    await db.flush()
    logger.info(
        "gate2.stale deal=%s claims=%d last_activity=%s",
        deal_id, len(stale_claims), last_activity,
    )
    return stale_claims

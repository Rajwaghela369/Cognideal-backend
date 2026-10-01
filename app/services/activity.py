"""``deals.last_activity_at`` -- the one column that reads "has anyone talked
to them?".

Nothing wrote it before this module existed, which made two things quietly
wrong: the ``stale_days`` filter on ``GET /deals`` already read the column and
so matched nothing, and Gate 2 staleness (docs/schema/README.md section 5) has
no input at all without it.

Two rules, and both matter more than they look:

**Monotonic.** The write is ``greatest(existing, moment)``, never an
assignment. Activity is a high-water mark; a backfilled transcript from June
must not make a deal that was touched yesterday look three months cold. Plain
assignment would mean the column's value depended on the order files happened
to be uploaded in.

**Dated by when it happened, not when it was recorded.** A document contributes
its ``occurred_at``, which is the same distinction ``documents`` draws between
``occurred_at`` and ``uploaded_at``: "recency ranking must use when the
conversation happened, not when the file was dragged in". Using ``now()`` here
would let an archive import mark every deal as freshly active, silently
resolving every ``gone_quiet`` risk in the pipeline.

Distinct from ``DAYS_IN_STAGE`` in ``queries.py``, and the comment there is the
clearest statement of why both exist: this one asks whether anyone is talking
to them, that one asks whether any of the talking is moving the deal.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Deal


async def touch_deal(
    db: AsyncSession,
    deal_id: Optional[uuid.UUID],
    when: Optional[datetime] = None,
) -> None:
    """Advance ``last_activity_at`` to ``when`` (default: now) if that is later.

    Issued as an UPDATE rather than through the ORM so the comparison happens
    in Postgres. Reading the row, comparing in Python and writing it back would
    lose the race between two concurrent writes on one deal -- and the whole
    point of the column is that several unrelated write paths feed it.

    ``deal_id`` is optional because ``documents`` may be attached to an account
    with no deal; there is nothing to touch in that case.

    Does not commit. The caller's transaction owns this, so activity is
    recorded exactly when the thing that caused it is -- a rolled-back upload
    must not leave the deal looking active.
    """
    if deal_id is None:
        return

    moment = func.now() if when is None else when
    await db.execute(
        update(Deal)
        .where(Deal.id == deal_id)
        .values(
            last_activity_at=func.greatest(
                func.coalesce(Deal.last_activity_at, moment), moment
            )
        )
    )

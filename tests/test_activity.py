"""``deals.last_activity_at`` -- monotonic, and dated by when things happened.

The regression these guard against is subtle and silent: an archive import of
old transcripts marking every deal as freshly active, which would resolve every
`gone_quiet` risk in the pipeline without anyone touching a deal.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import Deal
from app.services import activity


async def current(deal_id):
    async with SessionLocal() as session:
        return await session.scalar(
            select(Deal.last_activity_at).where(Deal.id == deal_id)
        )


async def test_starts_unset(deal_id):
    assert await current(deal_id) is None


async def test_advances_on_a_newer_timestamp(db, deal_id):
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    await activity.touch_deal(db, deal_id, yesterday)
    await db.commit()
    assert await current(deal_id) is not None


async def test_an_older_timestamp_does_not_regress_it(db, deal_id):
    """The high-water mark rule. A June transcript uploaded today is June."""
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    long_ago = datetime.now(timezone.utc) - timedelta(days=200)
    await activity.touch_deal(db, deal_id, yesterday)
    await db.commit()
    high_water = await current(deal_id)

    await activity.touch_deal(db, deal_id, long_ago)
    await db.commit()
    assert await current(deal_id) == high_water


async def test_default_is_now_and_advances(db, deal_id):
    old = datetime.now(timezone.utc) - timedelta(days=30)
    await activity.touch_deal(db, deal_id, old)
    await db.commit()
    before = await current(deal_id)

    await activity.touch_deal(db, deal_id)
    await db.commit()
    assert await current(deal_id) > before


async def test_a_null_deal_is_a_no_op(db):
    """Documents may be attached to an account with no deal."""
    await activity.touch_deal(db, None)
    await db.commit()

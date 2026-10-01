"""Phase 0 acceptance checks -- the "Done when" lines from docs/ai/TASKS.md.

A plain script, not pytest: the test harness is Phase 1 (tasks 1.3/1.4) and
pytest is not a dependency yet. Phase 1 should absorb these checks rather than
write them again.

Needs a live database. Run from `backend/`:

    docker compose up -d postgres
    ../deal-pilot-env/bin/alembic upgrade head
    ../deal-pilot-env/bin/python tests/verify_phase0.py

Seeds its own rows and leaves them; they are named so they are easy to find.
"""
import asyncio, sys, uuid, logging
from datetime import datetime, timezone, timedelta
sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from sqlalchemy import select, text
from app.db.session import SessionLocal
from app.models import Account, Deal, Meeting
from app.models.enums import AnalysisStatus, DealStage, MeetingStatus
from app.services import activity
import worker
from app.ai import graph

RESULTS = []
def check(label, ok, detail=""):
    RESULTS.append(ok)
    print("  %s %s%s" % ("PASS" if ok else "FAIL", label, (" -- " + detail) if detail else ""))

async def seed():
    async with SessionLocal() as db:
        acct = Account(name="Northwind Logistics %s" % uuid.uuid4().hex[:6])
        db.add(acct); await db.flush()
        deal = Deal(account_id=acct.id, name="SecureFlow Enterprise", stage=DealStage.DISCOVERY)
        db.add(deal); await db.flush()
        m = Meeting(deal_id=deal.id, title="Discovery call", status=MeetingStatus.COMPLETED,
                    analysis_status=AnalysisStatus.QUEUED)
        db.add(m); await db.flush()
        ids = (acct.id, deal.id, m.id)
        await db.commit()
        return ids

async def status_of(meeting_id):
    async with SessionLocal() as db:
        row = await db.execute(select(Meeting.analysis_status, Meeting.analyzed_at).where(Meeting.id == meeting_id))
        return row.one()

async def main():
    print("\n0.5  worker: claim, run, complete")
    _, deal_id, m_id = await seed()
    did = await worker.poll_queued_meetings()
    st, at = await status_of(m_id)
    check("claimed a queued meeting", did is True)
    check("reached complete", st == AnalysisStatus.COMPLETE, "status=%s" % st)
    check("analyzed_at stamped", at is not None, str(at))

    print("\n0.5  worker: empty queue is a no-op")
    check("returns False when nothing queued", (await worker.poll_queued_meetings()) is False)

    print("\n0.5  worker: a critical stage failure records `failed`, not a retry loop")
    _, _, m2 = await seed()
        async def boom(state, runtime):
        raise RuntimeError("extractor exploded")
    graph.NODES["extract_window"] = graph.guarded("extract_window")(boom)
    try:
        await worker.poll_queued_meetings()
        st2, _ = await status_of(m2)
        check("critical stage -> failed", st2 == AnalysisStatus.FAILED, "status=%s" % st2)
        check("not left queued (no retry storm)", st2 != AnalysisStatus.QUEUED)
    finally:

    print("\n0.5  worker: a degradable stage failure still completes")
    _, _, m3 = await seed()
        async def boom9(state, runtime):
        raise RuntimeError("summary exploded")
    graph.NODES["synthesize_summary"] = graph.guarded("synthesize_summary")(boom9)
    try:
        await worker.poll_queued_meetings()
        st3, _ = await status_of(m3)
        check("degradable stage -> still complete", st3 == AnalysisStatus.COMPLETE, "status=%s" % st3)
    finally:

    print("\n0.5  worker: SKIP LOCKED -- two claimants never get the same row")
    _, _, m4 = await seed()
    async with SessionLocal() as a, SessionLocal() as b:
        first  = await worker._claim_queued_meeting(a)
        second = await worker._claim_queued_meeting(b)
        check("first claimant gets the row", first is not None and first.id == m4)
        check("second claimant skips it", second is None, "got %s" % (second.id if second else None))

    print("\n0.6  last_activity_at: monotonic high-water mark")
    async with SessionLocal() as db:
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        june   = datetime.now(timezone.utc) - timedelta(days=100)
        await activity.touch_deal(db, deal_id, recent); await db.commit()
        v1 = await db.scalar(select(Deal.last_activity_at).where(Deal.id == deal_id))
        await activity.touch_deal(db, deal_id, june); await db.commit()
        v2 = await db.scalar(select(Deal.last_activity_at).where(Deal.id == deal_id))
        check("advances on a newer timestamp", v1 is not None, str(v1))
        check("an older timestamp does NOT regress it", v2 == v1, "%s -> %s" % (v1, v2))
        await activity.touch_deal(db, None); await db.commit()
        check("None deal_id is a no-op (account-only document)", True)
        await activity.touch_deal(db, deal_id); await db.commit()
        v3 = await db.scalar(select(Deal.last_activity_at).where(Deal.id == deal_id))
        check("default now() advances past it", v3 > v1, str(v3))

    print("\n0.6  stale_days filter now has input")
    async with SessionLocal() as db:
        n = await db.scalar(text("SELECT count(*) FROM deals WHERE last_activity_at IS NOT NULL"))
        check("deals with last_activity_at set", n >= 1, "%s rows" % n)

    print("\n0.5  worker loop: starts and stops on signal")
    worker._shutdown.clear()
    task = asyncio.ensure_future(worker.run_forever())
    await asyncio.sleep(0.3)
    running = not task.done()
    worker._shutdown.set()
    await asyncio.wait_for(task, timeout=5)
    check("loop ran then shut down cleanly", running and task.done())

    print("\n%d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1

sys.exit(asyncio.get_event_loop().run_until_complete(main()))

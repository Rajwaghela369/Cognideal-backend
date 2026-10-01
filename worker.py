"""The background worker.

``POST /deals/{id}/meetings/{id}/analysis`` has always set ``queued`` and
nothing consumed it. This is the consumer.

**Why there is no Celery, no Redis and no scheduler.** Compose has neither a
broker nor a cron container, and the work is already queued *in Postgres* --
``analysis_status`` is the queue. ``FOR UPDATE SKIP LOCKED`` is exactly the
primitive a job queue needs, so adding a broker would mean two sources of
truth about what is pending. Phase 10 adds two more poll queries to this same
loop (dirty deals, and the nightly sweep), which is why the loop is written as
a list of pollers rather than one query.

**Crash semantics.** The transaction is held across the whole run and committed
at the end. Kill the process mid-run and the lock dies with the connection, the
row is still ``queued``, and the next worker picks it up -- self-healing, with
no reaper process and no lease timestamps to expire.

The alternative -- claim the row, commit ``running``, then work -- looks more
observable but leaves a crashed run stuck in ``running`` forever, which needs a
reaper to fix. That trade is worth revisiting only when the UI needs to show
progress; the note is in docs/ai/TASKS.md task 0.5.

A *handled* failure in a critical stage is different from a crash: that is not
going to succeed on retry, so it is recorded as ``failed`` in a second
transaction rather than left to spin.

Run it with ``python -m worker`` from ``backend/``, or as the ``worker`` compose
service.
"""

import asyncio
import logging
import signal
from typing import Awaitable, Callable, List, Optional, Tuple

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.client import RunBudget
from app.ai.pipeline import CriticalStageFailed, run_meeting_analysis
from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Meeting
from app.models.enums import AnalysisStatus

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
)
logger = logging.getLogger("dealpilot.worker")

_shutdown = asyncio.Event()

# A poller claims at most one unit of work and returns True if it did. Returning
# False means "nothing to do", which is what puts the loop to sleep.
Poller = Tuple[str, Callable[[], Awaitable[bool]]]


async def _record_meeting_failure(meeting_id, error: str) -> None:
    """Mark a meeting ``failed`` in its own transaction.

    Separate transaction on purpose: the run's transaction has been rolled
    back, so the row is unlocked and nothing of the failed attempt survives.
    Writing the status in the rolled-back transaction would roll the status
    back too, and the meeting would be retried forever.
    """
    async with SessionLocal() as db:
        await db.execute(
            update(Meeting)
            .where(Meeting.id == meeting_id)
            .values(analysis_status=AnalysisStatus.FAILED)
        )
        await db.commit()
    # meetings has no analysis_error column, so the reason lives in the log.
    # Worth adding one in migration 0010 -- noted in docs/ai/TASKS.md.
    logger.error("meeting.analysis_failed meeting=%s error=%s", meeting_id, error)


async def poll_queued_meetings() -> bool:
    """Claim one queued meeting and analyse it."""
    async with SessionLocal() as db:
        meeting = await _claim_queued_meeting(db)
        if meeting is None:
            return False

        meeting_id = meeting.id
        logger.info("meeting.analysis_started meeting=%s", meeting_id)
        try:
            state = await run_meeting_analysis(db, meeting, RunBudget())
        except CriticalStageFailed as exc:
            await db.rollback()
            await _record_meeting_failure(meeting_id, str(exc))
            return True
        except Exception as exc:  # noqa: BLE001
            # Not attributable to a stage: leave the row queued and let it be
            # retried. An infrastructure blip should not consume the work item.
            await db.rollback()
            logger.exception("meeting.analysis_crashed meeting=%s error=%r", meeting_id, exc)
            return True

        meeting.analysis_status = AnalysisStatus.COMPLETE
        # Stage 11 (`finalize`) owns analyzed_at once it exists; until then the
        # worker stamps it so the Analyzer screen has a completion time.
        if meeting.analyzed_at is None:
            meeting.analyzed_at = func.now()
        await db.commit()

        logger.info(
            "meeting.analysis_complete meeting=%s tokens=%d degraded=%s",
            meeting_id, state.budget.spent, sorted(state.stage_errors) or "none",
        )
        return True


async def _claim_queued_meeting(db: AsyncSession) -> Optional[Meeting]:
    """One queued meeting, locked for the life of this transaction.

    ``SKIP LOCKED`` is what makes more than one worker safe: a second worker
    steps over the locked row instead of blocking on it. Oldest first, so a
    burst of uploads is analysed in the order it arrived.
    """
    return await db.scalar(
        select(Meeting)
        .where(Meeting.analysis_status == AnalysisStatus.QUEUED)
        .order_by(Meeting.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )


# Ordered by urgency. Phase 10 appends `poll_dirty_deals` and `poll_sweep` here;
# the loop needs no change to carry them.
POLLERS: List[Poller] = [
    ("queued_meetings", poll_queued_meetings),
]


async def run_forever() -> None:
    logger.info(
        "worker.started pollers=%s poll_interval=%ss ai_enabled=%s",
        [name for name, _ in POLLERS], settings.worker_poll_seconds, settings.ai_enabled,
    )
    while not _shutdown.is_set():
        did_work = False
        for name, poller in POLLERS:
            if _shutdown.is_set():
                break
            try:
                did_work = await poller() or did_work
            except Exception:  # noqa: BLE001 -- one poller must not kill the loop
                logger.exception("worker.poller_error poller=%s", name)

        if not did_work:
            # Wait on the shutdown event rather than sleeping, so SIGTERM is
            # acted on immediately instead of after the poll interval.
            try:
                await asyncio.wait_for(
                    _shutdown.wait(), timeout=settings.worker_poll_seconds
                )
            except asyncio.TimeoutError:
                pass

    logger.info("worker.stopped")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Finish the current unit of work, then exit.

    Without this, `docker compose down` sends SIGTERM and Python raises
    immediately -- rolling back a run that was nearly done. With it the row
    stays queued either way, but the log says the worker stopped rather than
    crashed, which is the difference between a clean deploy and an incident.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown.set)
        except NotImplementedError:  # pragma: no cover -- Windows
            signal.signal(sig, lambda *_: _shutdown.set())


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop)
    try:
        loop.run_until_complete(run_forever())
    finally:
        loop.close()


if __name__ == "__main__":
    main()

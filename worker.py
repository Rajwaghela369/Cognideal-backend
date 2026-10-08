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
import time
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.client import RunBudget
from app.ai.graph import run_meeting_analysis
from app.ai.stage_registry import CriticalStageFailed
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import SessionLocal
from app.models import ChatMessage, Deal, Meeting
from app.models.enums import AnalysisStatus, MessageStatus
from app.services import analysis as analysis_service

configure_logging()
logger = logging.getLogger("cognideal.worker")

_shutdown = asyncio.Event()

# A poller claims at most one unit of work and returns True if it did. Returning
# False means "nothing to do", which is what puts the loop to sleep.
Poller = Tuple[str, Callable[[], Awaitable[bool]]]


class _Backoff:
    """Crash accounting per work item, so a failing item is not hot-looped.

    A crash rolls back and leaves the row claimable -- right for a transient
    blip, but the claim queries order oldest-first, so the same row came
    straight back on the next iteration, with no sleep (the poller reported
    work), and every attempt paid for its model calls again. Now a crashed item
    sits out an exponential backoff and, after ``worker_max_attempts``, is
    given up on by the poller that owns it.

    In memory, per process, on purpose: a restart is a reasonable moment to try
    again, and persisting attempts would need a migration for every queue.
    """

    def __init__(self) -> None:
        self._state: Dict[Tuple[str, object], Tuple[int, float]] = {}

    def cooling(self, kind: str) -> List[object]:
        """Ids of this kind that must not be claimed yet."""
        now = time.monotonic()
        return [
            item_id for (k, item_id), (_, retry_at) in self._state.items()
            if k == kind and retry_at > now
        ]

    def crashed(self, kind: str, item_id: object) -> int:
        """Record a crash and schedule the retry; returns attempts so far."""
        attempts = self._state.get((kind, item_id), (0, 0.0))[0] + 1
        delay = settings.worker_retry_backoff_seconds * (2 ** (attempts - 1))
        self._state[(kind, item_id)] = (attempts, time.monotonic() + delay)
        if attempts < settings.worker_max_attempts:
            logger.warning(
                "worker.retry_scheduled kind=%s id=%s attempt=%d/%d in=%.0fs",
                kind, item_id, attempts, settings.worker_max_attempts, delay,
            )
        return attempts

    def exhausted(self, attempts: int) -> bool:
        return attempts >= settings.worker_max_attempts

    def clear(self, kind: str, item_id: object) -> None:
        self._state.pop((kind, item_id), None)


_backoff = _Backoff()


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
            .values(analysis_status=AnalysisStatus.FAILED, analysis_error=error)
        )
        await db.commit()
    logger.error("meeting.analysis_failed meeting=%s error=%s", meeting_id, error)


async def poll_queued_meetings() -> bool:
    """Claim one queued meeting and analyse it."""
    async with SessionLocal() as db:
        meeting = await _claim_queued_meeting(db)
        if meeting is None:
            return False
        meeting_id = meeting.id
        logger.info("meeting.analysis_started meeting=%s deal=%s", meeting_id, meeting.deal_id)
        started = time.monotonic()
        budget = RunBudget()
        try:
            state = await run_meeting_analysis(db, meeting, budget)
            # `analysis_status` and `analyzed_at` are `finalize`'s, so that one
            # writer owns them. This is the floor underneath it, not a second
            # writer: `finalize` is a *degradable* stage, so if it raises,
            # `guarded` swallows it -- and a run that never set a status is a
            # row the next poll finds `queued` and re-runs.
            if meeting.analysis_status != AnalysisStatus.COMPLETE:
                logger.warning(
                    "meeting.finalize_missed meeting=%s status=%s -- completing anyway",
                    meeting_id, meeting.analysis_status,
                )
                meeting.analysis_status = AnalysisStatus.COMPLETE
                meeting.analyzed_at = func.now()
            # Inside the try: a failed commit is a crash like any other and must
            # go through the backoff, not escape to the loop and leave the row
            # to be re-claimed immediately.
            await db.commit()
        except CriticalStageFailed as exc:
            await db.rollback()
            _backoff.clear("meeting", meeting_id)
            await _record_meeting_failure(meeting_id, str(exc))
            return True
        except Exception as exc:  # noqa: BLE001
            # Not attributable to a stage: leave the row queued and retry with
            # backoff. An infrastructure blip should not consume the work item,
            # and a persistent fault must not loop forever either.
            await db.rollback()
            logger.exception("meeting.analysis_crashed meeting=%s error=%r", meeting_id, exc)
            if _backoff.exhausted(_backoff.crashed("meeting", meeting_id)):
                _backoff.clear("meeting", meeting_id)
                await _record_meeting_failure(
                    meeting_id,
                    "gave up after %d attempts: %s: %s"
                    % (settings.worker_max_attempts, type(exc).__name__, exc),
                )
            return True
        _backoff.clear("meeting", meeting_id)
        logger.info(
            "meeting.analysis_complete meeting=%s seconds=%.1f tokens=%d facts=%d degraded=%s",
            meeting_id, time.monotonic() - started, budget.spent,
            len(state.get("written_fact_ids") or []),
            sorted(state.get("stage_errors") or {}) or "none",
        )
        return True


async def _claim_queued_meeting(db: AsyncSession) -> Optional[Meeting]:
    """One queued meeting, locked for the life of this transaction.

    ``SKIP LOCKED`` is what makes more than one worker safe: a second worker
    steps over the locked row instead of blocking on it. Oldest first, so a
    burst of uploads is analysed in the order it arrived. Meetings in crash
    backoff are skipped.
    """
    stmt = select(Meeting).where(Meeting.analysis_status == AnalysisStatus.QUEUED)
    cooling = _backoff.cooling("meeting")
    if cooling:
        stmt = stmt.where(Meeting.id.notin_(cooling))
    return await db.scalar(
        stmt.order_by(Meeting.created_at).limit(1).with_for_update(skip_locked=True)
    )


async def _run_deal(kind: str, *, clear_dirty: bool) -> bool:
    """Claim one deal for ``kind`` and run its Tier 2 analysis.

    Shared by the dirty-deal and sweep pollers, which differ only in which
    deal they claim and whether success clears the dirty flags.
    """
    claim = _claim_dirty_deal if clear_dirty else _claim_sweep_deal
    async with SessionLocal() as db:
        deal = await claim(db)
        if deal is None:
            return False
        deal_id = deal.id
        logger.info(
            "deal.%s_started deal=%s reason=%s", kind, deal_id,
            deal.analysis_dirty_reason if clear_dirty else "sweep",
        )
        started = time.monotonic()
        budget = RunBudget()
        try:
            await analysis_service.run_deal_analysis(db, deal_id, budget=budget)
            if clear_dirty:
                deal.analysis_dirty_first_at = None
                deal.analysis_dirty_last_at = None
                deal.analysis_dirty_reason = None
            deal.analysis_swept_at = func.now()
            await db.commit()
        except Exception:  # noqa: BLE001 -- rollback leaves the enqueue intact
            await db.rollback()
            logger.exception("deal.%s_crashed deal=%s", kind, deal_id)
            if _backoff.exhausted(_backoff.crashed(kind, deal_id)):
                _backoff.clear(kind, deal_id)
                await _give_up_on_deal(deal_id, clear_dirty=clear_dirty)
            return True
        _backoff.clear(kind, deal_id)
        logger.info(
            "deal.%s_complete deal=%s seconds=%.1f tokens=%d",
            kind, deal_id, time.monotonic() - started, budget.spent,
        )
        return True


async def _give_up_on_deal(deal_id, *, clear_dirty: bool) -> None:
    """Stop re-claiming a deal whose analysis keeps crashing.

    Stamps ``analysis_swept_at`` so the sweep leaves it for a full period, and
    for a dirty deal clears the flags -- the next real change re-enqueues it,
    which is a better moment to try than the next poll.
    """
    values = {"analysis_swept_at": func.now()}
    if clear_dirty:
        values.update(
            analysis_dirty_first_at=None,
            analysis_dirty_last_at=None,
            analysis_dirty_reason=None,
        )
    async with SessionLocal() as db:
        await db.execute(update(Deal).where(Deal.id == deal_id).values(**values))
        await db.commit()
    logger.error(
        "deal.analysis_gave_up deal=%s attempts=%d", deal_id, settings.worker_max_attempts
    )


async def poll_dirty_deals() -> bool:
    """Run one debounced Tier 2 analysis with a row lock as single-flight."""
    return await _run_deal("dirty_deal", clear_dirty=True)


async def _claim_dirty_deal(db: AsyncSession) -> Optional[Deal]:
    now = datetime.now(timezone.utc)
    quiet_cutoff = now - timedelta(seconds=settings.analysis_debounce_seconds)
    max_cutoff = now - timedelta(seconds=settings.analysis_max_debounce_seconds)
    stmt = select(Deal).where(
        Deal.analysis_dirty_first_at.is_not(None),
        (
            (Deal.analysis_dirty_last_at < quiet_cutoff)
            | (Deal.analysis_dirty_first_at < max_cutoff)
        ),
    )
    cooling = _backoff.cooling("dirty_deal")
    if cooling:
        stmt = stmt.where(Deal.id.notin_(cooling))
    return await db.scalar(
        stmt.order_by(Deal.analysis_dirty_last_at).limit(1).with_for_update(skip_locked=True)
    )


async def poll_sweep() -> bool:
    """Refresh one clean deal whose clock-dependent analysis is stale."""
    return await _run_deal("sweep", clear_dirty=False)


async def _claim_sweep_deal(db: AsyncSession) -> Optional[Deal]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.analysis_sweep_hours)
    stmt = select(Deal).where(
        Deal.analysis_dirty_first_at.is_(None),
        (
            Deal.analysis_swept_at.is_(None)
            | (Deal.analysis_swept_at < cutoff)
        ),
    )
    cooling = _backoff.cooling("sweep")
    if cooling:
        stmt = stmt.where(Deal.id.notin_(cooling))
    return await db.scalar(
        stmt.order_by(Deal.analysis_swept_at.asc().nulls_first())
        .limit(1)
        .with_for_update(skip_locked=True)
    )


async def reap_abandoned_streams() -> bool:
    """Finalize chat turns whose client never came back. Task 11.6.

    Task 9.3 commits an empty assistant row with `status='streaming'` *before*
    inference, so a refresh mid-answer can recover the turn. The gap was that
    nothing ever closed a row whose reader closed the tab: the generator stops
    being consumed, the request task is cancelled, and the row stays `streaming`
    forever -- read by the messages route as a turn still in flight.

    `error` rather than `complete`, with whatever content was checkpointed left
    in place. The answer was interrupted and may stop mid-sentence; calling that
    complete would present a truncated answer as a finished one, and partial
    content with an honest status is the more useful of the two.

    The window has to exceed the longest plausible single turn, or this reaps
    live streams -- `created_at` is set when the row is inserted, which is before
    the model has produced anything.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=settings.chat_stream_timeout_seconds
    )
    async with SessionLocal() as db:
        rows = list(
            (
                await db.scalars(
                    select(ChatMessage)
                    .where(
                        ChatMessage.status == MessageStatus.STREAMING,
                        ChatMessage.created_at < cutoff,
                    )
                    .limit(20)
                    .with_for_update(skip_locked=True)
                )
            ).all()
        )
        if not rows:
            return False
        for row in rows:
            row.status = MessageStatus.ERROR
            if not row.content:
                row.content = "(interrupted before any output was produced)"
            logger.info(
                "chat.stream_reaped message=%s session=%s chars=%d",
                row.id, row.session_id, len(row.content),
            )
        await db.commit()
        return True


# Ordered by urgency. Phase 10 appends `poll_dirty_deals` and `poll_sweep` here;
# the loop needs no change to carry them, which is also why 11.6's reaper is a
# four-line addition rather than a second process.
POLLERS: List[Poller] = [
    ("queued_meetings", poll_queued_meetings),
    ("dirty_deals", poll_dirty_deals),
    ("sweep", poll_sweep),
    ("abandoned_streams", reap_abandoned_streams),
]


def stop() -> None:
    """Ask the loop to finish its current unit of work and return."""
    _shutdown.set()


async def run_forever(*, embedded: bool = False) -> None:
    """The poll loop. Runs as its own process (``python -m worker``) or
    inside the API (``EMBEDDED_WORKER=true``, see ``app.main``) -- the same
    loop either way, and ``SKIP LOCKED`` keeps the two safe side by side.
    """
    global _shutdown
    # Re-created on the running loop unless a stop is already pending: before
    # Python 3.10 an Event binds to whatever loop existed when it was built,
    # and the module-level one predates uvicorn's.
    if not _shutdown.is_set():
        _shutdown = asyncio.Event()
    logger.info(
        "worker.started mode=%s pollers=%s poll_interval=%ss ai_enabled=%s",
        "embedded" if embedded else "standalone",
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
            loop.add_signal_handler(sig, stop)
        except NotImplementedError:  # pragma: no cover -- Windows
            signal.signal(sig, lambda *_: stop())


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

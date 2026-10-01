"""Worker claim and degradation semantics.

Absorbed from ``verify_phase0.py``, which was a plain script written before
pytest was a dependency. The crash test and the HTTP round trip stay in
``verify_phase0_http.py``: one needs a subprocess kill, the other a running
server, and neither belongs in the default suite.
"""

import pytest
import pytest_asyncio
from sqlalchemy import select

import worker
from app.ai import pipeline
from app.db.session import SessionLocal
from app.models import Meeting
from app.models.enums import AnalysisStatus, MeetingStatus


@pytest_asyncio.fixture
async def queued_meeting(db, deal_id):
    meeting = Meeting(
        deal_id=deal_id, title="Discovery call",
        status=MeetingStatus.COMPLETED, analysis_status=AnalysisStatus.QUEUED,
    )
    db.add(meeting)
    await db.flush()
    created = meeting.id
    await db.commit()
    return created


async def status_of(meeting_id):
    async with SessionLocal() as session:
        return await session.scalar(
            select(Meeting.analysis_status).where(Meeting.id == meeting_id)
        )


@pytest.fixture
def broken_stage():
    """Swap one stage for a failing one, then put it back."""
    originals = list(pipeline.STAGES)

    def _break(index, critical):
        async def boom(db, state):
            raise RuntimeError("stage %d exploded" % index)

        pipeline.STAGES[index] = pipeline.Stage(
            index=index, name=originals[index].name, critical=critical, run=boom
        )

    yield _break
    pipeline.STAGES[:] = originals


async def test_claims_and_completes(queued_meeting):
    assert await worker.poll_queued_meetings() is True
    assert await status_of(queued_meeting) == AnalysisStatus.COMPLETE


async def test_stamps_analyzed_at(queued_meeting):
    await worker.poll_queued_meetings()
    async with SessionLocal() as session:
        at = await session.scalar(
            select(Meeting.analyzed_at).where(Meeting.id == queued_meeting)
        )
    assert at is not None


async def test_empty_queue_is_a_no_op():
    assert await worker.poll_queued_meetings() is False


async def test_critical_stage_failure_records_failed(queued_meeting, broken_stage):
    """Stages 0-4 produce the facts. Without them nothing was analysed.

    Recorded as `failed` rather than left `queued`, because a handled failure
    will not succeed on retry -- leaving it queued is a retry storm.
    """
    broken_stage(2, critical=True)
    await worker.poll_queued_meetings()
    assert await status_of(queued_meeting) == AnalysisStatus.FAILED


async def test_degradable_stage_failure_still_completes(queued_meeting, broken_stage):
    """A failed summary must not discard facts that already landed."""
    broken_stage(9, critical=False)
    await worker.poll_queued_meetings()
    assert await status_of(queued_meeting) == AnalysisStatus.COMPLETE


async def test_skip_locked_prevents_double_claiming(queued_meeting):
    """Two workers must never get the same row.

    `SKIP LOCKED` makes the second claimant step over the locked row rather
    than block on it, which is what lets more than one worker run at all.
    """
    async with SessionLocal() as first, SessionLocal() as second:
        claimed = await worker._claim_queued_meeting(first)
        skipped = await worker._claim_queued_meeting(second)
        assert claimed is not None and claimed.id == queued_meeting
        assert skipped is None

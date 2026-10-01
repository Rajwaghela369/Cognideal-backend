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
from app.ai import graph
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
def broken_stage(monkeypatch):
    """Make one graph node raise.

    Patches the node the graph actually runs rather than a registry entry, so
    the test exercises the path the worker takes. The critical/degradable
    classification comes from `stage_registry`, which is the thing under test
    -- it is not passed in.
    """

    def _break(node_name):
        async def boom(state, runtime):
            raise RuntimeError("%s exploded" % node_name)

        monkeypatch.setitem(graph.NODES, node_name, graph.guarded(node_name)(boom))

    return _break


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
    broken_stage("parse_transcript")
    await worker.poll_queued_meetings()
    assert await status_of(queued_meeting) == AnalysisStatus.FAILED


def feed_the_branch(monkeypatch):
    """Make `drop_unevidenced` report one surviving fact.

    `anything_survived` routes to END when nothing survived -- correct, and it
    means the degradable stages are unreachable for a meeting with no
    transcript. Stands in for a real extraction so the test needs no model and
    no fixture corpus.

    Via `monkeypatch.setitem`, not a plain assignment: `graph.NODES` is module
    state, and mutating it leaks into every test that runs afterwards.
    """

    async def one_survivor(state, runtime):
        return {"surviving_facts": ["stand-in"], "written_fact_ids": ["stand-in"],
                "rejected_facts": []}

    monkeypatch.setitem(
        graph.NODES, "drop_unevidenced", graph.guarded("drop_unevidenced")(one_survivor)
    )


async def test_degradable_stage_failure_still_completes(queued_meeting, broken_stage, monkeypatch):
    """A failed summary must not discard facts that already landed.

    The rule the first graph conversion dropped: a raw LangGraph node raising
    propagates out of `ainvoke` and rolls back the transaction, verified facts
    included. `graph.guarded` restores it, and this is the assertion that
    would have caught its absence.
    """
    # The late stages are only reached when something survived Gate 0, so the
    # branch has to be fed: a meeting with no transcript ends at
    # `drop_unevidenced`. This stands in for a real extraction.
    feed_the_branch(monkeypatch)
    broken_stage("synthesize_summary")
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


# --------------------------------------------------------------------------
# the graph and the registry must not drift
# --------------------------------------------------------------------------


def test_every_declared_stage_is_a_node():
    """`stage_registry.STAGES` is the executable copy of README section 4.

    A stage declared there and missing from the graph is a stage that silently
    never runs -- which is exactly the class of bug the conversion from the
    sequential runner could have introduced.
    """
    from app.ai import stage_registry

    nodes = {n for n in graph.build_graph().get_graph().nodes if not n.startswith("__")}
    for stage in stage_registry.STAGES:
        assert stage.name in nodes, "stage %d (%s) is not a node" % (stage.index, stage.name)


def test_every_node_is_declared_or_structural():
    """The reverse: a node nobody declared.

    `plan_windows` is the one legitimate extra -- a LangGraph node returns
    state while an edge returns destinations, so the fan-out needs an origin to
    hang its conditional edge off.
    """
    from app.ai import stage_registry

    declared = {stage.name for stage in stage_registry.STAGES} | {"plan_windows"}
    nodes = {n for n in graph.build_graph().get_graph().nodes if not n.startswith("__")}
    assert nodes - declared == set()


def test_the_critical_split_is_stages_zero_to_four():
    from app.ai import stage_registry

    critical = [s.name for s in stage_registry.STAGES if s.critical]
    assert critical == [s.name for s in stage_registry.STAGES[:5]]
    assert stage_registry.LATE_STAGES == [
        s.name for s in stage_registry.STAGES if not s.critical
    ]

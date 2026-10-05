"""Phase 11: the correction surface, the state read, the reaper, the write tool."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import worker
from app.ai import stages
from app.ai.tools import PROPOSE_VERSION, ToolRegistry
from app.core.config import settings
from app.db.session import SessionLocal
from app.models import ChatMessage, ChatSession, Commitment, Deal, Recommendation
from app.models.enums import (
    ActionType,
    ChatRole,
    ChatScope,
    CommitmentStatus,
    MessageStatus,
    Origin,
    OwnerSide,
    RecommendationStatus,
)
from langchain_core.tools import ToolException


async def _commitment(db, deal_id, description="Send the SOC 2 report"):
    commitment = Commitment(
        deal_id=deal_id,
        description=description,
        owner_side=OwnerSide.US,
        status=CommitmentStatus.PENDING,
        origin=Origin.AI,
    )
    db.add(commitment)
    await db.flush()
    return commitment


# --------------------------------------------------------------------------
# 11.2 -- stage 7 files a correction instead of logging it
# --------------------------------------------------------------------------


async def test_correction_is_filed_as_a_suggestion(deal_id):
    async with SessionLocal() as db:
        commitment = await _commitment(db, deal_id)
        rec = await stages._file_correction(
            db, deal_id=deal_id, commitment_id=commitment.id,
            why="Maya confirmed the report went out on Tuesday.", facts=[],
        )
        await db.commit()

        assert rec is not None
        assert rec.action_type == ActionType.CORRECT_RECORD.value
        assert rec.status == RecommendationStatus.SUGGESTED
        assert rec.source_commitment_id == commitment.id
        # The proposal must not be the change. `missed_commitment` is only
        # trustworthy if a model cannot satisfy the thing it measures.
        assert commitment.status == CommitmentStatus.PENDING


async def test_a_second_run_updates_rather_than_duplicates(deal_id):
    """Two meetings can show the same promise kept; the index allows one row."""
    async with SessionLocal() as db:
        commitment = await _commitment(db, deal_id)
        first = await stages._file_correction(
            db, deal_id=deal_id, commitment_id=commitment.id,
            why="first reasoning", facts=[],
        )
        first_id = first.id
        await db.commit()

    async with SessionLocal() as db:
        commitment = await db.get(Commitment, commitment.id)
        second = await stages._file_correction(
            db, deal_id=deal_id, commitment_id=commitment.id,
            why="second reasoning", facts=[],
        )
        await db.commit()
        assert second.id == first_id
        assert second.rationale == "second reasoning"

    async with SessionLocal() as db:
        rows = list((await db.scalars(
            select(Recommendation).where(
                Recommendation.source_commitment_id == commitment.id
            )
        )).all())
        assert len(rows) == 1


async def test_a_satisfied_commitment_is_not_proposed_again(deal_id):
    """Raced with a human marking it, or with another meeting's run."""
    async with SessionLocal() as db:
        commitment = await _commitment(db, deal_id)
        commitment.status = CommitmentStatus.MET
        await db.flush()
        rec = await stages._file_correction(
            db, deal_id=deal_id, commitment_id=commitment.id,
            why="too late", facts=[],
        )
        assert rec is None


# --------------------------------------------------------------------------
# 11.3 -- the derived analysis state matches what the worker would claim
# --------------------------------------------------------------------------


async def _state(deal_id):
    from app.api.v1.routes.deals.risks import get_analysis_state

    async with SessionLocal() as db:
        deal = await db.get(Deal, deal_id)
        return (await get_analysis_state(deal=deal))["state"]


async def test_state_tracks_the_debounce_windows(deal_id):
    now = datetime.now(timezone.utc)

    async with SessionLocal() as db:
        deal = await db.get(Deal, deal_id)
        deal.analysis_swept_at = now
        await db.commit()
    assert await _state(deal_id) == "clean"

    # Just marked: inside the quiet window, so more edits are still expected.
    async with SessionLocal() as db:
        deal = await db.get(Deal, deal_id)
        deal.analysis_dirty_first_at = now
        deal.analysis_dirty_last_at = now
        await db.commit()
    assert await _state(deal_id) == "debouncing"

    # Quiet long enough that the worker's claim query would take it.
    async with SessionLocal() as db:
        deal = await db.get(Deal, deal_id)
        deal.analysis_dirty_last_at = now - timedelta(
            seconds=settings.analysis_debounce_seconds + 5
        )
        await db.commit()
    assert await _state(deal_id) == "due"

    # Not dirty, but past the sweep window.
    async with SessionLocal() as db:
        deal = await db.get(Deal, deal_id)
        deal.analysis_dirty_first_at = None
        deal.analysis_dirty_last_at = None
        deal.analysis_swept_at = now - timedelta(
            hours=settings.analysis_sweep_hours + 1
        )
        await db.commit()
    assert await _state(deal_id) == "stale"


async def test_due_and_the_worker_claim_agree(deal_id):
    """The endpoint is only useful if it cannot disagree with the claim query."""
    now = datetime.now(timezone.utc)
    async with SessionLocal() as db:
        deal = await db.get(Deal, deal_id)
        deal.analysis_dirty_first_at = now - timedelta(seconds=5)
        deal.analysis_dirty_last_at = now - timedelta(
            seconds=settings.analysis_debounce_seconds + 5
        )
        await db.commit()

    assert await _state(deal_id) == "due"
    async with SessionLocal() as db:
        claimed = await worker._claim_dirty_deal(db)
        assert claimed is not None and claimed.id == deal_id


# --------------------------------------------------------------------------
# 11.6 -- abandoned streams reach a terminal status
# --------------------------------------------------------------------------


async def test_reaper_closes_only_stale_streaming_rows(deal_id):
    async with SessionLocal() as db:
        session = ChatSession(scope=ChatScope.DEAL, deal_id=deal_id)
        db.add(session)
        await db.flush()

        stale = ChatMessage(
            session_id=session.id, role=ChatRole.ASSISTANT,
            content="partial answ", status=MessageStatus.STREAMING,
        )
        live = ChatMessage(
            session_id=session.id, role=ChatRole.ASSISTANT,
            content="", status=MessageStatus.STREAMING,
        )
        db.add_all([stale, live])
        await db.flush()
        # created_at is server-set, so age it explicitly.
        stale.created_at = datetime.now(timezone.utc) - timedelta(
            seconds=settings.chat_stream_timeout_seconds + 60
        )
        stale_id, live_id = stale.id, live.id
        await db.commit()

    assert await worker.reap_abandoned_streams() is True

    async with SessionLocal() as db:
        reaped = await db.get(ChatMessage, stale_id)
        untouched = await db.get(ChatMessage, live_id)
        # `error`, not `complete`: the answer stops mid-sentence and calling it
        # complete would present a truncated answer as a finished one.
        assert reaped.status == MessageStatus.ERROR
        assert reaped.content == "partial answ"
        assert untouched.status == MessageStatus.STREAMING

        await db.delete(await db.get(ChatSession, reaped.session_id))
        await db.commit()


async def test_reaper_is_a_noop_with_nothing_to_do():
    assert await worker.reap_abandoned_streams() is False


# --------------------------------------------------------------------------
# 11.8 -- propose_task suggests, and cannot do anything else
# --------------------------------------------------------------------------


async def test_propose_task_creates_a_suggestion_not_a_task(deal_id):
    async with SessionLocal() as db:
        registry = ToolRegistry(db, deal_id)
        await registry.propose_task(
            title="Ask the CFO to confirm the budget line",
            rationale="No economic buyer has engaged in three meetings.",
        )
        await db.commit()

        rec = await db.scalar(
            select(Recommendation).where(Recommendation.deal_id == deal_id)
        )
        assert rec is not None
        assert rec.status == RecommendationStatus.SUGGESTED
        assert rec.created_task_id is None
        assert rec.origin == Origin.AI
        # How a reviewer tells an agent suggestion from a detector one.
        assert rec.detector_version == PROPOSE_VERSION

        from app.models import Task
        assert await db.scalar(select(Task).where(Task.deal_id == deal_id)) is None


async def test_propose_task_refuses_another_deal(deal_id):
    async with SessionLocal() as db:
        registry = ToolRegistry(db, deal_id)
        with pytest.raises(ToolException):
            await registry.propose_task(
                title="Write to the wrong company",
                rationale="Scope must come from the session row, not the model.",
                deal_id=str(uuid.uuid4()),
            )


async def test_propose_task_is_the_only_write_tool():
    """A regression guard on the one rule §6 states: no second write tool."""
    registry = ToolRegistry(None, None)
    names = {tool.name for tool in registry.langchain_tools()}
    assert "propose_task" in names
    writers = {n for n in names if n.startswith(("create_", "update_", "delete_", "set_"))}
    assert writers == set()

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.ai import brief, client
from app.ai.schemas import BriefOut
from app.models import Meeting, MeetingBrief
from app.models.enums import MeetingType


@pytest.mark.asyncio
async def test_brief_persists_and_second_call_reuses_it(db, deal_id, monkeypatch):
    meeting = Meeting(
        deal_id=deal_id,
        title="Executive review",
        meeting_type=MeetingType.CHECK_IN,
        scheduled_at=datetime.now(timezone.utc),
    )
    db.add(meeting)
    await db.flush()
    calls = []

    async def fake_structured(*args, **kwargs):
        calls.append(kwargs["task"])
        return BriefOut(
            objectives=["Confirm the decision process"],
            key_risks=[],
            recommended_questions=["Who gives final approval?"],
            context_summary="The deal is in discovery.",
        ), client.AIRun("brief", "test-model", "brief@1")

    monkeypatch.setattr(client, "structured", fake_structured)

    first = await brief.generate(db, meeting)
    second = await brief.generate(db, meeting)

    assert first.id == second.id
    assert calls == ["brief"]
    assert await db.get(MeetingBrief, first.id) is not None
    await db.commit()


@pytest.mark.asyncio
async def test_force_replaces_the_stored_brief(db, deal_id, monkeypatch):
    meeting = Meeting(deal_id=deal_id, title="Review")
    db.add(meeting)
    await db.flush()
    counter = [0]

    async def fake_structured(*args, **kwargs):
        counter[0] += 1
        return BriefOut(
            objectives=["Objective %d" % counter[0]],
            key_risks=[],
            recommended_questions=[],
            context_summary="Context %d" % counter[0],
        ), client.AIRun("brief", "test-model", "brief@1")

    monkeypatch.setattr(client, "structured", fake_structured)
    first = await brief.generate(db, meeting)
    first_id = first.id
    await brief.generate(db, meeting, force=True)

    rows = list((await db.scalars(
        select(MeetingBrief).where(
            MeetingBrief.meeting_id == meeting.id
        )
    )).all())
    assert len(rows) == 1
    assert rows[0].id != first_id
    assert rows[0].context_summary == "Context 2"
    await db.commit()

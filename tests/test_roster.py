"""Attendee roster and name resolution -- tasks 2.2, 2.3, 2.4.

The rule under test throughout: **resolution is reluctant.** A wrong
`contact_id` silently corrupts every attendance-based risk and nothing in the
UI would reveal it, while an unresolved attendee is itself the
missing-stakeholder signal. So NULL is a result, not a failure.
"""

import json
import pathlib

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.ai import stages, tiebreak
from app.db.session import SessionLocal
from app.models import Account, Contact, Deal, DocumentChunk, Meeting, MeetingAttendee
from app.models.enums import MeetingStatus
from app.services import roster

MANIFEST = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "manifest.json").read_text()
)


# --------------------------------------------------------------------------
# pure functions
# --------------------------------------------------------------------------


def test_normalise_collapses_whitespace():
    assert roster.normalise("  Priya   Raman \n") == "Priya Raman"


def test_is_internal_uses_the_configured_name():
    """There is no users table, so one setting decides who 'we' are.

    A heuristic -- whoever makes commitments -- would misfile the first
    customer who promises something.
    """
    assert roster.is_internal("Maya Chen") is True
    assert roster.is_internal("maya  chen") is True
    assert roster.is_internal("Priya Raman") is False


# --------------------------------------------------------------------------
# resolution against contacts
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def account_with_contacts(db):
    account = Account(name="TEST roster")
    db.add(account)
    await db.flush()
    for first, last in (("Priya", "Raman"), ("Marcus", "Webb")):
        db.add(Contact(account_id=account.id, first_name=first, last_name=last))
    await db.flush()
    account_id = account.id
    await db.commit()
    yield account_id
    async with SessionLocal() as cleanup:
        obsolete = await cleanup.get(Account, account_id)
        if obsolete is not None:
            await cleanup.delete(obsolete)
            await cleanup.commit()


async def test_exact_name_links(db, account_with_contacts):
    result = await roster.resolve(db, account_with_contacts, "Priya Raman")
    assert result.decision == roster.LINKED
    assert result.contact_id is not None
    assert result.similarity == pytest.approx(1.0)


async def test_an_unknown_name_stays_null_and_calls_no_model(db, account_with_contacts):
    """A score near zero is an answer, not a doubt.

    This is the Tom Alvarez case: he speaks in every meeting and is nobody we
    know. Sending him to a model would invite an invention.
    """
    result = await roster.resolve(db, account_with_contacts, "Tom Alvarez")
    assert result.decision == roster.UNKNOWN
    assert result.contact_id is None
    assert result.needs_tiebreak is False


async def test_a_typo_is_tolerated_or_deferred_but_never_mislinked(db, account_with_contacts):
    result = await roster.resolve(db, account_with_contacts, "Priya Ramen")
    assert result.decision in (roster.LINKED, roster.AMBIGUOUS)
    if result.decision == roster.AMBIGUOUS:
        assert result.contact_id is None


async def test_two_similar_contacts_are_ambiguous_not_guessed(db, account_with_contacts):
    """The case a threshold cannot settle: one first name, two people."""
    async with SessionLocal() as session:
        session.add(Contact(account_id=account_with_contacts,
                            first_name="Priya", last_name="Shah"))
        await session.commit()

    result = await roster.resolve(db, account_with_contacts, "Priya")
    assert result.decision == roster.AMBIGUOUS
    assert result.contact_id is None
    assert len(result.candidates) >= 2


async def test_resolution_is_scoped_to_the_account(db, account_with_contacts):
    """Contacts belong to accounts; no foreign key expresses that for a speaker."""
    other = Account(name="TEST elsewhere")
    db.add(other)
    await db.flush()
    await db.commit()
    result = await roster.resolve(db, other.id, "Priya Raman")
    assert result.decision == roster.UNKNOWN
    async with SessionLocal() as cleanup:
        await cleanup.delete(await cleanup.get(Account, other.id))
        await cleanup.commit()


# --------------------------------------------------------------------------
# writing the roster
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def meeting_on(db, account_with_contacts):
    deal = Deal(account_id=account_with_contacts, name="SecureFlow", stage="discovery")
    db.add(deal)
    await db.flush()
    meeting = Meeting(deal_id=deal.id, title="Call", status=MeetingStatus.COMPLETED)
    db.add(meeting)
    await db.flush()
    ids = (meeting.id, account_with_contacts)
    await db.commit()
    return ids


async def test_sync_writes_one_row_per_speaker(db, meeting_on):
    meeting_id, account_id = meeting_on
    await roster.sync_roster(db, meeting_id, account_id,
                             ["Maya Chen", "Priya Raman", "Tom Alvarez"])
    await db.commit()
    rows = (await db.execute(
        select(MeetingAttendee).where(MeetingAttendee.meeting_id == meeting_id)
    )).scalars().all()
    assert {r.raw_name for r in rows} == {"Maya Chen", "Priya Raman", "Tom Alvarez"}
    by_name = {r.raw_name: r for r in rows}
    assert by_name["Priya Raman"].contact_id is not None
    assert by_name["Tom Alvarez"].contact_id is None
    assert by_name["Maya Chen"].is_internal is True


async def test_sync_is_idempotent_for_unresolved_names(db, meeting_on):
    """`uq_meeting_attendees_meeting_id_contact_id` is partial, so the database
    does not stop Tom being inserted twice. This function is the only writer."""
    meeting_id, account_id = meeting_on
    for _ in range(3):
        await roster.sync_roster(db, meeting_id, account_id, ["Tom Alvarez"])
        await db.commit()
    rows = (await db.execute(
        select(MeetingAttendee).where(MeetingAttendee.meeting_id == meeting_id)
    )).scalars().all()
    assert len(rows) == 1


async def test_sync_upgrades_null_to_a_link_but_never_the_reverse(db, meeting_on):
    """What happens when a human promotes a stakeholder fact into a contact."""
    meeting_id, account_id = meeting_on
    await roster.sync_roster(db, meeting_id, account_id, ["Dana Whitfield"])
    await db.commit()
    # Scoped to this meeting: the fixture corpus also has a Dana Whitfield
    # attendee, and an unscoped query returns whichever row the planner likes.
    row = await db.scalar(
        select(MeetingAttendee).where(
            MeetingAttendee.meeting_id == meeting_id,
            MeetingAttendee.raw_name == "Dana Whitfield",
        )
    )
    assert row.contact_id is None

    db.add(Contact(account_id=account_id, first_name="Dana", last_name="Whitfield"))
    await db.commit()
    await roster.sync_roster(db, meeting_id, account_id, ["Dana Whitfield"])
    await db.commit()
    await db.refresh(row)
    assert row.contact_id is not None

    linked = row.contact_id
    await roster.sync_roster(db, meeting_id, account_id, ["Dana Whitfield"])
    await db.commit()
    await db.refresh(row)
    assert row.contact_id == linked, "a human-set link must not be cleared"


# --------------------------------------------------------------------------
# stage 1 and the tiebreak
# --------------------------------------------------------------------------


def test_speakers_come_from_chunk_metadata_not_a_reparse():
    """ingest wrote the attribution at upload time; it must not be re-derived.

    Two parses that could disagree is a worse failure than one that is wrong,
    because only one of them is visible.
    """
    chunks = [
        DocumentChunk(chunk_metadata={"speakers": ["Priya Raman", "Maya Chen"]}),
        DocumentChunk(chunk_metadata={"speakers": ["Maya Chen", "Tom Alvarez"]}),
        DocumentChunk(chunk_metadata={}),
    ]
    assert stages.speakers_from_chunks(chunks) == ["Priya Raman", "Maya Chen", "Tom Alvarez"]


async def test_tiebreak_is_a_no_op_when_ai_is_disabled(db, account_with_contacts):
    """Resolution has to work with the provider off: no_economic_buyer cannot
    depend on a model being reachable.

    ``ai_enabled`` is forced off by the autouse ``_offline`` fixture rather
    than inherited from the environment -- this test passed for the wrong
    reason until a key was configured, at which point it made a real call.
    """
    result = await roster.resolve(db, account_with_contacts, "Priya Ramen")
    before = result.contact_id
    returned = await tiebreak.resolve_ambiguous(result)
    assert returned.contact_id == before


async def test_tiebreak_rejects_an_out_of_range_choice(db, account_with_contacts, monkeypatch):
    """Indices are validated against the list we supplied -- the whole reason
    the model answers with a number rather than a contact id."""
    from app.ai import client as ai_client

    result = await roster.resolve(db, account_with_contacts, "Priya")
    result.decision = roster.AMBIGUOUS

    async def fake_structured(*args, **kwargs):
        return tiebreak.SpeakerMatch(choice=99, reasoning="nonsense"), None

    monkeypatch.setattr(ai_client, "structured", fake_structured)
    monkeypatch.setattr(tiebreak.client, "structured", fake_structured)
    returned = await tiebreak.resolve_ambiguous(result)
    assert returned.contact_id is None


async def test_tiebreak_links_a_valid_choice(db, account_with_contacts, monkeypatch):
    result = await roster.resolve(db, account_with_contacts, "Priya")
    expected = result.candidates[0].contact_id

    async def fake_structured(*args, **kwargs):
        return tiebreak.SpeakerMatch(choice=1, reasoning="same person"), None

    monkeypatch.setattr(tiebreak.client, "structured", fake_structured)
    returned = await tiebreak.resolve_ambiguous(result)
    assert returned.contact_id == expected

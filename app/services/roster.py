"""Transcript speakers into ``meeting_attendees``, and names into contacts.

Why this is load-bearing rather than cosmetic: ``no_economic_buyer`` and
``single_threaded`` are computed from ``meeting_attendees``. Until something
writes that table from transcripts, those two rules read whatever a human
typed by hand -- they cannot hallucinate, but they can be confidently wrong
about an empty table, which is worse because it looks like an answer.

Resolution is deliberately reluctant. ``MeetingAttendee``'s own docstring makes
the case: ``raw_name`` is required and ``contact_id`` is the optional
*resolution* of it, because "a name that maps to no known contact is the raw
signal for 'missing stakeholder'". So an unresolved attendee is a product
feature, not a gap to be filled by guessing. A wrong ``contact_id`` is the
expensive error -- it silently corrupts every attendance-based risk and there is
nothing in the UI that would show it.

Three outcomes, by trigram similarity:

    >= roster_link_threshold, and clear of the runner-up   -> linked
    >= roster_candidate_threshold                          -> ambiguous, ask a model (2.4)
    below that                                             -> unknown, and no model is called

The last case matters for cost and for honesty: a score near zero is not doubt,
it is the answer "nobody we know", and sending it to a model invites an
invention.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Contact, MeetingAttendee
from app.services import ingest

_WHITESPACE = re.compile(r"\s+")

LINKED = "linked"
AMBIGUOUS = "ambiguous"
UNKNOWN = "unknown"


def normalise(name: str) -> str:
    """Collapse whitespace and trim. Not casefolded -- trigram similarity is
    case-insensitive on its own, and the original casing is what gets stored."""
    return _WHITESPACE.sub(" ", name).strip()


def is_internal(raw_name: str) -> bool:
    """Whether this speaker is us.

    A configured name rather than a heuristic (see ``ae_display_name``). One AE
    exists in the MVP, so a setting is both sufficient and honest; a guess based
    on who makes commitments would misfile the first customer who promises
    something.
    """
    return normalise(raw_name).casefold() == normalise(settings.ae_display_name).casefold()


@dataclass
class Candidate:
    contact_id: uuid.UUID
    full_name: str
    similarity: float


@dataclass
class Resolution:
    raw_name: str
    decision: str
    contact_id: Optional[uuid.UUID] = None
    similarity: Optional[float] = None
    candidates: List[Candidate] = field(default_factory=list)

    @property
    def needs_tiebreak(self) -> bool:
        return self.decision == AMBIGUOUS


async def candidates_for(
    db: AsyncSession, account_id: uuid.UUID, raw_name: str, limit: int = 5
) -> List[Candidate]:
    """Contacts on this account, ranked by trigram similarity to the name.

    Scoped to the account, never the whole table: contacts belong to accounts
    and a speaker in a Northwind meeting cannot be somebody at another company.
    Nothing in the schema enforces that, so it is enforced here.
    """
    full_name = (Contact.first_name + " " + Contact.last_name).label("full_name")
    score = func.similarity(Contact.first_name + " " + Contact.last_name, raw_name).label("score")
    rows = (
        await db.execute(
            select(Contact.id, full_name, score)
            .where(Contact.account_id == account_id)
            .order_by(score.desc())
            .limit(limit)
        )
    ).all()
    return [Candidate(contact_id=r[0], full_name=r[1], similarity=float(r[2])) for r in rows]


async def resolve(db: AsyncSession, account_id: uuid.UUID, raw_name: str) -> Resolution:
    """Decide whether this speaker is a known contact. No model call."""
    name = normalise(raw_name)
    ranked = await candidates_for(db, account_id, name)
    if not ranked or ranked[0].similarity < settings.roster_candidate_threshold:
        return Resolution(raw_name=name, decision=UNKNOWN, candidates=ranked)

    best = ranked[0]
    runner_up = ranked[1].similarity if len(ranked) > 1 else 0.0
    clear_of_runner_up = (best.similarity - runner_up) >= settings.roster_link_margin

    if best.similarity >= settings.roster_link_threshold and clear_of_runner_up:
        return Resolution(raw_name=name, decision=LINKED, contact_id=best.contact_id,
                          similarity=best.similarity, candidates=ranked)

    # Either not similar enough to be certain, or two contacts are too close to
    # separate -- two people named Priya is exactly the case a threshold cannot
    # settle. Both go to the tiebreak.
    return Resolution(raw_name=name, decision=AMBIGUOUS,
                      similarity=best.similarity, candidates=ranked)


async def speakers_from_transcript(transcript: str) -> List[str]:
    """Distinct speaker labels, in order of first appearance."""
    seen: List[str] = []
    for speaker, _, _ in ingest.parse_speaker_turns(transcript):
        name = normalise(speaker)
        if name not in seen:
            seen.append(name)
    return seen


async def sync_roster(
    db: AsyncSession,
    meeting_id: uuid.UUID,
    account_id: uuid.UUID,
    speakers: Sequence[str],
) -> Dict[str, Resolution]:
    """Write one ``meeting_attendees`` row per distinct speaker.

    Idempotent by ``(meeting_id, raw_name)``, which has to be enforced here:
    ``uq_meeting_attendees_meeting_id_contact_id`` is partial
    (``WHERE contact_id IS NOT NULL``), deliberately, so that a meeting may hold
    many unresolved attendees -- which also means it does not stop *Tom Alvarez*
    being inserted twice by two analysis runs. A unique index on
    ``(meeting_id, lower(raw_name))`` would move this into the database; until
    one exists this function is the only writer.

    Re-running may *upgrade* a row: an attendee who was unresolved becomes
    linked once the contact exists, which is what happens when a human promotes
    a ``stakeholder`` fact. It never downgrades a link to NULL -- a human who
    set that link outranks a similarity score.
    """
    existing = {
        normalise(row.raw_name).casefold(): row
        for row in (
            await db.execute(
                select(MeetingAttendee).where(MeetingAttendee.meeting_id == meeting_id)
            )
        ).scalars()
    }

    resolutions: Dict[str, Resolution] = {}
    for raw_name in speakers:
        name = normalise(raw_name)
        resolution = await resolve(db, account_id, name)
        resolutions[name] = resolution

        row = existing.get(name.casefold())
        if row is None:
            db.add(
                MeetingAttendee(
                    meeting_id=meeting_id,
                    raw_name=name,
                    contact_id=resolution.contact_id,
                    is_internal=is_internal(name),
                    attended=True,
                )
            )
        elif row.contact_id is None and resolution.contact_id is not None:
            row.contact_id = resolution.contact_id

    await db.flush()
    return resolutions

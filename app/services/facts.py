"""Writing extracted facts -- and the two rules that make the writes trustworthy.

This is the only place ``extracted_facts`` rows are created from model output,
and it exists so that two guarantees cannot be forgotten by a caller:

**Gate 0 runs inside the insert.** Not before it as a courtesy, not after it as
an audit. A claim whose citation does not resolve is never written, so the
database cannot hold an unverifiable fact even briefly
(docs/schema/README.md section 5).

**A claim with zero surviving links is a rejected write, not a flagged one.**
"No evidence" is a rejection. The service raises or skips; it never writes the
claim and leaves someone downstream to notice.

Everything here is side-effect-free until the caller commits -- the pipeline
owns the transaction, so a later stage failing rolls the facts back with it.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ExtractedFact
from app.models.enums import ClaimType, FactStatus, SourceKind
from app.services import claims, gate0

logger = logging.getLogger("cognideal.services.facts")


@dataclass
class RejectedFact:
    """A candidate that did not survive Gate 0, kept for the metric.

    The *rate* of these is the single most useful signal about extraction
    quality, so they are returned rather than silently dropped. They are not
    persisted: a row for something that failed verification is a row somebody
    will eventually render.
    """

    fact_type: str
    content: str
    snippet: str
    reason: str


@dataclass
class FactWriteResult:
    written: List[uuid.UUID] = field(default_factory=list)
    rejected: List[RejectedFact] = field(default_factory=list)

    @property
    def gate0_pass_rate(self) -> float:
        total = len(self.written) + len(self.rejected)
        return (len(self.written) / total) if total else 0.0


def citation_for(candidate: Any) -> Dict[str, Any]:
    """The citation dict Gate 0 and ``attach_evidence`` both consume.

    One shape for both so the thing checked is exactly the thing written. A
    separate mapping on each side is how a span gets verified and then stored
    with different offsets.
    """
    return {
        "source_kind": SourceKind.DOCUMENT,
        "chunk_id": candidate.chunk_id,
        "snippet": candidate.snippet,
        "char_start": candidate.char_start,
        "char_end": candidate.char_end,
        "speaker": candidate.speaker,
    }


async def write_facts(
    db: AsyncSession,
    *,
    deal_id: uuid.UUID,
    candidates: Sequence[Any],
    meeting_id: Optional[uuid.UUID] = None,
    document_id: Optional[uuid.UUID] = None,
    occurred_at: Optional[datetime] = None,
    checks: Optional[Dict[int, gate0.ClaimCheck]] = None,
) -> FactWriteResult:
    """Write the candidates that pass Gate 0. Returns what landed and what did not.

    ``checks`` lets a caller that has already run Gate 0 pass the results in --
    the pipeline does, so the work is not repeated. They are still *honoured*
    rather than trusted: a check that did not pass means no write, whoever
    computed it. Omitting them makes this function run the gate itself, which
    is what keeps it non-skippable for any future caller.
    """
    result = FactWriteResult()

    for index, candidate in enumerate(candidates):
        if candidate.rejected:
            result.rejected.append(RejectedFact(
                candidate.fact_type, candidate.content, candidate.snippet,
                candidate.rejected,
            ))
            continue

        citation = citation_for(candidate)
        check = (checks or {}).get(index)
        if check is None:
            check = await gate0.check_claim(db, candidate.content, [citation])

        if not check.passed:
            result.rejected.append(RejectedFact(
                candidate.fact_type, candidate.content, candidate.snippet,
                check.reason or "gate 0 failed",
            ))
            continue

        fact = ExtractedFact(
            deal_id=deal_id,
            meeting_id=meeting_id,
            document_id=document_id,
            fact_type=candidate.fact_type,
            content=candidate.content,
            payload=candidate.payload,
            confidence=candidate.confidence,
            # `pending` is the whole point: a model-written fact is a proposal
            # until a human promotes it, which is Gate 3.
            status=FactStatus.PENDING,
        )
        db.add(fact)
        await db.flush()

        await claims.attach_evidence(
            db,
            claim_type=ClaimType.FACT,
            claim_id=fact.id,
            deal_id=deal_id,
            source_kind=SourceKind.DOCUMENT,
            snippet=candidate.snippet,
            document_id=document_id,
            chunk_id=candidate.chunk_id,
            char_start=candidate.char_start,
            char_end=candidate.char_end,
            speaker=candidate.speaker,
            occurred_at=occurred_at,
            relevance=candidate.confidence,
            verification_status=check.links[0].status,
        )
        result.written.append(fact.id)

    await db.flush()
    logger.info(
        "facts.written deal=%s written=%d rejected=%d gate0_pass_rate=%.2f",
        deal_id, len(result.written), len(result.rejected), result.gate0_pass_rate,
    )
    for bad in result.rejected:
        logger.info("facts.rejected type=%s reason=%s content=%r",
                    bad.fact_type, bad.reason, bad.content[:70])
    return result


async def existing_fact_count(
    db: AsyncSession, document_id: uuid.UUID
) -> int:
    """How many facts this document already produced.

    Re-analysis is idempotent at the document level rather than the fact level:
    there is no natural key on a fact, so the only safe statement is "this
    document has already been extracted". The pipeline uses it to skip rather
    than duplicate.
    """
    from sqlalchemy import func, select

    return await db.scalar(
        select(func.count()).select_from(ExtractedFact)
        .where(ExtractedFact.document_id == document_id)
    ) or 0


# --------------------------------------------------------------------------
# Gate 3 -- human adjudication
#
# Where correctness is actually resolved. Gates 0-2 decide what a person is
# allowed to see; a person decides what is true.
#
# The schema describes this as promotion into a `commitment` or `task`, and
# that is right for the categories that name an action. It is not right for the
# other six: a `competitor` fact ("Meridian quoted 195k") is something to know,
# not something to do, and there is no table to promote it into. So accepting
# does one of two things depending on the category:
#
#   promote   `commitment` -- creates the commitments row, recorded in
#             promoted_to_type / promoted_to_id. The only category whose
#             payload maps onto an existing table unambiguously.
#   confirm   everything else -- flips `status` to `accepted` and nothing more.
#
# "And nothing more" undersells it. `dossier.build` selects facts
# `WHERE status = 'accepted'`, so confirming is what makes a fact visible to
# the AI detector at all. A pending fact is inert; a confirmed one is deal
# context the next analysis pass reasons over. That is the whole value of the
# gesture for six of the eight categories.
# --------------------------------------------------------------------------

#: Categories whose acceptance creates a row somewhere else, and where it goes.
#: Deliberately short. `requirement` is the tempting third entry and is left
#: out: "they need SCIM provisioning" is a fact about the customer, not work we
#: promised, and turning it into a task would put words in the user's mouth.
#: `stakeholder` has an obvious home in `deal_contacts` but the schema does not
#: sanction that target, and inventing it here would make this module the
#: authority on a decision that belongs in the schema docs.
PROMOTION_TARGETS = {"commitment": "commitment"}


def _parse_spoken_date(value: Optional[str]):
    """An ISO date, or nothing.

    Extraction records dates as they were spoken -- "Thursday the ninth",
    "the fifteenth of November" -- because the snippet must be verbatim. Those
    are not parseable without a reference year and a calendar, and a guessed
    due date on a commitment is worse than none: `missed_commitment` measures
    against it. So only an already-ISO value is taken, and anything else leaves
    `due_date` null, which the column allows.
    """
    if not value:
        return None
    from datetime import date as _date

    try:
        return _date.fromisoformat(value.strip()[:10])
    except (ValueError, AttributeError):
        return None


async def apply_decision(db: AsyncSession, fact: Any, to_status: Any) -> bool:
    """Record a human's verdict on a fact. Returns True if it promoted a row.

    Refuses three things, each because the alternative loses information:

    *   a `superseded` fact -- a later extraction already replaced it, and
        accepting the older claim would leave two live records of the same
        thing. Supersession is the pipeline's judgement, not a status a human
        is adjudicating.
    *   re-deciding a fact that has already promoted -- the commitment it
        created is a real row that something may already reference. Deleting
        that row releases the fact back to pending (see
        `services/task.release_source_fact` for the same rule on tasks), so the
        honest path is destructive and visible rather than a silent status flip.
    *   accepting into any status other than accepted/rejected -- `pending` is
        where a fact starts and `superseded` is the pipeline's to write.
    """
    from fastapi import HTTPException, status as http
    from app.models import Commitment
    from app.models.enums import FactStatus, Origin, OwnerSide

    if fact.status == FactStatus.SUPERSEDED or fact.status == FactStatus.SUPERSEDED.value:
        raise HTTPException(
            status_code=http.HTTP_409_CONFLICT,
            detail=(
                "A later extraction superseded this fact. Decide the one that "
                "replaced it -- accepting this would leave two live records of "
                "the same claim."
            ),
        )
    if fact.promoted_to_id is not None:
        raise HTTPException(
            status_code=http.HTTP_409_CONFLICT,
            detail=(
                f"Already accepted, and it created {fact.promoted_to_type} "
                f"{fact.promoted_to_id}. Delete that record to undo this -- "
                f"changing the status alone would leave it orphaned."
            ),
        )

    wanted = to_status.value if hasattr(to_status, "value") else to_status
    fact.status = wanted

    if wanted != FactStatus.ACCEPTED.value:
        return False

    target = PROMOTION_TARGETS.get(
        fact.fact_type.value if hasattr(fact.fact_type, "value") else fact.fact_type
    )
    if target is None:
        # Confirmed, not promoted. The status change is the whole effect, and
        # it is what puts this fact in front of the AI detector.
        return False

    payload = fact.payload or {}
    side = payload.get("owner_side")
    commitment = Commitment(
        deal_id=fact.deal_id,
        source_fact_id=fact.id,
        description=payload.get("what") or fact.content,
        owner_side=side if side in (OwnerSide.US.value, OwnerSide.CUSTOMER.value) else OwnerSide.US,
        owner_name=payload.get("owner_name"),
        due_date=_parse_spoken_date(payload.get("date")),
        # `origin='ai'` on the row: a person approved it, but the model is what
        # proposed it, and "how much of this record did the model write?" has to
        # stay answerable.
        origin=Origin.AI,
        confidence=fact.confidence,
    )
    db.add(commitment)
    await db.flush()

    fact.promoted_to_type = target
    fact.promoted_to_id = commitment.id
    return True

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

logger = logging.getLogger("dealpilot.services.facts")


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

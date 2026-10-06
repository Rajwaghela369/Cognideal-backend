"""Facts the extractor proposed, for a human to accept or reject.

One read and one write. The write is Gate 3 -- see
`services/facts.apply_decision` for why accepting means two different things
depending on the category, and why only `commitment` promotes.

The one rule this route must not forget, and therefore does not implement
itself: a claim whose newest Gate 1 verdict is `contradicted` or `unsupported`
is excluded by `queries.quarantine_filter`. Putting that predicate in
`queries.py` rather than here is deliberate -- it holds for every reader, and a
route that omitted it would render a quarantined claim as fact with nothing in
the response to reveal it.
"""

import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated

from app.api.deps import get_deal_or_404
from app.db.session import get_db
from app.models import ClaimEvidence, Deal, Evidence, ExtractedFact
from app.models.enums import ClaimType
from app.queries import quarantine_filter
from app.schemas.v1.deal.fact import (
    FactDecision,
    FactEvidence,
    FactFilters,
    FactListItem,
)
from app.services import analysis as analysis_service
from app.services import claims as claims_service
from app.services import facts as facts_service

router = APIRouter(prefix="/deals/{deal_id}/facts", tags=["facts"])


@router.get("", response_model=List[FactListItem])
async def list_facts(
    filters: Annotated[FactFilters, Query()],
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Pending and reviewed facts on this deal, newest first.

    Quarantined claims are absent. `verdict` is returned so `partial` can be
    shown with a caution badge rather than silently hidden -- a claim where one
    clause is evidenced and another is not is still worth a human's eye.
    """
    stmt = (
        select(ExtractedFact)
        .where(
            ExtractedFact.deal_id == deal.id,
            quarantine_filter(ClaimType.FACT, ExtractedFact.id),
        )
        .order_by(ExtractedFact.extracted_at.desc())
    )
    if filters.fact_type:
        stmt = stmt.where(ExtractedFact.fact_type.in_([f.value for f in filters.fact_type]))
    if filters.status:
        stmt = stmt.where(ExtractedFact.status.in_(filters.status))

    facts = list((await db.execute(stmt)).scalars())
    if not facts:
        return []

    fact_ids = [fact.id for fact in facts]
    verdicts = await claims_service.latest_verdicts(db, ClaimType.FACT, fact_ids)

    spans = (
        await db.execute(
            select(
                ClaimEvidence.claim_id,
                Evidence.id,
                Evidence.snippet,
                Evidence.speaker,
                Evidence.chunk_id,
                Evidence.char_start,
                Evidence.char_end,
                ClaimEvidence.verification_status,
            )
            .join(Evidence, Evidence.id == ClaimEvidence.evidence_id)
            .where(
                ClaimEvidence.claim_type == ClaimType.FACT,
                ClaimEvidence.claim_id.in_(fact_ids),
            )
        )
    ).all()
    by_fact = {}
    for row in spans:
        by_fact.setdefault(row[0], []).append(
            FactEvidence(
                evidence_id=row[1], snippet=row[2], speaker=row[3], chunk_id=row[4],
                char_start=row[5], char_end=row[6],
                verification_status=row[7].value if hasattr(row[7], "value") else row[7],
            )
        )

    return [
        FactListItem(
            id=fact.id,
            fact_type=fact.fact_type,
            content=fact.content,
            payload=fact.payload,
            status=fact.status.value if hasattr(fact.status, "value") else fact.status,
            confidence=float(fact.confidence) if fact.confidence is not None else None,
            verdict=verdicts.get(fact.id),
            meeting_id=fact.meeting_id,
            document_id=fact.document_id,
            extracted_at=fact.extracted_at,
            promoted_to_type=fact.promoted_to_type,
            promoted_to_id=fact.promoted_to_id,
            evidence=by_fact.get(fact.id, []),
        )
        for fact in facts
    ]


@router.patch("/{fact_id}", response_model=FactListItem)
async def decide_fact(
    fact_id: uuid.UUID,
    body: FactDecision,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Gate 3. Accept or reject one proposal.

    Both ids are matched, not just the fact's: fetching by id alone would let
    one deal's URL adjudicate another deal's fact.

    Tiering is the subtle part, and it follows from what actually reads a fact.

    **Tier 2 always.** `dossier.build` selects facts `WHERE status =
    'accepted'`, so a decision in either direction changes what the AI detector
    can see -- accepting puts a fact in front of it, rejecting takes one away.
    It is debounced rather than immediate on purpose: a reviewer works through a
    queue of twenty, and twenty detector runs where one will do is exactly what
    the quiet window exists to collapse.

    **Tier 1 only when something promoted.** The deterministic detector never
    reads `extracted_facts` -- zero references -- so a confirmed `competitor`
    fact gives it nothing to re-evaluate. A promoted `commitment` does: it
    writes a `commitments` row, and `missed_commitment` reads that table.

    No `origin` argument, and that matters. `mark_dirty` ignores writes where
    `origin == Origin.AI`, so passing it here -- on the reasoning that the fact
    came from the model -- would silently suppress tier 2 and make acceptance
    have no effect at all. The *decision* is the human's; that is the whole
    point of this gate.
    """
    fact = await db.scalar(
        select(ExtractedFact).where(
            ExtractedFact.id == fact_id, ExtractedFact.deal_id == deal.id
        )
    )
    if fact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fact {fact_id} not found on this deal",
        )

    promoted = await facts_service.apply_decision(db, fact, body.status)

    await analysis_service.record_change(
        db,
        deal.id,
        "fact %s" % (body.status.value if hasattr(body.status, "value") else body.status),
        tier1=promoted,
        tier2=True,
    )
    await db.commit()

    decided = await list_facts(
        FactFilters(fact_type=None, status=None), deal=deal, db=db
    )
    return next(item for item in decided if item.id == fact_id)

"""Facts the extractor proposed, for a human to accept or reject.

Read-only. Promotion is a separate write path (Gate 3) and does not belong on a
list endpoint.

The one rule this route must not forget, and therefore does not implement
itself: a claim whose newest Gate 1 verdict is `contradicted` or `unsupported`
is excluded by `queries.quarantine_filter`. Putting that predicate in
`queries.py` rather than here is deliberate -- it holds for every reader, and a
route that omitted it would render a quarantined claim as fact with nothing in
the response to reveal it.
"""

import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated

from app.api.deps import get_deal_or_404
from app.db.session import get_db
from app.models import ClaimEvidence, Deal, Evidence, ExtractedFact
from app.models.enums import ClaimType
from app.queries import quarantine_filter
from app.schemas.v1.deal.fact import FactEvidence, FactFilters, FactListItem
from app.services import claims as claims_service

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
            evidence=by_fact.get(fact.id, []),
        )
        for fact in facts
    ]

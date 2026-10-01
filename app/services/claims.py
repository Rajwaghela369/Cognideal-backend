"""Evidence attached to claims, and the orphan problem it creates.

`claim_evidence.claim_id` is a bare uuid with **no foreign key** -- it is
polymorphic over the five tables that hold model assertions, with `claim_type`
saying which. That is what lets one claim rest on several spans and one span
support several claims, but Postgres cannot then stop a claim being deleted out
from under its links.

So the service layer has to. Every delete that removes a claim -- or a deal, or
a document whose chunks the evidence points into -- must clear the links in the
same transaction, or `claim_evidence` fills with rows referencing nothing. This
module is the one place that knows how.
"""

import uuid
from typing import Iterable, List, Optional, Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ClaimEvidence, ClaimValidation, Evidence
from app.models.enums import ClaimType, SourceKind


async def attach_evidence(
    db: AsyncSession,
    *,
    claim_type: ClaimType,
    claim_id: uuid.UUID,
    deal_id: uuid.UUID,
    source_kind: SourceKind,
    snippet: str,
    record_ref: Optional[dict] = None,
    document_id: Optional[uuid.UUID] = None,
    chunk_id: Optional[uuid.UUID] = None,
    relevance: Optional[float] = None,
) -> Evidence:
    """Record one locatable source for one claim.

    ``snippet`` must be the **literal value**, not a description of it. Gate 0
    re-resolves ``record_ref`` and checks the field still holds exactly this
    string; that is also what makes a claim self-invalidate when the world
    moves on -- change the deal's stage and the citation goes `value_drifted`
    without anyone having to notice.
    """
    evidence = Evidence(
        deal_id=deal_id,
        source_kind=source_kind,
        document_id=document_id,
        chunk_id=chunk_id,
        record_ref=record_ref,
        snippet=snippet,
    )
    db.add(evidence)
    await db.flush()

    db.add(
        ClaimEvidence(
            claim_type=claim_type,
            claim_id=claim_id,
            evidence_id=evidence.id,
            relevance=relevance,
        )
    )
    return evidence


async def delete_claim_links(
    db: AsyncSession, claim_type: ClaimType, claim_ids: Sequence[uuid.UUID]
) -> None:
    """Clear the citation links for claims about to be deleted.

    Deletes the `evidence` rows too, but only those cited by nothing else --
    one span can support several claims, and dropping a risk must not silently
    remove the quote a commitment also rests on.

    Call this *before* deleting the claims, inside the same transaction. The
    `evidence_id` side of claim_evidence is a real foreign key, so the ordering
    matters: links first, then the evidence they were the last reference to.
    """
    if not claim_ids:
        return

    evidence_ids: List[uuid.UUID] = list(
        (
            await db.scalars(
                select(ClaimEvidence.evidence_id).where(
                    ClaimEvidence.claim_type == claim_type,
                    ClaimEvidence.claim_id.in_(claim_ids),
                )
            )
        ).all()
    )

    await db.execute(
        delete(ClaimEvidence).where(
            ClaimEvidence.claim_type == claim_type,
            ClaimEvidence.claim_id.in_(claim_ids),
        )
    )
    # Validation rows are keyed the same polymorphic way and have the same
    # problem.
    await db.execute(
        delete(ClaimValidation).where(
            ClaimValidation.claim_type == claim_type,
            ClaimValidation.claim_id.in_(claim_ids),
        )
    )
    await db.flush()

    if not evidence_ids:
        return

    still_cited = set(
        (
            await db.scalars(
                select(ClaimEvidence.evidence_id).where(
                    ClaimEvidence.evidence_id.in_(evidence_ids)
                )
            )
        ).all()
    )
    orphaned = [e for e in evidence_ids if e not in still_cited]
    if orphaned:
        await db.execute(delete(Evidence).where(Evidence.id.in_(orphaned)))


async def evidence_for(
    db: AsyncSession, claim_type: ClaimType, claim_id: uuid.UUID
) -> Iterable:
    """Citations for one claim, strongest first.

    `relevance` is how strongly *that* span supports *that* claim, which is
    what lets the UI show the best quote and collapse the marginal ones behind
    "show 2 more".
    """
    return (
        await db.execute(
            select(
                Evidence.id,
                Evidence.source_kind,
                Evidence.document_id,
                Evidence.chunk_id,
                Evidence.record_ref,
                Evidence.snippet,
                Evidence.char_start,
                Evidence.char_end,
                Evidence.speaker,
                Evidence.occurred_at,
                ClaimEvidence.relevance,
                ClaimEvidence.verification_status,
                ClaimEvidence.verified_at,
            )
            .join(ClaimEvidence, ClaimEvidence.evidence_id == Evidence.id)
            .where(
                ClaimEvidence.claim_type == claim_type,
                ClaimEvidence.claim_id == claim_id,
            )
            .order_by(ClaimEvidence.relevance.desc().nulls_last(), Evidence.id)
        )
    ).all()


async def delete_links_for_deal(db: AsyncSession, deal_id: uuid.UUID) -> None:
    """Clear every citation link belonging to a deal, before the deal is deleted.

    Deleting a deal cascades to its risks, commitments, facts and
    recommendations -- and `claim_evidence` points at all four with a bare
    uuid, so Postgres lets them all become orphans at once. This is the fix for
    the gap marked NOT YET HANDLED in the deal delete handler since Layer A.

    `evidence` itself is FK'd to the deal and cascades cleanly, so only the
    polymorphic side needs handling.
    """
    from app.models import Commitment, ExtractedFact, Recommendation, Risk

    for claim_type, model in (
        (ClaimType.RISK, Risk),
        (ClaimType.RECOMMENDATION, Recommendation),
        (ClaimType.COMMITMENT, Commitment),
        (ClaimType.FACT, ExtractedFact),
    ):
        ids = list(
            (await db.scalars(select(model.id).where(model.deal_id == deal_id))).all()
        )
        if ids:
            await db.execute(
                delete(ClaimEvidence).where(
                    ClaimEvidence.claim_type == claim_type,
                    ClaimEvidence.claim_id.in_(ids),
                )
            )
            await db.execute(
                delete(ClaimValidation).where(
                    ClaimValidation.claim_type == claim_type,
                    ClaimValidation.claim_id.in_(ids),
                )
            )
    await db.flush()


async def delete_links_for_document(
    db: AsyncSession, document_id: uuid.UUID
) -> None:
    """Clear citation links that point into a document about to be deleted.

    `documents -> document_chunks -> evidence` cascades, so deleting one
    document silently destroys every evidence row quoting it -- and every
    `claim_evidence` link to those rows survives, pointing at nothing. The
    claims themselves are left alone: a risk whose quote was deleted is now an
    uncited risk, which Gate 0 will mark, not a risk that never existed.
    """
    evidence_ids = list(
        (
            await db.scalars(
                select(Evidence.id).where(Evidence.document_id == document_id)
            )
        ).all()
    )
    if evidence_ids:
        await db.execute(
            delete(ClaimEvidence).where(ClaimEvidence.evidence_id.in_(evidence_ids))
        )
        await db.flush()

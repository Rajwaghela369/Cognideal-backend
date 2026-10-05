"""Phase 10 event tiers: invalidate, recompute, then enqueue regeneration."""

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import ClaimEvidence, Deal, Evidence
from app.models.enums import Origin, SourceKind
from app.services import gate0

_tier2_suppressed: ContextVar[bool] = ContextVar(
    "dealpilot_tier2_suppressed", default=False
)


@contextmanager
def suppress_tier2():
    """Suppress model regeneration for a bulk import; tiers 0 and 1 remain."""
    token = _tier2_suppressed.set(True)
    try:
        yield
    finally:
        _tier2_suppressed.reset(token)


def tier2_is_suppressed() -> bool:
    return settings.analysis_tier2_suppressed or _tier2_suppressed.get()


async def mark_dirty(
    db: AsyncSession,
    deal_id: uuid.UUID,
    reason: str,
    *,
    origin: Optional[Origin] = None,
) -> bool:
    """Debounced Tier 2 enqueue. AI-authored and bulk-import writes are ignored."""
    if tier2_is_suppressed() or origin == Origin.AI or origin == Origin.AI.value:
        return False
    await db.execute(
        update(Deal)
        .where(Deal.id == deal_id)
        .values(
            analysis_dirty_first_at=func.coalesce(
                Deal.analysis_dirty_first_at, func.now()
            ),
            analysis_dirty_last_at=func.now(),
            analysis_dirty_reason=reason[:500],
        )
    )
    return True


async def reverify_record_refs(
    db: AsyncSession,
    *,
    table: str,
    row_id: uuid.UUID,
    fields: Iterable[str],
) -> int:
    """Tier 0: refresh links naming changed record fields in this transaction."""
    names = tuple(set(fields))
    if not names:
        return 0
    await db.flush()
    rows = list((await db.scalars(
        select(Evidence).where(
            Evidence.source_kind == SourceKind.RECORD,
            Evidence.record_ref["table"].astext == table,
            Evidence.record_ref["id"].astext == str(row_id),
            Evidence.record_ref["field"].astext.in_(names),
        )
    )).all())
    for evidence in rows:
        check = await gate0.check_record_ref(db, evidence.record_ref, evidence.snippet)
        await db.execute(
            update(ClaimEvidence)
            .where(ClaimEvidence.evidence_id == evidence.id)
            .values(verification_status=check.status, verified_at=func.now())
        )
    return len(rows)


async def refresh_deterministic(db: AsyncSession, deal_id: uuid.UUID):
    """Tier 1: refresh exhaustive SQL risks immediately."""
    from app.services import detect

    await db.flush()
    deal = await db.get(Deal, deal_id)
    if deal is None:
        return None
    return await detect.run(db, deal)


async def record_change(
    db: AsyncSession,
    deal_id: uuid.UUID,
    reason: str,
    *,
    table: Optional[str] = None,
    row_id: Optional[uuid.UUID] = None,
    fields: Iterable[str] = (),
    tier1: bool = True,
    tier2: bool = True,
    origin: Optional[Origin] = None,
) -> None:
    """Apply the tiers selected for one domain event, in their required order."""
    if table and row_id and fields:
        await reverify_record_refs(
            db, table=table, row_id=row_id, fields=fields
        )
    if tier1:
        await refresh_deterministic(db, deal_id)
    if tier2:
        await mark_dirty(db, deal_id, reason, origin=origin)


async def run_deal_analysis(db: AsyncSession, deal_id: uuid.UUID, budget=None):
    """The worker's Tier 2 unit: deterministic floor, then the AI detector."""
    from app.ai import detect_ai

    deterministic = await refresh_deterministic(db, deal_id)
    ai_result = None
    if settings.ai_enabled:
        proposals, dossier = await detect_ai.propose(db, deal_id, budget=budget)
        ai_result = await detect_ai.apply(
            db, deal_id, proposals, dossier, budget=budget
        )
    return deterministic, ai_result

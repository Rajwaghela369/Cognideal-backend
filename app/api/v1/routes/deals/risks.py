"""The "needs attention" panel: risks with their suggested actions.

One card per risk, with its recommendation nested. Two parallel lists would
make the user read "no economic buyer" in one panel and "engage a stakeholder"
in another and work out that they are the same thing.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, List

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated

from app.api.deps import get_deal_or_404
from app.core.config import settings
from app.db.session import get_db
from app.models import ClaimEvidence, Deal, Recommendation, Risk, Task
from app.models.enums import ClaimType, RiskStatus, TaskStatus
from app.queries import SEVERITY_ORDER
from app.schemas.v1.deal.risk import (
    DealAnalysisState,
    DetectionResult,
    EvidenceItem,
    RecommendationAccept,
    RecommendationDetail,
    RecommendationDismiss,
    RecommendationFilters,
    RiskDetail,
    RiskFilters,
    RiskListItem,
    RiskUpdate,
)
from app.services import claims as claims_service
from app.services import detect as detect_service
from app.services import risk as risk_service

router = APIRouter(prefix="/deals/{deal_id}", tags=["risks"])

OPEN_STATUSES = (RiskStatus.OPEN.value, RiskStatus.MITIGATING.value)

# claim_evidence.claim_id carries no foreign key -- it is polymorphic over the
# five claim tables -- so this cannot be a relationship and has to be an
# explicit correlated subquery keyed on (claim_type, claim_id).
_EVIDENCE_COUNT = (
    select(func.count())
    .select_from(ClaimEvidence)
    .where(
        ClaimEvidence.claim_type == ClaimType.RISK.value,
        ClaimEvidence.claim_id == Risk.id,
    )
    .correlate(Risk)
    .scalar_subquery()
    .label("evidence_count")
)


def _recommendation_payload(rec: Recommendation, task_done: bool) -> dict:
    return {
        "id": rec.id,
        "title": rec.title,
        "action_type": rec.action_type,
        "priority": rec.priority,
        "rationale": rec.rationale,
        "status": rec.status,
        # Derived, never stored: a second copy of the task's status would go
        # stale the moment the task moved.
        "is_completed": task_done,
        "created_task_id": rec.created_task_id,
        "dismissal_reason": rec.dismissal_reason,
    }


async def _risk_rows(db: AsyncSession, deal_id: uuid.UUID, f: RiskFilters) -> List[dict]:
    """Risks with their recommendation and evidence count, in one round trip.

    Outer-joined to recommendations: a risk with no suggestion is still a risk
    worth showing, so an inner join would hide exactly the rows that need
    attention most.
    """
    stmt = (
        select(Risk, Recommendation, _EVIDENCE_COUNT, Task.status.label("task_status"))
        .outerjoin(Recommendation, Recommendation.source_risk_id == Risk.id)
        .outerjoin(Task, Task.id == Recommendation.created_task_id)
        .where(Risk.deal_id == deal_id)
    )

    if f.status:
        stmt = stmt.where(Risk.status.in_(f.status))
    if f.risk_type:
        stmt = stmt.where(Risk.risk_type.in_(f.risk_type))
    if f.severity:
        stmt = stmt.where(Risk.severity.in_(f.severity))
    if f.open is not None:
        stmt = stmt.where(
            Risk.status.in_(OPEN_STATUSES)
            if f.open
            else Risk.status.notin_(OPEN_STATUSES)
        )

    # Live risks first, then worst first, then oldest -- the order someone
    # working the panel top-down would want.
    stmt = stmt.order_by(
        Risk.status.notin_(OPEN_STATUSES),
        SEVERITY_ORDER,
        Risk.first_detected_at.asc(),
    )

    out = []
    for row in (await db.execute(stmt)).all():
        r, rec = row.Risk, row.Recommendation
        out.append(
            {
                "id": r.id,
                "risk_type": r.risk_type,
                "title": r.title,
                "description": r.description,
                "severity": r.severity,
                "status": r.status,
                "origin": r.origin,
                "confidence": r.confidence,
                "first_detected_at": r.first_detected_at,
                "last_seen_at": r.last_seen_at,
                "resolved_at": r.resolved_at,
                "evidence_count": row.evidence_count,
                "recommendation": (
                    _recommendation_payload(rec, row.task_status == TaskStatus.DONE)
                    if rec is not None
                    else None
                ),
            }
        )
    return out


@router.get("/risks", response_model=List[RiskListItem])
async def list_risks(
    filters: Annotated[RiskFilters, Query()],
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Unpaginated: a deal has at most ten risks, one per risk_type."""
    return await _risk_rows(db, deal.id, filters)


@router.get("/risks/{risk_id}", response_model=RiskDetail)
async def get_risk(
    risk_id: uuid.UUID,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await risk_service.get_risk_or_404(db, deal.id, risk_id)
    rows = await _risk_rows(db, deal.id, RiskFilters())
    payload = next(r for r in rows if r["id"] == risk_id)
    payload["evidence"] = await claims_service.evidence_for(
        db, ClaimType.RISK, risk_id
    )
    return payload


@router.get("/risks/{risk_id}/evidence", response_model=List[EvidenceItem])
async def get_risk_evidence(
    risk_id: uuid.UUID,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """The citations alone, for the "▸ 2 sources" expander."""
    await risk_service.get_risk_or_404(db, deal.id, risk_id)
    return await claims_service.evidence_for(db, ClaimType.RISK, risk_id)


@router.patch("/risks/{risk_id}", response_model=RiskDetail)
async def update_risk(
    risk_id: uuid.UUID,
    body: RiskUpdate,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Record the human decision. Only `status` is writable -- the claim itself
    belongs to the detector."""
    risk = await risk_service.get_risk_or_404(db, deal.id, risk_id)
    await risk_service.apply_status_change(db, risk, body.status)
    await db.commit()
    return await get_risk(risk_id, deal, db)


@router.get("/analysis", response_model=DealAnalysisState)
async def get_analysis_state(
    deal: Deal = Depends(get_deal_or_404),
) -> Any:
    """Is an AI pass pending for this deal, and why?

    Phase 10 marks deals dirty in the service layer and the worker claims them
    on a debounce, so between a stage change and the refreshed panel there is a
    window where the screen is showing a stale answer and has no way to say so.
    This is that missing half: the same columns the claim query reads, plus the
    state those columns put the deal in.

    Derived here rather than stored, against the same settings the worker uses
    (`_claim_dirty_deal` in `worker.py`). A stored state column would be a second
    copy of a decision the claim query already makes, and the two would drift the
    first time the debounce was retuned.
    """
    now = datetime.now(timezone.utc)
    quiet_cutoff = now - timedelta(seconds=settings.analysis_debounce_seconds)
    max_cutoff = now - timedelta(seconds=settings.analysis_max_debounce_seconds)
    sweep_cutoff = now - timedelta(hours=settings.analysis_sweep_hours)

    if deal.analysis_dirty_first_at is not None:
        due = (
            deal.analysis_dirty_last_at is not None
            and deal.analysis_dirty_last_at < quiet_cutoff
        ) or deal.analysis_dirty_first_at < max_cutoff
        state = "due" if due else "debouncing"
    elif deal.analysis_swept_at is None or deal.analysis_swept_at < sweep_cutoff:
        state = "stale"
    else:
        state = "clean"

    return {
        "deal_id": deal.id,
        "state": state,
        "dirty_first_at": deal.analysis_dirty_first_at,
        "dirty_last_at": deal.analysis_dirty_last_at,
        "dirty_reason": deal.analysis_dirty_reason,
        "swept_at": deal.analysis_swept_at,
        "debounce_seconds": settings.analysis_debounce_seconds,
        "max_debounce_seconds": settings.analysis_max_debounce_seconds,
        "sweep_hours": settings.analysis_sweep_hours,
    }


@router.post("/analysis", response_model=DetectionResult)
async def run_detection(
    deal: Deal = Depends(get_deal_or_404), db: AsyncSession = Depends(get_db)
) -> Any:
    """Run the deterministic detector over this deal.

    Six of the ten risk types need no model -- they are joins over tables that
    already exist. Writes risks and their recommendations in one pass, because
    the RiskType -> ActionType mapping means they are generated together and two
    triggers could disagree.

    The explicit trigger remains useful for an operator-requested refresh; the
    worker also runs this detector eagerly on relevant writes and in its daily
    sweep.
    """
    result = await detect_service.run(db, deal)
    await db.commit()
    return result


# --------------------------------------------------------------------------
# Recommendations
#
# Secondary to the risk panel above, which nests them. This list earns its place
# for the proactive recommendations that have no source_risk_id to nest under,
# and for evaluation -- what was suggested, accepted and dismissed is the only
# honest measure of whether the advice is worth anything.
# --------------------------------------------------------------------------


@router.get("/recommendations", response_model=List[RecommendationDetail])
async def list_recommendations(
    filters: Annotated[RecommendationFilters, Query()],
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    stmt = (
        select(Recommendation, Task.status.label("task_status"))
        .outerjoin(Task, Task.id == Recommendation.created_task_id)
        .where(Recommendation.deal_id == deal.id)
    )
    if filters.status:
        stmt = stmt.where(Recommendation.status.in_(filters.status))
    if filters.action_type:
        stmt = stmt.where(Recommendation.action_type.in_(filters.action_type))
    if filters.orphaned is not None:
        stmt = stmt.where(
            Recommendation.source_risk_id.is_(None)
            if filters.orphaned
            else Recommendation.source_risk_id.is_not(None)
        )
    stmt = stmt.order_by(Recommendation.generated_at.desc())

    out = []
    for row in (await db.execute(stmt)).all():
        rec = row.Recommendation
        out.append(
            {
                **{
                    c.name: getattr(rec, c.name)
                    for c in Recommendation.__table__.columns
                    if c.name not in ("created_at", "updated_at")
                },
                "is_completed": row.task_status == TaskStatus.DONE,
            }
        )
    return out


async def _recommendation_detail(
    db: AsyncSession, deal_id: uuid.UUID, rec_id: uuid.UUID
) -> Any:
    row = (
        await db.execute(
            select(Recommendation, Task.status.label("task_status"))
            .outerjoin(Task, Task.id == Recommendation.created_task_id)
            .where(Recommendation.id == rec_id, Recommendation.deal_id == deal_id)
        )
    ).first()
    rec = row.Recommendation
    return {
        **{
            c.name: getattr(rec, c.name)
            for c in Recommendation.__table__.columns
            if c.name not in ("created_at", "updated_at")
        },
        "is_completed": row.task_status == TaskStatus.DONE,
    }


@router.get("/recommendations/{rec_id}", response_model=RecommendationDetail)
async def get_recommendation(
    rec_id: uuid.UUID,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    await risk_service.get_recommendation_or_404(db, deal.id, rec_id)
    return await _recommendation_detail(db, deal.id, rec_id)


@router.get(
    "/recommendations/{rec_id}/evidence", response_model=List[EvidenceItem]
)
async def get_recommendation_evidence(
    rec_id: uuid.UUID,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """A recommendation inherits its risk's citations: the reason to act is the
    same evidence as the reason to worry."""
    await risk_service.get_recommendation_or_404(db, deal.id, rec_id)
    return await claims_service.evidence_for(db, ClaimType.RECOMMENDATION, rec_id)


@router.post("/recommendations/{rec_id}/accept", response_model=RecommendationDetail)
async def accept_recommendation(
    rec_id: uuid.UUID,
    body: RecommendationAccept,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Turn the suggestion into a task.

    The body is the prefilled form as the user edited it. `due_date` is required
    even though tasks.due_date is nullable -- accepting means committing to
    *when*, and undated committed work is how a task list becomes noise.
    """
    rec = await risk_service.get_recommendation_or_404(db, deal.id, rec_id)
    await risk_service.accept_recommendation(db, deal.id, rec, body)
    await db.commit()
    return await _recommendation_detail(db, deal.id, rec_id)


@router.post("/recommendations/{rec_id}/dismiss", response_model=RecommendationDetail)
async def dismiss_recommendation(
    rec_id: uuid.UUID,
    body: RecommendationDismiss,
    deal: Deal = Depends(get_deal_or_404),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Record that a human said no, and why.

    Remembered rather than deleted, for two reasons: the detector must not
    re-suggest it on the next run, and the reason is the only signal that says
    whether the suggestions are any good.
    """
    rec = await risk_service.get_recommendation_or_404(db, deal.id, rec_id)
    await risk_service.dismiss_recommendation(db, rec, body)
    await db.commit()
    return await _recommendation_detail(db, deal.id, rec_id)

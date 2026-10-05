"""Risk and recommendation decisions -- the human half of Layer C.

The detector proposes; these functions record what a person decided. That
split is the rule the whole design turns on: nothing the model or a SQL rule
produces becomes committed work without someone accepting it.
"""

import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Recommendation, Risk, Task
from app.models.enums import ClaimType, Origin, RecommendationStatus, RiskStatus, TaskStatus
from app.services import claims as claims_service


async def get_risk_or_404(
    db: AsyncSession, deal_id: uuid.UUID, risk_id: uuid.UUID
) -> Risk:
    risk = await db.scalar(
        select(Risk).where(Risk.id == risk_id, Risk.deal_id == deal_id)
    )
    if risk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Risk {risk_id} not found on deal {deal_id}",
        )
    return risk


async def get_recommendation_or_404(
    db: AsyncSession, deal_id: uuid.UUID, rec_id: uuid.UUID
) -> Recommendation:
    rec = await db.scalar(
        select(Recommendation).where(
            Recommendation.id == rec_id, Recommendation.deal_id == deal_id
        )
    )
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recommendation {rec_id} not found on deal {deal_id}",
        )
    return rec


async def apply_status_change(
    db: AsyncSession, risk: Risk, to_status: RiskStatus
) -> None:
    """Move a risk, maintaining ``resolved_at``.

    The fourth instance of this shape, after deal stage, task status and
    meeting status:

    *   -> resolved   stamp resolved_at
    *   back to open  clear it. A risk reopened while still carrying the time
                      it was resolved reads as closed to anything checking the
                      timestamp.
    *   -> dismissed  leave it NULL. Dismissed is not resolved -- the situation
                      did not change, a human decided it did not matter, and
                      conflating the two would corrupt any measure of how many
                      risks actually got fixed.

    Note that moving a risk out of `open` frees the partial unique index, so
    the detector can legitimately raise the same risk_type again later. That is
    intended: a stall that recurs in December is a new stall.
    """
    if risk.status == to_status:
        return

    was_resolved = risk.status == RiskStatus.RESOLVED
    risk.status = to_status

    if to_status == RiskStatus.RESOLVED:
        risk.resolved_at = func.now()
    elif was_resolved:
        risk.resolved_at = None


async def accept_recommendation(
    db: AsyncSession, deal_id: uuid.UUID, rec: Recommendation, body
) -> Task:
    """Turn a suggestion into committed work, in one transaction.

    Creates the task, points the recommendation at it, and records the
    decision. Three writes for one human action, so it cannot half-succeed
    with a task created and the recommendation still reading as unanswered.

    ``origin='ai'`` on the task: this work was suggested, not thought of by the
    person doing it, and "how much of my task list came from suggestions?" is
    worth being able to answer.
    """
    if rec.status == RecommendationStatus.ACCEPTED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Already accepted; it created task {rec.created_task_id}. "
                f"Edit that task rather than accepting again."
            ),
        )
    if rec.status == RecommendationStatus.DISMISSED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Already dismissed. Nothing re-opens a dismissed suggestion.",
        )

    task = Task(
        deal_id=deal_id,
        title=body.title or rec.title,
        description=body.description if body.description is not None else rec.rationale,
        due_date=body.due_date,
        priority=body.priority or rec.priority,
        status=TaskStatus.OPEN,
        origin=Origin.AI,
    )
    db.add(task)
    await db.flush()

    rec.created_task_id = task.id
    rec.status = RecommendationStatus.ACCEPTED
    rec.decided_at = func.now()
    # Task events refresh deterministic state, but never enqueue an AI pass.
    # This task itself is AI-originated, so the loop guard is explicit too.
    from app.services import analysis as analysis_service

    await analysis_service.record_change(
        db,
        deal_id,
        "AI recommendation accepted as task",
        tier1=True,
        tier2=False,
        origin=Origin.AI,
    )
    return task


async def dismiss_recommendation(
    db: AsyncSession, rec: Recommendation, body
) -> None:
    """Record that a human said no, and why.

    The record is the point. Without it the detector re-suggests the same thing
    on every run, and you learn nothing about *why* suggestions get refused --
    which is the only signal that tells you whether the advice is any good.
    """
    if rec.status == RecommendationStatus.ACCEPTED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Already accepted; it created task {rec.created_task_id}. "
                f"Cancel that task instead."
            ),
        )

    rec.status = RecommendationStatus.DISMISSED
    rec.dismissal_reason = body.reason.value
    rec.dismissal_note = body.note
    rec.decided_at = func.now()


async def delete_risk_evidence(db: AsyncSession, risk_ids) -> None:
    """Clear citations before risks are deleted.

    `claim_evidence.claim_id` has no foreign key -- it is polymorphic over five
    tables -- so Postgres will not stop a risk being deleted out from under its
    links. See services/claims.py.
    """
    await claims_service.delete_claim_links(db, ClaimType.RISK, list(risk_ids))

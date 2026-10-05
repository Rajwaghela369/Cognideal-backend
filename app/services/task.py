"""Task writes that are more than one statement, or that have a rule.

Same reasoning as services/deal.py: the seeder and the fact-acceptance flow
will need to make exactly these writes, and two implementations of "what
happens when a task is completed" drift.
"""

import uuid

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Deal, ExtractedFact, Task
from app.models.enums import FactStatus, TaskStatus
from app.services import activity
from app.services import analysis as analysis_service


async def apply_status_change(db: AsyncSession, task: Task, to_status: TaskStatus) -> None:
    """Move a task to a new status, maintaining ``completed_at``.

    The same shape as ``deal.apply_stage_change``: writing ``status`` alone is
    never correct.

    *   ``done``      -> stamp ``completed_at``.
    *   back to open  -> clear it. A reopened task carrying the timestamp of
                        the time it was briefly finished reads as done to
                        every query that checks it.
    *   ``cancelled`` -> leave it NULL. Cancelled is not completed, and a
                        completion date on abandoned work corrupts any future
                        "how much did we finish" measure.

    A no-op status write returns early, so a PATCH that merely echoes the
    current status does not re-stamp the timestamp.
    """
    if task.status == to_status:
        return

    was_done = task.status == TaskStatus.DONE
    task.status = to_status

    if to_status == TaskStatus.DONE:
        task.completed_at = func.now()
        await activity.touch_deal(db, task.deal_id)
    elif was_done:
        task.completed_at = None

    await analysis_service.record_change(
        db,
        task.deal_id,
        "task status changed",
        tier1=True,
        tier2=False,
    )


async def get_deal_for_task_or_422(db: AsyncSession, deal_id: uuid.UUID) -> Deal:
    """A bad deal_id would otherwise surface as a foreign-key IntegrityError at
    commit, which reaches the client as a 500."""
    deal = await db.get(Deal, deal_id)
    if deal is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Deal {deal_id} not found",
        )
    return deal


async def release_source_fact(db: AsyncSession, task: Task) -> None:
    """Undo the promotion before deleting a task that came from a fact.

    ``tasks.source_fact_id`` is a real foreign key, but the *reverse* pointer
    is not: ``extracted_facts.promoted_to_id`` is a bare uuid, because it is
    polymorphic over the tables a fact can be promoted into. So deleting a
    promoted task leaves the fact claiming it became a row that no longer
    exists -- the same orphan problem as claim_evidence on deal delete.

    Sending the fact back to ``pending`` rather than leaving it ``accepted``
    means it reappears in the review queue, which is the honest outcome: the
    work it described was thrown away, so the decision is open again.
    """
    if task.source_fact_id is None:
        return

    fact = await db.get(ExtractedFact, task.source_fact_id)
    if fact is None:
        return

    fact.promoted_to_type = None
    fact.promoted_to_id = None
    fact.status = FactStatus.PENDING
    fact.reviewed_at = None

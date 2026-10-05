"""Operational state for the AI layer. Task 11.5.

Top-level rather than per-deal: nothing here is about one deal, and the question
it answers -- "is the pipeline moving, and if not, why" -- is asked when no deal
id is to hand.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai import client
from app.core.config import settings
from app.db.session import get_db
from app.models import Deal, Meeting
from app.models.enums import AnalysisStatus
from app.schemas.v1.system import AIStatus
from app.services import analysis as analysis_service

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/ai-status", response_model=AIStatus)
async def ai_status(db: AsyncSession = Depends(get_db)) -> Any:
    """Configuration, this process's token budget, and the worker's queue depth.

    Three things that are invisible otherwise. The per-run log line in
    `client.py` says what happened; this says what is about to. The queue counts
    are the honest part -- they are Postgres rows, shared by every process. The
    budget is this interpreter's bucket and is labelled as such, because the
    worker runs in its own container with its own.
    """
    now = datetime.now(timezone.utc)
    sweep_cutoff = now - timedelta(hours=settings.analysis_sweep_hours)

    queued_meetings = await db.scalar(
        select(func.count())
        .select_from(Meeting)
        .where(Meeting.analysis_status == AnalysisStatus.QUEUED)
    )
    oldest_queued = await db.scalar(
        select(func.min(Meeting.updated_at)).where(
            Meeting.analysis_status == AnalysisStatus.QUEUED
        )
    )
    dirty_deals = await db.scalar(
        select(func.count())
        .select_from(Deal)
        .where(Deal.analysis_dirty_first_at.is_not(None))
    )
    oldest_dirty = await db.scalar(select(func.min(Deal.analysis_dirty_first_at)))
    sweep_backlog = await db.scalar(
        select(func.count())
        .select_from(Deal)
        .where(
            Deal.analysis_dirty_first_at.is_(None),
            (Deal.analysis_swept_at.is_(None)) | (Deal.analysis_swept_at < sweep_cutoff),
        )
    )

    # Reading the bucket must not charge it, so this goes through the read-only
    # property rather than take().
    available = client.governor().tokens_available
    capacity = settings.groq_tokens_per_minute
    deficit = max(0.0, capacity - available)
    # The bucket refills at capacity/60 tokens per second.
    seconds_to_full = round(deficit / (capacity / 60.0), 1) if capacity else 0.0

    return {
        "config": {
            "enabled": settings.ai_enabled,
            "model_primary": settings.groq_model_primary,
            "model_cheap": settings.groq_model_cheap,
            "tokens_per_minute": capacity,
            "requests_per_minute": settings.groq_requests_per_minute,
            "max_concurrency": settings.groq_max_concurrency,
            "debounce_seconds": settings.analysis_debounce_seconds,
            "max_debounce_seconds": settings.analysis_max_debounce_seconds,
            "sweep_hours": settings.analysis_sweep_hours,
            "tier2_suppressed": analysis_service.tier2_is_suppressed(),
        },
        "budget": {
            "process": "api:pid-%d" % os.getpid(),
            "tokens_available": round(available, 1),
            "tokens_capacity": capacity,
            "seconds_to_full": seconds_to_full,
        },
        "queues": {
            "queued_meetings": queued_meetings or 0,
            "oldest_queued_meeting_at": oldest_queued,
            "dirty_deals": dirty_deals or 0,
            "oldest_dirty_at": oldest_dirty,
            "sweep_backlog": sweep_backlog or 0,
        },
    }

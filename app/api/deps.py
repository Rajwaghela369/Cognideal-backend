import uuid
from dataclasses import dataclass
from typing import Any, List, Tuple

from fastapi import Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.db.session import get_db
from app.models import Deal


@dataclass(frozen=True)
class Pagination:
    limit: int
    offset: int


def pagination(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Pagination:
    return Pagination(limit=limit, offset=offset)


async def paginate(
    db: AsyncSession, stmt: Select, page: Pagination
) -> Tuple[List[Any], int]:
    """Run a statement twice: once counted, once windowed.

    `order_by(None)` strips the ORDER BY before counting -- sorting a subquery
    whose rows are only being counted is wasted work, and window functions in
    the ordering would otherwise have to be materialised.
    """
    total = await db.scalar(select(func.count()).select_from(stmt.order_by(None).subquery()))
    result = await db.execute(stmt.limit(page.limit).offset(page.offset))
    return result.all(), total or 0


async def get_deal_or_404(
    deal_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Deal:
    """Resolve a deal for the sub-resource routes.

    Without this the sub-resources would answer an unknown deal id with an
    empty list, which is indistinguishable from a real deal that simply has no
    meetings yet.
    """
    deal = await db.get(Deal, deal_id)
    if deal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Deal {deal_id} not found"
        )
    return deal

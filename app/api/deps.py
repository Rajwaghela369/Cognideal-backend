import uuid
from typing import Any, List, Tuple

from fastapi import Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.db.session import get_db
from app.models import Deal
from app.schemas.common import ListQuery


async def paginate(
    db: AsyncSession, stmt: Select, page: ListQuery
) -> Tuple[List[Any], int]:
    """Run a statement twice: once counted, once windowed.

    `order_by(None)` strips the ORDER BY before counting -- sorting a subquery
    whose rows are only being counted is wasted work, and window functions in
    the ordering would otherwise have to be materialised.

    `page` is the endpoint's own filter model, which carries `limit`/`offset`
    by inheriting `ListQuery`. There is deliberately no `Depends(pagination)`
    here any more: a second dependency reading the query string cannot
    coexist with a query-bound filter model that forbids extras -- see
    `ListQuery`. Removing it rather than leaving it unused is the point, so
    the next paginated endpoint cannot reintroduce the bug by reaching for it.
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

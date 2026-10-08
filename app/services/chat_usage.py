"""The chat assistant's daily question limit.

One shared cap -- ``chat_daily_message_limit`` questions per day, the day
starting at midnight in ``chat_limit_timezone`` -- because there are no user
accounts to divide it by. A question is a ``chat_messages`` row with
``role='user'``; nothing else is counted, so meeting analysis, briefs and risk
detection are unaffected, and so is the background session-title call.

The count is taken from the rows themselves rather than a separate counter, so
there is nothing to reset at midnight and nothing that can drift from what was
actually asked.
"""

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict

from fastapi import HTTPException, status
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import ChatMessage
from app.models.enums import ChatRole

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

logger = logging.getLogger("cognideal.chat.usage")

#: Serialises the check-then-insert of concurrent sends, so two people asking
#: at question 199 cannot both get through. Transaction-scoped: released when
#: the turn commits the user's message, or on rollback.
_LOCK_KEY = 0x63686174  # "chat"


def _zone():
    try:
        return ZoneInfo(settings.chat_limit_timezone) if ZoneInfo else timezone.utc
    except Exception:  # noqa: BLE001 -- a bad setting must not break chat
        logger.warning("chat.limit_timezone_unknown tz=%r -- using UTC", settings.chat_limit_timezone)
        return timezone.utc


def day_bounds(now: datetime = None):
    """Start of today and of tomorrow, in the limit's timezone."""
    zone = _zone()
    local = (now or datetime.now(timezone.utc)).astimezone(zone)
    start = datetime.combine(local.date(), time.min, tzinfo=zone)
    end = datetime.combine(local.date() + timedelta(days=1), time.min, tzinfo=zone)
    return start, end


async def usage(db: AsyncSession) -> Dict[str, Any]:
    """Today's questions against the limit. ``limit`` 0 means unlimited."""
    start, end = day_bounds()
    used = await db.scalar(
        select(func.count()).select_from(ChatMessage).where(
            ChatMessage.role == ChatRole.USER,
            ChatMessage.created_at >= start,
        )
    ) or 0
    limit = max(settings.chat_daily_message_limit, 0)
    return {
        "limit": limit,
        "used": used,
        "remaining": max(limit - used, 0) if limit else None,
        "resets_at": end,
        "timezone": getattr(end.tzinfo, "key", "UTC"),
    }


async def reserve_question(db: AsyncSession) -> Dict[str, Any]:
    """Refuse the question if today's limit is used up; otherwise hold the
    lock until the caller's transaction commits the question.

    Raises 429 with the reset time when the limit is reached.
    """
    if settings.chat_daily_message_limit <= 0:
        return await usage(db)
    await db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _LOCK_KEY})
    current = await usage(db)
    if current["used"] >= current["limit"]:
        resets = current["resets_at"]
        logger.info(
            "chat.limit_reached used=%d limit=%d resets_at=%s",
            current["used"], current["limit"], resets.isoformat(),
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            # No number in the message: the UI shows usage only as a bar.
            detail=(
                "Today's chat limit has been reached. Chat is available again at %s (%s time)."
                % (resets.strftime("%H:%M on %d %b"), current["timezone"])
            ),
            headers={"Retry-After": str(max(int((resets - datetime.now(timezone.utc)).total_seconds()), 1))},
        )
    return current

"""Stage transitions.

Responses only -- there is deliberately no request model. The one writer is
services.deal.apply_stage_change, reached through PATCH /deals/{id}, which
writes the history row, deals.stage and closed_at in one transaction. See
routes/stage_history.py for why appending here directly is not offered."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.common import ORM


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class StageHistoryEntry(BaseModel):
    model_config = ORM

    id: uuid.UUID
    from_stage: Optional[str] = None
    to_stage: str
    changed_at: datetime
    note: Optional[str] = None
    # Whole days spent in `to_stage` before the next transition. The newest
    # entry is open-ended, so it is measured against now() rather than left
    # null -- "still here, 58 days" is the number the timeline renders, and a
    # null would make every client reimplement that subtraction.
    days_in_stage: int = 0

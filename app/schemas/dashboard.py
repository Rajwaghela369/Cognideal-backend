import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PriorityDeal(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    account_id: uuid.UUID
    account_name: str
    name: str
    value: Optional[Decimal]
    stage: str
    risk_level: Optional[str]
    expected_close_date: Optional[date]
    next_action: Optional[str]
    last_activity_at: Optional[datetime]

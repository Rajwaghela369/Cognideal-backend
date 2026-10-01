"""Accounts.

Only AccountRef, because it is embedded in DealDetail. The account list and
detail models belong with the /accounts routes, which do not exist yet.
"""

import uuid
from typing import Optional

from pydantic import BaseModel

from app.schemas.common import ORM


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class AccountRef(BaseModel):
    """Minimal account identity, embedded wherever a record points at one."""

    model_config = ORM

    id: uuid.UUID
    name: str
    industry: Optional[str] = None

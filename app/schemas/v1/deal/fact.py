"""Extracted facts -- what the model asserted, awaiting a human.

Every row here is a *proposal*. `status='pending'` until somebody promotes it,
which is Gate 3 and the only place correctness is actually settled
(docs/schema/README.md section 5).

Two fields exist so a reader can judge rather than trust:

`verdict` is Gate 1's independent check -- whether the quoted span supports the
claim. `confidence` is the model's self-report about itself, which is a weak
and poorly calibrated signal; this corpus came back at exactly 1.00 across
every fact. They are returned side by side and never conflated: confidence is
never rendered as validation, and a high score never bypasses a gate.

Quarantined facts (`contradicted`, `unsupported`) are absent from every
response -- filtered by `queries.quarantine_filter`, not by this schema.
`partial` is returned, so the UI can show it with a caution badge.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from app.models.enums import FactStatus, FactType, Verdict
from app.schemas.common import ORM


class FactEvidence(BaseModel):
    """The span behind a fact. A fact with none never reaches here -- Gate 0
    rejects it on insert."""

    model_config = ORM

    evidence_id: uuid.UUID
    snippet: str
    speaker: Optional[str] = None
    chunk_id: Optional[uuid.UUID] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    verification_status: str


class FactListItem(BaseModel):
    model_config = ORM

    id: uuid.UUID
    fact_type: str
    content: str
    payload: Optional[dict] = None
    status: str
    #: The generator's self-report. Not validation -- see the module docstring.
    confidence: Optional[float] = None
    #: Gate 1's newest verdict, or null when not yet validated. Absent is a
    #: distinct state from any verdict and is not defaulted to a pass.
    verdict: Optional[Verdict] = None
    meeting_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    extracted_at: Optional[datetime] = None
    evidence: List[FactEvidence] = []


class FactFilters(BaseModel):
    fact_type: Optional[List[FactType]] = None
    status: Optional[List[FactStatus]] = None

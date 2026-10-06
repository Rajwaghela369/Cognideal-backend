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
from typing import List, Literal, Optional

from pydantic import BaseModel

from app.models.enums import FactStatus, FactType, Verdict
from app.schemas.common import ORM, WRITE


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
    #: Where accepting this fact sent it, for the categories that promote.
    #: Null for the six that only confirm -- which is not a failure, there is
    #: simply no table a `competitor` or `budget` fact belongs in.
    promoted_to_type: Optional[str] = None
    promoted_to_id: Optional[uuid.UUID] = None
    evidence: List[FactEvidence] = []


class FactDecision(BaseModel):
    """Gate 3: a human's verdict on a proposal.

    Only `accepted` and `rejected` are offered. `pending` is where a fact
    starts, and `superseded` belongs to the pipeline -- a human adjudicating a
    claim is not choosing between four states, they are answering one question.
    Typed as a two-value Literal rather than `FactStatus` so the other two are a
    422 listing what is legal, instead of a silent write of a status this
    endpoint has no business setting.
    """

    model_config = WRITE

    status: Literal[FactStatus.ACCEPTED, FactStatus.REJECTED]


class FactFilters(BaseModel):
    fact_type: Optional[List[FactType]] = None
    status: Optional[List[FactStatus]] = None

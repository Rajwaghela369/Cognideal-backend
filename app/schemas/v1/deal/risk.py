"""Risks and their recommendations -- the "needs attention" panel.

The recommendation is **nested inside the risk**, not a parallel list. The user
does not think "show me risks" then "show me recommendations"; they think "what
is wrong and what do I do about it", and those are one card. A recommendation
without its risk is advice with no reason; a risk without its recommendation is
a complaint with no fix.
"""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.enums import (
    ActionType,
    DismissalReason,
    Priority,
    RecommendationStatus,
    RiskStatus,
    RiskType,
    Severity,
)
from app.schemas.common import ORM, WRITE

# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """One locatable source for a claim.

    Two flavours, discriminated by `source_kind`:

    *   `document` -- `chunk_id` plus char offsets point at an exact stretch of
        a transcript, email or contract. Resolve it via GET /chunks/{id}.
    *   `record`   -- `record_ref` points at a field in our own database,
        e.g. {"table": "deals", "id": "...", "field": "expected_close_date"}.
        Most risk detection reasons over structured state, not over quotes, so
        without this flavour every deterministic risk would render uncited.
    *   `derived`  -- an assertion about the *absence* of rows, which cannot
        cite one.

    `snippet` is the literal value, which is what lets Gate 0 re-check that the
    source still says it -- and what makes a risk self-invalidate when the
    underlying field changes.
    """

    model_config = ORM

    id: uuid.UUID
    source_kind: str
    snippet: Optional[str] = None
    document_id: Optional[uuid.UUID] = None
    chunk_id: Optional[uuid.UUID] = None
    record_ref: Optional[dict] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    speaker: Optional[str] = None
    occurred_at: Optional[datetime] = None
    # How strongly *this* span supports *this* claim -- what lets the UI show
    # the best quote first and collapse the marginal ones.
    relevance: Optional[Decimal] = None
    verification_status: Optional[str] = None
    verified_at: Optional[datetime] = None


class RecommendationRef(BaseModel):
    """Minimal recommendation identity, embedded where a row points at one.

    The same shape and purpose as ``AccountRef``: enough to name the thing and
    link to it, without dragging its whole row along. Used by ``TaskDetail`` to
    answer "which suggestion did this task come from?".

    ``action_type`` is ``str`` rather than the enum, as every other response
    here is -- it is a ``text + CHECK`` column precisely because it churns as
    prompts are tuned, so a closed type on the way out would break a client
    the day a value is added.
    """

    model_config = ORM

    id: uuid.UUID
    title: str
    action_type: str
    rationale: Optional[str] = None


class RecommendationSummary(BaseModel):
    """The suggested action, as it appears inside a risk card."""

    model_config = ORM

    id: uuid.UUID
    title: str
    action_type: str
    priority: str
    rationale: Optional[str] = None
    status: str
    # Derived from created_task_id's task being done -- never stored. A second
    # copy of the task's status is a copy that goes stale, the same rule that
    # keeps `next_action` off the deals table.
    is_completed: bool = False
    created_task_id: Optional[uuid.UUID] = None
    dismissal_reason: Optional[str] = None


class RiskListItem(BaseModel):
    """One card in the panel."""

    model_config = ORM

    id: uuid.UUID
    risk_type: str
    title: str
    description: Optional[str] = None
    severity: str
    status: str
    origin: str
    confidence: Optional[Decimal] = None
    first_detected_at: datetime
    last_seen_at: datetime
    resolved_at: Optional[datetime] = None
    evidence_count: int = 0
    recommendation: Optional[RecommendationSummary] = None


class RiskDetail(RiskListItem):
    """The card expanded, with its citations."""

    evidence: List[EvidenceItem] = Field(default_factory=list)


class RecommendationDetail(BaseModel):
    model_config = ORM

    id: uuid.UUID
    deal_id: uuid.UUID
    source_risk_id: Optional[uuid.UUID] = None
    title: str
    description: Optional[str] = None
    rationale: Optional[str] = None
    action_type: str
    priority: str
    confidence: Optional[Decimal] = None
    status: str
    origin: str
    is_completed: bool = False
    created_task_id: Optional[uuid.UUID] = None
    dismissal_reason: Optional[str] = None
    dismissal_note: Optional[str] = None
    generated_at: datetime
    decided_at: Optional[datetime] = None


class DetectionResult(BaseModel):
    """What a detector run did."""

    risks_detected: int
    recommendations_written: int
    risks_auto_resolved: int


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


class RiskUpdate(BaseModel):
    """Only the human decision is writable.

    `risk_type`, `title`, `description`, `severity`, `confidence` and the
    timestamps belong to the detector. Editing what it asserted would destroy
    the record of what it asserted -- if a risk is wrong, dismiss it.
    """

    model_config = WRITE

    status: RiskStatus
    # Free text on the decision, not on the claim.
    note: Optional[str] = None


class RecommendationAccept(BaseModel):
    """The task this becomes, as the user edited it in the prefilled form.

    `due_date` is required even though `tasks.due_date` is nullable: a
    suggestion promoted to committed work with no date is how a task list turns
    into noise. The point of accepting is to commit to *when*.
    """

    model_config = WRITE

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Defaults to the recommendation's title",
    )
    description: Optional[str] = None
    due_date: date
    priority: Optional[Priority] = Field(
        default=None, description="Defaults to the recommendation's priority"
    )


class RecommendationDismiss(BaseModel):
    """Why the user said no.

    `reason` is countable -- "40% are `wrong`" says the detector needs work,
    "40% are `already_handled`" says it is right but late. `note` carries the
    detail a count cannot. A single free-text field would give neither.
    """

    model_config = WRITE

    reason: DismissalReason
    note: Optional[str] = None


class RiskFilters(BaseModel):
    """Query parameters of GET /deals/{deal_id}/risks.

    Unpaginated: a deal has at most ten risks, one per risk_type.
    """

    model_config = WRITE

    status: List[RiskStatus] = Field(default_factory=list)
    risk_type: List[RiskType] = Field(default_factory=list)
    severity: List[Severity] = Field(default_factory=list)
    open: Optional[bool] = Field(
        default=None,
        description="true = status is open or mitigating; false = resolved or "
        "dismissed. Shorthand so callers need not enumerate them.",
    )


class RecommendationFilters(BaseModel):
    """Query parameters of GET /deals/{deal_id}/recommendations.

    Secondary to the risk panel. Earns its place for the proactive
    recommendations that have no risk to nest under, and for evaluation --
    what was suggested, accepted and dismissed is the only honest measure of
    whether the advice is any good.
    """

    model_config = WRITE

    status: List[RecommendationStatus] = Field(default_factory=list)
    action_type: List[ActionType] = Field(default_factory=list)
    orphaned: Optional[bool] = Field(
        default=None,
        description="true lists recommendations with no source_risk_id -- the "
        "proactive ones, which the risk panel cannot show",
    )


class DealAnalysisState(BaseModel):
    """Phase 10's trigger state, read back.

    The worker's two claim queries are the authority on when a deal runs; this
    projection reports what they would decide *now* so a badge and the claim
    cannot disagree. `state` is derived, never stored:

    ``clean``       nothing pending, swept inside the window
    ``debouncing``  marked dirty, still inside the quiet window -- more edits
                    are expected and will collapse into one run
    ``due``         marked dirty and past the quiet or maximum window; the next
                    poll claims it
    ``stale``       not dirty, but `analysis_swept_at` is null or older than the
                    sweep window, so the nightly pass will pick it up

    ``debouncing`` and ``due`` are kept apart because they mean opposite things
    to a person watching: one says "still collecting your edits", the other says
    "running shortly". Collapsing them into "queued" would make the quiet window
    look like latency.
    """

    model_config = ORM

    deal_id: uuid.UUID
    state: str
    dirty_first_at: Optional[datetime] = None
    dirty_last_at: Optional[datetime] = None
    dirty_reason: Optional[str] = None
    swept_at: Optional[datetime] = None
    #: Echoed so a client can render "runs in ~40s" without hardcoding the
    #: server's debounce, and so a misconfiguration is visible rather than
    #: inferred from timing.
    debounce_seconds: int
    max_debounce_seconds: int
    sweep_hours: int
